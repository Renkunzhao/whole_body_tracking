from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_G = 9.81


class UniformHoppingCommand(CommandTerm):
    """Per-env (peak_height, stance_time) command, uniformly sampled within user ranges.

    ``command`` returns a ``[num_envs, 2]`` tensor of ``(h*, t_stance*)``. Per-env
    ``cycle_time = t_stance* + 2·sqrt(2·h*/g)`` is derived from the command and
    exposed as a property for phase-clock observations and rewards.

    Also tracks per-hop metrics (air time, peak base z): the current cycle's
    accumulators are latched into ``last_*`` buffers at each air->contact
    transition; ``last_*`` persists across resamples so the most recent hop is
    always readable. Logs cross-env means into ``env.extras["log"]`` each step.
    """

    cfg: UniformHoppingCommandCfg

    def __init__(self, cfg: UniformHoppingCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_cfg.name]
        cfg.sensor_cfg.resolve(env.scene)
        self.sensor: ContactSensor = env.scene.sensors[cfg.sensor_cfg.name]
        self._body_ids = cfg.sensor_cfg.body_ids

        n, dev = self.num_envs, self.device
        self._peak_h_target = torch.zeros(n, device=dev)
        self._stance_time_target = torch.zeros(n, device=dev)
        self._cycle_time = torch.zeros(n, device=dev)
        # episode-step index at which the current cycle started — latched on
        # every resample so that phase restarts from 0 at each command change
        # (Option A: reset-on-resample, accepts a brief phase_contact whip until
        # the next landing but keeps the semantics simple and local).
        self._cycle_start_step = torch.zeros(n, dtype=torch.long, device=dev)
        self._in_air = torch.zeros(n, dtype=torch.bool, device=dev)
        self._air_time = torch.zeros(n, device=dev)
        self._stance_accum = torch.zeros(n, device=dev)
        self._peak_z = torch.zeros(n, device=dev)
        # Base z latched at the most recent takeoff event. ``peak_h_target`` is
        # commanded *relative to takeoff height* (consistent with the ballistic
        # ``flight_time_target = 2·sqrt(2·h*/g)`` formula, which assumes landing
        # back at takeoff altitude), so the realized hop height must be measured
        # as ``last_peak_z - takeoff_z`` rather than absolute world z.
        self._takeoff_z = torch.zeros(n, device=dev)
        self._last_air_time = torch.zeros(n, device=dev)
        self._last_stance_time = torch.zeros(n, device=dev)
        self._last_peak_z = torch.zeros(n, device=dev)
        # One-step pulse flags latched each step by ``_update_metrics`` so that
        # downstream terms (e.g. landing-event rewards) can fire exactly on the
        # transition without duplicating the air/contact detection logic.
        self._just_landed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._just_took_off = torch.zeros(n, dtype=torch.bool, device=dev)

        # Per-env event counters for incremental-mean metrics. Reset only on env
        # reset (in ``reset()`` override), not on mid-episode command resample —
        # otherwise the running mean would be wiped every 3–8 s.
        self._landing_count = torch.zeros(n, device=dev)
        self._takeoff_count = torch.zeros(n, device=dev)

        # Tracking error metrics: per-env running mean of absolute error across
        # all events in the current episode. Base class mean-of-envs + zeros on
        # env reset; we separately zero the counters in ``reset()``.
        self.metrics["error_peak_height"] = torch.zeros(n, device=dev)
        self.metrics["error_stance_time"] = torch.zeros(n, device=dev)

    @property
    def command(self) -> torch.Tensor:
        return torch.stack((self._peak_h_target, self._stance_time_target), dim=-1)

    @property
    def peak_height(self) -> torch.Tensor:
        return self._peak_h_target

    @property
    def stance_time(self) -> torch.Tensor:
        return self._stance_time_target

    @property
    def cycle_time(self) -> torch.Tensor:
        return self._cycle_time

    @property
    def stance_fraction(self) -> torch.Tensor:
        return self._stance_time_target / self._cycle_time.clamp_min(1e-6)

    @property
    def phase(self) -> torch.Tensor:
        """Per-env normalized phase in [0, 1), measured from the last resample."""
        elapsed = (self._env.episode_length_buf - self._cycle_start_step).to(torch.float32) * self._env.step_dt
        return torch.remainder(elapsed / self._cycle_time.clamp_min(1e-6), 1.0)

    @property
    def last_peak_z(self) -> torch.Tensor:
        return self._last_peak_z

    @property
    def last_peak_height(self) -> torch.Tensor:
        """Realized apex height above takeoff for the most recent completed hop."""
        return self._last_peak_z - self._takeoff_z

    @property
    def last_air_time(self) -> torch.Tensor:
        return self._last_air_time

    @property
    def last_stance_time(self) -> torch.Tensor:
        return self._last_stance_time

    @property
    def just_landed(self) -> torch.Tensor:
        return self._just_landed

    @property
    def just_took_off(self) -> torch.Tensor:
        return self._just_took_off

    @property
    def flight_time_target(self) -> torch.Tensor:
        """Commanded ballistic flight time per env: ``2·sqrt(2·h*/g)``."""
        return 2.0 * torch.sqrt(2.0 * self._peak_h_target / _G)

    def _update_metrics(self):
        forces = self.sensor.data.net_forces_w_history[:, :, self._body_ids, :]
        contacts = torch.max(torch.norm(forces, dim=-1), dim=1)[0] > self.cfg.contact_threshold
        all_air = ~contacts.any(dim=1)
        z = self.robot.data.root_pos_w[:, 2]
        dt = self._env.step_dt
        zero = torch.zeros_like(self._air_time)

        # accumulate current-cycle air / stance time and peak z
        self._air_time = torch.where(all_air, self._air_time + dt, self._air_time)
        self._stance_accum = torch.where(~all_air, self._stance_accum + dt, self._stance_accum)
        self._peak_z = torch.where(all_air, torch.maximum(self._peak_z, z), self._peak_z)

        # landing event: latch air_time + peak_z, update peak-height running mean.
        # ``peak_h_target`` is relative to takeoff height, so the realized hop
        # height compared against it is ``last_peak_z - takeoff_z`` (where
        # ``takeoff_z`` was latched at the most recent ``just_took_off`` event).
        just_landed = self._in_air & ~all_air
        self._just_landed = just_landed
        self._last_air_time = torch.where(just_landed, self._air_time, self._last_air_time)
        self._last_peak_z = torch.where(just_landed, self._peak_z, self._last_peak_z)
        new_landing_count = torch.where(just_landed, self._landing_count + 1.0, self._landing_count)
        peak_err = torch.abs((self._last_peak_z - self._takeoff_z) - self._peak_h_target)
        # incremental mean: m_n = m_{n-1} + (x_n - m_{n-1}) / n
        self.metrics["error_peak_height"] = torch.where(
            just_landed,
            self.metrics["error_peak_height"]
            + (peak_err - self.metrics["error_peak_height"]) / new_landing_count.clamp_min(1.0),
            self.metrics["error_peak_height"],
        )
        self._landing_count = new_landing_count
        self._air_time = torch.where(just_landed, zero, self._air_time)
        self._peak_z = torch.where(just_landed, zero, self._peak_z)

        # takeoff event: latch stance_time and the takeoff base z (used by the
        # next landing's peak-height error), update stance-time running mean.
        just_took_off = ~self._in_air & all_air
        self._just_took_off = just_took_off
        self._takeoff_z = torch.where(just_took_off, z, self._takeoff_z)
        self._last_stance_time = torch.where(just_took_off, self._stance_accum, self._last_stance_time)
        new_takeoff_count = torch.where(just_took_off, self._takeoff_count + 1.0, self._takeoff_count)
        stance_err = torch.abs(self._last_stance_time - self._stance_time_target)
        self.metrics["error_stance_time"] = torch.where(
            just_took_off,
            self.metrics["error_stance_time"]
            + (stance_err - self.metrics["error_stance_time"]) / new_takeoff_count.clamp_min(1.0),
            self.metrics["error_stance_time"],
        )
        self._takeoff_count = new_takeoff_count
        self._stance_accum = torch.where(just_took_off, zero, self._stance_accum)

        self._in_air = all_air

    def _resample_command(self, env_ids: Sequence[int]):
        r = self.cfg.ranges
        n_ids = len(env_ids)
        dev = self.device
        self._peak_h_target[env_ids] = torch.empty(n_ids, device=dev).uniform_(*r.peak_height)
        self._stance_time_target[env_ids] = torch.empty(n_ids, device=dev).uniform_(*r.stance_time)
        t_flight = 2.0 * torch.sqrt(2.0 * self._peak_h_target[env_ids] / _G)
        self._cycle_time[env_ids] = self._stance_time_target[env_ids] + t_flight
        # Latch the current episode step as this cycle's origin so ``phase``
        # restarts from 0 right after resample.
        self._cycle_start_step[env_ids] = self._env.episode_length_buf[env_ids]
        self._in_air[env_ids] = False
        self._air_time[env_ids] = 0.0
        self._stance_accum[env_ids] = 0.0
        self._peak_z[env_ids] = 0.0
        self._takeoff_z[env_ids] = 0.0

    def _update_command(self):
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # Base class means+zeros self.metrics for env_ids; we additionally zero
        # our running-mean event counters so the next episode starts fresh.
        extras = super().reset(env_ids)
        if env_ids is None:
            self._landing_count.zero_()
            self._takeoff_count.zero_()
        else:
            self._landing_count[env_ids] = 0.0
            self._takeoff_count[env_ids] = 0.0
        return extras


@configclass
class UniformHoppingCommandCfg(CommandTermCfg):
    """Config for :class:`UniformHoppingCommand`."""

    class_type: type = UniformHoppingCommand
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
    contact_threshold: float = 5.0

    @configclass
    class Ranges:
        peak_height: tuple[float, float] = MISSING
        stance_time: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
