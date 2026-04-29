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


class UniformRebounceCommand(CommandTerm):
    """Per-env scalar ``peak_height`` target for the rebounce task.

    On each env reset, samples ``h_cmd ~ U(ranges.peak_height)`` and latches
    ``drop_z = base_z`` after the reset event teleports the robot to its drop
    height. Tracks periodic drop/rebound/apex state so rewards can fire once
    per airborne apex while the episode continues.

    Unlike :class:`UniformHoppingCommand`, this class does **not** depend on
    the contact sensor: rebounce semantics are encoded by root vertical
    velocity and base-state running max.
    """

    cfg: UniformRebounceCommandCfg

    def __init__(self, cfg: UniformRebounceCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_cfg.name]
        self._foot_asset = None
        self._foot_body_ids = None
        if cfg.foot_asset_cfg is not None:
            cfg.foot_asset_cfg.resolve(env.scene)
            self._foot_asset: Articulation = env.scene[cfg.foot_asset_cfg.name]
            self._foot_body_ids = cfg.foot_asset_cfg.body_ids
        n, dev = self.num_envs, self.device
        self._peak_h_target = torch.zeros(n, device=dev)
        # ``peak_z`` is the max of base_z taken **only during the rebound
        # ascent** — i.e. timesteps where the robot has already descended
        # (vz < 0 at some prior step) and is now moving upward (vz > 0). This
        # excludes the initial drop_z so a weak bounce that fails to recover
        # the drop height is correctly reported as ``realized_peak_height < 0``.
        self._peak_z = torch.zeros(n, device=dev)
        self._drop_z = torch.zeros(n, device=dev)
        self._has_descended = torch.zeros(n, dtype=torch.bool, device=dev)
        self._has_rebounded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._just_apex = torch.zeros(n, dtype=torch.bool, device=dev)
        self._apex_armed = torch.ones(n, dtype=torch.bool, device=dev)
        self._prev_vz = torch.zeros(n, device=dev)
        self._apex_count = torch.zeros(n, device=dev)
        self._target_apex_count = torch.zeros(n, device=dev)
        self.metrics["error_peak_height"] = torch.zeros(n, device=dev)
        self.metrics["apex_count"] = torch.zeros(n, device=dev)
        self.metrics["target_apex_count"] = torch.zeros(n, device=dev)

    @property
    def command(self) -> torch.Tensor:
        return self._peak_h_target.unsqueeze(-1)

    @property
    def peak_height(self) -> torch.Tensor:
        return self._peak_h_target

    @property
    def peak_z(self) -> torch.Tensor:
        return self._peak_z

    @property
    def drop_z(self) -> torch.Tensor:
        return self._drop_z

    @property
    def realized_peak_height(self) -> torch.Tensor:
        return self._peak_z - self._drop_z

    @property
    def rebound_height_error(self) -> torch.Tensor:
        return torch.abs(self._peak_z - self._drop_z)

    @property
    def has_descended(self) -> torch.Tensor:
        return self._has_descended

    @property
    def has_rebounded(self) -> torch.Tensor:
        return self._has_rebounded

    @property
    def just_apex(self) -> torch.Tensor:
        return self._just_apex

    @property
    def apex_count(self) -> torch.Tensor:
        return self._apex_count

    @property
    def target_apex_count(self) -> torch.Tensor:
        return self._target_apex_count

    def _foot_airborne_and_near(self) -> tuple[torch.Tensor, torch.Tensor]:
        ones = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if self._foot_asset is None or self._foot_body_ids is None:
            return ones, ones
        foot_z_local = self._foot_asset.data.body_pos_w[:, self._foot_body_ids, 2] - self._env.scene.env_origins[:, 2:3]
        min_foot_z = torch.min(foot_z_local, dim=1).values
        threshold = self.cfg.surface_z + self.cfg.foot_clearance
        return min_foot_z > threshold, min_foot_z <= threshold

    def _target_apex(self, airborne_apex: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        height_ok = torch.abs(z - self._drop_z) < self.cfg.apex_height_tolerance
        return airborne_apex & height_ok

    def _update_metrics(self):
        z = self.robot.data.root_pos_w[:, 2]
        vz = self.robot.data.root_lin_vel_w[:, 2]
        # Periodic state machine:
        # descending -> rebound ascent -> apex pulse -> re-arm for next fall.
        foot_airborne, foot_near = self._foot_airborne_and_near()
        self._apex_armed = self._apex_armed | foot_near
        apex = self._has_descended & self._has_rebounded & (self._prev_vz > 0.0) & (vz <= 0.0)
        airborne_apex = self._apex_armed & apex & foot_airborne
        target_apex = self._target_apex(airborne_apex, z)
        self._just_apex = airborne_apex

        ascending_after_descent = self._has_descended & (vz > 0)
        self._has_rebounded = self._has_rebounded | ascending_after_descent
        self._peak_z = torch.where(
            ascending_after_descent,
            torch.maximum(self._peak_z, z),
            self._peak_z,
        )
        apex_error = torch.abs(z - self._drop_z)
        new_apex_count = torch.where(airborne_apex, self._apex_count + 1.0, self._apex_count)
        self.metrics["error_peak_height"] = torch.where(
            airborne_apex,
            self.metrics["error_peak_height"]
            + (apex_error - self.metrics["error_peak_height"]) / new_apex_count.clamp_min(1.0),
            self.metrics["error_peak_height"],
        )
        self._apex_count = new_apex_count
        self._target_apex_count = torch.where(target_apex, self._target_apex_count + 1.0, self._target_apex_count)
        self.metrics["apex_count"][:] = self._apex_count
        self.metrics["target_apex_count"][:] = self._target_apex_count
        self._apex_armed = self._apex_armed & ~airborne_apex

        self._has_descended = self._has_descended | (vz < 0)
        # After a completed apex, clear rebound state so the next reward pulse
        # requires a fresh descent and fresh rebound ascent.
        self._has_descended = torch.where(apex, vz < 0, self._has_descended)
        self._has_rebounded = torch.where(apex, torch.zeros_like(self._has_rebounded), self._has_rebounded)
        self._peak_z = torch.where(apex, torch.zeros_like(self._peak_z), self._peak_z)
        self._prev_vz = vz.clone()

    def _resample_command(self, env_ids: Sequence[int]):
        # h_cmd is sampled and written by the ``reset_drop_from_height`` event,
        # which runs before ``command_manager.reset()`` and is responsible for
        # both the drop pose and the per-env target. This keeps the two values
        # aligned without requiring a chicken-and-egg ordering hack.
        del env_ids

    def _update_command(self):
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        z = self.robot.data.root_pos_w[:, 2]
        if env_ids is None:
            self._drop_z.copy_(z)
            self._peak_z.zero_()
            self._has_descended.zero_()
            self._has_rebounded.zero_()
            self._just_apex.zero_()
            self._apex_armed.fill_(True)
            self._prev_vz.zero_()
            self._apex_count.zero_()
            self._target_apex_count.zero_()
        else:
            self._drop_z[env_ids] = z[env_ids]
            self._peak_z[env_ids] = 0.0
            self._has_descended[env_ids] = False
            self._has_rebounded[env_ids] = False
            self._just_apex[env_ids] = False
            self._apex_armed[env_ids] = True
            self._prev_vz[env_ids] = 0.0
            self._apex_count[env_ids] = 0.0
            self._target_apex_count[env_ids] = 0.0
        return extras


@configclass
class UniformRebounceCommandCfg(CommandTermCfg):
    """Config for :class:`UniformRebounceCommand`."""

    class_type: type = UniformRebounceCommand
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    foot_asset_cfg: SceneEntityCfg | None = None
    foot_clearance: float = 0.0
    surface_z: float = 0.0
    apex_height_tolerance: float = 0.25

    @configclass
    class Ranges:
        peak_height: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
