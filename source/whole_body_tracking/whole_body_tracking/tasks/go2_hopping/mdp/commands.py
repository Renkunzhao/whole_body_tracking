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
    ``drop_height = base_height`` after the reset event teleports the robot to its
    independently sampled drop height. Tracks periodic drop/rebound/apex state
    so rewards can fire once per valid apex while the episode continues.

    Unlike :class:`UniformHoppingCommand`, this class does **not** depend on
    the contact sensor: rebounce semantics are encoded by root vertical
    velocity and geometric foot clearance.
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
        self._target_apex_height = torch.zeros(n, device=dev)
        self._drop_height = torch.zeros(n, device=dev)
        self._is_apex = torch.zeros(n, dtype=torch.bool, device=dev)
        self._last_apex_height = torch.zeros(n, device=dev)
        self._last_apex_target_height = torch.zeros(n, device=dev)
        self._apex_armed = torch.ones(n, dtype=torch.bool, device=dev)
        self._prev_vz = torch.zeros(n, device=dev)
        self._apex_count = torch.zeros(n, device=dev)
        self._height_matched_apex_count = torch.zeros(n, device=dev)
        self.metrics["error_peak_height"] = torch.zeros(n, device=dev)
        self.metrics["apex_count"] = torch.zeros(n, device=dev)
        self.metrics["height_matched_apex_count"] = torch.zeros(n, device=dev)

    @property
    def command(self) -> torch.Tensor:
        return self._target_apex_height.unsqueeze(-1)

    @property
    def drop_height(self) -> torch.Tensor:
        return self._drop_height

    @property
    def target_apex_height(self) -> torch.Tensor:
        return self._target_apex_height

    @property
    def is_apex(self) -> torch.Tensor:
        return self._is_apex

    @property
    def last_apex_height(self) -> torch.Tensor:
        return self._last_apex_height

    @property
    def last_apex_target_height(self) -> torch.Tensor:
        return self._last_apex_target_height

    def _feet_clearance_flags(self) -> tuple[torch.Tensor, torch.Tensor]:
        ones = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if self._foot_asset is None or self._foot_body_ids is None:
            return ones, ones
        foot_z_local = self._foot_asset.data.body_pos_w[:, self._foot_body_ids, 2] - self._env.scene.env_origins[:, 2:3]
        min_foot_z = torch.min(foot_z_local, dim=1).values
        max_foot_z = torch.max(foot_z_local, dim=1).values
        threshold = self.cfg.surface_z + self.cfg.foot_clearance
        return min_foot_z > threshold, max_foot_z <= threshold

    def _height_matched_apex(self, is_apex: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
        height_ok = torch.abs(height - self.target_apex_height) < self.cfg.apex_height_tolerance
        return is_apex & height_ok

    def _update_metrics(self):
        height = self.robot.data.root_pos_w[:, 2]
        vz = self.robot.data.root_lin_vel_w[:, 2]
        # Periodic event detector:
        # feet return below clearance -> arm -> upward-to-non-upward velocity
        # crossing with feet above clearance -> valid apex pulse -> disarm.
        feet_above_clearance, feet_below_clearance = self._feet_clearance_flags()
        self._apex_armed = self._apex_armed | feet_below_clearance
        is_apex = self._apex_armed & (self._prev_vz > 0.0) & (vz <= 0.0) & feet_above_clearance
        height_matched_apex = self._height_matched_apex(is_apex, height)
        self._is_apex = is_apex
        self._last_apex_height = torch.where(is_apex, height, self._last_apex_height)
        self._last_apex_target_height = torch.where(
            is_apex, self.target_apex_height, self._last_apex_target_height
        )

        apex_error = torch.abs(self._last_apex_height - self._last_apex_target_height)
        new_apex_count = torch.where(is_apex, self._apex_count + 1.0, self._apex_count)
        self.metrics["error_peak_height"] = torch.where(
            is_apex,
            self.metrics["error_peak_height"]
            + (apex_error - self.metrics["error_peak_height"]) / new_apex_count.clamp_min(1.0),
            self.metrics["error_peak_height"],
        )
        self._apex_count = new_apex_count
        self._height_matched_apex_count = torch.where(
            height_matched_apex,
            self._height_matched_apex_count + 1.0,
            self._height_matched_apex_count,
        )
        self.metrics["apex_count"][:] = self._apex_count
        self.metrics["height_matched_apex_count"][:] = self._height_matched_apex_count
        self._apex_armed = self._apex_armed & ~is_apex
        self._prev_vz = vz.clone()

    def _resample_command(self, env_ids: Sequence[int]):
        # On env reset, ``reset_drop_from_height`` has already sampled and
        # written the initial target before ``command_manager.reset()`` calls
        # this method. Preserve that reset-time target, but allow normal
        # mid-episode command resampling afterward.
        if isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(device=self.device)
        else:
            ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        resample_ids = ids[self.command_counter[ids] > 0]
        if len(resample_ids) == 0:
            return

        r = self.cfg.ranges
        self._target_apex_height[resample_ids] = torch.empty(len(resample_ids), device=self.device).uniform_(
            *r.peak_height
        )

    def _update_command(self):
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        height = self.robot.data.root_pos_w[:, 2]
        if env_ids is None:
            self._drop_height.copy_(height)
            self._is_apex.zero_()
            self._last_apex_height.zero_()
            self._last_apex_target_height.zero_()
            self._apex_armed.fill_(True)
            self._prev_vz.zero_()
            self._apex_count.zero_()
            self._height_matched_apex_count.zero_()
        else:
            self._drop_height[env_ids] = height[env_ids]
            self._is_apex[env_ids] = False
            self._last_apex_height[env_ids] = 0.0
            self._last_apex_target_height[env_ids] = 0.0
            self._apex_armed[env_ids] = True
            self._prev_vz[env_ids] = 0.0
            self._apex_count[env_ids] = 0.0
            self._height_matched_apex_count[env_ids] = 0.0
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


class EnergyMetricsCommand(CommandTerm):
    """Track motor mechanical work for the rebounce task."""

    cfg: EnergyMetricsCommandCfg

    def __init__(self, cfg: EnergyMetricsCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        cfg.asset_cfg.resolve(env.scene)
        self.robot: Articulation = env.scene[cfg.asset_cfg.name]
        self._joint_ids = cfg.asset_cfg.joint_ids

        n, dev = self.num_envs, self.device
        self._command = torch.zeros(n, 1, device=dev)
        self._positive_power = torch.zeros(n, device=dev)
        self._negative_power = torch.zeros(n, device=dev)
        self._absolute_power = torch.zeros(n, device=dev)

        self._negative_work = torch.zeros(n, device=dev)
        self._absolute_work = torch.zeros(n, device=dev)
        self._positive_interval_work = torch.zeros(n, device=dev)
        self._absolute_interval_work = torch.zeros(n, device=dev)
        self._apex_count = torch.zeros(n, device=dev)
        self._positive_work_per_height = torch.zeros(n, device=dev)
        self._absolute_work_per_height = torch.zeros(n, device=dev)
        self._positive_work_per_height_pulse = torch.zeros(n, device=dev)
        self._absolute_work_per_height_pulse = torch.zeros(n, device=dev)
        self._positive_work_per_target_height_pulse = torch.zeros(n, device=dev)
        self._absolute_work_per_target_height_pulse = torch.zeros(n, device=dev)

        self.metrics["positive_work_per_height"] = torch.zeros(n, device=dev)
        self.metrics["absolute_work_per_height"] = torch.zeros(n, device=dev)
        self.metrics["braking_ratio"] = torch.zeros(n, device=dev)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def positive_power(self) -> torch.Tensor:
        return self._positive_power

    @property
    def negative_power(self) -> torch.Tensor:
        return self._negative_power

    @property
    def absolute_power(self) -> torch.Tensor:
        return self._absolute_power

    def power(self, mode: str) -> torch.Tensor:
        if mode == "positive":
            return self._positive_power
        if mode == "absolute":
            return self._absolute_power
        raise ValueError(f"Unsupported energy penalty mode: {mode!r}. Expected 'positive' or 'absolute'.")

    def work_per_height_pulse(self, mode: str) -> torch.Tensor:
        if mode == "positive":
            return self._positive_work_per_height_pulse
        if mode == "absolute":
            return self._absolute_work_per_height_pulse
        raise ValueError(f"Unsupported energy penalty mode: {mode!r}. Expected 'positive' or 'absolute'.")

    def work_per_target_height_pulse(self, mode: str) -> torch.Tensor:
        if mode == "positive":
            return self._positive_work_per_target_height_pulse
        if mode == "absolute":
            return self._absolute_work_per_target_height_pulse
        raise ValueError(f"Unsupported energy penalty mode: {mode!r}. Expected 'positive' or 'absolute'.")

    def _update_metrics(self):
        torque = self.robot.data.applied_torque[:, self._joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self._joint_ids]
        joint_power = torque * joint_vel

        self._positive_power = torch.sum(torch.clamp(joint_power, min=0.0), dim=1)
        self._negative_power = torch.sum(torch.clamp(-joint_power, min=0.0), dim=1)
        self._absolute_power = self._positive_power + self._negative_power

        dt = self._env.step_dt
        self._negative_work += self._negative_power * dt
        self._absolute_work += self._absolute_power * dt
        self._positive_interval_work += self._positive_power * dt
        self._absolute_interval_work += self._absolute_power * dt

        self._positive_work_per_height_pulse.zero_()
        self._absolute_work_per_height_pulse.zero_()
        self._positive_work_per_target_height_pulse.zero_()
        self._absolute_work_per_target_height_pulse.zero_()

        if self.cfg.apex_command_name is not None:
            apex_cmd = self._env.command_manager.get_term(self.cfg.apex_command_name)
            is_apex = apex_cmd.is_apex
            is_apex_float = is_apex.float()
            apex_height = apex_cmd.last_apex_height.clamp_min(1e-6)
            target_height = apex_cmd.last_apex_target_height.clamp_min(1e-6)
            self._positive_work_per_height_pulse = torch.where(
                is_apex, self._positive_interval_work / apex_height, self._positive_work_per_height_pulse
            )
            self._absolute_work_per_height_pulse = torch.where(
                is_apex, self._absolute_interval_work / apex_height, self._absolute_work_per_height_pulse
            )
            self._positive_work_per_target_height_pulse = torch.where(
                is_apex,
                self._positive_interval_work / target_height,
                self._positive_work_per_target_height_pulse,
            )
            self._absolute_work_per_target_height_pulse = torch.where(
                is_apex,
                self._absolute_interval_work / target_height,
                self._absolute_work_per_target_height_pulse,
            )
            self._positive_interval_work = torch.where(
                is_apex, torch.zeros_like(self._positive_interval_work), self._positive_interval_work
            )
            self._absolute_interval_work = torch.where(
                is_apex, torch.zeros_like(self._absolute_interval_work), self._absolute_interval_work
            )
            new_apex_count = self._apex_count + is_apex_float
            self._positive_work_per_height = torch.where(
                is_apex,
                self._positive_work_per_height
                + (self._positive_work_per_height_pulse - self._positive_work_per_height)
                / new_apex_count.clamp_min(1.0),
                self._positive_work_per_height,
            )
            self._absolute_work_per_height = torch.where(
                is_apex,
                self._absolute_work_per_height
                + (self._absolute_work_per_height_pulse - self._absolute_work_per_height)
                / new_apex_count.clamp_min(1.0),
                self._absolute_work_per_height,
            )
            self._apex_count = new_apex_count

        self.metrics["positive_work_per_height"][:] = self._positive_work_per_height
        self.metrics["absolute_work_per_height"][:] = self._absolute_work_per_height
        self.metrics["braking_ratio"][:] = self._negative_work / self._absolute_work.clamp_min(1e-6)

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if env_ids is None:
            self._positive_power.zero_()
            self._negative_power.zero_()
            self._absolute_power.zero_()
            self._negative_work.zero_()
            self._absolute_work.zero_()
            self._positive_interval_work.zero_()
            self._absolute_interval_work.zero_()
            self._apex_count.zero_()
            self._positive_work_per_height.zero_()
            self._absolute_work_per_height.zero_()
            self._positive_work_per_height_pulse.zero_()
            self._absolute_work_per_height_pulse.zero_()
            self._positive_work_per_target_height_pulse.zero_()
            self._absolute_work_per_target_height_pulse.zero_()
        else:
            self._positive_power[env_ids] = 0.0
            self._negative_power[env_ids] = 0.0
            self._absolute_power[env_ids] = 0.0
            self._negative_work[env_ids] = 0.0
            self._absolute_work[env_ids] = 0.0
            self._positive_interval_work[env_ids] = 0.0
            self._absolute_interval_work[env_ids] = 0.0
            self._apex_count[env_ids] = 0.0
            self._positive_work_per_height[env_ids] = 0.0
            self._absolute_work_per_height[env_ids] = 0.0
            self._positive_work_per_height_pulse[env_ids] = 0.0
            self._absolute_work_per_height_pulse[env_ids] = 0.0
            self._positive_work_per_target_height_pulse[env_ids] = 0.0
            self._absolute_work_per_target_height_pulse[env_ids] = 0.0
        return extras


@configclass
class EnergyMetricsCommandCfg(CommandTermCfg):
    """Config for :class:`EnergyMetricsCommand`."""

    class_type: type = EnergyMetricsCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*"])
    apex_command_name: str | None = "hop"
