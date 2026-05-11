from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import wrap_to_pi
from isaaclab.utils.string import resolve_matching_names_values

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class joint_deviation_phase_exp(ManagerTermBase):
    """Per-joint exponential posture reward with different std for stance and flight.

    Reward: ``exp(-mean((err / std_per_joint)^2))`` where ``std_per_joint`` is picked
    per-joint and per-phase (stance vs flight) from the provided regex-to-std dicts.
    The stance/flight split is driven by the hopping command's ``phase`` and
    ``stance_fraction`` (both per-env), not by a global clock.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: Articulation = env.scene[asset_cfg.name]
        joint_names = [asset.joint_names[i] for i in asset_cfg.joint_ids]

        _, _, stance_vals = resolve_matching_names_values(cfg.params["std_stance"], joint_names)
        _, _, flight_vals = resolve_matching_names_values(cfg.params["std_flight"], joint_names)
        self._std_stance = torch.tensor(stance_vals, device=env.device, dtype=torch.float32)
        self._std_flight = torch.tensor(flight_vals, device=env.device, dtype=torch.float32)
        self._joint_ids = asset_cfg.joint_ids
        self._asset_name = asset_cfg.name

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        std_stance: dict[str, float],
        std_flight: dict[str, float],
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        del std_stance, std_flight, asset_cfg  # resolved in __init__
        asset: Articulation = env.scene[self._asset_name]
        err = asset.data.joint_pos[:, self._joint_ids] - asset.data.default_joint_pos[:, self._joint_ids]

        cmd = env.command_manager.get_term(command_name)
        in_stance = cmd.phase < cmd.stance_fraction  # [num_envs]
        std = torch.where(in_stance.unsqueeze(1), self._std_stance, self._std_flight)
        mean_sq = torch.mean(torch.square(err / std), dim=1)
        return torch.exp(-mean_sq)


def phase_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    contact_threshold: float,
) -> torch.Tensor:
    """Reward all feet matching the commanded stance/flight phase together."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contacts = torch.max(torch.norm(contact_forces, dim=-1), dim=1)[0] > contact_threshold

    cmd = env.command_manager.get_term(command_name)
    in_stance = cmd.phase < cmd.stance_fraction
    all_feet_contact = torch.all(contacts, dim=-1)
    all_feet_air = torch.all(~contacts, dim=-1)
    return torch.where(in_stance, all_feet_contact, all_feet_air).float()


class track_air_time(ManagerTermBase):
    """At-most-one-pulse-per-cycle landing reward for matching commanded flight time.

    Each hop cycle (between two consecutive phase wraps of the hopping command)
    can fire the landing pulse at most once. The pulse fires on the first
    air->contact transition that lands during the commanded stance window
    (``phase < stance_fraction``); subsequent mini-bounces in the same cycle
    yield zero, blocking the ``farm pulses by jittering during stance'' exploit
    that ``phase_contact`` alone cannot suppress. The pulse re-arms on phase
    wrap (cycle boundary) and on env reset. Pulse value is
    ``exp(-((last_air_time - t_flight*) / std)^2)``.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, dev = env.num_envs, env.device
        self._armed = torch.ones(n, dtype=torch.bool, device=dev)
        self._prev_phase = torch.zeros(n, device=dev)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        std: float,
    ) -> torch.Tensor:
        cmd = env.command_manager.get_term(command_name)
        phase = cmd.phase

        # Re-arm at phase wrap (phase decreased → new cycle started). Also
        # re-arms after env reset, since episode_length_buf resets and phase
        # falls back to 0 from whatever stale prev_phase we held.
        wrapped = phase < self._prev_phase
        self._armed = self._armed | wrapped
        self._prev_phase = phase.clone()

        in_stance = phase < cmd.stance_fraction
        fire = cmd.just_landed & in_stance & self._armed
        # Disarm after firing — subsequent landings in the same cycle yield 0.
        self._armed = self._armed & ~fire

        err = cmd.last_air_time - cmd.flight_time_target
        return fire.float() * torch.exp(-torch.square(err / std))


class in_place_xy_yaw_l2(ManagerTermBase):
    """Penalty for drifting away from the reset xy position and reset yaw."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset_name = asset_cfg.name
        self._xy0 = torch.zeros(env.num_envs, 2, device=env.device)
        self._yaw0 = torch.zeros(env.num_envs, device=env.device)
        self.reset()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        xy_std: float,
        yaw_std: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        del asset_cfg  # resolved in __init__
        asset: Articulation = env.scene[self._asset_name]
        xy_error = asset.data.root_pos_w[:, :2] - self._xy0
        yaw_error = wrap_to_pi(asset.data.heading_w - self._yaw0)
        return torch.sum(torch.square(xy_error / xy_std), dim=1) + torch.square(yaw_error / yaw_std)

    def reset(self, env_ids=None):
        asset: Articulation = self._env.scene[self._asset_name]
        if env_ids is None:
            self._xy0.copy_(asset.data.root_pos_w[:, :2])
            self._yaw0.copy_(asset.data.heading_w)
        else:
            self._xy0[env_ids] = asset.data.root_pos_w[env_ids, :2]
            self._yaw0[env_ids] = asset.data.heading_w[env_ids]


class rebounce_height_tracking_exp(ManagerTermBase):
    """One-step reward at each command-detected valid apex.

    The target is not "jump higher forever"; it is "reach the commanded
    apex height". The command term owns the apex state machine and latches the
    apex height/target at the detected event. This reward intentionally consumes
    that command event one manager step later, with value
    ``exp(-((apex_height - target_apex_height) / std)^2)`` multiplied by a flat-orientation gate.
    Foot clearance is handled by the command-owned valid-apex detector, not by
    another soft gate in this reward.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset_name = asset_cfg.name

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        std: float,
        orientation_std: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        del asset_cfg  # resolved in __init__
        robot: Articulation = env.scene[self._asset_name]
        cmd = env.command_manager.get_term(command_name)

        orientation_error = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)
        orientation_reward = torch.exp(-orientation_error / (orientation_std * orientation_std))
        height_error = torch.abs(cmd.last_apex_height - cmd.last_apex_target_height)
        return cmd.is_apex.float() * torch.exp(-torch.square(height_error / std)) * orientation_reward


def joint_mechanical_energy_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    mode: str = "positive",
) -> torch.Tensor:
    """Sparse motor-work-per-height penalty from :class:`EnergyMetricsCommand`.

    The command term emits a one-step pulse at each valid apex with work
    accumulated since the previous apex divided by the commanded target apex
    height. This keeps energy optimization from making the jump higher just to
    dilute the work-per-height penalty. Isaac Lab applies the global reward
    ``dt`` scaling.
    """
    energy_cmd = env.command_manager.get_term(command_name)
    return energy_cmd.work_per_target_height_pulse(mode)


class termination_term(ManagerTermBase):
    """Reward/penalty pulse for selected termination terms, including timeouts."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> torch.Tensor:
        del term_keys
        value = torch.zeros(env.num_envs, device=env.device)
        for term_name in self._term_names:
            value += env.termination_manager.get_term(term_name).float()
        return value


def phase_contact_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    contact_height: float,
    softness: float = 0.01,
    surface_z: float = 0.0,
) -> torch.Tensor:
    """Distance-based drop-in for ``phase_contact`` with a sigmoid-soft threshold.

    ``in_contact(foot)`` is a smooth sigmoid around ``surface_z + contact_height``
    with transition width ``softness`` (meters); values near 1 mean the foot is
    below the threshold, near 0 mean above. The stance/flight reward is then the
    (soft) AND of per-foot contact/air, giving continuous gradient even when feet
    only lift by a few millimeters. Works on rigid ground and, with a wrapper
    that supplies ``surface_z`` per env, on a trampoline too.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z_local = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2:3]
    in_contact = torch.sigmoid((surface_z + contact_height - foot_z_local) / softness)

    cmd = env.command_manager.get_term(command_name)
    in_stance = cmd.phase < cmd.stance_fraction
    all_feet_contact = in_contact.prod(dim=-1)
    all_feet_air = (1.0 - in_contact).prod(dim=-1)
    return torch.where(in_stance, all_feet_contact, all_feet_air)
