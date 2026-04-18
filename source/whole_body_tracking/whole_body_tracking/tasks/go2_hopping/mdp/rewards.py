from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
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


def air_time_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """Pulse reward on landing for matching the commanded single-hop flight time.

    Fires exactly one step per air->contact transition with value
    ``exp(-((last_air_time - t_flight*) / std)^2)``. Because the pulse is tied
    to a single continuous air phase, splitting one commanded flight window
    into multiple short hops incurs a much larger squared error on each landing
    than a single correctly-timed hop, breaking the double-hop local optimum
    that ``phase_contact`` alone admits.
    """
    cmd = env.command_manager.get_term(command_name)
    err = cmd.last_air_time - cmd.flight_time_target
    return cmd.just_landed.float() * torch.exp(-torch.square(err / std))


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
