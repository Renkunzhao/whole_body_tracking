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

    ``std_stance`` / ``std_flight`` are dicts mapping joint-name regex to std value,
    e.g. ``{".*_hip_joint": 0.08, ".*_thigh_joint": 0.3, ".*_calf_joint": 0.4}``.
    Every matched joint contributes; unmatched joints in ``asset_cfg.joint_names``
    will raise a strict-mode error from ``resolve_matching_names_values``.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: Articulation = env.scene[asset_cfg.name]
        # asset_cfg is already resolved by the manager; joint_ids is the authoritative list.
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
        cycle_time: float,
        stance_fraction: float,
        std_stance: dict[str, float],
        std_flight: dict[str, float],
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        del std_stance, std_flight, asset_cfg  # resolved in __init__
        asset: Articulation = env.scene[self._asset_name]
        err = asset.data.joint_pos[:, self._joint_ids] - asset.data.default_joint_pos[:, self._joint_ids]

        in_stance = _phase(env, cycle_time) < stance_fraction   # [num_envs]
        # broadcast per-env std selection: [num_envs, num_joints]
        std = torch.where(in_stance.unsqueeze(1), self._std_stance, self._std_flight)
        mean_sq = torch.mean(torch.square(err / std), dim=1)
        return torch.exp(-mean_sq)


def _phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    return torch.remainder(env.episode_length_buf.to(torch.float32) * env.step_dt / cycle_time, 1.0)


def phase_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    cycle_time: float,
    stance_fraction: float,
    contact_threshold: float,
) -> torch.Tensor:
    """Reward all feet matching the stance/flight phase together."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contacts = torch.max(torch.norm(contact_forces, dim=-1), dim=1)[0] > contact_threshold

    in_stance = _phase(env, cycle_time) < stance_fraction
    all_feet_contact = torch.all(contacts, dim=-1)
    all_feet_air = torch.all(~contacts, dim=-1)
    return torch.where(in_stance, all_feet_contact, all_feet_air).float()

def phase_contact_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    cycle_time: float,
    stance_fraction: float,
    contact_height: float,
    softness: float = 0.01,
    surface_z: float = 0.0,
) -> torch.Tensor:
    """Distance-based drop-in for ``phase_contact`` with a sigmoid-soft threshold.

    ``in_contact(foot)`` is a smooth sigmoid around ``surface_z + contact_height``
    with transition width ``softness`` (meters); values near 1 mean the foot is
    below the threshold, near 0 mean above. The stance/flight reward is then the
    (soft) AND of per-foot contact/air, giving continuous gradient even when feet
    only lift by a few millimeters.

    Works for rigid ground (pass the ground height as ``surface_z``) and any
    setup where you can provide a per-env surface height — for a trampoline,
    read the center node z and pass it in via a wrapper.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z_local = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2:3]
    in_contact = torch.sigmoid((surface_z + contact_height - foot_z_local) / softness)

    in_stance = _phase(env, cycle_time) < stance_fraction
    all_feet_contact = in_contact.prod(dim=-1)
    all_feet_air = (1.0 - in_contact).prod(dim=-1)
    return torch.where(in_stance, all_feet_contact, all_feet_air)
