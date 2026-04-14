from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
