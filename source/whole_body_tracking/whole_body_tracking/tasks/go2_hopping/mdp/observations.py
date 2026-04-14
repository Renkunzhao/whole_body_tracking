from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    """Current normalized hopping phase in [0, 1)."""
    return torch.remainder(env.episode_length_buf.to(torch.float32) * env.step_dt / cycle_time, 1.0)


def sin_cos_phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    """Sine/cosine encoding of the hopping phase."""
    phase_rad = 2.0 * math.pi * phase(env, cycle_time)
    return torch.stack((torch.sin(phase_rad), torch.cos(phase_rad)), dim=-1)
