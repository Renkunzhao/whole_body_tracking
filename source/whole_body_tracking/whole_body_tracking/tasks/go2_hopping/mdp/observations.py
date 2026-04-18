from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def sin_cos_phase(env: ManagerBasedRLEnv, command_name: str = "hop") -> torch.Tensor:
    """Sine/cosine encoding of the hopping phase, read from a hopping command term."""
    phase = env.command_manager.get_term(command_name).phase
    phase_rad = 2.0 * math.pi * phase
    return torch.stack((torch.sin(phase_rad), torch.cos(phase_rad)), dim=-1)
