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


def trampoline_properties(
    env: ManagerBasedRLEnv,
    event_name: str = "randomize_trampoline_properties",
    youngs_modulus_nominal: float = 8.0e4,
    mass_nominal: float = 10.0,
    mass_scale: float = 5.0,
    poissons_ratio_nominal: float = 0.35,
    poissons_ratio_scale: float = 0.10,
    dynamic_friction_nominal: float = 0.8,
    dynamic_friction_scale: float = 0.4,
    elasticity_damping_nominal: float = 0.02,
) -> torch.Tensor:
    """Normalized privileged trampoline parameters for teacher policies."""
    try:
        randomizer = env.event_manager.get_term_cfg(event_name).func
    except (AttributeError, KeyError, ValueError):
        return torch.zeros(env.num_envs, 5, device=env.device)

    def _value(name: str, default: float) -> torch.Tensor:
        value = getattr(randomizer, name, None)
        if value is None:
            return torch.full((env.num_envs,), default, device=env.device)
        return value.to(device=env.device, dtype=torch.float32).reshape(-1)[: env.num_envs]

    youngs_modulus = _value("last_youngs_moduli", youngs_modulus_nominal)
    mass = _value("last_masses", mass_nominal)
    poissons_ratio = _value("last_poissons_ratios", poissons_ratio_nominal)
    dynamic_friction = _value("last_dynamic_frictions", dynamic_friction_nominal)
    elasticity_damping = _value("last_elasticity_dampings", elasticity_damping_nominal)

    return torch.stack(
        (
            torch.log(youngs_modulus.clamp_min(1.0e-6) / youngs_modulus_nominal),
            (mass - mass_nominal) / mass_scale,
            (poissons_ratio - poissons_ratio_nominal) / poissons_ratio_scale,
            (dynamic_friction - dynamic_friction_nominal) / dynamic_friction_scale,
            torch.log(elasticity_damping.clamp_min(1.0e-6) / elasticity_damping_nominal),
        ),
        dim=-1,
    )
