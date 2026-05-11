from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from whole_body_tracking.sensors import get_or_create_dob_contact_sensor

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


def get_dob_contact_sensor(env: ManagerBasedRLEnv, backend: str = "gpu"):
    return get_or_create_dob_contact_sensor(env, backend=backend)


def dob_contact_forces(
    env: ManagerBasedRLEnv,
    force_scale: float = 1000.0,
    backend: str = "gpu",
) -> torch.Tensor:
    """Flattened per-foot DOB contact forces for privileged observations."""
    sensor = get_dob_contact_sensor(env, backend=backend)
    return sensor.data.foot_forces_w.reshape(env.num_envs, -1) / force_scale


def dob_contact_energy(
    env: ManagerBasedRLEnv,
    force_scale: float = 1000.0,
    power_scale: float = 1000.0,
    work_scale: float = 100.0,
    backend: str = "gpu",
) -> torch.Tensor:
    """Compact DOB contact energy state for privileged observations."""
    sensor = get_dob_contact_sensor(env, backend=backend)
    positive = sensor.data.hop_positive_work / work_scale
    negative = sensor.data.hop_negative_work / work_scale
    return_ratio = sensor.data.hop_positive_work / torch.clamp(-sensor.data.hop_negative_work, min=1.0e-6)
    return torch.stack(
        (
            sensor.data.total_force_w[:, 2] / force_scale,
            sensor.data.contact_power / power_scale,
            positive,
            negative,
            torch.clamp(return_ratio, 0.0, 5.0),
        ),
        dim=-1,
    )
