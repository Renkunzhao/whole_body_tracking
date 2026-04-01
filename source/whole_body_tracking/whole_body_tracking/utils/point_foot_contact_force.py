from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class PointFootContactForceCfg:
    """Configuration for a simplified flat-plane contact model."""

    plane_height: float
    normal_stiffness: float
    normal_damping: float
    tangential_damping: float
    friction_coeff: float


class PointFootContactForceModel:
    """Compute world-frame wrenches for a simplified flat-plane contact model."""

    def __init__(self, cfg: PointFootContactForceCfg):
        if cfg.normal_stiffness < 0.0:
            raise ValueError(f"normal_stiffness must be non-negative, got {cfg.normal_stiffness}.")
        if cfg.normal_damping < 0.0:
            raise ValueError(f"normal_damping must be non-negative, got {cfg.normal_damping}.")
        if cfg.tangential_damping < 0.0:
            raise ValueError(f"tangential_damping must be non-negative, got {cfg.tangential_damping}.")
        if cfg.friction_coeff < 0.0:
            raise ValueError(f"friction_coeff must be non-negative, got {cfg.friction_coeff}.")
        self.cfg = cfg

    def compute_wrenches(
        self,
        body_pos_w: torch.Tensor,
        body_lin_vel_w: torch.Tensor,
        env_origins: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute world-frame contact wrench for point contacts.

        Args:
            body_pos_w: Body origin positions with shape ``(N, 3)``.
            body_lin_vel_w: Body linear velocities with shape ``(N, 3)``.
            env_origins: Optional environment origins with shape ``(N, 3)``. When provided, the plane height
                is evaluated in the env-local frame.

        Returns:
            A tuple ``(force_w, torque_w, contact_pos_w, penetration, normal_force, tangential_force_norm,
            contact_active)``.

        Notes:
            Contact is evaluated directly at the body origin, torque is zero, and the tangential response is
            viscous tangential damping capped by Coulomb friction, without tangential spring or anchor state.
        """
        if body_pos_w.ndim != 2 or body_pos_w.shape[-1] != 3:
            raise ValueError(f"Expected body_pos_w to have shape (N, 3), got {tuple(body_pos_w.shape)}.")
        if body_lin_vel_w.shape != body_pos_w.shape:
            raise ValueError(
                "Expected body_lin_vel_w to match body_pos_w shape, "
                f"got {tuple(body_lin_vel_w.shape)} and {tuple(body_pos_w.shape)}."
            )
        if env_origins is not None and env_origins.shape != body_pos_w.shape:
            raise ValueError(
                "Expected env_origins to match body_pos_w shape, "
                f"got {tuple(env_origins.shape)} and {tuple(body_pos_w.shape)}."
            )

        contact_pos_w = body_pos_w
        contact_vel_w = body_lin_vel_w

        contact_height = contact_pos_w[:, 2]
        if env_origins is not None:
            contact_height = contact_height - env_origins[:, 2]

        plane_height = contact_height.new_full(contact_height.shape, self.cfg.plane_height)
        penetration = torch.clamp(plane_height - contact_height, min=0.0)
        contact_active = penetration > 0.0

        normal_force = self.cfg.normal_stiffness * penetration - self.cfg.normal_damping * contact_vel_w[:, 2]
        normal_force = torch.clamp(normal_force, min=0.0)
        normal_force = torch.where(contact_active, normal_force, torch.zeros_like(normal_force))

        total_force_w = torch.zeros_like(body_pos_w)
        total_force_w[:, 2] = normal_force

        tangential_vel_w = contact_vel_w.clone()
        tangential_vel_w[:, 2] = 0.0

        tangential_force_w = -self.cfg.tangential_damping * tangential_vel_w
        tangential_force_w = torch.where(
            contact_active.unsqueeze(-1), tangential_force_w, torch.zeros_like(tangential_force_w)
        )
        tangential_force_norm = torch.linalg.vector_norm(tangential_force_w, dim=-1)
        tangential_force_scale = torch.ones_like(tangential_force_norm)

        active_tangent = contact_active & (tangential_force_norm > 0.0)
        if torch.any(active_tangent):
            safe_norm = tangential_force_norm[active_tangent]
            max_tangential_force = self.cfg.friction_coeff * normal_force[active_tangent]
            tangential_force_scale[active_tangent] = torch.clamp(max_tangential_force / safe_norm, max=1.0)

        tangential_force_w *= tangential_force_scale.unsqueeze(-1)
        tangential_force_norm = torch.linalg.vector_norm(tangential_force_w, dim=-1)

        total_force_w += tangential_force_w

        torque_w = torch.zeros_like(total_force_w)

        return (
            total_force_w,
            torque_w,
            contact_pos_w,
            penetration,
            normal_force,
            tangential_force_norm,
            contact_active,
        )
