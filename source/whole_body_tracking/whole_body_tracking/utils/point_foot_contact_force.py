from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class PointFootContactForceCfg:
    """Configuration for a point-foot ground contact model on a flat plane."""

    plane_height: float
    normal_stiffness: float
    normal_damping: float
    tangential_stiffness: float
    tangential_damping: float
    friction_coeff: float


class PointFootContactForceModel:
    """Compute world-frame wrenches for point-foot contact against a flat plane."""

    def __init__(self, cfg: PointFootContactForceCfg):
        if cfg.normal_stiffness < 0.0:
            raise ValueError(f"normal_stiffness must be non-negative, got {cfg.normal_stiffness}.")
        if cfg.normal_damping < 0.0:
            raise ValueError(f"normal_damping must be non-negative, got {cfg.normal_damping}.")
        if cfg.tangential_stiffness < 0.0:
            raise ValueError(f"tangential_stiffness must be non-negative, got {cfg.tangential_stiffness}.")
        if cfg.tangential_damping < 0.0:
            raise ValueError(f"tangential_damping must be non-negative, got {cfg.tangential_damping}.")
        if cfg.friction_coeff < 0.0:
            raise ValueError(f"friction_coeff must be non-negative, got {cfg.friction_coeff}.")
        self.cfg = cfg

    def compute_wrenches(
        self,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        body_lin_vel_w: torch.Tensor,
        body_ang_vel_w: torch.Tensor,
        contact_point_offsets_local: torch.Tensor,
        env_origins: torch.Tensor | None = None,
        tangential_displacement_w: torch.Tensor | None = None,
        dt: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute world-frame contact wrench for point contacts.

        Args:
            body_pos_w: Body origin positions with shape ``(N, 3)``.
            body_quat_w: Body orientations as ``(w, x, y, z)`` with shape ``(N, 4)``.
            body_lin_vel_w: Body linear velocities with shape ``(N, 3)``.
            body_ang_vel_w: Body angular velocities with shape ``(N, 3)``.
            contact_point_offsets_local: Local-frame contact point offsets with shape ``(N, 3)``.
            env_origins: Optional environment origins with shape ``(N, 3)``. When provided, the plane height
                is evaluated in the env-local frame.
            tangential_displacement_w: Optional world-frame tangential spring displacement state with shape ``(N, 3)``.
            dt: Optional physics step used to integrate the tangential spring state.

        Returns:
            A tuple ``(force_w, torque_w, contact_pos_w, penetration, normal_force, tangential_force_norm,
            contact_active, tangential_displacement_w)``.

        Notes:
            This simplified variant keeps the same interface but ignores ``body_quat_w``, ``body_ang_vel_w``,
            and ``contact_point_offsets_local``. Contact is evaluated directly at the body origin, and torque is zero.
        """
        if body_pos_w.ndim != 2 or body_pos_w.shape[-1] != 3:
            raise ValueError(f"Expected body_pos_w to have shape (N, 3), got {tuple(body_pos_w.shape)}.")
        if body_quat_w.ndim != 2 or body_quat_w.shape[-1] != 4:
            raise ValueError(f"Expected body_quat_w to have shape (N, 4), got {tuple(body_quat_w.shape)}.")
        if body_lin_vel_w.shape != body_pos_w.shape:
            raise ValueError(
                "Expected body_lin_vel_w to match body_pos_w shape, "
                f"got {tuple(body_lin_vel_w.shape)} and {tuple(body_pos_w.shape)}."
            )
        if body_ang_vel_w.shape != body_pos_w.shape:
            raise ValueError(
                "Expected body_ang_vel_w to match body_pos_w shape, "
                f"got {tuple(body_ang_vel_w.shape)} and {tuple(body_pos_w.shape)}."
            )
        if contact_point_offsets_local.shape != body_pos_w.shape:
            raise ValueError(
                "Expected contact_point_offsets_local to match body_pos_w shape, "
                f"got {tuple(contact_point_offsets_local.shape)} and {tuple(body_pos_w.shape)}."
            )
        if env_origins is not None and env_origins.shape != body_pos_w.shape:
            raise ValueError(
                "Expected env_origins to match body_pos_w shape, "
                f"got {tuple(env_origins.shape)} and {tuple(body_pos_w.shape)}."
            )
        if tangential_displacement_w is not None and tangential_displacement_w.shape != body_pos_w.shape:
            raise ValueError(
                "Expected tangential_displacement_w to match body_pos_w shape, "
                f"got {tuple(tangential_displacement_w.shape)} and {tuple(body_pos_w.shape)}."
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

        if tangential_displacement_w is None:
            tangential_displacement_w = torch.zeros_like(body_pos_w)
        tangential_displacement_w = torch.where(
            contact_active.unsqueeze(-1), tangential_displacement_w, torch.zeros_like(tangential_displacement_w)
        )
        tangential_displacement_w[:, 2] = 0.0

        if dt is not None and self.cfg.tangential_stiffness > 0.0:
            tangential_displacement_w = tangential_displacement_w + tangential_vel_w * dt
            tangential_displacement_w[:, 2] = 0.0

        tangential_force_w = -self.cfg.tangential_damping * tangential_vel_w
        if self.cfg.tangential_stiffness > 0.0:
            tangential_force_w = tangential_force_w - self.cfg.tangential_stiffness * tangential_displacement_w
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

        slip_contacts = active_tangent & (tangential_force_scale < 1.0)
        if torch.any(slip_contacts) and self.cfg.tangential_stiffness > 0.0:
            tangential_displacement_w[slip_contacts] = -(
                tangential_force_w[slip_contacts] + self.cfg.tangential_damping * tangential_vel_w[slip_contacts]
            ) / self.cfg.tangential_stiffness
            tangential_displacement_w[slip_contacts, 2] = 0.0

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
            tangential_displacement_w,
        )
