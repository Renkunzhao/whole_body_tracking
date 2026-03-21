from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class VerticalSpringPlaneCfg:
    """Configuration for a minimal vertical spring-damper contact model."""

    plane_height: float
    stiffness: float
    damping: float
    contact_radius: float
    max_force: float | None = None


class VerticalSpringPlane:
    """Applies a one-sided vertical spring-damper force in the world frame."""

    def __init__(self, cfg: VerticalSpringPlaneCfg):
        if cfg.contact_radius < 0.0:
            raise ValueError(f"contact_radius must be non-negative, got {cfg.contact_radius}.")
        if cfg.max_force is not None and cfg.max_force < 0.0:
            raise ValueError(f"max_force must be non-negative, got {cfg.max_force}.")
        self.cfg = cfg

    def compute_force(
        self, root_pos_w: torch.Tensor, root_lin_vel_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the world-frame spring force from root position and linear velocity.

        Args:
            root_pos_w: Root positions in the world frame with shape ``(N, 3)``.
            root_lin_vel_w: Root linear velocities in the world frame with shape ``(N, 3)``.

        Returns:
            A tuple of ``(force_w, penetration, bottom_height)`` where:
            - ``force_w`` has shape ``(N, 3)``
            - ``penetration`` has shape ``(N,)``
            - ``bottom_height`` has shape ``(N,)``
        """
        if root_pos_w.ndim != 2 or root_pos_w.shape[-1] != 3:
            raise ValueError(f"Expected root_pos_w to have shape (N, 3), got {tuple(root_pos_w.shape)}.")
        if root_lin_vel_w.shape != root_pos_w.shape:
            raise ValueError(
                "Expected root_lin_vel_w to match root_pos_w shape, "
                f"got {tuple(root_lin_vel_w.shape)} and {tuple(root_pos_w.shape)}."
            )

        bottom_height = root_pos_w[:, 2] - self.cfg.contact_radius
        plane_height = bottom_height.new_full(bottom_height.shape, self.cfg.plane_height)
        penetration = torch.clamp(plane_height - bottom_height, min=0.0)

        force_z = self.cfg.stiffness * penetration - self.cfg.damping * root_lin_vel_w[:, 2]
        force_z = torch.clamp(force_z, min=0.0)
        if self.cfg.max_force is not None:
            force_z = torch.clamp(force_z, max=self.cfg.max_force)

        force_w = torch.zeros_like(root_pos_w)
        force_w[:, 2] = force_z
        return force_w, penetration, bottom_height
