from __future__ import annotations

import logging
from collections.abc import Sequence

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

from whole_body_tracking.utils.point_foot_contact_force import PointFootContactForceCfg, PointFootContactForceModel

logger = logging.getLogger(__name__)


class ContactForceAction(ActionTerm):
    """Applies simplified custom ground-contact wrenches at the selected body origins."""

    cfg: "ContactForceActionCfg"

    def __init__(self, cfg: "ContactForceActionCfg", env):
        super().__init__(cfg, env)

        self._export_IO_descriptor = False
        self._raw_actions = torch.zeros((self.num_envs, 0), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._contact_model = PointFootContactForceModel(
            PointFootContactForceCfg(
                plane_height=cfg.contact_plane_height,
                normal_stiffness=cfg.contact_normal_stiffness,
                normal_damping=cfg.contact_normal_damping,
                tangential_damping=cfg.contact_tangential_damping,
                friction_coeff=cfg.contact_friction_coeff,
            )
        )
        self._contact_enabled = cfg.contact_enabled

        if cfg.contact_body_names is None:
            self._contact_body_ids: list[int] = []
            self._contact_body_names: list[str] = []
            self._num_contact_bodies = 0
        else:
            body_ids, body_names = self._asset.find_bodies(cfg.contact_body_names, preserve_order=True)
            self._contact_body_ids = list(body_ids)
            self._contact_body_names = body_names
            self._num_contact_bodies = len(body_ids)

        self._last_penetration = torch.zeros((self.num_envs, self._num_contact_bodies), device=self.device)
        self._last_normal_force = torch.zeros_like(self._last_penetration)
        self._last_tangential_force_norm = torch.zeros_like(self._last_penetration)
        self._last_contact_active = torch.zeros(
            (self.num_envs, self._num_contact_bodies), device=self.device, dtype=torch.bool
        )

        logger.info(
            "Resolved custom foot contact bodies for %s: %s",
            self.__class__.__name__,
            self._contact_body_names,
        )

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the term."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The actions computed by the term after applying any processing."""
        return self._processed_actions

    @property
    def contact_penetration(self) -> torch.Tensor:
        """Last computed env-local penetration per contact body."""
        return self._last_penetration

    @property
    def contact_normal_force(self) -> torch.Tensor:
        """Last computed normal force magnitude per contact body."""
        return self._last_normal_force

    @property
    def contact_tangential_force_norm(self) -> torch.Tensor:
        """Last computed tangential force magnitude per contact body."""
        return self._last_tangential_force_norm

    @property
    def contact_active(self) -> torch.Tensor:
        """Whether each contact body was penetrating the support plane on the last step."""
        return self._last_contact_active

    def process_actions(self, actions: torch.Tensor):
        """Store the zero-dimensional action slice for API compatibility."""
        self._raw_actions = actions.to(self.device)
        self._processed_actions = self._raw_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids=env_ids)
        self._write_zero_contact_wrench(env_ids=env_ids)
        self._zero_contact_debug_buffers(env_ids=env_ids)

    def apply_actions(self):
        if self._contact_enabled:
            self._apply_custom_foot_contact()
        else:
            self._write_zero_contact_wrench()
            self._zero_contact_debug_buffers()

    def _apply_custom_foot_contact(self) -> None:
        if self._num_contact_bodies == 0:
            self._write_zero_contact_wrench()
            self._zero_contact_debug_buffers()
            return

        body_pos_w = self._asset.data.body_pos_w[:, self._contact_body_ids]
        body_lin_vel_w = self._asset.data.body_lin_vel_w[:, self._contact_body_ids]
        env_origins = self._env.scene.env_origins.unsqueeze(1).expand(-1, self._num_contact_bodies, -1)

        force_w, torque_w, _, penetration, normal_force, tangential_force_norm, contact_active = (
            self._contact_model.compute_wrenches(
                body_pos_w.reshape(-1, 3),
                body_lin_vel_w.reshape(-1, 3),
                env_origins=env_origins.reshape(-1, 3),
            )
        )

        force_w = force_w.view(self.num_envs, self._num_contact_bodies, 3)
        torque_w = torque_w.view(self.num_envs, self._num_contact_bodies, 3)
        self._last_penetration = penetration.view(self.num_envs, self._num_contact_bodies)
        self._last_normal_force = normal_force.view(self.num_envs, self._num_contact_bodies)
        self._last_tangential_force_norm = tangential_force_norm.view(self.num_envs, self._num_contact_bodies)
        self._last_contact_active = contact_active.view(self.num_envs, self._num_contact_bodies)

        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=force_w,
            torques=torque_w,
            body_ids=self._contact_body_ids,
            is_global=True,
        )

    def _write_zero_contact_wrench(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if self._num_contact_bodies == 0:
            return
        if env_ids is None:
            env_ids_arg = slice(None)
            num_envs = self.num_envs
        elif isinstance(env_ids, slice):
            if env_ids != slice(None):
                raise ValueError(f"Unsupported slice for env_ids: {env_ids}")
            env_ids_arg = env_ids
            num_envs = self.num_envs
        else:
            env_ids_arg = env_ids
            num_envs = len(env_ids)
        zero_force = torch.zeros((num_envs, self._num_contact_bodies, 3), device=self.device)
        zero_torque = torch.zeros_like(zero_force)
        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=zero_force,
            torques=zero_torque,
            body_ids=self._contact_body_ids,
            env_ids=env_ids_arg,
            is_global=True,
        )

    def _zero_contact_debug_buffers(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if self._num_contact_bodies == 0:
            return
        if env_ids is None or env_ids == slice(None):
            self._last_penetration.zero_()
            self._last_normal_force.zero_()
            self._last_tangential_force_norm.zero_()
            self._last_contact_active.zero_()
            return

        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._last_penetration[env_ids_tensor] = 0.0
        self._last_normal_force[env_ids_tensor] = 0.0
        self._last_tangential_force_norm[env_ids_tensor] = 0.0
        self._last_contact_active[env_ids_tensor] = False

@configclass
class ContactForceActionCfg(ActionTermCfg):
    """Configuration for custom foot-ground contact applied as a separate action term."""

    class_type: type = ContactForceAction

    contact_enabled: bool = False
    contact_plane_height: float = 0.0
    contact_normal_stiffness: float = 5000.0
    contact_normal_damping: float = 120.0
    contact_tangential_damping: float = 120.0
    contact_friction_coeff: float = 1.0
    contact_body_names: list[str] | None = None
