from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaacsim.core.utils.stage import get_current_stage
from pxr import Sdf

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject
from isaaclab.managers import EventTermCfg, ManagerTermBase
from isaaclab.utils.math import sample_uniform

from whole_body_tracking.utils.trampoline_deformable import (
    get_trampoline_youngs_moduli,
    set_trampoline_youngs_moduli,
    trampoline_mesh_prim_path,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _resolve_env_ids(env: "ManagerBasedEnv", env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        if env_ids != slice(None):
            raise ValueError(f"Unsupported slice for env_ids: {env_ids}")
        return torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


class RandomizeTrampolineProperties(ManagerTermBase):
    """Randomize deformable trampoline material stiffness and optional mass on reset."""

    def __init__(self, cfg: EventTermCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        asset_name = cfg.params.get("asset_name", "trampoline")
        self._asset: DeformableObject = env.scene[asset_name]
        self._material_view = self._asset.material_physx_view
        if self._material_view is None:
            raise RuntimeError("Trampoline randomization requires a deformable material view.")

        self._default_youngs_moduli = get_trampoline_youngs_moduli(self._material_view)
        mesh_prim_paths = sim_utils.find_matching_prim_paths(trampoline_mesh_prim_path(self._asset.cfg.prim_path))
        if len(mesh_prim_paths) != env.scene.num_envs:
            raise RuntimeError(
                f"Expected {env.scene.num_envs} trampoline mesh prims, found {len(mesh_prim_paths)} for "
                f"pattern '{trampoline_mesh_prim_path(self._asset.cfg.prim_path)}'."
            )

        stage = get_current_stage()
        self._mass_attrs = []
        self._default_masses = torch.zeros((env.scene.num_envs,), device=env.device, dtype=torch.float32)
        for env_id, prim_path in enumerate(mesh_prim_paths):
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(f"Invalid trampoline mesh prim: '{prim_path}'.")
            attr = prim.GetAttribute("physics:mass")
            if not attr.IsValid():
                attr = prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float)
            value = attr.Get()
            if value is None:
                raise RuntimeError(f"Trampoline mesh prim '{prim_path}' is missing a default physics:mass value.")
            self._mass_attrs.append(attr)
            self._default_masses[env_id] = float(value)

    def reset(self, env_ids=None) -> None:
        pass

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor,
        youngs_modulus_range: tuple[float, float],
        mass_range: tuple[float, float] | None = None,
        asset_name: str = "trampoline",
    ) -> None:
        env_ids_tensor = _resolve_env_ids(env, env_ids)

        youngs_moduli = sample_uniform(
            youngs_modulus_range[0],
            youngs_modulus_range[1],
            (len(env_ids_tensor),),
            device=env.device,
        ).to(dtype=torch.float32)
        set_trampoline_youngs_moduli(self._material_view, youngs_moduli, env_ids_tensor)

        if mass_range is None:
            return

        masses = sample_uniform(mass_range[0], mass_range[1], (len(env_ids_tensor),), device=env.device).to(
            dtype=torch.float32
        )
        with Sdf.ChangeBlock():
            for env_id, mass in zip(env_ids_tensor.tolist(), masses.tolist(), strict=True):
                self._mass_attrs[env_id].Set(float(mass))


def reapply_trampoline_pinning(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    action_term_name: str = "trampoline_pin",
) -> None:
    """Refresh and rewrite trampoline pinning targets after reset randomization."""
    term = env.action_manager.get_term(action_term_name)
    term.refresh_targets(env_ids)
    term.write_targets(env_ids)
