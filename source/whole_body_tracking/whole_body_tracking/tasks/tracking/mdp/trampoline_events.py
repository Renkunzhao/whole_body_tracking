from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaacsim.core.utils.stage import get_current_stage
from pxr import Sdf

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject
from isaaclab.managers import EventTermCfg, ManagerTermBase
from isaaclab.utils.math import sample_log_uniform, sample_uniform

from whole_body_tracking.utils.trampoline_deformable import (
    get_trampoline_damping_scales,
    get_trampoline_dynamic_frictions,
    get_trampoline_elasticity_dampings,
    get_trampoline_poissons_ratios,
    get_trampoline_youngs_moduli,
    set_trampoline_damping_scales,
    set_trampoline_dynamic_frictions,
    set_trampoline_elasticity_dampings,
    set_trampoline_poissons_ratios,
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
        self._default_dynamic_frictions = get_trampoline_dynamic_frictions(self._material_view)
        self._default_elasticity_dampings = get_trampoline_elasticity_dampings(self._material_view)
        self._default_damping_scales = get_trampoline_damping_scales(self._material_view)
        self._default_poissons_ratios = get_trampoline_poissons_ratios(self._material_view)
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

        self._last_youngs_moduli = self._default_youngs_moduli.to(device=env.device).reshape(-1).clone()
        self._last_dynamic_frictions = self._default_dynamic_frictions.to(device=env.device).reshape(-1).clone()
        self._last_elasticity_dampings = self._default_elasticity_dampings.to(device=env.device).reshape(-1).clone()
        self._last_damping_scales = self._default_damping_scales.to(device=env.device).reshape(-1).clone()
        self._last_poissons_ratios = self._default_poissons_ratios.to(device=env.device).reshape(-1).clone()
        self._last_masses = self._default_masses.clone()

    @property
    def last_youngs_moduli(self) -> torch.Tensor:
        """Most recent Young's modulus values written by this reset event."""
        return self._last_youngs_moduli

    @property
    def last_dynamic_frictions(self) -> torch.Tensor:
        """Most recent deformable material dynamic friction values written by this reset event."""
        return self._last_dynamic_frictions

    @property
    def last_elasticity_dampings(self) -> torch.Tensor:
        """Most recent deformable material elasticity damping values written by this reset event."""
        return self._last_elasticity_dampings

    @property
    def last_damping_scales(self) -> torch.Tensor:
        """Most recent deformable material damping scale values written by this reset event."""
        return self._last_damping_scales

    @property
    def last_poissons_ratios(self) -> torch.Tensor:
        """Most recent deformable material Poisson's ratio values written by this reset event."""
        return self._last_poissons_ratios

    @property
    def last_masses(self) -> torch.Tensor:
        """Most recent trampoline mass values written by this reset event."""
        return self._last_masses

    def reset(self, env_ids=None) -> None:
        pass

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor,
        youngs_modulus_range: tuple[float, float],
        youngs_modulus_distribution: str = "uniform",
        mass_range: tuple[float, float] | None = None,
        dynamic_friction_range: tuple[float, float] | None = None,
        elasticity_damping_range: tuple[float, float] | None = None,
        damping_scale_range: tuple[float, float] | None = None,
        poissons_ratio_range: tuple[float, float] | None = None,
        randomization_start_step: int = 0,
        fixed_youngs_modulus_range: tuple[float, float] | None = None,
        fixed_mass_range: tuple[float, float] | None = None,
        fixed_dynamic_friction_range: tuple[float, float] | None = None,
        fixed_elasticity_damping_range: tuple[float, float] | None = None,
        fixed_damping_scale_range: tuple[float, float] | None = None,
        fixed_poissons_ratio_range: tuple[float, float] | None = None,
        asset_name: str = "trampoline",
    ) -> None:
        env_ids_tensor = _resolve_env_ids(env, env_ids)
        use_fixed = randomization_start_step > 0 and env.common_step_counter < randomization_start_step
        if use_fixed:
            youngs_modulus_range = fixed_youngs_modulus_range or youngs_modulus_range
            mass_range = fixed_mass_range if fixed_mass_range is not None else mass_range
            dynamic_friction_range = (
                fixed_dynamic_friction_range if fixed_dynamic_friction_range is not None else dynamic_friction_range
            )
            elasticity_damping_range = (
                fixed_elasticity_damping_range
                if fixed_elasticity_damping_range is not None
                else elasticity_damping_range
            )
            damping_scale_range = fixed_damping_scale_range if fixed_damping_scale_range is not None else damping_scale_range
            poissons_ratio_range = (
                fixed_poissons_ratio_range if fixed_poissons_ratio_range is not None else poissons_ratio_range
            )

        if youngs_modulus_distribution == "uniform":
            youngs_moduli = sample_uniform(
                youngs_modulus_range[0],
                youngs_modulus_range[1],
                (len(env_ids_tensor),),
                device=env.device,
            ).to(dtype=torch.float32)
        elif youngs_modulus_distribution == "log_uniform":
            min_youngs, max_youngs = youngs_modulus_range
            if min_youngs <= 0.0 or max_youngs <= 0.0:
                raise ValueError(
                    "Log-uniform Young's modulus randomization requires positive bounds, "
                    f"got {youngs_modulus_range}."
                )
            youngs_moduli = sample_log_uniform(
                min_youngs,
                max_youngs,
                (len(env_ids_tensor),),
                device=env.device,
            ).to(dtype=torch.float32)
        else:
            raise ValueError(
                f"Unsupported Young's modulus distribution: {youngs_modulus_distribution!r}. "
                "Expected 'uniform' or 'log_uniform'."
            )
        set_trampoline_youngs_moduli(self._material_view, youngs_moduli, env_ids_tensor)
        self._last_youngs_moduli[env_ids_tensor] = youngs_moduli

        if dynamic_friction_range is not None:
            dynamic_frictions = sample_uniform(
                dynamic_friction_range[0],
                dynamic_friction_range[1],
                (len(env_ids_tensor),),
                device=env.device,
            ).to(dtype=torch.float32)
            set_trampoline_dynamic_frictions(self._material_view, dynamic_frictions, env_ids_tensor)
            self._last_dynamic_frictions[env_ids_tensor] = dynamic_frictions

        if elasticity_damping_range is not None:
            elasticity_dampings = sample_uniform(
                elasticity_damping_range[0],
                elasticity_damping_range[1],
                (len(env_ids_tensor),),
                device=env.device,
            ).to(dtype=torch.float32)
            set_trampoline_elasticity_dampings(self._material_view, elasticity_dampings, env_ids_tensor)
            self._last_elasticity_dampings[env_ids_tensor] = elasticity_dampings

        if damping_scale_range is not None:
            damping_scales = sample_uniform(
                damping_scale_range[0],
                damping_scale_range[1],
                (len(env_ids_tensor),),
                device=env.device,
            ).to(dtype=torch.float32)
            set_trampoline_damping_scales(self._material_view, damping_scales, env_ids_tensor)
            self._last_damping_scales[env_ids_tensor] = damping_scales

        if poissons_ratio_range is not None:
            poissons_ratios = sample_uniform(
                poissons_ratio_range[0],
                poissons_ratio_range[1],
                (len(env_ids_tensor),),
                device=env.device,
            ).to(dtype=torch.float32)
            set_trampoline_poissons_ratios(self._material_view, poissons_ratios, env_ids_tensor)
            self._last_poissons_ratios[env_ids_tensor] = poissons_ratios

        if mass_range is None:
            return

        masses = sample_uniform(mass_range[0], mass_range[1], (len(env_ids_tensor),), device=env.device).to(
            dtype=torch.float32
        )
        self._last_masses[env_ids_tensor] = masses
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
