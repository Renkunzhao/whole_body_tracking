from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from isaacsim.core.utils.stage import get_current_stage
from pxr import Sdf

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject
from isaaclab.managers import EventTermCfg, ManagerTermBase
from isaaclab.utils.math import sample_uniform

from whole_body_tracking.utils.trampoline_deformable import (
    TRAMPOLINE_PIN_WIDTH,
    TRAMPOLINE_SIM_RESOLUTION,
    TRAMPOLINE_SIM_RESOLUTION_ATTR,
    TRAMPOLINE_THICKNESS,
    TRAMPOLINE_THICKNESS_ATTR,
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


def _read_float_attr(prim, name: str) -> float | None:
    attr = prim.GetAttribute(name)
    if not attr.IsValid():
        return None
    value = attr.Get()
    return None if value is None else float(value)


def _read_int_attr(prim, name: str) -> int | None:
    attr = prim.GetAttribute(name)
    if not attr.IsValid():
        return None
    value = attr.Get()
    return None if value is None else int(value)


def _infer_mesh_height(prim) -> float:
    points_attr = prim.GetAttribute("points")
    if not points_attr.IsValid():
        return TRAMPOLINE_THICKNESS
    points = points_attr.Get()
    if points is None or len(points) == 0:
        return TRAMPOLINE_THICKNESS
    z_values = [float(point[2]) for point in points]
    return max(z_values) - min(z_values)


def _youngs_ranges_for_envs(
    sim_resolutions: torch.Tensor,
    env_ids: torch.Tensor,
    default_range: tuple[float, float] | None,
    range_by_sim_resolution: Mapping[int | str, tuple[float, float]] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if default_range is None:
        if range_by_sim_resolution is None:
            raise ValueError(
                "Young's modulus randomization requires either youngs_modulus_range or "
                "youngs_modulus_range_by_sim_resolution."
            )
        min_values = torch.empty((len(env_ids),), device=env_ids.device, dtype=torch.float32)
        max_values = torch.empty((len(env_ids),), device=env_ids.device, dtype=torch.float32)
    else:
        min_values = torch.full((len(env_ids),), float(default_range[0]), device=env_ids.device, dtype=torch.float32)
        max_values = torch.full((len(env_ids),), float(default_range[1]), device=env_ids.device, dtype=torch.float32)
    if range_by_sim_resolution is None:
        return min_values, max_values

    env_sim_resolutions = torch.round(sim_resolutions[env_ids]).to(dtype=torch.long)
    matched = torch.zeros_like(env_sim_resolutions, dtype=torch.bool)
    for sim_resolution, youngs_range in range_by_sim_resolution.items():
        mask = env_sim_resolutions == int(sim_resolution)
        if not torch.any(mask):
            continue
        min_values[mask] = float(youngs_range[0])
        max_values[mask] = float(youngs_range[1])
        matched |= mask
    if not torch.all(matched):
        missing = sorted(set(env_sim_resolutions[~matched].detach().cpu().tolist()))
        raise ValueError(
            "Missing youngs_modulus_range_by_sim_resolution entries for trampoline sim resolutions: "
            f"{missing}."
        )
    return min_values, max_values


def _sample_youngs_moduli(
    sim_resolutions: torch.Tensor,
    env_ids: torch.Tensor,
    youngs_modulus_range: tuple[float, float] | None,
    youngs_modulus_range_by_sim_resolution: Mapping[int | str, tuple[float, float]] | None,
    distribution: str,
) -> torch.Tensor:
    min_youngs, max_youngs = _youngs_ranges_for_envs(
        sim_resolutions,
        env_ids,
        youngs_modulus_range,
        youngs_modulus_range_by_sim_resolution,
    )
    if torch.any(min_youngs > max_youngs):
        raise ValueError("Young's modulus randomization ranges must have min <= max.")

    samples = torch.rand_like(min_youngs)
    if distribution == "uniform":
        return min_youngs + samples * (max_youngs - min_youngs)
    if distribution == "log_uniform":
        if torch.any(min_youngs <= 0.0) or torch.any(max_youngs <= 0.0):
            raise ValueError("Log-uniform Young's modulus randomization requires positive bounds.")
        return torch.exp(torch.log(min_youngs) + samples * (torch.log(max_youngs) - torch.log(min_youngs)))
    raise ValueError(
        f"Unsupported Young's modulus distribution: {distribution!r}. Expected 'uniform' or 'log_uniform'."
    )


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
        self._thicknesses = torch.zeros((env.scene.num_envs,), device=env.device, dtype=torch.float32)
        self._sim_resolutions = torch.zeros((env.scene.num_envs,), device=env.device, dtype=torch.float32)
        for env_id, prim_path in enumerate(mesh_prim_paths):
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(f"Invalid trampoline mesh prim: '{prim_path}'.")
            self._thicknesses[env_id] = _read_float_attr(prim, TRAMPOLINE_THICKNESS_ATTR) or _infer_mesh_height(prim)
            self._sim_resolutions[env_id] = float(
                _read_int_attr(prim, TRAMPOLINE_SIM_RESOLUTION_ATTR) or TRAMPOLINE_SIM_RESOLUTION
            )
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
        self._last_pin_widths = torch.full(
            (env.scene.num_envs,),
            float(cfg.params.get("default_pin_width", TRAMPOLINE_PIN_WIDTH)),
            device=env.device,
            dtype=torch.float32,
        )

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

    @property
    def last_pin_widths(self) -> torch.Tensor:
        """Most recent trampoline pin-width values written by this reset event."""
        return self._last_pin_widths

    @property
    def trampoline_thicknesses(self) -> torch.Tensor:
        """Spawn-time cooked FEM mesh thickness for each trampoline."""
        return self._thicknesses

    @property
    def trampoline_sim_resolutions(self) -> torch.Tensor:
        """Spawn-time simulation hexahedral resolution for each trampoline."""
        return self._sim_resolutions

    def reset(self, env_ids=None) -> None:
        pass

    def _randomize_pin_widths(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor,
        pin_width_range: tuple[float, float],
        action_term_name: str,
    ) -> None:
        pin_widths = sample_uniform(pin_width_range[0], pin_width_range[1], (len(env_ids),), device=env.device).to(
            dtype=torch.float32
        )
        self._last_pin_widths[env_ids] = pin_widths
        try:
            action_term = env.action_manager.get_term(action_term_name)
        except (AttributeError, KeyError, ValueError) as exc:
            raise RuntimeError(
                f"Trampoline pin-width randomization requires action term '{action_term_name}'."
            ) from exc
        if not hasattr(action_term, "set_pin_widths"):
            raise RuntimeError(f"Action term '{action_term_name}' does not support pin-width randomization.")
        action_term.set_pin_widths(pin_widths, env_ids)

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor,
        youngs_modulus_range: tuple[float, float] | None = None,
        youngs_modulus_range_by_sim_resolution: Mapping[int | str, tuple[float, float]] | None = None,
        youngs_modulus_distribution: str = "uniform",
        mass_range: tuple[float, float] | None = None,
        dynamic_friction_range: tuple[float, float] | None = None,
        elasticity_damping_range: tuple[float, float] | None = None,
        damping_scale_range: tuple[float, float] | None = None,
        poissons_ratio_range: tuple[float, float] | None = None,
        pin_width_range: tuple[float, float] | None = None,
        randomization_start_step: int = 0,
        fixed_youngs_modulus_range: tuple[float, float] | None = None,
        fixed_youngs_modulus_range_by_sim_resolution: Mapping[int | str, tuple[float, float]] | None = None,
        fixed_mass_range: tuple[float, float] | None = None,
        fixed_dynamic_friction_range: tuple[float, float] | None = None,
        fixed_elasticity_damping_range: tuple[float, float] | None = None,
        fixed_damping_scale_range: tuple[float, float] | None = None,
        fixed_poissons_ratio_range: tuple[float, float] | None = None,
        fixed_pin_width_range: tuple[float, float] | None = None,
        pinning_action_term_name: str = "trampoline_pin",
        asset_name: str = "trampoline",
    ) -> None:
        env_ids_tensor = _resolve_env_ids(env, env_ids)
        use_fixed = randomization_start_step > 0 and env.common_step_counter < randomization_start_step
        if use_fixed:
            if fixed_youngs_modulus_range_by_sim_resolution is not None:
                youngs_modulus_range_by_sim_resolution = fixed_youngs_modulus_range_by_sim_resolution
            elif fixed_youngs_modulus_range is not None:
                youngs_modulus_range_by_sim_resolution = None
            if fixed_youngs_modulus_range is not None:
                youngs_modulus_range = fixed_youngs_modulus_range
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
            pin_width_range = fixed_pin_width_range if fixed_pin_width_range is not None else pin_width_range

        youngs_moduli = _sample_youngs_moduli(
            self._sim_resolutions,
            env_ids_tensor,
            youngs_modulus_range,
            youngs_modulus_range_by_sim_resolution,
            youngs_modulus_distribution,
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

        if pin_width_range is not None:
            self._randomize_pin_widths(env, env_ids_tensor, pin_width_range, pinning_action_term_name)

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
