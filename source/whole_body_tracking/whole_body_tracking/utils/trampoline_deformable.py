from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from pxr import Sdf

TRAMPOLINE_RADIUS = 1.5
TRAMPOLINE_THICKNESS = 0.03
TRAMPOLINE_TOP_Z = 0.0
TRAMPOLINE_CENTER_Z = TRAMPOLINE_TOP_Z - 0.5 * TRAMPOLINE_THICKNESS
TRAMPOLINE_PIN_WIDTH = 0.4
TRAMPOLINE_CONTACT_OFFSET = 0.01
TRAMPOLINE_MASS = 10.0
TRAMPOLINE_YOUNGS_MODULUS = 8.0e4
TRAMPOLINE_SIM_RESOLUTION = 10
TRAMPOLINE_THICKNESS_ATTR = "userProperties:trampolineThickness"
TRAMPOLINE_SIM_RESOLUTION_ATTR = "userProperties:trampolineSimResolution"


def _set_or_create_attr(prim, name: str, type_name, value) -> None:
    attr = prim.GetAttribute(name)
    if not attr.IsValid():
        attr = prim.CreateAttribute(name, type_name)
    attr.Set(value)


def _make_trampoline_mesh_cfg(
    *,
    thickness: float,
    mass: float,
    youngs_modulus: float,
    sim_resolution: int,
    contact_offset: float,
) -> sim_utils.MeshCylinderCfg:
    return sim_utils.MeshCylinderCfg(
        radius=TRAMPOLINE_RADIUS,
        height=thickness,
        axis="Z",
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(
            solver_position_iteration_count=24,
            vertex_velocity_damping=0.05,
            sleep_damping=1.0,
            sleep_threshold=0.01,
            settling_threshold=0.02,
            self_collision=False,
            simulation_hexahedral_resolution=sim_resolution,
            contact_offset=contact_offset,
            rest_offset=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.95), metallic=0.05),
        physics_material=sim_utils.DeformableBodyMaterialCfg(
            dynamic_friction=0.8,
            youngs_modulus=youngs_modulus,
            poissons_ratio=0.35,
            elasticity_damping=0.02,
            damping_scale=1.0,
        ),
    )


def spawn_top_aligned_trampoline_cylinder(
    prim_path: str,
    cfg: sim_utils.MeshCylinderCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a cylinder whose top surface is aligned to the requested translation z."""
    top_translation = (0.0, 0.0, TRAMPOLINE_TOP_Z) if translation is None else translation
    center_translation = (
        float(top_translation[0]),
        float(top_translation[1]),
        float(top_translation[2]) - 0.5 * float(cfg.height),
    )
    prim = sim_utils.spawn_mesh_cylinder(prim_path, cfg, center_translation, orientation, **kwargs)
    mesh_prim = sim_utils.get_current_stage().GetPrimAtPath(trampoline_mesh_prim_path(str(prim_path)))
    if mesh_prim.IsValid():
        _set_or_create_attr(mesh_prim, TRAMPOLINE_THICKNESS_ATTR, Sdf.ValueTypeNames.Float, float(cfg.height))
        if cfg.deformable_props is not None:
            _set_or_create_attr(
                mesh_prim,
                TRAMPOLINE_SIM_RESOLUTION_ATTR,
                Sdf.ValueTypeNames.Int,
                int(cfg.deformable_props.simulation_hexahedral_resolution),
            )
    return prim


def make_trampoline_cfg(
    prim_path: str,
    *,
    center_z: float | None = None,
    thickness: float = TRAMPOLINE_THICKNESS,
    mass: float = TRAMPOLINE_MASS,
    youngs_modulus: float = TRAMPOLINE_YOUNGS_MODULUS,
    sim_resolution: int = TRAMPOLINE_SIM_RESOLUTION,
    contact_offset: float = TRAMPOLINE_CONTACT_OFFSET,
    debug_vis: bool = False,
) -> DeformableObjectCfg:
    """Create the shared deformable trampoline configuration."""
    if center_z is None:
        center_z = TRAMPOLINE_TOP_Z - 0.5 * float(thickness)
    return DeformableObjectCfg(
        prim_path=prim_path,
        spawn=_make_trampoline_mesh_cfg(
            thickness=thickness,
            mass=mass,
            youngs_modulus=youngs_modulus,
            sim_resolution=sim_resolution,
            contact_offset=contact_offset,
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, center_z)),
        debug_vis=debug_vis,
    )


def make_trampoline_bucket_cfg(
    prim_path: str,
    *,
    geometry_buckets: Sequence[tuple[float, int]],
    top_z: float = TRAMPOLINE_TOP_Z,
    mass: float = TRAMPOLINE_MASS,
    youngs_modulus: float = TRAMPOLINE_YOUNGS_MODULUS,
    contact_offset: float = TRAMPOLINE_CONTACT_OFFSET,
    random_choice: bool = False,
    debug_vis: bool = False,
) -> DeformableObjectCfg:
    """Create a trampoline config that assigns pre-cooked thickness/resolution buckets per env."""
    if len(geometry_buckets) == 0:
        raise ValueError("At least one trampoline geometry bucket is required.")
    assets_cfg = []
    for thickness, sim_resolution in geometry_buckets:
        cfg = _make_trampoline_mesh_cfg(
            thickness=float(thickness),
            mass=mass,
            youngs_modulus=youngs_modulus,
            sim_resolution=int(sim_resolution),
            contact_offset=contact_offset,
        )
        cfg.func = spawn_top_aligned_trampoline_cylinder
        assets_cfg.append(cfg)

    return DeformableObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=assets_cfg, random_choice=random_choice),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, top_z)),
        debug_vis=debug_vis,
    )


def build_trampoline_sim_node_mask(trampoline: DeformableObject) -> torch.Tensor:
    """Return a valid-node mask for padded heterogeneous soft-body views."""
    element_indices = trampoline.root_physx_view.get_sim_element_indices()
    element_indices = torch.as_tensor(element_indices, device=trampoline.device, dtype=torch.long)
    if element_indices.ndim != 3 or element_indices.shape[-1] != 4:
        return torch.ones(
            trampoline.data.default_nodal_state_w.shape[:2],
            device=trampoline.device,
            dtype=torch.bool,
        )

    valid_elements = element_indices.amax(dim=-1) != element_indices.amin(dim=-1)
    scatter_indices = torch.where(valid_elements.unsqueeze(-1), element_indices + 1, 0).reshape(
        element_indices.shape[0], -1
    )
    valid_node_mask = torch.zeros(
        (element_indices.shape[0], trampoline.max_sim_vertices_per_body + 1),
        device=trampoline.device,
        dtype=torch.bool,
    )
    valid_node_mask.scatter_(1, scatter_indices.clamp(min=0, max=trampoline.max_sim_vertices_per_body), True)
    return valid_node_mask[:, 1:]


def resolve_trampoline_center_node_ids(
    default_nodal_state_w: torch.Tensor,
    valid_node_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Resolve the center simulation-node id independently for each environment."""
    nodal_pos = default_nodal_state_w[..., :3]
    if valid_node_mask is None:
        valid_node_mask = torch.ones(nodal_pos.shape[:2], device=nodal_pos.device, dtype=torch.bool)
    valid_count = valid_node_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=nodal_pos.dtype)
    center_xy = (nodal_pos[..., :2] * valid_node_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_count.unsqueeze(
        -1
    )
    radial_distance = torch.linalg.vector_norm(nodal_pos[..., :2] - center_xy, dim=-1)
    radial_distance = torch.where(valid_node_mask, radial_distance, torch.full_like(radial_distance, torch.inf))
    return radial_distance.argmin(dim=1)


def build_trampoline_kinematic_targets(
    default_nodal_state_w: torch.Tensor,
    nodal_kinematic_target: torch.Tensor,
    pin_width: float | torch.Tensor = TRAMPOLINE_PIN_WIDTH,
    valid_node_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create kinematic targets that pin the outer rim of a deformable trampoline."""
    targets = nodal_kinematic_target.clone()
    targets[..., :3] = default_nodal_state_w[..., :3]
    targets[..., 3] = 1.0

    nodal_pos = default_nodal_state_w[..., :3]
    if valid_node_mask is None:
        valid_node_mask = torch.ones(nodal_pos.shape[:2], device=nodal_pos.device, dtype=torch.bool)
    valid_count = valid_node_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=nodal_pos.dtype)
    center_xy = (nodal_pos[..., :2] * valid_node_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_count.unsqueeze(
        -1
    )
    radial_distance = torch.linalg.vector_norm(nodal_pos[..., :2] - center_xy, dim=-1)

    masked_radial_distance = torch.where(
        valid_node_mask,
        radial_distance,
        torch.full_like(radial_distance, -torch.inf),
    )
    rim_radius = masked_radial_distance.max(dim=1, keepdim=True).values.clamp_min(0.0)
    pin_width_tensor = torch.as_tensor(pin_width, device=nodal_pos.device, dtype=nodal_pos.dtype)
    if pin_width_tensor.ndim == 0:
        pin_width_tensor = pin_width_tensor.reshape(1, 1)
    elif pin_width_tensor.ndim == 1:
        pin_width_tensor = pin_width_tensor.reshape(-1, 1)
    if pin_width_tensor.shape[0] == 1 and nodal_pos.shape[0] != 1:
        pin_width_tensor = pin_width_tensor.expand(nodal_pos.shape[0], 1)
    if pin_width_tensor.shape != rim_radius.shape:
        raise ValueError(f"Expected pin_width shape {tuple(rim_radius.shape)}, got {tuple(pin_width_tensor.shape)}.")
    pin_threshold = torch.clamp(rim_radius - pin_width_tensor, min=0.0)
    pinned_mask = valid_node_mask & (radial_distance >= pin_threshold)
    center_node_ids = resolve_trampoline_center_node_ids(default_nodal_state_w, valid_node_mask)

    targets[..., 3] = torch.where(
        pinned_mask,
        torch.zeros_like(targets[..., 3]),
        torch.ones_like(targets[..., 3]),
    )
    return targets, pinned_mask, center_node_ids


def make_trampoline_node_marker_cfg(
    prim_path: str,
    color: tuple[float, float, float],
) -> VisualizationMarkersCfg:
    """Create a marker config for visualizing trampoline nodes."""
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "node": sim_utils.SphereCfg(
                radius=0.012,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        },
    )


def build_trampoline_node_visualizers() -> tuple[VisualizationMarkers, VisualizationMarkers]:
    """Create pinned (red) and free (green) node visualizers for the deformable trampoline."""
    pinned_visualizer = VisualizationMarkers(
        make_trampoline_node_marker_cfg("/Visuals/TrampolinePinnedNodes", color=(1.0, 0.2, 0.2))
    )
    free_visualizer = VisualizationMarkers(
        make_trampoline_node_marker_cfg("/Visuals/TrampolineFreeNodes", color=(0.2, 1.0, 0.2))
    )
    return pinned_visualizer, free_visualizer


def update_trampoline_node_visualizers(
    trampoline: DeformableObject,
    pinned_mask: torch.Tensor,
    pinned_visualizer: VisualizationMarkers | None,
    free_visualizer: VisualizationMarkers | None,
    valid_node_mask: torch.Tensor | None = None,
) -> None:
    """Update the pinned/free node marker positions from current nodal state."""
    if pinned_visualizer is None or free_visualizer is None:
        return

    env0_mask = pinned_mask[0]
    if valid_node_mask is not None:
        env0_valid = valid_node_mask[0]
    else:
        env0_valid = torch.ones_like(env0_mask)
    env0_nodal_pos_w = trampoline.data.nodal_pos_w[0]
    pinned_visualizer.visualize(translations=env0_nodal_pos_w[env0_mask])
    free_visualizer.visualize(translations=env0_nodal_pos_w[env0_valid & ~env0_mask])


def trampoline_mesh_prim_path(root_prim_path: str) -> str:
    """Return the mesh prim path inside a spawned mesh-cylinder trampoline."""
    return f"{root_prim_path}/geometry/mesh"


def _get_trampoline_material_property(material_view, plural_getter_name: str, singular_getter_name: str) -> torch.Tensor:
    """Read a deformable material property from whichever material-view API is available."""
    getter = getattr(material_view, plural_getter_name, None)
    if getter is not None:
        values = getter()
    else:
        values = getattr(material_view, singular_getter_name)()
    return torch.as_tensor(values, device="cpu", dtype=torch.float32).clone()


def get_trampoline_youngs_moduli(material_view) -> torch.Tensor:
    """Read Young's modulus values from the available material view API."""
    return _get_trampoline_material_property(material_view, "get_youngs_moduli", "get_youngs_modulus")


def get_trampoline_dynamic_frictions(material_view) -> torch.Tensor:
    """Read deformable material dynamic friction values."""
    return _get_trampoline_material_property(material_view, "get_dynamic_frictions", "get_dynamic_friction")


def get_trampoline_elasticity_dampings(material_view) -> torch.Tensor:
    """Read deformable material elasticity damping values."""
    return _get_trampoline_material_property(material_view, "get_elasticity_dampings", "get_damping")


def get_trampoline_damping_scales(material_view) -> torch.Tensor:
    """Read deformable material damping scale values."""
    return _get_trampoline_material_property(material_view, "get_damping_scales", "get_damping_scale")


def get_trampoline_poissons_ratios(material_view) -> torch.Tensor:
    """Read deformable material Poisson's ratio values."""
    return _get_trampoline_material_property(material_view, "get_poissons_ratios", "get_poissons_ratio")


def _as_column_tensor(values: torch.Tensor, *, device: str | torch.device | None = None) -> torch.Tensor:
    """Convert material properties to the column layout expected by the PhysX tensor API."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device=device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    elif tensor.ndim != 2 or tensor.shape[1] != 1:
        raise ValueError(f"Expected a scalar, vector, or column tensor, got shape {tuple(tensor.shape)}.")
    return tensor.contiguous()


def _set_trampoline_material_property(
    material_view,
    values: torch.Tensor,
    env_ids: torch.Tensor,
    *,
    plural_setter_name: str,
    singular_setter_name: str,
    singular_getter_name: str,
    property_name: str,
) -> None:
    """Write a deformable material property using whichever material-view API is available."""
    env_ids = torch.as_tensor(env_ids, dtype=torch.long).reshape(-1).contiguous()

    setter = getattr(material_view, plural_setter_name, None)
    if setter is not None:
        values = _as_column_tensor(values)
        if values.shape[0] == 1 and env_ids.numel() > 1:
            values = values.expand(env_ids.numel(), 1).clone()
        if values.shape[0] != env_ids.numel():
            raise ValueError(f"Expected {env_ids.numel()} {property_name} values, got {values.shape[0]}.")
        setter(values, indices=env_ids)
    else:
        # The low-level PhysX tensor view expects a full `(count, 1)` material buffer
        # even when `indices` selects only a subset of environments.
        current_values = _as_column_tensor(getattr(material_view, singular_getter_name)()).clone()
        env_ids = env_ids.to(device=current_values.device)
        values = _as_column_tensor(values, device=current_values.device)
        if values.shape[0] == 1 and env_ids.numel() > 1:
            values = values.expand(env_ids.numel(), 1).clone()
        if values.shape[0] != env_ids.numel():
            raise ValueError(f"Expected {env_ids.numel()} {property_name} values, got {values.shape[0]}.")
        current_values[env_ids] = values
        getattr(material_view, singular_setter_name)(current_values, indices=env_ids)


def set_trampoline_youngs_moduli(material_view, values: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write Young's modulus values using whichever material-view API is available."""
    _set_trampoline_material_property(
        material_view,
        values,
        env_ids,
        plural_setter_name="set_youngs_moduli",
        singular_setter_name="set_youngs_modulus",
        singular_getter_name="get_youngs_modulus",
        property_name="Young's modulus",
    )


def set_trampoline_dynamic_frictions(material_view, values: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write deformable material dynamic friction values."""
    _set_trampoline_material_property(
        material_view,
        values,
        env_ids,
        plural_setter_name="set_dynamic_frictions",
        singular_setter_name="set_dynamic_friction",
        singular_getter_name="get_dynamic_friction",
        property_name="dynamic friction",
    )


def set_trampoline_elasticity_dampings(material_view, values: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write deformable material elasticity damping values."""
    _set_trampoline_material_property(
        material_view,
        values,
        env_ids,
        plural_setter_name="set_elasticity_dampings",
        singular_setter_name="set_damping",
        singular_getter_name="get_damping",
        property_name="elasticity damping",
    )


def set_trampoline_damping_scales(material_view, values: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write deformable material damping scale values."""
    _set_trampoline_material_property(
        material_view,
        values,
        env_ids,
        plural_setter_name="set_damping_scales",
        singular_setter_name="set_damping_scale",
        singular_getter_name="get_damping_scale",
        property_name="damping scale",
    )


def set_trampoline_poissons_ratios(material_view, values: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write deformable material Poisson's ratio values."""
    _set_trampoline_material_property(
        material_view,
        values,
        env_ids,
        plural_setter_name="set_poissons_ratios",
        singular_setter_name="set_poissons_ratio",
        singular_getter_name="get_poissons_ratio",
        property_name="Poisson's ratio",
    )
