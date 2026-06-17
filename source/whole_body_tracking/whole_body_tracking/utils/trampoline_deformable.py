from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

TRAMPOLINE_RADIUS = 1.5
TRAMPOLINE_THICKNESS = 0.1
TRAMPOLINE_TOP_Z = 0.0
TRAMPOLINE_CENTER_Z = TRAMPOLINE_TOP_Z - 0.5 * TRAMPOLINE_THICKNESS
TRAMPOLINE_PIN_RADIUS = TRAMPOLINE_RADIUS
TRAMPOLINE_MASS = 10.0
TRAMPOLINE_YOUNGS_MODULUS = 8.0e4
TRAMPOLINE_SIM_RESOLUTION = 15


def make_trampoline_cfg(
    prim_path: str,
    *,
    center_z: float = TRAMPOLINE_CENTER_Z,
    mass: float = TRAMPOLINE_MASS,
    youngs_modulus: float = TRAMPOLINE_YOUNGS_MODULUS,
    sim_resolution: int = TRAMPOLINE_SIM_RESOLUTION,
    debug_vis: bool = False,
) -> DeformableObjectCfg:
    """Create the shared deformable trampoline configuration."""
    return DeformableObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.MeshCylinderCfg(
            radius=TRAMPOLINE_RADIUS,
            height=TRAMPOLINE_THICKNESS,
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
                # contact_offset=0.01,
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
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, center_z)),
        debug_vis=debug_vis,
    )


def build_trampoline_kinematic_targets(
    default_nodal_state_w: torch.Tensor,
    nodal_kinematic_target: torch.Tensor,
    pin_radius: float | torch.Tensor = TRAMPOLINE_PIN_RADIUS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create kinematic targets that pin nodes outside the usable trampoline radius."""
    targets = nodal_kinematic_target.clone()
    targets[..., :3] = default_nodal_state_w[..., :3]
    targets[..., 3] = 1.0

    nodal_pos = default_nodal_state_w[..., :3]
    xy_min = nodal_pos[..., :2].amin(dim=1, keepdim=True)
    xy_max = nodal_pos[..., :2].amax(dim=1, keepdim=True)
    center_xy = 0.5 * (xy_min + xy_max)
    radial_distance = torch.linalg.vector_norm(nodal_pos[..., :2] - center_xy, dim=-1)

    pin_radius_column = _as_column_tensor(pin_radius, device=radial_distance.device).to(dtype=radial_distance.dtype)
    edge_radius = radial_distance.max(dim=1, keepdim=True).values
    pin_threshold = torch.minimum(torch.clamp(pin_radius_column, min=0.0), edge_radius)
    pinned_mask = radial_distance >= pin_threshold
    center_node_ids = radial_distance.argmin(dim=1)

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
) -> None:
    """Update the pinned/free node marker positions from current nodal state."""
    if pinned_visualizer is None or free_visualizer is None:
        return

    env0_mask = pinned_mask[0]
    env0_nodal_pos_w = trampoline.data.nodal_pos_w[0]
    pinned_visualizer.visualize(translations=env0_nodal_pos_w[env0_mask])
    free_visualizer.visualize(translations=env0_nodal_pos_w[~env0_mask])


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


def _as_column_tensor(values: float | torch.Tensor, *, device: str | torch.device | None = None) -> torch.Tensor:
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
