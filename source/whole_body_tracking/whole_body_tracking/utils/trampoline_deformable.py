from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObjectCfg

TRAMPOLINE_RADIUS = 1.5
TRAMPOLINE_THICKNESS = 0.1
TRAMPOLINE_TOP_Z = 0.0
TRAMPOLINE_CENTER_Z = TRAMPOLINE_TOP_Z - 0.5 * TRAMPOLINE_THICKNESS
TRAMPOLINE_PIN_WIDTH = 0.7
TRAMPOLINE_MASS = 10.0
TRAMPOLINE_YOUNGS_MODULUS = 8.0e3
TRAMPOLINE_SIM_RESOLUTION = 10
TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE = (8.0e3, 8.0e4)
TRAMPOLINE_DR_MASS_RANGE = (10.0, 10.0)


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
                contact_offset=0.01,
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
    pin_width: float = TRAMPOLINE_PIN_WIDTH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create kinematic targets that pin the outer rim of a deformable trampoline."""
    targets = nodal_kinematic_target.clone()
    targets[..., :3] = default_nodal_state_w[..., :3]
    targets[..., 3] = 1.0

    nodal_pos = default_nodal_state_w[..., :3]
    center_xy = nodal_pos[..., :2].mean(dim=1, keepdim=True)
    radial_distance = torch.linalg.vector_norm(nodal_pos[..., :2] - center_xy, dim=-1)

    rim_radius = radial_distance.max(dim=1, keepdim=True).values
    pin_threshold = torch.clamp(rim_radius - pin_width, min=0.0)
    pinned_mask = radial_distance >= pin_threshold
    center_node_ids = radial_distance.argmin(dim=1)

    targets[..., 3] = torch.where(
        pinned_mask,
        torch.zeros_like(targets[..., 3]),
        torch.ones_like(targets[..., 3]),
    )
    return targets, pinned_mask, center_node_ids


def trampoline_mesh_prim_path(root_prim_path: str) -> str:
    """Return the mesh prim path inside a spawned mesh-cylinder trampoline."""
    return f"{root_prim_path}/geometry/mesh"


def get_trampoline_youngs_moduli(material_view) -> torch.Tensor:
    """Read Young's modulus values from the available material view API."""
    getter = getattr(material_view, "get_youngs_moduli", None)
    if getter is not None:
        values = getter()
    else:
        values = material_view.get_youngs_modulus()
    return torch.as_tensor(values, device="cpu", dtype=torch.float32).clone()


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


def set_trampoline_youngs_moduli(material_view, values: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write Young's modulus values using whichever material-view API is available."""
    env_ids = torch.as_tensor(env_ids, dtype=torch.long).reshape(-1).contiguous()

    setter = getattr(material_view, "set_youngs_moduli", None)
    if setter is not None:
        values = _as_column_tensor(values)
        if values.shape[0] == 1 and env_ids.numel() > 1:
            values = values.expand(env_ids.numel(), 1).clone()
        if values.shape[0] != env_ids.numel():
            raise ValueError(f"Expected {env_ids.numel()} Young's modulus values, got {values.shape[0]}.")
        setter(values, indices=env_ids)
    else:
        # The low-level PhysX tensor view expects a full `(count, 1)` material buffer
        # even when `indices` selects only a subset of environments.
        current_values = _as_column_tensor(material_view.get_youngs_modulus()).clone()
        env_ids = env_ids.to(device=current_values.device)
        values = _as_column_tensor(values, device=current_values.device)
        if values.shape[0] == 1 and env_ids.numel() > 1:
            values = values.expand(env_ids.numel(), 1).clone()
        if values.shape[0] != env_ids.numel():
            raise ValueError(f"Expected {env_ids.numel()} Young's modulus values, got {values.shape[0]}.")
        current_values[env_ids] = values
        material_view.set_youngs_modulus(current_values, indices=env_ids)
