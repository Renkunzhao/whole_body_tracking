from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


DEFAULT_OUTPUT = Path("logs/isaaclab_ball_drop_trampoline_sweep.csv")
BALL_RADIUS = 0.022
DEFAULT_BALL_MASS = 4.02
DEFAULT_BALL_HEIGHT = 1.0
DEFAULT_SIM_DT = 0.002
DEFAULT_SIM_TIME = 4.0
DEFAULT_STABLE_VZ_THRESHOLD = 0.05
DEFAULT_STABLE_WINDOW_S = 0.2
DEFAULT_APEX_VZ_HYSTERESIS = 0.05
FALLTHROUGH_BALL_Z = -2.0
DEFAULT_TRAMPOLINE_MASS = 10.0
DEFAULT_YOUNGS_MODULUS = 8.0e6
DEFAULT_DYNAMIC_FRICTION = 0.8
DEFAULT_ELASTICITY_DAMPING = 0.01
DEFAULT_DAMPING_SCALE = 1.0
DEFAULT_POISSONS_RATIO = 0.35
DEFAULT_YOUNGS_SWEEP = (8.0e5, 8.0e6)
DEFAULT_TRAMPOLINE_MASS_SWEEP = (5.0, 15.0)
DEFAULT_DYNAMIC_FRICTION_SWEEP = (0.4, 1.2)
DEFAULT_ELASTICITY_DAMPING_SWEEP = (0.01, 0.1)
DEFAULT_DAMPING_SCALE_SWEEP: tuple[float, ...] = ()
DEFAULT_POISSONS_RATIO_SWEEP = (0.25, 0.45)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IsaacLab ball-drop sweeps on the deformable trampoline.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV output path.")
    parser.add_argument("--sim_time", type=float, default=DEFAULT_SIM_TIME, help="Simulation duration in seconds.")
    parser.add_argument("--sim_dt", type=float, default=DEFAULT_SIM_DT, help="Simulation timestep in seconds.")
    parser.add_argument("--ball_mass", type=float, default=DEFAULT_BALL_MASS, help="Ball mass in kilograms.")
    parser.add_argument("--ball_height", type=float, default=DEFAULT_BALL_HEIGHT, help="Initial ball center height.")
    parser.add_argument("--sim_resolution", type=int, default=None, help="Hexahedral resolution used at spawn time.")
    parser.add_argument("--thickness", type=float, default=None, help="Trampoline thickness in meters.")
    parser.add_argument("--stable_vz_threshold", type=float, default=DEFAULT_STABLE_VZ_THRESHOLD, help="Vertical speed threshold for stable-time detection.")
    parser.add_argument("--stable_window_s", type=float, default=DEFAULT_STABLE_WINDOW_S, help="Required consecutive low-speed duration for stable-time detection.")
    parser.add_argument("--apex_vz_hysteresis", type=float, default=DEFAULT_APEX_VZ_HYSTERESIS, help="Velocity hysteresis used to arm apex detection and suppress jitter.")
    parser.add_argument("--youngs_modulus", type=float, nargs="*", default=list(DEFAULT_YOUNGS_SWEEP), help="Extra Young's modulus values to sweep.")
    parser.add_argument("--trampoline_mass", type=float, nargs="*", default=list(DEFAULT_TRAMPOLINE_MASS_SWEEP), help="Extra trampoline mass values to sweep.")
    parser.add_argument("--dynamic_friction", type=float, nargs="*", default=list(DEFAULT_DYNAMIC_FRICTION_SWEEP), help="Extra deformable material dynamic friction values to sweep.")
    parser.add_argument("--elasticity_damping", type=float, nargs="*", default=list(DEFAULT_ELASTICITY_DAMPING_SWEEP), help="Extra elasticity damping values to sweep.")
    parser.add_argument("--damping_scale", type=float, nargs="*", default=list(DEFAULT_DAMPING_SCALE_SWEEP), help="Extra damping scale values to sweep.")
    parser.add_argument("--poissons_ratio", type=float, nargs="*", default=list(DEFAULT_POISSONS_RATIO_SWEEP), help="Extra Poisson's ratio values to sweep.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402
from pxr import Sdf  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, DeformableObject, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from whole_body_tracking.utils.trampoline_deformable import (  # noqa: E402
    TRAMPOLINE_RADIUS,
    TRAMPOLINE_SIM_RESOLUTION,
    TRAMPOLINE_THICKNESS,
    build_trampoline_kinematic_targets,
    make_trampoline_cfg,
    set_trampoline_damping_scales,
    set_trampoline_dynamic_frictions,
    set_trampoline_elasticity_dampings,
    set_trampoline_poissons_ratios,
    set_trampoline_youngs_moduli,
    trampoline_mesh_prim_path,
)


if args_cli.sim_resolution is None:
    args_cli.sim_resolution = TRAMPOLINE_SIM_RESOLUTION
if args_cli.thickness is None:
    args_cli.thickness = TRAMPOLINE_THICKNESS


def format_float_label(value: float) -> str:
    return f"{value:g}"


def finite_or_nan(value: float) -> float:
    return value if math.isfinite(value) else float("nan")


def add_single_factor_sweep(
    conditions: list[dict[str, Any]],
    nominal: dict[str, Any],
    label: str,
    field_name: str,
    values: list[float],
) -> None:
    for value in values:
        if value == nominal[field_name]:
            continue
        conditions.append({**nominal, "condition": f"{label}_{format_float_label(value)}", field_name: value})


def build_conditions(args: argparse.Namespace) -> list[dict[str, Any]]:
    nominal = {
        "condition": "nominal",
        "trampoline_mass": DEFAULT_TRAMPOLINE_MASS,
        "youngs_modulus": DEFAULT_YOUNGS_MODULUS,
        "dynamic_friction": DEFAULT_DYNAMIC_FRICTION,
        "elasticity_damping": DEFAULT_ELASTICITY_DAMPING,
        "damping_scale": DEFAULT_DAMPING_SCALE,
        "poissons_ratio": DEFAULT_POISSONS_RATIO,
    }

    conditions = [nominal]
    add_single_factor_sweep(conditions, nominal, "mass", "trampoline_mass", args.trampoline_mass)
    add_single_factor_sweep(conditions, nominal, "youngs", "youngs_modulus", args.youngs_modulus)
    add_single_factor_sweep(conditions, nominal, "dynamic_friction", "dynamic_friction", args.dynamic_friction)
    add_single_factor_sweep(conditions, nominal, "elasticity_damping", "elasticity_damping", args.elasticity_damping)
    add_single_factor_sweep(conditions, nominal, "damping_scale", "damping_scale", args.damping_scale)
    add_single_factor_sweep(conditions, nominal, "poissons_ratio", "poissons_ratio", args.poissons_ratio)
    return conditions


def make_ball_cfg(prim_path: str, ball_height: float, ball_mass: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=ball_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7, dynamic_friction=0.6, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.35, 0.15), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, ball_height)),
    )


def reset_ball(scene: InteractiveScene, ball: RigidObject) -> None:
    env_ids = torch.arange(scene.num_envs, device=ball.device, dtype=torch.long)
    root_state = ball.data.default_root_state.clone()
    root_state[:, 0:3] += scene.env_origins
    root_state[:, 7:] = 0.0
    ball.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    ball.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
    ball.reset()


def reset_trampoline(scene: InteractiveScene, trampoline: DeformableObject, targets: torch.Tensor) -> None:
    env_ids = torch.arange(scene.num_envs, device=trampoline.device, dtype=torch.long)
    trampoline.write_nodal_state_to_sim(trampoline.data.default_nodal_state_w, env_ids=env_ids)
    trampoline.write_nodal_kinematic_target_to_sim(targets, env_ids=env_ids)
    trampoline.reset(env_ids=env_ids)


def set_trampoline_masses(trampoline: DeformableObject, masses: torch.Tensor) -> None:
    mesh_prim_paths = sim_utils.find_matching_prim_paths(trampoline_mesh_prim_path(trampoline.cfg.prim_path))
    if len(mesh_prim_paths) != int(masses.numel()):
        raise RuntimeError(
            f"Expected {masses.numel()} trampoline mesh prims, found {len(mesh_prim_paths)} for "
            f"pattern '{trampoline_mesh_prim_path(trampoline.cfg.prim_path)}'."
        )

    stage = get_current_stage()
    with Sdf.ChangeBlock():
        for prim_path, mass in zip(mesh_prim_paths, masses.cpu().tolist(), strict=True):
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(f"Invalid trampoline mesh prim: '{prim_path}'.")
            attr = prim.GetAttribute("physics:mass")
            if not attr.IsValid():
                attr = prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float)
            attr.Set(float(mass))


def classify_final_state(
    *,
    fallthrough: bool,
    fell_off_edge: bool,
    stable: bool,
    second_apex_found: bool,
    first_apex_found: bool,
    first_min_found: bool,
) -> str:
    if fallthrough:
        return "fallthrough"
    if fell_off_edge:
        return "fell_off_edge"
    if stable:
        return "stable"
    if second_apex_found:
        return "second_apex"
    if first_apex_found:
        return "first_apex"
    if first_min_found:
        return "first_min"
    return "not_stable"


def main() -> None:
    conditions = build_conditions(args_cli)

    sim = SimulationContext(sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.device))

    @configclass
    class BallDropSceneCfg(InteractiveSceneCfg):
        ball: RigidObjectCfg = make_ball_cfg("{ENV_REGEX_NS}/Ball", args_cli.ball_height, args_cli.ball_mass)
        trampoline = make_trampoline_cfg(
            "{ENV_REGEX_NS}/Trampoline",
            thickness=args_cli.thickness,
            mass=DEFAULT_TRAMPOLINE_MASS,
            youngs_modulus=DEFAULT_YOUNGS_MODULUS,
            sim_resolution=args_cli.sim_resolution,
        )
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )

    scene_cfg = BallDropSceneCfg(num_envs=len(conditions), env_spacing=4.0, replicate_physics=False)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.update(args_cli.sim_dt)

    ball: RigidObject = scene["ball"]
    trampoline: DeformableObject = scene["trampoline"]
    if trampoline.material_physx_view is None:
        raise RuntimeError("Failed to create deformable trampoline material view.")

    env_ids = torch.arange(scene.num_envs, device=trampoline.device, dtype=torch.long)
    env_ids_cpu = env_ids.cpu()
    set_trampoline_masses(
        trampoline,
        torch.tensor([condition["trampoline_mass"] for condition in conditions], dtype=torch.float32),
    )
    set_trampoline_youngs_moduli(
        trampoline.material_physx_view,
        torch.tensor([condition["youngs_modulus"] for condition in conditions], dtype=torch.float32),
        env_ids_cpu,
    )
    set_trampoline_dynamic_frictions(
        trampoline.material_physx_view,
        torch.tensor([condition["dynamic_friction"] for condition in conditions], dtype=torch.float32),
        env_ids_cpu,
    )
    set_trampoline_elasticity_dampings(
        trampoline.material_physx_view,
        torch.tensor([condition["elasticity_damping"] for condition in conditions], dtype=torch.float32),
        env_ids_cpu,
    )
    set_trampoline_damping_scales(
        trampoline.material_physx_view,
        torch.tensor([condition["damping_scale"] for condition in conditions], dtype=torch.float32),
        env_ids_cpu,
    )
    set_trampoline_poissons_ratios(
        trampoline.material_physx_view,
        torch.tensor([condition["poissons_ratio"] for condition in conditions], dtype=torch.float32),
        env_ids_cpu,
    )

    targets, pinned_mask, center_node_ids = build_trampoline_kinematic_targets(
        trampoline.data.default_nodal_state_w,
        trampoline.data.nodal_kinematic_target,
    )
    reset_ball(scene, ball)
    reset_trampoline(scene, trampoline, targets)

    center_z0 = trampoline.data.nodal_pos_w[env_ids, center_node_ids, 2].detach().clone()
    static_sag_m = -center_z0
    min_center_z = center_z0.clone()
    min_ball_z = ball.data.root_pos_w[:, 2].detach().clone()
    min_ball_xy = ball.data.root_pos_w[:, :2].detach().clone()
    first_min_ball_z_m = torch.full((scene.num_envs,), float("nan"), device=ball.device)
    first_min_ball_z_time_s = torch.full((scene.num_envs,), float("nan"), device=ball.device)
    first_min_armed = torch.zeros(scene.num_envs, dtype=torch.bool, device=ball.device)
    first_apex_found = torch.zeros(scene.num_envs, dtype=torch.bool, device=ball.device)
    first_apex_armed = torch.zeros(scene.num_envs, dtype=torch.bool, device=ball.device)
    first_apex_time_s = torch.full((scene.num_envs,), float("nan"), device=ball.device)
    first_apex_height_m = torch.full((scene.num_envs,), float("nan"), device=ball.device)
    second_apex_found = torch.zeros(scene.num_envs, dtype=torch.bool, device=ball.device)
    second_apex_armed = torch.zeros(scene.num_envs, dtype=torch.bool, device=ball.device)
    second_apex_time_s = torch.full((scene.num_envs,), float("nan"), device=ball.device)
    second_apex_height_m = torch.full((scene.num_envs,), float("nan"), device=ball.device)
    previous_ball_vz = torch.full((scene.num_envs,), float("nan"), device=ball.device)
    stable_step_count = torch.zeros(scene.num_envs, dtype=torch.int32, device=ball.device)
    stable_window_steps = max(1, int(round(args_cli.stable_window_s / args_cli.sim_dt)))
    stable = torch.zeros(scene.num_envs, dtype=torch.bool, device=ball.device)
    stable_time_s = torch.full((scene.num_envs,), float("nan"), device=ball.device)
    stable_ball_z = torch.full((scene.num_envs,), float("nan"), device=ball.device)
    stable_compression = torch.full((scene.num_envs,), float("nan"), device=ball.device)

    for step in range(int(args_cli.sim_time / args_cli.sim_dt)):
        trampoline.write_nodal_kinematic_target_to_sim(targets)
        ball.write_data_to_sim()
        sim.step()
        scene.update(args_cli.sim_dt)

        t = (step + 1) * args_cli.sim_dt
        ball_pos = ball.data.root_pos_w
        ball_z = ball_pos[:, 2]
        ball_vz = ball.data.root_lin_vel_w[:, 2]
        center_z = trampoline.data.nodal_pos_w[env_ids, center_node_ids, 2]
        compression = center_z0 - center_z
        min_center_z = torch.minimum(min_center_z, center_z)
        new_min_ball = ball_z < min_ball_z
        min_ball_z = torch.where(new_min_ball, ball_z, min_ball_z)
        min_ball_xy[new_min_ball] = ball_pos[new_min_ball, :2]

        low_speed = torch.abs(ball_vz) <= args_cli.stable_vz_threshold
        active_stable_search = ~stable
        stable_step_count = torch.where(active_stable_search & low_speed, stable_step_count + 1, stable_step_count)
        stable_step_count = torch.where(active_stable_search & (~low_speed), torch.zeros_like(stable_step_count), stable_step_count)
        new_stable = active_stable_search & (stable_step_count >= stable_window_steps)
        stable_time_s[new_stable] = t
        stable_ball_z[new_stable] = ball_z[new_stable]
        stable_compression[new_stable] = compression[new_stable]
        stable |= new_stable

        finite_previous = torch.isfinite(previous_ball_vz)
        first_min_armed |= ball_vz < 0.0
        new_first_min = (
            finite_previous
            & first_min_armed
            & torch.isnan(first_min_ball_z_time_s)
            & (previous_ball_vz < 0.0)
            & (ball_vz >= 0.0)
        )
        first_min_ball_z_m[new_first_min] = ball_z[new_first_min]
        first_min_ball_z_time_s[new_first_min] = t

        first_apex_ready = torch.isfinite(first_min_ball_z_time_s) & (~first_apex_found)
        first_apex_armed |= first_apex_ready & (ball_vz > args_cli.apex_vz_hysteresis)
        new_first_apex = first_apex_ready & first_apex_armed & finite_previous & (previous_ball_vz > 0.0) & (ball_vz <= 0.0)
        first_apex_found |= new_first_apex
        first_apex_time_s[new_first_apex] = t
        first_apex_height_m[new_first_apex] = ball_z[new_first_apex]

        second_apex_ready = first_apex_found & (~second_apex_found)
        second_apex_armed |= (
            second_apex_ready
            & (~second_apex_armed)
            & finite_previous
            & (previous_ball_vz <= 0.0)
            & (ball_vz > 0.0)
            & (t > first_apex_time_s)
        )
        new_second_apex = second_apex_ready & second_apex_armed & finite_previous & (previous_ball_vz > 0.0) & (ball_vz <= 0.0)
        second_apex_found |= new_second_apex
        second_apex_time_s[new_second_apex] = t
        second_apex_height_m[new_second_apex] = ball_z[new_second_apex]

        previous_ball_vz = ball_vz.clone()

    rows = []
    dropped_below = min_ball_z < FALLTHROUGH_BALL_Z
    min_ball_radius = torch.linalg.vector_norm(min_ball_xy, dim=1)
    fell_off_edge = dropped_below & (min_ball_radius >= TRAMPOLINE_RADIUS)
    fallthrough = dropped_below & (~fell_off_edge)

    for env_id, condition in enumerate(conditions):
        is_fallthrough = bool(fallthrough[env_id].cpu())
        is_fell_off_edge = bool(fell_off_edge[env_id].cpu())
        is_stable = bool(stable[env_id].cpu()) and not is_fallthrough and not is_fell_off_edge
        first_apex_height = float(first_apex_height_m[env_id].cpu())
        second_apex_height = float(second_apex_height_m[env_id].cpu())
        damping_ratio = (
            second_apex_height / first_apex_height
            if bool(first_apex_found[env_id].cpu()) and bool(second_apex_found[env_id].cpu()) and first_apex_height > 0.0
            else float("nan")
        )
        rows.append(
            {
                "condition": condition["condition"],
                "ball_mass": args_cli.ball_mass,
                "ball_height": args_cli.ball_height,
                "thickness": args_cli.thickness,
                "trampoline_mass": condition["trampoline_mass"],
                "sim_resolution": args_cli.sim_resolution,
                "pinned_node_count": int(pinned_mask[env_id].sum().item()),
                "youngs_modulus": condition["youngs_modulus"],
                "dynamic_friction": condition["dynamic_friction"],
                "elasticity_damping": condition["elasticity_damping"],
                "damping_scale": condition["damping_scale"],
                "poissons_ratio": condition["poissons_ratio"],
                "sim_time": args_cli.sim_time,
                "sim_dt": args_cli.sim_dt,
                "static_sag_m": finite_or_nan(float(static_sag_m[env_id].cpu())),
                "fallthrough": int(is_fallthrough),
                "fell_off_edge": int(is_fell_off_edge),
                "final_state": classify_final_state(
                    fallthrough=is_fallthrough,
                    fell_off_edge=is_fell_off_edge,
                    stable=is_stable,
                    second_apex_found=bool(second_apex_found[env_id].cpu()),
                    first_apex_found=bool(first_apex_found[env_id].cpu()),
                    first_min_found=math.isfinite(float(first_min_ball_z_time_s[env_id].cpu())),
                ),
                "min_ball_z_m": float(min_ball_z[env_id].cpu()),
                "min_ball_z_x_m": float(min_ball_xy[env_id, 0].cpu()),
                "min_ball_z_y_m": float(min_ball_xy[env_id, 1].cpu()),
                "first_min_ball_z_m": finite_or_nan(float(first_min_ball_z_m[env_id].cpu())),
                "first_min_ball_z_time_s": finite_or_nan(float(first_min_ball_z_time_s[env_id].cpu())),
                "first_apex_height_m": finite_or_nan(first_apex_height),
                "first_apex_time_s": finite_or_nan(float(first_apex_time_s[env_id].cpu())),
                "rebound_height_m": finite_or_nan(first_apex_height),
                "second_apex_height_m": finite_or_nan(second_apex_height),
                "second_apex_time_s": finite_or_nan(float(second_apex_time_s[env_id].cpu())),
                "damping_ratio": finite_or_nan(damping_ratio),
                "max_compression_m": float((center_z0[env_id] - min_center_z[env_id]).cpu()),
                "stable": int(is_stable),
                "stable_time_s": finite_or_nan(float(stable_time_s[env_id].cpu()) if is_stable else float("nan")),
                "stable_ball_z_m": finite_or_nan(float(stable_ball_z[env_id].cpu()) if is_stable else float("nan")),
                "stable_compression_m": finite_or_nan(float(stable_compression[env_id].cpu()) if is_stable else float("nan")),
            }
        )

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    with args_cli.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row, flush=True)
    print(f"WROTE {args_cli.output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
