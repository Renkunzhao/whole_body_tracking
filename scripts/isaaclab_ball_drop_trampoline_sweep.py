from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


DEFAULT_OUTPUT = Path("logs/isaaclab_ball_drop_trampoline_sweep.csv")
BALL_RADIUS = 0.022
BALL_MASS = 4.02
DEFAULT_BALL_HEIGHT = 1.0
DEFAULT_SIM_DT = 0.002
DEFAULT_SIM_TIME = 4.0
CONTACT_START_BOTTOM_Z = 0.015
CONTACT_END_BOTTOM_Z = 0.035
DEFAULT_ELASTICITY_DAMPING = 0.02
DEFAULT_DAMPING_SCALE = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IsaacLab ball-drop sweeps on the deformable trampoline.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV output path.")
    parser.add_argument("--sim_time", type=float, default=DEFAULT_SIM_TIME, help="Simulation duration in seconds.")
    parser.add_argument("--sim_dt", type=float, default=DEFAULT_SIM_DT, help="Simulation timestep in seconds.")
    parser.add_argument("--ball_height", type=float, default=DEFAULT_BALL_HEIGHT, help="Initial ball center height.")
    parser.add_argument("--youngs_modulus", type=float, nargs="*", default=[4.0e4, 1.6e5], help="Extra Young's modulus values to sweep.")
    parser.add_argument("--elasticity_damping", type=float, nargs="*", default=[0.005, 0.05], help="Extra elasticity damping values to sweep.")
    parser.add_argument("--damping_scale", type=float, nargs="*", default=[0.2], help="Extra damping scale values to sweep.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, DeformableObject, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from whole_body_tracking.utils.trampoline_deformable import (  # noqa: E402
    TRAMPOLINE_MASS,
    TRAMPOLINE_SIM_RESOLUTION,
    TRAMPOLINE_YOUNGS_MODULUS,
    build_trampoline_kinematic_targets,
    make_trampoline_cfg,
    set_trampoline_damping_scales,
    set_trampoline_elasticity_dampings,
    set_trampoline_youngs_moduli,
)


def format_float_label(value: float) -> str:
    return f"{value:g}"


def finite_or_nan(value: float) -> float:
    return value if math.isfinite(value) else float("nan")


def build_conditions(args: argparse.Namespace) -> list[dict[str, Any]]:
    nominal = {
        "condition": "nominal",
        "youngs_modulus": TRAMPOLINE_YOUNGS_MODULUS,
        "elasticity_damping": DEFAULT_ELASTICITY_DAMPING,
        "damping_scale": DEFAULT_DAMPING_SCALE,
    }
    single_factor_sweeps = (
        ("youngs", "youngs_modulus", args.youngs_modulus),
        ("elasticity_damping", "elasticity_damping", args.elasticity_damping),
        ("damping_scale", "damping_scale", args.damping_scale),
    )

    conditions = [nominal]
    for label, field_name, values in single_factor_sweeps:
        conditions.extend(
            {**nominal, "condition": f"{label}_{format_float_label(value)}", field_name: value}
            for value in values
        )
    return conditions


def make_ball_cfg(prim_path: str, ball_height: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL_MASS),
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


def main() -> None:
    conditions = build_conditions(args_cli)

    sim = SimulationContext(sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.device))

    @configclass
    class BallDropSceneCfg(InteractiveSceneCfg):
        ball: RigidObjectCfg = make_ball_cfg("{ENV_REGEX_NS}/Ball", args_cli.ball_height)
        trampoline = make_trampoline_cfg(
            "{ENV_REGEX_NS}/Trampoline",
            mass=TRAMPOLINE_MASS,
            youngs_modulus=TRAMPOLINE_YOUNGS_MODULUS,
            sim_resolution=TRAMPOLINE_SIM_RESOLUTION,
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
    set_trampoline_youngs_moduli(
        trampoline.material_physx_view,
        torch.tensor([condition["youngs_modulus"] for condition in conditions], dtype=torch.float32),
        env_ids.cpu(),
    )
    set_trampoline_elasticity_dampings(
        trampoline.material_physx_view,
        torch.tensor([condition["elasticity_damping"] for condition in conditions], dtype=torch.float32),
        env_ids.cpu(),
    )
    set_trampoline_damping_scales(
        trampoline.material_physx_view,
        torch.tensor([condition["damping_scale"] for condition in conditions], dtype=torch.float32),
        env_ids.cpu(),
    )

    targets, _, center_node_ids = build_trampoline_kinematic_targets(
        trampoline.data.default_nodal_state_w,
        trampoline.data.nodal_kinematic_target,
    )
    reset_ball(scene, ball)
    reset_trampoline(scene, trampoline, targets)

    center_z0 = trampoline.data.nodal_pos_w[env_ids, center_node_ids, 2].detach().clone()
    min_center_z = center_z0.clone()
    min_ball_z = ball.data.root_pos_w[:, 2].detach().clone()
    contact_started = torch.zeros(scene.num_envs, dtype=torch.bool, device=ball.device)
    released = torch.zeros(scene.num_envs, dtype=torch.bool, device=ball.device)
    contact_start_s = torch.full((scene.num_envs,), float("nan"), device=ball.device)
    contact_end_s = torch.full((scene.num_envs,), float("nan"), device=ball.device)
    impact_vz = torch.full((scene.num_envs,), float("nan"), device=ball.device)
    release_vz = torch.full((scene.num_envs,), float("nan"), device=ball.device)
    max_rebound = torch.full((scene.num_envs,), -float("inf"), device=ball.device)

    for step in range(int(args_cli.sim_time / args_cli.sim_dt)):
        trampoline.write_nodal_kinematic_target_to_sim(targets)
        ball.write_data_to_sim()
        sim.step()
        scene.update(args_cli.sim_dt)

        t = (step + 1) * args_cli.sim_dt
        ball_z = ball.data.root_pos_w[:, 2]
        ball_vz = ball.data.root_lin_vel_w[:, 2]
        bottom_z = ball_z - BALL_RADIUS
        center_z = trampoline.data.nodal_pos_w[env_ids, center_node_ids, 2]
        min_center_z = torch.minimum(min_center_z, center_z)
        min_ball_z = torch.minimum(min_ball_z, ball_z)

        new_contact = (~contact_started) & (bottom_z <= CONTACT_START_BOTTOM_Z)
        contact_start_s[new_contact] = t
        impact_vz[new_contact] = ball_vz[new_contact]
        contact_started |= new_contact

        new_release = contact_started & (~released) & (bottom_z >= CONTACT_END_BOTTOM_Z) & (ball_vz > 0.0)
        contact_end_s[new_release] = t
        release_vz[new_release] = ball_vz[new_release]
        released |= new_release
        max_rebound = torch.where(released, torch.maximum(max_rebound, ball_z), max_rebound)

    rows = []
    for env_id, condition in enumerate(conditions):
        start = float(contact_start_s[env_id].cpu())
        end = float(contact_end_s[env_id].cpu())
        rebound_height = float(max_rebound[env_id].cpu()) if bool(released[env_id].cpu()) else float("nan")
        rows.append(
            {
                "condition": condition["condition"],
                "youngs_modulus": condition["youngs_modulus"],
                "elasticity_damping": condition["elasticity_damping"],
                "damping_scale": condition["damping_scale"],
                "contact_started": int(bool(contact_started[env_id].cpu())),
                "released": int(bool(released[env_id].cpu())),
                "contact_start_s": finite_or_nan(start),
                "contact_duration_s": finite_or_nan(end - start if math.isfinite(start) and math.isfinite(end) else float("nan")),
                "impact_vz_mps": finite_or_nan(float(impact_vz[env_id].cpu())),
                "release_vz_mps": finite_or_nan(float(release_vz[env_id].cpu())),
                "max_compression_m": float((center_z0[env_id] - min_center_z[env_id]).cpu()),
                "min_ball_z_m": float(min_ball_z[env_id].cpu()),
                "rebound_height_m": finite_or_nan(rebound_height),
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
