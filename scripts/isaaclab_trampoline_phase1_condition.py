from __future__ import annotations

import argparse
import csv
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


DEFAULT_ARTIFACT_ROOT = Path("logs/isaaclab_trampoline_phase1_runs")
BALL_RADIUS = 0.022
BALL_MASS = 4.02
DEFAULT_BALL_HEIGHT = 1.0
DEFAULT_SIM_DT = 0.002
DEFAULT_SIM_TIME = 10.0
CONTACT_START_BOTTOM_Z = 0.015
CONTACT_END_BOTTOM_Z = 0.035
DEFAULT_STABLE_VZ_THRESHOLD = 0.05
DEFAULT_STABLE_WINDOW_S = 0.2
DEFAULT_APEX_VZ_HYSTERESIS = 0.05
FALLTHROUGH_BALL_Z = -2.0
DEFAULT_THICKNESS = 0.1
DEFAULT_TRAMPOLINE_MASS = 10.0
DEFAULT_SIM_RESOLUTION = 15
DEFAULT_YOUNGS = 8.0e4
DEFAULT_ELASTICITY_DAMPING = 0.02
DEFAULT_DAMPING_SCALE = 0.1
DEFAULT_VIDEO_WIDTH = 1280
DEFAULT_VIDEO_HEIGHT = 720
DEFAULT_VIDEO_FPS = 30
DEFAULT_CAMERA_EYE = (3.0, -3.0, 2.2)
DEFAULT_CAMERA_TARGET = (0.0, 0.0, -0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one IsaacLab trampoline Phase 1 condition.")
    parser.add_argument("--artifact_root", type=Path, default=DEFAULT_ARTIFACT_ROOT, help="Root directory for per-run artifacts.")
    parser.add_argument("--run_dir", type=Path, default=None, help="Explicit artifact directory for this run.")
    parser.add_argument("--label", type=str, required=True, help="Condition label written to CSV.")
    parser.add_argument("--sim_time", type=float, default=DEFAULT_SIM_TIME, help="Simulation duration in seconds.")
    parser.add_argument("--sim_dt", type=float, default=DEFAULT_SIM_DT, help="Simulation timestep in seconds.")
    parser.add_argument("--ball_height", type=float, default=DEFAULT_BALL_HEIGHT, help="Initial ball center height.")
    parser.add_argument("--ball_mass", type=float, default=BALL_MASS, help="Ball mass in kilograms.")
    parser.add_argument("--thickness", type=float, default=DEFAULT_THICKNESS, help="Trampoline thickness in meters.")
    parser.add_argument("--trampoline_mass", type=float, default=DEFAULT_TRAMPOLINE_MASS, help="Trampoline mass in kilograms.")
    parser.add_argument("--sim_resolution", type=int, default=DEFAULT_SIM_RESOLUTION, help="Hexahedral resolution used at spawn time.")
    parser.add_argument("--youngs_modulus", type=float, default=DEFAULT_YOUNGS, help="Young's modulus used for the trampoline material.")
    parser.add_argument("--elasticity_damping", type=float, default=DEFAULT_ELASTICITY_DAMPING, help="Elasticity damping used for the trampoline material.")
    parser.add_argument("--damping_scale", type=float, default=DEFAULT_DAMPING_SCALE, help="Damping scale used for the trampoline material.")
    parser.add_argument("--stable_vz_threshold", type=float, default=DEFAULT_STABLE_VZ_THRESHOLD, help="Vertical speed threshold for stable-time detection.")
    parser.add_argument("--stable_window_s", type=float, default=DEFAULT_STABLE_WINDOW_S, help="Required consecutive low-speed duration for stable-time detection.")
    parser.add_argument("--apex_vz_hysteresis", type=float, default=DEFAULT_APEX_VZ_HYSTERESIS, help="Velocity hysteresis used to arm apex detection and suppress jitter.")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True, help="Record an MP4 for this run.")
    parser.add_argument("--video_width", type=int, default=DEFAULT_VIDEO_WIDTH, help="Video width in pixels.")
    parser.add_argument("--video_height", type=int, default=DEFAULT_VIDEO_HEIGHT, help="Video height in pixels.")
    parser.add_argument("--video_fps", type=int, default=DEFAULT_VIDEO_FPS, help="Video frame rate.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, DeformableObject, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from whole_body_tracking.utils.trampoline_deformable import (  # noqa: E402
    TRAMPOLINE_RADIUS,
    build_trampoline_kinematic_targets,
    make_trampoline_cfg,
    set_trampoline_damping_scales,
    set_trampoline_elasticity_dampings,
    set_trampoline_youngs_moduli,
)


try:
    import imageio.v2 as imageio
except ImportError as exc:  # pragma: no cover - dependency should exist in the IsaacLab env
    imageio = None
    _IMAGEIO_IMPORT_ERROR = exc
else:
    _IMAGEIO_IMPORT_ERROR = None


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - dependency should exist in the IsaacLab env
    plt = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None


def finite_or_nan(value: float) -> float:
    return value if math.isfinite(value) else float("nan")


def sanitize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "run"


def format_float_label(value: float) -> str:
    return f"{value:g}"


def build_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return args.run_dir.expanduser().resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [
        timestamp,
        sanitize_token(args.label),
        f"t{format_float_label(args.sim_time)}",
        f"dt{format_float_label(args.sim_dt)}",
        f"h{format_float_label(args.ball_height)}",
        f"bm{format_float_label(args.ball_mass)}",
        f"sr{args.sim_resolution}",
        f"th{format_float_label(args.thickness)}",
        f"tm{format_float_label(args.trampoline_mass)}",
        f"E{format_float_label(args.youngs_modulus)}",
        f"ed{format_float_label(args.elasticity_damping)}",
        f"ds{format_float_label(args.damping_scale)}",
    ]
    return args.artifact_root.expanduser().resolve() / "__".join(parts)


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


@configclass
class Phase1SceneCfg(InteractiveSceneCfg):
    ball: RigidObjectCfg = make_ball_cfg("{ENV_REGEX_NS}/Ball", args_cli.ball_height, args_cli.ball_mass)
    trampoline = make_trampoline_cfg(
        "{ENV_REGEX_NS}/Trampoline",
        thickness=args_cli.thickness,
        mass=args_cli.trampoline_mass,
        youngs_modulus=args_cli.youngs_modulus,
        sim_resolution=args_cli.sim_resolution,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
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


def create_video_camera() -> Camera:
    camera_cfg = CameraCfg(
        prim_path="/World/Phase1Camera",
        update_period=0,
        height=args_cli.video_height,
        width=args_cli.video_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    return Camera(cfg=camera_cfg)


def set_video_camera_view(camera: Camera, sim: SimulationContext) -> None:
    camera.set_world_poses_from_view(
        torch.tensor([DEFAULT_CAMERA_EYE], device=sim.device),
        torch.tensor([DEFAULT_CAMERA_TARGET], device=sim.device),
    )


def _as_numpy_rgb(frame: Any) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        array = frame.detach().cpu().numpy()
    else:
        array = np.asarray(frame)
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array, 0.0, 255.0)
        array = array.astype(np.uint8)
    return np.ascontiguousarray(array)


def write_summary_row(output_path: Path, row: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_timeseries_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_timeseries(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    if plt is None:
        raise RuntimeError(f"matplotlib is required for plots: {_MATPLOTLIB_IMPORT_ERROR}")

    times = np.asarray([row["time_s"] for row in rows], dtype=float)
    ball_z = np.asarray([row["ball_z_m"] for row in rows], dtype=float)
    ball_vz = np.asarray([row["ball_vz_mps"] for row in rows], dtype=float)
    center_z = np.asarray([row["trampoline_center_z_m"] for row in rows], dtype=float)
    center_vz = np.asarray([row["trampoline_center_vz_mps"] for row in rows], dtype=float)
    bottom_z = np.asarray([row["ball_bottom_z_m"] for row in rows], dtype=float)
    compression = np.asarray([row["compression_m"] for row in rows], dtype=float)

    state_path = output_dir / "phase1_vertical_state.png"
    compression_path = output_dir / "phase1_compression.png"

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(times, ball_z, label="ball z")
    ax[0].plot(times, center_z, label="trampoline center z")
    ax[0].plot(times, bottom_z, label="ball bottom z", linestyle="--", alpha=0.7)
    ax[0].axhline(CONTACT_START_BOTTOM_Z, color="tab:red", linestyle=":", linewidth=1, label="contact start threshold")
    ax[0].axhline(CONTACT_END_BOTTOM_Z, color="tab:green", linestyle=":", linewidth=1, label="contact end threshold")
    ax[0].set_ylabel("position [m]")
    ax[0].legend(loc="best")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(times, ball_vz, label="ball vz")
    ax[1].plot(times, center_vz, label="trampoline center vz")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("velocity [m/s]")
    ax[1].legend(loc="best")
    ax[1].grid(True, alpha=0.3)
    fig.suptitle("Phase 1 vertical state")
    fig.tight_layout()
    fig.savefig(state_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, compression, label="center compression")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("compression [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.suptitle("Phase 1 trampoline compression")
    fig.tight_layout()
    fig.savefig(compression_path, dpi=160)
    plt.close(fig)

    return [state_path, compression_path]


def main() -> None:
    run_dir = build_run_dir(args_cli)
    run_dir.mkdir(parents=True, exist_ok=True)
    video_path = run_dir / "phase1_video.mp4"
    timeseries_path = run_dir / "phase1_trajectory.csv"
    summary_path = run_dir / "phase1_summary.csv"

    sim = SimulationContext(sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.device))
    scene_cfg = Phase1SceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)
    scene = InteractiveScene(scene_cfg)
    camera = create_video_camera() if args_cli.video else None
    sim.reset()
    scene.update(args_cli.sim_dt)

    ball: RigidObject = scene["ball"]
    trampoline: DeformableObject = scene["trampoline"]
    if trampoline.material_physx_view is None:
        raise RuntimeError("Failed to create deformable trampoline material view.")

    env_ids_cpu = torch.arange(scene.num_envs, device=trampoline.device, dtype=torch.long).cpu()
    material_view = trampoline.material_physx_view

    def write_material_property(setter, value: float) -> None:
        setter(material_view, torch.tensor([value], dtype=torch.float32), env_ids_cpu)

    write_material_property(set_trampoline_youngs_moduli, args_cli.youngs_modulus)
    write_material_property(set_trampoline_elasticity_dampings, args_cli.elasticity_damping)
    write_material_property(set_trampoline_damping_scales, args_cli.damping_scale)

    targets, pinned_mask, center_node_ids = build_trampoline_kinematic_targets(
        trampoline.data.default_nodal_state_w,
        trampoline.data.nodal_kinematic_target,
    )
    reset_ball(scene, ball)
    reset_trampoline(scene, trampoline, targets)

    if camera is not None:
        set_video_camera_view(camera, sim)
        camera.update(dt=args_cli.sim_dt)

    center_z0 = float(trampoline.data.nodal_pos_w[0, center_node_ids[0], 2])
    min_center_z = center_z0
    min_ball_z = float(ball.data.root_pos_w[0, 2])
    min_ball_z_xy = (float(ball.data.root_pos_w[0, 0]), float(ball.data.root_pos_w[0, 1]))
    contact_started = False
    released = False
    contact_start_s = float("nan")
    contact_end_s = float("nan")
    impact_vz = float("nan")
    release_vz = float("nan")
    max_rebound = -float("inf")
    first_apex_found = False
    apex_armed = False
    previous_ball_vz = float("nan")
    first_apex_time_s = float("nan")
    first_apex_height_m = float("nan")
    stable_step_count = 0
    stable_window_steps = max(1, int(round(args_cli.stable_window_s / args_cli.sim_dt)))
    stable = False
    stable_time_s = float("nan")
    stable_ball_z = float("nan")
    stable_compression = float("nan")
    rows: list[dict[str, Any]] = []

    video_writer = None
    video_frame_stride = max(1, int(round(1.0 / max(args_cli.video_fps * args_cli.sim_dt, 1.0e-9)))) if args_cli.video else 1
    if args_cli.video:
        if imageio is None:
            raise RuntimeError(f"imageio is required for video output: {_IMAGEIO_IMPORT_ERROR}")
        video_writer = imageio.get_writer(video_path, fps=args_cli.video_fps, macro_block_size=1)

    completed_successfully = False
    loop_start_time = time.perf_counter()
    try:
        for step in range(int(args_cli.sim_time / args_cli.sim_dt)):
            trampoline.write_nodal_kinematic_target_to_sim(targets)
            ball.write_data_to_sim()
            sim.step()
            scene.update(args_cli.sim_dt)
            if camera is not None:
                camera.update(dt=args_cli.sim_dt)

            t = (step + 1) * args_cli.sim_dt
            ball_pos = ball.data.root_pos_w[0]
            ball_vel = ball.data.root_lin_vel_w[0]
            ball_z = float(ball_pos[2])
            ball_vz = float(ball_vel[2])
            bottom_z = ball_z - BALL_RADIUS
            center_pos = trampoline.data.nodal_pos_w[0, center_node_ids[0], :3]
            center_vel_tensor = getattr(trampoline.data, "nodal_vel_w", None)
            if center_vel_tensor is not None:
                center_vel = center_vel_tensor[0, center_node_ids[0], :3]
                center_vz = float(center_vel[2])
            else:
                center_vz = float("nan")
            center_z = float(center_pos[2])
            compression = float(center_z0 - center_z)

            min_center_z = min(min_center_z, center_z)
            if ball_z < min_ball_z:
                min_ball_z = ball_z
                min_ball_z_xy = (float(ball_pos[0]), float(ball_pos[1]))

            if not stable:
                stable_step_count = stable_step_count + 1 if abs(ball_vz) <= args_cli.stable_vz_threshold else 0
                if stable_step_count >= stable_window_steps:
                    stable = True
                    stable_time_s = t
                    stable_ball_z = ball_z
                    stable_compression = compression

            if contact_started and not first_apex_found:
                if ball_vz > args_cli.apex_vz_hysteresis:
                    apex_armed = True
                if apex_armed and math.isfinite(previous_ball_vz) and previous_ball_vz > 0.0 and ball_vz <= 0.0:
                    first_apex_found = True
                    first_apex_time_s = t
                    first_apex_height_m = ball_z
                    apex_armed = False
            previous_ball_vz = ball_vz

            if not contact_started and bottom_z <= CONTACT_START_BOTTOM_Z:
                contact_started = True
                contact_start_s = t
                impact_vz = ball_vz

            if contact_started and not released and bottom_z >= CONTACT_END_BOTTOM_Z and ball_vz > 0.0:
                released = True
                contact_end_s = t
                release_vz = ball_vz

            if released:
                max_rebound = max(max_rebound, ball_z)

            rows.append(
                {
                    "step": step,
                    "time_s": t,
                    "ball_x_m": float(ball_pos[0]),
                    "ball_y_m": float(ball_pos[1]),
                    "ball_z_m": ball_z,
                    "ball_vx_mps": float(ball_vel[0]),
                    "ball_vy_mps": float(ball_vel[1]),
                    "ball_vz_mps": ball_vz,
                    "ball_bottom_z_m": bottom_z,
                    "trampoline_center_x_m": float(center_pos[0]),
                    "trampoline_center_y_m": float(center_pos[1]),
                    "trampoline_center_z_m": center_z,
                    "trampoline_center_vx_mps": float(center_vel[0]) if center_vel_tensor is not None else float("nan"),
                    "trampoline_center_vy_mps": float(center_vel[1]) if center_vel_tensor is not None else float("nan"),
                    "trampoline_center_vz_mps": center_vz,
                    "compression_m": compression,
                    "stable": int(stable),
                    "apex_armed": int(apex_armed),
                    "first_apex_found": int(first_apex_found),
                    "contact_started": int(contact_started),
                    "released": int(released),
                }
            )

            if video_writer is not None and step % video_frame_stride == 0:
                frame = _as_numpy_rgb(camera.data.output["rgb"])  # type: ignore[index]
                video_writer.append_data(frame)

        wall_time_s = time.perf_counter() - loop_start_time
        trajectory_path = timeseries_path
        write_timeseries_csv(trajectory_path, rows)

        dropped_below = min_ball_z < FALLTHROUGH_BALL_Z
        if dropped_below:
            min_ball_z_r = math.sqrt(min_ball_z_xy[0] ** 2 + min_ball_z_xy[1] ** 2)
            fell_off_edge = min_ball_z_r >= TRAMPOLINE_RADIUS
            fallthrough = dropped_below and not fell_off_edge
        else:
            fell_off_edge = False
            fallthrough = False
        if fallthrough or fell_off_edge:
            stable = False
            stable_time_s = float("nan")
            stable_ball_z = float("nan")
            stable_compression = float("nan")

        row = {
            "label": args_cli.label,
            "run_dir": str(run_dir),
            "ball_mass": args_cli.ball_mass,
            "ball_height": args_cli.ball_height,
            "thickness": args_cli.thickness,
            "trampoline_mass": args_cli.trampoline_mass,
            "sim_resolution": args_cli.sim_resolution,
            "pinned_node_count": int(pinned_mask[0].sum().item()),
            "youngs_modulus": args_cli.youngs_modulus,
            "elasticity_damping": args_cli.elasticity_damping,
            "damping_scale": args_cli.damping_scale,
            "sim_time": args_cli.sim_time,
            "sim_dt": args_cli.sim_dt,
            "wall_time_s": round(wall_time_s, 2),
            "video": int(args_cli.video),
            "stable": int(stable),
            "stable_time_s": finite_or_nan(stable_time_s),
            "stable_ball_z_m": finite_or_nan(stable_ball_z),
            "stable_compression_m": finite_or_nan(stable_compression),
            "fallthrough": int(fallthrough),
            "fell_off_edge": int(fell_off_edge),
            "min_ball_z_x_m": min_ball_z_xy[0],
            "min_ball_z_y_m": min_ball_z_xy[1],
            "first_apex_found": int(first_apex_found),
            "first_apex_time_s": finite_or_nan(first_apex_time_s),
            "first_apex_height_m": finite_or_nan(first_apex_height_m),
            "contact_started": int(contact_started),
            "released": int(released),
            "contact_start_s": finite_or_nan(contact_start_s),
            "contact_duration_s": finite_or_nan(contact_end_s - contact_start_s if contact_started and released else float("nan")),
            "impact_vz_mps": finite_or_nan(impact_vz),
            "release_vz_mps": finite_or_nan(release_vz),
            "max_compression_m": float(center_z0 - min_center_z),
            "min_ball_z_m": float(min_ball_z),
            "rebound_height_m": finite_or_nan(first_apex_height_m),
        }

        write_summary_row(summary_path, row)
        plot_paths = plot_timeseries(rows, run_dir)

        print(row, flush=True)
        print(f"WROTE {summary_path}", flush=True)
        print(f"WROTE {trajectory_path}", flush=True)
        for plot_path in plot_paths:
            print(f"WROTE {plot_path}", flush=True)
        if args_cli.video:
            print(f"WROTE {video_path}", flush=True)
        print(f"RUN_DIR {run_dir}", flush=True)
        completed_successfully = True
    finally:
        if video_writer is not None:
            video_writer.close()
        if completed_successfully:
            print("[INFO] Artifacts complete; exiting without Isaac Sim shutdown to avoid close() hang.", flush=True)
            os._exit(0)
        simulation_app.close()


if __name__ == "__main__":
    main()
