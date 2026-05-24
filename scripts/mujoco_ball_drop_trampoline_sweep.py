from __future__ import annotations

import argparse
import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


DEFAULT_ASSET_DIR = Path("/home/rkz/code/unitree_ws/src/unitree_mujoco/unitree_robots/go2")
DEFAULT_OUTPUT_ROOT = Path("logs/mujoco_ball_drop_runs")
SUMMARY_FILENAME = "ball_drop_summary.csv"
TRAJECTORY_FILENAME = "ball_drop_trajectory.csv"
VIDEO_FILENAME = "ball_drop_video.mp4"
VERTICAL_STATE_PLOT_FILENAME = "ball_drop_vertical_state.png"
COMPRESSION_PLOT_FILENAME = "ball_drop_compression.png"
SWEEP_SUMMARY_FILENAME = "ball_drop_sweep_summary.csv"
BALL_RADIUS = 0.022
DEFAULT_BALL_HEIGHT = 1.0
DEFAULT_SIM_TIME = 4.0
CONTACT_FORCE_THRESHOLD = 1.0e-3
DEFAULT_STABLE_VZ_THRESHOLD = 0.05
DEFAULT_STABLE_WINDOW_S = 0.2
DEFAULT_APEX_VZ_HYSTERESIS = 0.05
FALLTHROUGH_BALL_Z = -2.0
DEFAULT_TRAMPOLINE_MASS = 10.0
DEFAULT_TRAMPOLINE_RADIUS = 0.03
DEFAULT_TRAMPOLINE_SPACING = 1.5
DEFAULT_EDGE_SOLREF = "0.01 1"
DEFAULT_EDGE_SOLIMP = "0.8 0.9 0.001 0.1 6"
DEFAULT_BALL_X = 0.0
DEFAULT_VIDEO_WIDTH = 1280
DEFAULT_VIDEO_HEIGHT = 720
DEFAULT_VIDEO_FPS = 60

CONDITION_FIELDS: dict[str, type] = {
    "sim_time": float,
    "ball_height": float,
    "force_threshold": float,
    "stable_vz_threshold": float,
    "stable_window_s": float,
    "apex_vz_hysteresis": float,
    "mass": float,
    "radius": float,
    "spacing": float,
    "solref": str,
    "solimp": str,
    "ball_x": float,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MuJoCo ball-drop conditions on the Go2 trampoline flexcomp.")
    parser.add_argument("--asset_dir", type=Path, default=DEFAULT_ASSET_DIR, help="Directory containing ball.xml and trampoline.xml.")
    parser.add_argument("--output", type=Path, default=None, help="Summary CSV path. If omitted, a run directory is auto-created under logs/mujoco_ball_drop_runs/.")
    parser.add_argument("--label", type=str, default="nominal", help="Condition label used when no sweep is configured.")
    parser.add_argument("--sim_time", type=float, default=DEFAULT_SIM_TIME, help="Simulation duration in seconds.")
    parser.add_argument("--ball_height", type=float, default=DEFAULT_BALL_HEIGHT, help="Initial ball center height.")
    parser.add_argument("--force_threshold", type=float, default=CONTACT_FORCE_THRESHOLD, help="Touch force threshold for contact state.")
    parser.add_argument("--stable_vz_threshold", type=float, default=DEFAULT_STABLE_VZ_THRESHOLD, help="Vertical speed threshold for stable-time detection.")
    parser.add_argument("--stable_window_s", type=float, default=DEFAULT_STABLE_WINDOW_S, help="Required consecutive low-speed duration for stable-time detection.")
    parser.add_argument("--apex_vz_hysteresis", type=float, default=DEFAULT_APEX_VZ_HYSTERESIS, help="Velocity hysteresis used to arm apex detection and suppress jitter.")
    parser.add_argument("--mass", type=float, default=DEFAULT_TRAMPOLINE_MASS, help="Flexcomp trampoline mass.")
    parser.add_argument("--radius", type=float, default=DEFAULT_TRAMPOLINE_RADIUS, help="Flexcomp radius.")
    parser.add_argument("--spacing", type=float, default=DEFAULT_TRAMPOLINE_SPACING, help="Flexcomp spacing.")
    parser.add_argument("--solref", type=str, default=DEFAULT_EDGE_SOLREF, help="Edge solref value.")
    parser.add_argument("--solimp", type=str, default=DEFAULT_EDGE_SOLIMP, help="Edge solimp value.")
    parser.add_argument("--ball_x", type=float, default=DEFAULT_BALL_X, help="Initial ball x-offset in meters.")
    parser.add_argument("--sweep", action="append", nargs="+", default=[], metavar=("PARAM", "VALUE"), help="Cartesian-product sweep, e.g. --sweep solref '0.012 1' '0.015 1'.")
    parser.add_argument("--sweep_config", type=Path, default=None, help="YAML or JSON file with base, sweep, and/or conditions.")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True, help="Save an MP4 video for each condition.")
    parser.add_argument("--video_width", type=int, default=DEFAULT_VIDEO_WIDTH, help="Video width in pixels.")
    parser.add_argument("--video_height", type=int, default=DEFAULT_VIDEO_HEIGHT, help="Video height in pixels.")
    parser.add_argument("--video_fps", type=int, default=DEFAULT_VIDEO_FPS, help="Video frame rate.")
    return parser.parse_args()


def format_float_label(value: float) -> str:
    return f"{value:g}"


def format_value_label(value: Any) -> str:
    if isinstance(value, float):
        return format_float_label(value)
    return sanitize_token(str(value))


def sanitize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "condition"


def is_sweep_mode(args: argparse.Namespace) -> bool:
    return bool(args.sweep or args.sweep_config is not None)


def build_output_stem(label: str, conditions: list[tuple[str, dict[str, Any]]]) -> str:
    if len(conditions) != 1:
        return sanitize_token(label or "sweep")
    condition_label, condition = conditions[0]
    parts = [
        sanitize_token(condition_label),
        f"t{format_float_label(condition['sim_time'])}",
        f"h{format_float_label(condition['ball_height'])}",
        f"m{format_float_label(condition['mass'])}",
        f"r{format_float_label(condition['radius'])}",
        f"sp{format_float_label(condition['spacing'])}",
        f"x{format_float_label(condition['ball_x'])}",
        f"solref{sanitize_token(str(condition['solref']))}",
    ]
    return "__".join(parts)


def resolve_output_path(args: argparse.Namespace, conditions: list[tuple[str, dict[str, Any]]], sweep_mode: bool) -> Path:
    if args.output is not None:
        return args.output.expanduser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = DEFAULT_OUTPUT_ROOT / f"{timestamp}__{build_output_stem(args.label, conditions)}"
    return output_dir / (SWEEP_SUMMARY_FILENAME if sweep_mode else SUMMARY_FILENAME)


def condition_artifact_paths(output_path: Path, label: str, sweep_mode: bool) -> tuple[Path, Path]:
    if sweep_mode:
        run_dir = output_path.parent / sanitize_token(label)
        return run_dir, run_dir / SUMMARY_FILENAME
    return output_path.parent, output_path


def load_sweep_config(path: Path) -> dict[str, Any]:
    with path.expanduser().open(encoding="utf-8") as config_file:
        if path.suffix.lower() == ".json":
            config = json.load(config_file)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML sweep configs require PyYAML. Use JSON or install PyYAML.") from exc
            config = yaml.safe_load(config_file)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Sweep config must be a mapping: {path}")
    return config


def coerce_condition_field(field_name: str, value: Any) -> Any:
    if field_name not in CONDITION_FIELDS:
        valid = ", ".join(sorted(CONDITION_FIELDS))
        raise ValueError(f"Unsupported sweep parameter {field_name!r}. Valid names: {valid}.")
    return CONDITION_FIELDS[field_name](value)


def base_condition_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {field_name: getattr(args, field_name) for field_name in CONDITION_FIELDS}


def apply_config_options(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if "output" in config:
        args.output = Path(config["output"])
    if "video" in config:
        if not isinstance(config["video"], bool):
            raise ValueError("sweep_config field 'video' must be a boolean.")
        args.video = config["video"]
    if "asset_dir" in config:
        args.asset_dir = Path(config["asset_dir"])
    for field_name in ("video_width", "video_height", "video_fps"):
        if field_name in config:
            setattr(args, field_name, int(config[field_name]))


def apply_config_base(base_condition: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    raw_base = config.get("base", {})
    if raw_base is None:
        return base_condition
    if not isinstance(raw_base, dict):
        raise ValueError("sweep_config field 'base' must be a mapping.")
    condition = dict(base_condition)
    for field_name, value in raw_base.items():
        condition[str(field_name)] = coerce_condition_field(str(field_name), value)
    return condition


def parse_cli_sweeps(sweep_specs: list[list[str]]) -> dict[str, list[Any]]:
    sweeps: dict[str, list[Any]] = {}
    for sweep_spec in sweep_specs:
        if len(sweep_spec) < 2:
            raise ValueError(f"Expected --sweep PARAM VALUE [VALUE ...], got {sweep_spec}.")
        field_name = sweep_spec[0]
        if field_name in sweeps:
            raise ValueError(f"Duplicate --sweep for parameter {field_name!r}.")
        sweeps[field_name] = [coerce_condition_field(field_name, value) for value in sweep_spec[1:]]
    return sweeps


def parse_config_sweeps(raw_sweeps: Any) -> dict[str, list[Any]]:
    if raw_sweeps is None:
        return {}
    if not isinstance(raw_sweeps, dict):
        raise ValueError("sweep_config field 'sweep' must be a mapping.")
    sweeps = {}
    for field_name, raw_values in raw_sweeps.items():
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        sweeps[str(field_name)] = [coerce_condition_field(str(field_name), value) for value in values]
    return sweeps


def build_product_conditions(base_condition: dict[str, Any], sweeps: dict[str, list[Any]]) -> list[tuple[str, dict[str, Any]]]:
    if not sweeps:
        return []
    conditions = []
    field_names = list(sweeps)
    for values in product(*(sweeps[field_name] for field_name in field_names)):
        condition = dict(base_condition)
        label_parts = []
        for field_name, value in zip(field_names, values, strict=True):
            condition[field_name] = value
            label_parts.append(f"{field_name}_{format_value_label(value)}")
        conditions.append((sanitize_token("__".join(label_parts)), condition))
    return conditions


def parse_config_conditions(base_condition: dict[str, Any], raw_conditions: Any) -> list[tuple[str, dict[str, Any]]]:
    if raw_conditions is None:
        return []
    if not isinstance(raw_conditions, list):
        raise ValueError("sweep_config field 'conditions' must be a list of mappings.")
    conditions = []
    for index, raw_condition in enumerate(raw_conditions):
        if not isinstance(raw_condition, dict):
            raise ValueError("Each sweep_config condition must be a mapping.")
        label = sanitize_token(str(raw_condition.get("label", f"condition_{index:03d}")))
        condition = dict(base_condition)
        for field_name, value in raw_condition.items():
            if field_name == "label":
                continue
            condition[str(field_name)] = coerce_condition_field(str(field_name), value)
        conditions.append((label, condition))
    return conditions


def build_conditions(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    config = load_sweep_config(args.sweep_config) if args.sweep_config is not None else {}
    apply_config_options(args, config)
    base_condition = apply_config_base(base_condition_from_args(args), config)
    sweeps = parse_config_sweeps(config.get("sweep"))
    sweeps.update(parse_cli_sweeps(args.sweep))
    conditions = build_product_conditions(base_condition, sweeps)
    conditions.extend(parse_config_conditions(base_condition, config.get("conditions")))
    if conditions:
        return conditions
    return [(sanitize_token(str(config.get("label", args.label))), base_condition)]


def sensor_slice(model: Any, name: str) -> slice:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise RuntimeError(f"Sensor '{name}' not found in assembled MuJoCo model.")
    start = int(model.sensor_adr[sensor_id])
    dim = int(model.sensor_dim[sensor_id])
    return slice(start, start + dim)


def body_mass(model: Any, name: str) -> float:
    return float(np.asarray(model.body(name).mass).reshape(-1)[0])


def build_model(
    asset_dir: Path,
    *,
    ball_height: float,
    ball_x: float = DEFAULT_BALL_X,
    mass: float = DEFAULT_TRAMPOLINE_MASS,
    radius: float = DEFAULT_TRAMPOLINE_RADIUS,
    spacing: float = DEFAULT_TRAMPOLINE_SPACING,
    solref: str = DEFAULT_EDGE_SOLREF,
    solimp: str = DEFAULT_EDGE_SOLIMP,
    offwidth: int = DEFAULT_VIDEO_WIDTH,
    offheight: int = DEFAULT_VIDEO_HEIGHT,
) -> Any:
    ball = ET.parse(asset_dir / "ball.xml").getroot()
    trampoline = ET.parse(asset_dir / "trampoline.xml").getroot()

    ball_body = ball.find("worldbody").find("body[@name='foot']")
    if ball_body is None:
        raise RuntimeError("Could not find body 'foot' in ball.xml.")
    ball_body.set("pos", f"{ball_x:g} 0 {ball_height:g}")

    flex = trampoline.find("worldbody").find("flexcomp[@name='trampoline']")
    if flex is None:
        raise RuntimeError("Could not find flexcomp 'trampoline' in trampoline.xml.")
    flex.set("mass", f"{mass:g}")
    flex.set("radius", f"{radius:g}")
    flex.set("spacing", f"{spacing:g} {spacing:g} {spacing:g}")

    edge = flex.find("edge")
    if edge is None:
        raise RuntimeError("Could not find trampoline flexcomp edge equality settings.")
    edge.set("solref", solref)
    edge.set("solimp", solimp)

    world = ET.Element("worldbody")
    for child in ball.find("worldbody"):
        world.append(child)
    for child in trampoline.find("worldbody"):
        world.append(child)
    ET.SubElement(world, "light", {"name": "video_light", "pos": "0 -3 4", "dir": "0 1 -1", "diffuse": "1 1 1"})

    model_xml = ET.Element("mujoco", {"model": "ball_exact_trampoline"})
    ET.SubElement(
        model_xml,
        "option",
        {
            "timestep": "0.001",
            "solver": "PGS",
            "tolerance": "1e-6",
            "integrator": "implicitfast",
        },
    )
    ET.SubElement(model_xml, "size", {"memory": "10000M"})
    visual = ET.SubElement(model_xml, "visual")
    ET.SubElement(visual, "global", {"offwidth": str(offwidth), "offheight": str(offheight)})
    model_xml.append(world)

    sensor = ball.find("sensor")
    if sensor is None:
        raise RuntimeError("Could not find sensor block in ball.xml.")
    model_xml.append(sensor)

    return mujoco.MjModel.from_xml_string(ET.tostring(model_xml, encoding="unicode"))


def finite_or_nan(value: float) -> float:
    return value if math.isfinite(value) else float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_timeseries(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for MuJoCo ball-drop plots.") from exc

    times = np.asarray([row["time_s"] for row in rows], dtype=float)
    ball_z = np.asarray([row["ball_z_m"] for row in rows], dtype=float)
    ball_vz = np.asarray([row["ball_vz_mps"] for row in rows], dtype=float)
    center_z = np.asarray([row["trampoline_center_z_m"] for row in rows], dtype=float)
    center_vz = np.asarray([row["trampoline_center_vz_mps"] for row in rows], dtype=float)
    bottom_z = np.asarray([row["ball_bottom_z_m"] for row in rows], dtype=float)
    compression = np.asarray([row["compression_m"] for row in rows], dtype=float)

    state_path = output_dir / VERTICAL_STATE_PLOT_FILENAME
    compression_path = output_dir / COMPRESSION_PLOT_FILENAME

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(times, ball_z, label="ball z")
    ax[0].plot(times, center_z, label="trampoline center z")
    ax[0].plot(times, bottom_z, label="ball bottom z", linestyle="--", alpha=0.7)
    ax[0].set_ylabel("position [m]")
    ax[0].legend(loc="best")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(times, ball_vz, label="ball vz")
    ax[1].plot(times, center_vz, label="trampoline center vz")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("velocity [m/s]")
    ax[1].legend(loc="best")
    ax[1].grid(True, alpha=0.3)
    fig.suptitle("MuJoCo trampoline ball-drop vertical state")
    fig.tight_layout()
    fig.savefig(state_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, compression, label="center compression")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("compression [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.suptitle("MuJoCo trampoline ball-drop compression")
    fig.tight_layout()
    fig.savefig(compression_path, dpi=160)
    plt.close(fig)

    return [state_path, compression_path]


def make_video_camera() -> Any:
    camera = mujoco.MjvCamera()
    camera.lookat[:] = np.array([0.0, 0.0, 0.25])
    camera.distance = 4.0
    camera.azimuth = 135.0
    camera.elevation = -25.0
    return camera


def make_video_writer(args: argparse.Namespace, output_dir: Path):
    if not args.video:
        return None, ""
    if args.video_fps <= 0 or args.video_width <= 0 or args.video_height <= 0:
        raise ValueError("Video width, height, and fps must be positive.")
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio is required for MuJoCo MP4 output.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / VIDEO_FILENAME
    return imageio.get_writer(video_path, fps=args.video_fps, macro_block_size=1), str(video_path)


def run_condition(args: argparse.Namespace, label: str, condition: dict[str, Any], run_dir: Path, summary_path: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = run_dir / TRAJECTORY_FILENAME

    model = build_model(
        args.asset_dir,
        ball_height=condition["ball_height"],
        ball_x=condition["ball_x"],
        mass=condition["mass"],
        radius=condition["radius"],
        spacing=condition["spacing"],
        solref=condition["solref"],
        solimp=condition["solimp"],
        offwidth=args.video_width,
        offheight=args.video_height,
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    touch_slice = sensor_slice(model, "foot_touch")
    pos_slice = sensor_slice(model, "foot_pos")
    vel_slice = sensor_slice(model, "foot_linvel")
    center_id = int(np.argmin(np.linalg.norm(data.flexvert_xpos[:, :2], axis=1)))
    center_z0 = float(data.flexvert_xpos[center_id, 2])
    previous_center_pos = data.flexvert_xpos[center_id, :3].copy()

    contact_started = False
    released = False
    contact_start_s = float("nan")
    contact_end_s = float("nan")
    impact_vz = float("nan")
    release_vz = float("nan")
    min_center_z = center_z0
    min_ball_z = float(data.sensordata[pos_slice][2])
    peak_force = 0.0
    impulse = 0.0
    previous_contact = False
    first_apex_found = False
    apex_armed = False
    previous_ball_vz = float("nan")
    first_apex_time_s = float("nan")
    first_apex_height_m = float("nan")
    stable_step_count = 0
    stable_window_steps = max(1, int(round(condition["stable_window_s"] / model.opt.timestep)))
    stable = False
    stable_time_s = float("nan")
    stable_ball_z = float("nan")
    stable_compression = float("nan")
    rows: list[dict[str, Any]] = []

    video_writer, video_path_text = make_video_writer(args, run_dir)
    renderer = None
    camera = None
    video_frame_stride = max(1, int(round(1.0 / max(args.video_fps * model.opt.timestep, 1.0e-9))))

    try:
        if video_writer is not None:
            renderer = mujoco.Renderer(model, height=args.video_height, width=args.video_width)
            camera = make_video_camera()
        step = 0
        while data.time < condition["sim_time"]:
            mujoco.mj_step(model, data)
            force = float(data.sensordata[touch_slice][0])
            ball_pos = data.sensordata[pos_slice].copy()
            ball_vel = data.sensordata[vel_slice].copy()
            ball_z = float(ball_pos[2])
            ball_vz = float(ball_vel[2])
            ball_bottom_z = ball_z - BALL_RADIUS
            center_pos = data.flexvert_xpos[center_id, :3].copy()
            center_vel = (center_pos - previous_center_pos) / model.opt.timestep
            previous_center_pos = center_pos
            center_z = float(center_pos[2])
            center_vz = float(center_vel[2])
            compression = float(center_z0 - center_z)
            in_contact = force > condition["force_threshold"]

            min_center_z = min(min_center_z, center_z)
            min_ball_z = min(min_ball_z, ball_z)
            peak_force = max(peak_force, force)
            impulse += max(force, 0.0) * model.opt.timestep

            if not stable:
                stable_step_count = stable_step_count + 1 if abs(ball_vz) <= condition["stable_vz_threshold"] else 0
                if stable_step_count >= stable_window_steps:
                    stable = True
                    stable_time_s = float(data.time)
                    stable_ball_z = ball_z
                    stable_compression = compression

            if contact_started and not first_apex_found:
                if ball_vz > condition["apex_vz_hysteresis"]:
                    apex_armed = True
                if apex_armed and math.isfinite(previous_ball_vz) and previous_ball_vz > 0.0 and ball_vz <= 0.0:
                    first_apex_found = True
                    first_apex_time_s = float(data.time)
                    first_apex_height_m = ball_z
                    apex_armed = False
            previous_ball_vz = ball_vz

            if in_contact and not previous_contact and not contact_started:
                contact_started = True
                contact_start_s = float(data.time)
                impact_vz = ball_vz

            if contact_started and previous_contact and not in_contact and not released:
                released = True
                contact_end_s = float(data.time)
                release_vz = ball_vz

            rows.append(
                {
                    "step": step,
                    "time_s": float(data.time),
                    "ball_x_m": float(ball_pos[0]),
                    "ball_y_m": float(ball_pos[1]),
                    "ball_z_m": ball_z,
                    "ball_vx_mps": float(ball_vel[0]),
                    "ball_vy_mps": float(ball_vel[1]),
                    "ball_vz_mps": ball_vz,
                    "ball_bottom_z_m": ball_bottom_z,
                    "trampoline_center_x_m": float(center_pos[0]),
                    "trampoline_center_y_m": float(center_pos[1]),
                    "trampoline_center_z_m": center_z,
                    "trampoline_center_vx_mps": float(center_vel[0]),
                    "trampoline_center_vy_mps": float(center_vel[1]),
                    "trampoline_center_vz_mps": center_vz,
                    "compression_m": compression,
                    "contact_force_N": force,
                    "stable": int(stable),
                    "apex_armed": int(apex_armed),
                    "first_apex_found": int(first_apex_found),
                    "contact_started": int(contact_started),
                    "released": int(released),
                }
            )

            if renderer is not None and video_writer is not None and step % video_frame_stride == 0:
                renderer.update_scene(data, camera=camera)
                video_writer.append_data(renderer.render())

            previous_contact = in_contact
            step += 1
    finally:
        if video_writer is not None:
            video_writer.close()
        if renderer is not None:
            renderer.close()

    write_csv(trajectory_path, rows)
    plot_paths = plot_timeseries(rows, run_dir)

    fallthrough = min_ball_z < FALLTHROUGH_BALL_Z
    if fallthrough:
        stable = False
        stable_time_s = float("nan")
        stable_ball_z = float("nan")
        stable_compression = float("nan")

    row = {
        "label": label,
        "run_dir": str(run_dir),
        "ball_mass": body_mass(model, "foot"),
        "ball_height": condition["ball_height"],
        "trampoline_mass": condition["mass"],
        "trampoline_radius": condition["radius"],
        "trampoline_spacing": condition["spacing"],
        "edge_solref": condition["solref"],
        "edge_solimp": condition["solimp"],
        "ball_x_m": condition["ball_x"],
        "sim_time": condition["sim_time"],
        "sim_dt": float(model.opt.timestep),
        "video": int(args.video),
        "stable": int(stable),
        "stable_time_s": finite_or_nan(stable_time_s),
        "stable_ball_z_m": finite_or_nan(stable_ball_z),
        "stable_compression_m": finite_or_nan(stable_compression),
        "fallthrough": int(fallthrough),
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
        "peak_force_N": peak_force,
        "impulse_Ns": impulse,
        "video_path": video_path_text,
    }
    write_csv(summary_path, [row])

    print(row, flush=True)
    print(f"WROTE {summary_path}", flush=True)
    print(f"WROTE {trajectory_path}", flush=True)
    for plot_path in plot_paths:
        print(f"WROTE {plot_path}", flush=True)
    if args.video:
        print(f"WROTE {video_path_text}", flush=True)
    return row


def main() -> None:
    args = parse_args()
    sweep_mode = is_sweep_mode(args)
    conditions = build_conditions(args)
    output_path = resolve_output_path(args, conditions, sweep_mode)
    rows = []
    for label, condition in conditions:
        run_dir, summary_path = condition_artifact_paths(output_path, label, sweep_mode)
        rows.append(run_condition(args, label, condition, run_dir, summary_path))

    if sweep_mode:
        write_csv(output_path, rows)
        print(f"WROTE {output_path}", flush=True)
    print(f"RUN_DIR {output_path.parent}", flush=True)


if __name__ == "__main__":
    main()
