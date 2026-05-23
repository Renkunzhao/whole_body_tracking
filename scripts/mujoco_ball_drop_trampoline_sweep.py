from __future__ import annotations

import argparse
import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


DEFAULT_ASSET_DIR = Path("/home/rkz/code/unitree_ws/src/unitree_mujoco/unitree_robots/go2")
DEFAULT_OUTPUT = Path("logs/mujoco_ball_drop_trampoline_sweep.csv")
DEFAULT_BALL_MASS = 4.02
DEFAULT_BALL_HEIGHT = 1.0
DEFAULT_SIM_TIME = 4.0
CONTACT_FORCE_THRESHOLD = 1.0e-3
DEFAULT_STABLE_VZ_THRESHOLD = 0.05
DEFAULT_STABLE_WINDOW_S = 0.2
DEFAULT_APEX_VZ_HYSTERESIS = 0.05
FALLTHROUGH_BALL_Z = -2.0
DEFAULT_USABLE_RADIUS = 1.5
DEFAULT_TRAMPOLINE_MASS = 10.0
DEFAULT_TRAMPOLINE_RADIUS = 0.03
DEFAULT_TRAMPOLINE_SPACING = 1.5
DEFAULT_EDGE_SOLREF = "0.01 1"
DEFAULT_EDGE_SOLIMP = "0.8 0.9 0.001 0.1 6"
DEFAULT_BALL_X = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MuJoCo ball-drop sweeps on the Go2 trampoline flexcomp.")
    parser.add_argument("--asset_dir", type=Path, default=DEFAULT_ASSET_DIR, help="Directory containing ball.xml and trampoline.xml.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV output path.")
    parser.add_argument("--trajectory_dir", type=Path, default=None, help="Optional directory for per-condition trajectory CSV files.")
    parser.add_argument("--sim_time", type=float, default=DEFAULT_SIM_TIME, help="Simulation duration in seconds.")
    parser.add_argument("--ball_mass", type=float, default=DEFAULT_BALL_MASS, help="Ball mass in kilograms.")
    parser.add_argument("--ball_height", type=float, default=DEFAULT_BALL_HEIGHT, help="Initial ball center height.")
    parser.add_argument("--force_threshold", type=float, default=CONTACT_FORCE_THRESHOLD, help="Touch force threshold for contact state.")
    parser.add_argument("--stable_vz_threshold", type=float, default=DEFAULT_STABLE_VZ_THRESHOLD, help="Vertical speed threshold for stable-time detection.")
    parser.add_argument("--stable_window_s", type=float, default=DEFAULT_STABLE_WINDOW_S, help="Required consecutive low-speed duration for stable-time detection.")
    parser.add_argument("--apex_vz_hysteresis", type=float, default=DEFAULT_APEX_VZ_HYSTERESIS, help="Velocity hysteresis used to arm apex detection and suppress jitter.")
    parser.add_argument("--solref", type=str, nargs="*", default=["0.002 1", "0.005 1", "0.02 1"], help="Extra edge solref values to sweep.")
    parser.add_argument("--radius", type=float, nargs="*", default=[0.02, 0.05, 0.10], help="Extra flexcomp radius values to sweep.")
    parser.add_argument("--mass", type=float, nargs="*", default=[1.0, 5.0, 30.0], help="Extra flexcomp mass values to sweep.")
    parser.add_argument("--spacing", type=float, nargs="*", default=[1.0, 2.0], help="Extra flexcomp spacing values to sweep.")
    parser.add_argument("--ball_x", type=float, nargs="*", default=[0.15, 0.30], help="Extra ball x-offset values to sweep.")
    return parser.parse_args()


def format_float_label(value: float) -> str:
    return f"{value:g}"


def sensor_slice(model: mujoco.MjModel, name: str) -> slice:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise RuntimeError(f"Sensor '{name}' not found in assembled MuJoCo model.")
    start = int(model.sensor_adr[sensor_id])
    dim = int(model.sensor_dim[sensor_id])
    return slice(start, start + dim)


def set_ball_mass(ball_body: ET.Element, ball_mass: float) -> None:
    inertial = ball_body.find("inertial")
    if inertial is not None:
        inertial.set("mass", f"{ball_mass:g}")
        return

    geoms = ball_body.findall(".//geom")
    if not geoms:
        raise RuntimeError("Could not find a geom or inertial element for the MuJoCo ball mass.")
    geoms[0].set("mass", f"{ball_mass:g}")
    for geom in geoms[1:]:
        geom.attrib.pop("mass", None)


def build_conditions(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    conditions: list[tuple[str, dict[str, Any]]] = [("nominal", {})]
    conditions.extend((f"solref_{value.split()[0]}", {"solref": value}) for value in args.solref)
    conditions.extend((f"radius_{format_float_label(value)}", {"radius": value}) for value in args.radius)
    conditions.extend((f"mass_{format_float_label(value)}", {"mass": value}) for value in args.mass)
    conditions.extend((f"spacing_{format_float_label(value)}", {"spacing": value}) for value in args.spacing)
    conditions.extend((f"offset_x_{format_float_label(value)}", {"ball_x": value}) for value in args.ball_x)
    return conditions


def build_model(
    asset_dir: Path,
    *,
    ball_mass: float,
    ball_height: float,
    ball_x: float = DEFAULT_BALL_X,
    mass: float = DEFAULT_TRAMPOLINE_MASS,
    radius: float = DEFAULT_TRAMPOLINE_RADIUS,
    spacing: float = DEFAULT_TRAMPOLINE_SPACING,
    solref: str = DEFAULT_EDGE_SOLREF,
    solimp: str = DEFAULT_EDGE_SOLIMP,
) -> mujoco.MjModel:
    ball = ET.parse(asset_dir / "ball.xml").getroot()
    trampoline = ET.parse(asset_dir / "trampoline.xml").getroot()

    ball_body = ball.find("worldbody").find("body[@name='foot']")
    if ball_body is None:
        raise RuntimeError("Could not find body 'foot' in ball.xml.")
    ball_body.set("pos", f"{ball_x:g} 0 {ball_height:g}")
    set_ball_mass(ball_body, ball_mass)

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
    model_xml.append(world)

    sensor = ball.find("sensor")
    if sensor is None:
        raise RuntimeError("Could not find sensor block in ball.xml.")
    model_xml.append(sensor)

    return mujoco.MjModel.from_xml_string(ET.tostring(model_xml, encoding="unicode"))


def finite_or_nan(value: float) -> float:
    return value if math.isfinite(value) else float("nan")


def top_center_flex_vertex_id(data: mujoco.MjData) -> int:
    xy = data.flexvert_xpos[:, :2]
    center_xy = 0.5 * (xy.min(axis=0) + xy.max(axis=0))
    radial_distance = np.linalg.norm(xy - center_xy, axis=1)
    min_radial_distance = float(radial_distance.min())
    center_candidates = np.flatnonzero(radial_distance <= min_radial_distance + 1.0e-9)
    return int(center_candidates[np.argmax(data.flexvert_xpos[center_candidates, 2])])


def write_trajectory_csv(trajectory_dir: Path, label: str, rows: list[dict[str, Any]]) -> Path:
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = trajectory_dir / f"{label}_trajectory.csv"
    with trajectory_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return trajectory_path


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


def run_condition(args: argparse.Namespace, label: str, overrides: dict[str, Any]) -> dict[str, Any]:
    model = build_model(args.asset_dir, ball_mass=args.ball_mass, ball_height=args.ball_height, **overrides)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    touch_slice = sensor_slice(model, "foot_touch")
    pos_slice = sensor_slice(model, "foot_pos")
    vel_slice = sensor_slice(model, "foot_linvel")
    center_id = top_center_flex_vertex_id(data)
    center_z0 = float(data.flexvert_xpos[center_id, 2])
    static_sag_m = -center_z0

    min_center_z = center_z0
    min_ball_z = float(data.sensordata[pos_slice][2])
    min_ball_z_xy = (float(data.sensordata[pos_slice][0]), float(data.sensordata[pos_slice][1]))
    first_min_ball_z_m = float("nan")
    first_min_ball_z_time_s = float("nan")
    first_min_armed = False
    first_apex_found = False
    first_apex_armed = False
    first_apex_time_s = float("nan")
    first_apex_height_m = float("nan")
    second_apex_found = False
    second_apex_armed = False
    second_apex_time_s = float("nan")
    second_apex_height_m = float("nan")
    previous_ball_vz = float("nan")
    stable_step_count = 0
    stable_window_steps = max(1, int(round(args.stable_window_s / model.opt.timestep)))
    stable = False
    stable_time_s = float("nan")
    stable_ball_z = float("nan")
    stable_compression = float("nan")
    peak_force = 0.0
    impulse = 0.0
    trajectory_rows: list[dict[str, Any]] = []
    step = 0

    while data.time < args.sim_time:
        mujoco.mj_step(model, data)
        force = float(data.sensordata[touch_slice][0])
        ball_z = float(data.sensordata[pos_slice][2])
        ball_vz = float(data.sensordata[vel_slice][2])
        ball_x = float(data.sensordata[pos_slice][0])
        ball_y = float(data.sensordata[pos_slice][1])
        center_z = float(data.flexvert_xpos[center_id, 2])
        center_vz = float(data.flexvert_xvel[center_id, 2]) if getattr(data, "flexvert_xvel", None) is not None else float("nan")
        compression = center_z0 - center_z

        min_center_z = min(min_center_z, center_z)
        if ball_z < min_ball_z:
            min_ball_z = ball_z
            min_ball_z_xy = (ball_x, ball_y)
        peak_force = max(peak_force, force)
        impulse += max(force, 0.0) * model.opt.timestep

        if not stable:
            stable_step_count = stable_step_count + 1 if abs(ball_vz) <= args.stable_vz_threshold else 0
            if stable_step_count >= stable_window_steps:
                stable = True
                stable_time_s = float(data.time)
                stable_ball_z = ball_z
                stable_compression = compression

        if math.isfinite(previous_ball_vz):
            if not first_min_armed and ball_vz < 0.0:
                first_min_armed = True

            if first_min_armed and math.isnan(first_min_ball_z_time_s) and previous_ball_vz < 0.0 and ball_vz >= 0.0:
                first_min_ball_z_m = ball_z
                first_min_ball_z_time_s = float(data.time)

            if math.isfinite(first_min_ball_z_time_s) and not first_apex_found:
                if ball_vz > args.apex_vz_hysteresis:
                    first_apex_armed = True
                if first_apex_armed and previous_ball_vz > 0.0 and ball_vz <= 0.0:
                    first_apex_found = True
                    first_apex_time_s = float(data.time)
                    first_apex_height_m = ball_z

            if first_apex_found and not second_apex_found:
                if not second_apex_armed and previous_ball_vz <= 0.0 and ball_vz > 0.0 and data.time > first_apex_time_s:
                    second_apex_armed = True
                if second_apex_armed and previous_ball_vz > 0.0 and ball_vz <= 0.0:
                    second_apex_found = True
                    second_apex_time_s = float(data.time)
                    second_apex_height_m = ball_z

        previous_ball_vz = ball_vz
        if args.trajectory_dir is not None:
            trajectory_rows.append(
                {
                    "step": step,
                    "time_s": float(data.time),
                    "ball_x_m": ball_x,
                    "ball_y_m": ball_y,
                    "ball_z_m": ball_z,
                    "ball_vx_mps": float(data.sensordata[vel_slice][0]),
                    "ball_vy_mps": float(data.sensordata[vel_slice][1]),
                    "ball_vz_mps": ball_vz,
                    "trampoline_center_z_m": center_z,
                    "trampoline_center_vz_mps": center_vz,
                    "compression_m": compression,
                    "force_N": force,
                    "stable": int(stable),
                    "first_min_found": int(math.isfinite(first_min_ball_z_time_s)),
                    "first_apex_found": int(first_apex_found),
                    "second_apex_found": int(second_apex_found),
                }
            )
        step += 1

    dropped_below = min_ball_z < FALLTHROUGH_BALL_Z
    min_ball_z_r = math.hypot(min_ball_z_xy[0], min_ball_z_xy[1])
    usable_radius = overrides.get("spacing", DEFAULT_USABLE_RADIUS)
    fell_off_edge = bool(dropped_below and min_ball_z_r >= usable_radius)
    fallthrough = bool(dropped_below and not fell_off_edge)
    if fallthrough or fell_off_edge:
        stable = False
        stable_time_s = float("nan")
        stable_ball_z = float("nan")
        stable_compression = float("nan")

    damping_ratio = (
        second_apex_height_m / first_apex_height_m
        if first_apex_found and second_apex_found and first_apex_height_m > 0.0
        else float("nan")
    )
    trajectory_path = ""
    if args.trajectory_dir is not None and trajectory_rows:
        trajectory_path = str(write_trajectory_csv(args.trajectory_dir, label, trajectory_rows))

    return {
        "condition": label,
        "ball_mass": args.ball_mass,
        "ball_height_m": args.ball_height,
        "trampoline_mass": overrides.get("mass", DEFAULT_TRAMPOLINE_MASS),
        "trampoline_radius": overrides.get("radius", DEFAULT_TRAMPOLINE_RADIUS),
        "trampoline_spacing": overrides.get("spacing", DEFAULT_TRAMPOLINE_SPACING),
        "usable_radius_m": usable_radius,
        "edge_solref": overrides.get("solref", DEFAULT_EDGE_SOLREF),
        "edge_solimp": overrides.get("solimp", DEFAULT_EDGE_SOLIMP),
        "ball_x_m": overrides.get("ball_x", DEFAULT_BALL_X),
        "trajectory_path": trajectory_path,
        "static_sag_m": finite_or_nan(static_sag_m),
        "fallthrough": int(fallthrough),
        "fell_off_edge": int(fell_off_edge),
        "final_state": classify_final_state(
            fallthrough=fallthrough,
            fell_off_edge=fell_off_edge,
            stable=stable,
            second_apex_found=second_apex_found,
            first_apex_found=first_apex_found,
            first_min_found=math.isfinite(first_min_ball_z_time_s),
        ),
        "min_ball_z_m": min_ball_z,
        "min_ball_z_x_m": min_ball_z_xy[0],
        "min_ball_z_y_m": min_ball_z_xy[1],
        "first_min_ball_z_m": finite_or_nan(first_min_ball_z_m),
        "first_min_ball_z_time_s": finite_or_nan(first_min_ball_z_time_s),
        "first_apex_height_m": finite_or_nan(first_apex_height_m),
        "first_apex_time_s": finite_or_nan(first_apex_time_s),
        "rebound_height_m": finite_or_nan(first_apex_height_m),
        "second_apex_height_m": finite_or_nan(second_apex_height_m),
        "second_apex_time_s": finite_or_nan(second_apex_time_s),
        "damping_ratio": finite_or_nan(damping_ratio),
        "max_compression_m": float(center_z0 - min_center_z),
        "stable": int(stable),
        "stable_time_s": finite_or_nan(stable_time_s),
        "stable_ball_z_m": finite_or_nan(stable_ball_z),
        "stable_compression_m": finite_or_nan(stable_compression),
        "peak_force_N": peak_force,
        "impulse_Ns": impulse,
    }


def main() -> None:
    args = parse_args()
    rows = [run_condition(args, label, overrides) for label, overrides in build_conditions(args)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row, flush=True)
    print(f"WROTE {args.output}", flush=True)


if __name__ == "__main__":
    main()
