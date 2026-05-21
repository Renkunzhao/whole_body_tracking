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
BALL_RADIUS = 0.022
DEFAULT_BALL_HEIGHT = 1.0
DEFAULT_SIM_TIME = 4.0
CONTACT_FORCE_THRESHOLD = 1.0e-3
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
    parser.add_argument("--sim_time", type=float, default=DEFAULT_SIM_TIME, help="Simulation duration in seconds.")
    parser.add_argument("--ball_height", type=float, default=DEFAULT_BALL_HEIGHT, help="Initial ball center height.")
    parser.add_argument("--force_threshold", type=float, default=CONTACT_FORCE_THRESHOLD, help="Touch force threshold for contact state.")
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


def run_condition(args: argparse.Namespace, label: str, overrides: dict[str, Any]) -> dict[str, Any]:
    model = build_model(args.asset_dir, ball_height=args.ball_height, **overrides)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    touch_slice = sensor_slice(model, "foot_touch")
    pos_slice = sensor_slice(model, "foot_pos")
    vel_slice = sensor_slice(model, "foot_linvel")
    center_id = int(np.argmin(np.linalg.norm(data.flexvert_xpos[:, :2], axis=1)))
    center_z0 = float(data.flexvert_xpos[center_id, 2])

    contact_started = False
    released = False
    contact_start_s = float("nan")
    contact_end_s = float("nan")
    impact_vz = float("nan")
    release_vz = float("nan")
    min_center_z = center_z0
    min_ball_z = float(data.sensordata[pos_slice][2])
    max_rebound = -float("inf")
    peak_force = 0.0
    impulse = 0.0
    previous_contact = False

    while data.time < args.sim_time:
        mujoco.mj_step(model, data)
        force = float(data.sensordata[touch_slice][0])
        ball_z = float(data.sensordata[pos_slice][2])
        ball_vz = float(data.sensordata[vel_slice][2])
        center_z = float(data.flexvert_xpos[center_id, 2])
        in_contact = force > args.force_threshold

        min_center_z = min(min_center_z, center_z)
        min_ball_z = min(min_ball_z, ball_z)
        peak_force = max(peak_force, force)
        impulse += max(force, 0.0) * model.opt.timestep

        if in_contact and not previous_contact and not contact_started:
            contact_started = True
            contact_start_s = float(data.time)
            impact_vz = ball_vz

        if contact_started and previous_contact and not in_contact and not released:
            released = True
            contact_end_s = float(data.time)
            release_vz = ball_vz

        if released:
            max_rebound = max(max_rebound, ball_z)

        previous_contact = in_contact

    if contact_started and released:
        contact_duration_s = contact_end_s - contact_start_s
    else:
        contact_duration_s = float("nan")

    if released:
        rebound_height_m = max_rebound
    else:
        rebound_height_m = float("nan")

    return {
        "condition": label,
        "trampoline_mass": overrides.get("mass", DEFAULT_TRAMPOLINE_MASS),
        "trampoline_radius": overrides.get("radius", DEFAULT_TRAMPOLINE_RADIUS),
        "trampoline_spacing": overrides.get("spacing", DEFAULT_TRAMPOLINE_SPACING),
        "edge_solref": overrides.get("solref", DEFAULT_EDGE_SOLREF),
        "edge_solimp": overrides.get("solimp", DEFAULT_EDGE_SOLIMP),
        "ball_x_m": overrides.get("ball_x", DEFAULT_BALL_X),
        "ball_height_m": args.ball_height,
        "contact_started": int(contact_started),
        "released": int(released),
        "contact_start_s": finite_or_nan(contact_start_s),
        "contact_duration_s": finite_or_nan(contact_duration_s),
        "impact_vz_mps": finite_or_nan(impact_vz),
        "release_vz_mps": finite_or_nan(release_vz),
        "max_compression_m": float(center_z0 - min_center_z),
        "min_ball_z_m": min_ball_z,
        "rebound_height_m": finite_or_nan(rebound_height_m),
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
