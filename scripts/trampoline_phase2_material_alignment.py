from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MUJOCO_SCRIPT = SCRIPT_DIR / "mujoco_ball_drop_trampoline_sweep.py"
ISAACLAB_CONDITION_SCRIPT = SCRIPT_DIR / "isaaclab_trampoline_phase1_condition.py"
DEFAULT_ARTIFACT_ROOT = Path("logs/trampoline_phase2_material_alignment_runs")
DEFAULT_MUJOCO_ASSET_DIR = Path("/home/rkz/code/unitree_ws/src/unitree_mujoco/unitree_robots/go2")
G1_TOTAL_MASS_KG = 33.341142022
DEFAULT_BALL_MASS = 0.5 * G1_TOTAL_MASS_KG
DEFAULT_BALL_HEIGHT = 1.0
DEFAULT_SIM_TIME = 4.0
DEFAULT_SIM_DT = 0.002
DEFAULT_SIM_RESOLUTION = 15
DEFAULT_THICKNESS = 0.1
DEFAULT_YOUNGS_GRID = (8.0e5, 8.0e6)
DEFAULT_MASS_GRID = (5.0, 10.0, 15.0)
DEFAULT_ELASTICITY_DAMPING_GRID = (0.01, 0.1)
DEFAULT_DAMPING_SCALE_GRID = (1.0,)
DEFAULT_DYNAMIC_FRICTION_GRID = (0.8,)
DEFAULT_POISSONS_RATIO_GRID = (0.25, 0.45)
CURVE_FIELDS = {
    "ball_z_m": 1.0,
    "ball_vz_mps": 5.0,
    "compression_m": 0.2,
}
METRIC_FIELDS = {
    "max_compression_m": 0.1,
    "first_min_ball_z_m": 0.2,
    "first_min_ball_z_time_s": 0.5,
    "rebound_height_m": 0.2,
    "first_apex_time_s": 1.0,
    "damping_ratio": 0.2,
    "stable_time_s": 2.0,
}


@dataclass(frozen=True)
class Candidate:
    index: int
    youngs_modulus: float
    trampoline_mass: float
    elasticity_damping: float
    damping_scale: float
    dynamic_friction: float
    poissons_ratio: float


@dataclass(frozen=True)
class TargetPaths:
    summary: Path
    trajectory: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: fit IsaacLab material parameters to one MuJoCo ball-drop target curve.")
    parser.add_argument("--artifact_root", type=Path, default=DEFAULT_ARTIFACT_ROOT, help="Root directory for Phase 2 artifacts.")
    parser.add_argument("--mujoco_asset_dir", type=Path, default=DEFAULT_MUJOCO_ASSET_DIR, help="Directory containing MuJoCo ball.xml and trampoline.xml.")
    parser.add_argument("--mujoco_summary", type=Path, default=None, help="Existing one-row MuJoCo target summary CSV to reuse.")
    parser.add_argument("--mujoco_trajectory", type=Path, default=None, help="Existing MuJoCo target trajectory CSV to reuse.")
    parser.add_argument("--sim_time", type=float, default=DEFAULT_SIM_TIME, help="Simulation duration for both simulators.")
    parser.add_argument("--sim_dt", type=float, default=DEFAULT_SIM_DT, help="IsaacLab simulation timestep.")
    parser.add_argument("--ball_mass", type=float, default=DEFAULT_BALL_MASS, help="Ball mass in kilograms.")
    parser.add_argument("--ball_height", type=float, default=DEFAULT_BALL_HEIGHT, help="Ball center height.")
    parser.add_argument("--sim_resolution", type=int, default=DEFAULT_SIM_RESOLUTION, help="Fixed Phase 1 IsaacLab hexahedral resolution.")
    parser.add_argument("--thickness", type=float, default=DEFAULT_THICKNESS, help="Fixed Phase 1 IsaacLab trampoline thickness.")
    parser.add_argument("--youngs_modulus", type=float, nargs="+", default=list(DEFAULT_YOUNGS_GRID), help="IsaacLab Young's modulus grid.")
    parser.add_argument("--trampoline_mass", type=float, nargs="+", default=list(DEFAULT_MASS_GRID), help="IsaacLab trampoline mass grid.")
    parser.add_argument("--elasticity_damping", type=float, nargs="+", default=list(DEFAULT_ELASTICITY_DAMPING_GRID), help="IsaacLab elasticity damping grid.")
    parser.add_argument("--damping_scale", type=float, nargs="+", default=list(DEFAULT_DAMPING_SCALE_GRID), help="IsaacLab damping scale grid.")
    parser.add_argument("--dynamic_friction", type=float, nargs="+", default=list(DEFAULT_DYNAMIC_FRICTION_GRID), help="IsaacLab dynamic friction grid.")
    parser.add_argument("--poissons_ratio", type=float, nargs="+", default=list(DEFAULT_POISSONS_RATIO_GRID), help="IsaacLab Poisson's ratio grid.")
    parser.add_argument("--headless", action="store_true", default=True, help="Launch Isaac Sim headless.")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=False, help="Record IsaacLab candidate videos.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running simulators.")
    return parser.parse_args()


def finite_or_nan(value: float) -> float:
    return value if math.isfinite(value) else float("nan")


def parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def format_float_label(value: float) -> str:
    return f"{value:g}"


def sanitize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "run"


def candidate_label(candidate: Candidate) -> str:
    parts = [
        f"cand{candidate.index:03d}",
        f"E{format_float_label(candidate.youngs_modulus)}",
        f"tm{format_float_label(candidate.trampoline_mass)}",
        f"ed{format_float_label(candidate.elasticity_damping)}",
        f"ds{format_float_label(candidate.damping_scale)}",
        f"df{format_float_label(candidate.dynamic_friction)}",
        f"nu{format_float_label(candidate.poissons_ratio)}",
    ]
    return sanitize_token("__".join(parts))


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print(shlex.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_candidates(args: argparse.Namespace) -> list[Candidate]:
    candidates = []
    for index, values in enumerate(
        itertools.product(
            args.youngs_modulus,
            args.trampoline_mass,
            args.elasticity_damping,
            args.damping_scale,
            args.dynamic_friction,
            args.poissons_ratio,
        )
    ):
        youngs_modulus, trampoline_mass, elasticity_damping, damping_scale, dynamic_friction, poissons_ratio = values
        candidates.append(
            Candidate(
                index=index,
                youngs_modulus=youngs_modulus,
                trampoline_mass=trampoline_mass,
                elasticity_damping=elasticity_damping,
                damping_scale=damping_scale,
                dynamic_friction=dynamic_friction,
                poissons_ratio=poissons_ratio,
            )
        )
    return candidates


def resolve_mujoco_target(args: argparse.Namespace, run_root: Path) -> TargetPaths:
    if (args.mujoco_summary is None) != (args.mujoco_trajectory is None):
        raise ValueError("--mujoco_summary and --mujoco_trajectory must be provided together.")
    if args.mujoco_summary is not None and args.mujoco_trajectory is not None:
        return TargetPaths(args.mujoco_summary.expanduser().resolve(), args.mujoco_trajectory.expanduser().resolve())

    summary_path = run_root / "mujoco_target_summary.csv"
    trajectory_dir = run_root / "mujoco_target_trajectory"
    cmd = [
        sys.executable,
        str(MUJOCO_SCRIPT),
        "--asset_dir",
        str(args.mujoco_asset_dir),
        "--output",
        str(summary_path),
        "--trajectory_dir",
        str(trajectory_dir),
        "--sim_time",
        str(args.sim_time),
        "--ball_mass",
        str(args.ball_mass),
        "--ball_height",
        str(args.ball_height),
        "--solref",
        "--radius",
        "--mass",
        "--spacing",
        "--ball_x",
    ]
    run_command(cmd, dry_run=args.dry_run)
    return TargetPaths(summary_path, trajectory_dir / "nominal_trajectory.csv")


def run_isaaclab_candidate(args: argparse.Namespace, run_root: Path, candidate: Candidate) -> tuple[Path, Path, Path]:
    label = candidate_label(candidate)
    run_dir = run_root / "isaaclab_candidates" / label
    cmd = [
        sys.executable,
        str(ISAACLAB_CONDITION_SCRIPT),
        "--label",
        label,
        "--run_dir",
        str(run_dir),
        "--sim_time",
        str(args.sim_time),
        "--sim_dt",
        str(args.sim_dt),
        "--ball_height",
        str(args.ball_height),
        "--ball_mass",
        str(args.ball_mass),
        "--thickness",
        str(args.thickness),
        "--trampoline_mass",
        str(candidate.trampoline_mass),
        "--sim_resolution",
        str(args.sim_resolution),
        "--youngs_modulus",
        str(candidate.youngs_modulus),
        "--dynamic_friction",
        str(candidate.dynamic_friction),
        "--elasticity_damping",
        str(candidate.elasticity_damping),
        "--damping_scale",
        str(candidate.damping_scale),
        "--poissons_ratio",
        str(candidate.poissons_ratio),
    ]
    if args.headless:
        cmd.append("--headless")
    cmd.append("--video" if args.video else "--no-video")
    run_command(cmd, dry_run=args.dry_run)
    return run_dir, run_dir / "phase1_summary.csv", run_dir / "phase1_trajectory.csv"


def read_one_row(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in {path}, found {len(rows)}.")
    return rows[0]


def load_curve(path: Path) -> dict[str, np.ndarray]:
    rows = read_csv_rows(path)
    if not rows:
        raise RuntimeError(f"No rows in trajectory CSV: {path}")
    columns: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            columns.setdefault(key, []).append(parse_float(value))
    return {key: np.asarray(values, dtype=float) for key, values in columns.items()}


def curve_loss(target_curve: dict[str, np.ndarray], candidate_curve: dict[str, np.ndarray]) -> float:
    if "time_s" not in target_curve or "time_s" not in candidate_curve:
        return float("nan")
    target_time = target_curve["time_s"]
    candidate_time = candidate_curve["time_s"]
    start_time = max(float(np.nanmin(target_time)), float(np.nanmin(candidate_time)))
    end_time = min(float(np.nanmax(target_time)), float(np.nanmax(candidate_time)))
    evaluation_time = target_time[(target_time >= start_time) & (target_time <= end_time)]
    if evaluation_time.size == 0:
        return float("nan")

    residuals = []
    for field, scale in CURVE_FIELDS.items():
        if field not in target_curve or field not in candidate_curve:
            continue
        target_values = np.interp(evaluation_time, target_time, target_curve[field])
        candidate_values = np.interp(evaluation_time, candidate_time, candidate_curve[field])
        residuals.append((candidate_values - target_values) / scale)
    if not residuals:
        return float("nan")
    residual = np.concatenate(residuals)
    return float(np.sqrt(np.nanmean(residual * residual)))


def metric_loss(target_summary: dict[str, str], candidate_summary: dict[str, str]) -> float:
    residuals = []
    for field, scale in METRIC_FIELDS.items():
        target_value = parse_float(target_summary.get(field))
        candidate_value = parse_float(candidate_summary.get(field))
        if not math.isfinite(target_value) or not math.isfinite(candidate_value):
            continue
        residuals.append((candidate_value - target_value) / max(abs(target_value), scale))
    if not residuals:
        return float("nan")
    return float(math.sqrt(sum(value * value for value in residuals) / len(residuals)))


def combined_score(curve_score: float, metric_score: float) -> float:
    if math.isfinite(curve_score) and math.isfinite(metric_score):
        return 0.6 * curve_score + 0.4 * metric_score
    if math.isfinite(curve_score):
        return curve_score
    if math.isfinite(metric_score):
        return metric_score
    return float("inf")


def build_result_row(
    candidate: Candidate,
    run_dir: Path,
    summary_path: Path,
    trajectory_path: Path,
    target_summary: dict[str, str],
    target_curve: dict[str, np.ndarray],
) -> dict[str, Any]:
    candidate_summary = read_one_row(summary_path)
    candidate_curve = load_curve(trajectory_path)
    curve_score = curve_loss(target_curve, candidate_curve)
    metric_score = metric_loss(target_summary, candidate_summary)
    score = combined_score(curve_score, metric_score)
    return {
        "label": candidate_label(candidate),
        "score": finite_or_nan(score),
        "curve_loss": finite_or_nan(curve_score),
        "metric_loss": finite_or_nan(metric_score),
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "trajectory_path": str(trajectory_path),
        "youngs_modulus": candidate.youngs_modulus,
        "trampoline_mass": candidate.trampoline_mass,
        "elasticity_damping": candidate.elasticity_damping,
        "damping_scale": candidate.damping_scale,
        "dynamic_friction": candidate.dynamic_friction,
        "poissons_ratio": candidate.poissons_ratio,
        "target_max_compression_m": target_summary.get("max_compression_m", ""),
        "candidate_max_compression_m": candidate_summary.get("max_compression_m", ""),
        "target_first_min_ball_z_m": target_summary.get("first_min_ball_z_m", ""),
        "candidate_first_min_ball_z_m": candidate_summary.get("first_min_ball_z_m", ""),
        "target_rebound_height_m": target_summary.get("rebound_height_m", ""),
        "candidate_rebound_height_m": candidate_summary.get("rebound_height_m", ""),
        "target_first_apex_time_s": target_summary.get("first_apex_time_s", ""),
        "candidate_first_apex_time_s": candidate_summary.get("first_apex_time_s", ""),
        "target_damping_ratio": target_summary.get("damping_ratio", ""),
        "candidate_damping_ratio": candidate_summary.get("damping_ratio", ""),
        "candidate_final_state": candidate_summary.get("final_state", ""),
    }


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.artifact_root.expanduser().resolve() / f"{timestamp}__phase2_material_alignment"
    if not args.dry_run:
        run_root.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates(args)
    target_paths = resolve_mujoco_target(args, run_root)
    if args.dry_run:
        for candidate in candidates:
            run_isaaclab_candidate(args, run_root, candidate)
        print(f"DRY_RUN candidates={len(candidates)} run_root={run_root}", flush=True)
        return

    target_summary = read_one_row(target_paths.summary)
    target_curve = load_curve(target_paths.trajectory)
    rows = []
    for candidate in candidates:
        run_dir, summary_path, trajectory_path = run_isaaclab_candidate(args, run_root, candidate)
        rows.append(build_result_row(candidate, run_dir, summary_path, trajectory_path, target_summary, target_curve))

    rows.sort(key=lambda row: parse_float(row["score"]))
    results_path = run_root / "phase2_alignment_results.csv"
    write_csv(results_path, rows)

    print(f"WROTE {results_path}", flush=True)
    print(f"RUN_DIR {run_root}", flush=True)
    for row in rows[: min(5, len(rows))]:
        print(
            "TOP "
            f"score={row['score']} label={row['label']} "
            f"E={row['youngs_modulus']} mass={row['trampoline_mass']} "
            f"ed={row['elasticity_damping']} ds={row['damping_scale']} "
            f"df={row['dynamic_friction']} nu={row['poissons_ratio']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
