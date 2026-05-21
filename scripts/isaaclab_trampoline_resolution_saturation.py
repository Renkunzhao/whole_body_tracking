from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
CONDITION_SCRIPT = SCRIPT_DIR / "isaaclab_trampoline_phase1_condition.py"
DEFAULT_ARTIFACT_ROOT = Path("logs/isaaclab_trampoline_resolution_saturation_runs")

G1_TOTAL_MASS_KG = 33.341142022
DEFAULT_BALL_MASS = 0.5 * G1_TOTAL_MASS_KG
DEFAULT_BALL_HEIGHT = 1.0
DEFAULT_RESOLUTIONS = (8, 10, 12, 15, 18, 20, 24)
DEFAULT_STABILITY_THRESHOLD = 0.05

# Keep these in sync with go2_rebounce_env_cfg.py without importing IsaacLab task packages here.
TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE = (8.0e5, 8.0e6)
TRAMPOLINE_DR_MASS_RANGE = (5.0, 15.0)
TRAMPOLINE_DR_ELASTICITY_DAMPING_RANGE = (0.01, 0.1)
TRAMPOLINE_DR_DAMPING_SCALE_RANGE = (1.0, 1.0)
THICKNESS_VALUES = (0.03, 0.1)

DYNAMIC_METRICS = (
    "max_compression_m",
    "contact_duration_s",
    "release_vz_mps",
    "rebound_height_m",
)


@dataclass(frozen=True)
class ParameterGroup:
    label: str
    thickness: float
    trampoline_mass: float
    youngs_modulus: float
    elasticity_damping: float
    damping_scale: float


def parse_float(value: str) -> float:
    if value.lower() == "nan":
        return float("nan")
    return float(value)


def relative_change(current: float, previous: float) -> float:
    if not math.isfinite(current) or not math.isfinite(previous):
        return float("nan")
    denominator = max(abs(previous), 1.0e-9)
    return abs(current - previous) / denominator


def build_parameter_groups() -> list[ParameterGroup]:
    thickness_groups = (
        ("thin", min(THICKNESS_VALUES)),
        ("thick", max(THICKNESS_VALUES)),
    )
    material_groups = (
        ("soft", TRAMPOLINE_DR_MASS_RANGE[0], TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE[0]),
        ("stiff", TRAMPOLINE_DR_MASS_RANGE[1], TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE[1]),
    )
    damping_groups = (
        ("low_damp", TRAMPOLINE_DR_ELASTICITY_DAMPING_RANGE[0], TRAMPOLINE_DR_DAMPING_SCALE_RANGE[0]),
        ("high_damp", TRAMPOLINE_DR_ELASTICITY_DAMPING_RANGE[1], TRAMPOLINE_DR_DAMPING_SCALE_RANGE[1]),
    )

    groups = []
    for thickness_label, thickness in thickness_groups:
        for material_label, trampoline_mass, youngs_modulus in material_groups:
            for damping_label, elasticity_damping, damping_scale in damping_groups:
                groups.append(
                    ParameterGroup(
                        label=f"{thickness_label}_{material_label}_{damping_label}",
                        thickness=thickness,
                        trampoline_mass=trampoline_mass,
                        youngs_modulus=youngs_modulus,
                        elasticity_damping=elasticity_damping,
                        damping_scale=damping_scale,
                    )
                )
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IsaacLab trampoline resolution saturation tests.")
    parser.add_argument("--artifact_root", type=Path, default=DEFAULT_ARTIFACT_ROOT, help="Root directory for saturation artifacts.")
    parser.add_argument("--sim_time", type=float, default=4.0, help="Simulation duration passed to each condition run.")
    parser.add_argument("--sim_dt", type=float, default=0.002, help="Simulation timestep passed to each condition run.")
    parser.add_argument("--ball_mass", type=float, default=DEFAULT_BALL_MASS, help="Ball mass in kilograms; default is half of G1 URDF mass.")
    parser.add_argument("--ball_height", type=float, default=DEFAULT_BALL_HEIGHT, help="Ball center height passed to each condition run.")
    parser.add_argument("--resolutions", type=int, nargs="+", default=list(DEFAULT_RESOLUTIONS), help="Resolution grid to test.")
    parser.add_argument("--stability_threshold", type=float, default=DEFAULT_STABILITY_THRESHOLD, help="Relative-change threshold for saturation detection.")
    parser.add_argument("--headless", action="store_true", default=True, help="Launch Isaac Sim headless.")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True, help="Record per-run videos.")
    return parser.parse_args()


def run_condition(args: argparse.Namespace, group: ParameterGroup, resolution: int, run_dir: Path) -> Path:
    label = f"{group.label}_res{resolution}"
    cmd = [
        sys.executable,
        str(CONDITION_SCRIPT),
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
        str(group.thickness),
        "--trampoline_mass",
        str(group.trampoline_mass),
        "--sim_resolution",
        str(resolution),
        "--youngs_modulus",
        str(group.youngs_modulus),
        "--elasticity_damping",
        str(group.elasticity_damping),
        "--damping_scale",
        str(group.damping_scale),
    ]
    if args.headless:
        cmd.append("--headless")
    cmd.append("--video" if args.video else "--no-video")

    print(f"[INFO] Running {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return run_dir / "phase1_summary.csv"


def read_summary_row(summary_path: Path) -> dict[str, Any]:
    with summary_path.open(newline="", encoding="utf-8") as summary_file:
        rows = list(csv.DictReader(summary_file))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in {summary_path}, found {len(rows)}")
    return rows[0]


def build_run_row(group: ParameterGroup, resolution: int, summary_path: Path) -> dict[str, Any]:
    row = read_summary_row(summary_path)
    row.update(
        {
            "group": group.label,
            "resolution": resolution,
            "summary_path": str(summary_path),
        }
    )
    return row


def finite_difference(current: float, previous: float) -> float:
    if not math.isfinite(current) or not math.isfinite(previous):
        return float("nan")
    return current - previous


def group_rows_by_resolution(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_group.setdefault(row["group"], []).append(row)
    for group_rows in rows_by_group.values():
        group_rows.sort(key=lambda row: int(row["resolution"]))
    return rows_by_group


def add_delta_columns(rows: list[dict[str, Any]]) -> None:
    for group_rows in group_rows_by_resolution(rows).values():
        previous = None
        for row in group_rows:
            for metric in DYNAMIC_METRICS:
                if previous is None:
                    row[f"delta_{metric}_vs_prev"] = float("nan")
                    row[f"relative_change_{metric}_vs_prev"] = float("nan")
                    continue
                current_value = parse_float(str(row[metric]))
                previous_value = parse_float(str(previous[metric]))
                row[f"delta_{metric}_vs_prev"] = finite_difference(current_value, previous_value)
                row[f"relative_change_{metric}_vs_prev"] = relative_change(current_value, previous_value)
            previous = row


def is_stable_step(row: dict[str, Any], threshold: float) -> bool:
    for metric in DYNAMIC_METRICS:
        change = parse_float(str(row[f"relative_change_{metric}_vs_prev"]))
        if not math.isfinite(change) or change > threshold:
            return False
    return True


def summarize_groups(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    summaries = []
    for group, group_rows in sorted(group_rows_by_resolution(rows).items()):
        # Start at index=2 to require two consecutive stable relative-change steps.
        stable_steps = [is_stable_step(row, threshold) for row in group_rows]
        recommended_resolution: Any = ""
        saturation_reason = "not_stable_in_grid"
        for index in range(2, len(group_rows)):
            if stable_steps[index - 1] and stable_steps[index]:
                recommended_resolution = group_rows[index - 1]["resolution"]
                saturation_reason = "two_consecutive_stable_steps"
                break

        if recommended_resolution == "" and not any(int(row["released"]) == 1 for row in group_rows):
            saturation_reason = "no_release_in_grid"

        last_row = group_rows[-1]
        summaries.append(
            {
                "group": group,
                "recommended_resolution": recommended_resolution,
                "saturation_reason": saturation_reason,
                "max_tested_resolution": last_row["resolution"],
                "last_released": last_row["released"],
                "last_max_compression_m": last_row["max_compression_m"],
                "last_contact_duration_s": last_row["contact_duration_s"],
                "last_release_vz_mps": last_row["release_vz_mps"],
                "last_rebound_height_m": last_row["rebound_height_m"],
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = args.artifact_root.expanduser().resolve() / f"{timestamp}__resolution_saturation"
    groups = build_parameter_groups()
    rows = []

    for group in groups:
        for resolution in args.resolutions:
            run_dir = sweep_root / group.label / f"resolution_{resolution}"
            summary_path = run_condition(args, group, resolution, run_dir)
            rows.append(build_run_row(group, resolution, summary_path))

    add_delta_columns(rows)
    group_summaries = summarize_groups(rows, args.stability_threshold)
    runs_path = sweep_root / "resolution_saturation_runs.csv"
    group_summary_path = sweep_root / "resolution_saturation_group_summary.csv"
    write_csv(runs_path, rows)
    write_csv(group_summary_path, group_summaries)

    unresolved_groups = [row["group"] for row in group_summaries if row["recommended_resolution"] == ""]
    if unresolved_groups:
        print(f"GLOBAL_RECOMMENDED_RESOLUTION unresolved groups={','.join(unresolved_groups)}", flush=True)
    else:
        recommended = [int(row["recommended_resolution"]) for row in group_summaries]
        print(f"GLOBAL_RECOMMENDED_RESOLUTION {max(recommended)}", flush=True)
    print(f"WROTE {runs_path}", flush=True)
    print(f"WROTE {group_summary_path}", flush=True)
    print(f"RUN_DIR {sweep_root}", flush=True)


if __name__ == "__main__":
    main()
