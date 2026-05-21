from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CONDITION_SCRIPT = SCRIPT_DIR / "isaaclab_trampoline_phase1_condition.py"
DEFAULT_ARTIFACT_ROOT = Path("logs/isaaclab_trampoline_phase1_runs")

PHASE1_CONDITIONS: list[tuple[str, list[str]]] = [
    ("nominal", []),
    ("resolution_8", ["--sim_resolution", "8"]),
    ("resolution_20", ["--sim_resolution", "20"]),
    ("thickness_0.03", ["--thickness", "0.03"]),
    ("thickness_0.1", ["--thickness", "0.1"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the IsaacLab trampoline Phase 1 structure sweep.")
    parser.add_argument("--artifact_root", type=Path, default=DEFAULT_ARTIFACT_ROOT, help="Root directory for auto-named per-run artifacts.")
    parser.add_argument("--sim_time", type=float, default=4.0, help="Simulation duration passed to each condition run.")
    parser.add_argument("--sim_dt", type=float, default=0.002, help="Simulation timestep passed to each condition run.")
    parser.add_argument("--ball_height", type=float, default=1.0, help="Ball height passed to each condition run.")
    parser.add_argument("--headless", action="store_true", default=True, help="Launch Isaac Sim headless.")
    parser.add_argument("--video", action="store_true", default=True, help="Record video for each condition.")
    return parser.parse_args()


def run_condition(args: argparse.Namespace, label: str, extra_args: list[str], artifact_root: Path) -> None:
    cmd = [
        sys.executable,
        str(CONDITION_SCRIPT),
        "--label",
        label,
        "--artifact_root",
        str(artifact_root),
        "--sim_time",
        str(args.sim_time),
        "--sim_dt",
        str(args.sim_dt),
        "--ball_height",
        str(args.ball_height),
    ]
    if args.headless:
        cmd.append("--headless")
    if args.video:
        cmd.append("--video")
    cmd.extend(extra_args)
    print(f"[INFO] Running {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = args.artifact_root.expanduser().resolve() / f"{timestamp}__phase1_sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    for label, extra_args in PHASE1_CONDITIONS:
        run_condition(args, label, extra_args, sweep_root)

    print(f"WROTE {sweep_root}", flush=True)


if __name__ == "__main__":
    main()
