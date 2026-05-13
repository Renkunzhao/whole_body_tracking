"""Analyze rebounce play joint-velocity CSV logs.

The input is produced by ``scripts/rsl_rl/play-rebounce.py``.  It writes
phase-split velocity statistics and plots useful for selecting a
joint-velocity deadband.  If the CSV also contains DOB contact-force columns,
it writes a 3x4 foot-force comparison plot for GPU and Pinocchio estimates.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = "scripts/logs/rebounce_joint_vel.csv"
EXCEED_THRESHOLDS = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
PERCENTILES = (50, 75, 80, 85, 90, 95, 99)
CONTACT_FORCE_BACKENDS = ("gpu", "pinocchio")
CONTACT_FORCE_FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
CONTACT_FORCE_AXES = ("x", "y", "z")


def _joint_name(column: str) -> str:
    return column.removeprefix("joint_vel/")


def _joint_group(joint_name: str) -> str:
    if "_hip_joint" in joint_name:
        return "hip"
    if "_thigh_joint" in joint_name:
        return "thigh"
    if "_calf_joint" in joint_name:
        return "calf"
    return "other"


def _phase_masks(df: pd.DataFrame, vz_threshold: float) -> dict[str, pd.Series]:
    is_air = df["is_air"].astype(bool)
    is_apex = df["is_apex"].astype(bool)
    root_vz = df["root_vz"].astype(float)
    return {
        "all": pd.Series(True, index=df.index),
        "air": is_air,
        "ground": ~is_air,
        "apex_band": is_air & (root_vz.abs() < vz_threshold),
        "flight_up": is_air & (root_vz > vz_threshold),
        "flight_down": is_air & (root_vz < -vz_threshold),
        "apex_pulse": is_apex,
    }


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        row = {
            "count": 0,
            "mean_abs": np.nan,
            "rms": np.nan,
            "max_abs": np.nan,
        }
        row.update({f"p{p}_abs": np.nan for p in PERCENTILES})
        row.update({f"frac_abs_gt_{threshold:g}": np.nan for threshold in EXCEED_THRESHOLDS})
        return row

    abs_values = np.abs(values)
    row = {
        "count": int(values.size),
        "mean_abs": float(abs_values.mean()),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "max_abs": float(abs_values.max()),
    }
    row.update({f"p{p}_abs": float(np.percentile(abs_values, p)) for p in PERCENTILES})
    row.update(
        {f"frac_abs_gt_{threshold:g}": float(np.mean(abs_values > threshold)) for threshold in EXCEED_THRESHOLDS}
    )
    return row


def _build_joint_summary(df: pd.DataFrame, joint_cols: list[str], phase_masks: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for phase, mask in phase_masks.items():
        phase_df = df.loc[mask, joint_cols]
        for col in joint_cols:
            joint = _joint_name(col)
            rows.append(
                {
                    "phase": phase,
                    "joint": joint,
                    "group": _joint_group(joint),
                    **_stats(phase_df[col].to_numpy()),
                }
            )
    return pd.DataFrame(rows)


def _build_group_summary(df: pd.DataFrame, joint_cols: list[str], phase_masks: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    groups = sorted({_joint_group(_joint_name(col)) for col in joint_cols})
    for phase, mask in phase_masks.items():
        for group in groups:
            cols = [col for col in joint_cols if _joint_group(_joint_name(col)) == group]
            values = df.loc[mask, cols].to_numpy().reshape(-1)
            rows.append({"phase": phase, "group": group, **_stats(values)})
    return pd.DataFrame(rows)


def _build_deadband_recommendation(group_summary: pd.DataFrame, preferred_phase: str) -> pd.DataFrame:
    rows = []
    for group in ("hip", "thigh", "calf"):
        phase_row = group_summary[(group_summary["phase"] == preferred_phase) & (group_summary["group"] == group)]
        source_phase = preferred_phase
        if phase_row.empty or int(phase_row.iloc[0]["count"]) < 12:
            phase_row = group_summary[(group_summary["phase"] == "air") & (group_summary["group"] == group)]
            source_phase = "air"
        if phase_row.empty:
            continue
        row = phase_row.iloc[0]
        rows.append(
            {
                "group": group,
                "source_phase": source_phase,
                "samples": int(row["count"]),
                "gentle_deadband_p90": row["p90_abs"],
                "recommended_deadband_p85": row["p85_abs"],
                "stricter_deadband_p80": row["p80_abs"],
                "current_frac_above_recommended": 0.15,
            }
        )
    return pd.DataFrame(rows)


def _mask_segments(time_s: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    segments = []
    start = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = time_s[index]
        if start is not None and (not active or index == len(mask) - 1):
            end_index = index if not active else index
            segments.append((start, time_s[end_index]))
            start = None
    return segments


def _shade_air(ax, time_s: np.ndarray, is_air: np.ndarray) -> None:
    for start, end in _mask_segments(time_s, is_air):
        ax.axvspan(start, end, color="0.92", linewidth=0)


def _plot_joint_timeseries(
    df: pd.DataFrame,
    joint_cols: list[str],
    deadbands: pd.DataFrame,
    output_path: Path,
) -> None:
    time_s = df["sim_time_s"].to_numpy()
    is_air = df["is_air"].astype(bool).to_numpy()
    root_vz = df["root_vz"].to_numpy()
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
    axes = axes.reshape(-1)
    band_by_group = {
        row["group"]: float(row["recommended_deadband_p85"])
        for _, row in deadbands.iterrows()
        if np.isfinite(row["recommended_deadband_p85"])
    }
    ylim_by_group = {}
    for group in sorted({_joint_group(_joint_name(col)) for col in joint_cols}):
        cols = [col for col in joint_cols if _joint_group(_joint_name(col)) == group]
        values = df[cols].to_numpy().reshape(-1)
        values = values[np.isfinite(values)]
        max_abs = float(np.max(np.abs(values))) if values.size else 1.0
        if group in band_by_group:
            max_abs = max(max_abs, band_by_group[group])
        max_abs = max(max_abs * 1.05, 1.0)
        ylim_by_group[group] = (-max_abs, max_abs)

    for ax, col in zip(axes, joint_cols):
        joint = _joint_name(col)
        group = _joint_group(joint)
        _shade_air(ax, time_s, is_air)
        ax.plot(time_s, df[col].to_numpy(), color="#1f77b4", linewidth=1.0)
        if group in band_by_group:
            band = band_by_group[group]
            ax.axhline(band, color="#d62728", linestyle="--", linewidth=0.8)
            ax.axhline(-band, color="#d62728", linestyle="--", linewidth=0.8)
        ax.set_title(joint, fontsize=9)
        if group in ylim_by_group:
            ax.set_ylim(*ylim_by_group[group])
        ax.grid(True, alpha=0.25)
    axes[0].plot([], [], color="0.7", linewidth=6, label="air")
    axes[0].plot([], [], color="#d62728", linestyle="--", label="p85 deadband")
    axes[0].legend(loc="upper right", fontsize=8)
    for ax in axes[-4:]:
        ax.set_xlabel("time [s]")
    for ax in axes[::4]:
        ax.set_ylabel("joint vel [rad/s]")
    fig.suptitle("Rebounce joint velocities. Grey spans are air by foot-clearance gate.", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(time_s, root_vz, color="#2ca02c", linewidth=1.0)
    ax.axhline(0.5, color="0.4", linestyle="--", linewidth=0.8)
    ax.axhline(-0.5, color="0.4", linestyle="--", linewidth=0.8)
    _shade_air(ax, time_s, is_air)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("root vz [m/s]")
    ax.set_title("Root vertical velocity with air spans")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path.with_name(output_path.stem + "_root_vz.png"), dpi=160)
    plt.close(fig)


def _plot_group_histograms(
    df: pd.DataFrame,
    joint_cols: list[str],
    phase_masks: dict[str, pd.Series],
    deadbands: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    phases = ("air", "ground", "apex_band")
    colors = {"air": "#1f77b4", "ground": "#ff7f0e", "apex_band": "#2ca02c"}
    for ax, group in zip(axes, ("hip", "thigh", "calf")):
        cols = [col for col in joint_cols if _joint_group(_joint_name(col)) == group]
        for phase in phases:
            values = np.abs(df.loc[phase_masks[phase], cols].to_numpy().reshape(-1))
            values = values[np.isfinite(values)]
            if values.size:
                ax.hist(values, bins=30, density=True, histtype="step", linewidth=1.5, color=colors[phase], label=phase)
        recommendation = deadbands[deadbands["group"] == group]
        if not recommendation.empty:
            ax.axvline(float(recommendation.iloc[0]["recommended_deadband_p85"]), color="#d62728", linestyle="--")
        ax.set_title(f"{group} |dq| distribution")
        ax.set_xlabel("|joint vel| [rad/s]")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _contact_force_col(backend: str, foot_name: str, axis: str) -> str:
    return f"contact_force/{backend}/{foot_name}/{axis}"


def _contact_force_valid_col(backend: str) -> str:
    return f"contact_force/{backend}/valid"


def _available_contact_force_backends(df: pd.DataFrame) -> list[str]:
    backends = []
    for backend in CONTACT_FORCE_BACKENDS:
        required_cols = [
            _contact_force_col(backend, foot_name, axis)
            for foot_name in CONTACT_FORCE_FEET
            for axis in CONTACT_FORCE_AXES
        ]
        if all(col in df.columns for col in required_cols):
            backends.append(backend)
    return backends


def _force_series(df: pd.DataFrame, backend: str, foot_name: str, axis: str) -> np.ndarray:
    values = df[_contact_force_col(backend, foot_name, axis)].to_numpy(dtype=float)
    valid_col = _contact_force_valid_col(backend)
    if valid_col in df.columns:
        valid = df[valid_col].astype(bool).to_numpy()
        values = values.copy()
        values[~valid] = np.nan
    return values


def _contact_force_ylim(df: pd.DataFrame, backends: list[str], axis: str) -> tuple[float, float]:
    values = []
    for backend in backends:
        for foot_name in CONTACT_FORCE_FEET:
            force = _force_series(df, backend, foot_name, axis)
            values.append(force[np.isfinite(force)])
    values = [value for value in values if value.size]
    if not values:
        return (-1.0, 1.0)

    merged = np.concatenate(values)
    if axis in ("x", "y"):
        max_abs = max(float(np.max(np.abs(merged))) * 1.05, 1.0)
        return (-max_abs, max_abs)

    low = min(float(np.min(merged)) * 1.05, 0.0)
    high = max(float(np.max(merged)) * 1.05, 1.0)
    if np.isclose(low, high):
        high = low + 1.0
    return (low, high)


def _plot_contact_force_timeseries(df: pd.DataFrame, output_path: Path) -> bool:
    backends = _available_contact_force_backends(df)
    if not backends:
        return False

    time_s = df["sim_time_s"].to_numpy()
    is_air = df["is_air"].astype(bool).to_numpy()
    colors = {"gpu": "#1f77b4", "pinocchio": "#d62728"}
    ylims = {axis: _contact_force_ylim(df, backends, axis) for axis in CONTACT_FORCE_AXES}

    fig, axes = plt.subplots(3, 4, figsize=(18, 9), sharex=True)
    for row, axis in enumerate(CONTACT_FORCE_AXES):
        for col, foot_name in enumerate(CONTACT_FORCE_FEET):
            ax = axes[row, col]
            _shade_air(ax, time_s, is_air)
            for backend in backends:
                ax.plot(
                    time_s,
                    _force_series(df, backend, foot_name, axis),
                    color=colors.get(backend, None),
                    linewidth=0.9,
                    label=backend,
                )
            ax.axhline(0.0, color="0.3", linewidth=0.6, alpha=0.6)
            ax.set_ylim(*ylims[axis])
            ax.set_title(f"{foot_name} force {axis}", fontsize=9)
            ax.grid(True, alpha=0.25)

    axes[0, 0].plot([], [], color="0.7", linewidth=6, label="air")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")
    for row, axis in enumerate(CONTACT_FORCE_AXES):
        axes[row, 0].set_ylabel(f"force {axis} [N]")
    fig.suptitle("DOB foot contact force comparison. Grey spans are air by foot-clearance gate.", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze rebounce joint-velocity CSV logs.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="CSV generated by play-rebounce.py.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for plots and summary CSVs.")
    parser.add_argument("--vz_threshold", type=float, default=0.5, help="Apex-band threshold: is_air and |root_vz| < this.")
    parser.add_argument(
        "--deadband_phase",
        type=str,
        default="flight_up",
        choices=("air", "flight_up", "flight_down", "apex_band", "apex_pulse", "ground", "all"),
        help="Phase used for the deadband recommendation.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    joint_cols = [col for col in df.columns if col.startswith("joint_vel/")]
    if not joint_cols:
        raise RuntimeError(f"No joint_vel/* columns found in {input_path}.")

    phase_masks = _phase_masks(df, args.vz_threshold)
    joint_summary = _build_joint_summary(df, joint_cols, phase_masks)
    group_summary = _build_group_summary(df, joint_cols, phase_masks)
    deadbands = _build_deadband_recommendation(group_summary, preferred_phase=args.deadband_phase)

    joint_summary_path = output_dir / "rebounce_joint_vel_summary_by_joint.csv"
    group_summary_path = output_dir / "rebounce_joint_vel_summary_by_group.csv"
    deadband_path = output_dir / "rebounce_joint_vel_deadband_recommendation.csv"
    joint_summary.to_csv(joint_summary_path, index=False, float_format="%.6f")
    group_summary.to_csv(group_summary_path, index=False, float_format="%.6f")
    deadbands.to_csv(deadband_path, index=False, float_format="%.6f")

    time_plot_path = output_dir / "rebounce_joint_vel_timeseries.png"
    hist_plot_path = output_dir / "rebounce_joint_vel_histograms.png"
    force_plot_path = output_dir / "rebounce_contact_force_timeseries.png"
    _plot_joint_timeseries(df, joint_cols, deadbands, time_plot_path)
    _plot_group_histograms(df, joint_cols, phase_masks, deadbands, hist_plot_path)
    wrote_force_plot = _plot_contact_force_timeseries(df, force_plot_path)

    print(f"Input: {input_path}")
    print(f"Samples: {len(df)}")
    print("Phase sample counts:")
    for phase, mask in phase_masks.items():
        print(f"  {phase:12s}: {int(mask.sum())}")
    print(f"\nSuggested joint-velocity deadbands from {args.deadband_phase} |dq| percentiles:")
    for _, row in deadbands.iterrows():
        print(
            f"  {row['group']:6s}: p80={row['stricter_deadband_p80']:.2f}, "
            f"p85={row['recommended_deadband_p85']:.2f}, p90={row['gentle_deadband_p90']:.2f} rad/s "
            f"(source={row['source_phase']}, samples={int(row['samples'])})"
        )
    print("\nWrote:")
    print(f"  {joint_summary_path}")
    print(f"  {group_summary_path}")
    print(f"  {deadband_path}")
    print(f"  {time_plot_path}")
    print(f"  {hist_plot_path}")
    print(f"  {time_plot_path.with_name(time_plot_path.stem + '_root_vz.png')}")
    if wrote_force_plot:
        print(f"  {force_plot_path}")
    else:
        print("  contact force plot skipped: no contact_force/{gpu,pinocchio} foot columns found")


if __name__ == "__main__":
    main()
