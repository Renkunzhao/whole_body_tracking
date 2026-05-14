"""Analyze rebounce play CSV logs produced by ``scripts/rsl_rl/play-rebounce.py``."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONTACT_FORCE_BACKENDS = ("gpu", "pinocchio")
CONTACT_FORCE_FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
CONTACT_FORCE_AXES = ("x", "y", "z")
DEADBAND_PHASES = ("air", "flight_up", "flight_down", "apex_band", "apex_pulse", "ground", "all")
JOINT_GROUPS = ("hip", "thigh", "calf")


def _flight_mask(df: pd.DataFrame) -> pd.Series:
    if "in_flight" in df.columns:
        return df["in_flight"].astype(bool)
    return df["is_air"].astype(bool)


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
    is_air = _flight_mask(df)
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


def _finite_abs(values: np.ndarray) -> np.ndarray:
    values = np.abs(np.asarray(values, dtype=float).reshape(-1))
    return values[np.isfinite(values)]


def _group_abs_values(df: pd.DataFrame, joint_cols: list[str], mask: pd.Series, group: str) -> np.ndarray:
    cols = [col for col in joint_cols if _joint_group(_joint_name(col)) == group]
    if not cols:
        return np.empty(0)
    return _finite_abs(df.loc[mask, cols].to_numpy())


def _deadbands_by_group(
    df: pd.DataFrame,
    joint_cols: list[str],
    phase_masks: dict[str, pd.Series],
    preferred_phase: str,
) -> list[dict[str, float | int | str]]:
    rows = []
    for group in JOINT_GROUPS:
        source_phase = preferred_phase
        values = _group_abs_values(df, joint_cols, phase_masks[preferred_phase], group)
        if values.size < 12 and preferred_phase != "air":
            source_phase = "air"
            values = _group_abs_values(df, joint_cols, phase_masks[source_phase], group)
        if values.size == 0:
            continue
        p80, p85, p90 = np.percentile(values, [80, 85, 90])
        rows.append(
            {
                "group": group,
                "source_phase": source_phase,
                "samples": int(values.size),
                "p80": float(p80),
                "p85": float(p85),
                "p90": float(p90),
            }
        )
    return rows


def _signed_finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    return values


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


def _shade_flight(ax, time_s: np.ndarray, in_flight: np.ndarray) -> None:
    for start, end in _mask_segments(time_s, in_flight):
        ax.axvspan(start, end, color="0.92", linewidth=0)


def _plot_robot_trampoline_vertical_state(df: pd.DataFrame, output_path: Path) -> None:
    time_s = df["sim_time_s"].to_numpy()
    in_flight = _flight_mask(df).to_numpy()
    has_trampoline_center = {"trampoline_center/z", "trampoline_center/vz"}.issubset(df.columns)

    fig, axes = plt.subplots(2, 1, figsize=(14, 5.5), sharex=True)
    pos_ax, vel_ax = axes
    for ax in axes:
        _shade_flight(ax, time_s, in_flight)
        ax.grid(True, alpha=0.25)

    pos_ax.plot(time_s, df["root_z"].to_numpy(), color="#1f77b4", linewidth=1.0, label="robot root z")
    vel_ax.plot(time_s, df["root_vz"].to_numpy(), color="#1f77b4", linewidth=1.0, label="robot root vz")
    if has_trampoline_center:
        pos_ax.plot(
            time_s,
            df["trampoline_center/z"].to_numpy(),
            color="#d62728",
            linewidth=1.0,
            label="trampoline center z",
        )
        vel_ax.plot(
            time_s,
            df["trampoline_center/vz"].to_numpy(),
            color="#d62728",
            linewidth=1.0,
            label="trampoline center vz",
        )
    if "trampoline/compression" in df.columns:
        pos_ax.plot(
            time_s,
            df["trampoline/compression"].to_numpy(),
            color="#2ca02c",
            linestyle="--",
            linewidth=1.0,
            label="trampoline compression",
        )

    vel_ax.axhline(0.0, color="0.3", linewidth=0.7, alpha=0.7)
    vel_ax.axhline(0.5, color="0.4", linestyle="--", linewidth=0.8)
    vel_ax.axhline(-0.5, color="0.4", linestyle="--", linewidth=0.8)
    pos_ax.plot([], [], color="0.7", linewidth=6, label="flight")
    pos_ax.set_ylabel("z / compression [m]")
    vel_ax.set_ylabel("vz [m/s]")
    vel_ax.set_xlabel("time [s]")
    pos_ax.legend(loc="upper right", fontsize=8)
    vel_ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Robot root and trampoline center vertical state. Grey spans are DOB flight state.", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_joint_timeseries(
    df: pd.DataFrame,
    joint_cols: list[str],
    deadbands: list[dict[str, float | int | str]],
    output_path: Path,
    vertical_output_path: Path,
) -> None:
    time_s = df["sim_time_s"].to_numpy()
    in_flight = _flight_mask(df).to_numpy()
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
    axes = axes.reshape(-1)
    band_by_group = {
        str(row["group"]): float(row["p85"])
        for row in deadbands
        if np.isfinite(float(row["p85"]))
    }
    ylim_by_group = {}
    for group in sorted({_joint_group(_joint_name(col)) for col in joint_cols}):
        cols = [col for col in joint_cols if _joint_group(_joint_name(col)) == group]
        values = _signed_finite(df[cols].to_numpy())
        max_abs = float(np.max(np.abs(values))) if values.size else 1.0
        if group in band_by_group:
            max_abs = max(max_abs, band_by_group[group])
        max_abs = max(max_abs * 1.05, 1.0)
        ylim_by_group[group] = (-max_abs, max_abs)

    for ax, col in zip(axes, joint_cols):
        joint = _joint_name(col)
        group = _joint_group(joint)
        _shade_flight(ax, time_s, in_flight)
        ax.plot(time_s, df[col].to_numpy(), color="#1f77b4", linewidth=1.0)
        if group in band_by_group:
            band = band_by_group[group]
            ax.axhline(band, color="#d62728", linestyle="--", linewidth=0.8)
            ax.axhline(-band, color="#d62728", linestyle="--", linewidth=0.8)
        ax.set_title(joint, fontsize=9)
        if group in ylim_by_group:
            ax.set_ylim(*ylim_by_group[group])
        ax.grid(True, alpha=0.25)
    axes[0].plot([], [], color="0.7", linewidth=6, label="flight")
    axes[0].plot([], [], color="#d62728", linestyle="--", label="p85 deadband")
    axes[0].legend(loc="upper right", fontsize=8)
    for ax in axes[-4:]:
        ax.set_xlabel("time [s]")
    for ax in axes[::4]:
        ax.set_ylabel("joint vel [rad/s]")
    fig.suptitle("Rebounce joint velocities. Grey spans are DOB flight state.", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    _plot_robot_trampoline_vertical_state(df, vertical_output_path)


def _plot_group_histograms(
    df: pd.DataFrame,
    joint_cols: list[str],
    phase_masks: dict[str, pd.Series],
    deadbands: list[dict[str, float | int | str]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    phases = ("air", "ground", "apex_band")
    colors = {"air": "#1f77b4", "ground": "#ff7f0e", "apex_band": "#2ca02c"}
    band_by_group = {str(row["group"]): float(row["p85"]) for row in deadbands}
    for ax, group in zip(axes, JOINT_GROUPS):
        for phase in phases:
            values = _group_abs_values(df, joint_cols, phase_masks[phase], group)
            if values.size:
                ax.hist(values, bins=30, density=True, histtype="step", linewidth=1.5, color=colors[phase], label=phase)
        if group in band_by_group:
            ax.axvline(band_by_group[group], color="#d62728", linestyle="--")
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
    in_flight = _flight_mask(df).to_numpy()
    colors = {"gpu": "#1f77b4", "pinocchio": "#d62728"}
    ylims = {axis: _contact_force_ylim(df, backends, axis) for axis in CONTACT_FORCE_AXES}

    fig, axes = plt.subplots(3, 4, figsize=(18, 9), sharex=True)
    for row, axis in enumerate(CONTACT_FORCE_AXES):
        for col, foot_name in enumerate(CONTACT_FORCE_FEET):
            ax = axes[row, col]
            _shade_flight(ax, time_s, in_flight)
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

    axes[0, 0].plot([], [], color="0.7", linewidth=6, label="flight")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")
    for row, axis in enumerate(CONTACT_FORCE_AXES):
        axes[row, 0].set_ylabel(f"force {axis} [N]")
    fig.suptitle("DOB foot contact force comparison. Grey spans are DOB flight state.", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def analyze_rebounce_play(
    input_path: str | os.PathLike,
    output_dir: str | os.PathLike | None = None,
    *,
    vz_threshold: float = 0.5,
    deadband_phase: str = "flight_up",
) -> list[Path]:
    """Generate rebounce play plots and print phase-split velocity statistics."""
    if deadband_phase not in DEADBAND_PHASES:
        raise ValueError(f"deadband_phase must be one of {DEADBAND_PHASES}, got {deadband_phase!r}.")

    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    joint_cols = [col for col in df.columns if col.startswith("joint_vel/")]
    if not joint_cols:
        raise RuntimeError(f"No joint_vel/* columns found in {input_path}.")

    phase_masks = _phase_masks(df, vz_threshold)
    deadbands = _deadbands_by_group(df, joint_cols, phase_masks, preferred_phase=deadband_phase)

    time_plot_path = output_dir / "rebounce_play_joint_velocity_timeseries.png"
    hist_plot_path = output_dir / "rebounce_play_joint_velocity_histograms.png"
    vertical_plot_path = output_dir / "rebounce_play_vertical_state.png"
    force_plot_path = output_dir / "rebounce_play_contact_force_timeseries.png"
    _plot_joint_timeseries(df, joint_cols, deadbands, time_plot_path, vertical_plot_path)
    _plot_group_histograms(df, joint_cols, phase_masks, deadbands, hist_plot_path)
    wrote_force_plot = _plot_contact_force_timeseries(df, force_plot_path)

    print(f"Input: {input_path}")
    print(f"Samples: {len(df)}")
    print("Phase sample counts:")
    for phase, mask in phase_masks.items():
        print(f"  {phase:12s}: {int(mask.sum())}")
    print(f"\nSuggested joint-velocity deadbands from {deadband_phase} |dq| percentiles:")
    for row in deadbands:
        print(
            f"  {str(row['group']):6s}: p80={float(row['p80']):.2f}, "
            f"p85={float(row['p85']):.2f}, p90={float(row['p90']):.2f} rad/s "
            f"(source={row['source_phase']}, samples={int(row['samples'])})"
        )
    print("\nWrote:")
    print(f"  {time_plot_path}")
    print(f"  {hist_plot_path}")
    print(f"  {vertical_plot_path}")
    if wrote_force_plot:
        print(f"  {force_plot_path}")
    else:
        print("  contact force plot skipped: no contact_force/{gpu,pinocchio} foot columns found")
    written_paths = [time_plot_path, hist_plot_path, vertical_plot_path]
    if wrote_force_plot:
        written_paths.append(force_plot_path)
    return written_paths
