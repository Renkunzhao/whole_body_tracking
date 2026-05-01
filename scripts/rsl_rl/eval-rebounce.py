"""Headless evaluation for Go2 rebounce policies."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import math
import pathlib
import re
import sys
import time
from collections import Counter
from itertools import product

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a rebounce RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--wandb_path", type=str, required=True, help="Wandb run path (entity/project/run_id[/model]).")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel envs for headless evaluation.")
parser.add_argument("--episodes_per_condition", type=int, default=20, help="Episodes collected for each condition.")
parser.add_argument("--target_heights", type=float, nargs="+", default=[0.5, 0.65, 0.8])
parser.add_argument("--youngs_moduli", type=float, nargs="+", default=[4.0e4, 8.0e4, 1.6e5])
parser.add_argument("--masses", type=float, nargs="+", default=[10.0])
parser.add_argument(
    "--drop_heights",
    type=float,
    nargs="*",
    default=None,
    help="Optional drop heights. If omitted, each condition uses drop_height=target_height.",
)
parser.add_argument(
    "--csv_path",
    type=str,
    default=None,
    help="Optional path for aggregate CSV output. The wandb run id is appended to the file name.",
)
parser.add_argument("--print_episodes", action="store_true", help="Print every episode summary.")
parser.add_argument("--progress_interval", type=int, default=16, help="Print progress after this many episodes.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if hasattr(args_cli, "headless"):
    args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.task_utils import apply_play_overrides


def _mean(values):
    values = [value for value in values if not math.isnan(value)]
    return float("nan") if not values else sum(values) / len(values)


def _std(values):
    values = [value for value in values if not math.isnan(value)]
    if len(values) < 2:
        return 0.0 if len(values) == 1 else float("nan")
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _rmse(values):
    return float("nan") if not values else math.sqrt(sum(value * value for value in values) / len(values))


def _p90(values):
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, math.ceil(0.9 * len(sorted_values)) - 1)
    return sorted_values[index]


def _fmt(value: float, precision: int = 3):
    if math.isnan(value):
        return "nan"
    return f"{value:.{precision}f}"


def _wandb_run_id(wandb_path: str) -> str:
    parts = [part for part in wandb_path.strip("/").split("/") if part]
    if not parts:
        return "unknown"
    if len(parts) >= 2 and parts[-1].startswith("model"):
        return parts[-2]
    if len(parts) >= 3:
        return parts[2]
    return parts[-1]


def _sanitize_file_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return sanitized or "unknown"


def _csv_path_with_wandb_id(path: str, wandb_path: str) -> str:
    run_id = _sanitize_file_component(_wandb_run_id(wandb_path))
    if "{wandb_id}" in path:
        return path.replace("{wandb_id}", run_id)

    path_obj = pathlib.Path(path)
    if path_obj.suffix:
        return str(path_obj.with_name(f"{path_obj.stem}_{run_id}{path_obj.suffix}"))
    return str(path_obj / f"rebounce_eval_{run_id}.csv")


def _wrapped_angle_error(angle: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    delta = angle - reference
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def _get_rebounce_handles(env):
    command_manager = getattr(env.unwrapped, "command_manager", None)
    if command_manager is None or "hop" not in command_manager.active_terms:
        raise RuntimeError("This task has no 'hop' command.")
    hop_command = command_manager.get_term("hop")
    energy_command = command_manager.get_term("energy") if "energy" in command_manager.active_terms else None
    return hop_command, energy_command


def _get_done_reasons(env, env_id: int) -> list[str]:
    termination_manager = env.unwrapped.termination_manager
    reasons = []
    for name in termination_manager.active_terms:
        if bool(termination_manager.get_term(name)[env_id]):
            reasons.append(name)
    return reasons


def _set_condition(env, target_height: float, drop_height: float, youngs_modulus: float, mass: float):
    unwrapped = env.unwrapped
    hop_command = unwrapped.command_manager.get_term("hop")
    hop_command.cfg.ranges.peak_height = (target_height, target_height)
    hop_command.cfg.resampling_time_range = (1.0e9, 1.0e9)
    if hasattr(unwrapped.cfg.commands, "hop"):
        unwrapped.cfg.commands.hop.ranges.peak_height = (target_height, target_height)
        unwrapped.cfg.commands.hop.resampling_time_range = (1.0e9, 1.0e9)

    reset_cfg = unwrapped.event_manager.get_term_cfg("reset_drop")
    reset_cfg.params["drop_height_range"] = (drop_height, drop_height)
    if hasattr(unwrapped.cfg.events, "reset_drop"):
        unwrapped.cfg.events.reset_drop.params["drop_height_range"] = (drop_height, drop_height)

    try:
        trampoline_cfg = unwrapped.event_manager.get_term_cfg("randomize_trampoline_properties")
    except ValueError:
        return
    trampoline_cfg.params["youngs_modulus_range"] = (youngs_modulus, youngs_modulus)
    trampoline_cfg.params["youngs_modulus_distribution"] = "uniform"
    trampoline_cfg.params["mass_range"] = (mass, mass)
    if hasattr(unwrapped.cfg.events, "randomize_trampoline_properties"):
        params = unwrapped.cfg.events.randomize_trampoline_properties.params
        params["youngs_modulus_range"] = (youngs_modulus, youngs_modulus)
        params["youngs_modulus_distribution"] = "uniform"
        params["mass_range"] = (mass, mass)


class EpisodeStats:
    def __init__(self, env, env_id: int, energy_command):
        self.env_id = env_id
        self.energy_command = energy_command
        self.reset(env)

    def reset(self, env):
        robot = env.unwrapped.scene["robot"]
        i = self.env_id
        self.start_xy = robot.data.root_pos_w[i, :2].clone()
        self.start_yaw = euler_xyz_from_quat(robot.data.root_quat_w[i : i + 1])[2][0].clone()
        self.height_errors = []
        self.height_abs_errors = []
        self.height_ratios = []
        self.pos_work_target = []
        self.abs_work_target = []
        self.pos_work_height = []
        self.abs_work_height = []
        self.apex_count = 0
        self.height_matched_count = 0
        self.orientation_error_sum = 0.0
        self.yaw_error_sq_sum = 0.0
        self.step_count = 0
        self.last_xy_drift = 0.0
        self.last_yaw_drift = 0.0
        self.negative_work = 0.0
        self.absolute_work = 0.0

    def update_state(self, env):
        robot = env.unwrapped.scene["robot"]
        i = self.env_id
        root_xy = robot.data.root_pos_w[i, :2]
        yaw = euler_xyz_from_quat(robot.data.root_quat_w[i : i + 1])[2][0]
        yaw_error = _wrapped_angle_error(yaw, self.start_yaw)
        orientation_error = torch.sum(torch.square(robot.data.projected_gravity_b[i, :2]))

        self.last_xy_drift = float(torch.linalg.vector_norm(root_xy - self.start_xy))
        self.last_yaw_drift = abs(float(yaw_error))
        self.orientation_error_sum += float(orientation_error)
        self.yaw_error_sq_sum += float(yaw_error * yaw_error)
        self.step_count += 1

        if self.energy_command is not None:
            dt = env.unwrapped.step_dt
            self.negative_work += float(self.energy_command.negative_power[i]) * dt
            self.absolute_work += float(self.energy_command.absolute_power[i]) * dt

    def record_apex(self, hop_command):
        i = self.env_id
        self.apex_count += 1
        apex_height = float(hop_command.last_apex_height[i])
        target_height = float(hop_command.last_apex_target_height[i])
        error = apex_height - target_height
        abs_error = abs(error)
        self.height_errors.append(error)
        self.height_abs_errors.append(abs_error)
        self.height_ratios.append(apex_height / max(target_height, 1.0e-6))
        if abs_error < hop_command.cfg.apex_height_tolerance:
            self.height_matched_count += 1

        if self.energy_command is not None:
            self.pos_work_target.append(float(self.energy_command.work_per_target_height_pulse("positive")[i]))
            self.abs_work_target.append(float(self.energy_command.work_per_target_height_pulse("absolute")[i]))
            self.pos_work_height.append(float(self.energy_command.work_per_height_pulse("positive")[i]))
            self.abs_work_height.append(float(self.energy_command.work_per_height_pulse("absolute")[i]))

    def summary(self, reasons: list[str]) -> dict[str, float | int | str | bool]:
        failed_reasons = [reason for reason in reasons if reason != "time_out"]
        success = "time_out" in reasons and not failed_reasons
        reason = "+".join(reasons) if reasons else "unknown"
        orientation_rms = math.sqrt(self.orientation_error_sum / max(self.step_count, 1))
        yaw_rms = math.sqrt(self.yaw_error_sq_sum / max(self.step_count, 1))
        braking_ratio = self.negative_work / max(self.absolute_work, 1.0e-6)
        return {
            "success": success,
            "reason": reason,
            "apex": self.apex_count,
            "matched": self.height_matched_count,
            "mae": _mean(self.height_abs_errors),
            "rmse": _rmse(self.height_errors),
            "bias": _mean(self.height_errors),
            "h_over_target": _mean(self.height_ratios),
            "p90": _p90(self.height_abs_errors),
            "pos_target": _mean(self.pos_work_target),
            "abs_target": _mean(self.abs_work_target),
            "pos_height": _mean(self.pos_work_height),
            "abs_height": _mean(self.abs_work_height),
            "braking_ratio": braking_ratio,
            "xy_drift": self.last_xy_drift,
            "yaw_drift": self.last_yaw_drift,
            "yaw_rms": yaw_rms,
            "orientation_rms": orientation_rms,
        }


def _aggregate(condition: dict[str, float], episodes: list[dict[str, float | int | str | bool]]) -> dict[str, float | str]:
    failure_counts = Counter(str(ep["reason"]) for ep in episodes if not bool(ep["success"]))
    result: dict[str, float | str] = {
        **condition,
        "episodes": len(episodes),
        "success_rate": sum(1 for ep in episodes if bool(ep["success"])) / max(len(episodes), 1),
        "failure_distribution": (
            ";".join(f"{key}:{value}" for key, value in sorted(failure_counts.items())) if failure_counts else "none"
        ),
    }
    metric_names = [
        "apex",
        "matched",
        "mae",
        "rmse",
        "bias",
        "h_over_target",
        "p90",
        "pos_target",
        "abs_target",
        "pos_height",
        "abs_height",
        "braking_ratio",
        "xy_drift",
        "yaw_drift",
        "yaw_rms",
        "orientation_rms",
    ]
    for name in metric_names:
        values = [float(ep[name]) for ep in episodes]
        result[f"{name}_mean"] = _mean(values)
        result[f"{name}_std"] = _std(values)
    return result


def _print_episode(condition: dict[str, float], episode_index: int, stats: dict[str, float | int | str | bool]):
    print(
        f"[EP {episode_index:04d}] E={condition['youngs_modulus']:.2e} h={condition['target_height']:.2f} "
        f"ok={int(stats['success'])} reason={stats['reason']} apex={stats['apex']} "
        f"mae={_fmt(stats['mae'])} bias={_fmt(stats['bias'])} h/t={_fmt(stats['h_over_target'])} "
        f"posT={_fmt(stats['pos_target'], 1)} absT={_fmt(stats['abs_target'], 1)}",
        flush=True,
    )


def _print_condition_result(result: dict[str, float | str]):
    print(
        f"[RESULT] E={result['youngs_modulus']:.2e} target={result['target_height']:.2f} "
        f"drop={result['drop_height']:.2f} mass={result['mass']:.2f} "
        f"success={100.0 * result['success_rate']:.1f}% fail={result['failure_distribution']} "
        f"apex={_fmt(result['apex_mean'])}±{_fmt(result['apex_std'])} "
        f"mae={_fmt(result['mae_mean'])}±{_fmt(result['mae_std'])} "
        f"bias={_fmt(result['bias_mean'])}±{_fmt(result['bias_std'])} "
        f"h/t={_fmt(result['h_over_target_mean'])}±{_fmt(result['h_over_target_std'])} "
        f"posT={_fmt(result['pos_target_mean'], 1)}±{_fmt(result['pos_target_std'], 1)} "
        f"absT={_fmt(result['abs_target_mean'], 1)}±{_fmt(result['abs_target_std'], 1)} "
        f"xy={_fmt(result['xy_drift_mean'])}±{_fmt(result['xy_drift_std'])}",
        flush=True,
    )


def _prepare_csv(path: str):
    path_obj = pathlib.Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text("", encoding="utf-8")
    print(f"[INFO] Streaming aggregate CSV rows to: {path_obj}")


def _format_csv_value(value):
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.3f}"
    return value


def _append_csv_row(path: str, row: dict[str, float | str]):
    path_obj = pathlib.Path(path)
    write_header = not path_obj.exists() or path_obj.stat().st_size == 0
    formatted_row = {key: _format_csv_value(value) for key, value in row.items()}
    with path_obj.open("a", newline="", encoding="utf-8") as stream:
        fieldnames = list(row.keys())
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(formatted_row)
    print(f"[INFO] Appended CSV row: {path_obj}", flush=True)


def _load_checkpoint_from_wandb() -> str:
    import wandb

    run_path = args_cli.wandb_path
    api = wandb.Api()
    if "model" in args_cli.wandb_path:
        run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
    wandb_run = api.run(run_path)
    files = [file.name for file in wandb_run.files() if "model" in file.name]
    if "model" in args_cli.wandb_path:
        file = args_cli.wandb_path.split("/")[-1]
    else:
        file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

    wandb_file = wandb_run.file(str(file))
    wandb_file.download("./logs/rsl_rl/temp", replace=True)
    print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
    return f"./logs/rsl_rl/temp/{file}"


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Evaluate with RSL-RL agent."""
    env_cfg = apply_play_overrides(env_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs
    if hasattr(env_cfg.commands, "hop"):
        env_cfg.commands.hop.resampling_time_range = (1.0e9, 1.0e9)

    resume_path = _load_checkpoint_from_wandb()

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    hop_command, energy_command = _get_rebounce_handles(env)
    aggregate_rows = []
    drop_heights = args_cli.drop_heights
    csv_path = _csv_path_with_wandb_id(args_cli.csv_path, args_cli.wandb_path) if args_cli.csv_path is not None else None
    conditions = []
    for youngs_modulus, mass, target_height in product(args_cli.youngs_moduli, args_cli.masses, args_cli.target_heights):
        condition_drop_heights = drop_heights if drop_heights is not None and len(drop_heights) > 0 else [target_height]
        for drop_height in condition_drop_heights:
            conditions.append(
                {
                    "youngs_modulus": float(youngs_modulus),
                    "mass": float(mass),
                    "target_height": float(target_height),
                    "drop_height": float(drop_height),
                }
            )

    if csv_path is not None:
        _prepare_csv(csv_path)

    total_conditions = len(conditions)
    print(
        f"[INFO] Evaluating {total_conditions} conditions with num_envs={args_cli.num_envs}, "
        f"episodes_per_condition={args_cli.episodes_per_condition}",
        flush=True,
    )

    for condition_index, condition in enumerate(conditions, start=1):
        youngs_modulus = condition["youngs_modulus"]
        mass = condition["mass"]
        target_height = condition["target_height"]
        drop_height = condition["drop_height"]
        condition = {
            "youngs_modulus": float(youngs_modulus),
            "mass": float(mass),
            "target_height": float(target_height),
            "drop_height": float(drop_height),
        }
        print(
            f"[COND {condition_index:03d}/{total_conditions:03d}] "
            f"E={youngs_modulus:.2e} mass={mass:.2f} target={target_height:.2f} "
            f"drop={drop_height:.2f} episodes={args_cli.episodes_per_condition}",
            flush=True,
        )
        _set_condition(env, float(target_height), float(drop_height), float(youngs_modulus), float(mass))
        obs, _ = env.reset()
        stats = [EpisodeStats(env, env_id, energy_command) for env_id in range(env.unwrapped.num_envs)]
        episodes = []
        condition_start_time = time.monotonic()
        last_progress_episodes = 0

        while simulation_app.is_running() and len(episodes) < args_cli.episodes_per_condition:
            for episode_stats in stats:
                episode_stats.update_state(env)

            with torch.inference_mode():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            apex_env_ids = hop_command.is_apex.nonzero(as_tuple=True)[0].tolist()
            for env_id in apex_env_ids:
                stats[env_id].record_apex(hop_command)

            done_env_ids = dones.nonzero(as_tuple=True)[0].tolist()
            for env_id in done_env_ids:
                if len(episodes) >= args_cli.episodes_per_condition:
                    stats[env_id].reset(env)
                    continue
                episode_summary = stats[env_id].summary(_get_done_reasons(env, env_id))
                episodes.append(episode_summary)
                if args_cli.print_episodes:
                    _print_episode(condition, len(episodes), episode_summary)
                stats[env_id].reset(env)
            if (
                len(episodes) == args_cli.episodes_per_condition
                or len(episodes) - last_progress_episodes >= max(args_cli.progress_interval, 1)
            ):
                elapsed = max(time.monotonic() - condition_start_time, 1.0e-6)
                rate = len(episodes) / elapsed
                print(
                    f"[PROGRESS {condition_index:03d}/{total_conditions:03d}] "
                    f"episodes={len(episodes)}/{args_cli.episodes_per_condition} "
                    f"elapsed={elapsed:.1f}s rate={rate:.2f} ep/s",
                    flush=True,
                )
                last_progress_episodes = len(episodes)

        result = _aggregate(condition, episodes)
        aggregate_rows.append(result)
        _print_condition_result(result)
        if csv_path is not None:
            _append_csv_row(csv_path, result)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
