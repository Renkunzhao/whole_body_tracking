"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import math
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--target_height", type=float, default=None, help="Fix the rebounce target height during play.")
parser.add_argument("--drop_height", type=float, default=None, help="Fix the reset drop height during play.")
parser.add_argument("--youngs_modulus", type=float, default=None, help="Fix trampoline Young's modulus during play.")
parser.add_argument("--trampoline_mass", type=float, default=None, help="Fix trampoline mass during play.")
parser.add_argument("--dynamic_friction", type=float, default=None, help="Fix trampoline dynamic friction during play.")
parser.add_argument("--elasticity_damping", type=float, default=None, help="Fix trampoline elasticity damping during play.")
parser.add_argument("--damping_scale", type=float, default=None, help="Fix trampoline damping scale during play.")
parser.add_argument("--poissons_ratio", type=float, default=None, help="Fix trampoline Poisson's ratio during play.")
parser.add_argument("--video", action="store_true", default=False, help="Record a real-time video of the play loop.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in env steps).")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# offscreen camera is required for video capture; safe to enable alongside GUI
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from analyze_rebounce_play import analyze_rebounce_play
from whole_body_tracking.sensors import create_dob_contact_sensor, get_or_create_dob_contact_sensor
from whole_body_tracking.utils.exporter import attach_onnx_metadata, get_policy_export_normalizer
from whole_body_tracking.utils.task_utils import apply_play_overrides


CONTACT_FORCE_FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
CONTACT_FORCE_AXES = ("x", "y", "z")


def _mean(values):
    return float("nan") if not values else sum(values) / len(values)


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


def _get_rebounce_debug_handles(env):
    command_manager = getattr(env.unwrapped, "command_manager", None)
    if command_manager is None or "hop" not in command_manager.active_terms:
        print("[WARN]: Cannot print apex heights because this task has no 'hop' command.")
        return None, None
    hop_command = command_manager.get_term("hop")
    energy_command = command_manager.get_term("energy") if "energy" in command_manager.active_terms else None
    return hop_command, energy_command


def _get_trampoline_randomizer(env):
    event_manager = getattr(env.unwrapped, "event_manager", None)
    if event_manager is None:
        return None
    try:
        event_term = event_manager.get_term_cfg("randomize_trampoline_properties").func
    except ValueError:
        return None
    if not hasattr(event_term, "last_youngs_moduli") or not hasattr(event_term, "last_masses"):
        return None
    return event_term


def _maybe_print_property(parts: list[str], randomizer, attr_name: str, label: str, fmt: str = ".3f"):
    values = getattr(randomizer, attr_name, None)
    if values is None:
        return
    parts.append(f"{label}={float(values[0]):{fmt}}")


def _print_trampoline_params(env, reset_count: int):
    randomizer = _get_trampoline_randomizer(env)
    if randomizer is None:
        return
    parts = [
        f"E={float(randomizer.last_youngs_moduli[0]):.2e}",
        f"m={float(randomizer.last_masses[0]):.2f}",
    ]
    _maybe_print_property(parts, randomizer, "last_dynamic_frictions", "mu", ".2f")
    _maybe_print_property(parts, randomizer, "last_elasticity_dampings", "damp", ".3f")
    _maybe_print_property(parts, randomizer, "last_damping_scales", "ds", ".2f")
    _maybe_print_property(parts, randomizer, "last_poissons_ratios", "nu", ".2f")
    print(f"[TRAMP R{reset_count:03d}] " + " ".join(parts), flush=True)


class RebouncePlayCsvLogger:
    """Temporary play logger for rebounce diagnostics."""

    def __init__(self, env, hop_command, csv_path: str, contact_sensors: dict[str, object] | None = None):
        self.env = env
        self.hop_command = hop_command
        self.robot = env.unwrapped.scene["robot"]
        self.trampoline = self._get_trampoline_asset()
        self.trampoline_center_node_id = self._resolve_trampoline_center_node_id()
        self.csv_path = os.path.abspath(os.path.expanduser(csv_path))
        self.contact_sensors = contact_sensors or {}
        self.contact_sensor_labels = tuple(self.contact_sensors.keys())
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._write_header()
        print(f"[INFO]: Logging rebounce play data to {self.csv_path}", flush=True)

    def _write_header(self):
        header = [
            "global_step",
            "episode",
            "episode_step",
            "sim_time_s",
            "is_air",
            "feet_below_clearance",
            "is_apex",
            "root_z",
            "root_vz",
            "min_foot_z",
            "max_foot_z",
            "trampoline_center/z",
            "trampoline_center/vz",
        ]
        for label in self.contact_sensor_labels:
            header.append(f"contact_force/{label}/valid")
            header.extend(f"contact_force/{label}/total/{axis}" for axis in CONTACT_FORCE_AXES)
            for foot_name in CONTACT_FORCE_FOOT_NAMES:
                header.extend(f"contact_force/{label}/{foot_name}/{axis}" for axis in CONTACT_FORCE_AXES)
        header.extend(f"joint_vel/{name}" for name in self.robot.joint_names)
        self._writer.writerow(header)

    def _air_flags(self) -> tuple[int | float, int | float]:
        if self.hop_command is None or not hasattr(self.hop_command, "_feet_clearance_flags"):
            return float("nan"), float("nan")
        feet_above_clearance, feet_below_clearance = self.hop_command._feet_clearance_flags()
        return int(bool(feet_above_clearance[0].item())), int(bool(feet_below_clearance[0].item()))

    def _foot_z_range(self) -> tuple[float, float]:
        foot_asset = getattr(self.hop_command, "_foot_asset", None) if self.hop_command is not None else None
        foot_body_ids = getattr(self.hop_command, "_foot_body_ids", None) if self.hop_command is not None else None
        if foot_asset is None or foot_body_ids is None:
            return float("nan"), float("nan")
        foot_z_local = foot_asset.data.body_pos_w[0, foot_body_ids, 2] - self.env.unwrapped.scene.env_origins[0, 2]
        return float(torch.min(foot_z_local).item()), float(torch.max(foot_z_local).item())

    def _get_trampoline_asset(self):
        try:
            return self.env.unwrapped.scene["trampoline"]
        except (KeyError, AttributeError):
            return None

    def _resolve_trampoline_center_node_id(self) -> int | None:
        if self.trampoline is None:
            return None
        default_nodal_state = getattr(self.trampoline.data, "default_nodal_state_w", None)
        if default_nodal_state is None:
            return None
        nodal_pos = default_nodal_state[0, :, :3]
        center_xy = nodal_pos[:, :2].mean(dim=0, keepdim=True)
        radial_distance = torch.linalg.vector_norm(nodal_pos[:, :2] - center_xy, dim=-1)
        return int(torch.argmin(radial_distance).item())

    def _trampoline_center_z_values(self) -> tuple[float, float]:
        if self.trampoline is None or self.trampoline_center_node_id is None:
            return float("nan"), float("nan")
        try:
            center_z = self.trampoline.data.nodal_pos_w[0, self.trampoline_center_node_id, 2]
            center_vz = self.trampoline.data.nodal_vel_w[0, self.trampoline_center_node_id, 2]
        except (AttributeError, IndexError, RuntimeError):
            return float("nan"), float("nan")
        return float(center_z.item()), float(center_vz.item())

    def _contact_sensor_values(self, sensor) -> list[str]:
        nan_values = ["nan"] * (1 + 3 + 3 * len(CONTACT_FORCE_FOOT_NAMES))
        data = getattr(sensor, "data", None)
        if data is None:
            return nan_values

        try:
            valid = int(bool(data.valid[0].item()))
            total_force = data.total_force_w[0].detach().cpu().tolist()
            foot_forces = data.foot_forces_w[0].detach().cpu()
        except (AttributeError, IndexError, RuntimeError):
            return nan_values

        values = [str(valid)]
        values.extend(f"{float(value):.6f}" for value in total_force[:3])
        for foot_id in range(len(CONTACT_FORCE_FOOT_NAMES)):
            if foot_id >= foot_forces.shape[0]:
                values.extend(["nan", "nan", "nan"])
            else:
                values.extend(f"{float(value):.6f}" for value in foot_forces[foot_id, :3].tolist())
        return values

    def _contact_values(self) -> list[str]:
        values = []
        for label in self.contact_sensor_labels:
            values.extend(self._contact_sensor_values(self.contact_sensors[label]))
        return values

    def record(self, global_step: int, episode: int, sim_time_s: float):
        is_air, feet_below_clearance = self._air_flags()
        min_foot_z, max_foot_z = self._foot_z_range()
        trampoline_center_z, trampoline_center_vz = self._trampoline_center_z_values()
        episode_step = int(self.env.unwrapped.episode_length_buf[0].item())
        root_z = float(self.robot.data.root_pos_w[0, 2].item())
        root_vz = float(self.robot.data.root_lin_vel_w[0, 2].item())
        is_apex = int(bool(self.hop_command is not None and self.hop_command.is_apex[0].item()))
        joint_vel = self.robot.data.joint_vel[0].detach().cpu().tolist()
        self._writer.writerow(
            [
                global_step,
                episode,
                episode_step,
                f"{sim_time_s:.6f}",
                is_air,
                feet_below_clearance,
                is_apex,
                f"{root_z:.6f}",
                f"{root_vz:.6f}",
                f"{min_foot_z:.6f}",
                f"{max_foot_z:.6f}",
                f"{trampoline_center_z:.6f}",
                f"{trampoline_center_vz:.6f}",
                *self._contact_values(),
                *[f"{value:.6f}" for value in joint_vel],
            ]
        )
        if global_step % 100 == 0:
            self._file.flush()

    def close(self):
        self._file.flush()
        self._file.close()


def _get_done_reasons(env, env_id: int = 0) -> list[str]:
    termination_manager = getattr(env.unwrapped, "termination_manager", None)
    if termination_manager is None:
        return []
    reasons = []
    for name in termination_manager.active_terms:
        if bool(termination_manager.get_term(name)[env_id]):
            reasons.append(name)
    return reasons


def _wrapped_angle_error(angle: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    delta = angle - reference
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def _set_fixed_play_condition(env_cfg):
    commands = getattr(env_cfg, "commands", None)
    if commands is not None and hasattr(commands, "hop"):
        commands.hop.resampling_time_range = (1.0e9, 1.0e9)
        if args_cli.target_height is not None:
            commands.hop.ranges.peak_height = (args_cli.target_height, args_cli.target_height)
            curriculum = getattr(env_cfg, "curriculum", None)
            if curriculum is not None and hasattr(curriculum, "hopping_init_height"):
                curriculum.hopping_init_height = None

    events = getattr(env_cfg, "events", None)
    if events is not None and hasattr(events, "reset_drop"):
        if args_cli.drop_height is not None:
            events.reset_drop.params["drop_height_range"] = (args_cli.drop_height, args_cli.drop_height)
        elif args_cli.target_height is not None:
            events.reset_drop.params["drop_height_range"] = (args_cli.target_height, args_cli.target_height)

    if events is not None and hasattr(events, "randomize_trampoline_properties"):
        params = events.randomize_trampoline_properties.params
        params["randomization_start_step"] = 0
        if args_cli.youngs_modulus is not None:
            params["youngs_modulus_range"] = (args_cli.youngs_modulus, args_cli.youngs_modulus)
            params["youngs_modulus_distribution"] = "uniform"
        if args_cli.trampoline_mass is not None:
            params["mass_range"] = (args_cli.trampoline_mass, args_cli.trampoline_mass)
        if args_cli.dynamic_friction is not None:
            params["dynamic_friction_range"] = (args_cli.dynamic_friction, args_cli.dynamic_friction)
        if args_cli.elasticity_damping is not None:
            params["elasticity_damping_range"] = (args_cli.elasticity_damping, args_cli.elasticity_damping)
        if args_cli.damping_scale is not None:
            params["damping_scale_range"] = (args_cli.damping_scale, args_cli.damping_scale)
        if args_cli.poissons_ratio is not None:
            params["poissons_ratio_range"] = (args_cli.poissons_ratio, args_cli.poissons_ratio)

    return env_cfg


def _run_id_for_video(wandb_path: str | None, resume_path: str) -> str:
    if wandb_path:
        segments = [segment for segment in wandb_path.split("/") if segment]
        if segments and "model" in segments[-1]:
            segments = segments[:-1]
        if segments:
            return segments[-1]
    # local --load_run: use the run folder name (parent of the checkpoint file)
    return os.path.basename(os.path.dirname(os.path.abspath(resume_path))) or "local"


def _video_name_prefix(args, run_id: str) -> str:
    parts = [f"run-{run_id}"]
    keys = [
        ("target_height", "h", ".2f"),
        ("youngs_modulus", "E", ".1e"),
        ("trampoline_mass", "m", ".1f"),
        ("dynamic_friction", "mu", ".2f"),
        ("elasticity_damping", "damp", ".3f"),
        ("damping_scale", "ds", ".2f"),
        ("poissons_ratio", "nu", ".2f"),
    ]
    for attr, label, fmt in keys:
        value = getattr(args, attr, None)
        if value is None:
            continue
        parts.append(f"{label}{format(float(value), fmt)}")
    return "_".join(parts)


def _play_output_dir(resume_path: str, name_prefix: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(resume_path)), "play-rebounce", name_prefix)


def _play_csv_path(play_output_dir: str) -> str:
    return os.path.join(play_output_dir, "rebounce_play.csv")


def _run_rebounce_play_analysis(csv_path: str, output_dir: str) -> None:
    print(f"[INFO]: Running rebounce play analysis in {output_dir}", flush=True)
    try:
        analyze_rebounce_play(csv_path, output_dir)
    except Exception as exc:
        print(f"[WARN]: Rebounce play analysis failed: {exc}", flush=True)


def _download_checkpoint_from_wandb(wandb_path: str) -> str:
    import wandb

    run_path = wandb_path

    api = wandb.Api()
    if "model" in wandb_path:
        run_path = "/".join(wandb_path.split("/")[:-1])
    wandb_run = api.run(run_path)
    files = [file.name for file in wandb_run.files() if "model" in file.name]
    if not files:
        raise RuntimeError(f"No model checkpoint files found in wandb run '{run_path}'.")

    if "model" in wandb_path:
        checkpoint_file = wandb_path.split("/")[-1]
    else:
        checkpoint_file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

    wandb_file = wandb_run.file(str(checkpoint_file))
    wandb_file.download("./logs/rsl_rl/temp", replace=True)

    print(f"[INFO]: Loading model checkpoint from: {run_path}/{checkpoint_file}")
    return f"./logs/rsl_rl/temp/{checkpoint_file}"


class EpisodeStats:
    def __init__(self, env, energy_command):
        self.energy_command = energy_command
        self.reset(env)

    def reset(self, env):
        robot = env.unwrapped.scene["robot"]
        self.start_xy = robot.data.root_pos_w[0, :2].clone()
        self.start_yaw = euler_xyz_from_quat(robot.data.root_quat_w[0:1])[2][0].clone()
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
        root_xy = robot.data.root_pos_w[0, :2]
        yaw = euler_xyz_from_quat(robot.data.root_quat_w[0:1])[2][0]
        yaw_error = _wrapped_angle_error(yaw, self.start_yaw)
        orientation_error = torch.sum(torch.square(robot.data.projected_gravity_b[0, :2]))

        self.last_xy_drift = float(torch.linalg.vector_norm(root_xy - self.start_xy))
        self.last_yaw_drift = abs(float(yaw_error))
        self.orientation_error_sum += float(orientation_error)
        self.yaw_error_sq_sum += float(yaw_error * yaw_error)
        self.step_count += 1

        if self.energy_command is not None:
            dt = env.unwrapped.step_dt
            self.negative_work += float(self.energy_command.negative_power[0]) * dt
            self.absolute_work += float(self.energy_command.absolute_power[0]) * dt

    def record_apex(self, hop_command):
        self.apex_count += 1
        apex_height = float(hop_command.last_apex_height[0])
        target_height = float(hop_command.last_apex_target_height[0])
        error = apex_height - target_height
        abs_error = abs(error)
        self.height_errors.append(error)
        self.height_abs_errors.append(abs_error)
        self.height_ratios.append(apex_height / max(target_height, 1.0e-6))
        if abs_error < hop_command.cfg.apex_height_tolerance:
            self.height_matched_count += 1

        if self.energy_command is not None:
            self.pos_work_target.append(float(self.energy_command.work_per_target_height_pulse("positive")[0]))
            self.abs_work_target.append(float(self.energy_command.work_per_target_height_pulse("absolute")[0]))
            self.pos_work_height.append(float(self.energy_command.work_per_height_pulse("positive")[0]))
            self.abs_work_height.append(float(self.energy_command.work_per_height_pulse("absolute")[0]))

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


def _print_episode_summary(episode_index: int, stats: dict[str, float | int | str | bool]):
    print(
        f"[EP{episode_index:03d}] ok={int(stats['success'])} reason={stats['reason']} "
        f"apex={stats['apex']} match={stats['matched']} "
        f"mae={_fmt(stats['mae'])} rmse={_fmt(stats['rmse'])} bias={_fmt(stats['bias'])} "
        f"h/t={_fmt(stats['h_over_target'])} p90={_fmt(stats['p90'])} "
        f"posT={_fmt(stats['pos_target'], 1)} absT={_fmt(stats['abs_target'], 1)} "
        f"brake={_fmt(stats['braking_ratio'])} xy={_fmt(stats['xy_drift'])} yaw={_fmt(stats['yaw_drift'])} "
        f"ori={_fmt(stats['orientation_rms'])}",
        flush=True,
    )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # Align with mjlab: play mode is chosen by this entry script, while env-specific overrides live on the config.
    env_cfg = apply_play_overrides(env_cfg)
    env_cfg = _set_fixed_play_condition(env_cfg)
    env_cfg.terminations = None
    if hasattr(env_cfg.rewards, "failed_termination"):
        env_cfg.rewards.failed_termination = None
    env_cfg.scene.num_envs = 1

    if args_cli.wandb_path:
        resume_path = _download_checkpoint_from_wandb(args_cli.wandb_path)
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    run_id = _run_id_for_video(args_cli.wandb_path, resume_path)
    name_prefix = _video_name_prefix(args_cli, run_id)
    play_output_dir = _play_output_dir(resume_path, name_prefix)
    os.makedirs(play_output_dir, exist_ok=True)

    # create isaac environment
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # wrap with RecordVideo on the raw gym env, before RL-specific wrappers
    if args_cli.video:
        video_folder = play_output_dir
        video_name_prefix = "rebounce_play"
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": video_name_prefix,
            "disable_logger": True,
        }
        step_dt_estimate = float(env_cfg.sim.dt) * int(env_cfg.decimation)
        print(
            f"[INFO]: Recording video -> {video_folder}/{video_name_prefix}-step-0.mp4 "
            f"({args_cli.video_length} steps = {args_cli.video_length * step_dt_estimate:.2f}s).",
            flush=True,
        )
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load previously trained model
    runner_class_name = getattr(agent_cfg, "class_name", "OnPolicyRunner")
    if runner_class_name == "DistillationRunner":
        ppo_runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif runner_class_name == "OnPolicyRunner":
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported RSL-RL runner class: {runner_class_name}")
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to ONNX with the same observation normalizer used by inference
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(
        ppo_runner.alg.policy,
        path=export_model_dir,
        normalizer=get_policy_export_normalizer(ppo_runner.alg.policy),
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)

    obs, _ = env.reset()
    hop_command, energy_command = _get_rebounce_debug_handles(env)
    reset_count = 0
    _print_trampoline_params(env, reset_count)
    apex_count = 0
    episode_count = 0
    episode_stats = EpisodeStats(env, energy_command)
    pinocchio_dob_sensor = create_dob_contact_sensor(env, backend="pinocchio", fallback_to_gpu=True)
    gpu_dob_sensor = get_or_create_dob_contact_sensor(env, backend="gpu", update=False)
    contact_sensors = {
        "pinocchio": pinocchio_dob_sensor,
        "gpu": gpu_dob_sensor,
    }
    print(
        "[INFO]: DOB contact sensors: "
        f"pinocchio columns use backend={getattr(pinocchio_dob_sensor, 'dob_backend', 'unknown')}, "
        f"gpu columns use backend={getattr(gpu_dob_sensor, 'dob_backend', 'unknown')}",
        flush=True,
    )
    play_csv_path = _play_csv_path(play_output_dir)
    play_logger = RebouncePlayCsvLogger(env, hop_command, play_csv_path, contact_sensors=contact_sensors)
    play_step_count = 0
    sim_step_dt = float(env.unwrapped.step_dt)
    video_step_count = 0

    # simulate environment
    try:
        while simulation_app.is_running():
            episode_stats.update_state(env)
            # run policy inference without putting mutable environment buffers into inference mode
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            play_step_count += 1
            for sensor in contact_sensors.values():
                sensor.update()
            play_logger.record(play_step_count, episode_count, play_step_count * sim_step_dt)

            if hop_command is not None and bool(hop_command.is_apex[0]):
                apex_count += 1
                apex_height = float(hop_command.last_apex_height[0])
                target_height = float(hop_command.last_apex_target_height[0])
                drop_height = float(hop_command.drop_height[0])
                error = apex_height - target_height
                episode_stats.record_apex(hop_command)
                energy_text = ""
                if energy_command is not None:
                    positive_work_per_height = float(energy_command.work_per_height_pulse("positive")[0])
                    absolute_work_per_height = float(energy_command.work_per_height_pulse("absolute")[0])
                    energy_text = f" pos/h={positive_work_per_height:.1f} abs/h={absolute_work_per_height:.1f} J/m"
                dob_metrics = pinocchio_dob_sensor.consume_hop_metrics(0, target_height)
                energy_text += (
                    f" dob+={dob_metrics['positive_work_per_height']:.1f} "
                    f"dob-={dob_metrics['negative_work_per_height']:.1f} "
                    f"dobR={dob_metrics['return_ratio']:.2f} "
                    f"dobFz={dob_metrics['peak_total_force_z']:.0f}N"
                )
                print(
                    f"[A{apex_count:03d}] h={apex_height:.3f}/{target_height:.3f} "
                    f"e={error:+.3f} d={drop_height:.3f}{energy_text}",
                    flush=True,
                )

            if bool(dones[0]):
                episode_count += 1
                _print_episode_summary(episode_count, episode_stats.summary(_get_done_reasons(env)))
                dob_metrics = pinocchio_dob_sensor.episode_metrics(0)
                print(
                    f"[DOB{episode_count:03d}] dobW+={dob_metrics['positive_work']:.1f} "
                    f"dobW-={dob_metrics['negative_work']:.1f} dobR={dob_metrics['return_ratio']:.2f} "
                    f"dobFz={dob_metrics['peak_total_force_z']:.0f}N",
                    flush=True,
                )
                reset_count += 1
                apex_count = 0
                episode_stats.reset(env)
                for sensor in contact_sensors.values():
                    sensor.reset()
                _print_trampoline_params(env, reset_count)

            if args_cli.video:
                video_step_count += 1
                if video_step_count >= args_cli.video_length:
                    print(f"[INFO]: Video recording complete after {video_step_count} steps. Exiting.", flush=True)
                    break
    except KeyboardInterrupt:
        print("[INFO]: Interrupted by user. Finalizing video (if recording) before exit...", flush=True)
    finally:
        play_logger.close()
        env.close()
        _run_rebounce_play_analysis(play_csv_path, play_output_dir)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
