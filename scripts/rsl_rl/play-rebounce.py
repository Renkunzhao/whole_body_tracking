"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--wandb_path", type=str, required=True, help="Wandb run path (entity/project/run_id[/model]).")
parser.add_argument("--target_height", type=float, default=None, help="Fix the rebounce target height during play.")
parser.add_argument("--drop_height", type=float, default=None, help="Fix the reset drop height during play.")
parser.add_argument("--youngs_modulus", type=float, default=None, help="Fix trampoline Young's modulus during play.")
parser.add_argument("--trampoline_mass", type=float, default=None, help="Fix trampoline mass during play.")
parser.add_argument("--dynamic_friction", type=float, default=None, help="Fix trampoline dynamic friction during play.")
parser.add_argument("--elasticity_damping", type=float, default=None, help="Fix trampoline elasticity damping during play.")
parser.add_argument("--damping_scale", type=float, default=None, help="Fix trampoline damping scale during play.")
parser.add_argument("--poissons_ratio", type=float, default=None, help="Fix trampoline Poisson's ratio during play.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

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
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.task_utils import apply_play_overrides


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
    # Align with mjlab: play mode is chosen by this entry script, while env-specific overrides live on the config.
    env_cfg = apply_play_overrides(env_cfg)
    env_cfg = _set_fixed_play_condition(env_cfg)
    env_cfg.scene.num_envs = 1

    import wandb

    run_path = args_cli.wandb_path

    api = wandb.Api()
    if "model" in args_cli.wandb_path:
        run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
    wandb_run = api.run(run_path)
    # loop over files in the run
    files = [file.name for file in wandb_run.files() if "model" in file.name]
    # files are all model_xxx.pt find the largest filename
    if "model" in args_cli.wandb_path:
        file = args_cli.wandb_path.split("/")[-1]
    else:
        file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

    wandb_file = wandb_run.file(str(file))
    wandb_file.download("./logs/rsl_rl/temp", replace=True)

    print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
    resume_path = f"./logs/rsl_rl/temp/{file}"

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

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

    obs, _ = env.reset()
    hop_command, energy_command = _get_rebounce_debug_handles(env)
    reset_count = 0
    _print_trampoline_params(env, reset_count)
    apex_count = 0
    episode_count = 0
    episode_stats = EpisodeStats(env, energy_command)

    # simulate environment
    while simulation_app.is_running():
        episode_stats.update_state(env)
        # run policy inference without putting mutable environment buffers into inference mode
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

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
            print(
                f"[A{apex_count:03d}] h={apex_height:.3f}/{target_height:.3f} "
                f"e={error:+.3f} d={drop_height:.3f}{energy_text}",
                flush=True,
            )

        if bool(dones[0]):
            episode_count += 1
            _print_episode_summary(episode_count, episode_stats.summary(_get_done_reasons(env)))
            reset_count += 1
            apex_count = 0
            episode_stats.reset(env)
            _print_trampoline_params(env, reset_count)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
