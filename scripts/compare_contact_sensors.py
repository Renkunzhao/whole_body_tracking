"""Compare GPU/PhysX and Pinocchio DOB contact sensors on a trained policy.

The script loads an RSL-RL checkpoint, rolls out the selected task, and updates
both contact sensors from the same simulator state. It is meant for checking
whether the GPU sensor is close enough to the original Pinocchio estimator on
the actual rebounce policy trajectory.

Example:
    python scripts/compare_contact_sensors.py --headless \
        --task Go2-Rebounce-Trampoline \
        --load_run RUN_DIR --checkpoint model_5000.pt \
        --num_envs 16 --steps 500
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
import time

from isaaclab.app import AppLauncher

RSL_RL_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent / "rsl_rl"
sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))
import cli_args  # isort: skip  # noqa: E402


parser = argparse.ArgumentParser(description="Compare DOB contact sensors on an RSL-RL policy rollout.")
parser.add_argument("--task", type=str, default="Go2-Rebounce-Trampoline", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of parallel envs.")
parser.add_argument("--steps", type=int, default=500, help="Benchmark steps after warmup.")
parser.add_argument("--warmup", type=int, default=20, help="Warmup steps before measuring sensor differences.")
parser.add_argument("--pinocchio_workers", type=int, default=1, help="Worker threads for the Pinocchio sensor.")
parser.add_argument("--target_height", type=float, default=None, help="Fix the rebounce target height.")
parser.add_argument("--drop_height", type=float, default=None, help="Fix the reset drop height.")
parser.add_argument("--youngs_modulus", type=float, default=None, help="Fix trampoline Young's modulus.")
parser.add_argument("--trampoline_mass", type=float, default=None, help="Fix trampoline mass.")
parser.add_argument("--dynamic_friction", type=float, default=None, help="Fix trampoline dynamic friction.")
parser.add_argument("--elasticity_damping", type=float, default=None, help="Fix trampoline elasticity damping.")
parser.add_argument("--damping_scale", type=float, default=None, help="Fix trampoline damping scale.")
parser.add_argument("--poissons_ratio", type=float, default=None, help="Fix trampoline Poisson's ratio.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Everything after Isaac Sim is launched ----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import whole_body_tracking.tasks  # noqa: E402,F401
from whole_body_tracking.sensors import (  # noqa: E402
    create_dob_contact_sensor,
    is_pinocchio_dob_available,
)
from whole_body_tracking.utils.task_utils import apply_play_overrides  # noqa: E402


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _mean(values: list[float]) -> float:
    return float("nan") if not values else sum(values) / len(values)


def _fmt(value: float, precision: int = 4) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{precision}f}"


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


def _load_local_checkpoint(agent_cfg: RslRlOnPolicyRunnerCfg) -> str:
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    return resume_path


def _set_fixed_condition(env_cfg):
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


def _load_policy(env, agent_cfg: RslRlOnPolicyRunnerCfg, resume_path: str):
    runner_class_name = getattr(agent_cfg, "class_name", "OnPolicyRunner")
    if runner_class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif runner_class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported RSL-RL runner class: {runner_class_name}")
    runner.load(resume_path)
    return runner.get_inference_policy(device=env.unwrapped.device)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg = apply_play_overrides(env_cfg)
    env_cfg = _set_fixed_condition(env_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs

    if not is_pinocchio_dob_available():
        raise RuntimeError(
            "Cannot compare GPU and Pinocchio DOB sensors because the optional "
            "'pinocchio' package is not installed in this environment."
        )

    resume_path = (
        _download_checkpoint_from_wandb(args_cli.wandb_path)
        if args_cli.wandb_path
        else _load_local_checkpoint(agent_cfg)
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    print(f"\n[INFO] Task: {args_cli.task} | num_envs: {num_envs} | device: {device}")

    policy = _load_policy(env, agent_cfg, resume_path)
    pin_sensor = create_dob_contact_sensor(
        env,
        backend="pinocchio",
        fallback_to_gpu=False,
        num_workers=args_cli.pinocchio_workers,
    )
    gpu_sensor = create_dob_contact_sensor(env, backend="gpu")

    obs, _ = env.reset()
    print(f"[INFO] Warming up for {args_cli.warmup} policy steps")
    for _ in range(args_cli.warmup):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        pin_sensor.update(force=True)
        gpu_sensor.update(force=True)
    _sync()

    pin_times: list[float] = []
    gpu_times: list[float] = []
    force_mean_diffs: list[float] = []
    force_max_diffs: list[float] = []
    tau_mean_diffs: list[float] = []
    tau_joint_mean_diffs: list[float] = []
    tau_max_diffs: list[float] = []
    tau_joint_max_diffs: list[float] = []
    valid_steps = 0

    print(f"[INFO] Running {args_cli.steps} measured policy steps")
    for _ in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

        _sync()
        t0 = time.perf_counter()
        pin_sensor.update(force=True)
        _sync()
        pin_times.append(time.perf_counter() - t0)

        _sync()
        t1 = time.perf_counter()
        gpu_sensor.update(force=True)
        _sync()
        gpu_times.append(time.perf_counter() - t1)

        valid = pin_sensor.data.valid & gpu_sensor.data.valid
        if valid.any():
            pin_f = pin_sensor.data.foot_forces_w[valid]
            gpu_f = gpu_sensor.data.foot_forces_w[valid]
            pin_tau = pin_sensor.data.tau_residual[valid]
            gpu_tau = gpu_sensor.data.tau_residual[valid]

            force_delta = (pin_f - gpu_f).abs()
            tau_delta = (pin_tau - gpu_tau).abs()
            tau_joint_delta = tau_delta[:, 6:]
            force_mean_diffs.append(force_delta.mean().item())
            force_max_diffs.append(force_delta.max().item())
            tau_mean_diffs.append(tau_delta.mean().item())
            tau_joint_mean_diffs.append(tau_joint_delta.mean().item())
            tau_max_diffs.append(tau_delta.max().item())
            tau_joint_max_diffs.append(tau_joint_delta.max().item())
            valid_steps += 1

    env.close()

    pin_ms = _mean(pin_times) * 1000.0
    gpu_ms = _mean(gpu_times) * 1000.0
    speedup = pin_ms / gpu_ms if gpu_ms > 0.0 else float("nan")

    print("\n" + "=" * 72)
    print(f"  task                 : {args_cli.task}")
    print(f"  checkpoint           : {resume_path}")
    print(f"  num_envs             : {num_envs}")
    print(f"  measured steps       : {args_cli.steps} (valid: {valid_steps})")
    print("-" * 72)
    print(f"  Pinocchio sensor     : {pin_ms:.3f} ms/step")
    print(f"  GPU/PhysX sensor     : {gpu_ms:.3f} ms/step")
    print(f"  speedup              : {speedup:.1f}x")
    if force_mean_diffs:
        print("-" * 72)
        print(f"  Mean |delta foot_forces_w|       : {_fmt(_mean(force_mean_diffs))} N")
        print(f"  Max  |delta foot_forces_w|       : {_fmt(max(force_max_diffs))} N")
        print(f"  Mean |delta tau_residual all|    : {_fmt(_mean(tau_mean_diffs))} N*m")
        print(f"  Max  |delta tau_residual all|    : {_fmt(max(tau_max_diffs))} N*m")
        print(f"  Mean |delta tau_residual joints| : {_fmt(_mean(tau_joint_mean_diffs))} N*m")
        print(f"  Max  |delta tau_residual joints| : {_fmt(max(tau_joint_max_diffs))} N*m")
        print()
        print("  Note: all-row tau_residual includes floating-base rows 0:6, where")
        print("  Pinocchio uses body-frame base velocity and PhysX uses world-frame")
        print("  base velocity. Joint rows 6:18 and foot forces are the main comparison.")
    print("=" * 72)


if __name__ == "__main__":
    main()
    simulation_app.close()
