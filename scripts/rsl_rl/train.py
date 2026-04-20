# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wand registry.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
import pickle

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner, MyOnPolicyRunner
from whole_body_tracking.utils.task_utils import env_cfg_requires_motion


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _normalize_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _install_render_toggle_hotkey(key_name: str = "V") -> None:
    """Toggle Isaac Sim viewport rendering with a hotkey (default: V), à la IsaacGym."""
    if args_cli.headless:
        return
    try:
        import carb.input
        import omni.appwindow
        from isaaclab.sim import SimulationContext
    except ImportError as exc:
        print(f"[WARN] Render-toggle hotkey unavailable: {exc}")
        return

    state = {"render_on": True, "prev_mode": None}

    def _on_keyboard_event(event, *_):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        if event.input != getattr(carb.input.KeyboardInput, key_name):
            return True
        sim = SimulationContext.instance()
        if sim is None:
            return True
        state["render_on"] = not state["render_on"]
        if state["render_on"]:
            target = state["prev_mode"] or sim.RenderMode.FULL_RENDERING
        else:
            state["prev_mode"] = sim.render_mode
            target = sim.RenderMode.NO_RENDERING
        sim.set_render_mode(target)
        print(f"[INFO] Viewport rendering {'ON' if state['render_on'] else 'OFF'} (mode={target.name}).")
        return True

    appwindow = omni.appwindow.get_default_app_window()
    keyboard = appwindow.get_keyboard()
    sub = carb.input.acquire_input_interface().subscribe_to_keyboard_events(keyboard, _on_keyboard_event)
    # keep a reference so the subscription is not garbage-collected
    _install_render_toggle_hotkey._sub = sub  # type: ignore[attr-defined]
    print(f"[INFO] Press '{key_name}' to toggle viewport rendering.")


def _download_checkpoint_from_wandb(wandb_path: str) -> tuple[str, object]:
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
    return f"./logs/rsl_rl/temp/{checkpoint_file}", wandb_run


def _load_motion_file_from_wandb_run(env_cfg, wandb_run) -> None:
    art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
    if art is None:
        raise RuntimeError("No motion artifact found in the wandb run.")
    env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    registry_name = _normalize_optional_str(args_cli.registry_name)
    wandb_path = _normalize_optional_str(args_cli.wandb_path)
    wandb_run = None
    resume_path = None

    if wandb_path is not None:
        resume_path, wandb_run = _download_checkpoint_from_wandb(wandb_path)

    # load the motion file from the wandb registry
    requires_motion = env_cfg_requires_motion(env_cfg)
    if requires_motion:
        if registry_name is not None:
            if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
                registry_name += ":latest"
            import wandb

            api = wandb.Api()
            artifact = api.artifact(registry_name)
            env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
        elif wandb_run is not None:
            _load_motion_file_from_wandb_run(env_cfg, wandb_run)
        else:
            raise ValueError("Tracking tasks require either --registry_name or --wandb_path.")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    _install_render_toggle_hotkey("V")
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if requires_motion:
        runner = MotionOnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
        )
    else:
        runner = MyOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if resume_path is None and agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if resume_path is not None:
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "env.pkl"), "wb") as f:
        pickle.dump(env_cfg, f)
    with open(os.path.join(log_dir, "params", "agent.pkl"), "wb") as f:
        pickle.dump(agent_cfg, f)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
