"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--wandb_path", type=str, required=True, help="Wandb run path (entity/project/run_id[/model]).")
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

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.task_utils import apply_play_overrides


PEAK_HEIGHT_UPDATE_INTERVAL_S = 5.0
PEAK_HEIGHT_PLAY_RANGES = (
    (0.1, 0.1),
    (0.2, 0.2),
    (0.3, 0.3),
    (0.4, 0.4),
    (0.5, 0.5),
    (0.6, 0.6),
    (0.7, 0.7),
    (0.8, 0.8),
    (0.9, 0.9),
    (1.0, 1.0),
)


def _set_peak_height_range(env, peak_height_range: tuple[float, float]) -> bool:
    command_manager = getattr(env.unwrapped, "command_manager", None)
    if command_manager is None or "hop" not in command_manager.active_terms:
        print("[WARN]: Cannot update peak_height range because this task has no 'hop' command.")
        return False

    hop_command = command_manager.get_term("hop")
    hop_command.cfg.ranges.peak_height = peak_height_range
    env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
    hop_command._resample(env_ids)
    print(f"[INFO]: Set hop peak_height range to {peak_height_range}.")
    return True


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # Align with mjlab: play mode is chosen by this entry script, while env-specific overrides live on the config.
    env_cfg = apply_play_overrides(env_cfg)
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
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    peak_height_index = 0
    # peak_height_updates_enabled = _set_peak_height_range(env, PEAK_HEIGHT_PLAY_RANGES[peak_height_index])
    elapsed_time_s = 0.0
    next_peak_height_update_s = PEAK_HEIGHT_UPDATE_INTERVAL_S
    obs = env.get_observations()
    hop_command = env.unwrapped.command_manager.get_term("hop") if "hop" in env.unwrapped.command_manager.active_terms else None
    hop_count = 0
    # simulate environment
    while simulation_app.is_running():
        # if peak_height_updates_enabled and elapsed_time_s >= next_peak_height_update_s:
        #     peak_height_index = (peak_height_index + 1) % len(PEAK_HEIGHT_PLAY_RANGES)
        #     _set_peak_height_range(env, PEAK_HEIGHT_PLAY_RANGES[peak_height_index])
        #     obs = env.get_observations()
        #     next_peak_height_update_s += PEAK_HEIGHT_UPDATE_INTERVAL_S

        # run policy inference without putting mutable environment buffers into inference mode
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        elapsed_time_s += env.unwrapped.step_dt

        if hop_command is not None and bool(hop_command.just_landed[0]):
            hop_count += 1
            peak_h = float(hop_command.last_peak_height[0])
            peak_z = float(hop_command.last_peak_z[0])
            air_t = float(hop_command.last_air_time[0])
            print(f"[HOP {hop_count:04d}] peak_height={peak_h:.3f} m (peak_z={peak_z:.3f} m), air_time={air_t:.3f} s")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
