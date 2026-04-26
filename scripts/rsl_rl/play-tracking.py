"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
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
import os
import pathlib
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
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import export_policy_as_onnx
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from whole_body_tracking.utils.task_utils import apply_play_overrides, env_cfg_requires_motion

class _JumpTracker:
    """Per-env takeoff/landing detector that prints peak base-z per jump.

    Detects flight by foot contact forces and tracks both robot and motion-reference
    apex z-positions during each air phase.
    """

    def __init__(
        self,
        unwrapped,
        base_idx: int,
        foot_sensor_idx: list[int],
        contact_threshold: float = 5.0,
    ) -> None:
        self.unwrapped = unwrapped
        self.robot = unwrapped.scene["robot"]
        self.contact_sensor = unwrapped.scene.sensors["contact_forces"]
        self.base_idx = base_idx
        self.foot_sensor_idx = foot_sensor_idx
        self.contact_threshold = contact_threshold

        cm = getattr(unwrapped, "command_manager", None)
        self.motion_cmd = cm.get_term("motion") if cm is not None and "motion" in cm.active_terms else None

        n = unwrapped.num_envs
        dev = unwrapped.device
        self.in_air_prev = torch.zeros(n, dtype=torch.bool, device=dev)
        self.peak_z = torch.full((n,), -1e9, device=dev)
        self.peak_z_ref = torch.full((n,), -1e9, device=dev)
        self.takeoff_z = torch.zeros(n, device=dev)
        self.hop_count = 0

    def update(self) -> None:
        forces = self.contact_sensor.data.net_forces_w_history[:, :, self.foot_sensor_idx, :]
        contacts = torch.max(torch.norm(forces, dim=-1), dim=1)[0] > self.contact_threshold
        all_air = ~contacts.any(dim=1)

        z = self.robot.data.body_pos_w[:, self.base_idx, 2]
        z_ref = self.motion_cmd.anchor_pos_w[:, 2] if self.motion_cmd is not None else None

        self.peak_z = torch.where(all_air, torch.maximum(self.peak_z, z), self.peak_z)
        if z_ref is not None:
            self.peak_z_ref = torch.where(all_air, torch.maximum(self.peak_z_ref, z_ref), self.peak_z_ref)

        just_took_off = all_air & ~self.in_air_prev
        self.takeoff_z = torch.where(just_took_off, z, self.takeoff_z)

        just_landed = ~all_air & self.in_air_prev
        for env_id in torch.where(just_landed)[0].tolist():
            self.hop_count += 1
            apex_z = float(self.peak_z[env_id])
            apex_h = apex_z - float(self.takeoff_z[env_id])
            if z_ref is not None:
                ref_apex_z = float(self.peak_z_ref[env_id])
                msg = (
                    f"[HOP {self.hop_count:04d}] env={env_id} "
                    f"peak_z={apex_z:.3f}m height={apex_h:.3f}m | "
                    f"ref peak_z={ref_apex_z:.3f}m diff={apex_z - ref_apex_z:+.3f}m"
                )
            else:
                msg = f"[HOP {self.hop_count:04d}] env={env_id} peak_z={apex_z:.3f}m height={apex_h:.3f}m"
            print(msg)

        self.peak_z = torch.where(just_landed, torch.full_like(self.peak_z, -1e9), self.peak_z)
        self.peak_z_ref = torch.where(just_landed, torch.full_like(self.peak_z_ref, -1e9), self.peak_z_ref)
        self.in_air_prev = all_air


def _make_jump_tracker(env) -> _JumpTracker | None:
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "scene"):
        return None
    if "contact_forces" not in unwrapped.scene.sensors:
        print("[INFO] Jump tracker disabled: no 'contact_forces' sensor in scene.")
        return None
    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene.sensors["contact_forces"]
    sensor_body_names = contact_sensor.body_names

    base_candidates = ["base", "torso_link", "pelvis", "trunk"]
    base_idx = next((robot.body_names.index(n) for n in base_candidates if n in robot.body_names), None)
    if base_idx is None:
        print(f"[WARN] Jump tracker disabled: no base body among {base_candidates} in {robot.body_names}.")
        return None

    foot_candidates = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    foot_sensor_idx = [sensor_body_names.index(n) for n in foot_candidates if n in sensor_body_names]
    if not foot_sensor_idx:
        print(f"[INFO] Jump tracker disabled: no recognized foot bodies in {sensor_body_names}.")
        return None

    print(f"[INFO] Jump tracker: base='{robot.body_names[base_idx]}', feet={[sensor_body_names[i] for i in foot_sensor_idx]}.")
    return _JumpTracker(unwrapped, base_idx, foot_sensor_idx)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    # Align with mjlab: play mode is chosen by this entry script, while env-specific overrides live on the config.
    env_cfg = apply_play_overrides(env_cfg)
    env_cfg.scene.num_envs = 1

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
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

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

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

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if env_cfg_requires_motion(env_cfg):
        export_motion_policy_as_onnx(
            env.unwrapped,
            ppo_runner.alg.policy,
            path=export_model_dir,
            filename="policy.onnx",
        )
    else:
        export_policy_as_onnx(
            ppo_runner.alg.policy,
            path=export_model_dir,
            filename="policy.onnx",
        )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    # reset environment
    obs = env.get_observations()
    jump_tracker = _make_jump_tracker(env)
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if jump_tracker is not None:
            jump_tracker.update()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
