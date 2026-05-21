import os
import time
from collections import deque
from typing import Any

import numpy as np
import torch

from rsl_rl.env import VecEnv
from rsl_rl.runners import DistillationRunner
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import store_code_state

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - dependency is expected in the IsaacLab env
    imageio = None

from whole_body_tracking.utils.exporter import (
    attach_onnx_metadata,
    export_motion_policy_as_onnx,
    get_policy_export_normalizer,
)


class CheckpointVideoMixin:
    def configure_checkpoint_video(
        self,
        *,
        enabled: bool,
        output_dir: str,
        length_s: float,
        fps: int,
        wandb_key: str = "checkpoint_eval/video",
    ) -> None:
        self.checkpoint_video_cfg = {
            "enabled": enabled,
            "output_dir": output_dir,
            "length_s": length_s,
            "fps": fps,
            "wandb_key": wandb_key,
        }

    def _get_checkpoint_video_render(self):
        """Return the env render() callable if checkpoint video recording is ready, else None."""
        cfg = getattr(self, "checkpoint_video_cfg", None)
        if not cfg or not cfg["enabled"]:
            return None
        if imageio is None:
            print("[WARN]: imageio is unavailable; skipping checkpoint video.", flush=True)
            return None
        render = getattr(getattr(self.env, "env", None), "render", None)
        if render is None:
            print("[WARN]: Training env does not expose render(); skipping checkpoint video.", flush=True)
            return None
        return render

    def _record_checkpoint_video(self, iteration: int) -> bool:
        """Roll out the current policy and save a fixed-length eval video. Returns True if recorded."""
        render = self._get_checkpoint_video_render()
        if render is None:
            return False
        cfg = self.checkpoint_video_cfg

        os.makedirs(cfg["output_dir"], exist_ok=True)
        video_path = os.path.join(cfg["output_dir"], f"checkpoint_eval_{iteration:06d}.mp4")
        step_dt = float(getattr(self.env.unwrapped, "step_dt", 1.0 / max(cfg["fps"], 1)))
        rollout_steps = max(1, int(round(cfg["length_s"] / step_dt)))
        frame_stride = max(1, int(round(1.0 / max(cfg["fps"] * step_dt, 1.0e-9))))
        policy = self.get_inference_policy(device=self.device)

        print(
            f"[INFO]: Recording checkpoint eval video iteration={iteration} "
            f"steps={rollout_steps} path= {video_path}",
            flush=True,
        )
        self.eval_mode()
        frame_count = 0
        writer = imageio.get_writer(video_path, fps=cfg["fps"], macro_block_size=1)
        try:
            with torch.inference_mode():
                obs, _ = self.env.reset()
                obs = obs.to(self.device)
                for step in range(rollout_steps):
                    actions = policy(obs)
                    obs, _, _, _ = self.env.step(actions.to(self.env.device))
                    obs = obs.to(self.device)
                    if step % frame_stride == 0:
                        frame = render()
                        if frame is not None:
                            writer.append_data(_as_numpy_rgb(frame))
                            frame_count += 1
        finally:
            writer.close()
            self.train_mode()

        if frame_count == 0:
            print(f"[WARN]: No frames were rendered for checkpoint eval video: {video_path}", flush=True)
        elif self.logger_type == "wandb" and not self.disable_logs:
            wandb.log(
                {
                    cfg["wandb_key"]: wandb.Video(video_path, fps=cfg["fps"], format="mp4"),
                    "checkpoint_eval/iteration": iteration,
                },
                step=iteration,
            )
        with torch.inference_mode():
            self.env.reset()
        return True

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # Mirrors rsl_rl.runners.OnPolicyRunner.learn; the only added lines are tagged
        # "checkpoint-video addition" so they are easy to keep in sync with upstream.
        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                    # checkpoint-video addition: refresh obs only if a rollout actually ran.
                    if self._record_checkpoint_video(it):
                        obs = self.env.get_observations().to(self.device)

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
            # checkpoint-video addition: record one final eval video after the last save.
            self._record_checkpoint_video(self.current_learning_iteration)


def _as_numpy_rgb(frame: Any) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        array = frame.detach().cpu().numpy()
    else:
        array = np.asarray(frame)
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array, 0.0, 255.0)
        array = array.astype(np.uint8)
    return np.ascontiguousarray(array)


class MyOnPolicyRunner(CheckpointVideoMixin, OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(
                self.alg.policy,
                path=policy_path,
                normalizer=get_policy_export_normalizer(self.alg.policy),
                filename=filename,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MyDistillationRunner(DistillationRunner):
    def save(self, path: str, infos=None):
        """Save the distilled student and export a deployable ONNX policy."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(
                self.alg.policy,
                path=policy_path,
                normalizer=get_policy_export_normalizer(self.alg.policy),
                filename=filename,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(CheckpointVideoMixin, OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped,
                self.alg.policy,
                path=policy_path,
                normalizer=get_policy_export_normalizer(self.alg.policy),
                filename=filename,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
