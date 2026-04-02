import os
import warnings

import wandb
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx, export_policy_as_onnx


def _export_wandb_onnx(runner: OnPolicyRunner, path: str, *, motion: bool):
    if runner.logger_type not in ["wandb"]:
        return

    policy_path = path.split("model")[0]
    filename = policy_path.split("/")[-2] + ".onnx"
    obs_normalizer = getattr(runner, "obs_normalizer", None)
    try:
        if motion:
            export_motion_policy_as_onnx(
                runner.env.unwrapped,
                runner.alg.policy,
                normalizer=obs_normalizer,
                path=policy_path,
                filename=filename,
            )
        else:
            export_policy_as_onnx(
                runner.alg.policy,
                normalizer=obs_normalizer,
                path=policy_path,
                filename=filename,
            )
        attach_onnx_metadata(runner.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
        wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
    except Exception as exc:
        mode = "motion ONNX" if motion else "ONNX"
        warnings.warn(f"Skipping {mode} export for checkpoint '{path}' due to: {exc}")


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device="cpu",
        registry_name: str | None = None,
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
                self.env.unwrapped, self.alg.policy, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

        if self.logger_type in ["wandb"] and self.registry_name is not None:
            wandb.run.use_artifact(self.registry_name)
            self.registry_name = None
