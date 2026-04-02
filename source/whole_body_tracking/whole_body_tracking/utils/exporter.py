# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os

import onnx
import torch

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

from whole_body_tracking.tasks.tracking.mdp import MotionCommand


def export_policy_as_onnx(
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose: bool = False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _StableOnnxPolicyExporter(actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose: bool = False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _StableOnnxPolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic, normalizer=None, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.forward = self.forward_lstm
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h_out, c_out) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h_out, c_out

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path: str, filename: str):
        self.to("cpu")
        was_training = self.training
        self.eval()
        try:
            if self.is_recurrent:
                obs = torch.zeros(1, self.rnn.input_size)
                h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                torch.onnx.export(
                    self,
                    (obs, h_in, c_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "c_in"],
                    output_names=["actions", "h_out", "c_out"],
                    dynamic_axes={},
                    dynamo=False,
                )
            else:
                obs = torch.zeros(1, self.actor[0].in_features)
                torch.onnx.export(
                    self,
                    obs,
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs"],
                    output_names=["actions"],
                    dynamic_axes={},
                    dynamo=False,
                )
        finally:
            if was_training:
                self.train()


class _OnnxMotionPolicyExporter(_StableOnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose: bool = False):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion")

        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(x)),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path: str, filename: str):
        self.to("cpu")
        was_training = self.training
        self.eval()
        obs = torch.zeros(1, self.actor[0].in_features)
        time_step = torch.zeros(1, 1)
        try:
            torch.onnx.export(
                self,
                (obs, time_step),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "time_step"],
                output_names=[
                    "actions",
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                ],
                dynamic_axes={},
                dynamo=False,
            )
        finally:
            if was_training:
                self.train()


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr)


def _legacy_manager_metadata(env: ManagerBasedRLEnv, run_path: str) -> dict[str, object]:
    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    robot = env.scene["robot"]
    default_joint_pos = getattr(robot.data, "default_joint_pos_nominal", robot.data.default_joint_pos[0]).cpu().tolist()
    metadata: dict[str, object] = {
        "run_path": run_path,
        "joint_names": robot.data.joint_names,
        "joint_stiffness": robot.data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": robot.data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": default_joint_pos,
        "command_names": list(env.command_manager.active_terms),
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
    }

    if "motion" in env.command_manager.active_terms:
        motion_term = env.command_manager.get_term("motion")
        metadata["anchor_body_name"] = motion_term.cfg.anchor_body_name
        metadata["body_names"] = motion_term.cfg.body_names

    return metadata


def attach_onnx_metadata(
    env: ManagerBasedRLEnv | DirectRLEnv,
    run_path: str,
    path: str,
    filename="policy.onnx",
) -> None:
    onnx_path = os.path.join(path, filename)

    if hasattr(env, "get_export_metadata"):
        metadata = env.get_export_metadata(run_path)
    elif isinstance(env, ManagerBasedRLEnv):
        metadata = _legacy_manager_metadata(env, run_path)
    else:
        raise TypeError(f"Unsupported environment type for ONNX metadata export: {type(env)!r}")

    model = onnx.load(onnx_path)

    for key, value in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = key
        entry.value = list_to_csv_str(value) if isinstance(value, list) else str(value)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
