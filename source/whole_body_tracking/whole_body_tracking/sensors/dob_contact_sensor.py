from __future__ import annotations

import math
import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pinocchio as pin
import torch

from whole_body_tracking.robots.go2 import GO2_CSV_JOINT_NAMES, GO2_URDF_PATH
from whole_body_tracking.utils.disturbance_observer import LagrangianDisturbanceObserver


@dataclass
class DobContactSensorData:
    foot_forces_w: torch.Tensor
    total_force_w: torch.Tensor
    contact_power: torch.Tensor
    positive_work: torch.Tensor
    negative_work: torch.Tensor
    absolute_work: torch.Tensor
    hop_positive_work: torch.Tensor
    hop_negative_work: torch.Tensor
    hop_absolute_work: torch.Tensor
    peak_total_force_z: torch.Tensor
    hop_peak_total_force_z: torch.Tensor
    tau_residual: torch.Tensor
    valid: torch.Tensor


class DobContactSensor:
    FOOT_BLOCKS = (("FL_foot", 1), ("FR_foot", 2), ("RL_foot", 3), ("RR_foot", 4))

    def __init__(
        self,
        env,
        asset_name: str = "robot",
        joint_names: tuple[str, ...] = GO2_CSV_JOINT_NAMES,
        urdf_path: str = GO2_URDF_PATH,
        foot_blocks: tuple[tuple[str, int], ...] = FOOT_BLOCKS,
        update_rate_hz: float = 50.0,
        num_workers: int | None = None,
    ):
        self.env = env.unwrapped
        self.robot = env.unwrapped.scene[asset_name]
        self.device = self.robot.device
        self.num_envs = self.robot.data.joint_pos.shape[0]
        self.dt = env.unwrapped.step_dt
        self.update_interval_steps = max(1, math.ceil(1.0 / (update_rate_hz * self.dt)))
        self.sensor_dt = self.update_interval_steps * self.dt
        self.foot_blocks = foot_blocks
        self.num_workers = max(1, min(num_workers or (os.cpu_count() or 1), self.num_envs))
        self.joint_ids, resolved_joint_names = self.robot.find_joints(joint_names, preserve_order=True)
        if tuple(resolved_joint_names) != tuple(joint_names):
            raise RuntimeError(f"DOB joint order mismatch: {resolved_joint_names}")

        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.observers = [
            LagrangianDisturbanceObserver(self.model, self.model.createData()) for _ in range(self.num_workers)
        ]
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers) if self.num_workers > 1 else None
        self.q = np.zeros((self.num_envs, self.model.nq))
        self.v = np.zeros((self.num_envs, self.model.nv))
        self.tau = np.zeros((self.num_envs, self.model.nv))
        self.previous_v = np.zeros((self.num_envs, self.model.nv))
        self.last_update_step = -1
        self.last_compute_step = -self.update_interval_steps

        self.data = DobContactSensorData(
            foot_forces_w=torch.zeros(self.num_envs, len(foot_blocks), 3, device=self.device),
            total_force_w=torch.zeros(self.num_envs, 3, device=self.device),
            contact_power=torch.zeros(self.num_envs, device=self.device),
            positive_work=torch.zeros(self.num_envs, device=self.device),
            negative_work=torch.zeros(self.num_envs, device=self.device),
            absolute_work=torch.zeros(self.num_envs, device=self.device),
            hop_positive_work=torch.zeros(self.num_envs, device=self.device),
            hop_negative_work=torch.zeros(self.num_envs, device=self.device),
            hop_absolute_work=torch.zeros(self.num_envs, device=self.device),
            peak_total_force_z=torch.zeros(self.num_envs, device=self.device),
            hop_peak_total_force_z=torch.zeros(self.num_envs, device=self.device),
            tau_residual=torch.zeros(self.num_envs, self.model.nv, device=self.device),
            valid=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )

    def __del__(self):
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.shutdown(wait=False)

    def reset(self, env_ids: torch.Tensor | list[int] | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.data.positive_work[env_ids] = 0.0
        self.data.negative_work[env_ids] = 0.0
        self.data.absolute_work[env_ids] = 0.0
        self.data.hop_positive_work[env_ids] = 0.0
        self.data.hop_negative_work[env_ids] = 0.0
        self.data.hop_absolute_work[env_ids] = 0.0
        self.data.peak_total_force_z[env_ids] = 0.0
        self.data.hop_peak_total_force_z[env_ids] = 0.0
        self.data.valid[env_ids] = False

    def update(self, env_ids: torch.Tensor | list[int] | None = None, force: bool = False):
        reset_ids = (self.env.episode_length_buf == 0).nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            self.reset(reset_ids)
        if env_ids is None and not force and self.last_update_step == self.env.common_step_counter:
            return
        if env_ids is None:
            self.last_update_step = self.env.common_step_counter

        if env_ids is None:
            env_ids_np = np.arange(self.num_envs)
            env_ids_torch = slice(None)
        else:
            env_ids_torch = env_ids
            env_ids_np = torch.as_tensor(env_ids, device="cpu").numpy() if isinstance(env_ids, torch.Tensor) else np.asarray(env_ids)

        self._read_state(env_ids_torch, env_ids_np)
        valid = self.data.valid[env_ids_torch].detach().cpu().numpy().astype(bool)
        warmup_ids = env_ids_np[~valid]
        if warmup_ids.size > 0:
            warmup_ids_t = torch.as_tensor(warmup_ids, device=self.device, dtype=torch.long)
            self.previous_v[warmup_ids, :] = self.v[warmup_ids]
            self.data.valid[warmup_ids_t] = True

        compute_due = force or (self.env.common_step_counter - self.last_compute_step >= self.update_interval_steps)
        compute_ids = env_ids_np[valid] if compute_due else np.empty(0, dtype=int)
        if compute_ids.size > 0:
            self._compute_contact(compute_ids)
            if env_ids is None:
                self.last_compute_step = self.env.common_step_counter

        self.previous_v[env_ids_np, :] = self.v[env_ids_np]

    def consume_hop_metrics(self, env_id: int, height_scale: float = 1.0) -> dict[str, float]:
        scale = max(height_scale, 1.0e-6)
        positive = float(self.data.hop_positive_work[env_id])
        negative = float(self.data.hop_negative_work[env_id])
        metrics = {
            "positive_work_per_height": positive / scale,
            "negative_work_per_height": negative / scale,
            "return_ratio": positive / max(-negative, 1.0e-6),
            "peak_total_force_z": float(self.data.hop_peak_total_force_z[env_id]),
        }
        self.data.hop_positive_work[env_id] = 0.0
        self.data.hop_negative_work[env_id] = 0.0
        self.data.hop_absolute_work[env_id] = 0.0
        self.data.hop_peak_total_force_z[env_id] = 0.0
        return metrics

    def episode_metrics(self, env_id: int) -> dict[str, float]:
        positive = float(self.data.positive_work[env_id])
        negative = float(self.data.negative_work[env_id])
        return {
            "positive_work": positive,
            "negative_work": negative,
            "return_ratio": positive / max(-negative, 1.0e-6),
            "peak_total_force_z": float(self.data.peak_total_force_z[env_id]),
        }

    def _read_state(self, env_ids_torch, env_ids_np: np.ndarray):
        root_quat_wxyz = self.robot.data.root_quat_w[env_ids_torch].detach().cpu().numpy()
        self.q[env_ids_np, :3] = self.robot.data.root_pos_w[env_ids_torch].detach().cpu().numpy()
        self.q[env_ids_np, 3:7] = np.roll(root_quat_wxyz, -1, axis=-1)
        self.q[env_ids_np, 7:] = self.robot.data.joint_pos[env_ids_torch][:, self.joint_ids].detach().cpu().numpy()

        self.v[env_ids_np, :3] = self.robot.data.root_lin_vel_b[env_ids_torch].detach().cpu().numpy()
        self.v[env_ids_np, 3:6] = self.robot.data.root_ang_vel_b[env_ids_torch].detach().cpu().numpy()
        self.v[env_ids_np, 6:] = self.robot.data.joint_vel[env_ids_torch][:, self.joint_ids].detach().cpu().numpy()

        self.tau[env_ids_np, :] = 0.0
        self.tau[env_ids_np, 6:] = self.robot.data.applied_torque[env_ids_torch][:, self.joint_ids].detach().cpu().numpy()

    def _compute_contact(self, env_ids: np.ndarray):
        chunks = [chunk for chunk in np.array_split(env_ids, min(self.num_workers, env_ids.size)) if chunk.size > 0]
        if len(chunks) == 1:
            results = [self._compute_chunk(chunks[0], self.observers[0])]
        else:
            results = list(
                self.executor.map(
                    lambda args: self._compute_chunk(*args),
                    ((chunk, self.observers[i]) for i, chunk in enumerate(chunks)),
                )
            )

        for env_ids_chunk, foot_forces, total_forces, contact_power, tau_residual in results:
            env_ids_t = torch.as_tensor(env_ids_chunk, device=self.device, dtype=torch.long)
            foot_forces_t = torch.as_tensor(foot_forces, device=self.device, dtype=torch.float32)
            total_forces_t = torch.as_tensor(total_forces, device=self.device, dtype=torch.float32)
            contact_power_t = torch.as_tensor(contact_power, device=self.device, dtype=torch.float32)
            tau_residual_t = torch.as_tensor(tau_residual, device=self.device, dtype=torch.float32)
            work = contact_power_t * self.sensor_dt

            self.data.foot_forces_w[env_ids_t] = foot_forces_t
            self.data.total_force_w[env_ids_t] = total_forces_t
            self.data.contact_power[env_ids_t] = contact_power_t
            self.data.tau_residual[env_ids_t] = tau_residual_t
            self.data.positive_work[env_ids_t] += torch.clamp(work, min=0.0)
            self.data.negative_work[env_ids_t] += torch.clamp(work, max=0.0)
            self.data.absolute_work[env_ids_t] += torch.abs(work)
            self.data.hop_positive_work[env_ids_t] += torch.clamp(work, min=0.0)
            self.data.hop_negative_work[env_ids_t] += torch.clamp(work, max=0.0)
            self.data.hop_absolute_work[env_ids_t] += torch.abs(work)
            self.data.peak_total_force_z[env_ids_t] = torch.maximum(
                self.data.peak_total_force_z[env_ids_t], total_forces_t[:, 2]
            )
            self.data.hop_peak_total_force_z[env_ids_t] = torch.maximum(
                self.data.hop_peak_total_force_z[env_ids_t], total_forces_t[:, 2]
            )

    def _compute_chunk(self, env_ids: np.ndarray, observer: LagrangianDisturbanceObserver):
        foot_forces = np.zeros((env_ids.size, len(self.foot_blocks), 3))
        total_forces = np.zeros((env_ids.size, 3))
        contact_power = np.zeros(env_ids.size)
        tau_residuals = np.zeros((env_ids.size, self.model.nv))

        for local_id, env_id in enumerate(env_ids):
            tau_residual = observer.compute_residual_from_velocity_difference(
                self.q[env_id],
                self.v[env_id],
                self.tau[env_id],
                self.previous_v[env_id],
                self.dt,
            )
            tau_residuals[local_id] = tau_residual

            for foot_id, (frame_name, block_index) in enumerate(self.foot_blocks):
                force = observer.estimate_contact_force_from_residual_block(
                    self.q[env_id],
                    tau_residual,
                    frame_name,
                    block_index,
                )
                foot_velocity = observer.compute_frame_position_jacobian(self.q[env_id], frame_name) @ self.v[env_id]
                foot_forces[local_id, foot_id] = force
                total_forces[local_id] += force
                contact_power[local_id] += float(force @ foot_velocity)

        return env_ids, foot_forces, total_forces, contact_power, tau_residuals
