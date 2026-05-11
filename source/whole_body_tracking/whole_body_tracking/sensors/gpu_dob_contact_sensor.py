"""GPU-native disturbance-observer contact sensor using PhysX dynamics APIs.

Key differences from DobContactSensor (Pinocchio-based):
- All computation runs on GPU via batched PyTorch operations — no CPU transfer during update.
- Uses PhysX get_generalized_mass_matrices / get_coriolis_and_centrifugal_compensation_forces /
  get_gravity_compensation_forces / get_jacobians instead of Pinocchio CRBA/ABA/forwardKinematics.
- Base velocity convention: world frame (PhysX native) instead of body frame (Pinocchio FreeFlyer).
  Joint-space rows of tau_residual (used for contact force estimation) are unaffected by this
  difference to first order; base-joint inertia coupling does differ but is small at locomotion speeds.
- DOF column ordering is permuted to match joint_names (CSV / Pinocchio ordering) so the block
  extraction for contact forces is identical to the Pinocchio version.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from whole_body_tracking.robots.go2 import (
    GO2_CSV_JOINT_NAMES,
    GO2_FOOT_BODY_NAMES,
)

_FOOT_BODY_NAMES_DEFAULT = GO2_FOOT_BODY_NAMES


@dataclass
class GpuDobContactSensorData:
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


class GpuDobContactSensor:
    """Disturbance-observer contact force estimator that runs entirely on GPU.

    The sensor estimates per-foot contact forces by solving:
        J_block^T @ f = tau_residual_block
    where tau_residual = M @ (ddq_measured - ddq_free) and ddq_free = M^{-1} @ (tau - h).

    All dynamics quantities (M, h, J) are fetched from PhysX on GPU.  DOF columns of the
    PhysX generalized quantities are permuted to match ``joint_names`` ordering so that the
    block-index convention (block 1 = joints 6:9, block 2 = 9:12, …) is preserved.
    """

    # (body_name, block_index) — same convention as DobContactSensor
    FOOT_BLOCKS: tuple[tuple[str, int], ...] = (
        ("FL_foot", 1),
        ("FR_foot", 2),
        ("RL_foot", 3),
        ("RR_foot", 4),
    )

    def __init__(
        self,
        env,
        asset_name: str = "robot",
        joint_names: tuple[str, ...] = GO2_CSV_JOINT_NAMES,
        foot_body_names: tuple[str, ...] = _FOOT_BODY_NAMES_DEFAULT,
        foot_blocks: tuple[tuple[str, int], ...] = FOOT_BLOCKS,
        update_rate_hz: float = 50.0,
    ) -> None:
        self.env = env.unwrapped
        self.robot = env.unwrapped.scene[asset_name]
        self.device = self.robot.device
        self.num_envs = self.robot.data.joint_pos.shape[0]
        self.dt = env.unwrapped.step_dt
        self.update_interval_steps = max(1, math.ceil(1.0 / (update_rate_hz * self.dt)))
        self.sensor_dt = self.update_interval_steps * self.dt
        self.foot_blocks = foot_blocks

        # ------------------------------------------------------------------
        # DOF permutation: convert PhysX ordering → joint_names (CSV) order.
        # Isaac Lab's robot.data.joint_pos uses the same DOF ordering as PhysX.
        # find_joints returns indices that select joint_names in that ordering.
        # ------------------------------------------------------------------
        joint_ids, resolved = self.robot.find_joints(joint_names, preserve_order=True)
        if tuple(resolved) != tuple(joint_names):
            raise RuntimeError(f"GpuDobContactSensor: joint order mismatch: {resolved}")
        self._joint_ids = joint_ids  # list[int], length n_joints

        n_joints = len(joint_ids)
        # Full generalized coordinate permutation (base stays at 0:6):
        # perm[6 + k] = 6 + joint_ids[k]  for k in 0..n_joints-1
        perm = torch.cat([
            torch.arange(6, device=self.device, dtype=torch.long),
            torch.tensor(joint_ids, device=self.device, dtype=torch.long) + 6,
        ])
        self._perm = perm          # (6 + n_joints,)
        self._nv = 6 + n_joints    # 18 for Go2

        # ------------------------------------------------------------------
        # Foot link indices in PhysX Jacobian.
        # PhysX link ordering matches Isaac Lab body ordering.
        # For floating-base, root = index 0; subsequent links follow URDF tree.
        # ------------------------------------------------------------------
        physx_link_names = list(self.robot.root_physx_view.shared_metatype.link_names)
        self._foot_link_ids = torch.tensor(
            [physx_link_names.index(name) for name, _ in foot_blocks],
            device=self.device, dtype=torch.long,
        )
        self._num_feet = len(foot_blocks)

        # Previous generalized velocity (world frame, CSV-permuted order), for ddq finite diff
        self._prev_v = torch.zeros(self.num_envs, self._nv, device=self.device)

        self.last_update_step = -1
        self.last_compute_step = -self.update_interval_steps

        self.data = GpuDobContactSensorData(
            foot_forces_w=torch.zeros(self.num_envs, self._num_feet, 3, device=self.device),
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
            tau_residual=torch.zeros(self.num_envs, self._nv, device=self.device),
            valid=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )

    def reset(self, env_ids: torch.Tensor | list[int] | slice | None = None) -> None:
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

    def update(self, env_ids: torch.Tensor | list[int] | None = None, force: bool = False) -> None:
        reset_ids = (self.env.episode_length_buf == 0).nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            self.reset(reset_ids)

        if env_ids is None and not force and self.last_update_step == self.env.common_step_counter:
            return
        if env_ids is None:
            self.last_update_step = self.env.common_step_counter

        # Current generalised velocity in CSV joint ordering, world-frame base
        v_cur = self._get_generalized_velocity()  # (num_envs, nv)

        if env_ids is None:
            # Capture already-valid mask BEFORE warmup so newly initialized envs
            # are not included in this step's compute (prev_v == v_cur → ddq = 0).
            already_valid = self.data.valid.clone()

            warmup_ids = (~already_valid).nonzero(as_tuple=False).flatten()
            if warmup_ids.numel() > 0:
                self._prev_v[warmup_ids] = v_cur[warmup_ids]
                self.data.valid[warmup_ids] = True

            compute_due = force or (
                self.env.common_step_counter - self.last_compute_step >= self.update_interval_steps
            )
            if compute_due:
                compute_ids = already_valid.nonzero(as_tuple=False).flatten()
                if compute_ids.numel() > 0:
                    self._compute_contact(compute_ids, v_cur)
                self.last_compute_step = self.env.common_step_counter

            self._prev_v[:] = v_cur
        else:
            ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            already_valid = self.data.valid[ids_t].clone()

            warmup_ids = ids_t[~already_valid]
            if warmup_ids.numel() > 0:
                self._prev_v[warmup_ids] = v_cur[warmup_ids]
                self.data.valid[warmup_ids] = True

            compute_ids = ids_t[already_valid]
            if compute_ids.numel() > 0:
                self._compute_contact(compute_ids, v_cur)

            self._prev_v[ids_t] = v_cur[ids_t]

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_generalized_velocity(self) -> torch.Tensor:
        """Return (num_envs, nv) generalized velocity in world frame + CSV joint order."""
        v = torch.empty(self.num_envs, self._nv, device=self.device)
        v[:, :3] = self.robot.data.root_lin_vel_w   # world-frame linear
        v[:, 3:6] = self.robot.data.root_ang_vel_w  # world-frame angular
        v[:, 6:] = self.robot.data.joint_vel[:, self._joint_ids]  # CSV order
        return v

    def _compute_contact(self, env_ids: torch.Tensor, v_cur: torch.Tensor) -> None:
        """Compute contact forces and update self.data for the given env_ids."""
        physx = self.robot.root_physx_view
        p = self._perm  # (nv,) DOF permutation

        # ---- Dynamics quantities from PhysX (GPU, world frame) ----------------
        M_full = physx.get_generalized_mass_matrices()  # (num_envs_physx, nv, nv)
        h_full = (
            physx.get_coriolis_and_centrifugal_compensation_forces()
            + physx.get_gravity_compensation_forces()
        )  # (num_envs_physx, nv)
        J_full = physx.get_jacobians()  # (num_envs_physx, num_links, 6, nv)

        # Slice to requested envs and apply DOF permutation
        M = M_full[env_ids][:, p][:, :, p]   # (|ids|, nv, nv)
        h = h_full[env_ids][:, p]            # (|ids|, nv)
        J = J_full[env_ids][:, :, :, p]      # (|ids|, num_links, 6, nv)

        v = v_cur[env_ids]              # (|ids|, nv)  already CSV-permuted
        v_prev = self._prev_v[env_ids]  # (|ids|, nv)

        # Build tau: only joint rows are non-zero (base is unactuated)
        tau = torch.zeros_like(v)
        tau[:, 6:] = self.robot.data.applied_torque[env_ids][:, self._joint_ids]

        # ---- DOB residual -------------------------------------------------------
        ddq_measured = (v - v_prev) / self.dt                # (|ids|, nv)
        ddq_free = torch.linalg.solve(M, (tau - h))          # (|ids|, nv)
        tau_residual = torch.bmm(M, (ddq_measured - ddq_free).unsqueeze(-1)).squeeze(-1)  # (|ids|, nv)

        # ---- Foot contact forces via block Jacobian solve -----------------------
        n = env_ids.shape[0]
        foot_forces = torch.zeros(n, self._num_feet, 3, device=self.device)
        for foot_id, (_, block_index) in enumerate(self.foot_blocks):
            start = 6 + 3 * (block_index - 1)  # identical block convention to Pinocchio
            J_foot_lin = J[:, self._foot_link_ids[foot_id], :3, :]  # (n, 3, nv) linear Jacobian
            J_block = J_foot_lin[:, :, start : start + 3]            # (n, 3, 3)
            tau_block = tau_residual[:, start : start + 3]            # (n, 3)
            # Solve J_block^T @ f = tau_block  →  f = (J_block^T)^{-1} @ tau_block
            foot_forces[:, foot_id] = torch.linalg.solve(J_block.mT, tau_block)

        # ---- Foot velocity and contact power ------------------------------------
        J_feet_lin = J[:, self._foot_link_ids, :3, :]        # (n, num_feet, 3, nv)
        v_feet = torch.einsum("nfij,nj->nfi", J_feet_lin, v) # (n, num_feet, 3)
        contact_power = (foot_forces * v_feet).sum(dim=(-1, -2))  # (n,)

        total_forces = foot_forces.sum(dim=1)  # (n, 3)
        work = contact_power * self.sensor_dt

        # ---- Write back to data tensors -----------------------------------------
        self.data.foot_forces_w[env_ids] = foot_forces
        self.data.total_force_w[env_ids] = total_forces
        self.data.contact_power[env_ids] = contact_power
        self.data.tau_residual[env_ids] = tau_residual
        self.data.positive_work[env_ids] += work.clamp(min=0.0)
        self.data.negative_work[env_ids] += work.clamp(max=0.0)
        self.data.absolute_work[env_ids] += work.abs()
        self.data.hop_positive_work[env_ids] += work.clamp(min=0.0)
        self.data.hop_negative_work[env_ids] += work.clamp(max=0.0)
        self.data.hop_absolute_work[env_ids] += work.abs()
        self.data.peak_total_force_z[env_ids] = torch.maximum(
            self.data.peak_total_force_z[env_ids], total_forces[:, 2]
        )
        self.data.hop_peak_total_force_z[env_ids] = torch.maximum(
            self.data.hop_peak_total_force_z[env_ids], total_forces[:, 2]
        )
