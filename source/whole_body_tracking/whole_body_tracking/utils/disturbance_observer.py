from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np
import pinocchio as pin


ArrayLike = Union[np.ndarray, Sequence[float]]


class LagrangianDisturbanceObserver:
    def __init__(self, model: pin.Model, data: pin.Data) -> None:
        self.model = model
        self.data = data
        self.nq = model.nq
        self.nv = model.nv
        self.M = np.zeros((self.nv, self.nv))
        self.h = np.zeros(self.nv)
        self.ddq_free = np.zeros(self.nv)
        self.tau_residual = np.zeros(self.nv)
        self.tau_external = np.zeros(self.nv)

    def compute_lagrangian_terms(self, q: ArrayLike, v: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        q = self._vec(q, self.nq, "q")
        v = self._vec(v, self.nv, "v")
        self.M[:, :] = pin.crba(self.model, self.data, q)
        self.M[:, :] = 0.5 * (self.M + self.M.T)
        self.h[:] = pin.nonLinearEffects(self.model, self.data, q, v)
        return self.M, self.h

    def compute_residual_from_velocity_difference(
        self,
        q: ArrayLike,
        v: ArrayLike,
        tau: ArrayLike,
        v_previous: ArrayLike,
        dt: float,
    ) -> np.ndarray:
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        v = self._vec(v, self.nv, "v")
        v_previous = self._vec(v_previous, self.nv, "v_previous")
        ddq_measured = (v - v_previous) / dt
        return self.compute_residual(q, v, tau, ddq_measured)

    def compute_residual(self, q: ArrayLike, v: ArrayLike, tau: ArrayLike, ddq_measured: ArrayLike) -> np.ndarray:
        tau = self._vec(tau, self.nv, "tau")
        ddq_measured = self._vec(ddq_measured, self.nv, "ddq_measured")
        M, h = self.compute_lagrangian_terms(q, v)
        self.ddq_free[:] = np.linalg.solve(M, tau - h)
        self.tau_residual[:] = M @ (ddq_measured - self.ddq_free)
        return self.tau_residual

    def compute_frame_position_jacobian(
        self,
        q: ArrayLike,
        frame_name: str,
        reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    ) -> np.ndarray:
        q = self._vec(q, self.nq, "q")
        frame_id = self._frame_id(frame_name)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.framesForwardKinematics(self.model, self.data, q)
        J6 = pin.getFrameJacobian(self.model, self.data, frame_id, reference_frame)
        return np.asarray(J6[:3, :], dtype=float)

    def estimate_contact_forces_from_residual_blocks(
        self,
        q: ArrayLike,
        tau_residual: ArrayLike,
        frame_blocks: Sequence[tuple[str, int]],
        base_nv: int = 6,
        reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    ) -> dict[str, np.ndarray]:
        return {
            frame_name: self.estimate_contact_force_from_residual_block(
                q, tau_residual, frame_name, block_index, base_nv, reference_frame
            )
            for frame_name, block_index in frame_blocks
        }

    def estimate_contact_force_from_residual_block(
        self,
        q: ArrayLike,
        tau_residual: ArrayLike,
        frame_name: str,
        block_index: int,
        base_nv: int = 6,
        reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    ) -> np.ndarray:
        tau_residual = self._vec(tau_residual, self.nv, "tau_residual")
        start = self._contact_block_start(block_index, base_nv)
        J = self.compute_frame_position_jacobian(q, frame_name, reference_frame)
        return np.linalg.solve(J[:, start : start + 3].T, tau_residual[start : start + 3])

    def _contact_block_start(self, block_index: int, base_nv: int) -> int:
        if block_index < 1:
            raise ValueError("block_index starts from 1.")
        start = base_nv + 3 * (block_index - 1)
        if start + 3 > self.nv:
            raise ValueError(f"block_index {block_index} maps outside nv={self.nv}.")
        return start

    def _frame_id(self, frame_name: str) -> int:
        if not self.model.existFrame(frame_name):
            raise ValueError(f"Unknown frame name: {frame_name}")
        return self.model.getFrameId(frame_name)

    @staticmethod
    def _vec(x: ArrayLike, size: int, name: str) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != size:
            raise ValueError(f"{name} must have length {size}, got {x.size}.")
        return x
