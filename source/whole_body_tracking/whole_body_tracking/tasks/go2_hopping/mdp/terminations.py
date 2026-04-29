from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class apex_reached(ManagerTermBase):
    """Terminates when the robot reaches the apex of its rebound.

    Detects the ``vz`` positive-to-non-positive transition gated on the
    rebounce command's ``has_descended`` flag, so the t=0 instant (where vz
    starts at 0 and dips below threshold on the first sim step) cannot fire
    spuriously. ``command_name`` must point at a :class:`UniformRebounceCommand`.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_vz = torch.zeros(env.num_envs, device=env.device)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset_name = asset_cfg.name
        foot_asset_cfg: SceneEntityCfg | None = cfg.params.get("foot_asset_cfg")
        self._foot_asset_name = None if foot_asset_cfg is None else foot_asset_cfg.name
        self._foot_body_ids = None if foot_asset_cfg is None else foot_asset_cfg.body_ids

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        vz_threshold: float = 0.0,
        foot_asset_cfg: SceneEntityCfg | None = None,
        foot_clearance: float | None = None,
        surface_z: float = 0.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        del asset_cfg, foot_asset_cfg  # resolved in __init__
        robot: Articulation = env.scene[self._asset_name]
        vz = robot.data.root_lin_vel_w[:, 2]
        cmd = env.command_manager.get_term(command_name)
        # Apex: was clearly ascending (prev_vz > threshold) and is no longer.
        # The has_descended gate prevents the t=0 instant — when prev_vz is the
        # init zero and the first sim step makes vz <= 0 — from firing.
        apex = cmd.has_descended & cmd.has_rebounded & (self._prev_vz > vz_threshold) & (vz <= vz_threshold)
        if foot_clearance is not None and self._foot_asset_name is not None:
            foot_asset: Articulation = env.scene[self._foot_asset_name]
            foot_z_local = foot_asset.data.body_pos_w[:, self._foot_body_ids, 2] - env.scene.env_origins[:, 2:3]
            min_foot_z = torch.min(foot_z_local, dim=1).values
            apex = apex & (min_foot_z > surface_z + foot_clearance)
        self._prev_vz = vz.clone()
        return apex

    def reset(self, env_ids=None):
        if env_ids is None:
            self._prev_vz.zero_()
        else:
            self._prev_vz[env_ids] = 0.0


class no_airborne_apex_timeout(ManagerTermBase):
    """Terminates when no airborne apex has occurred for too long.

    An airborne apex is a root vertical velocity transition from upward to
    non-upward motion while the feet are above the support surface. This keeps
    continuous rebounce training from settling into a stable standing policy
    that only waits for the episode timeout.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_vz = torch.zeros(env.num_envs, device=env.device)
        self._time_since_apex = torch.zeros(env.num_envs, device=env.device)
        self._apex_armed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset_name = asset_cfg.name
        foot_asset_cfg: SceneEntityCfg | None = cfg.params.get("foot_asset_cfg")
        if foot_asset_cfg is not None:
            foot_asset_cfg.resolve(env.scene)
        self._foot_asset_name = None if foot_asset_cfg is None else foot_asset_cfg.name
        self._foot_body_ids = None if foot_asset_cfg is None else foot_asset_cfg.body_ids

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        timeout: float,
        foot_asset_cfg: SceneEntityCfg | None = None,
        foot_clearance: float | None = None,
        surface_z: float = 0.0,
        vz_threshold: float = 0.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        del asset_cfg, foot_asset_cfg  # resolved in __init__
        robot: Articulation = env.scene[self._asset_name]
        vz = robot.data.root_lin_vel_w[:, 2]

        apex = (self._prev_vz > vz_threshold) & (vz <= vz_threshold)
        foot_airborne = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        foot_near = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        if foot_clearance is not None and self._foot_asset_name is not None:
            foot_asset: Articulation = env.scene[self._foot_asset_name]
            foot_z_local = foot_asset.data.body_pos_w[:, self._foot_body_ids, 2] - env.scene.env_origins[:, 2:3]
            min_foot_z = torch.min(foot_z_local, dim=1).values
            foot_threshold = surface_z + foot_clearance
            foot_airborne = min_foot_z > foot_threshold
            foot_near = min_foot_z <= foot_threshold
        self._apex_armed = self._apex_armed | foot_near
        apex = self._apex_armed & apex & foot_airborne

        self._time_since_apex += env.step_dt
        self._time_since_apex = torch.where(apex, torch.zeros_like(self._time_since_apex), self._time_since_apex)
        self._apex_armed = self._apex_armed & ~apex
        timed_out = self._time_since_apex > timeout
        self._prev_vz = vz.clone()
        return timed_out

    def reset(self, env_ids=None):
        if env_ids is None:
            self._prev_vz.zero_()
            self._time_since_apex.zero_()
            self._apex_armed.fill_(True)
        else:
            self._prev_vz[env_ids] = 0.0
            self._time_since_apex[env_ids] = 0.0
            self._apex_armed[env_ids] = True
