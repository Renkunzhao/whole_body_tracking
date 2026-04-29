from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_drop_from_height(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str = "hop",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    drop_z_offset: float = 0.0,
    drop_height_range: tuple[float, float] | None = None,
) -> None:
    """Reset the robot dropped from a sampled height for the rebounce task.

    Samples ``h_cmd ~ U(cmd.cfg.ranges.peak_height)`` per env, writes it into
    the rebounce command's target buffer, then teleports the robot to an
    independently sampled drop height. Because event-mode resets run before
    ``command_manager.reset()``, this is the single place that samples the
    per-episode target — the command term's ``_resample_command`` is
    intentionally a no-op.

    Args:
        drop_z_offset: Added to the sampled drop height to set the base z.
        drop_height_range: Optional independent sampling range for the initial
            drop height. If omitted, the reset falls back to the old behavior
            ``h_drop = h_cmd``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    n_ids = len(env_ids)
    dev = asset.device

    # Sample h_cmd from the command's configured range and write to the
    # command's per-env target buffer (the source of truth read by obs and
    # downstream rewards / metrics).
    h_lo, h_hi = cmd.cfg.ranges.peak_height
    h_cmd = torch.empty(n_ids, device=dev).uniform_(h_lo, h_hi)
    cmd._peak_h_target[env_ids] = h_cmd
    if drop_height_range is None:
        h_drop = h_cmd
    else:
        h_drop = torch.empty(n_ids, device=dev).uniform_(*drop_height_range)

    # Build root state from the asset's default, shifted into the env's local
    # frame, with z overridden by the sampled drop height in that env frame.
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, 0:3] = root_state[:, 0:3] + env.scene.env_origins[env_ids]
    root_state[:, 2] = env.scene.env_origins[env_ids, 2] + drop_z_offset + h_drop
    root_state[:, 7:13] = 0.0  # zero linear and angular velocity

    asset.write_root_pose_to_sim(root_state[:, 0:7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)

    # Default joint pose and zero joint velocity. A pre-landing crouch could
    # be substituted here later if the default standing pose causes harsh
    # initial impacts at low h_cmd.
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
