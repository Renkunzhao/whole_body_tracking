from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class apex_reached(ManagerTermBase):
    """Terminates on the command-owned valid-apex event."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
    ) -> torch.Tensor:
        cmd = env.command_manager.get_term(command_name)
        return cmd.is_apex


class no_valid_apex_timeout(ManagerTermBase):
    """Terminates when no valid apex has occurred for too long.

    The command term owns the apex state machine. This termination only tracks
    elapsed time since the last command-owned valid-apex pulse.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._time_since_valid_apex = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        timeout: float,
        command_name: str = "hop",
    ) -> torch.Tensor:
        cmd = env.command_manager.get_term(command_name)

        self._time_since_valid_apex += env.step_dt
        self._time_since_valid_apex = torch.where(
            cmd.is_apex,
            torch.zeros_like(self._time_since_valid_apex),
            self._time_since_valid_apex,
        )
        return self._time_since_valid_apex > timeout

    def reset(self, env_ids=None):
        if env_ids is None:
            self._time_since_valid_apex.zero_()
        else:
            self._time_since_valid_apex[env_ids] = 0.0
