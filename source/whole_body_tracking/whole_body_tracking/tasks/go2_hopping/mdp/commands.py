from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class HoppingMetricsCommand(CommandTerm):
    """Tracks per-hop air time and peak base height and logs them to wandb.

    A "hop" is the interval during which all monitored feet are off the ground.
    At the air->contact transition the current accumulators are latched into
    per-env buffers; these persist between hops so the last completed hop is
    always readable.

    Logging is performed every env step by writing cross-env means into
    ``env.extras["log"]``; RSL-RL appends that dict each step so one wandb
    point per iteration is the mean over (mean over all envs) across
    ``num_steps_per_env``.

    This term emits no actual command; ``command`` returns an empty tensor.
    """

    cfg: HoppingMetricsCommandCfg

    def __init__(self, cfg: HoppingMetricsCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_cfg.name]
        cfg.sensor_cfg.resolve(env.scene)
        self.sensor: ContactSensor = env.scene.sensors[cfg.sensor_cfg.name]
        self._body_ids = cfg.sensor_cfg.body_ids

        n, dev = self.num_envs, self.device
        self._in_air = torch.zeros(n, dtype=torch.bool, device=dev)
        self._air_time = torch.zeros(n, device=dev)
        self._peak_z = torch.zeros(n, device=dev)
        self._last_air_time = torch.zeros(n, device=dev)
        self._last_peak_z = torch.zeros(n, device=dev)

    @property
    def command(self) -> torch.Tensor:
        return torch.empty(self.num_envs, 0, device=self.device)

    def _update_metrics(self):
        forces = self.sensor.data.net_forces_w_history[:, :, self._body_ids, :]
        contacts = torch.max(torch.norm(forces, dim=-1), dim=1)[0] > self.cfg.contact_threshold
        all_air = ~contacts.any(dim=1)
        z = self.robot.data.root_pos_w[:, 2]

        self._air_time = torch.where(all_air, self._air_time + self._env.step_dt, self._air_time)
        self._peak_z = torch.where(all_air, torch.maximum(self._peak_z, z), self._peak_z)

        just_landed = self._in_air & ~all_air
        self._last_air_time = torch.where(just_landed, self._air_time, self._last_air_time)
        self._last_peak_z = torch.where(just_landed, self._peak_z, self._last_peak_z)
        zero = torch.zeros_like(self._air_time)
        self._air_time = torch.where(just_landed, zero, self._air_time)
        self._peak_z = torch.where(just_landed, zero, self._peak_z)
        self._in_air = all_air

        log = self._env.extras.setdefault("log", {})
        log["Metrics/hop/peak_height"] = self._last_peak_z.mean().item()
        log["Metrics/hop/air_time"] = self._last_air_time.mean().item()

    def _resample_command(self, env_ids: Sequence[int]):
        self._in_air[env_ids] = False
        self._air_time[env_ids] = 0.0
        self._peak_z[env_ids] = 0.0
        self._last_air_time[env_ids] = 0.0
        self._last_peak_z[env_ids] = 0.0

    def _update_command(self):
        pass


@configclass
class HoppingMetricsCommandCfg(CommandTermCfg):
    class_type: type = HoppingMetricsCommand
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
    contact_threshold: float = 5.0
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
