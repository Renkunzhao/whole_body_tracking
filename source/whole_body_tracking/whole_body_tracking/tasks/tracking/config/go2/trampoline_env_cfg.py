from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.config.go2.flat_env_cfg import (
    Go2FlatEnvCfg,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg import ActionsCfg as BaseActionsCfg
from whole_body_tracking.tasks.tracking.tracking_env_cfg import EventCfg as BaseEventCfg
from whole_body_tracking.utils.trampoline_deformable import (
    TRAMPOLINE_DR_MASS_RANGE,
    TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE,
    make_trampoline_cfg,
)


@configclass
class Go2TrampolineSceneCfg(InteractiveSceneCfg):
    """Scene config for the deformable Go2 trampoline tracking task."""

    robot: ArticulationCfg = MISSING
    trampoline: DeformableObjectCfg = make_trampoline_cfg("{ENV_REGEX_NS}/Trampoline")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )


@configclass
class Go2TrampolineActionsCfg(BaseActionsCfg):
    trampoline_pin = mdp.TrampolinePinningActionCfg(asset_name="trampoline")


@configclass
class Go2TrampolineEventCfg(BaseEventCfg):
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    randomize_trampoline_properties = EventTerm(
        func=mdp.RandomizeTrampolineProperties,
        mode="reset",
        params={
            "asset_name": "trampoline",
            "youngs_modulus_range": TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE,
            "mass_range": TRAMPOLINE_DR_MASS_RANGE,
        },
    )
    reapply_trampoline_pinning = EventTerm(func=mdp.reapply_trampoline_pinning, mode="reset")


@configclass
class Go2TrampolineEnvCfg(Go2FlatEnvCfg):
    scene: Go2TrampolineSceneCfg = Go2TrampolineSceneCfg(num_envs=256, env_spacing=4.0, replicate_physics=False)
    actions: Go2TrampolineActionsCfg = Go2TrampolineActionsCfg()
    events: Go2TrampolineEventCfg = Go2TrampolineEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.foot_contact.contact_enabled = False


@configclass
class Go2TrampolineWoStateEstimationEnvCfg(Go2TrampolineEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
