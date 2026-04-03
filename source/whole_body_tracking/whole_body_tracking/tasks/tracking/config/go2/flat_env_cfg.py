from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.go2 import (
    GO2_ACTION_SCALE,
    GO2_FOOT_BODY_NAMES,
    GO2_NON_FOOT_CONTACT_BODY_NAMES,
    GO2_TRACKING_ANCHOR_BODY_NAME,
    GO2_TRACKING_BODY_NAMES,
    get_go2_cfg,
    get_go2_spawn_cfg,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


GO2_TRACKING_CFG = get_go2_cfg(
    spawn=get_go2_spawn_cfg(
        enabled_self_collisions=False,
        max_depenetration_velocity=5.0,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
        contact_offset=0.005,
        rest_offset=0.0,
        enable_gyroscopic_forces=True,
    )
)


@configclass
class Go2FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = GO2_TRACKING_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
        self.viewer.body_name = GO2_TRACKING_ANCHOR_BODY_NAME

        self.actions.joint_pos.scale = GO2_ACTION_SCALE

        self.commands.motion.anchor_body_name = GO2_TRACKING_ANCHOR_BODY_NAME
        self.commands.motion.body_names = list(GO2_TRACKING_BODY_NAMES)

        self.events.base_com.params['asset_cfg'] = SceneEntityCfg('robot', body_names=GO2_TRACKING_ANCHOR_BODY_NAME)
        self.events.physics_material.params['asset_cfg'] = SceneEntityCfg('robot', body_names=list(GO2_FOOT_BODY_NAMES))
        self.events.physics_material.params['static_friction_range'] = (0.3, 1.2)
        self.events.physics_material.params['dynamic_friction_range'] = (0.3, 1.2)
        self.events.physics_material.params['restitution_range'] = (0.0, 0.0)
        self.events.physics_material.params['make_consistent'] = True

        self.rewards.motion_global_anchor_pos.weight = 1.5
        self.rewards.motion_global_anchor_pos.params['std'] = 0.2
        self.rewards.motion_global_anchor_ori.weight = 1.5
        self.rewards.motion_body_pos.weight = 1.5
        self.rewards.motion_body_pos.params['std'] = 0.2
        self.rewards.motion_body_ang_vel.weight = 2.0
        self.rewards.motion_body_ang_vel.params['std'] = 6.28
        # mjlab's Go2 setup keeps the self-collision cost effectively inactive for this task.
        # Penalizing all non-foot PhysX contacts is harsher and tends to make aerial phases too conservative.
        self.rewards.undesired_contacts.weight = 0.0
        self.rewards.undesired_contacts.params['sensor_cfg'] = SceneEntityCfg(
            'contact_forces', body_names=list(GO2_NON_FOOT_CONTACT_BODY_NAMES)
        )

        self.terminations.anchor_pos.params['threshold'] = 0.5
        self.terminations.ee_body_pos.params['threshold'] = 0.6
        self.terminations.ee_body_pos.params['body_names'] = list(GO2_FOOT_BODY_NAMES)

    def apply_play_overrides(self):
        self.episode_length_s = int(1e9)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.commands.motion.pose_range = {}
        self.commands.motion.velocity_range = {}
        self.commands.motion.sampling_mode = "start"
        self.commands.motion.debug_vis = False
        self.scene.contact_forces.debug_vis = False
        # Keep the play camera static so manual mouse control is not overridden by asset tracking.
        self.viewer.origin_type = "world"
        return self


@configclass
class Go2FlatWoStateEstimationEnvCfg(Go2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
