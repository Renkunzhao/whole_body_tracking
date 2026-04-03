from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.go2 import (
    GO2_CSV_JOINT_NAMES,
    GO2_FOOT_BODY_NAMES,
    GO2_NON_FOOT_CONTACT_BODY_NAMES,
    GO2_TRACKING_ANCHOR_BODY_NAME,
    get_go2_actuators,
    get_go2_cfg,
    get_go2_init_state,
    get_go2_spawn_cfg,
)
from whole_body_tracking.utils.trampoline_deformable import (
    TRAMPOLINE_PIN_WIDTH,
    TRAMPOLINE_RADIUS,
    TRAMPOLINE_THICKNESS,
    TRAMPOLINE_TOP_Z,
    make_trampoline_cfg,
)


def _ordered_action_scale(action_scale_map: dict[str, float]) -> tuple[float, ...]:
    return tuple(float(action_scale_map[joint_name]) for joint_name in GO2_CSV_JOINT_NAMES)


GO2_HOPPING_DEFAULT_JOINT_POS = {
    "FL_hip_joint": 0.1,
    "RL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "FR_thigh_joint": 0.8,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,
    "RL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RR_calf_joint": -1.5,
}
GO2_HOPPING_ACTION_SCALE = 0.25
GO2_HOPPING_TERMINATION_BODY_NAMES = (GO2_TRACKING_ANCHOR_BODY_NAME,)
GO2_HOPPING_PENALIZED_CONTACT_BODY_NAMES = tuple(
    name for name in GO2_NON_FOOT_CONTACT_BODY_NAMES if "thigh" in name or "calf" in name
)
GO2_HOPPING_CFG = get_go2_cfg(
    spawn=get_go2_spawn_cfg(
        enabled_self_collisions=True,
        max_depenetration_velocity=1.0,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=0,
        contact_offset=0.01,
        rest_offset=0.0,
        enable_gyroscopic_forces=True,
    ),
    init_state=get_go2_init_state(
        pos=(0.0, 0.0, 0.42),
        joint_pos=GO2_HOPPING_DEFAULT_JOINT_POS,
    ),
    actuators=get_go2_actuators(
        hip_thigh_stiffness=20.0,
        hip_thigh_damping=0.5,
        hip_thigh_armature=0.0,
        hip_thigh_effort_limit=23.7,
        hip_thigh_velocity_limit=30.1,
        calf_stiffness=20.0,
        calf_damping=0.5,
        calf_armature=0.0,
        calf_effort_limit=45.43,
        calf_velocity_limit=15.70,
    ),
)
GO2_HOPPING_ACTION_SCALE_MAP = {name: GO2_HOPPING_ACTION_SCALE for name in GO2_CSV_JOINT_NAMES}


@configclass
class CommandRangesCfg:
    lin_vel_x = (-1.0, 1.0)
    lin_vel_y = (-1.0, 1.0)
    ang_vel_yaw = (-1.0, 1.0)
    heading = (-3.14, 3.14)


@configclass
class CommandsCfg:
    curriculum = True
    max_curriculum = 2.0
    num_commands = 4
    resampling_time = 5.0
    heading_command = False
    command_xy_deadzone = 0.1
    ranges: CommandRangesCfg = CommandRangesCfg()


@configclass
class DomainRandCfg:
    randomize_friction = True
    friction_range = (0.4, 0.8)

    push_robots = True
    push_interval_s = 4.0
    max_push_vel_xy = 0.4
    max_push_ang_vel = 0.6

    randomize_base_mass = True
    added_base_mass_range = (-1.0, 1.0)

    randomize_link_mass = True
    multiplied_link_mass_range = (0.9, 1.1)

    randomize_base_com = True
    added_base_com_range = (-0.02, 0.02)

    randomize_pd_gains = True
    stiffness_multiplier_range = (0.9, 1.1)
    damping_multiplier_range = (0.9, 1.1)

    randomize_motor_zero_offset = True
    motor_zero_offset_range = (-0.035, 0.035)

    add_obs_latency = True
    randomize_obs_motor_latency = True
    randomize_obs_imu_latency = True
    range_obs_motor_latency = (1, 3)
    range_obs_imu_latency = (1, 3)

    add_cmd_action_latency = True
    randomize_cmd_action_latency = True
    range_cmd_action_latency = (1, 3)


@configclass
class RewardScalesCfg:
    termination = -0.0
    tracking_lin_vel = 2.0
    tracking_ang_vel = 2.0
    lin_vel_z = 0.05
    ang_vel_xy = 0.2
    orientation = 0.6
    torques = -0.0002
    dof_vel = -0.0
    dof_acc = -5.5e-4
    base_height = 1.0
    feet_air_time = 1.0
    collision = -1.0
    feet_stumble = -0.0
    action_rate = -0.01
    default_pos = -0.1
    default_hip_pos = 0.3
    feet_contact_forces = -0.01
    jump = 2.0
    feet_clearance = 0.5


@configclass
class RewardsCfg:
    scales: RewardScalesCfg = RewardScalesCfg()
    only_positive_rewards = False
    tracking_sigma = 0.25
    soft_dof_pos_limit = 0.9
    soft_dof_vel_limit = 1.0
    soft_torque_limit = 1.0
    base_height_target = 0.3
    max_contact_force = 100.0
    cycle_time = 1.5
    target_feet_height = 0.05


@configclass
class ObsScalesCfg:
    lin_vel = 2.0
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05
    height_measurements = 5.0
    quat = 1.0


@configclass
class NormalizationCfg:
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    clip_observations = 100.0
    clip_actions = 100.0


@configclass
class NoiseScalesCfg:
    dof_pos = 0.01
    dof_vel = 1.5
    lin_vel = 0.1
    ang_vel = 0.2
    gravity = 0.05
    quat = 0.1
    height_measurements = 0.1


@configclass
class NoiseCfg:
    add_noise = True
    noise_level = 1.0
    noise_scales: NoiseScalesCfg = NoiseScalesCfg()


@configclass
class Go2HoppingFlatEnvCfg(DirectRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg(
        eye=(10.0, 0.0, 6.0),
        lookat=(11.0, 5.0, 3.0),
        origin_type="world",
    )
    episode_length_s = 24.0
    decimation = 4
    action_space = 12
    observation_space = 470
    state_space = 210

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.5,
            gpu_max_rigid_contact_count=2**23,
        ),
    )
    terrain: TerrainImporterCfg | None = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    trampoline: DeformableObjectCfg | None = None
    trampoline_radius = TRAMPOLINE_RADIUS
    trampoline_pin_width = TRAMPOLINE_PIN_WIDTH
    trampoline_thickness = TRAMPOLINE_THICKNESS
    trampoline_surface_height = TRAMPOLINE_TOP_Z
    trampoline_robot_clearance = 0.0
    usable_radius = TRAMPOLINE_RADIUS
    use_plain_trampoline_visual = False
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)
    robot = GO2_HOPPING_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    action_scale = _ordered_action_scale(GO2_HOPPING_ACTION_SCALE_MAP)
    foot_body_names = GO2_FOOT_BODY_NAMES
    penalized_contact_body_names = GO2_HOPPING_PENALIZED_CONTACT_BODY_NAMES
    termination_body_names = GO2_HOPPING_TERMINATION_BODY_NAMES
    non_foot_contact_body_names = GO2_NON_FOOT_CONTACT_BODY_NAMES
    anchor_body_name = GO2_TRACKING_ANCHOR_BODY_NAME
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=False,
    )

    commands: CommandsCfg = CommandsCfg()
    domain_rand: DomainRandCfg = DomainRandCfg()
    rewards: RewardsCfg = RewardsCfg()
    normalization: NormalizationCfg = NormalizationCfg()
    noise: NoiseCfg = NoiseCfg()

    def apply_play_overrides(self):
        self.episode_length_s = float(1.0e9)
        self.noise.add_noise = False
        self.domain_rand.push_robots = False
        self.viewer.eye = (1.8, 1.8, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.3)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.body_name = self.anchor_body_name
        return self


@configclass
class Go2HoppingTrampolineEnvCfg(Go2HoppingFlatEnvCfg):
    terrain: TerrainImporterCfg | None = None
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=max(3.0, 2.0 * float(TRAMPOLINE_RADIUS) + 2.0),
        replicate_physics=False,
    )
    trampoline: DeformableObjectCfg | None = make_trampoline_cfg(
        "/World/envs/env_.*/Trampoline",
        center_z=float(TRAMPOLINE_TOP_Z) - 0.5 * float(TRAMPOLINE_THICKNESS),
        debug_vis=False,
    )
    usable_radius = float(TRAMPOLINE_RADIUS)
    use_plain_trampoline_visual = False
