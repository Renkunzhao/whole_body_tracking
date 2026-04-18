from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.envs.mdp.curriculums import modify_reward_weight
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.go2_hopping.mdp as mdp
from whole_body_tracking.robots.go2 import (
    GO2_ACTION_SCALE,
    GO2_FOOT_BODY_NAMES,
    get_go2_cfg,
    get_go2_spawn_cfg,
)
from whole_body_tracking.tasks.tracking.mdp import (
    RandomizeTrampolineProperties,
    TrampolinePinningActionCfg,
    reapply_trampoline_pinning,
)
from whole_body_tracking.utils.trampoline_deformable import (
    TRAMPOLINE_DR_MASS_RANGE,
    TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE,
    make_trampoline_cfg,
)

##
# Scene definition
##
VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

GO2_HOPPING_CFG = get_go2_cfg(
    spawn=get_go2_spawn_cfg(
        enabled_self_collisions=True,
        max_depenetration_velocity=5.0,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
        contact_offset=0.005,
        rest_offset=0.0,
        enable_gyroscopic_forces=True,
    )
)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Base scene: robot, lights, contact sensor. Terrain is added by subclasses."""

    # robots
    robot: ArticulationCfg = GO2_HOPPING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


@configclass
class FlatSceneCfg(MySceneCfg):
    """Scene with a flat ground plane."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    twist = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.5,
        rel_heading_envs=0.3,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
    )
    hop = mdp.UniformHoppingCommandCfg(
        asset_cfg=SceneEntityCfg("robot"),
        sensor_cfg=SceneEntityCfg("contact_forces", body_names=list(GO2_FOOT_BODY_NAMES)),
        contact_threshold=5.0,
        ranges=mdp.UniformHoppingCommandCfg.Ranges(
            peak_height=(0.05, 1.0),
            stance_time=(0.1, 0.50),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=GO2_ACTION_SCALE, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        phase = ObsTerm(func=mdp.sin_cos_phase, params={"command_name": "hop"})
        hop_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "hop"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(GO2_FOOT_BODY_NAMES)),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )

    # reset
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    phase_contact = RewTerm(
        func=mdp.phase_contact,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(GO2_FOOT_BODY_NAMES)),
            "command_name": "hop",
            "contact_threshold": 5.0,
        },
    )
    # phase_contact = RewTerm(
    #     func=mdp.phase_contact_distance,
    #     weight=2.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=list(GO2_FOOT_BODY_NAMES)),
    #         "command_name": "hop",
    #         "contact_height": 0.024,   # 对齐静止脚 z，让 sigmoid 在最大梯度处（σ=0.5, σ'=0.25），bootstrap 最快
    #         "softness": 0.005,          # 过渡区 ±1cm
    #         "surface_z": 0.0,
    #     },
    # )
    # note: this term sum all joints' deviations, so the weight should be divided by the number of joints to keep the reward magnitude consistent when changing the robot
    # joint_deviation_l1 = RewTerm(func=mdp.joint_deviation_l1, weight=0.0)
    joint_deviation_phase_exp = RewTerm(
        func=mdp.joint_deviation_phase_exp,
        weight=0.0,
        params={
            "command_name": "hop",
            "std_stance": {
                ".*_hip_joint": 0.3,   # 严格：站立时 hip 不能晃
                ".*_thigh_joint": 0.3,
                ".*_calf_joint": 0.6,
            },
            "std_flight": {
                ".*_hip_joint": 0.3,   # 飞行时放松一点
                ".*_thigh_joint": 0.3,  # 膝大幅摆动
                ".*_calf_joint": 0.6,
            },
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
    )
    track_linear_velocity = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.0,
        params={
            "command_name": "twist",  # 换成你的 command term 名
            "std": math.sqrt(0.25),           # 典型值
        },
    )
    track_angular_velocity = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.0,
        params={
            "command_name": "twist",  # 换成你的 command term 名
            "std": math.sqrt(0.5),           # 典型值
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    base_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    non_foot_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*_foot$).*"),
            "threshold": 1.0,
        },
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.

    Phase 1 (steps 0 ~ JOINT_DEVIATION_START): learn basic hopping —
        active terms: phase_contact (+2.0), action_rate_l2 (-1e-2), joint_pos_limits (-10.0).
    Phase 2 (steps JOINT_DEVIATION_START ~ TRACK_ANG_VEL_START): add pose regularization.
    Phase 3 (steps TRACK_ANG_VEL_START+): add angular velocity tracking.
    """

    enable_joint_deviation = CurrTerm(
        func=modify_reward_weight,
        # params={"term_name": "joint_deviation_l1", "weight": -0.1, "num_steps": 300*24 },
        params={"term_name": "joint_deviation_phase_exp", "weight": 0.5, "num_steps": 300*24 },
    )
    enable_track_linear_velocity = CurrTerm(
        func=modify_reward_weight,
        params={"term_name": "track_linear_velocity", "weight": 1.0, "num_steps": 300*24},
    )
    enable_track_angular_velocity = CurrTerm(
        func=modify_reward_weight,
        params={"term_name": "track_angular_velocity", "weight": 1.0, "num_steps": 300*24},
    )
    pass


##
# Environment configuration
##


@configclass
class Go2HoppingEnvCfg(ManagerBasedRLEnvCfg):
    """Base hopping environment configuration (terrain-agnostic).

    Subclasses must provide a concrete scene with a terrain/support asset.
    """

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.4)
        self.viewer.origin_type = "world"
        self.viewer.asset_name = None
        self.viewer.body_name = None

    def apply_play_overrides(self):
        self.episode_length_s = 1.0e9
        self.commands.twist.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.twist.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.twist.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.twist.heading_command = False
        self.commands.twist.rel_standing_envs = 0.0
        self.commands.hop.ranges.peak_height = (0.265, 0.265)
        self.commands.hop.ranges.stance_time = (0.6, 0.6)
        self.events.push_robot = None
        return self


@configclass
class Go2HoppingFlatEnvCfg(Go2HoppingEnvCfg):
    """Hopping on a flat ground plane."""

    scene: FlatSceneCfg = FlatSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics_material = self.scene.terrain.physics_material


##
# Trampoline variant
##


@configclass
class TrampolineSceneCfg(MySceneCfg):
    """Scene with a deformable trampoline instead of a rigid ground plane."""

    trampoline: DeformableObjectCfg = make_trampoline_cfg("{ENV_REGEX_NS}/Trampoline")


@configclass
class TrampolineActionsCfg(ActionsCfg):
    """Actions with an extra term to pin the trampoline rim each step."""

    trampoline_pin = TrampolinePinningActionCfg(asset_name="trampoline")


@configclass
class TrampolineEventCfg(EventCfg):
    """Events with trampoline material/mass randomization and pinning refresh on reset."""

    randomize_trampoline_properties = EventTerm(
        func=RandomizeTrampolineProperties,
        mode="reset",
        params={
            "asset_name": "trampoline",
            "youngs_modulus_range": TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE,
            "mass_range": TRAMPOLINE_DR_MASS_RANGE,
        },
    )
    reapply_trampoline_pinning = EventTerm(func=reapply_trampoline_pinning, mode="reset")


@configclass
class Go2HoppingTrampolineEnvCfg(Go2HoppingEnvCfg):
    """Hopping on a deformable trampoline."""

    scene: TrampolineSceneCfg = TrampolineSceneCfg(num_envs=2048, env_spacing=4.0, replicate_physics=False)
    actions: TrampolineActionsCfg = TrampolineActionsCfg()
    events: TrampolineEventCfg = TrampolineEventCfg()
