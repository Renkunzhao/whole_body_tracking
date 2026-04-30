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
REBOUNCE_HEIGHT_RANGE = (0.5, 0.8)

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

    hop = mdp.UniformRebounceCommandCfg(
        asset_cfg=SceneEntityCfg("robot"),
        foot_asset_cfg=SceneEntityCfg("robot", body_names=list(GO2_FOOT_BODY_NAMES)),
        foot_clearance=0.08,
        surface_z=0.0,
        apex_height_tolerance=0.25,
        # Initial target is sampled by the reset event so it can be decoupled
        # from the initial drop height. During a 20 s rollout, resample the
        # target at most about once to test command adaptation.
        resampling_time_range=(10.0, 20.0),
        ranges=mdp.UniformRebounceCommandCfg.Ranges(
            peak_height=REBOUNCE_HEIGHT_RANGE,
        ),
    )
    energy = mdp.EnergyMetricsCommandCfg(
        asset_cfg=SceneEntityCfg("robot", joint_names=[".*"]),
        apex_command_name="hop",
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
        hop_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "hop"})
        base_pos = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_quat = ObsTerm(func=mdp.root_quat_w,params={"make_quat_unique": True})
        # projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(PolicyCfg):
        # base_pos = ObsTerm(func=mdp.root_pos_w)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

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

    # reset — rebounce: sample target apex height and initial drop height
    # independently, teleport robot to the drop height with zero velocity and
    # default joint pose, and write the sampled target into the command buffer.
    reset_drop = EventTerm(
        func=mdp.reset_drop_from_height,
        mode="reset",
        params={
            "command_name": "hop",
            "asset_cfg": SceneEntityCfg("robot"),
            "drop_height_offset": 0.0,
            "drop_height_range": REBOUNCE_HEIGHT_RANGE,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    failed_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-250.0,
        params={"term_keys": ["base_orientation", "non_foot_contact", "no_valid_apex_timeout"]},
    )
    rebounce_height = RewTerm(
        func=mdp.rebounce_height_tracking_exp,
        weight=50.0,
        params={
            "command_name": "hop",
            "std": 0.10,
            "orientation_std": 0.35,
        },
    )
    energy_penalty = RewTerm(
        func=mdp.joint_mechanical_energy_penalty,
        weight=-2.5e-2,
        params={
            "command_name": "energy",
            "mode": "positive",
        },
    )
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    in_place_xy_yaw = RewTerm(
        func=mdp.in_place_xy_yaw_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "xy_std": 0.25,
            "yaw_std": 0.5,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    base_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.6})
    non_foot_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*_foot$).*"),
            "threshold": 1.0,
        },
    )
    no_valid_apex_timeout = DoneTerm(
        func=mdp.no_valid_apex_timeout,
        params={
            "command_name": "hop",
            "timeout": 2.0,
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
    pass


##
# Environment configuration
##


@configclass
class Go2RebounceEnvCfg(ManagerBasedRLEnvCfg):
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
        self.episode_length_s = 20.0
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
        self.commands.hop.ranges.peak_height = REBOUNCE_HEIGHT_RANGE
        self.events.reset_drop.params["drop_height_range"] = REBOUNCE_HEIGHT_RANGE
        return self


@configclass
class Go2RebounceFlatEnvCfg(Go2RebounceEnvCfg):
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
class Go2RebounceTrampolineEnvCfg(Go2RebounceEnvCfg):
    """Hopping on a deformable trampoline."""

    scene: TrampolineSceneCfg = TrampolineSceneCfg(num_envs=2048, env_spacing=4.0, replicate_physics=False)
    actions: TrampolineActionsCfg = TrampolineActionsCfg()
    events: TrampolineEventCfg = TrampolineEventCfg()
