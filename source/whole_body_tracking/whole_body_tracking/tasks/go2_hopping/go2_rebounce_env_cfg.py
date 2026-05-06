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
from whole_body_tracking.utils.trampoline_deformable import make_trampoline_cfg

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
REBOUNCE_OBS_HISTORY_LENGTH = 10
# Early phase: make hopping discovery easy before restoring the full command and trampoline DR ranges.
HOPPING_INIT_STEP = 1000 * 24
HOPPING_INIT_HEIGHT_RANGE = (0.5, 0.5)
TRAMPOLINE_FIXED_YOUNGS_MODULUS_RANGE = (8.0e4, 8.0e4)
TRAMPOLINE_FIXED_MASS_RANGE = (10.0, 10.0)
TRAMPOLINE_FIXED_DYNAMIC_FRICTION_RANGE = (0.8, 0.8)
TRAMPOLINE_FIXED_ELASTICITY_DAMPING_RANGE = (0.02, 0.02)
TRAMPOLINE_FIXED_DAMPING_SCALE_RANGE = (1.0, 1.0)
TRAMPOLINE_FIXED_POISSONS_RATIO_RANGE = (0.35, 0.35)
REBOUNCE_HEIGHT_RANGE = (0.5, 1.2)
TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE = (2.0e4, 8.0e4)
TRAMPOLINE_DR_MASS_RANGE = (5.0, 15.0)
TRAMPOLINE_DR_DYNAMIC_FRICTION_RANGE = (0.4, 1.2)
TRAMPOLINE_DR_ELASTICITY_DAMPING_RANGE = (0.01, 0.1)
TRAMPOLINE_DR_DAMPING_SCALE_RANGE = (1.0, 1.0)
TRAMPOLINE_DR_POISSONS_RATIO_RANGE = (0.25, 0.45)

# Delay energy optimization until after robust hopping and trampoline adaptation are learned.
ENERGY_PENALTY_START_STEP = 5000 * 24


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
        apex_height_tolerance=0.05,
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
        # base_pos = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.05, n_max=0.05))
        # base_quat = ObsTerm(func=mdp.root_quat_w,params={"make_quat_unique": True})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0
            self.flatten_history_dim = True

    @configclass
    class PrivilegedCfg(PolicyCfg):
        base_pos = ObsTerm(func=mdp.root_pos_w)
        base_quat = ObsTerm(func=mdp.root_quat_w,params={"make_quat_unique": True})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False
            self.history_length = 0

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class HistoryObservationsCfg(ObservationsCfg):
    """Deployable observations with flattened actor history for MLP ablations."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = REBOUNCE_OBS_HISTORY_LENGTH

    policy: PolicyCfg = PolicyCfg()
    critic: ObservationsCfg.PrivilegedCfg = ObservationsCfg.PrivilegedCfg()


@configclass
class RecurrentObservationsCfg(ObservationsCfg):
    """Instantaneous deployable observations for recurrent policies."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 0

    @configclass
    class PrivilegedCfg(ObservationsCfg.PrivilegedCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 0

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class TeacherObservationsCfg(ObservationsCfg):
    """Teacher observations with root state and true trampoline parameters."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        base_pos = ObsTerm(func=mdp.root_pos_w)
        base_quat = ObsTerm(func=mdp.root_quat_w,params={"make_quat_unique": True})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        trampoline_properties = ObsTerm(func=mdp.trampoline_properties)

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False
            self.history_length = 0

    @configclass
    class PrivilegedCfg(ObservationsCfg.PrivilegedCfg):
        trampoline_properties = ObsTerm(func=mdp.trampoline_properties)

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class RecurrentTeacherObservationsCfg(RecurrentObservationsCfg):
    """Recurrent teacher observations with root state and true trampoline parameters."""

    @configclass
    class PolicyCfg(RecurrentObservationsCfg.PolicyCfg):
        base_pos = ObsTerm(func=mdp.root_pos_w)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        trampoline_properties = ObsTerm(func=mdp.trampoline_properties)

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    @configclass
    class PrivilegedCfg(RecurrentObservationsCfg.PrivilegedCfg):
        trampoline_properties = ObsTerm(func=mdp.trampoline_properties)

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class DistillationObservationsCfg:
    """Student-teacher observations for action distillation.

    The student receives the same deployable actor history used by the
    flattened-history baseline. The teacher group must match the privileged
    teacher PPO actor observation exactly so its checkpoint can be loaded into
    RSL-RL's StudentTeacher module.
    """

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = REBOUNCE_OBS_HISTORY_LENGTH

    @configclass
    class TeacherCfg(TeacherObservationsCfg.PolicyCfg):
        pass

    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()


@configclass
class RecurrentDistillationObservationsCfg:
    """Student-teacher observations for recurrent student distillation."""

    @configclass
    class PolicyCfg(RecurrentObservationsCfg.PolicyCfg):
        pass

    @configclass
    class TeacherCfg(TeacherObservationsCfg.PolicyCfg):
        pass

    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()


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
        # func=mdp.joint_mechanical_energy_penalty,
        # weight=-2.5e-2,
        # params={
        #     "command_name": "energy",
        #     "mode": "positive",
        # },
        func=mdp.joint_mechanical_energy_penalty,
        weight=0.0,
        params={
            "command_name": "energy",
            "mode": "absolute",
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

    The rebounce command starts at a fixed low target while the policy discovers
    hopping. Trampoline randomization starts after the same init phase, and the
    energy penalty starts later once robust rebounding has been learned.
    ``modify_reward_weight`` uses manager steps.
    """

    hopping_init_height = CurrTerm(
        func=mdp.set_rebounce_command_height_range,
        params={
            "command_name": "hop",
            "init_peak_height_range": HOPPING_INIT_HEIGHT_RANGE,
            "peak_height_range": REBOUNCE_HEIGHT_RANGE,
            "num_steps": HOPPING_INIT_STEP,
        },
    )
    enable_energy_penalty = CurrTerm(
        func=modify_reward_weight,
        params={
            "term_name": "energy_penalty",
            "weight": -1.5e-2,
            "num_steps": ENERGY_PENALTY_START_STEP,
        },
    )


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
            "youngs_modulus_distribution": "log_uniform",
            "mass_range": TRAMPOLINE_DR_MASS_RANGE,
            "dynamic_friction_range": TRAMPOLINE_DR_DYNAMIC_FRICTION_RANGE,
            "elasticity_damping_range": TRAMPOLINE_DR_ELASTICITY_DAMPING_RANGE,
            "damping_scale_range": TRAMPOLINE_DR_DAMPING_SCALE_RANGE,
            "poissons_ratio_range": TRAMPOLINE_DR_POISSONS_RATIO_RANGE,
            "randomization_start_step": HOPPING_INIT_STEP,
            "fixed_youngs_modulus_range": TRAMPOLINE_FIXED_YOUNGS_MODULUS_RANGE,
            "fixed_mass_range": TRAMPOLINE_FIXED_MASS_RANGE,
            "fixed_dynamic_friction_range": TRAMPOLINE_FIXED_DYNAMIC_FRICTION_RANGE,
            "fixed_elasticity_damping_range": TRAMPOLINE_FIXED_ELASTICITY_DAMPING_RANGE,
            "fixed_damping_scale_range": TRAMPOLINE_FIXED_DAMPING_SCALE_RANGE,
            "fixed_poissons_ratio_range": TRAMPOLINE_FIXED_POISSONS_RATIO_RANGE,
        },
    )
    reapply_trampoline_pinning = EventTerm(func=reapply_trampoline_pinning, mode="reset")


@configclass
class Go2RebounceTrampolineEnvCfg(Go2RebounceEnvCfg):
    """Hopping on a deformable trampoline."""

    scene: TrampolineSceneCfg = TrampolineSceneCfg(num_envs=2048, env_spacing=4.0, replicate_physics=False)
    actions: TrampolineActionsCfg = TrampolineActionsCfg()
    events: TrampolineEventCfg = TrampolineEventCfg()


@configclass
class Go2RebounceTrampolineHistoryEnvCfg(Go2RebounceTrampolineEnvCfg):
    """Trampoline rebounce with flattened actor observation history."""

    observations: HistoryObservationsCfg = HistoryObservationsCfg()


@configclass
class Go2RebounceTrampolineTeacherEnvCfg(Go2RebounceTrampolineEnvCfg):
    """Trampoline rebounce teacher with true trampoline parameters in actor observations."""

    observations: TeacherObservationsCfg = TeacherObservationsCfg()


@configclass
class Go2RebounceTrampolineStudentEnvCfg(Go2RebounceTrampolineEnvCfg):
    """Distilled MLP student with deployable actor history and privileged teacher observations."""

    observations: DistillationObservationsCfg = DistillationObservationsCfg()


@configclass
class Go2RebounceTrampolineRnnEnvCfg(Go2RebounceTrampolineEnvCfg):
    """Trampoline rebounce with recurrent policy observations."""

    observations: RecurrentObservationsCfg = RecurrentObservationsCfg()


@configclass
class Go2RebounceTrampolineRnnTeacherEnvCfg(Go2RebounceTrampolineEnvCfg):
    """Recurrent trampoline rebounce teacher with true trampoline parameters."""

    observations: RecurrentTeacherObservationsCfg = RecurrentTeacherObservationsCfg()


@configclass
class Go2RebounceTrampolineRnnStudentEnvCfg(Go2RebounceTrampolineEnvCfg):
    """Distilled recurrent student with deployable observations and privileged teacher observations."""

    observations: RecurrentDistillationObservationsCfg = RecurrentDistillationObservationsCfg()
