from __future__ import annotations

import math
from collections.abc import Mapping

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR


GO2_URDF_PATH = f"{ASSET_DIR}/legged_models/unitree_go2/urdf/go2.urdf"

# Rotor inertia extracted from the Go2 URDF.
ROTOR_INERTIA = 0.000111842
HIP_GEAR_RATIO = 6.33
KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.92

NATURAL_FREQ = 10.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0

GO2_CSV_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

GO2_TRACKING_ANCHOR_BODY_NAME = "base"
GO2_TRACKING_BODY_NAMES = (
    "base",
    "FL_hip",
    "FL_thigh",
    "FL_foot",
    "FR_hip",
    "FR_thigh",
    "FR_foot",
    "RL_hip",
    "RL_thigh",
    "RL_foot",
    "RR_hip",
    "RR_thigh",
    "RR_foot",
)
GO2_FOOT_BODY_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
GO2_NON_FOOT_CONTACT_BODY_NAMES = (
    "base",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
)

GO2_HIP_AND_THIGH_JOINT_NAMES = tuple(name for name in GO2_CSV_JOINT_NAMES if not name.endswith("calf_joint"))
GO2_CALF_JOINT_NAMES = tuple(name for name in GO2_CSV_JOINT_NAMES if name.endswith("calf_joint"))

GO2_DEFAULT_INIT_POS = (0.0, 0.0, 0.4)
GO2_DEFAULT_JOINT_POS = {
    ".*_thigh_joint": 0.9,
    ".*_calf_joint": -1.8,
    ".*R_hip_joint": 0.1,
    ".*L_hip_joint": -0.1,
}
GO2_DEFAULT_JOINT_VEL = {".*": 0.0}


def _reflected_inertia(rotor_inertia: float, gear_ratio: float) -> float:
    return rotor_inertia * gear_ratio**2


HIP_ARMATURE = _reflected_inertia(ROTOR_INERTIA, HIP_GEAR_RATIO)
KNEE_ARMATURE = _reflected_inertia(ROTOR_INERTIA, KNEE_GEAR_RATIO)

HIP_EFFORT_LIMIT = 23.7
KNEE_EFFORT_LIMIT = 45.43
HIP_VELOCITY_LIMIT = 30.1
KNEE_VELOCITY_LIMIT = 15.70

HIP_STIFFNESS = HIP_ARMATURE * NATURAL_FREQ**2
KNEE_STIFFNESS = KNEE_ARMATURE * NATURAL_FREQ**2
HIP_DAMPING = 2.0 * DAMPING_RATIO * HIP_ARMATURE * NATURAL_FREQ
KNEE_DAMPING = 2.0 * DAMPING_RATIO * KNEE_ARMATURE * NATURAL_FREQ


def get_go2_spawn_cfg(
    *,
    activate_contact_sensors: bool = True,
    enabled_self_collisions: bool = True,
    max_depenetration_velocity: float = 1.0,
    solver_position_iteration_count: int = 8,
    solver_velocity_iteration_count: int = 4,
    enable_gyroscopic_forces: bool = True,
    contact_offset: float | None = None,
    rest_offset: float | None = None,
) -> sim_utils.UrdfFileCfg:
    collision_props = None
    if contact_offset is not None or rest_offset is not None:
        collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=0.0 if contact_offset is None else contact_offset,
            rest_offset=0.0 if rest_offset is None else rest_offset,
        )

    return sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=GO2_URDF_PATH,
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=max_depenetration_velocity,
            enable_gyroscopic_forces=enable_gyroscopic_forces,
            solver_position_iteration_count=solver_position_iteration_count,
            solver_velocity_iteration_count=solver_velocity_iteration_count,
        ),
        collision_props=collision_props,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=enabled_self_collisions,
            solver_position_iteration_count=solver_position_iteration_count,
            solver_velocity_iteration_count=solver_velocity_iteration_count,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    )


def get_go2_init_state(
    *,
    pos: tuple[float, float, float] = GO2_DEFAULT_INIT_POS,
    joint_pos: Mapping[str, float] | None = None,
    joint_vel: Mapping[str, float] | None = None,
) -> ArticulationCfg.InitialStateCfg:
    return ArticulationCfg.InitialStateCfg(
        pos=pos,
        joint_pos=dict(GO2_DEFAULT_JOINT_POS if joint_pos is None else joint_pos),
        joint_vel=dict(GO2_DEFAULT_JOINT_VEL if joint_vel is None else joint_vel),
    )


def get_go2_actuators(
    *,
    hip_thigh_stiffness: float = HIP_STIFFNESS,
    hip_thigh_damping: float = HIP_DAMPING,
    hip_thigh_armature: float = HIP_ARMATURE,
    hip_thigh_effort_limit: float = HIP_EFFORT_LIMIT,
    hip_thigh_velocity_limit: float = HIP_VELOCITY_LIMIT,
    calf_stiffness: float = KNEE_STIFFNESS,
    calf_damping: float = KNEE_DAMPING,
    calf_armature: float = KNEE_ARMATURE,
    calf_effort_limit: float = KNEE_EFFORT_LIMIT,
    calf_velocity_limit: float = KNEE_VELOCITY_LIMIT,
) -> dict[str, ImplicitActuatorCfg]:
    return {
        "hip_thigh": ImplicitActuatorCfg(
            joint_names_expr=list(GO2_HIP_AND_THIGH_JOINT_NAMES),
            effort_limit_sim=hip_thigh_effort_limit,
            velocity_limit_sim=hip_thigh_velocity_limit,
            stiffness=hip_thigh_stiffness,
            damping=hip_thigh_damping,
            armature=hip_thigh_armature,
        ),
        "calf": ImplicitActuatorCfg(
            joint_names_expr=list(GO2_CALF_JOINT_NAMES),
            effort_limit_sim=calf_effort_limit,
            velocity_limit_sim=calf_velocity_limit,
            stiffness=calf_stiffness,
            damping=calf_damping,
            armature=calf_armature,
        ),
    }


def get_go2_cfg(
    *,
    spawn: sim_utils.UrdfFileCfg | None = None,
    init_state: ArticulationCfg.InitialStateCfg | None = None,
    actuators: Mapping[str, ImplicitActuatorCfg] | None = None,
    soft_joint_pos_limit_factor: float = 0.9,
) -> ArticulationCfg:
    return ArticulationCfg(
        spawn=get_go2_spawn_cfg() if spawn is None else spawn,
        init_state=get_go2_init_state() if init_state is None else init_state,
        soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
        actuators=dict(get_go2_actuators() if actuators is None else actuators),
    )


def get_go2_action_scale(actuators: Mapping[str, ImplicitActuatorCfg] | None = None) -> dict[str, float]:
    action_scale: dict[str, float] = {}
    actuator_cfgs = get_go2_actuators() if actuators is None else dict(actuators)
    for actuator in actuator_cfgs.values():
        effort_limit = actuator.effort_limit_sim
        stiffness = actuator.stiffness
        joint_names = tuple(str(name) for name in actuator.joint_names_expr)
        if effort_limit is None:
            continue
        if not isinstance(effort_limit, dict):
            effort_limit = {name: effort_limit for name in joint_names}
        if not isinstance(stiffness, dict):
            stiffness = {name: stiffness for name in joint_names}
        for joint_name in joint_names:
            if joint_name in effort_limit and joint_name in stiffness and stiffness[joint_name]:
                action_scale[joint_name] = 0.25 * effort_limit[joint_name] / stiffness[joint_name]
    return action_scale


GO2_CFG = get_go2_cfg()
GO2_ACTION_SCALE = get_go2_action_scale(GO2_CFG.actuators)
