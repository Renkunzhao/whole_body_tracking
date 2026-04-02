import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


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

GO2_CFG = UNITREE_GO2_CFG.replace(
    spawn=UNITREE_GO2_CFG.spawn.replace(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            # Give PhysX more room to resolve hard landing contacts during aerial flips.
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.278),
        joint_pos={
            ".*thigh_joint": 0.9,
            ".*calf_joint": -1.8,
            ".*R_hip_joint": 0.1,
            ".*L_hip_joint": -0.1,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "go2_hip_and_thigh": DCMotorCfg(
            joint_names_expr=list(GO2_HIP_AND_THIGH_JOINT_NAMES),
            stiffness=HIP_STIFFNESS,
            damping=HIP_DAMPING,
            armature=HIP_ARMATURE,
            effort_limit=HIP_EFFORT_LIMIT,
            effort_limit_sim=HIP_EFFORT_LIMIT,
            saturation_effort=HIP_EFFORT_LIMIT,
            velocity_limit=HIP_VELOCITY_LIMIT,
            velocity_limit_sim=HIP_VELOCITY_LIMIT,
            friction=0.0,
        ),
        "go2_calf": DCMotorCfg(
            joint_names_expr=list(GO2_CALF_JOINT_NAMES),
            stiffness=KNEE_STIFFNESS,
            damping=KNEE_DAMPING,
            armature=KNEE_ARMATURE,
            effort_limit=KNEE_EFFORT_LIMIT,
            effort_limit_sim=KNEE_EFFORT_LIMIT,
            saturation_effort=KNEE_EFFORT_LIMIT,
            velocity_limit=KNEE_VELOCITY_LIMIT,
            velocity_limit_sim=KNEE_VELOCITY_LIMIT,
            friction=0.0,
        ),
    },
)

GO2_ACTION_SCALE = {
    name: 0.25 * HIP_EFFORT_LIMIT / HIP_STIFFNESS for name in GO2_HIP_AND_THIGH_JOINT_NAMES
}
GO2_ACTION_SCALE.update({name: 0.25 * KNEE_EFFORT_LIMIT / KNEE_STIFFNESS for name in GO2_CALF_JOINT_NAMES})
