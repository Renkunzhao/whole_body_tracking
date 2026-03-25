import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

ROTOR_INERTIA = 0.000111842
HIP_GEAR_RATIO = 6.33
KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.92

HIP_ARMATURE = ROTOR_INERTIA * HIP_GEAR_RATIO**2
KNEE_ARMATURE = ROTOR_INERTIA * KNEE_GEAR_RATIO**2

HIP_EFFORT_LIMIT = 23.7
KNEE_EFFORT_LIMIT = 45.43
HIP_VELOCITY_LIMIT = 30.1
KNEE_VELOCITY_LIMIT = 15.70

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

HIP_STIFFNESS = HIP_ARMATURE * NATURAL_FREQ**2
KNEE_STIFFNESS = KNEE_ARMATURE * NATURAL_FREQ**2
HIP_DAMPING = 2.0 * DAMPING_RATIO * HIP_ARMATURE * NATURAL_FREQ
KNEE_DAMPING = 2.0 * DAMPING_RATIO * KNEE_ARMATURE * NATURAL_FREQ

GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/legged_models/unitree_go2/urdf/go2.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.278),
        joint_pos={
            ".*_thigh_joint": 0.9,
            ".*_calf_joint": -1.8,
            ".*R_hip_joint": 0.1,
            ".*L_hip_joint": -0.1,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_thigh": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            effort_limit_sim=HIP_EFFORT_LIMIT,
            velocity_limit_sim=HIP_VELOCITY_LIMIT,
            stiffness=HIP_STIFFNESS,
            damping=HIP_DAMPING,
            armature=HIP_ARMATURE,
        ),
        "calf": ImplicitActuatorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit_sim=KNEE_EFFORT_LIMIT,
            velocity_limit_sim=KNEE_VELOCITY_LIMIT,
            stiffness=KNEE_STIFFNESS,
            damping=KNEE_DAMPING,
            armature=KNEE_ARMATURE,
        ),
    },
)

GO2_ACTION_SCALE = {}
for a in GO2_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            GO2_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
