"""Unified trampoline smoke script for G1, Go2, or ball actors.

The two main switches are:

- ``--actor {g1,go2,ball}``
- ``--trampoline_mode {builtin,spring}``

Notes:

- ``builtin`` uses Isaac Lab's deformable body support with pinned rim nodes.
- ``spring`` uses a custom flat support model:
  - ``ball``: a simplified spring model at the rigid-body origin
  - ``g1``: a simplified spring model at the two ankle-link origins
  - ``go2``: a simplified spring model at the four foot-link origins
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Load G1, Go2, or ball on a built-in or custom trampoline.")
parser.add_argument("--actor", type=str, choices=("g1", "go2", "ball"), default="go2", help="Actor placed on the trampoline.")
parser.add_argument(
    "--trampoline_mode",
    type=str,
    choices=("builtin", "spring"),
    default="spring",
    help="Trampoline implementation to use.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of scene instances to spawn.")
parser.add_argument("--env_spacing", type=float, default=4.0, help="Environment spacing. Defaults to 4.0 when omitted.")
parser.add_argument(
    "--passive",
    action="store_true",
    default=False,
    help="If set for robot actors, do not command the default standing pose.",
)
# built-in deformable trampoline options
parser.add_argument("--pin_width", type=float, default=0.4, help="Pinned rim width in meters for the deformable trampoline.")
parser.add_argument("--youngs_modulus", type=float, default=8.0e4, help="Built-in trampoline Young's modulus.")
parser.add_argument("--mass", type=float, default=10.0, help="Built-in trampoline mass.")
parser.add_argument("--sim_resolution", type=int, default=10, help="Built-in trampoline hexahedral resolution.")

# custom contact model options
parser.add_argument("--normal_stiffness", type=float, default=5000.0, help="Custom model normal stiffness.")
parser.add_argument("--normal_damping", type=float, default=120.0, help="Custom model normal damping.")
parser.add_argument("--tangential_damping", type=float, default=120.0, help="Custom model tangential damping.")
parser.add_argument("--friction_coeff", type=float, default=1.0, help="Custom model friction coefficient.")

# ball options
parser.add_argument("--ball_radius", type=float, default=0.18, help="Ball radius in meters.")
parser.add_argument("--ball_mass", type=float, default=6.0, help="Ball mass in kilograms.")
parser.add_argument(
    "--ball_horizontal_speed",
    type=float,
    default=1.0,
    help="Horizontal speed magnitude assigned to the ball at each reset; direction is randomized.",
)
parser.add_argument(
    "--ball_height",
    type=float,
    default=0.8,
    help="Initial ball center height. Defaults to support height + 0.8m.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.robots.go2 import GO2_CFG
from whole_body_tracking.utils.point_foot_contact_force import PointFootContactForceCfg, PointFootContactForceModel
from whole_body_tracking.utils.trampoline_deformable import (
    TRAMPOLINE_THICKNESS,
    build_trampoline_kinematic_targets,
    build_trampoline_node_visualizers,
    make_trampoline_cfg,
    update_trampoline_node_visualizers,
)


SPRING_PLANE_RADIUS = 1.5
SPRING_PLANE_THICKNESS = 0.02
G1_CONTACT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
GO2_CONTACT_BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


@configclass
class UnifiedTrampolineSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg | None = None
    ball: RigidObjectCfg | None = None
    trampoline: DeformableObjectCfg | None = None
    spring_visual: AssetBaseCfg | None = None
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

def make_ball_cfg(prim_path: str, ball_height: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.SphereCfg(
            radius=args_cli.ball_radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.ball_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7 if args_cli.trampoline_mode == "builtin" else 0.0,
                dynamic_friction=0.6 if args_cli.trampoline_mode == "builtin" else 0.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.35, 0.15), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, ball_height)),
    )


def make_spring_visual_cfg(prim_path: str, surface_height: float) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.MeshCylinderCfg(
            radius=SPRING_PLANE_RADIUS,
            height=SPRING_PLANE_THICKNESS,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.95), metallic=0.05),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, surface_height - 0.5 * SPRING_PLANE_THICKNESS)),
    )


def build_scene_cfg(surface_height: float, ball_height: float, env_spacing: float) -> UnifiedTrampolineSceneCfg:
    scene_cfg = UnifiedTrampolineSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=env_spacing,
        replicate_physics=args_cli.trampoline_mode != "builtin",
    )
    if args_cli.actor == "g1":
        scene_cfg.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.actor == "go2":
        scene_cfg.robot = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        scene_cfg.ball = make_ball_cfg("{ENV_REGEX_NS}/Ball", ball_height)

    if args_cli.trampoline_mode == "builtin":
        scene_cfg.trampoline = make_trampoline_cfg(
            "{ENV_REGEX_NS}/Trampoline",
            center_z=surface_height - 0.5 * TRAMPOLINE_THICKNESS,
            mass=args_cli.mass,
            youngs_modulus=args_cli.youngs_modulus,
            sim_resolution=args_cli.sim_resolution,
        )
    else:
        scene_cfg.spring_visual = make_spring_visual_cfg("{ENV_REGEX_NS}/SpringPlaneVisual", surface_height)
    return scene_cfg


def set_camera(sim: SimulationContext) -> None:
    sim.set_camera_view(eye=[3.0, 3.0, 2.0], target=[0.0, 0.0, 0.5])


def clear_ball_wrench(ball: RigidObject) -> None:
    zero_force = torch.zeros((ball.num_instances, ball.num_bodies, 3), device=ball.device)
    zero_torque = torch.zeros_like(zero_force)
    ball.permanent_wrench_composer.set_forces_and_torques(
        forces=zero_force,
        torques=zero_torque,
        is_global=True,
    )


def reset_ball(scene: InteractiveScene, ball: RigidObject) -> None:
    env_ids = torch.arange(scene.num_envs, device=ball.device, dtype=torch.long)
    default_root_state = ball.data.default_root_state.clone()
    default_root_state[:, 0:3] += scene.env_origins
    if args_cli.ball_horizontal_speed > 0.0:
        headings = sample_uniform(0.0, 2.0 * torch.pi, (scene.num_envs,), device=ball.device)
        default_root_state[:, 7] = args_cli.ball_horizontal_speed * torch.cos(headings)
        default_root_state[:, 8] = args_cli.ball_horizontal_speed * torch.sin(headings)
    ball.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
    ball.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
    ball.reset()
    clear_ball_wrench(ball)


def reset_robot(scene: InteractiveScene, robot: Articulation, surface_height: float) -> None:
    env_ids = torch.arange(scene.num_envs, device=robot.device, dtype=torch.long)
    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, 0:3] += scene.env_origins
    default_root_state[:, 2] += surface_height
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
    robot.reset()


def reset_deformable_trampoline(
    scene: InteractiveScene,
    trampoline: DeformableObject,
    trampoline_targets: torch.Tensor,
) -> None:
    env_ids = torch.arange(scene.num_envs, device=trampoline.device, dtype=torch.long)
    trampoline.write_nodal_state_to_sim(trampoline.data.default_nodal_state_w, env_ids=env_ids)
    trampoline.write_nodal_kinematic_target_to_sim(trampoline_targets, env_ids=env_ids)
    trampoline.reset(env_ids=env_ids)


def resolve_robot_contact_body_names(actor: str) -> list[str]:
    if actor == "g1":
        return G1_CONTACT_BODY_NAMES
    if actor == "go2":
        return GO2_CONTACT_BODY_NAMES
    raise ValueError(f"Unsupported robot actor for spring contact: {actor}.")


def _build_spring_model() -> PointFootContactForceModel:
    return PointFootContactForceModel(
        PointFootContactForceCfg(
            plane_height=0.0,
            normal_stiffness=args_cli.normal_stiffness,
            normal_damping=args_cli.normal_damping,
            tangential_damping=args_cli.tangential_damping,
            friction_coeff=args_cli.friction_coeff,
        )
    )


SPRING_RUNTIME_SCALAR_KEYS = (
    "last_penetration",
    "last_normal_force",
    "last_tangential_force_norm",
)


def _build_spring_runtime(
    device: str | torch.device,
    num_envs: int,
    num_bodies: int,
    **extra: object,
) -> dict[str, object]:
    scalar_shape = (num_envs, num_bodies)
    return {
        "model": _build_spring_model(),
        **{key: torch.zeros(scalar_shape, device=device) for key in SPRING_RUNTIME_SCALAR_KEYS},
        **extra,
    }


def build_robot_spring_runtime(robot: Articulation, actor: str, num_envs: int) -> dict[str, object]:
    contact_body_names = resolve_robot_contact_body_names(actor)
    body_ids, body_names = robot.find_bodies(contact_body_names, preserve_order=True)
    if len(body_ids) != len(contact_body_names):
        raise RuntimeError(f"Failed to resolve custom contact bodies: expected {contact_body_names}, got {body_names}.")

    return _build_spring_runtime(
        robot.device,
        num_envs,
        len(body_ids),
        body_ids=list(body_ids),
        body_names=body_names,
    )


def build_ball_spring_runtime(ball: RigidObject, num_envs: int) -> dict[str, object]:
    return _build_spring_runtime(
        ball.device,
        num_envs,
        1,
        last_force_w=torch.zeros((num_envs, 3), device=ball.device),
    )


def clear_robot_spring_contact(robot: Articulation, runtime: dict[str, object]) -> None:
    body_ids = runtime["body_ids"]
    num_bodies = len(body_ids)
    zero_force = torch.zeros((robot.num_instances, num_bodies, 3), device=robot.device)
    robot.permanent_wrench_composer.set_forces_and_torques(
        forces=zero_force,
        torques=torch.zeros_like(zero_force),
        body_ids=body_ids,
        is_global=True,
    )
    for key in SPRING_RUNTIME_SCALAR_KEYS:
        runtime[key].zero_()


def clear_ball_spring_contact(ball: RigidObject, runtime: dict[str, object]) -> None:
    clear_ball_wrench(ball)
    runtime["last_force_w"].zero_()
    for key in SPRING_RUNTIME_SCALAR_KEYS:
        runtime[key].zero_()


def apply_robot_spring_contact(
    scene: InteractiveScene,
    robot: Articulation,
    runtime: dict[str, object],
) -> None:
    body_ids = runtime["body_ids"]
    num_bodies = len(body_ids)
    data = robot.data
    body_pos_w = data.body_pos_w[:, body_ids]
    body_lin_vel_w = data.body_lin_vel_w[:, body_ids]
    env_origins = scene.env_origins.unsqueeze(1).expand(-1, num_bodies, -1)

    force_w, _, _, penetration, normal_force, tangential_force_norm, _ = runtime["model"].compute_wrenches(
        body_pos_w.reshape(-1, 3),
        body_lin_vel_w.reshape(-1, 3),
        env_origins=env_origins.reshape(-1, 3),
    )
    print(f"body_pos_w={body_pos_w.reshape(-1, 3)}")
    print(f"body_lin_vel_w={body_lin_vel_w.reshape(-1, 3)}")
    print(f"force_w={force_w}")

    force_w = force_w.view(scene.num_envs, num_bodies, 3)
    runtime.update(
        last_penetration=penetration.view(scene.num_envs, num_bodies),
        last_normal_force=normal_force.view(scene.num_envs, num_bodies),
        last_tangential_force_norm=tangential_force_norm.view(scene.num_envs, num_bodies),
    )

    robot.permanent_wrench_composer.set_forces_and_torques(
        forces=force_w,
        torques=torch.zeros_like(force_w),
        body_ids=body_ids,
        is_global=True,
    )


def apply_ball_spring_contact(
    scene: InteractiveScene,
    ball: RigidObject,
    runtime: dict[str, object],
) -> None:
    force_w, _, _, penetration, normal_force, tangential_force_norm, _ = runtime["model"].compute_wrenches(
        ball.data.root_pos_w,
        ball.data.root_lin_vel_w,
        env_origins=scene.env_origins,
    )

    runtime.update(
        last_penetration=penetration.view(scene.num_envs, 1),
        last_normal_force=normal_force.view(scene.num_envs, 1),
        last_tangential_force_norm=tangential_force_norm.view(scene.num_envs, 1),
        last_force_w=force_w,
    )

    ball.permanent_wrench_composer.set_forces_and_torques(
        forces=force_w.unsqueeze(1),
        torques=torch.zeros_like(force_w).unsqueeze(1),
        is_global=True,
    )


def print_mode_summary(surface_height: float, ball_height: float | None) -> None:
    summary = (
        f"[INFO]: Loaded trampoline scene — actor={args_cli.actor}, mode={args_cli.trampoline_mode}, "
        f"num_envs={args_cli.num_envs}, passive={args_cli.passive}, surface_height={surface_height:.3f}"
    )
    if args_cli.actor == "ball" and ball_height is not None:
        summary += f", ball_height={ball_height:.3f}"
    print(summary)


def main() -> None:
    env_spacing = args_cli.env_spacing
    reset_interval = 500

    sim_dt = 0.005
    sim = SimulationContext(sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device))
    set_camera(sim)

    ball_height = args_cli.ball_height
    scene_cfg = build_scene_cfg(0.0, ball_height, env_spacing)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.update(sim_dt)

    actor = scene["robot"] if args_cli.actor != "ball" else scene["ball"]

    trampoline = None
    trampoline_targets = None
    trampoline_pinned_mask = None
    trampoline_center_node_ids = None
    trampoline_pinned_node_visualizer = None
    trampoline_free_node_visualizer = None
    if args_cli.trampoline_mode == "builtin":
        trampoline = scene["trampoline"]
        if trampoline.material_physx_view is None:
            raise RuntimeError("Failed to create deformable trampoline material view.")
        trampoline_targets, trampoline_pinned_mask, trampoline_center_node_ids = build_trampoline_kinematic_targets(
            trampoline.data.default_nodal_state_w,
            trampoline.data.nodal_kinematic_target,
            pin_width=args_cli.pin_width,
        )
        trampoline_pinned_node_visualizer, trampoline_free_node_visualizer = build_trampoline_node_visualizers()

    spring_ball_runtime = None
    spring_robot_runtime = None
    if args_cli.trampoline_mode == "spring":
        if args_cli.actor == "ball":
            spring_ball_runtime = build_ball_spring_runtime(actor, scene.num_envs)
        else:
            spring_robot_runtime = build_robot_spring_runtime(actor, args_cli.actor, scene.num_envs)
            print(f"[INFO]: Resolved custom contact bodies: {spring_robot_runtime['body_names']}")

    if args_cli.actor == "ball":
        reset_ball(scene, actor)
        if spring_ball_runtime is not None:
            clear_ball_spring_contact(actor, spring_ball_runtime)
    else:
        reset_robot(scene, actor, 0.0)
        if spring_robot_runtime is not None:
            clear_robot_spring_contact(actor, spring_robot_runtime)

    if trampoline is not None:
        reset_deformable_trampoline(scene, trampoline, trampoline_targets)

    print_mode_summary(0.0, ball_height)
    if trampoline_pinned_mask is not None:
        pinned_node_count = int(trampoline_pinned_mask[0].sum().item())
        free_node_count = int((~trampoline_pinned_mask[0]).sum().item())
        print(f"[INFO]: Trampoline nodes — pinned={pinned_node_count}, free={free_node_count}")
        update_trampoline_node_visualizers(
            trampoline,
            trampoline_pinned_mask,
            trampoline_pinned_node_visualizer,
            trampoline_free_node_visualizer,
        )

    step = 0
    while simulation_app.is_running():
        if step > 0 and step % reset_interval == 0:
            if args_cli.actor == "ball":
                reset_ball(scene, actor)
                if spring_ball_runtime is not None:
                    clear_ball_spring_contact(actor, spring_ball_runtime)
            else:
                reset_robot(scene, actor, 0.0)
                if spring_robot_runtime is not None:
                    clear_robot_spring_contact(actor, spring_robot_runtime)
            if trampoline is not None:
                reset_deformable_trampoline(scene, trampoline, trampoline_targets)
                print("[INFO]: Resetting built-in trampoline.")
            else:
                print("[INFO]: Resetting spring trampoline scene.")

        if trampoline is not None:
            trampoline.write_nodal_kinematic_target_to_sim(trampoline_targets)
        elif spring_ball_runtime is not None:
            apply_ball_spring_contact(scene, actor, spring_ball_runtime)
        elif spring_robot_runtime is not None:
            apply_robot_spring_contact(scene, actor, spring_robot_runtime)

        if args_cli.actor != "ball" and not args_cli.passive:
            actor.set_joint_position_target(actor.data.default_joint_pos.clone())

        actor.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        if trampoline is not None and trampoline_pinned_mask is not None:
            update_trampoline_node_visualizers(
                trampoline,
                trampoline_pinned_mask,
                trampoline_pinned_node_visualizer,
                trampoline_free_node_visualizer,
            )

        step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
