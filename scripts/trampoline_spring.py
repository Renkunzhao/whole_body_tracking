# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ball-drop demo driven by a custom spring-damper terrain interface.

This demo avoids built-in soft bodies and instead applies a world-frame vertical
spring force to a rigid ball when its bottom penetrates a reference plane.

Example:

    ./isaaclab.sh -p /home/rkz/code/whole_body_tracking/scripts/trampoline_spring.py
    ./isaaclab.sh -p /home/rkz/code/whole_body_tracking/scripts/trampoline_spring.py --stiffness 5000 --damping 120
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drop a rigid ball onto a custom vertical spring-damper plane.")
parser.add_argument("--stiffness", type=float, default=5000.0, help="Spring stiffness in N/m.")
parser.add_argument("--damping", type=float, default=120.0, help="Spring damping in N/(m/s).")
parser.add_argument("--plane_height", type=float, default=0.75, help="Reference spring plane height in meters.")
parser.add_argument("--ball_height", type=float, default=1.55, help="Initial ball center height in meters.")
parser.add_argument("--ball_radius", type=float, default=0.18, help="Ball radius in meters.")
parser.add_argument("--ball_mass", type=float, default=6.0, help="Ball mass in kilograms.")
parser.add_argument("--reset_interval", type=int, default=360, help="Number of simulation steps between resets.")
parser.add_argument("--print_interval", type=int, default=30, help="Number of simulation steps between status prints.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext

from whole_body_tracking.utils.spring_terrain import VerticalSpringPlane, VerticalSpringPlaneCfg

SPRING_PLANE_RADIUS = 1.5
SPRING_PLANE_THICKNESS = 0.02


def design_scene() -> RigidObject:
    """Spawn a light-weight scene with a visual spring plane and a rigid ball."""
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.85, 0.85, 0.85))
    light_cfg.func("/World/Light", light_cfg)

    spring_plane_cfg = sim_utils.MeshCylinderCfg(
        radius=SPRING_PLANE_RADIUS,
        height=SPRING_PLANE_THICKNESS,
        axis="Z",
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.95), metallic=0.05),
    )
    spring_plane_cfg.func(
        "/World/SpringPlaneVisual",
        spring_plane_cfg,
        translation=(0.0, 0.0, args_cli.plane_height - 0.01),
    )

    ball_cfg = RigidObjectCfg(
        prim_path="/World/Ball",
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
                static_friction=0.0,
                dynamic_friction=0.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.35, 0.15), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, args_cli.ball_height)),
    )
    return RigidObject(cfg=ball_cfg)


def clear_ball_wrench(ball: RigidObject):
    """Clear the persistent external wrench on the ball."""
    zero_force = torch.zeros((ball.num_instances, ball.num_bodies, 3), device=ball.device)
    zero_torque = torch.zeros_like(zero_force)
    ball.permanent_wrench_composer.set_forces_and_torques(
        forces=zero_force,
        torques=zero_torque,
        is_global=True,
    )


def reset_ball(ball: RigidObject, default_root_state: torch.Tensor):
    """Restore the ball to its default state and clear external forces."""
    ball.write_root_pose_to_sim(default_root_state[:, :7])
    ball.write_root_velocity_to_sim(default_root_state[:, 7:])
    ball.reset()
    clear_ball_wrench(ball)
    ball.update(0.0)


def step_ball_with_spring(
    sim: SimulationContext,
    ball: RigidObject,
    spring_plane: VerticalSpringPlane,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the custom spring force and advance the simulation by one physics step."""
    force_w, penetration, bottom_height = spring_plane.compute_force(
        ball.data.root_pos_w,
        ball.data.root_lin_vel_w,
    )
    zero_torque = torch.zeros_like(force_w)
    ball.permanent_wrench_composer.set_forces_and_torques(
        forces=force_w.unsqueeze(1),
        torques=zero_torque.unsqueeze(1),
        is_global=True,
    )
    ball.write_data_to_sim()
    sim.step()
    ball.update(sim.get_physics_dt())
    return force_w, penetration, bottom_height


def run_simulator(sim: SimulationContext, ball: RigidObject, spring_plane: VerticalSpringPlane):
    """Run the custom spring plane demo."""
    default_root_state = ball.data.default_root_state.clone()
    step_count = 0

    while simulation_app.is_running():
        if step_count % args_cli.reset_interval == 0:
            reset_ball(ball, default_root_state)
            print("[INFO]: Resetting ball state...")
            step_count = 0

        force_w, penetration, _ = step_ball_with_spring(sim, ball, spring_plane)
        step_count += 1

        if step_count % args_cli.print_interval == 0:
            print(
                f"[INFO]: Ball z = {ball.data.root_pos_w[0, 2].item():.3f}, "
                f"vz = {ball.data.root_lin_vel_w[0, 2].item():.3f}, "
                f"penetration = {penetration[0].item():.4f}, "
                f"force_z = {force_w[0, 2].item():.3f}"
            )


def main():
    """Main entry-point for the ball-drop spring demo."""
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[4.0, -3.5, 2.5], target=[0.0, 0.0, args_cli.plane_height])

    ball = design_scene()
    sim.reset()

    spring_plane = VerticalSpringPlane(
        VerticalSpringPlaneCfg(
            plane_height=args_cli.plane_height,
            stiffness=args_cli.stiffness,
            damping=args_cli.damping,
            contact_radius=args_cli.ball_radius,
        )
    )

    default_root_state = ball.data.default_root_state.clone()
    reset_ball(ball, default_root_state)
    print(
        "[INFO]: Starting spring plane demo with "
        f"k={args_cli.stiffness:.1f}, c={args_cli.damping:.1f}, "
        f"plane_height={args_cli.plane_height:.3f}, ball_height={args_cli.ball_height:.3f}"
    )
    print(
        "[INFO]: Future training integration should update the spring force at every physics substep, "
        "for example in ActionTerm.apply_actions() or a scene-side updater instead of events.interval."
    )
    run_simulator(sim, ball, spring_plane)


if __name__ == "__main__":
    main()
    simulation_app.close()
