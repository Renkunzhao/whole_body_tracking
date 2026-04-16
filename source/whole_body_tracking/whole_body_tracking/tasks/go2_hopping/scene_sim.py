"""Minimal PD-control smoke test for the Go2 hopping FlatSceneCfg.

Loads ``FlatSceneCfg`` with ``num_envs=1``, drives every joint to its default
position via the GO2 articulation's built-in PD gains, and steps the simulation
continuously. Useful for eyeballing the terrain, robot spawn, and contact
sensor before running the full RL env.

Run:

    python scripts/go2_hopping_flat_pd.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PD control smoke test for Go2 on the hopping scene.")
parser.add_argument("--reset_interval", type=int, default=5000, help="Steps between resets.")
parser.add_argument(
    "--scene", type=str, default="trampoline", choices=["flat", "trampoline"], help="Which scene to load."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from whole_body_tracking.robots.go2 import GO2_FOOT_BODY_NAMES
from whole_body_tracking.tasks.go2_hopping.go2_hopping_env_cfg import FlatSceneCfg, TrampolineSceneCfg
from whole_body_tracking.utils.trampoline_deformable import build_trampoline_kinematic_targets


def reset_robot(scene: InteractiveScene) -> None:
    robot = scene["robot"]
    env_ids = torch.arange(scene.num_envs, device=robot.device, dtype=torch.long)
    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, 0:3] += scene.env_origins
    robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(),
        robot.data.default_joint_vel.clone(),
        env_ids=env_ids,
    )
    robot.reset()


def pin_trampoline(scene: InteractiveScene) -> None:
    trampoline = scene["trampoline"]
    targets, _, _ = build_trampoline_kinematic_targets(
        trampoline.data.default_nodal_state_w,
        trampoline.data.nodal_kinematic_target,
    )
    trampoline.write_nodal_kinematic_target_to_sim(targets)


def main() -> None:
    sim_dt = 0.005
    sim = SimulationContext(sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device))
    sim.set_camera_view(eye=[2.0, 2.0, 1.2], target=[0.0, 0.0, 0.3])

    if args_cli.scene == "flat":
        scene_cfg = FlatSceneCfg(num_envs=1, env_spacing=2.5)
        sim.cfg.physics_material = scene_cfg.terrain.physics_material
    else:
        scene_cfg = TrampolineSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.update(sim_dt)

    robot = scene["robot"]
    reset_robot(scene)
    if args_cli.scene == "trampoline":
        pin_trampoline(scene)

    foot_ids, foot_names = robot.find_bodies(list(GO2_FOOT_BODY_NAMES), preserve_order=True)
    print(f"[INFO]: Loaded {args_cli.scene} scene with {scene.num_envs} env. Driving joints to default pose.")
    print(f"[INFO]: Resolved foot bodies: {foot_names} -> ids {foot_ids}")

    step = 0
    while simulation_app.is_running():
        if step > 0 and step % args_cli.reset_interval == 0:
            reset_robot(scene)
            if args_cli.scene == "trampoline":
                pin_trampoline(scene)
            print(f"[INFO]: Reset at step {step}.")

        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        robot.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if step % 50 == 0:
            foot_pos_w = robot.data.body_pos_w[0, foot_ids]  # [4, 3], env 0
            parts = [f"{name}=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})" for name, p in zip(foot_names, foot_pos_w.tolist())]
            print(f"[step {step:5d}] feet_w: " + "  ".join(parts))

        step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
