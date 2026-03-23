"""Load a single G1 robot on flat ground and hold the default standing pose.

A minimal stand-alone script that mirrors the scene setup used by the RL
training / play pipeline (same robot config, same sim parameters, same
terrain material) but without the full RL environment overhead.

Examples:

    ./isaaclab.sh -p scripts/load_g1.py
    ./isaaclab.sh -p scripts/load_g1.py --num_envs 4
    ./isaaclab.sh -p scripts/load_g1.py --passive
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Load G1 on flat ground.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robots to spawn.")
parser.add_argument(
    "--passive",
    action="store_true",
    default=False,
    help="If set, do not command the default standing pose (robot will collapse under gravity).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

ROBOT_PRIM_PATH = "/World/G1"
ENV_SPACING = 2.5


def design_scene(num_envs: int) -> Articulation:
    """Spawn flat ground, lighting, and G1 robot(s)."""
    # -- ground (matches tracking_env_cfg.py)
    terrain_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        num_envs=num_envs,
        env_spacing=ENV_SPACING,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    TerrainImporter(terrain_cfg)

    # -- lights (matches tracking_env_cfg.py)
    sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0).func(
        "/World/light", sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
    )
    sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0).func(
        "/World/skyLight", sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0)
    )

    # -- robot
    robot_cfg = G1_CYLINDER_CFG.replace(prim_path=ROBOT_PRIM_PATH)
    return Articulation(robot_cfg)


def reset_robot(robot: Articulation) -> None:
    """Reset to default root state and joint positions."""
    root_state = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()


def run_simulator(sim: SimulationContext, robot: Articulation) -> None:
    """Step the simulation, optionally commanding the default standing pose."""
    sim_dt = sim.get_physics_dt()
    step = 0

    while simulation_app.is_running():
        if step % 500 == 0:
            reset_robot(robot)

        if not args_cli.passive:
            robot.set_joint_position_target(robot.data.default_joint_pos.clone())

        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        step += 1


def main() -> None:
    # sim settings aligned with tracking_env_cfg (dt=0.005)
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 1.5], target=[0.0, 0.0, 0.7])

    robot = design_scene(args_cli.num_envs)

    sim.reset()
    reset_robot(robot)
    print(
        f"[INFO]: G1 loaded — num_envs={args_cli.num_envs}, passive={args_cli.passive}, "
    )
    run_simulator(sim, robot)


if __name__ == "__main__":
    main()
    simulation_app.close()
