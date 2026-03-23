"""Load G1 on a deformable trampoline and hold the default standing pose.

Examples:

    ./isaaclab.sh -p scripts/load_g1_trampoline.py
    ./isaaclab.sh -p scripts/load_g1_trampoline.py --passive
    ./isaaclab.sh -p scripts/load_g1_trampoline.py --randomize_on_reset
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Load G1 on a deformable trampoline.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of scene instances to spawn.")
parser.add_argument(
    "--passive",
    action="store_true",
    default=False,
    help="If set, do not command the default standing pose.",
)
parser.add_argument("--reset_interval", type=int, default=500, help="Number of simulation steps between resets.")
parser.add_argument(
    "--youngs_modulus",
    type=float,
    default=8.0e4,
    help="Fixed Young's modulus used when --randomize_on_reset is disabled.",
)
parser.add_argument(
    "--mass",
    type=float,
    default=10.0,
    help="Fixed trampoline mass used when --randomize_on_reset is disabled.",
)
parser.add_argument(
    "--sim_resolution",
    type=int,
    default=10,
    help="Hexahedral simulation resolution for the deformable trampoline.",
)
parser.add_argument(
    "--randomize_on_reset",
    action="store_true",
    default=False,
    help="If set, sample trampoline Young's modulus and mass on every reset.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaacsim.core.utils.stage import get_current_stage
from pxr import Sdf

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.utils.trampoline_deformable import (
    TRAMPOLINE_DR_MASS_RANGE,
    TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE,
    TRAMPOLINE_PIN_WIDTH,
    build_trampoline_kinematic_targets,
    make_trampoline_cfg,
    set_trampoline_youngs_moduli,
    trampoline_mesh_prim_path,
)


@configclass
class SmokeSceneCfg(InteractiveSceneCfg):
    robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    trampoline = make_trampoline_cfg("{ENV_REGEX_NS}/Trampoline")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


def resolve_trampoline_mass_attrs(trampoline) -> list:
    stage = get_current_stage()
    attrs = []
    for prim_path in sim_utils.find_matching_prim_paths(trampoline_mesh_prim_path(trampoline.cfg.prim_path)):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Invalid trampoline mesh prim: '{prim_path}'.")
        attr = prim.GetAttribute("physics:mass")
        if not attr.IsValid():
            attr = prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float)
        attrs.append(attr)
    return attrs


def apply_trampoline_properties(trampoline, mass_attrs: list, env_ids: torch.Tensor) -> tuple[float, float]:
    if args_cli.randomize_on_reset:
        youngs_moduli = sample_uniform(
            TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE[0],
            TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE[1],
            (len(env_ids),),
            device=trampoline.device,
        ).to(dtype=torch.float32)
        masses = sample_uniform(
            TRAMPOLINE_DR_MASS_RANGE[0],
            TRAMPOLINE_DR_MASS_RANGE[1],
            (len(env_ids),),
            device=trampoline.device,
        ).to(dtype=torch.float32)
    else:
        youngs_moduli = torch.full((len(env_ids),), args_cli.youngs_modulus, device=trampoline.device, dtype=torch.float32)
        masses = torch.full((len(env_ids),), args_cli.mass, device=trampoline.device, dtype=torch.float32)

    set_trampoline_youngs_moduli(trampoline.material_physx_view, youngs_moduli, env_ids)
    with Sdf.ChangeBlock():
        for env_id, mass in zip(env_ids.tolist(), masses.tolist(), strict=True):
            mass_attrs[env_id].Set(float(mass))

    return float(youngs_moduli[0].item()), float(masses[0].item())


def reset_scene(scene: InteractiveScene, robot, trampoline, trampoline_targets: torch.Tensor, mass_attrs: list) -> tuple[float, float]:
    env_ids = torch.arange(scene.num_envs, device=robot.device, dtype=torch.long)

    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, 0:3] += scene.env_origins
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()

    robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
    robot.reset()

    trampoline.write_nodal_state_to_sim(trampoline.data.default_nodal_state_w, env_ids=env_ids)
    youngs_modulus, mass = apply_trampoline_properties(trampoline, mass_attrs, env_ids)
    trampoline.write_nodal_kinematic_target_to_sim(trampoline_targets, env_ids=env_ids)
    trampoline.reset(env_ids=env_ids)
    return youngs_modulus, mass


def main() -> None:
    scene_cfg = SmokeSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=False)
    scene_cfg.trampoline = make_trampoline_cfg(
        "{ENV_REGEX_NS}/Trampoline",
        mass=args_cli.mass,
        youngs_modulus=args_cli.youngs_modulus,
        sim_resolution=args_cli.sim_resolution,
    )

    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    sim.set_camera_view(eye=[2.5, 2.5, 1.8], target=[0.0, 0.0, 0.7])
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.update(dt=sim.get_physics_dt())

    robot = scene["robot"]
    trampoline = scene["trampoline"]
    if trampoline.material_physx_view is None:
        raise RuntimeError("Trampoline smoke scene failed to create a deformable material view.")

    trampoline_targets, _, center_node_ids = build_trampoline_kinematic_targets(
        trampoline.data.default_nodal_state_w,
        trampoline.data.nodal_kinematic_target,
        pin_width=TRAMPOLINE_PIN_WIDTH,
    )
    mass_attrs = resolve_trampoline_mass_attrs(trampoline)
    youngs_modulus, mass = reset_scene(scene, robot, trampoline, trampoline_targets, mass_attrs)

    print(
        f"[INFO]: G1 trampoline scene loaded — num_envs={args_cli.num_envs}, passive={args_cli.passive}, "
        f"youngs_modulus={youngs_modulus:.1f}, mass={mass:.3f}"
    )

    sim_dt = sim.get_physics_dt()
    step = 0
    while simulation_app.is_running():
        if step > 0 and step % args_cli.reset_interval == 0:
            youngs_modulus, mass = reset_scene(scene, robot, trampoline, trampoline_targets, mass_attrs)
            print(
                f"[INFO]: Resetting trampoline scene — youngs_modulus={youngs_modulus:.1f}, mass={mass:.3f}"
            )

        trampoline.write_nodal_kinematic_target_to_sim(trampoline_targets)
        if not args_cli.passive:
            robot.set_joint_position_target(robot.data.default_joint_pos.clone())

        robot.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if step % 200 == 0:
            center_height = trampoline.data.nodal_pos_w[0, center_node_ids[0], 2].item()
            print(f"[INFO]: step={step}, trampoline center z={center_height:.3f}")

        step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
