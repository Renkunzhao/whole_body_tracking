# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trampoline demo built from a deformable disk and a rigid ball.

This is an Isaac Lab analogue of the MuJoCo ``flexcomp`` trampoline:

- a thin deformable cylinder acts as the trampoline membrane
- nodes near the outer rim are pinned through kinematic targets
- a rigid sphere is dropped onto the center and periodically reset

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/01_assets/trampoline.py

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse
import json
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Drop a rigid ball onto a pinned deformable trampoline.")
parser.add_argument("--reset_interval", type=int, default=360, help="Number of simulation steps between resets.")
parser.add_argument(
    "--pin_width",
    type=float,
    default=0.7,
    help="Width of the pinned rim on the trampoline in meters.",
)
parser.add_argument(
    "--youngs_modulus",
    type=float,
    default=8.0e2,
    help="Young's modulus of the deformable trampoline material.",
)
parser.add_argument(
    "--sim_resolution",
    type=int,
    default=5,
    help="Hexahedral simulation resolution for the deformable trampoline.",
)
parser.add_argument(
    "--show_sim_nodes",
    action="store_true",
    help="Visualize simulation mesh nodes of the trampoline.",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=120,
    help="Number of warmup steps before timing when benchmark mode is enabled.",
)
parser.add_argument(
    "--benchmark_steps",
    type=int,
    default=0,
    help="If greater than zero, run a fixed-length benchmark and exit after printing metrics.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import SimulationContext

TRAMPOLINE_RADIUS = 1.5
TRAMPOLINE_THICKNESS = 0.08
TRAMPOLINE_HEIGHT = 0.75
TRAMPOLINE_MASS = 1.0
BALL_RADIUS = 0.18
BALL_MASS = 6.0
BALL_HEIGHT = 1.55

SIM_NODE_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/TrampolineSimNodes",
    markers={
        "node": sim_utils.SphereCfg(
            radius=0.012,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.15, 0.15)),
        ),
    },
)


def design_scene(debug_vis: bool = True) -> dict[str, DeformableObject | RigidObject]:
    """Design the scene."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    trampoline_cfg = DeformableObjectCfg(
        prim_path="/World/Trampoline",
        spawn=sim_utils.MeshCylinderCfg(
            radius=TRAMPOLINE_RADIUS,
            height=TRAMPOLINE_THICKNESS,
            axis="Z",
            mass_props=sim_utils.MassPropertiesCfg(mass=TRAMPOLINE_MASS),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                solver_position_iteration_count=24,
                vertex_velocity_damping=0.05,
                sleep_damping=1.0,
                sleep_threshold=0.01,
                settling_threshold=0.02,
                self_collision=False,
                simulation_hexahedral_resolution=args_cli.sim_resolution,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.95), metallic=0.05),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                dynamic_friction=0.8,
                youngs_modulus=args_cli.youngs_modulus,
                poissons_ratio=0.35,
                elasticity_damping=0.02,
                damping_scale=1.0,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, TRAMPOLINE_HEIGHT)),
        debug_vis=debug_vis,
    )
    trampoline = DeformableObject(cfg=trampoline_cfg)

    ball_cfg = RigidObjectCfg(
        prim_path="/World/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.35, 0.15), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, BALL_HEIGHT)),
    )
    ball = RigidObject(cfg=ball_cfg)

    return {"trampoline": trampoline, "ball": ball}


def build_trampoline_targets(
    trampoline: DeformableObject, pin_width: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create kinematic targets that pin the outer rim of the trampoline."""
    targets = trampoline.data.nodal_kinematic_target.clone()
    targets[..., :3] = trampoline.data.default_nodal_state_w[..., :3]
    targets[..., 3] = 1.0

    nodal_pos = trampoline.data.default_nodal_state_w[..., :3]
    center_xy = nodal_pos[..., :2].mean(dim=1, keepdim=True)
    radial_distance = torch.linalg.vector_norm(nodal_pos[..., :2] - center_xy, dim=-1)

    rim_radius = radial_distance.max(dim=1, keepdim=True).values
    pin_threshold = torch.clamp(rim_radius - pin_width, min=0.0)
    pinned_mask = radial_distance >= pin_threshold
    center_node_ids = radial_distance.argmin(dim=1)

    targets[..., 3] = torch.where(
        pinned_mask,
        torch.zeros_like(targets[..., 3]),
        torch.ones_like(targets[..., 3]),
    )
    return targets, pinned_mask, center_node_ids


def reset_scene(
    trampoline: DeformableObject,
    ball: RigidObject,
    trampoline_targets: torch.Tensor,
    ball_default_state: torch.Tensor,
):
    """Reset the deformable trampoline and the rigid ball."""
    trampoline.write_nodal_state_to_sim(trampoline.data.default_nodal_state_w)
    trampoline.write_nodal_kinematic_target_to_sim(trampoline_targets)
    trampoline.reset()

    ball.write_root_pose_to_sim(ball_default_state[:, :7])
    ball.write_root_velocity_to_sim(ball_default_state[:, 7:])
    ball.reset()


def update_node_visualizers(
    trampoline: DeformableObject,
    sim_node_visualizer: VisualizationMarkers | None,
):
    """Update optional node visualizers."""
    if sim_node_visualizer is not None:
        sim_node_visualizer.visualize(translations=trampoline.data.nodal_pos_w[0])


def step_simulation(
    sim: sim_utils.SimulationContext,
    sim_dt: float,
    trampoline: DeformableObject,
    ball: RigidObject,
    trampoline_targets: torch.Tensor,
    sim_node_visualizer: VisualizationMarkers | None,
):
    """Advance the trampoline scene by one physics step."""
    trampoline.write_nodal_kinematic_target_to_sim(trampoline_targets)
    ball.write_data_to_sim()

    sim.step()

    trampoline.update(sim_dt)
    ball.update(sim_dt)
    update_node_visualizers(trampoline, sim_node_visualizer)


def maybe_synchronize(device: str):
    """Synchronize CUDA work before timing summaries when applicable."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)


def benchmark_simulator(
    sim: sim_utils.SimulationContext,
    trampoline: DeformableObject,
    ball: RigidObject,
    trampoline_targets: torch.Tensor,
    ball_default_state: torch.Tensor,
    center_node_ids: torch.Tensor,
    pinned_mask: torch.Tensor,
):
    """Run a fixed number of steps and report wall-clock performance."""
    sim_dt = sim.get_physics_dt()

    reset_scene(trampoline, ball, trampoline_targets, ball_default_state)
    print(
        f"[INFO]: Starting warmup for resolution {args_cli.sim_resolution} "
        f"({args_cli.warmup_steps} steps)...",
        flush=True,
    )

    for _ in range(args_cli.warmup_steps):
        step_simulation(sim, sim_dt, trampoline, ball, trampoline_targets, sim_node_visualizer=None)

    print(
        f"[INFO]: Warmup complete. Starting timed benchmark "
        f"({args_cli.benchmark_steps} steps)...",
        flush=True,
    )
    maybe_synchronize(args_cli.device)
    start_time = time.perf_counter()
    for _ in range(args_cli.benchmark_steps):
        step_simulation(sim, sim_dt, trampoline, ball, trampoline_targets, sim_node_visualizer=None)
    maybe_synchronize(args_cli.device)
    elapsed = time.perf_counter() - start_time

    center_height = trampoline.data.nodal_pos_w[0, center_node_ids[0], 2].item()
    return {
        "resolution": args_cli.sim_resolution,
        "warmup_steps": args_cli.warmup_steps,
        "benchmark_steps": args_cli.benchmark_steps,
        "wall_time_s": elapsed,
        "step_time_ms": elapsed * 1000.0 / args_cli.benchmark_steps,
        "steps_per_s": args_cli.benchmark_steps / elapsed,
        "sim_seconds_per_second": (args_cli.benchmark_steps * sim_dt) / elapsed,
        "sim_vertices": trampoline.max_sim_vertices_per_body,
        "sim_elements": trampoline.max_sim_elements_per_body,
        "collision_vertices": trampoline.max_collision_vertices_per_body,
        "collision_elements": trampoline.max_collision_elements_per_body,
        "pinned_nodes": int(pinned_mask.sum().item()),
        "total_nodes": int(pinned_mask.numel()),
        "ball_height": ball.data.root_pos_w[0, 2].item(),
        "center_height": center_height,
    }


def run_simulator(
    sim: sim_utils.SimulationContext,
    trampoline: DeformableObject,
    ball: RigidObject,
    trampoline_targets: torch.Tensor,
    ball_default_state: torch.Tensor,
    center_node_ids: torch.Tensor,
    sim_node_visualizer: VisualizationMarkers | None,
):
    """Run the simulation loop."""
    sim_dt = sim.get_physics_dt()
    step_count = 0

    while simulation_app.is_running():
        if step_count % args_cli.reset_interval == 0:
            reset_scene(trampoline, ball, trampoline_targets, ball_default_state)
            update_node_visualizers(trampoline, sim_node_visualizer)
            print("[INFO]: Resetting trampoline and ball...")
            step_count = 0

        step_simulation(sim, sim_dt, trampoline, ball, trampoline_targets, sim_node_visualizer)
        step_count += 1

        if step_count % 60 == 0:
            center_height = trampoline.data.nodal_pos_w[0, center_node_ids[0], 2].item()
            print(
                f"[INFO]: Ball z = {ball.data.root_pos_w[0, 2].item():.3f}, "
                f"trampoline center z = {center_height:.3f}"
            )


def main():
    """Main function."""
    benchmark_mode = args_cli.benchmark_steps > 0
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[4.5, -4.0, 2.7], target=[0.0, 0.0, TRAMPOLINE_HEIGHT])

    setup_start_time = time.perf_counter()
    scene_entities = design_scene(debug_vis=not benchmark_mode)

    sim.reset()

    trampoline = scene_entities["trampoline"]
    ball = scene_entities["ball"]
    sim_node_visualizer = None
    if args_cli.show_sim_nodes and not benchmark_mode:
        sim_node_visualizer = VisualizationMarkers(SIM_NODE_MARKER_CFG)

    trampoline_targets, pinned_mask, center_node_ids = build_trampoline_targets(
        trampoline, pin_width=args_cli.pin_width
    )
    ball_default_state = ball.data.default_root_state.clone()
    setup_elapsed_s = time.perf_counter() - setup_start_time

    pinned_node_count = int(pinned_mask.sum().item())
    total_node_count = pinned_mask.numel()
    print("[INFO]: Setup complete...")
    print(
        "[INFO]: Simulation mesh: "
        f"{trampoline.max_sim_vertices_per_body} vertices, "
        f"{trampoline.max_sim_elements_per_body} elements."
    )
    print(
        "[INFO]: Collision mesh: "
        f"{trampoline.max_collision_vertices_per_body} vertices, "
        f"{trampoline.max_collision_elements_per_body} elements."
    )
    print(f"[INFO]: Pinned {pinned_node_count} / {total_node_count} trampoline nodes.")
    print(f"[INFO]: Setup time: {setup_elapsed_s:.3f} s", flush=True)

    if benchmark_mode:
        benchmark_report = benchmark_simulator(
            sim,
            trampoline,
            ball,
            trampoline_targets,
            ball_default_state,
            center_node_ids,
            pinned_mask,
        )
        benchmark_report["setup_time_s"] = setup_elapsed_s
        print("[BENCHMARK_JSON]" + json.dumps(benchmark_report, sort_keys=True), flush=True)
        return

    reset_scene(trampoline, ball, trampoline_targets, ball_default_state)
    update_node_visualizers(trampoline, sim_node_visualizer)
    run_simulator(
        sim,
        trampoline,
        ball,
        trampoline_targets,
        ball_default_state,
        center_node_ids,
        sim_node_visualizer,
    )


if __name__ == "__main__":
    main()
    simulation_app.close(
        wait_for_replicator=False,
        skip_cleanup=args_cli.benchmark_steps > 0,
    )
