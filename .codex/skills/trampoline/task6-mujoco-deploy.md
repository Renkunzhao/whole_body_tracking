# Task 6: Mujoco Deploy Gap

This document tracks the Go2 rebounce deployment thread after the IsaacLab
baseline moved toward Mujoco and eventually hardware.

## Current Setup

- IsaacLab side: `Go2-Rebounce-Trampoline-Baseline` with deployable 43-dim
  observation can remain stable on many randomized IsaacLab deformable
  trampoline conditions.
- Mujoco trampoline:
  `/home/rkz/code/unitree_ws/src/unitree_mujoco/unitree_robots/go2/trampoline.xml`.
- Mujoco deploy config:
  `/home/rkz/code/unitree_ws/src/legged_rl_deploy/policies/go2/wbt/rebounce/baseline/config.yaml`.
- Deploy config currently uses:
  - `ll_dt = 0.005`;
  - `policy_dt = 0.02`;
  - ONNX policy with `input_dim = 43`, `output_dim = 12`;
  - observation terms matching the IsaacLab deployable policy group:
    `hop_command`, `projected_gravity`, `base_ang_vel_B`, `joint_pos`,
    `joint_vel`, `last_action`;
  - joint order mapping from IsaacLab order to Unitree low-level motor order.
- Observed issue: the IsaacLab DR baseline does not reliably jump on the
  Mujoco trampoline. It often falls, with orientation control especially weak.

## Working Hypothesis

Do not assume the first solution is simply "make DR wider".

There are three distinct failure classes:

1. Deploy plumbing mismatch.
   Observation order, frame convention, scaling, joint mapping, default joint
   offsets, action scaling, previous-action semantics, policy hold, PD gains,
   torque limits, latency, and IMU conventions can each break a policy even if
   the policy is good.

2. Physics-family mismatch.
   IsaacLab deformable trampoline DR randomizes material and mass parameters
   inside one simulator/model family. Mujoco `flexcomp` may have different
   spatial modes, contact impulse timing, solver damping, foot-contact
   behavior, and center-patch phase response. A scalar DR range in Isaac does
   not automatically cover a different trampoline model family.

3. Partial-observation adaptation limit.
   The baseline actor has no direct trampoline phase, center velocity, contact
   force, or base linear velocity. It can only infer trampoline state through
   proprioception and projected gravity. That may work inside Isaac's coherent
   DR family, but fail when Mujoco produces an out-of-family phase/contact
   response, especially during asymmetric touchdown and release.

## First Debug Priority

Before changing training, make Mujoco and Isaac logs comparable.

### Geometry Alignment

Start by aligning the geometric convention before tuning dynamics.

- Real trampoline radius target: `1.5 m`.
- IsaacLab deformable trampoline uses `TRAMPOLINE_RADIUS = 1.5` and
  `TRAMPOLINE_THICKNESS = 0.03`. The thickness is a simulation/contact
  thickness to reduce membrane penetration, not the literal fabric thickness.
- Keep the IsaacLab top surface near `z=0` by using
  `TRAMPOLINE_CENTER_Z = -0.5 * TRAMPOLINE_THICKNESS`.
- Mujoco `flexcomp radius` is not the full trampoline radius. For the current
  `type="disc" count="3 3 1"` setup, the full disc radius is controlled mostly
  by `spacing`; the current `spacing="1.5 1.5 1.5"` gives an approximately
  `1.5 m` radius.
- Mujoco `flexcomp radius` should be treated as the element/contact radius.
  Use `radius=".05"` for an approximate `0.1 m` contact thickness, and set the
  flex `pos` z to `-.05` so its top contact surface remains around `z=0`.
- IsaacLab uses a volumetric FEM deformable object. The exact node count is
  generated from `simulation_hexahedral_resolution`, and the current pinning
  marks the outer radial band across the volume as kinematic.
- The current Mujoco file uses `flexcomp type="disc" dim="2" count="3 3 1"`,
  so it is a single-layer 2D flex with eight pinned rim nodes and only one free
  center node. This topology is intentionally simple, but it is not a close
  structural match to the IsaacLab FEM trampoline.
- `play-rebounce.py` visualizes IsaacLab trampoline nodes by default: pinned
  nodes are red and free nodes are green. Use this view before deciding whether
  to increase Mujoco `count` or retune the flexible equality constraints.
- Geometry DR should distinguish reset-time parameters from cooked-mesh
  parameters. `pin_width_range` is reset-randomizable because it only changes
  which nodes are kinematic. True FEM mesh thickness and
  `simulation_hexahedral_resolution` are fixed at spawn/cooking time, so Go2
  rebounce uses spawn-time cooked-geometry buckets for those axes.

1. Validate deploy plumbing on a simple condition:
   - verify 43-dim observation order and scale against IsaacLab export;
   - check projected gravity frame/sign;
   - check base angular velocity body/world frame and scale;
   - check joint order after `joint_ids_map`;
   - check `joint_pos` offset equals IsaacLab default joint position handling;
   - check `last_action` is the raw policy output, not scaled target position;
   - check action scale/offset and PD gains match IsaacLab play behavior;
   - check torque saturation and command update timing.

2. Run a short sim2sim observation/action comparison:
   - same reset pose, same target height, same first policy steps;
   - log observation vector, action vector, joint targets, q, dq, body attitude;
   - compare signs and magnitudes term by term.

3. Calibrate trampoline response, not just XML parameters:
   - ball-drop or robot-drop test at fixed height;
   - compare contact duration, maximum compression, rebound height,
     center `z/vz`, total impulse, peak force, and damping ratio;
   - tune Mujoco trampoline XML or expand Isaac DR only after this response
     mismatch is measured.

## DR Strategy

Wider DR is useful only after the missing axes are represented.

Reasonable DR additions if Mujoco differs:

- contact friction/restitution and foot contact geometry;
- trampoline rest height and contact surface offset;
- effective stiffness/damping mapped from measured contact duration and
  rebound ratio;
- lateral/roll-pitch contact asymmetry and touchdown offset;
- motor strength, PD gain, torque limit, action latency, observation latency;
- IMU noise/bias and projected-gravity corruption.

Blindly widening Young's modulus, mass, and damping may produce harder
training without covering the actual Mujoco failure.

## Research Direction

If deploy plumbing and trampoline calibration are reasonable but Mujoco still
fails, treat this as a modeling/adaptation problem.

Teacher side:

- Give a privileged teacher trampoline phase information:
  center or local patch `z`, `vz`, compression, release/compression flag,
  DOB/contact force, and optionally stance impulse statistics.
- Add phase-aware rewards:
  - reward useful impulse during release phase;
  - penalize bad-phase positive work while trampoline is still moving down;
  - penalize braking during trampoline release;
  - keep orientation stable through touchdown, max compression, and liftoff.
- Preserve deployable student observation; privileged signals are teacher-only
  unless real hardware has equivalent sensors.

Student side:

- Distill teacher into a deployable student with history, GRU/RNN, or RMA-style
  latent dynamics estimation.
- The intended latent should encode trampoline phase/dynamics from
  proprioception, not true XML/material parameters.
- Compare against the plain MLP+DR baseline to see whether the gap is actually
  caused by missing temporal inference.

## Orientation-Specific Notes

Poor orientation control in Mujoco likely comes from stance-phase contact
timing and asymmetric foot impulse, not only from airborne posture.

Useful diagnostics:

- roll/pitch RMS and peak around touchdown, max compression, and liftoff;
- per-foot vertical impulse during stance;
- center-of-pressure or force asymmetry if available;
- action and torque saturation near touchdown;
- trampoline center/patch phase at touchdown and liftoff;
- base angular velocity right after contact.

Potential training changes:

- stance-gated roll/pitch or projected-gravity reward around touchdown and
  release;
- touchdown posture regularization;
- left-right impulse symmetry or foot force symmetry during stance;
- phase-aware work penalty so the robot does not actively create the wrong
  trampoline phase.

## Recommended Sequence

1. Make Mujoco deploy logs comparable with IsaacLab play logs.
2. Fix any observation/action/PD/timing mismatch first.
3. Calibrate Mujoco trampoline response against a fixed IsaacLab condition.
4. Add targeted DR axes for the measured gap.
5. If baseline still fails, train a phase-aware privileged teacher.
6. Distill to a deployable history/RNN/RMA student.
