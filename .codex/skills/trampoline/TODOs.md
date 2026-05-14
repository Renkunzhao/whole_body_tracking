# Trampoline TODOs

This file is the incremental backlog and work log for trampoline-related work in `whole_body_tracking`.

## How To Use

- Keep open work items in the backlog section.
- Append newest progress entries to the work log section, ideally with dates.
- When one backlog item turns into a substantial implementation thread, capture the detailed design in a dedicated `task-*.md` file and link it from here.

## Code Preferences

- Prefer explicit, minimal interfaces over protective compatibility layers. For CLI parameters, use canonical names directly and avoid alias tables unless the user asks for backwards compatibility.

## Backlog

## Active: Go2 Continuous Rebounce

Current baseline:
- deformable trampoline with rim pinning;
- canonical PPO baseline task id `Go2-Rebounce-Trampoline-Baseline`;
- base-state PPO ablation task id `Go2-Rebounce-Trampoline-Baseline-Base`;
- independent drop height and target apex height;
- command-owned valid-apex state machine;
- deployable instantaneous actor observation without base position or base
  linear velocity;
- separate `Go2-Rebounce-Trampoline-history` env for flattened-history MLP
  ablations (`history_length=5`);
- privileged critic with base state;
- energy metrics/reward available; energy penalty is delayed until 1000 PPO
  iterations;
- temporary valid-apex bootstrap bonus is active before 1000 PPO iterations and
  then switched off by curriculum;
- trampoline material/pinning DR is delayed until 1000 PPO iterations; early
  resets use the fixed trampoline `(E=8e4, mass=10, mu=0.8, damping=0.02,
  nu=0.35)`;
- trampoline geometry varies by spawn-time buckets
  `(thickness, simulation_hexahedral_resolution)`, currently the 2x2 set
  `(0.03/0.10, 10/20)`, with each env assigned one cooked FEM mesh for the
  run;
- Young's modulus DR is conditioned on cooked simulation resolution:
  resolution 10 samples `(8e3, 8e4)`, and resolution 20 samples `(8e4, 8e5)`;
- `damping_scale` DR must stay within PhysX's `[0, 1]` range; current Go2
  rebounce uses `(0.5, 1.0)`;
- teacher task variants expose root position, base linear velocity, and
  normalized true trampoline material/geometry parameters to an instantaneous actor for
  privileged upper-bound/RMA experiments;
- fixed-condition evaluation through `scripts/rsl_rl/eval-rebounce.py`.

Near-term experiments:
- train and evaluate the RNN policy under the same deployable observation;
- test RMA/latent dynamics estimator after history/RNN baselines;
- evaluate high-target/high-damping stress conditions;
- measure real-robot observation noise and update corruption ranges.
- debug Mujoco deploy gap documented in
  [task6-mujoco-deploy.md](task6-mujoco-deploy.md): first verify deploy
  observation/action plumbing and trampoline response calibration, then decide
  whether wider DR is enough or phase-aware teacher/student training is needed.

## Maintenance

- Keep `go2_rebounce.tex` aligned with the current task definition and latest
  evaluation conclusions.
- Keep `Structure.md` focused on the current architecture, not early prototype
  notes.
- Move old implementation notes to task-specific docs instead of leaving them
  as active backlog.

## Work Log

- `YYYY-MM-DD`: Add new entries here in reverse chronological order.
- `2026-05-14`: Fixed two PhysX limits exposed by resolution-20 trampoline
  training: `damping_scale` is clamped/configured inside the valid `[0, 1]`
  interval, and trampoline rebounce raises PhysX `gpu_collision_stack_size` to
  `2**28` for contact-rich soft-body scenes.
- `2026-05-14`: Changed Go2 rebounce geometry DR to the 2x2 bucket set
  `thickness in {0.03, 0.10}` x `simulation_hexahedral_resolution in {10, 20}`
  and made Young's modulus DR resolution-conditioned: `(8e3, 8e4)` for
  resolution 10 and `(8e4, 8e5)` for resolution 20.
- `2026-05-14`: Added true spawn-time trampoline geometry buckets. Go2 rebounce
  now cycles cooked FEM meshes across envs with
  `(thickness, simulation_hexahedral_resolution)` buckets, keeps all top
  surfaces aligned at `z=0`, and uses per-env valid-node masks/center-node ids
  for mixed-resolution pinning and phase metrics.
- `2026-05-14`: Started Mujoco deploy thread. Mujoco trampoline and Go2 WBT
  rebounce deploy config exist, but the IsaacLab DR baseline that is stable in
  Isaac often falls in Mujoco, especially through poor orientation control.
  Added a dedicated sim2sim/research plan in `task6-mujoco-deploy.md`.
- `2026-05-14`: Added reset-time trampoline pinning DR plumbing for Go2
  rebounce: `pin_width_range` can randomize the kinematic rim width. True FEM
  mesh thickness remains a spawn/cooking-time parameter.
- `2026-05-05`: Split the flattened-history MLP ablation into a separate
  `Go2-Rebounce-Trampoline-history` environment; the base trampoline task is
  now instantaneous deployable observation.
- `2026-05-04`: Updated teacher observation variants to use instantaneous
  privileged actor inputs: root position, base linear velocity, and normalized
  true trampoline parameters.
- `2026-05-03`: Delayed trampoline parameter randomization until 1000 PPO
  iterations so early RNN training starts from the fixed trampoline that can
  discover rebounding.
- `2026-05-03`: Added a temporary valid-apex bootstrap reward for early
  RNN/partial-observation training; it is disabled after 1000 PPO iterations.
- `2026-05-02`: H=5 flattened-history MLP was close to no-history overall but
  under-jumped more at high target height; H=10 often became passive. Added
  RNN task/config as the next adaptation baseline.
- `2026-05-02`: Delayed `energy_penalty` curriculum until 1000 PPO iterations
  so RNN/MLP policies learn sustained rebounding before work minimization.
- `2026-05-01`: Enabled `history_length=10` for the deployable actor observation in Go2 rebounce. Critic remains instantaneous privileged state.
- `2026-05-01`: Go2 rebounce is now the main trampoline path. MLP+DR baseline works with deployable observations; next steps are observation history, RNN, then RMA.
- `2026-04-02`: Tracking-Trampoline-Go2-v0 and Tracking-Flat-Go2-Wo-State-Estimation-v0
