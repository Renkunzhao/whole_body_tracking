# Trampoline Structure

This file is the current architecture map for trampoline-related work in
`whole_body_tracking`.

## Canonical Paths

- Go2 continuous rebounce on a deformable trampoline is the current main
  research path.
- The deformable trampoline is implemented with Isaac Lab `DeformableObject`
  plus rim pinning; rigid-deformable contact sensors are not used for task
  logic.
- The custom spring-damper contact model remains an experimental/legacy path.

## Main Entry Points

- `source/whole_body_tracking/whole_body_tracking/tasks/go2_hopping/go2_rebounce_env_cfg.py`
  defines the Go2 rebounce task, observations, rewards, terminations, and
  trampoline DR ranges.
- `source/whole_body_tracking/whole_body_tracking/tasks/go2_hopping/mdp/commands.py`
  owns the rebounce command state, valid-apex state machine, height metrics,
  and energy metrics.
- `source/whole_body_tracking/whole_body_tracking/tasks/go2_hopping/mdp/rewards.py`
  consumes command-owned apex pulses for height and energy rewards.
- `source/whole_body_tracking/whole_body_tracking/tasks/go2_hopping/mdp/terminations.py`
  consumes command-owned apex pulses for no-valid-apex timeout.
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/trampoline_events.py`
  randomizes deformable trampoline material/mass and reapplies pinning.
- `source/whole_body_tracking/whole_body_tracking/utils/trampoline_deformable.py`
  builds the shared deformable trampoline config and wraps material property
  getters/setters.
- `scripts/rsl_rl/play-rebounce.py` is the visual play/debug script.
- `scripts/rsl_rl/eval-rebounce.py` is the headless fixed-condition evaluation
  script.
- `source/whole_body_tracking/whole_body_tracking/tasks/go2_hopping/go2_rebounce.tex`
  is the living experiment note.

## Current Task State

- Episode length: 20 s.
- Target apex height and initial drop height are sampled independently.
- Apex termination has been removed; success is timeout without failure.
- The command term is the single owner of apex state.
- Valid apex detection uses vertical velocity sign change plus foot-clearance
  geometry, not contact sensors.
- The actor observation is deployable: target command, projected gravity, base
  angular velocity, joint state, and previous action. The critic remains
  privileged with base position, quaternion, and linear velocity.
- The current trampoline DR parameters are Young's modulus, mass, dynamic
  friction, elasticity damping, damping scale, and Poisson ratio.

## Evaluation State

- `eval-rebounce.py` supports fixed-condition Cartesian-product sweeps through
  repeated canonical `--sweep PARAM VALUE...` arguments.
- CSV rows are written per condition with three-decimal formatting and the
  wandb run id appended to the output name.
- The useful stress cases are high target height and high material damping;
  moderate target-height cases are already easy for MLP+DR.

## Next Research Sequence

1. Keep deployable-observation MLP+DR as the current baseline.
2. Add MLP observation history.
3. Test RNN policies under the same deployable observation.
4. Move to RMA/latent dynamics estimation if history or recurrence shows a
   meaningful benefit.
5. Measure real-robot observation noise and update corruption ranges.

## Invariants

- Do not provide true trampoline parameters directly to the deployable actor.
- Reward and termination terms should consume command-owned event state instead
  of maintaining duplicate apex detectors.
- Prefer explicit canonical CLI/config names; do not add alias tables unless
  backwards compatibility is explicitly requested.
