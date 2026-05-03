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
- independent drop height and target apex height;
- command-owned valid-apex state machine;
- deployable actor observation without base position or base linear velocity;
- flattened-history MLP ablation completed for `history_length=5` and
  `history_length=10`;
- privileged critic with base state;
- energy metrics/reward available; energy penalty is delayed until 1000 PPO
  iterations;
- temporary valid-apex bootstrap bonus is active before 1000 PPO iterations and
  then switched off by curriculum;
- trampoline DR is delayed until 1000 PPO iterations; early resets use the
  fixed trampoline `(E=8e4, mass=10, mu=0.8, damping=0.02, nu=0.35)`;
- fixed-condition evaluation through `scripts/rsl_rl/eval-rebounce.py`.

Near-term experiments:
- train and evaluate the RNN policy under the same deployable observation;
- test RMA/latent dynamics estimator after history/RNN baselines;
- evaluate high-target/high-damping stress conditions;
- measure real-robot observation noise and update corruption ranges.

## Maintenance

- Keep `go2_rebounce.tex` aligned with the current task definition and latest
  evaluation conclusions.
- Keep `Structure.md` focused on the current architecture, not early prototype
  notes.
- Move old implementation notes to task-specific docs instead of leaving them
  as active backlog.

## Work Log

- `YYYY-MM-DD`: Add new entries here in reverse chronological order.
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
