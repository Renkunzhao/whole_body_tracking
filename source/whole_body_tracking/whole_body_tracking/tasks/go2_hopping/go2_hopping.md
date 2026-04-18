---
name: Trampoline hopping research plan (Go2)
description: Hierarchical two-stage plan — a frozen flat-trained low-level policy with (apex_height, stance_time) commands, plus a trampoline-stage meta-policy that chooses those commands per hop to exploit elasticity.
type: project
originSessionId: 34b1e57f-482d-479f-93ed-c4916a61a53a
---

Research plan for the Go2 trampoline hopping task.

## Goal

Produce a hopping policy that **demonstrably exploits** trampoline elasticity, not merely survives on it. A flat-trained policy already "works" on trampoline (user verified via play), so visible hopping alone is not a convincing demonstration. The target deliverable is a publishable result showing **less joint mechanical work per unit peak height on trampoline than on flat**, attributable to the policy's use of spring dynamics.

## Physical decoupling (load-bearing insight)

- `flight_time` ↔ commanded peak jump height (ballistic, policy-controllable via takeoff vz): `t_flight = 2·√(2h/g)`.
- `stance_time` ↔ physics of mass-spring system, `t_stance ≈ π·√(m_eff / k_eff)` (on trampoline, dictated by the spring; on flat, determined by the robot's own compliance + control).
- Implication: do **not** command `stance_fraction`, and do **not** hard-code `cycle_time`. If we did, the policy would fight the spring on trampoline instead of exploiting it. Prefer rewards that shape **outcomes** (height, energy), not gait timing.
- Trampoline scenes have **no contact sensor** — any design relying on foot-contact signals for inference (not just reward) fails to transfer.

## Architecture: hierarchical two-stage

### Stage 1 — flat stage: low-level "motor skill" policy

Train `μ(obs, h*, t_stance*) → joint_action` on flat ground with a 2-D command:
- `h*` (target apex height) sampled from e.g. `[0.05, 0.25] m`.
- `t_stance*` (target stance duration) sampled from e.g. `[0.15, 0.60] s`, covering the expected trampoline natural stance times as a superset.

Per-env `cycle_time = t_stance* + 2·√(2h*/g)` is derived from the command; the phase clock (`sin_cos_phase` obs, `phase_contact` reward) uses this per-env cycle_time tensor, not a global constant.

Sampling strategy: draw `(h*, t_stance*)` jointly from the full box (not independently narrow), so the low-level's "capability surface" covers all combinations the high-level might later request. Start with a narrow curriculum around the currently-working fixed point `(0.12, 0.45)` and widen.

**Rewards (low-level, flat)**:
- `phase_contact` — per-env `stance_fraction = t_stance* / (t_stance* + t_flight(h*))`.
- `apex_height_tracking_exp` — at `just_landed`, `exp(-((last_peak_z - h*)/σ)²)`.
- `joint_deviation_phase_exp` — keep, with stance/flight std split driven by the per-env phase.
- `action_rate_l2`, `joint_pos_limits` — keep.
- Inject a small `joint_power_l1` (negative) late in curriculum so the low-level doesn't waste energy even on flat; this primes it for efficient execution once commands come from the meta-policy.

**Domain randomization (flat)**: increase `push_robot` amplitude in z and vz so the low-level has seen trampoline-like launch perturbations before transfer.

### Stage 2 — trampoline stage: frozen low-level + meta-policy

Low-level is **strictly frozen** (weights, BN/LN stats). Any improvement in energy/height on trampoline is therefore attributable to the meta-policy's command selection — this is what makes "exploits elasticity" falsifiable.

Train `π_H(s_H) → (h*, t_stance*)` on trampoline (with material/mass DR):
- **Time scale**: meta-policy steps once per hop, at each `just_landed` event. It is a semi-MDP — one meta transition covers many sim steps.
- **Low-level command update**: on each meta step, the new `(h*, t_stance*)` is written into the command buffer **and the phase clock is reset to 0** to align with the landing event. Without this reset, the low-level sees a mid-cycle command jump, which is OOD.
- **Action clipping**: meta output is hard-clipped to the low-level's flat training range. The low-level is only robust inside that box.

**Meta observations** `s_H` (fixed-length vector, no privileged DR info in the actor):
- `h_user` — top-level user command for apex height (the research evaluation target).
- Last-hop summary: `achieved_peak_z`, `achieved_stance_time`, `joint_energy_last_hop`.
- Last N=3 hops' summaries (short history replaces RNN).
- Short-window proprio statistics: base `ang_vel`, projected gravity mean/var.
- Optional for asymmetric critic only: DR parameters (`E`, `m`). Never in actor.

**Meta action**: residual form `(Δh, Δt_stance)` added onto `(h_user, 0.45)`, zero-initialized, so the untrained meta-policy is the identity "pass user command down, use flat-default stance". Learning is pure residual improvement.

**Meta rewards** (this is where the paper lives):
- Primary: `exp(-((achieved_peak_z - h_user)/σ)²)` at each hop — track the user apex target.
- Primary (negative): `−α · joint_energy_last_hop` — encodes the elasticity-exploitation objective. α ramped via curriculum from 0 to target.
- Do **not** add `phase_contact` on trampoline (no contact sensor) and do **not** track `t_stance` (it is now the meta action, not a target).

Expected emergent behavior: meta-policy converges near `t_stance* ≈ π·√(m/k)` (resonance). This is not hand-coded — it is a learned consequence of the energy penalty.

## Evaluation metrics (what proves elasticity is exploited)

- **Primary**: joint mechanical work per unit peak height (`Σ|τ·q̇|dt / h_peak`). Must be lower on trampoline than flat for the same commanded `h_user`.
- **Secondary**: peak-height above flat capability. Cap actuator torque / action scale so flat maxes out at `H_max`; show trampoline achieves ≥ 1.5 × `H_max` with the same low-level.
- **Tertiary**: resonance lock — measured hopping cadence scales with `√(m/k)` across trampolines.
- **Interpretability bonus**: log the meta-policy's chosen `(h*, t_stance*)` vs DR params. If the learned mapping reproduces `π·√(m/k)`, that is a clean figure for the paper.

## Required ablations

1. **Frozen low-level, no meta** (pass `h_user` and default `t_stance = 0.45` straight through): baseline showing the naive transfer.
2. **Frozen low-level, hand-tuned `t_stance* = π·√(m/k)`** (from privileged DR): shows whether the meta-policy recovers the analytical optimum or finds something better.
3. **Joint fine-tuning of low-level on trampoline, no hierarchy**: standard continued PPO. Compare energy/height — the hierarchical architecture must not lose to this to justify the complexity story.

## Implementation order

1. **Low-level double command**: extend `HoppingMetricsCommand` ([mdp/commands.py:16](mdp/commands.py#L16)) to sample `(h*, t_stance*)` per env; make `command` return the 2-D tensor. Resampling time 3–8 s.
2. **Per-env phase clock**: refactor `_phase` ([mdp/rewards.py:61](mdp/rewards.py#L61)) and `phase` ([mdp/observations.py:12](mdp/observations.py#L12)) to read `cycle_time` as a tensor from the command term. On any per-env command resample, latch the current step as `cycle_start[env]` and compute phase as `(now - cycle_start) / cycle_time[env]` to avoid discontinuities.
3. **Reward wiring**: `phase_contact` uses per-env `stance_fraction`. Add `apex_height_tracking_exp` keyed on `just_landed`. Drop `HOPPING_CYCLE_TIME` / `HOPPING_STANCE_FRACTION` constants.
4. **Flat evaluation sweep**: fix `h_user = 0.15`, sweep `t_stance ∈ [0.2, 0.6]`, plot `achieved_h` and `joint_energy` heatmaps. This characterizes the low-level's capability surface; later used to verify the meta-policy operates within it.
5. **Meta-policy env**: trampoline scene, frozen low-level, meta acts once per `just_landed`. Separate PPO instance with small MLP, low LR, residual action head zero-initialized.
6. **Ablations**: run the three listed.

## Risks and mitigations

- **Low-level OOD on extreme `(h*, t_stance*)` pairs**: mitigate by jointly-dense flat sampling + meta output clipping. Not independent-marginal sampling.
- **Phase discontinuity at command switch**: reset phase clock on landing when the meta writes a new command. Non-negotiable.
- **`joint_deviation_phase_exp` std split at extreme stance_fractions**: the two-std posture reward was designed around `stance_fraction ≈ 0.45`. If flat sampling covers `stance_fraction ∈ [0.15, 0.8]`, make sure the posture std choice is reasonable across the range (consider widening or using a single std until proven necessary).
- **Meta credit-assignment sparsity**: ~10–20 hops per trampoline episode means few meta transitions per rollout. Mitigate with many parallel envs (≥4096) and accumulating across multiple episodes before update.
- **Low-level transfer fragility**: flat-trained low-level may fail on trampoline even with correct commands if it never saw strong vertical launch disturbance. Mitigate by aggressive `push_robot` z/vz randomization during flat training.

## Non-goals / things to explicitly avoid

- Commanding `stance_fraction` directly, or hard-coding `cycle_time` as a global constant — breaks the physical decoupling.
- Contact-driven observations for the low-level — the trampoline scene has no contact sensor, so any policy relying on it for inference will not transfer. Contact signals remain fair game for **flat-stage rewards only**.
- Fine-tuning the low-level during stage 2 — destroys the attribution story.
- Adding `t_stance` tracking reward on trampoline — `t_stance` is the meta action, not a target.
