# Task 5: Go2 Continuous Rebounce

This is the current main trampoline research thread.

## Goal

Train Go2 to continuously rebound on a deformable trampoline while tracking a
commanded apex height, remaining roughly in place, and minimizing unnecessary
motor work.

## Current Setup

- Task id: `Go2-Rebounce-Trampoline`.
- Main config:
  `source/whole_body_tracking/whole_body_tracking/tasks/go2_hopping/go2_rebounce_env_cfg.py`.
- Episode length: 20 s.
- Target apex height and reset drop height are sampled independently.
- Success is a normal timeout without failure; apex is no longer a termination.
- Valid apex is detected from root vertical velocity sign change plus
  kinematic foot clearance.
- Reward and termination terms consume the command-owned valid-apex pulse.
- Actor observation is deployable. The flattened-history MLP ablation tested
  `history_length=5` and `history_length=10`; the critic observation is
  privileged and instantaneous.

## Current Trampoline DR

- `youngs_modulus`: log-uniform `(2.0e4, 8.0e4)`.
- `mass`: uniform `(5.0, 15.0)`.
- `dynamic_friction`: uniform `(0.4, 1.2)`.
- `elasticity_damping`: uniform `(0.01, 0.1)`.
- `damping_scale`: fixed `1.0`.
- `poissons_ratio`: uniform `(0.25, 0.45)`.

Do not feed these true parameters directly to the deployable actor.

## Evaluation

- Use `scripts/rsl_rl/eval-rebounce.py` for headless evaluation.
- Repeated `--sweep PARAM VALUE...` arguments form a Cartesian product.
- Useful stress cases are high target height and high material damping.
- Report success rate, failure distribution, apex count, height MAE/RMSE/bias,
  height ratio, work per target height, braking ratio, drift, and orientation
  RMS.

## Next Sequence

1. MLP + deployable observation + DR baseline.
2. MLP + observation/action history ablation (`history_length=5` was close to
   no-history; `history_length=10` often became passive).
3. RNN policy under the same instantaneous deployable observation.
4. RMA/latent dynamics estimator if temporal methods show benefit.
5. Real-robot observation-noise measurement and corruption tuning.
