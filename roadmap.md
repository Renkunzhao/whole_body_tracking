# Sim-to-Real Legged Robot Hopping and Flipping on a Trampoline

## Task 1: Integrate Trampoline into Training with Domain Randomization

Incorporate a soft-body trampoline into the training environment, with domain randomization over `youngs_modulus` and `trampoline_mass`.

**Ref:**
- `/home/rkz/code/whole_body_tracking/scripts/trampoline.py`

## Task 2: Port Hopping Behavior to Isaac Lab

Migrate the existing hopping controller from IsaacGym to Isaac Lab.

**Ref:**
- `/home/rkz/code/Isaacgym/src/My_unitree_go2_gym/legged_gym/envs/Go2_MoB/GO2_JUMP/go2_jump_env.py`
- `/home/rkz/code/mjlab/src/mjlab/tasks/hopping/hopping_env_cfg.py`

## Task 3: Adapt Humanoid Motion Tracking to Quadruped (Go2)

Adapt the existing humanoid (G1) whole-body motion tracking pipeline for quadruped (Go2) use.

**Ref:**
- `/home/rkz/code/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
- `/home/rkz/code/mjlab/src/mjlab/tasks/tracking/config/go2/env_cfgs.py`

## Task 4: Custom contact model

Replace trampoline with customized contact model
