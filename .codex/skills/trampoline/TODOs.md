# Trampoline TODOs

This file is the incremental backlog and work log for trampoline-related work in `whole_body_tracking`.

## How To Use

- Keep open work items in the backlog section.
- Append newest progress entries to the work log section, ideally with dates.
- When one backlog item turns into a substantial implementation thread, capture the detailed design in a dedicated `task-*.md` file and link it from here.

## Backlog

## Task 1: Isaaclab - DeformableObject

Incorporate a soft-body trampoline into the training environment, with domain randomization over `youngs_modulus` and `trampoline_mass`.

**Ref:**
- `/home/rkz/code/whole_body_tracking/scripts/trampoline.py`

## Task 2: Go2 Hopping

Migrate the existing hopping controller from IsaacGym to Isaac Lab.

**Ref:**
- `/home/rkz/code/Isaacgym/src/My_unitree_go2_gym/legged_gym/envs/Go2_MoB/GO2_JUMP/go2_jump_env.py`
- `/home/rkz/code/mjlab/src/mjlab/tasks/hopping/hopping_env_cfg.py`

## Task 3: Go2 Tracking

Adapt the existing humanoid (G1) whole-body motion tracking pipeline for quadruped (Go2) use.

**Ref:**
- `/home/rkz/code/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
- `/home/rkz/code/mjlab/src/mjlab/tasks/tracking/config/go2/env_cfgs.py`

## Task 4: Custom contact model

Replace trampoline with customized contact model

## Work Log

- `YYYY-MM-DD`: Add new entries here in reverse chronological order.
- `2026-04-02`: Tracking-Trampoline-Go2-v0 and Tracking-Flat-Go2-Wo-State-Estimation-v0
