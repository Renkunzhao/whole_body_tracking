# Task 1：用 DeformableObject 将 Trampoline 接入 G1 Tracking 训练

## 摘要
- 新增独立任务 `Tracking-Trampoline-G1-v0`，不改现有 `Tracking-Flat-G1-*`。
- 复用当前 G1 资产；`G1_CYLINDER_CFG` 已使用 `main.urdf`，本任务不再切换机器人碰撞版本。
- trampoline 使用 Isaac Lab `DeformableObject`，并让膜面上表面对齐 env-local `z=0.0`。
- 训练默认使用 `num_envs=256`、`env_spacing=4.0`、`scene.replicate_physics=False`。

## 接口与实现改动
- 新增共享 helper `whole_body_tracking/utils/trampoline_deformable.py`：
  - 默认参数：`radius=1.5`、`thickness=0.08`、`center_z=-0.04`、`pin_width=0.7`、`mass=1.0`、`youngs_modulus=8e2`、`sim_resolution=5`
  - `make_trampoline_cfg(...)`
  - `build_trampoline_kinematic_targets(...)`
- 新增 zero-dim action term：
  - `TrampolinePinningAction`
  - `TrampolinePinningActionCfg`
  - 每个 physics substep 调 `write_nodal_kinematic_target_to_sim(...)`，持续 pin 住外圈节点
- 新增 reset event：
  - `RandomizeTrampolineProperties`
  - `reapply_trampoline_pinning`
  - reset 顺序固定为：`reset_scene_to_default` -> material/mass randomization -> re-pin
- `tracking_env_cfg.py` 允许 scene 没有 terrain，避免 trampoline scene 初始化失败。
- 新增 `G1TrampolineSceneCfg`、`TrampolineActionsCfg`、`TrampolineEventCfg`、`G1TrampolineEnvCfg`。
- 注册新任务 `Tracking-Trampoline-G1-v0`，并新增 `G1TrampolinePPORunnerCfg` 复用 flat PPO 超参，只改 `experiment_name`。
- 新增 `scripts/load_g1_trampoline.py` 作为 smoke 脚本，用来单独验证 scene / pinning / reset randomization。

## 默认参数
- `youngs_modulus ∈ [2e2, 2e3]`
- `TRAMPOLINE_MASS ∈ [0.5, 1.5]`
- `actions.foot_contact.contact_enabled = False`
- trampoline 下方不放刚性支撑地面

## 测试计划
- `gym.make("Tracking-Trampoline-G1-v0")` 可成功创建。
- 单 env 下 G1 能与 deformable trampoline 正常接触，外圈持续 pinned。
- 多 env 下不会因为 3m 直径 trampoline 导致相互重叠。
- 多次 reset 后：
  - `youngs_modulus` 落在 `[2e2, 2e3]`
  - `mass` 落在 `[0.5, 1.5]`
- `train.py` / `play.py` 用 `task=Tracking-Trampoline-G1-v0` 即可启动。
- `load_g1_trampoline.py` 可用于 scene 级 smoke，`play.py` 用于端到端验证。

## 假设与默认
- v1 不重写 reward、observation、termination。
- v1 不接 phase-2 custom contact；soft-body 接触是唯一支撑来源。
- 若当前 Isaac 版本里 runtime `physics:mass` 写入没有稳定物理效果，可把 `mass_range=None` 作为默认降级，只保留 `youngs_modulus` randomization。
