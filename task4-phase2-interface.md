# Phase 2：通过 URDF 关闭 G1 下肢碰撞，自定义脚地接触接管支撑

## 摘要
- 第二阶段不再使用任何运行时 collision filter、collision group、或动态 `collisionEnabled` 修改。
- 改为直接在 URDF 中删除 G1 膝盖以下 link 的 collision，使 PhysX 在建模阶段就不再为这些部位创建对应地面碰撞。
- 第二阶段只覆盖 `play` 路径，不改 `train.py`；目标仍然是让 G1 在平地场景下，脚底支撑来自自定义接触模型，而不是 PhysX 地面接触。
- 当前关闭碰撞的范围是整段下肢末端链：`left/right_knee_link`、`left/right_ankle_pitch_link`、`left/right_ankle_roll_link`。

## 接口与实现改动
- 保留纯 tensor 模块 `source/whole_body_tracking/whole_body_tracking/utils/point_foot_contact_force.py`：
  - `PointFootContactForceCfg`
  - `PointFootContactForceModel.compute_wrenches(...)`
- 接触模型保持为“平地 + 每只脚单接触点 + 接触点施力”：
  - 接触 body 为 `left_ankle_roll_link`、`right_ankle_roll_link`
  - 局部接触点偏移固定取 G1 foot frame：`(0.04, 0.0, -0.037)`
  - 地面高度默认 env-local `z=0.0`
  - 接触点速度：`v_cp = v_body + omega x r`
  - 法向力：`F_n = max(k_n * penetration - c_n * v_n, 0)`
  - 切向力：`F_t = -k_t * Δx_t - c_t * v_t`，并裁剪到 `||F_t|| <= mu * ||F_n||`
  - `Δx_t` 为每只脚维护的切向弹簧位移；接触建立后积分切向速度，接触断开和 reset 时清零
  - 写回 wrench 时使用 `force = F_n + F_t`，`torque = r x force`
- `tasks/tracking/mdp/actions.py` 中将正常 PD 和脚地接触拆成两个 action term：
  - `joint_pos` 使用标准 joint position PD
  - `foot_contact` 使用 `contact_*` 字段并逐 physics substep 施加自定义接触 wrench
  - `foot_contact` 内部维护每只脚的切向弹簧状态，实现最小 stick-slip
  - 不再保留 `collision_filtered_body_names`
  - 不再提供 `initialize_ground_collision_filter()`
  - `foot_contact.reset()` 时清零 persistent wrench
  - `foot_contact.reset()` 和离地时同时清零切向弹簧位移
- `tracking_env_cfg.py` 中 `joint_pos` 和 `foot_contact` 分开配置，默认 `foot_contact.contact_enabled=False`
- `config/g1/flat_env_cfg.py` 只保留 G1 专属接触参数：
  - `contact_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]`
  - `contact_point_offsets_local = ((0.04, 0.0, -0.037), (0.04, 0.0, -0.037))`
- 自定义接触参数包含：
  - `contact_normal_stiffness`
  - `contact_normal_damping`
  - `contact_tangential_stiffness`
  - `contact_tangential_damping`
  - `contact_friction_coeff`
- `source/whole_body_tracking/whole_body_tracking/robots/g1.py` 改为使用去掉下肢碰撞的 URDF：
  - `main_no_lower_leg_collision.urdf`
- `scripts/load_g1.py` 作为独立验证脚本，直接复用当前 G1 配置，用来确认 no-lower-leg-collision URDF 下的行为
- `scripts/rsl_rl/play.py` 保持尽量干净：
  - 不再单独增加 `--use_custom_foot_contact` 或 `--contact_*` 这类脚本级参数
  - 是否启用自定义脚地接触，统一通过 Hydra 覆盖 `env.actions.foot_contact.*` 完成
  - 例如：`env.actions.foot_contact.contact_enabled=true`

## URDF 方案说明
- 这一版不再依赖“先创建碰撞体、再在 USD/PhysX 中把碰撞关掉”的方式。
- 改为直接维护一份 `main_no_lower_leg_collision.urdf`，从源头删除膝盖以下对应 link 的 collision 元素。
- 这样做的优点是：
  - 不依赖 instanceable / instance proxy 的可编辑性
  - 不依赖 `replicate_physics`、collision group 或 stage 上的后处理
  - scene 创建完成后，PhysX 不会为这些下肢部位生成对应 ground contact
- 代价是：
  - 这是一条 asset-level 修改路径
  - 如果后面要恢复原始碰撞，需要切回原版 `main.urdf`

## 测试计划
- 用 `scripts/load_g1.py --passive` 单独验证：
  - 膝盖以下 link 应能穿过地面
- 通过 Hydra 打开自定义脚地接触运行 G1 play：
  - 机器人应仍能通过自定义脚地力获得支撑
- 示例：
  - `env.actions.foot_contact.contact_enabled=true`
- 关闭自定义接触、但保持使用 no-lower-leg-collision URDF：
  - 机器人应明显下沉，证明支撑不再来自 PhysX 地面碰撞
- 将 `env.actions.foot_contact.contact_tangential_damping=0` 或 `env.actions.foot_contact.contact_friction_coeff=0`：
  - 应出现更明显打滑，验证切向力通路生效
- 将 `env.actions.foot_contact.contact_tangential_stiffness=0`：
  - 会退化回接近“纯切向阻尼 + 动摩擦”的行为，脚更容易滑
- reset 后确认 persistent wrench 被清零

## 假设与默认
- Phase 2 只做 `Tracking-Flat-G1-*`，不提前抽象到 GO2 或其它 humanoid
- 只做平地，不做 rough terrain 法向查询
- 每只脚单接触点是刻意简化；后续如果要更真实支撑面，再扩到两点或四点脚底
- ball 的自定义 spring 路径已并入统一的点接触模型，不再单独维护 `spring_terrain.py`
- 当前优先保证“下肢地面碰撞确实消失、自定义接触确实接管”，暂不处理由此带来的 reward / termination 设计细化
