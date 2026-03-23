# Ball Drop 第一阶段

## 目标
- 用自定义接口验证 training 里 future terrain dynamics 的最小形态。
- 不用 built-in soft object，只根据物体的位置和速度计算竖直弹簧阻尼力，让球落下后弹起。
- phase 1 先把接口做干净，再决定如何接回 training env。

## 已实现内容
- 新增 [spring_terrain.py](/home/rkz/code/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/utils/spring_terrain.py)
  - 提供 `VerticalSpringPlaneCfg`
  - 提供 `VerticalSpringPlane`
  - 核心接口为 `compute_force(root_pos_w, root_lin_vel_w)`
- 新增 [trampoline_spring.py](/home/rkz/code/whole_body_tracking/scripts/trampoline_spring.py)
  - 使用 `AppLauncher` + `SimulationContext`
  - 只创建一个刚体球和一个纯视觉圆盘
  - 不创建 deformable object
  - 每个 physics step 用 `permanent_wrench_composer` 把自定义 spring force 写回仿真

## 力模型
- 参考平面高度为 `plane_height`
- 球底高度为 `bottom_height = root_pos_w[:, 2] - contact_radius`
- 穿透深度为 `penetration = clamp(plane_height - bottom_height, min=0.0)`
- 竖直力为 `force_z = stiffness * penetration - damping * root_lin_vel_w[:, 2]`
- 最终使用单边约束 `force_z = clamp(force_z, min=0.0)`
- 只沿 world `+Z` 方向施力，不做切向力，不做接触点 torque

## Demo 默认参数
- `stiffness=5000.0`
- `damping=120.0`
- `plane_height=0.75`
- `ball_height=1.55`
- `ball_radius=0.18`
- `ball_mass=6.0`
- `reset_interval=360`
- `print_interval=30`

## 运行方式
```bash
./isaaclab.sh -p /home/rkz/code/whole_body_tracking/scripts/trampoline_spring.py
./isaaclab.sh -p /home/rkz/code/whole_body_tracking/scripts/trampoline_spring.py --stiffness 5000 --damping 120
./isaaclab.sh -p /home/rkz/code/whole_body_tracking/scripts/trampoline_spring.py --headless
```

## 日志检查点
- 球底高于参考平面时，`penetration=0` 且 `force_z=0`
- 球进入平面后，`penetration>0` 且 `force_z>0`
- 球离开接触区后，`force_z` 会回到 `0`

## 后续接回 training 的约束
- spring 更新必须放在每个 physics substep 都会执行的路径里。
- 推荐放在自定义 `ActionTerm.apply_actions()` 或 scene-side updater。
- 不要放在 `events.interval`，因为它只在 env step 执行一次，不适合 spring-damper 这种逐 physics step 更新的接口。
