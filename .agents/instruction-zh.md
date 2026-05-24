# Trampoline cross-sim calibration instructions

## 目标

在 IsaacLab 中训练 hopping policy，在 MuJoCo 中测试，再逐步上实物。当前问题不是“视觉上像不像”，而是两种仿真器里的 trampoline 行为在动力学上不一致，所以需要先把**可观测行为**量化，再谈参数对照和 DR 扩展。

## 文档分工

- 当前长期计划、指标定义、参数映射、执行原则：`.agents/instruction.md`
- 实验日志、结果表、失败模式、修改记录：`.agents/logs.md`

每次记录实验日志时必须写清：

- 这次跑了几次实验
- 在哪个仿真器里跑的
- 输出目录或 CSV 路径
- 主要指标和结论

## 当前代码落点

- IsaacLab trampoline：`source/whole_body_tracking/whole_body_tracking/utils/trampoline_deformable.py`
- MuJoCo trampoline：`/home/rkz/code/unitree_ws/src/unitree_mujoco/unitree_robots/go2/trampoline.xml`
- MuJoCo ball-drop 单次/sweep 统一脚本：`scripts/mujoco_trampoline_ball_drop.py`
- IsaacLab ball-drop 单次/sweep 统一脚本：`scripts/isaaclab_trampoline_ball_drop.py`
- 现有 policy 评测脚本：`scripts/rsl_rl/eval-rebounce.py`
- 相关部署/解释文档：`.codex/skills/trampoline/task6-mujoco-deploy.md`

## 阶段性结论

1. **两种模型不是同一类参数化。**
   - IsaacLab 是体积 FEM / deformable object，参数是 `resolution`, `thickness`, `Young's modulus & mass`, `damping`, `possion`, `friction` 。 
   - MuJoCo 这里是 `flexcomp`，更像“离散柔性网格 + 约束/接触求解”的模型，关键参数是 `count / spacing / radius / mass / edge solref / edge solimp`。
   - 经验映射关系，下面只是**粗对照**，不是严格等价：

| IsaacLab | MuJoCo | 说明 |
|---|---|---|
| `youngs_modulus` | `solref / solimp`、边约束刚度 | 都会影响整体软硬 |
| `elasticity_damping` / `damping_scale` | 边约束阻尼、接触衰减 | 都会影响回弹速度和能量损失 |
| `mass` | `mass` | 可直接对照 |
| IsaacLab default contact offset | `radius` / 接触几何厚度 | IsaacLab 侧不再手动指定；MuJoCo `radius` 仍影响接触包络 |
| `simulation_hexahedral_resolution` | `count` | 都影响离散程度和局部响应 |
| usable pin radius / edge pinning | pinned rim nodes / pin 选择 | 都影响边界约束范围；实物半径固定时不应作为 sweep 参数 |

2. **不能只按外观一一对应。**
   - 相同外观不代表相同接触时序、能量返还、压缩量、回弹相位。
   - 因此对照应基于指标，而不是先验假设某个参数完全等价。
3. **最重要的是先做“响应对齐”，再做 policy 对齐。**
   - 先测 trampoline 自身的静态/动态响应。
   - 再测 robot drop / hop 过程中的相位、冲量和姿态稳定性。
4. **固定边缘 pinning 规则**
   - 实物有效半径固定为 `1.5 m`，所以不再把 `pin_width` 当作可调参数。
   - 代码应先根据实际网格节点计算边缘半径 `edge_radius = max(||node_xy - center_xy||)`，再固定所有 `radial_distance >= min(usable_radius, edge_radius)` 的边缘节点。
   - 当前 `usable_radius = 1.5`；如果网格边缘半径大于 1.5，则固定 1.5 m 外侧的节点；如果离散化导致边缘半径小于等于 1.5，则至少固定最外圈边缘节点。
   - 这个规则和 `simulation_hexahedral_resolution` 相关：不同分辨率会改变边缘节点分布，但不应该通过手调 `pin_width` 改变实物有效半径。
5. **先固定 IsaacLab 中无法随机化或不应 reset-time 随机化的参数。**
   - `simulation_hexahedral_resolution`、`thickness` 等需要首先固定下来。
   - IsaacLab `DeformableBodyPropertiesCfg.contact_offset` 默认是 `None`，表示不显式修改 PhysX 默认值；trampoline 不再手动指定或 sweep 这个参数。
   - `simulation_hexahedral_resolution` 分辨率应该同时考虑保真度和速度，**当前固定为 15**（训练慢时可降到 10）；分辨率选择不依赖饱和假设，见下方测试结论。
   - `thickness` 


## 当前配置摘要

### IsaacLab

`source/whole_body_tracking/whole_body_tracking/utils/trampoline_deformable.py`
`source/whole_body_tracking/whole_body_tracking/tasks/go2_hopping/go2_rebounce_env_cfg.py`:

- radius: `1.5`
- usable pin radius: `1.5`
- pinned nodes: 自动固定 `radial_distance >= min(usable_radius, edge_radius)` 的节点
- contact offset: IsaacLab default (`DeformableBodyPropertiesCfg.contact_offset=None`)
- sim resolution: `15`
- thickness: `0.1`
- Young's modulus: `8.0e5` - `8.0e6`
- mass: `5.0` - `15.0`
- elasticity damping: `0.01` - `0.1`
- damping scale: `1.0` - `1.0`
- Poisson's ratio: `0.25` - `0.45`
- dynamic friction: `0.4` - `1.2`

### MuJoCo

`/home/rkz/code/unitree_ws/src/unitree_mujoco/unitree_robots/go2/trampoline.xml`

- `flexcomp type="disc" dim="2" count="3 3 1"`
- `spacing="1.5 1.5 1.5"`
- `radius=".03"`
- `mass="10"`
- pinned rim nodes: `0 1 2 3 5 6 7 8`
- edge constraint: `solref="0.01 1"`
- edge constraint: `solimp="0.8 0.9 0.001 0.1 6"`


## Phase 1：IsaacLab 固定结构参数研究计划

目标：先确定 reset 时无法或不应随机化的结构/离散化参数，避免后续材料参数 sweep 被接触几何和边界条件污染，包括 `resolution`, `thickness`。

### Ball-drop 对比指标及注意事项
不同质量球，从多个高度下落，isaaclab 没有 rigid-deformable 接触力传感器，只能依靠速度来判断状态

| 指标 | 建议定义 |
|---|---|
| 静态下沉量 | 在无压缩时 top-center 节点位置 | 理论应该是0, 反映弹簧自身重力影响 |
| 首次压缩最低点，时间 | 从 `t=0` 开始，小球第一次竖直速度 `|vz|`从负到正时的位置，时间 |
| 首次回弹高度，时间 | 从 `t=0` 开始，小球第一次竖直速度 `|vz|`从正到负时的位置，时间 |
| 阻尼比/衰减率 | 第二次回弹高度的幅值衰减 |
| 最终状态 | 穿膜，掉下边缘，是否稳定 |
| 稳定位置，时间 | 从 `t=0` 开始，小球竖直速度 `|vz|` 连续低于阈值一段时间的位置，时间 |

### Resolution 

- Saturation 测试

1. 固定其他参数，逐步增加 `simulation_hexahedral_resolution`，用 ball-drop 动态指标判断“分辨率增加带来的变软/响应变化是否达到上限”。
2. 条件：小球质量为 G1 URDF 总质量的一半（当前约 `16.67 kg`），小球高度 `1.0 m`；只组合 `thickness=0.03/0.10`、`mass+youngs_modulus` 的 DR 上下限、`elasticity_damping+damping_scale` 的 DR 上下限；当前 Young's modulus sweep 边界为 `8e5 / 8e6`。
3. 如果各参数组在某个 resolution 后 top-center `max_compression_m`、`stable_time_s`、`first_apex_time_s`、`first_apex_height_m` 连续两档相对变化都小于阈值（默认 `5%`），则认为该组基本达到上限；只有所有组都有推荐值时才输出全局候选分辨率，并取各组推荐值的最大值。

- 测试结论（2026-05-21）：
   - **饱和假设失败**：提升分辨率会系统性增大形变量，最终导致穿膜，不存在"动态指标收敛到上限"的饱和点。无法用分辨率间的相对变化来判断哪个分辨率更准确，因为没有外部参考真值（实物数据或解析解），高分辨率结果本身也不可信。
   - 分辨率只有**稳定性上限**（不穿膜的最大值），没有可量化的误差下限；选择依据是计算代价和数值稳定性，而不是收敛性。
   - `resolution=20–40` 在 `Young's modulus=8e6`、`dt=0.002` 下部分组可 release；`resolution=50` 在所有组全部穿膜（节点数 376–600，dt 不足以稳定）。
   - 旧 `max_compression_m` 指标曾用只按 xy 选出的 center node；厚 mesh 不同 resolution 可能选到 top/bottom 不同 z 层，因此已改为 top-center node 以保证跨分辨率可比。

### Thickness
在较大范围 0.03 - 0.3 测试厚度对仿真速度和ball drop指标的影响

注意事项： 
- 研究仿真速度，因为怕厚度高了相同分辨率导致节点增加
- 之前发现在使用dob判断contact的版本，thickness=0.03可能导致train不出来，且 thick 太小容易穿膜，所以确定 thick 之后仍需要后续确认
   - 测试 g1/go2 fixed PD 时是否有穿膜 
   - 测试能否训练出可以跳跃的policy

- 测试结论（2026-05-21，res=15，E=8e6，ed=0.01，tm=15，ball_mass=16.67 kg，ball_height=1.0 m，`--no-video`）
- artifact：`logs/isaaclab_trampoline_phase1_runs/thickness_sweep/`

| thickness | wall_time_s | pinned_nodes | static_sag_m | first_min_ball_z_m | first_min_time_s | first_apex_height_m | first_apex_time_s | second_apex_height_m | damping_ratio | max_compression_m | stable_time_s | 异常 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.03 | 53.2 | 120 | -0.003 | -0.132 | 0.518 | 0.182 | 0.798 | 0.044 | 0.240 | 0.152 | 2.20 | — |
| 0.05 | 52.8 | 120 | -0.003 | -0.150 | 0.524 | 0.173 | 0.810 | 0.020 | 0.117 | 0.170 | 1.91 | — |
| 0.10 | 53.4 | 120 | -0.003 | -0.127 | 0.512 | 0.200 | 0.794 | 0.033 | 0.165 | 0.158 | 1.85 | — |
| 0.15 | 53.8 | 120 | -0.003 | -0.083 | 0.496 | 0.198 | 0.754 | 0.045 | 0.225 | 0.103 | 1.78 | — |
| 0.20 | 52.7 | 120 | -0.003 | -0.054 | 0.484 | 0.187 | 0.720 | 0.048 | 0.258 | 0.083 | — | fell_off_edge |
| 0.30 | 57.4 | 180 | -0.003 | -0.049 | 0.480 | 0.213 | 0.724 | 0.059 | 0.279 | 0.062 | 1.64 | 节点数增至 180 |

关键结论：
- **wall_time_s 在 0.03–0.20 之间基本一致（52–54 s）**，thickness=0.30 略慢（57.4 s），主因是节点数从 120 增至 180；thickness 本身对计算代价影响不大
- **thickness 越大，压缩越浅、接触越早结束**：`first_min_ball_z_m` 从 -0.132 m 单调增加到 -0.049 m，`first_min_time_s` 从 0.524 s 提前到 0.480 s
- **回弹高度（first_apex_height_m）对 thickness 不敏感**，在 0.17–0.21 m 之间无明显趋势
- **thickness=0.20 出现 fell_off_edge**（球沿边缘落下，非穿膜），其余所有条件均正常 stable；原因待查，可能与 res=15 在该厚度下的边缘节点布局有关
- **thickness=0.30 节点数增至 180**，wall_time 略增，`second_apex_height_m` 最高（0.059 m），能量衰减最慢
- **当前沿用 thickness=0.10** 是合理默认值：节点数 120、wall_time ~53 s、压缩量居中、数值稳定

### 脚本和输出规范

- MuJoCo 统一脚本：`scripts/mujoco_trampoline_ball_drop.py`
- IsaacLab 统一脚本：`scripts/isaaclab_trampoline_ball_drop.py`
- MuJoCo 单次运行 artifact 根目录：`logs/mujoco_ball_drop_runs/`
- IsaacLab 单次运行 artifact 根目录：`logs/isaaclab_trampoline_ball_drop_runs/`
- 每次运行会按时间和关键仿真参数创建独立文件夹。
- 单次运行文件夹内包含：
  - `ball_drop_summary.csv`：单次 summary 指标。
  - `ball_drop_trajectory.csv`：逐步记录小球位置/速度、trampoline top-center 位置/速度、compression、stable/apex/contact/release 标志。
  - `ball_drop_params.yaml`：本次运行参数，和 summary CSV 存放在同一目录。
  - `ball_drop_vertical_state.png`：小球和 trampoline 中心点竖直位置/速度的 2x1 图。
  - `ball_drop_compression.png`：trampoline 中心压缩量图。
  - `ball_drop_video.mp4`：视频。
- sweep 模式仍由同一脚本执行：传 `--sweep PARAM VALUE...` 或 `--sweep_config YAML_OR_JSON`，在 sweep 根目录写出 `ball_drop_sweep_summary.csv`。

示例命令：

```bash
python scripts/mujoco_trampoline_ball_drop.py --label nominal
python scripts/mujoco_trampoline_ball_drop.py --sweep solref "0.012 1" "0.015 1" "0.018 1"
python scripts/isaaclab_trampoline_ball_drop.py --headless --label nominal
python scripts/isaaclab_trampoline_ball_drop.py --headless --no-video --sweep_name damping_sweep --sweep elasticity_damping 0.01 0.03 0.1
```

### Phase 1 结论
- 分辨率15, thick 0.1
- 后续根据需要调整

## 后续测试

- **中心 vs 偏心落点**
   - 中心落点和轻微偏心落点分开测
   - 看 yaw/roll/pitch 敏感性

- **机器人静载、落下**
   - 机器人站立不动，逐步抬升/下压
