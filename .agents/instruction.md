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
- MuJoCo ball-drop sweep：`scripts/mujoco_ball_drop_trampoline_sweep.py`
- IsaacLab material/reset-time ball-drop sweep：`scripts/isaaclab_ball_drop_trampoline_sweep.py`
- IsaacLab Phase 1 单条件脚本：`scripts/isaaclab_trampoline_phase1_condition.py`
- IsaacLab Phase 1 driver：`scripts/isaaclab_trampoline_phase1_sweep.py`
- 现有 policy 评测脚本：`scripts/rsl_rl/eval-rebounce.py`
- 相关部署/解释文档：`.codex/skills/trampoline/task6-mujoco-deploy.md`

## 阶段性结论

1. **两种模型不是同一类参数化。**
   - IsaacLab 是体积 FEM / deformable object，参数是 `resolution`, `thickness`, `Young's modulus & mass`, `damping`, `possion` 等。
   - MuJoCo 这里是 `flexcomp`，更像“离散柔性网格 + 约束/接触求解”的模型，关键参数是 `count / spacing / radius / mass / pin ids / edge solref / edge solimp`。
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
      - 测试结论（2026-05-21，resolution saturation 实验）：
         - **饱和假设失败**：提升分辨率会系统性增大形变量，最终导致穿膜，不存在"动态指标收敛到上限"的饱和点。无法用分辨率间的相对变化来判断哪个分辨率更准确，因为没有外部参考真值（实物数据或解析解），高分辨率结果本身也不可信。
         - 分辨率只有**稳定性上限**（不穿膜的最大值），没有可量化的误差下限；选择依据是计算代价和数值稳定性，而不是收敛性。
         - `resolution=20–40` 在 `Young's modulus=8e6`、`dt=0.002` 下部分组可 release；`resolution=50` 在所有组全部穿膜（节点数 376–600，dt 不足以稳定）。
         - 旧 `max_compression_m` 指标曾用只按 xy 选出的 center node；厚 mesh 不同 resolution 可能选到 top/bottom 不同 z 层，因此已改为 top-center node 以保证跨分辨率可比。
   - `thickness` 厚度应该优先测试
      - 之前发现在使用dob判断contact的版本，thickness=0.03可能导致train不出来，所以不要低于0.03
      - 在较大范围 0.03 - 0.3 测试厚度对ball drop指标的影响
      - 测试 g1/go2 fixed PD 时是否有穿膜 


## 当前配置摘要

### IsaacLab

`source/whole_body_tracking/whole_body_tracking/utils/trampoline_deformable.py`

- radius: `1.5`
- thickness: `0.1`
- mass: `10.0`
- Young's modulus: `8.0e4`
- sim resolution: `15`
- contact offset: IsaacLab default (`DeformableBodyPropertiesCfg.contact_offset=None`)
- dynamic friction: `0.8`
- elasticity damping: `0.02`
- damping scale: `1.0`
- Poisson's ratio: `0.35`
- usable pin radius: `1.5`
- pinned nodes: 自动固定 `radial_distance >= min(usable_radius, edge_radius)` 的节点

### MuJoCo

`/home/rkz/code/unitree_ws/src/unitree_mujoco/unitree_robots/go2/trampoline.xml`

- `flexcomp type="disc" dim="2" count="3 3 1"`
- `spacing="1.5 1.5 1.5"`
- `radius=".03"`
- `mass="10"`
- pinned rim nodes: `0 1 2 3 5 6 7 8`
- edge constraint: `solref="0.01 1"`
- edge constraint: `solimp="0.8 0.9 0.001 0.1 6"`

## 对比指标

### 几何/静态指标

| 指标 | 含义 | 为什么重要 |
|---|---|---|
| 顶面高度 `z_top` | trampoline 顶面是否都对齐到 `z=0` 附近 | 避免把几何偏差误判成动力学偏差 |
| 中心点位置 | 中心节点/中心区域的静态位置 | 作为局部变形参考系 |
| 有效半径 | 真实参与承载的范围 | 影响 touchdown 时接触面积 |
| 静态下沉量 | 在固定载荷下的压缩量 | 最直接反映“软硬” |
| 静态恢复量 | 卸载后恢复程度 | 反映残余变形与阻尼 |
| 横向漂移 | 载荷后中心是否偏移 | 反映不对称性 |

### 动态 trampoline 指标

| 指标 | 建议定义 |
|---|---|
| 接触持续时间 | 进入接触到完全离开接触的时间 |
| 最大压缩量 | 触地过程中最小 `z` 或最大形变量 |
| 回弹高度 | 离地后的第一峰值高度 |
| 峰值竖直速度 | 接触后 `vz` 最大值 |
| 峰值法向力 | 最大接触力或等效冲量峰值 |
| 总冲量 | `∫F dt` |
| 能量返还率 | `rebound_energy / impact_energy` 或等价定义 |
| 阻尼比/衰减率 | 连续几次 bounce 的幅值衰减 |
| 相位延迟 | touchdown 到最大压缩、到 release 的时间 |
| 非对称指标 | 左右/前后载荷不均引起的 yaw/roll/pitch 偏差 |

### Policy 级指标

这些可以直接对齐现有 `scripts/rsl_rl/eval-rebounce.py` 的输出：

- `success_rate`
- `height_success_rate`
- `mae / rmse / bias / p90`
- `h_over_target`
- `pos_target / abs_target / pos_height / abs_height`
- `braking_ratio`
- `contact_return_ratio`
- `xy_drift`
- `yaw_drift / yaw_rms`
- `orientation_rms`

这组指标很好，因为它们已经把“跳得高不高”“有没有在错误相位刹车”“姿态有没有乱掉”分开了。

## 测试矩阵

### A. 纯 trampoline 响应测试

目的：先不看 policy，只看 trampoline 本身。

1. **球体自由落体**
   - 固定质量球，从多个高度下落
   - 记录接触时间、压缩量、回弹高度、冲量、峰值力
2. **机器人静载/半静载压缩**
   - 机器人站立不动，逐步抬升/下压
   - 测静态下沉量、恢复量、横向漂移
3. **中心 vs 偏心落点**
   - 中心落点和轻微偏心落点分开测
   - 看 yaw/roll/pitch 敏感性

### B. Robot drop / hop playback

目的：评估 trampoline 与 robot 的耦合。

1. 固定初始姿态、固定 drop height
2. 固定同一 policy 的前几步动作
3. 对比两仿真中的：
   - 触地时序
   - 最大压缩
   - 释放相位
   - 回弹峰值
   - 姿态响应

### C. Policy transfer 对照

目的：看 gap 是否来自 trampoline 建模，而不是别的。

1. IsaacLab 训练好的 policy 原样放进 MuJoCo
2. 记录 failure mode 分类：
   - 过早刹车
   - 触地过深
   - 回弹相位错位
   - 姿态发散
   - 横向漂移过大
3. 再看是否需要：
   - 扩 DR
   - 加 history / RNN
   - 做 teacher-student distillation

## 参数扫掠建议

### IsaacLab 侧优先扫的参数

- `youngs_modulus`
- `mass`
- `elasticity_damping`
- `damping_scale`
- `dynamic_friction`
- `simulation_hexahedral_resolution`
- 边缘 pinning 规则 / usable radius

### MuJoCo 侧优先扫的参数

- `spacing`
- `radius`
- `mass`
- `pin ids` / pinned rim width
- `edge solref`
- `edge solimp`
- `count`

### 建议扫掠方式

先做单因素小网格，再做局部联动：

1. 先固定几何，再扫 stiffness / damping
2. 再固定动态参数，扫 pinning / discretization
3. 最后做 cross-sim 匹配，找“行为接近”的区域，而不是参数相同的点

## 经验映射关系

下面只是**粗对照**，不是严格等价：

| IsaacLab | MuJoCo | 说明 |
|---|---|---|
| `youngs_modulus` | `solref / solimp`、边约束刚度 | 都会影响整体软硬 |
| `elasticity_damping` / `damping_scale` | 边约束阻尼、接触衰减 | 都会影响回弹速度和能量损失 |
| `mass` | `mass` | 可直接对照 |
| IsaacLab default contact offset | `radius` / 接触几何厚度 | IsaacLab 侧不再手动指定；MuJoCo `radius` 仍影响接触包络 |
| `simulation_hexahedral_resolution` | `count` | 都影响离散程度和局部响应 |
| usable pin radius / edge pinning | pinned rim nodes / pin 选择 | 都影响边界约束范围；实物半径固定时不应作为 sweep 参数 |

## 目前最值得优先验证的假设

1. **MuJoCo 和 IsaacLab 的差异首先体现在接触相位，而不是单纯刚度。**
2. **同样的视觉形变可能对应不同的能量返还和峰值时序。**
3. **policy 在 IsaacLab 成功，不代表它学到了可跨模型泛化的 trampoline 相位表征。**
4. **如果要缩小 gap，最有效的路径很可能是“测量—对齐—再训练”，而不是单纯加宽 DR。**

## Phase 1：IsaacLab 固定结构参数研究计划

目标：先确定 reset 时无法或不应随机化的结构/离散化参数，避免后续材料参数 sweep 被接触几何和边界条件污染。

### 需要优先固定的参数

| 参数 | 生命周期 | 为什么先测 | 建议第一轮取值 |
|---|---|---|---|
| contact offset | IsaacLab/PhysX default | 不手动指定，避免把引擎接触包络作为可调参数 | fixed default (`None`) |
| `simulation_hexahedral_resolution` | spawn/cook-time | 改变 FEM 离散化和局部变形模式；用 resolution saturation 测试选择接近动态指标上限的最小分辨率 | `8 / 10 / 12 / 15 / 18 / 20 / 24` |
| `thickness` | spawn-time geometry | 改变几何厚度、顶面/中心位置和整体柔度 | `0.03 / 0.10` |
| edge pinning rule | kinematic target / boundary condition | 由网格边缘半径和 `usable_radius=1.5` 自动决定固定节点 | 不 sweep `pin_width` |

### 执行顺序

1. 先跑 nominal，确认当前结构基线：`thickness=0.1`、`simulation_hexahedral_resolution=15`、`usable_radius=1.5`，并使用 IsaacLab 默认 contact offset。
2. 跑独立的 resolution saturation 测试：在多组 rebounce DR 边界参数下固定其他参数，逐步增加 `simulation_hexahedral_resolution`，用 ball-drop 动态指标判断“分辨率增加带来的变软/响应变化是否达到上限”。
3. saturation 测试第一轮固定高冲击条件：小球质量为 G1 URDF 总质量的一半（当前约 `16.67 kg`），小球高度 `1.0 m`；只组合 `thickness=0.03/0.10`、`mass+youngs_modulus` 的 DR 上下限、`elasticity_damping+damping_scale` 的 DR 上下限；当前 Young's modulus sweep 边界为 `8e5 / 8e6`。
4. 如果各参数组在某个 resolution 后 top-center `max_compression_m`、`contact_duration_s`、`release_vz_mps`、`rebound_height_m` 连续两档相对变化都小于阈值（默认 `5%`），则认为该组基本达到上限；只有所有组都有推荐值时才输出全局候选分辨率，并取各组推荐值的最大值。
5. 不再扫 `pin_width`；边界固定由网格边缘半径和实物有效半径自动决定。
6. 不再扫 `contact_offset`；它保持 IsaacLab/PhysX 默认值。

### Phase 1 判据

优先选择同时满足以下条件的结构参数：

- `contact_duration_s`、`max_compression_m`、`rebound_height_m` 没有离群跳变。
- 不出现 drop through 异常。
- 计算代价可接受，能支持后续批量训练和评测。

### Phase 1 脚本和输出规范

- 单条件脚本：`scripts/isaaclab_trampoline_phase1_condition.py`
- Phase 1 driver：`scripts/isaaclab_trampoline_phase1_sweep.py`
- Resolution saturation driver：`scripts/isaaclab_trampoline_resolution_saturation.py`
- 单次运行 artifact 根目录：`logs/isaaclab_trampoline_phase1_runs/`
- Resolution saturation artifact 根目录：`logs/isaaclab_trampoline_resolution_saturation_runs/`
- 每次单条件运行会按时间和参数创建独立文件夹，目录名包含 `label / sim_time / sim_dt / ball_height / ball_mass / sim_resolution / thickness / trampoline_mass / youngs_modulus / elasticity_damping / damping_scale`。
- 单次运行文件夹内包含：
  - `phase1_summary.csv`：单次 summary 指标。
  - `phase1_trajectory.csv`：逐步记录小球位置/速度、trampoline 中心点位置/速度、compression、contact/release 标志。
  - `phase1_vertical_state.png`：小球和 trampoline 中心点竖直位置/速度的 2x1 图。
  - `phase1_compression.png`：trampoline 中心压缩量图。
  - `phase1_video.mp4`：视频。
- CSV 路径由运行目录自动生成，不通过 `--output` 手动指定。
- Resolution saturation 汇总输出：`resolution_saturation_runs.csv` 和 `resolution_saturation_group_summary.csv`。

示例命令：

```bash
python scripts/isaaclab_trampoline_phase1_condition.py --headless --label nominal
python scripts/isaaclab_trampoline_phase1_sweep.py --headless
python scripts/isaaclab_trampoline_resolution_saturation.py --headless --no-video
```

## 当前下一步

1. 先在 IsaacLab 中找能 release 的结构基线，不要直接进入大规模 policy 训练。
2. 优先运行 resolution saturation 测试，确认训练用 `simulation_hexahedral_resolution` 是否已经接近动态响应上限。
3. saturation 第一轮覆盖 `thickness=0.03/0.10`、rebounce DR 中 `mass+youngs_modulus` 上下限、`elasticity_damping+damping_scale` 上下限；小球质量固定为 G1 总重一半，高度固定为 `1.0 m`。
4. 如果结构参数仍然 no-release，再进入材料参数：
   - `elasticity_damping=0.005`
   - `damping_scale=0.2`
5. 找到 IsaacLab 可 release 组合后，再对齐 MuJoCo nominal 的 contact timing、compression、release velocity、rebound height。
