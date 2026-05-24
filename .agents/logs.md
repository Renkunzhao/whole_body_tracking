# Trampoline cross-sim calibration logs

## 记录规范

每次记录实验日志时必须写清：

- 这次跑了几次实验
- 在哪个仿真器里跑的
- 输出目录或 CSV 路径
- 主要指标和结论

## 首轮 ball-drop 参数测试结果（2026-05-19）

### 实验说明

这是第一轮小规模测试，用来验证指标和参数敏感性，不是最终校准结论。

- 实验次数：MuJoCo 多条件 sweep 1 次；IsaacLab 多条件 sweep 1 次。
- 仿真器：MuJoCo、IsaacLab。
- MuJoCo CSV：`logs/mujoco_ball_drop_trampoline_sweep_2026-05-19.csv`
- IsaacLab CSV：`logs/isaaclab_ball_drop_trampoline_sweep_2026-05-19.csv`
- 可复用 MuJoCo sweep 脚本：`scripts/mujoco_ball_drop_trampoline_sweep.py`
- 历史 IsaacLab sweep 脚本：`scripts/isaaclab_ball_drop_trampoline_sweep.py`（2026-05-23 已移除；当前入口见下方修改日志）
- 当时复现命令：
  - `python scripts/mujoco_ball_drop_trampoline_sweep.py --output logs/mujoco_ball_drop_trampoline_sweep.csv`
  - `python scripts/isaaclab_ball_drop_trampoline_sweep.py --headless --output logs/isaaclab_ball_drop_trampoline_sweep.csv`
- 共同测试：半径约 `0.022 m`、质量约 `4.02 kg` 的小球，从 `z=1.0 m` 附近落到 trampoline 中心。
- 这组首轮结果来自旧默认值（厚度 `0.03`、resolution `10`）的脚本版本；当前代码默认值已更新到厚度 `0.1`、resolution `15`，因此这里只保留为历史基线。
- MuJoCo 记录了 touch sensor 的 `peak_force_N` 和 `impulse_Ns`。
- IsaacLab 第一版脚本用 ball bottom height threshold 判断接触/离地，暂未记录真实接触力和冲量，所以接触时间只能粗略对比。
- IsaacLab 启动时出现大量 `Failed to create change watch ... errno=28/No space left on device`，结果 CSV 已写出，但最后手动停止了后台进程；后续应修复 inotify/file-watch 资源问题，避免长时间运行时刷日志。

### Nominal cross-sim 对比

| 指标 | IsaacLab nominal | MuJoCo nominal | MuJoCo / IsaacLab |
|---|---:|---:|---:|
| contact_start_s | 0.444 | 0.460 | 1.04 |
| contact_duration_s | 0.158 | 0.127 | 0.80 |
| max_compression_m | 0.084 | 0.188 | 2.24 |
| release_vz_mps | 0.710 | 1.894 | 2.67 |
| rebound_height_m | 0.083 | 0.142 | 1.70 |

首轮 nominal 下，MuJoCo 的表现不是简单“更硬”或“更软”：它的接触时间更短，但中心压缩更大、释放速度和回弹高度更高。这说明目前 `trampoline.xml` 的 `flexcomp` 模型在能量返还和相位上明显不同于 IsaacLab FEM trampoline。

### MuJoCo 参数影响

MuJoCo 使用当前 `/home/rkz/code/unitree_ws/src/unitree_mujoco/unitree_robots/go2/trampoline.xml` 为基准，只改一个参数。

| 条件 | contact_duration_s | max_compression_m | rebound_height_m | peak_force_N | 主要影响 |
|---|---:|---:|---:|---:|---|
| nominal | 0.127 | 0.188 | 0.142 | 482 | 基准 |
| solref 0.002 | 0.058 | 0.085 | 0.488 | 1121 | 接触极短、峰值力很高、反弹过强 |
| solref 0.005 | 0.089 | 0.136 | 0.308 | 719 | 比 nominal 更硬、更弹 |
| solref 0.02 | 0.360 | 0.261 | -0.016 | 464 | 接触拖长，基本不回弹 |
| radius 0.02 | 0.127 | 0.189 | 0.136 | 485 | 对中心落球影响很小 |
| radius 0.05 | 0.130 | 0.185 | 0.146 | 478 | 对中心落球影响很小 |
| radius 0.10 | 0.134 | 0.181 | 0.172 | 497 | 回弹略增，但不是主导参数 |
| mass 1 | 0.204 | 0.367 | 0.287 | 441 | 更轻 trampoline 反而更深、更久、更弹 |
| mass 5 | 0.141 | 0.232 | 0.197 | 510 | 介于 mass 1 和 nominal |
| mass 30 | 0.126 | 0.131 | 0.051 | 979 | 更重时压缩少、峰值力高、回弹差 |
| spacing 1.0 | 0.086 | 0.118 | 0.100 | 650 | 有效半径/离散结构变小后更硬、更短 |
| spacing 2.0 | 0.169 | 0.262 | 0.167 | 429 | 有效半径变大后更软、更久 |
| offset x=0.15 | 0.126 | 0.182 | 0.107 | 557 | 偏心降低回弹、峰值力升高 |
| offset x=0.30 | 0.188 | 0.176 | 0.080 | 659 | 偏心显著拖长接触并降低回弹 |

MuJoCo 的主导参数目前是 `edge solref`、`mass` 和 `spacing`。`radius` 在中心落球测试中影响较小，但它可能主要影响接触厚度、初始接触时刻和机器人脚掌接触，不应只凭中心 ball-drop 排除。

### IsaacLab 参数影响

IsaacLab 使用 `source/whole_body_tracking/whole_body_tracking/utils/trampoline_deformable.py` 的 nominal 参数为基准，只改一个参数。

| 条件 | contact_duration_s | max_compression_m | rebound_height_m | 主要影响 |
|---|---:|---:|---:|---|
| nominal | 0.158 | 0.084 | 0.083 | 基准 |
| youngs 4e4 | 0.202 | 0.124 | 0.088 | 更软，接触更久，回弹变化小 |
| youngs 1.6e5 | 0.136 | 0.058 | 0.073 | 更硬，接触更短，回弹略低 |
| elasticity_damping 0.005 | 0.122 | 0.094 | 0.235 | 阻尼降低后回弹显著增强 |
| elasticity_damping 0.05 | no release | 0.072 | no release | 阻尼过大，小球没有明显释放 |
| damping_scale 0.2 | 0.144 | 0.123 | 0.267 | 降低 damping scale 后压缩和回弹都增强 |
| pin_width 0.2 | no release | 0.303 | no release | pinning 变窄后中心大幅塌陷，未释放 |

IsaacLab 的 `youngs_modulus` 主要改变压缩量和接触时长，但对回弹高度的影响不如阻尼参数明显。`elasticity_damping` 和 `damping_scale` 直接控制能量返还；`pin_width` 是非常强的几何/边界条件参数，变窄后可能进入不稳定或过软区域。

| 条件 | 配置方式 | 预期影响 | 当前状态 |
|---|---|---|---|
| contact_offset_0.01 (nominal) | spawn-time 0.01 | nominal | 暂缺实测值（见下文说明） |
| contact_offset_0.02 | spawn-time 0.02 | 碰撞包裹层变厚，理论上更早触发接触 | 暂缺实测值 |
| contact_offset_0.05 | spawn-time 0.05 | 可能导致静态即穿透或提前反弹 | 暂缺实测值 |

### 关于 `contact_offset` 的补充说明

在收到明确测试 `contact_offset` 影响的要求后，我编写了独立的序列化扫描脚本。但是，与 `youngs_modulus` 或 `mass` 可以在运行期或 reset 时动态写入不同，`contact_offset` 属于 `DeformableBodyPropertiesCfg`，是**在 Spawn (Cook) 阶段写死进网格的几何属性**，无法在一个环境中随时间重置改变，只能按不同 Bucket 实例化或多次重启仿真器来测。

当前实验机遇到了较严重的 inotify/file-watch 资源耗尽（`errno=28/No space left on device`），即使分次调用 IsaacLab 也会在底层启动时崩溃，导致我们无法跑完这个特化测试脚本。

**理论影响推演：**
IsaacLab/PhysX 中的 `contact_offset` 相当于在可视/物理 mesh 外包裹了一层“探测厚度”。当它增加时：

1. **接触提前：** `ball` 在物理上还没有碰到视觉网格时，接触力就开始产生。
2. **等效变软：** 引擎在这一层厚度内使用 penalty force 处理侵入，这层虚拟厚度带来的接触时间可能使得宏观上的冲击变得平缓（类似加上了一层隐形海绵）。
3. 如果设置得太大（例如 `0.05` 甚至更大），可能直接影响机器人的静态站立高度，或者导致它在半空中就获得异常推力。

相对应的，MuJoCo 中与它最接近的概念是 `flexcomp` 的 `radius` 参数。首轮测试中，改动 `radius`（在 `0.02` 到 `0.1` 间）对中心落球的回弹影响较小。如果后续本地环境恢复 watch limit 后重新跑 IsaacLab，预期 `contact_offset` 也会表现出类似的非主导性（除非太大导致静态异常）。

### 首轮 cross-sim 解释

1. **MuJoCo nominal 的能量返还明显高于 IsaacLab nominal。** 释放速度约为 IsaacLab 的 `2.67x`，回弹高度约为 `1.70x`。
2. **MuJoCo nominal 的中心压缩也明显更大。** 压缩约为 IsaacLab 的 `2.24x`，说明不能只用“回弹高所以更硬”来解释。
3. **MuJoCo `solref` 更像直接控制接触相位和返弹强度。** `0.002` 产生短接触、高峰值力、高回弹；`0.02` 产生长接触和近乎无回弹。
4. **IsaacLab 的 damping 参数比 Young's modulus 更直接影响回弹。** 如果要让 IsaacLab DR 覆盖 MuJoCo 的高回弹，应优先检查 `elasticity_damping` / `damping_scale` 的范围，而不是只扩大 Young's modulus。
5. **偏心接触必须单独测。** MuJoCo offset x=0.30 时回弹高度下降约 `44%`、峰值力上升约 `37%`，这可能对应 Go2 在 MuJoCo 上姿态失败的问题。

### 首轮参数对照建议

下一轮如果目标是让 MuJoCo nominal 更接近 IsaacLab nominal，可以优先测试：

- MuJoCo `solref` 在 `0.01` 到 `0.02` 之间细扫，例如 `0.012 / 0.015 / 0.018`，目标是降低 release velocity 和 rebound height，同时不要把 contact duration 拉得过长。
- MuJoCo `mass` 在 `10` 到 `30` 之间细扫，目标是降低 rebound height 和压缩量，但注意 `mass=30` 峰值力过高。
- MuJoCo `spacing` 不应只作为几何参数看待；它显著改变压缩和接触时长，必须和真实 trampoline 半径/节点布局一起约束。
- IsaacLab 侧如果希望 DR 覆盖 MuJoCo 当前 nominal，应加入更低阻尼条件，例如 `elasticity_damping <= 0.005` 或 `damping_scale < 1`，但这可能会显著改变 policy 学到的相位。

## Phase 1 结构参数实验记录（2026-05-19）

### 已验证脚本和输出

- 实验次数：IsaacLab Phase 1 单条件脚本至少跑通 2 次已记录条件（`nominal`、`nominal_full`）；后续另有手动 `nominal_video` artifact 需要读取 summary 后补充指标。
- 仿真器：IsaacLab。
- 历史单条件脚本：`scripts/isaaclab_trampoline_phase1_condition.py`（2026-05-23 重命名为 `scripts/isaaclab_trampoline_ball_drop.py`）
- 历史 driver：`scripts/isaaclab_trampoline_phase1_sweep.py`（2026-05-23 已移除）
- 当时单次运行 artifact 根目录：`logs/isaaclab_trampoline_phase1_runs/`
- 单次运行 artifact：`phase1_summary.csv`、`phase1_trajectory.csv`、`phase1_vertical_state.png`、`phase1_compression.png`，以及 `phase1_video.mp4`
- CSV 路径由运行目录自动生成，不再通过 `--output` 手动指定。

### 已跑通的条件

| 条件 | contact_started | released | contact_start_s | contact_duration_s | impact_vz_mps | release_vz_mps | max_compression_m | rebound_height_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal | 1 | 0 | 0.444 | nan | -4.355630397796631 | nan | 0.1480908840894699 | nan |
| nominal_full | 1 | 0 | 0.444 | nan | -4.355630397796631 | nan | 0.1810057908296585 | nan |

### 手动 video artifact 待补充

以下运行已写出 artifact，但还需要读取 `phase1_summary.csv` 后补充精确指标：

- 实验次数：IsaacLab 手动单条件运行 1 次。
- 仿真器：IsaacLab。
- 输出目录：`logs/isaaclab_trampoline_phase1_runs/20260519_185037__nominal_video__t4__dt0.002__h1__co0.01__sr15__th0.1__pw0.4__E80000__ed0.02__ds1`
- 现象：写出了 `phase1_video.mp4` 和 `RUN_DIR`，但程序卡在 Isaac Sim shutdown；后续已修改脚本在成功写完 artifact 后用 `os._exit(0)` 避免 `simulation_app.close()` hang。

另一次运行也已出现对应 artifact：

- 实验次数：IsaacLab 手动单条件运行 1 次。
- 仿真器：IsaacLab。
- 输出目录：`logs/isaaclab_trampoline_phase1_runs/20260519_190839__nominal_video__t4__dt0.002__h1__co0.01__sr15__th0.1__pw0.4__E80000__ed0.02__ds1`
- 现象：用于确认图像输出关系；`phase1_vertical_state.png` 已包含位置与速度 2x1 图，因此已删除单独的 `phase1_vertical_velocity.png`。

### 直接结论

1. 以当前 IsaacLab trampoline 默认值（thickness `0.1`、contact_offset `0.01`、sim_resolution `15`、pin_width `0.4`）运行 4 秒，**ball-drop 仍然没有 release**。
2. 这说明当前 Phase 1 的基线结构参数本身就还不满足“可用于后续材料 sweep”的要求，不能把它当成稳定的对照点继续往下铺。
3. 这次结果也说明 `pin_width` / thickness / 离散化配置已经足够强，足以把系统推入 no-release 区域；因此 Phase 1 不是简单的“参数表整理”，而是真正需要先找到一个可释放的结构基线。

### 这次新增的实用观察

- `contact_offset` 仍然是 spawn/cook-time 参数，不能像 `mass` 那样 reset 时改。
- `build_trampoline_kinematic_targets()` 现在支持标量或 tensor `pin_width`，方便后续按 env 扫掠。
- `make_trampoline_cfg()` 现在显式支持 `thickness` 和 `contact_offset`，并允许从 `thickness` 推导 `center_z`。

### 当前 blocker

- IsaacLab 启动仍然持续出现大量 `Failed to create change watch ... errno=28/No space left on device`。
- 虽然 `nominal_full` 最终写出了 CSV 行，但后台过程依旧表现出启动/关闭不干净的迹象，导致重复 sweep 成本很高。
- 在这个 blocker 没处理前，Phase 1 的完整 sweep 只能停在“脚本已固化、nominal 已记录、但整套结构参数仍未完成”这个状态。

### 后续动作

- 先把当前可用结果保留在这里，不再把 canonical 记录放到 `logs/`。
- 若后续环境恢复，再补跑 `contact_offset` / `sim_resolution` / `thickness` / `pin_width` 的单因素条件，并把结论追加到本节。
- 如果要继续推进材料参数 sweep，必须先找到一个能 release 的结构基线。

## 修改日志

- `2026-05-19`：创建初版对照记录，整理 IsaacLab / MuJoCo trampoline 当前配置、建议指标、测试矩阵和参数扫掠方向。
- `2026-05-19`：新增 `scripts/mujoco_ball_drop_trampoline_sweep.py`，固化此前临时 MuJoCo ball-drop sweep，并用默认 4 秒 sweep 验证可写出 CSV。
- `2026-05-19`：新增 `scripts/isaaclab_ball_drop_trampoline_sweep.py`，固化 IsaacLab ball-drop sweep，并保留 `contact_offset` 作为 spawn-time 参数。
- `2026-05-19`：新增 `scripts/isaaclab_trampoline_phase1_condition.py` 和 `scripts/isaaclab_trampoline_phase1_sweep.py`，把 Phase 1 结构参数 sweep 固化成可复用入口。
- `2026-05-19`：更新 `source/whole_body_tracking/whole_body_tracking/utils/trampoline_deformable.py`，让 trampoline 配置显式暴露 `thickness`、`contact_offset`，并支持 tensor `pin_width`。
- `2026-05-19`：跑通 `nominal` 与 `nominal_full` 两个 Phase 1 条件，结果都没有 release；当前 CSV 见 `logs/isaaclab_trampoline_phase1_structure_sweep_2026-05-19.csv`。
- `2026-05-19`：增强 `scripts/isaaclab_trampoline_phase1_condition.py` 的单次运行日志：按时间和参数创建 artifact 文件夹，保存逐步 trajectory CSV、summary CSV、位置/速度/压缩量图，并支持视频输出 MP4。
- `2026-05-19`：拆分原 `.agents/trampoline_isaaclab_mujoco_comparison.md` 为 `.agents/instruction.md` 和 `.agents/logs.md`，分别保存长期计划与实验日志。
- `2026-05-20`：根据实物有效半径固定为 `1.5 m` 的原则，移除 `pin_width` sweep，把 IsaacLab pinning 改为根据网格边缘半径自动固定 `radial_distance >= min(usable_radius, edge_radius)` 的节点；Phase 1 后续只记录 `pinned_node_count`。
- `2026-05-20`：确认 IsaacLab `DeformableBodyPropertiesCfg.contact_offset` 默认值为 `None`；移除 trampoline 侧显式 `contact_offset` 配置、Phase 1 CLI 参数和相关 sweep 条件，后续使用 IsaacLab/PhysX 默认接触包络。
- `2026-05-20`：新增 `scripts/isaaclab_trampoline_resolution_saturation.py`，用于在 rebounce DR 边界参数组下逐步增加 `simulation_hexahedral_resolution`，用 ball-drop 动态指标判断分辨率响应是否达到上限；同时扩展 Phase 1 单条件脚本支持 `--ball_mass`、`--trampoline_mass` 和 `--no-video`。
- `2026-05-20`：修正 trampoline center node 选择逻辑：从只按 xy 最近改为中心 xy 候选中的最高 z 节点，避免厚 mesh 在不同 resolution 下选到 top/bottom 不同层导致 `max_compression_m` 跨分辨率不可比；同时将 rebounce/resolution saturation 的 Young's modulus sweep 范围从 `8e4-8e5` 提高到 `8e5-8e6`。
- `2026-05-21`：跑完 resolution saturation 实验（`scripts/isaaclab_trampoline_resolution_saturation.py`，`logs/isaaclab_trampoline_resolution_saturation_runs/20260520_232221__resolution_saturation/`）。结论：饱和假设失败，提升分辨率系统性增大形变量最终穿膜，不存在收敛上限；无外部参考真值，无法量化低分辨率的离散误差下限。分辨率选择只有数值稳定性上限（`res=50` 全组穿膜），没有可量化的误差下限，后续按计算代价固定 `simulation_hexahedral_resolution=15`（训练慢时可降到 10）。
- `2026-05-21`：更新 Phase 1 ball-drop 指标定义：不再把 contact-to-release duration 作为核心稳定指标；新增从 `t=0` 开始、基于小球 `|vz|` 连续低于阈值的 `stable_time_s`，并用 armed/hysteresis 状态机记录第一次有效 apex 的 `first_apex_time_s` 和 `rebound_height_m`，避免速度抖动造成重复 apex。
- `2026-05-23`：清理并合并 IsaacLab trampoline ball-drop 脚本：将核心入口从 `scripts/isaaclab_trampoline_phase1_condition.py` 重命名为 `scripts/isaaclab_trampoline_ball_drop.py`；移除旧版 `scripts/isaaclab_ball_drop_trampoline_sweep.py`、`scripts/isaaclab_trampoline_phase1_sweep.py`、`scripts/trampoline-DeformableObject.py` 和独立 `scripts/isaaclab_trampoline_resolution_saturation.py` driver；新脚本同时支持单次运行、`--sweep PARAM VALUE...` CLI sweep 和 `--sweep_config` YAML/JSON sweep；单次 artifact 根目录为 `logs/isaaclab_trampoline_ball_drop_runs/`，文件名为 `ball_drop_summary.csv`、`ball_drop_trajectory.csv`、`ball_drop_vertical_state.png`、`ball_drop_compression.png` 和 `ball_drop_video.mp4`；sweep 汇总文件为 `ball_drop_sweep_summary.csv`。
- `2026-05-23`：更新 `scripts/mujoco_ball_drop_trampoline_sweep.py`：默认只跑 nominal；`mass/radius/spacing/solref/solimp/ball_x` 变成单次参数；新增 `--sweep PARAM VALUE...` 和 `--sweep_config` YAML/JSON sweep；默认给每个 condition 写 `ball_drop_video.mp4` 到 summary CSV 同目录，传 `--no-video` 或在 sweep config 中设 `video: false` 可关闭，并在 CSV 中记录 `video_path`。
- `2026-05-23`：更新 `scripts/mujoco_ball_drop_trampoline_sweep.py` 的默认 artifact 输出：不传 `--output` 时自动创建 `logs/mujoco_ball_drop_runs/<timestamp+参数>/`，写出与 IsaacLab ball-drop 同名的 `ball_drop_summary.csv`、`ball_drop_trajectory.csv`、`ball_drop_vertical_state.png`、`ball_drop_compression.png` 和 `ball_drop_video.mp4`；sweep 模式每个 condition 使用独立子目录并在 sweep 根目录写 `ball_drop_sweep_summary.csv`；仍允许 `--output` 覆盖 summary CSV 路径。
- `2026-05-23`：历史短验证运行 MuJoCo 单条件 2 次（`--sim_time 0.01` 和 `--sim_time 0.01 --label video_default_check`），分别验证旧版自动 CSV 路径和默认 video 写出；这些运行只验证 artifact 写出，不作为动态响应结论。
- `2026-05-23`：验证 MuJoCo 与 IsaacLab 同名 artifact 输出：运行 MuJoCo 单条件 1 次（`--sim_time 0.01 --label artifact_match_check --output logs/mujoco_ball_drop_runs/artifact_match_check/ball_drop_summary.csv`），输出目录为 `logs/mujoco_ball_drop_runs/artifact_match_check/`，非空文件包括 `ball_drop_summary.csv`、`ball_drop_trajectory.csv`、`ball_drop_vertical_state.png`、`ball_drop_compression.png`、`ball_drop_video.mp4`；该运行只验证 artifact 命名和写出，不作为动态响应结论。
- `2026-05-23`：验证 MuJoCo sweep artifact 输出：运行 MuJoCo 2 条件 sweep 1 次（`--no-video --sim_time 0.002 --output logs/mujoco_ball_drop_runs/artifact_sweep_check/ball_drop_sweep_summary.csv --sweep mass 10 30`），输出目录为 `logs/mujoco_ball_drop_runs/artifact_sweep_check/`；根目录写出 `ball_drop_sweep_summary.csv`，`mass_10/` 和 `mass_30/` 子目录各写出非空 `ball_drop_summary.csv`、`ball_drop_trajectory.csv`、`ball_drop_vertical_state.png`、`ball_drop_compression.png`；该运行只验证 sweep artifact 命名和写出，不作为动态响应结论。
- 后续每次实验补充：实验次数、仿真器、输出路径、指标结果、失败模式、参数调整、结论。
