# Trampoline cross-sim calibration instructions

## Goal

Train a hopping policy in IsaacLab, test it in MuJoCo, then gradually deploy to hardware. The current problem is not "visual similarity" but dynamic inconsistency of trampoline behavior between the two simulators. We must first quantify **observable behavior**, then discuss parameter alignment and DR expansion.

## Document responsibilities

- Long-term plan, metric definitions, parameter mappings, execution principles: `.agents/instruction.md`
- Experiment logs, result tables, failure modes, change history: `.agents/logs.md`

Every experiment log entry must include:

- Number of runs
- Which simulator
- Output directory or CSV path
- Key metrics and conclusions

## Code locations

- IsaacLab trampoline：`source/whole_body_tracking/whole_body_tracking/utils/trampoline_deformable.py`
- MuJoCo trampoline：`/home/rkz/code/unitree_ws/src/unitree_mujoco/unitree_robots/go2/trampoline.xml`
- MuJoCo ball-drop 单次/sweep 统一脚本：`scripts/mujoco_trampoline_ball_drop.py`
- IsaacLab ball-drop 单次/sweep 统一脚本：`scripts/isaaclab_trampoline_ball_drop.py`
- 现有 policy 评测脚本：`scripts/rsl_rl/eval-rebounce.py`
- 相关部署/解释文档：`.codex/skills/trampoline/task6-mujoco-deploy.md`

## Interim conclusions

1. **The two models use different parameterizations.**
   - IsaacLab uses volumetric FEM / deformable object; parameters are `resolution`, `thickness`, `Young's modulus & mass`, `damping`, `poisson`, `friction`.
   - MuJoCo uses `flexcomp`, closer to a "discrete flexible mesh + constraint/contact solver" model; key parameters are `count / spacing / radius / mass / edge solref / edge solimp`.
   - Empirical mapping — **rough correspondence only**, not strict equivalence:

| IsaacLab | MuJoCo | Notes |
|---|---|---|
| `youngs_modulus` | `solref / solimp`, edge constraint stiffness | Both affect overall stiffness |
| `elasticity_damping` / `damping_scale` | Edge constraint damping, contact decay | Both affect rebound speed and energy loss |
| `mass` | `mass` | Direct comparison |
| IsaacLab default contact offset | `radius` / contact geometry thickness | IsaacLab side no longer set manually; MuJoCo `radius` still affects contact envelope |
| `simulation_hexahedral_resolution` | `count` | Both affect discretization and local response |
| usable pin radius / edge pinning | pinned rim nodes / pin selection | Both affect boundary constraint range; should not be swept when physical radius is fixed |

2. **Cannot map parameters one-to-one by appearance.**
   - Same appearance does not mean same contact timing, energy return, compression, or rebound phase.
   - Alignment should be based on metrics, not prior assumptions of parameter equivalence.
3. **"Response alignment" must come before policy alignment.**
   - First measure static/dynamic trampoline response.
   - Then measure contact phase, impulse, and attitude stability during robot drop/hop.
4. **Fixed edge pinning rule**
   - Physical usable radius is fixed at `1.5 m`, so `pin_width` is no longer a tunable parameter.
   - Code should first compute `edge_radius = max(||node_xy - center_xy||)` from actual mesh nodes, then pin all nodes with `radial_distance >= min(usable_radius, edge_radius)`.
   - Current `usable_radius = 1.5`; if mesh edge radius > 1.5, pin nodes outside 1.5 m; if discretization makes edge radius ≤ 1.5, pin at least the outermost ring.
   - This rule interacts with `simulation_hexahedral_resolution`: different resolutions change edge node distribution, but the physical usable radius must not be changed by manually tuning `pin_width`.
5. **Fix IsaacLab parameters that cannot or should not be randomized at reset time first.**
   - `simulation_hexahedral_resolution`, `thickness`, etc. must be fixed before material sweeps.
   - IsaacLab `DeformableBodyPropertiesCfg.contact_offset` defaults to `None`, meaning no explicit override of the PhysX default; trampoline no longer sets or sweeps this parameter.
   - `simulation_hexahedral_resolution` should balance fidelity and speed; **currently fixed at 15** (can be reduced to 10 if training is too slow). Resolution selection does not rely on a saturation assumption — see test conclusions below.
   - `thickness`


## Current configuration summary

### IsaacLab

`source/whole_body_tracking/whole_body_tracking/utils/trampoline_deformable.py`
`source/whole_body_tracking/whole_body_tracking/tasks/go2_hopping/go2_rebounce_env_cfg.py`:

- radius: `1.5`
- usable pin radius: `1.5`
- pinned nodes: automatically pin nodes with `radial_distance >= min(usable_radius, edge_radius)`
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

### Current ball-drop alignment candidates

These are empirical IsaacLab parameter candidates for the current MuJoCo trampoline endpoints. Use them as the response-alignment anchors when setting training DR ranges; full run metrics and conclusions remain in `.agents/logs.md`.

| Target MuJoCo endpoint | MuJoCo parameters | IsaacLab candidate parameters | Key status |
|---|---|---|---|
| nominal / `mass=10` | `mass=10`, `radius=0.03`, `spacing=1.5`, `solref="0.01 1"`, `solimp="0.8 0.9 0.001 0.1 6"` | `thickness=0.05`, `trampoline_mass=7.5`, `youngs_modulus=1.5e5`, `elasticity_damping=0.005`, `damping_scale=0.5` | Dynamic compression and first rebound height are close; steady-state compression is still too shallow. See `.agents/logs.md` entries `20260524_011500` and the recorded candidate immediately after it. |
| heavy / `mass=100` | `mass=100`, `radius=0.03`, `spacing=1.5`, `solref="0.01 1"`, `solimp="0.8 0.9 0.001 0.1 6"` | `thickness=0.05`, `trampoline_mass=20`, `youngs_modulus=3e5`, `elasticity_damping=0.005`, `damping_scale=0.5` | Dynamic compression is close; rebound remains slightly high and steady-state compression too shallow. See `.agents/logs.md` entries `20260524_013112` and the recorded candidate immediately after it. |


## Phase 1: IsaacLab fixed structural parameter study

Goal: identify structural/discretization parameters that cannot or should not be randomized at reset time, to avoid contaminating subsequent material parameter sweeps with contact geometry and boundary condition artifacts. Covers `resolution` and `thickness`.

### Ball-drop metrics and notes

Balls of different masses dropped from multiple heights. IsaacLab has no rigid-deformable contact force sensor, so state must be inferred from velocity.

| Metric | Definition |
|---|---|
| Static sag | Top-center node position before any compression — should be ~0; reflects mesh self-weight |
| First compression minimum, time | Position and time when ball `vz` first crosses zero from negative to positive |
| First rebound apex, time | Position and time when ball `vz` first crosses zero from positive to negative |
| Damping ratio | Amplitude decay from first apex to second apex |
| Final state | Fallthrough, fell off edge, or stable |
| Stable position, time | Position and time when `|vz|` stays below threshold for a continuous window |

### Resolution

- Saturation test

1. Fix all other parameters, sweep `simulation_hexahedral_resolution`, and use ball-drop dynamic metrics to determine whether the softening effect of increasing resolution reaches a saturation limit.
2. Conditions: ball mass = half of G1 URDF total mass (~`16.67 kg`), ball height = `1.0 m`; combine `thickness=0.03/0.10` with DR boundary values of `mass+youngs_modulus` and `elasticity_damping+damping_scale`; Young's modulus sweep range `8e5 / 8e6`.
3. If for a given group all four top-center metrics (`max_compression_m`, `stable_time_s`, `first_apex_time_s`, `first_apex_height_m`) change by less than the threshold (default `5%`) for two consecutive resolution steps, the group is considered saturated. A global candidate resolution is output only when all groups have a recommendation; the maximum across groups is taken.

- Test conclusions (2026-05-21):
   - **Saturation hypothesis failed**: increasing resolution systematically increases deformation, eventually causing fallthrough. There is no saturation point where dynamic metrics converge to an upper limit. Relative changes between resolution levels cannot be used to judge which resolution is more accurate, because there is no external ground truth (physical measurements or analytic solutions), and high-resolution results are themselves unreliable.
   - Resolution has only a **stability upper limit** (maximum that avoids fallthrough), no quantifiable error lower bound. The selection criterion is computational cost and numerical stability, not convergence.
   - `resolution=20–40` with `Young's modulus=8e6` and `dt=0.002` allows some groups to release; `resolution=50` causes fallthrough in all groups (376–600 nodes, dt insufficient for stability).
   - The old `max_compression_m` metric selected center nodes by xy only; thick meshes at different resolutions may select top or bottom z layers differently. Changed to top-center node to ensure cross-resolution comparability.

### Thickness

Test the effect of thickness on simulation speed and ball-drop metrics over the range 0.03–0.3.

Notes:
- Investigate simulation speed, since higher thickness at the same resolution may increase node count.
- Previous finding with DOB contact detection: `thickness=0.03` may prevent successful training, and very small thickness is prone to fallthrough. After fixing thickness, the following must still be confirmed:
   - Whether fallthrough occurs with g1/go2 fixed PD
   - Whether a hopping policy can be trained

- MuJoCo 统一脚本：`scripts/mujoco_trampoline_ball_drop.py`
- MuJoCo 运行 artifact 根目录：`logs/mujoco_ball_drop_runs/`
- MuJoCo 单次运行会按时间和参数创建独立文件夹，目录名包含 `label / sim_time / ball_height / trampoline_mass / radius / spacing / ball_x / solref`；传 `--output` 时可覆盖 summary CSV 路径。
- IsaacLab 统一脚本：`scripts/isaaclab_trampoline_ball_drop.py`
- IsaacLab 运行 artifact 根目录：`logs/isaaclab_trampoline_ball_drop_runs/`
- IsaacLab 单次运行会按时间和参数创建独立文件夹，目录名包含 `label / sim_time / sim_dt / ball_height / ball_mass / sim_resolution / thickness / trampoline_mass / youngs_modulus / elasticity_damping / damping_scale`。
- MuJoCo 和 IsaacLab 单次运行文件夹内都包含：
  - `ball_drop_summary.csv`：单次 summary 指标。
  - `ball_drop_trajectory.csv`：逐步记录小球位置/速度、trampoline top-center 位置/速度、compression、stable/apex/contact/release 标志。
  - `ball_drop_params.yaml`：本次运行参数，和 summary CSV 存放在同一目录。
  - `ball_drop_vertical_state.png`：小球和 trampoline 中心点竖直位置/速度的 2x1 图。
  - `ball_drop_compression.png`：trampoline 中心压缩量图。
  - `ball_drop_video.mp4`：视频；MuJoCo 和 IsaacLab 都可传 `--no-video` 关闭。
- 单次 summary CSV 路径由运行目录自动生成；MuJoCo 可用 `--output` 覆盖，IsaacLab 不通过 `--output` 手动指定。
- sweep 模式仍由同一脚本执行：传 `--sweep PARAM VALUE...` 或 `--sweep_config YAML_OR_JSON` 时，脚本会展开条件，并在 sweep 根目录写出 `ball_drop_sweep_summary.csv`。

| thickness | wall_time_s | pinned_nodes | static_sag_m | first_min_ball_z_m | first_min_time_s | first_apex_height_m | first_apex_time_s | second_apex_height_m | damping_ratio | max_compression_m | stable_time_s | anomaly |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.03 | 53.2 | 120 | -0.003 | -0.132 | 0.518 | 0.182 | 0.798 | 0.044 | 0.240 | 0.152 | 2.20 | — |
| 0.05 | 52.8 | 120 | -0.003 | -0.150 | 0.524 | 0.173 | 0.810 | 0.020 | 0.117 | 0.170 | 1.91 | — |
| 0.10 | 53.4 | 120 | -0.003 | -0.127 | 0.512 | 0.200 | 0.794 | 0.033 | 0.165 | 0.158 | 1.85 | — |
| 0.15 | 53.8 | 120 | -0.003 | -0.083 | 0.496 | 0.198 | 0.754 | 0.045 | 0.225 | 0.103 | 1.78 | — |
| 0.20 | 52.7 | 120 | -0.003 | -0.054 | 0.484 | 0.187 | 0.720 | 0.048 | 0.258 | 0.083 | — | fell_off_edge |
| 0.30 | 57.4 | 180 | -0.003 | -0.049 | 0.480 | 0.213 | 0.724 | 0.059 | 0.279 | 0.062 | 1.64 | node count increased to 180 |

Key conclusions:
- **wall_time_s is nearly identical across 0.03–0.20 (52–54 s)**; thickness=0.30 is slightly slower (57.4 s) primarily because node count increases from 120 to 180. Thickness itself has little effect on computational cost.
- **Greater thickness → shallower compression, earlier contact end**: `first_min_ball_z_m` increases monotonically from -0.132 m to -0.049 m; `first_min_time_s` shifts earlier from 0.524 s to 0.480 s.
- **Rebound apex height (`first_apex_height_m`) is insensitive to thickness**, ranging 0.17–0.21 m with no clear trend.
- **thickness=0.20 produced fell_off_edge** (ball slid off the edge, not a fallthrough); all other conditions reached stable. Cause unknown — possibly related to res=15 edge node layout at this thickness.
- **thickness=0.30 increases node count to 180**, slightly higher wall_time, and the highest `second_apex_height_m` (0.059 m), indicating slowest energy decay.
- **Current default thickness=0.10** is a reasonable choice: 120 nodes, wall_time ~53 s, moderate compression, numerically stable.

### Scripts and output conventions

- MuJoCo unified script: `scripts/mujoco_trampoline_ball_drop.py`
- IsaacLab unified script: `scripts/isaaclab_trampoline_ball_drop.py`
- MuJoCo per-run artifact root: `logs/mujoco_ball_drop_runs/`
- IsaacLab per-run artifact root: `logs/isaaclab_trampoline_ball_drop_runs/`
- Each run creates a timestamped directory whose name encodes the label and key simulator parameters.
- Per-run directory contains:
  - `ball_drop_summary.csv`: single-run summary metrics.
  - `ball_drop_trajectory.csv`: per-step ball position/velocity, trampoline top-center position/velocity, compression, and stable/apex/contact/release flags.
  - `ball_drop_params.yaml`: run parameters, stored in the same directory as the summary CSV.
  - `ball_drop_vertical_state.png`: 2×1 plot of ball and trampoline center vertical position/velocity.
  - `ball_drop_compression.png`: trampoline center compression over time.
  - `ball_drop_video.mp4`: video.
- Sweep mode uses the same script with `--sweep PARAM VALUE...` or `--sweep_config YAML_OR_JSON`, and writes `ball_drop_sweep_summary.csv` in the sweep root.

Example commands:

```bash
python scripts/mujoco_trampoline_ball_drop.py --label nominal
python scripts/mujoco_trampoline_ball_drop.py --sweep solref "0.012 1" "0.015 1" "0.018 1"
python scripts/mujoco_trampoline_ball_drop.py --sweep_config mujoco_sweep.yaml
python scripts/isaaclab_trampoline_ball_drop.py --headless --label nominal
python scripts/isaaclab_trampoline_ball_drop.py --headless --no-video --sweep_name damping_sweep --sweep elasticity_damping 0.01 0.03 0.1
python scripts/isaaclab_trampoline_ball_drop.py --sweep_config sweep.yaml
```

## Next steps

- **Center vs. off-center drop**
   - Test center and slightly off-center drop points separately
   - Measure yaw/roll/pitch sensitivity

- **Robot static load and drop**
   - Robot standing still, gradually raised/lowered
