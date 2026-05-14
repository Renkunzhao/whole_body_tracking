# Phase-Aware Teacher Reward 实现计划（给 Codex）

## 0. 目标

当前 pipeline 目标是：

1. 先训练一个 privileged teacher。
2. teacher 可以看到 trampoline 中心或局部 patch 的位置、速度，以及真实 trampoline 物理参数。
3. 通过 phase-aware reward，让 teacher 学会利用 trampoline 的回弹能量，而不是学成主动高频 pogo。
4. 再用 teacher-student / RMA，让 student 只依赖 deployable observation history 或 RNN hidden state 隐式推断 trampoline phase 和 dynamics。

本计划只实现 **teacher 训练阶段的 phase-aware reward**。

---

## 1. 需要修改的文件

优先检查并修改：

```text
go2_rebounce_env_cfg.py
mdp/commands.py
mdp/rewards.py
```

大致职责：

```text
go2_rebounce_env_cfg.py
- 增加 teacher observation term
- 增加 phase-aware reward 配置
- 增加 reward weight / beta 参数

mdp/commands.py
- 维护 bounce-level phase buffers
- 维护 stance / liftoff 状态
- 记录 q_phase，供 apex reward 使用

mdp/rewards.py
- 实现 phase-aware apex reward
- 可选实现 bad-phase motor work penalty
- 可选实现 impact / hard-smash penalty
```

---

## 2. Trampoline phase state 定义

读取 trampoline center 或 robot 下方 patch 的高度和速度。

第一版可以用 trampoline center：

```math
z_s(t) = z_center(t)
u_s(t) = dz_center(t) / dt
```

更好的版本是 robot 下方 patch：

```math
z_s(t) = weighted average of trampoline vertices under robot base / feet
u_s(t) = weighted average vertical velocity of that patch
```

定义压缩量：

```math
d_s(t) = z_rest - z_s(t)
dot_d_s(t) = -u_s(t)
```

物理含义：

```text
d_s > 0, u_s < 0:
    trampoline 被压缩，且仍在向下运动，compression phase

d_s large, u_s approx 0:
    接近最大压缩

d_s > 0, u_s > 0:
    trampoline 仍被压缩，且正在向上释放，release phase

d_s approx 0, u_s > 0:
    接近或已经过 rest height，可能偏晚
```

理想 liftoff 通常应发生在：

```text
d_s > 0 and u_s > 0
```

即 trampoline 仍压缩且正在向上回弹。

---

## 3. Teacher observation 增加项

只给 privileged teacher 增加 trampoline phase observation。

不要给 deployable student actor 增加这些量。

建议 teacher obs 增加：

```text
z_s
u_s
d_s
dot_d_s
```

建议做 normalization：

```text
z_s_norm = z_s / z_scale
u_s_norm = u_s / u_scale
d_s_norm = d_s / d_scale
dot_d_s_norm = dot_d_s / u_scale
```

推荐初值：

```text
d_scale = 0.10 ~ 0.20 m
u_scale = 1.0 ~ 2.0 m/s
```

teacher observation group 最终类似：

```text
teacher_obs = [
    deployable_obs,
    privileged_base_obs,
    true_trampoline_params,
    trampoline_phase_obs,
]
```

policy / deployable obs 保持不变。

---

## 4. 需要维护的 per-env buffers

在 command term 或 stateful MDP helper 中维护：

```python
# trampoline phase state
last_z_s
current_z_s
current_u_s
current_d_s
current_dot_d_s

# stance state
in_stance
prev_in_stance

# stance accumulators
stance_release_impulse
stance_total_impulse

# liftoff latched values
last_liftoff_d
last_liftoff_u
last_q_liftoff
last_q_stance
last_q_phase

# optional diagnostics
last_touchdown_vz
last_max_normal_force
```

reward 读取 `last_q_phase`。

---

## 5. Stance / contact 检测

优先使用真实或估计的 foot normal contact force。

推荐顺序：

```text
1. deformable contact force if available
2. DOB estimated foot force
3. existing contact sensor
4. fallback: foot height close to trampoline surface
```

定义：

```math
Fz_total = sum_i max(Fz_i, 0)
```

stance flag：

```python
in_stance = Fz_total > Fz_contact_threshold
```

初始阈值：

```text
Fz_contact_threshold = 20 ~ 50 N total
```

如噪声大，用 hysteresis：

```text
enter stance if Fz_total > F_enter
exit stance if Fz_total < F_exit
F_enter > F_exit
```

---

## 6. Release gate

定义一个 smooth release gate，表示 trampoline 处于“压缩且向上回弹”状态。

```math
g_rel(t)
=
sigmoid((u_s(t) - u_min_rel) / s_u)
*
sigmoid((d_s(t) - d_min_rel) / s_d)
```

推荐初值：

```text
u_min_rel = 0.05 ~ 0.15 m/s
d_min_rel = 0.005 ~ 0.02 m
s_u = 0.05 ~ 0.15 m/s
s_d = 0.005 ~ 0.02 m
```

---

## 7. Stance phase score

在 stance 中累计支撑冲量有多少发生在 release phase。

每个 control step，如果 `in_stance`：

```math
J_rel += Fz_total * g_rel * dt
J_tot += Fz_total * dt
```

liftoff 时计算：

```math
q_stance = J_rel / (J_tot + eps)
```

并 clamp：

```python
q_stance = torch.clamp(q_stance, 0.0, 1.0)
```

解释：

```text
q_stance 接近 1:
    支撑冲量主要发生在 trampoline 向上 release 阶段

q_stance 接近 0:
    支撑冲量主要发生在 compression 或错误相位
```

这个量是比例，不会直接鼓励更大冲击。

---

## 8. Liftoff phase score

liftoff 时 latch：

```math
d_lo = d_s(t_liftoff)
u_lo = u_s(t_liftoff)
```

目标高度归一化：

```math
h_bar = clamp((h_star - h_min) / (h_max - h_min), 0, 1)
```

推荐：

```text
h_min = 0.5
h_max = 1.2
```

手写 target-conditioned phase window：

```math
u_min(h*) = u0 + u1 * h_bar
d_min(h*) = d0 + d1 * h_bar
d_max(h*) = d2 + d3 * h_bar
```

推荐初值：

```text
u0 = 0.05 m/s
u1 = 0.30 m/s

d0 = 0.005 m
d1 = 0.04 m

d2 = 0.10 m
d3 = 0.05 m

s_u_lo = 0.10 m/s
s_d_lo = 0.02 m
```

计算：

```math
q_lo =
sigmoid((u_lo - u_min(h*)) / s_u_lo)
*
sigmoid((d_lo - d_min(h*)) / s_d_lo)
*
sigmoid((d_max(h*) - d_lo) / s_d_lo)
```

解释：

```text
第一项:
    liftoff 时 trampoline 向上速度要足够

第二项:
    liftoff 时 trampoline 仍有压缩，不能太晚

第三项:
    liftoff 不能发生在过深压缩处，不能太早主动弹走
```

最后 clamp：

```python
q_lo = torch.clamp(q_lo, 0.0, 1.0)
```

---

## 9. Combined phase score

liftoff 时合成：

```math
q_phase = alpha * q_lo + (1 - alpha) * q_stance
```

推荐：

```text
alpha = 0.5
```

实现：

```python
q_phase = torch.clamp(q_phase, 0.0, 1.0)

last_q_phase[env_ids] = q_phase
last_q_liftoff[env_ids] = q_lo
last_q_stance[env_ids] = q_stance
last_liftoff_d[env_ids] = d_lo
last_liftoff_u[env_ids] = u_lo
```

---

## 10. Phase-aware apex reward

当前 height quality：

```math
q_h =
exp(-(abs(last_apex_height - last_apex_target_height) / sigma_h)^2)
*
g_flat
```

其中：

```text
sigma_h = 0.10 m
```

新增 phase-aware apex reward：

```math
r_apex_phase
=
I[valid_apex_delayed]
*
q_h
*
((1 - beta) + beta * q_phase)
```

注意：不要一开始直接用：

```math
q_h * q_phase
```

因为太稀疏。

使用 mixed form：

```math
q_h * ((1 - beta) + beta * q_phase)
```

这样前期仍然有 height tracking signal，后期 phase alignment 逐渐变重要。

---

## 11. Phase beta curriculum

推荐 curriculum：

```text
Stage 0:
    beta = 0.0
    只学稳定 rebounce 和 height tracking

Stage 1:
    beta = 0.3
    弱 phase alignment

Stage 2:
    beta = 0.6
    phase alignment 明显参与优化

Stage 3:
    beta = 0.8
    final style shaping
```

如果当前项目中 curriculum 不方便，先暴露 config：

```python
phase_reward_beta = 0.3
```

后续手动调参。

不要从 beta = 1.0 开始。

---

## 12. Pseudocode: phase state update

```python
def update_phase_state(env, dt):
    # 1. Read trampoline patch / center state.
    z_s = get_trampoline_patch_height(env)
    u_s = (z_s - last_z_s) / dt
    d_s = z_rest - z_s
    dot_d_s = -u_s

    # 2. Read contact force.
    Fz_total = get_total_foot_normal_force(env)
    in_stance_now = Fz_total > Fz_contact_threshold

    # 3. Release gate.
    g_rel = torch.sigmoid((u_s - u_min_rel) / s_u_rel)
    g_rel = g_rel * torch.sigmoid((d_s - d_min_rel) / s_d_rel)

    # 4. Touchdown.
    touchdown = (~prev_in_stance) & in_stance_now
    if touchdown.any():
        stance_release_impulse[touchdown] = 0.0
        stance_total_impulse[touchdown] = 0.0
        max_Fz_this_stance[touchdown] = Fz_total[touchdown]
        touchdown_vz[touchdown] = base_vz[touchdown]

    # 5. Accumulate during stance.
    stance_ids = in_stance_now
    stance_release_impulse[stance_ids] += Fz_total[stance_ids] * g_rel[stance_ids] * dt
    stance_total_impulse[stance_ids] += Fz_total[stance_ids] * dt
    max_Fz_this_stance[stance_ids] = torch.maximum(
        max_Fz_this_stance[stance_ids],
        Fz_total[stance_ids],
    )

    # 6. Liftoff.
    liftoff = prev_in_stance & (~in_stance_now)
    if liftoff.any():
        q_stance = stance_release_impulse[liftoff] / (
            stance_total_impulse[liftoff] + eps
        )
        q_stance = torch.clamp(q_stance, 0.0, 1.0)

        d_lo = d_s[liftoff]
        u_lo = u_s[liftoff]

        h_star_lo = h_star[liftoff]
        h_bar = torch.clamp((h_star_lo - h_min) / (h_max - h_min), 0.0, 1.0)

        u_min = u0 + u1 * h_bar
        d_min = d0 + d1 * h_bar
        d_max = d2 + d3 * h_bar

        q_lo = torch.sigmoid((u_lo - u_min) / s_u_lo)
        q_lo = q_lo * torch.sigmoid((d_lo - d_min) / s_d_lo)
        q_lo = q_lo * torch.sigmoid((d_max - d_lo) / s_d_lo)
        q_lo = torch.clamp(q_lo, 0.0, 1.0)

        q_phase = alpha * q_lo + (1.0 - alpha) * q_stance
        q_phase = torch.clamp(q_phase, 0.0, 1.0)

        last_q_phase[liftoff] = q_phase
        last_q_liftoff[liftoff] = q_lo
        last_q_stance[liftoff] = q_stance
        last_liftoff_d[liftoff] = d_lo
        last_liftoff_u[liftoff] = u_lo

    # 7. Save previous state.
    prev_in_stance = in_stance_now
    last_z_s = z_s
```

---

## 13. Pseudocode: phase-aware apex reward

```python
def phase_aware_apex_reward(env):
    valid = command.valid_apex_delayed

    e_h = torch.abs(
        command.last_apex_height - command.last_apex_target_height
    )

    q_h = torch.exp(-torch.square(e_h / sigma_h))
    q_h = q_h * flat_body_gate(env)

    q_phase = torch.clamp(command.last_q_phase, 0.0, 1.0)

    phase_factor = (1.0 - beta) + beta * q_phase

    return valid.float() * q_h * phase_factor
```

先沿用原来的 height reward weight：

```text
+50
```

---

## 14. Optional: bad-phase motor work penalty

第一版可以先不加，只 logging。

如果要加，使用 motor-side positive work，而不是 foot contact power：

```math
P_motor_pos = sum_j max(tau_j * qdot_j, 0)
```

定义 late compression gate：

```math
g_late_comp =
sigmoid((d_s - d_late) / s_d)
*
sigmoid((u_late - abs(u_s)) / s_u)
```

定义目标升高需求：

```math
rho_up =
sigmoid((h_star - last_apex_height) / s_h)
```

允许正功窗口：

```math
g_allow_push = clamp(g_rel + rho_up * g_late_comp, 0, 1)
```

bad push：

```math
r_bad_push =
P_motor_pos * (1 - g_allow_push)
```

最终 reward 中以负号加入：

```text
- lambda_bad_push * r_bad_push
```

含义：

```text
release 阶段允许正功
目标升高时 late compression 允许部分正功
early compression 中的大正功会被惩罚
```

建议初值：

```text
lambda_bad_push = 1e-4 ~ 1e-3
```

---

## 15. Optional: impact / hard-smash penalty

用于防止 teacher 学会用很大冲击制造 phase reward。

可先 logging，再决定是否进 reward。

形式一：max normal force

```math
r_impact =
max(0, (Fz_max - F_safe) / F_scale)^2
```

形式二：touchdown vertical velocity

```math
r_td_vel =
max(0, (abs(vz_touchdown) - v_safe) / v_scale)^2
```

权重必须小，避免压制正常 trampoline loading。

---

## 16. Logging metrics

新增：

```text
Metrics/phase/q_phase_mean
Metrics/phase/q_liftoff_mean
Metrics/phase/q_stance_mean
Metrics/phase/liftoff_d_mean
Metrics/phase/liftoff_u_mean
Metrics/phase/release_impulse_ratio
Metrics/phase/stance_total_impulse
Metrics/phase/stance_release_impulse

Metrics/trampoline/z_s_mean
Metrics/trampoline/u_s_mean
Metrics/trampoline/d_s_mean
Metrics/trampoline/d_dot_s_mean
```

保留已有：

```text
apex count
height matched apex count
height success rate
mae
rmse
bias
h_over_target
positive work per height
absolute work per height
braking ratio
xy drift
yaw drift
orientation rms
failure distribution
```

建议额外导出 phase portrait 数据：

```text
(d_s, u_s) trajectory
touchdown points
liftoff points
valid apex points
```

好的策略应该让 liftoff points 聚集在：

```text
d_s > 0 and u_s > 0
```

---

## 17. 初始配置汇总

```python
phase_reward_beta = 0.3
phase_alpha_liftoff = 0.5

sigma_h = 0.10

u_min_rel = 0.10
d_min_rel = 0.01
s_u_rel = 0.10
s_d_rel = 0.01

h_min = 0.5
h_max = 1.2

u0 = 0.05
u1 = 0.30

d0 = 0.005
d1 = 0.04

d2 = 0.10
d3 = 0.05

s_u_lo = 0.10
s_d_lo = 0.02

Fz_contact_threshold = 30.0

eps = 1e-6
```

---

## 18. 训练顺序

推荐顺序：

```text
1. 保留原 height reward，训练/验证 privileged teacher baseline。

2. 只给 teacher 加 trampoline phase observation，reward 不变。
   检查没有 break，开始 logging phase variables。

3. 加 phase-aware apex reward，beta = 0.3。
   观察 height success、MAE、q_phase、视频行为。

4. 如果稳定，beta 增加到 0.6。
   检查 high target / high damping 是否 under-jump。

5. 如果仍稳定，再尝试 beta = 0.8。

6. 只有在 phase reward 稳定后，再考虑 bad-phase motor work penalty。
```

---

## 19. 成功标准

好的 phase-aware teacher 应该表现为：

```text
success rate 高
height success rate 高
MAE 低
apex count 合理
liftoff points 集中在 d_s > 0, u_s > 0
q_phase 提高
q_stance 提高
early-compression positive motor work 下降
positive work per height 下降或 rebound efficiency 变好
hard impact 没有明显增加
non-foot contact 没有增加
```

坏现象：

```text
height tracking 下降
teacher high target under-jump
teacher 学成慢跳 / passive 边界策略
teacher 用很大冲击砸 trampoline
q_phase 上升但 height success 下降
apex count 崩掉
failure 增加
```

应对：

```text
降低 beta
放宽 u_min / d_min
推迟 phase reward curriculum
先不要加 bad-work penalty
只 logging phase metrics，重新观察视频
```

---

## 20. Student 阶段留待后续

teacher 训练完成后再做 student。

student 不应该看到：

```text
z_s, u_s, d_s, dot_d_s
true trampoline parameters
root position / root linear velocity
```

student 使用：

```text
deployable observation history
or RNN hidden state
or RMA-style latent estimator
```

可以加入 auxiliary phase estimation loss：

```math
L_phase =
||d_hat - d_s||^2
+
||u_hat - u_s||^2
+
||sin_phi_hat - sin_phi||^2
+
||cos_phi_hat - cos_phi||^2
```

其中：

```math
phi = atan2(u_s / u_scale, d_s / d_scale)
```

最终 student 学的是从本体历史中隐式估计 trampoline phase / dynamics，并模仿 phase-aware teacher。

---

## 21. 最核心公式

最终 teacher reward 主项：

```math
r_apex_phase =
I[valid_apex]
*
q_height
*
((1 - beta) + beta * q_phase)
```

其中：

```math
q_phase = 0.5 * q_liftoff + 0.5 * q_stance
```

```text
q_liftoff:
    liftoff 是否发生在 trampoline 仍压缩且向上回弹的窗口

q_stance:
    stance 支撑冲量有多少比例发生在 trampoline release phase
```

这让 teacher 的节奏标准来自 trampoline 物理状态，而不是人为周期、apex 频率或纯省力。
