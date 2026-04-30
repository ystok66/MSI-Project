# Tutor Utility 各项计算细节报告

生成时间：2026-04-23  
范围：`cls_option_tutor/one_hint_tutor` 当前主线 `advantage_delta` 与机制线 normalized `eta-mix`。

本文专门解释 tutor utility 里每一项的具体含义、代码里如何估计、为什么加入，以及它如何影响 hint 选择。

---

## 1. 统一符号

一个候选 hint 记作：

```text
h
```

teach query 的最大尝试数：

```text
T = max_attempts_main
```

当前主线和 frontier 中通常是：

```text
T = 5
H = 1
T_no_tutor_bonus = T + H = 6
```

learner 第一次选中 correct option 的 attempt 记作：

```text
tau
```

如果 learner 在预算内没有选中 correct，则：

```text
tau = None
```

planner 不知道真实 tau，只能用 shadow learner rollout 估计：

```text
P_h(tau = t), t = 1..T
```

代码中对应字段：

```text
pred_p_tau_1_to_6
```

虽然字段名带 `1_to_6`，实际长度由当前 rollout horizon 决定；T=5 时就是 5 个概率。

---

## 2. 候选 hint 后的 shadow rollout 如何产生统计量

对每个 candidate hint：

1. clone shadow learner；
2. 对 shadow learner apply hint：

```text
scorer.incremental_study([(hint_words, hint_output)])
```

3. 在 teach menu 上模拟 learner pick；
4. 如果选错，则 reveal 该 wrong option 的真实 rendered output；
5. 按 planning update mode 更新 shadow state；
6. 重复直到 correct 或达到 T。

当前主要使用两类 rollout：

| 路径 | 用途 |
|---|---|
| score-table MC rollout | proxy / refine 的主要快速估计 |
| first-reveal cached CLS | 对最可能的 first wrong reveal 做语义更新缓存，提高 post-reveal dynamics 可信度 |

每个 rollout path 记录：

```text
success
first_correct_attempt
safe_wrong_count
risk_count
damage_sum
attempt-level correct probability/rank
```

多条 rollout path 聚合后得到：

```text
success_prob
pred_p_tau_1_to_6
safe_wrong_mean
risk_count_mean
wrong_before_correct_mean
pred_attempt_correct_prob_mean
pred_attempt_correct_rank_mean
eval_cell_acc
```

---

## 3. `P(tau <= T)`：bounded teach success

### 定义

```text
P_h(tau <= T)
```

表示 learner 在 T 次尝试内选中 correct option 的预测概率。

### 代码中如何计算

rollout 每条 path 如果成功，会记录：

```text
first_correct_attempt = t
```

聚合时：

```text
pred_p_tau_1_to_6[t-1] += path_weight
success_prob = sum_t pred_p_tau_1_to_6[t-1]
```

MC rollout 中：

```text
success_prob = success_count / n_rollouts
pred_p_tau_1_to_6 = tau_counts / n_rollouts
```

beam rollout 中：

```text
success_prob = sum_path weight(path) * 1[path success]
pred_p_tau_1_to_6[t-1] = sum_path weight(path) * 1[tau=t]
```

### 为什么需要

这是最核心项。它回答：

```text
这个 hint 是否让 learner 在有限 teach budget 内完成当前题？
```

主线 ablation 已经显示，去掉 success 项后 tutor success 从 `0.55` 掉到 `0.20`，所以这是主线 utility 的核心驱动力。

---

## 4. `P(tau > T)` / fail probability

### 定义

```text
P_h(tau > T) = 1 - P_h(tau <= T)
```

代码中：

```text
fail_prob = max(0, 1 - success_prob)
```

### 在主线里为什么说它冗余

主线是 delta utility：

```text
Delta U(h) = U(h) - U(no_tutor_T+H)
```

如果只看 success/fail 两项：

```text
lambda_success * P_success
- lambda_fail * P_fail
```

因为：

```text
P_fail = 1 - P_success
```

所以：

```text
lambda_success * P_success - lambda_fail * (1 - P_success)
= (lambda_success + lambda_fail) * P_success - lambda_fail
```

做 delta 后常数项抵消：

```text
Delta = (lambda_success + lambda_fail) * Delta P_success
```

因此在主线 `advantage_delta` 下，`lambda_success=3` 和 `lambda_fail=8` 等价于：

```text
effective success weight = 11
```

这就是为什么报告里说 fail term 可并入 success term。

---

## 5. `P(3 <= tau <= T)`：band success

### 定义

```text
P_h(3 <= tau <= T)
```

代码函数：

```text
band_success_prob(aggregate, cfg)
```

具体计算：

```text
tau = pred_p_tau_1_to_6
tau_min = target_tau_min = 3
tau_max = rollout horizon
band = sum(tau[tau_min-1 : tau_max])
```

### 注意：为什么 bonus baseline 的 band 更宽

如果计算的是 no-tutor bonus baseline，horizon 是：

```text
T + H
```

代码有意让 band 跟着实际 rollout horizon 走，所以 no-tutor bonus 的 band 是：

```text
3..T+H
```

而 tutor 的 band 是：

```text
3..T
```

这是公平控制的一部分，因为 no tutor 确实被多给了 H 次尝试。

### 为什么需要

只优化 success 会鼓励“马上撞对”。band success 鼓励 learner 在合理教学窗口内成功，避免：

```text
tau <= 2 且 eval 无提升
```

这种“太早成功但没学会”的浅层胜利。

---

## 6. `P(tau < 3)`：early success

### 定义

```text
P_h(tau < target_tau_min)
```

当前：

```text
target_tau_min = 3
early = P(tau=1) + P(tau=2)
```

代码函数：

```text
early_success_prob(aggregate, cfg)
```

### 用途

在 transfer utility 中，early success 是负项：

```text
- lambda_transfer_early * P(tau < 3)
```

因为 transfer-oriented tutor 不希望 learner 太快猜中而少看有用 reveal。

在 fast utility 中，early 不被惩罚，反而通过 `P(tau<=2)` 被鼓励。

---

## 7. `P(tau <= 2)`：fast success

### 定义

```text
P_h(tau <= 2)
```

机制线 fast utility 中显式使用：

```text
10 * P(tau <= 2)
```

### 代码中如何计算

优先使用：

```text
pred_tau_le2_exact
```

如果没有该字段，则退化为：

```text
pred_p_tau_1_to_6[0] + pred_p_tau_1_to_6[1]
```

代码函数：

```text
tau_le2_prob(aggregate, cfg)
```

### `pred_tau_le2_exact` 的来源

在 score-table / first-reveal cached 路径里，可以近似枚举两步成功：

```text
P(tau=1) = p0(correct)
P(tau=2) = sum_{a != correct} p0(a) * p1(correct | first wrong=a)
```

其中：

```text
p0
```

是 hint 后第一步 pick distribution；

```text
p1(correct | first wrong=a)
```

是 learner 第一步选错 `a`、看到 reveal `a -> output(a)` 后，第二步选 correct 的概率。

first-reveal cached CLS 会对 top wrong actions 构建 post-reveal score table，所以这个量比单纯 initial rank 更动态。

### 为什么需要

`soft_tau_center=2` 只是弱 shaping，不等价于直接优化 fast success。机制线里加入 `P(tau<=2)` 是为了让 fast tutor 真正偏向：

```text
更快完成当前 teach
更少 wrong reveal
更少 teach update
```

---

## 8. `R_tau(h)`：soft timing reward

### 定义

```text
R_tau(h)
  = sum_t P_h(tau=t) * exp(-(t - c)^2 / (2 sigma^2))
```

代码函数：

```text
soft_tau_score(aggregate, cfg)
```

当前主线参数：

```text
soft_tau_center = 4.5
soft_tau_sigma  = 1.5
```

### 具体计算

对 `pred_p_tau_1_to_6` 中每个 attempt：

```text
score += P(tau=t) * GaussianReward(t; c, sigma)
```

其中：

```text
GaussianReward(t; c, sigma)
  = exp(-(t-c)^2 / (2 sigma^2))
```

### 为什么需要

它比 hard band 平滑。hard band 只区分是否落在 `[3,T]`，而 soft tau 给每个 tau 一个连续 reward。

主线中它的作用是：

```text
偏好不要太早也不要太晚成功
```

机制线上我们发现它不是强 fast 控制旋钮；真正 fast 行为需要显式 `P(tau<=2)`。

---

## 9. `E[safe_wrong_before_correct]`

### 定义

```text
E_h[safe_wrong_before_correct]
```

表示 learner 在选中 correct 前，平均会选错多少个 non-risky wrong options。

在 no-risk setting 中，基本等价于：

```text
E[wrong_before_correct]
```

因为所有 wrong options 都是 safe。

### 代码中如何计算

每条 rollout path 维护：

```text
safe_wrong_count
risk_count
damage_sum
```

如果 pick wrong option：

```text
if damage == 0:
  safe_wrong_count += 1
else:
  risk_count += 1
```

聚合：

```text
safe_wrong_mean = mean_path safe_wrong_count
```

MC rollout 中是简单平均：

```text
safe_wrong_mean = safe_wrong_sum / n_rollouts
```

beam rollout 中是 path probability 加权平均：

```text
safe_wrong_mean = sum_path weight(path) * safe_wrong_count(path)
```

### 为什么在主线里是正项

主线里：

```text
+ lambda_exposure * E[safe_wrong_before_correct]
```

直觉：某些 wrong reveal 是教学性 exposure。learner 看到 wrong option 的真实 output，可能学到 operator / composition 规律，从而帮助 transfer。

### 为什么在 fast utility 里是负项

fast component 里用：

```text
- 4 * E[wrong_before_correct]
```

因为 fast tutor 的目标是：

```text
尽快做对
少看 wrong reveal
少发生 teach-time update
```

这正是 fast-vs-transfer 机制分析的关键差异。

---

## 10. `E[wrong_before_correct]`

### 定义

```text
E[wrong_before_correct]
  = E[safe_wrong_count + risky_wrong_count]
```

代码函数：

```text
wrong_before_correct_mean(aggregate)
```

优先读取：

```text
wrong_before_correct_mean
```

否则退化为：

```text
safe_wrong_mean + risk_count_mean
```

### 用途

主要用于 fast utility：

```text
U_fast includes -4 * E[wrong_before_correct]
```

它比 `tau` 更直接对应：

```text
learner 在 teach 中看到了多少 wrong reveal / update
```

---

## 11. `initial_correct_margin`

### 定义

hint 后、还没有任何 teach pick/reveal 前，shadow learner 的初始 pick distribution：

```text
p0(j)
```

correct option 的概率：

```text
p0(correct)
```

最强 wrong option 的概率：

```text
max_{j != correct} p0(j)
```

margin：

```text
initial_correct_margin
  = p0(correct) - max_{j != correct} p0(j)
```

### 代码中如何计算

在 `_initial_hint_stats_shadow` 和 score-table prefilter 里计算：

```text
initial_correct_prob_mean
initial_correct_rank_mean
initial_correct_margin_mean
```

### 为什么 fast utility 用它

fast component：

```text
+ 2 * initial_correct_margin
```

它鼓励 correct option 在初始 policy 上就明显领先，而不是 rank=1 但 margin 很小。

注意：initial rank/margin 是静态 proxy，不足以证明真实 tau<=2。因此现在 fast utility 同时使用动态项 `P(tau<=2)`。

---

## 12. `EvalCell(h)` / `EvalCellProxy(h)`

### 真实 eval cell

teach 结束后，用真实 learner 当前 scorer 对 eval items 做 direct prediction。

cell accuracy：

```text
EvalCell
  = correct_cells / total_cells
```

其中每个 eval item 比较：

```text
predicted_output vs true_output
```

### Planner 中的 eval proxy

主线 `advantage_delta` 中，eval-aware refine 可以来自 rollout 的 leaf eval 或 static proxy，具体取决于配置。

机制线 normalized eta-mix 当前使用：

```text
transfer_eval_proxy_mode = static_subset
```

具体做法：

1. 对 candidate hint apply 到 shadow learner；
2. 不跑完整 teach trajectory；
3. 在小 eval subset 上直接算 cell accuracy；
4. 写入：

```text
eval_cell_acc
eval_proxy_cell_acc
```

eval subset 当前配置：

```text
transfer_eval_proxy_n_per_diff = 3
transfer_eval_proxy_max_items = 9
```

### 为什么需要

之前 transfer utility 里的 eval proxy 是 0，导致 transfer mode 看不到候选之间的 eval 差异。接入 static eval proxy 后：

```text
selected_pred_eval_cell_mean > 0
```

transfer gate 才能按 eval improvement 选择 hint。

---

## 13. `C_collapse(h)`：collapse penalty

### 目的

惩罚那些看起来能帮助 learner，但在 wrong reveal 后让 correct option probability / rank 出现不稳定跳变的 hint。

这里的 collapse penalty 更准确地说是：

```text
post-reveal instability / over-aggressive reveal sensitivity penalty
```

代码函数：

```text
conservative_reveal_penalty(aggregate, cfg)
```

### 输入

rollout 会记录每个 attempt 的 correct probability：

```text
pred_attempt_correct_prob_mean = [p1, p2, p3, ...]
```

其中 `p_t` 是 rollout 中到达第 t 次尝试时，correct option 的平均 pick probability。

### 具体计算

代码读取非 None 的概率序列：

```text
vals = [p1, p2, p3, ...]
```

如果长度小于 2，penalty 为 0。

然后：

```text
first_gain = vals[1] - vals[0] - margin
if first_gain > 0:
  raw_penalty += first_jump_weight * first_gain
```

之后从第三个值开始：

```text
for cur in vals[2:]:
  gain = cur - prev - margin
  if gain > 0:
    raw_penalty += gain
  prev = cur
```

最终：

```text
C_collapse = conservative_reveal_penalty_weight * raw_penalty
```

当前常用参数：

```text
conservative_reveal_penalty_weight = 3.0
conservative_reveal_first_jump_weight = 2.0
conservative_reveal_monotone_margin = 0.01
```

### 为什么是惩罚“上升”而不是下降

这个 penalty 是 conservative reveal penalty。它惩罚 correct probability 在 reveal 后出现过强上升，因为这类 hint/rollout 可能依赖“过强的 reveal boost”而不是稳定语义学习，容易造成 planner 过度乐观。

早期开发中还检查过 probability/rank drop 的 collapse 诊断；当前 utility 中这个函数实际惩罚的是 reveal 后过强的 monotone gain。

### 在效用里的作用

主线：

```text
- lambda_collapse * C_collapse
```

transfer component：

```text
- 3 * C_collapse
```

fast component 当前：

```text
- 0 * C_collapse
```

也就是说 fast mode 基本不关心 collapse stability，而主线/transfer mode 会回避 post-reveal 过度敏感候选。

---

## 14. `expected_attempt_cost`

### 定义

```text
E[tau-like cost]
```

代码函数：

```text
expected_attempt_cost(aggregate, cfg)
```

具体：

```text
expected = sum_t t * P(tau=t)
fail_mass = 1 - sum_t P(tau=t)
expected += (T + 1) * fail_mass
```

失败被当作成本 `T+1`。

### 用途

主要用于旧的 `min_updates` / diagnostic utility：

```text
- lambda_min_updates_tau * expected_attempt_cost
```

当前主线和 normalized eta frontier 不以它为主。

---

## 15. `time_reward_mean`

### 定义

旧 reward：

```text
attempt_time_reward(t)
  = exp(-(t - target_attempt)^2 / (2 sigma_tau^2))
```

如果没有成功：

```text
reward = 0
```

### 用途

主要用于 legacy / success_gated utility。当前主线更多用 `soft_tau_score`，机制线用显式 `P(tau<=2)` 和 band/early。

---

## 16. Transfer gate 的具体计算

transfer-like mode 包括：

```text
advantage_transfer
advantage_mix with eta <= 0.25
```

当：

```text
transfer_gate_mode = eval_delta
```

使用：

```text
eval_cell_delta
  = candidate.eval_cell_acc - reference.eval_cell_acc

success_floor
  = max(
      transfer_success_floor,
      reference.success_prob - transfer_success_slack
    )
```

当前参数：

```text
transfer_delta_eval_min = 0.005
transfer_success_floor  = 0.15
transfer_success_slack  = 0.05
```

通过 gate 条件：

```text
eval_cell_delta >= 0.005
and candidate.success_prob >= max(0.15, reference.success_prob - 0.05)
```

### 为什么不用 band gate

transfer tutor 的目标不是一定提高 teach success，而是：

```text
提高 eval，且 teach 不完全崩
```

因此不能用 `no_band_candidate` 或 `early_only_candidates` 这种 search tutor gate 卡掉 transfer candidates。

---

## 17. Delta utility 与 no-tutor bonus baseline

所有核心 deployable utility 最终都可以使用 delta：

```text
Delta U(h)
  = U(h) - U(reference)
```

reference 通常是：

```text
no_tutor_T+H
```

也就是 no tutor 多给 H 次尝试。

这是为了公平比较：

```text
tutor gets 1 hint
no tutor gets 1 extra attempt
```

代码中 `_bonus_cfg` 会把 no-tutor baseline 的 rollout horizon 改成：

```text
T + H
```

---

## 18. Normalized eta-mix 的具体计算

未归一化混合：

```text
U_eta_raw(h)
  = eta * U_fast(h)
  + (1 - eta) * U_transfer(h)
```

问题：`U_fast` 和 `U_transfer` 尺度不同，所以 `eta=0.5` 不一定是中点。

当前 normalized mix：

```text
z_fast(h)
  = (U_fast(h) - mean_candidates U_fast) / std_candidates U_fast

z_transfer(h)
  = (U_transfer(h) - mean_candidates U_transfer) / std_candidates U_transfer

U_eta(h)
  = eta * z_fast(h)
  + (1 - eta) * z_transfer(h)
```

代码中会把以下字段写入 candidate record：

```text
fast_component_delta
transfer_component_delta
fast_component_z
transfer_component_z
selection_score
```

最终排序使用：

```text
selection_score = U_eta
```

---

## 19. 各 utility 的完整公式汇总

### `advantage_delta` 主线

```text
U_adv(h)
  = 3   * P(tau <= T)
  + 2   * EvalCell(h)
  + 2   * R_tau(h)
  + 0.5 * E[safe_wrong_before_correct]
  - 8   * P(tau > T)
  - 3   * C_collapse(h)
```

实际比较：

```text
Delta U_adv(h)
  = U_adv(h) - U_adv(no_tutor_T+H)
```

等价 success 权重：

```text
3 + 8 = 11
```

### `U_fast`

```text
U_fast(h)
  = 8  * P(tau <= T)
  + 10 * P(tau <= 2)
  - 4  * E[wrong_before_correct]
  + 2  * initial_correct_margin
  - 0  * C_collapse(h)
```

### `U_transfer`

```text
U_transfer(h)
  = 2 * P(tau <= T)
  + 6 * EvalCellProxy(h)
  + 4 * P(3 <= tau <= T)
  + 2 * E[safe_wrong_exposure]
  - 4 * P(tau < 3)
  - 3 * C_collapse(h)
```

### Normalized `U_eta`

```text
U_eta(h)
  = eta * z(U_fast(h))
  + (1 - eta) * z(U_transfer(h))
```

---

## 20. 每个项的直觉总结

| 项 | 计算来源 | 鼓励什么 | 风险 |
|---|---|---|---|
| `P(tau<=T)` | rollout success probability | 当前 teach 完成 | 可能只追求做对 |
| `P(tau>T)` | `1 - success` | 惩罚失败 | delta 下可并入 success |
| `P(3<=tau<=T)` | tau distribution sum | 教学窗口成功 | 可能牺牲快速完成 |
| `P(tau<3)` | early tau mass | 惩罚太早成功 | 不适合 fast mode |
| `P(tau<=2)` | first-reveal dynamic / tau sum | 快速完成 | 可能减少学习 exposure |
| `R_tau` | tau distribution 的 Gaussian reward | 平滑 timing preference | 只是弱 shaping |
| `safe_wrong_mean` | rollout wrong reveal count | 有用 exposure | 多 exposure 不一定有用 |
| `wrong_before_correct` | safe + risky wrong count | fast mode 中要减少 | 过少可能 eval 弱 |
| `initial_margin` | initial pick distribution | correct 初始领先更稳 | 静态 proxy，不够动态 |
| `EvalCellProxy` | hint 后小 eval subset | transfer | proxy 可能不准 |
| `C_collapse` | attempt-level correct prob jump | 稳定 post-reveal dynamics | 太强会保守 |

