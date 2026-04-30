# cls_option_tutor / one_hint_tutor 当前系统详细说明

生成日期：2026-04-30  
范围：`cls_option_tutor/one_hint_tutor` 当前 one-hint tutor 主线、utility ablation、fast/transfer frontier 机制线。  
主要代码入口：

```text
cls_option_tutor/one_hint_tutor/protocol.py
cls_option_tutor/one_hint_tutor/learner_runner.py
cls_option_tutor/one_hint_tutor/inverse_particles.py
cls_option_tutor/tutor/inverse_predictor.py
cls_option_tutor/tutor/learner_model.py
cls_option_tutor/one_hint_tutor/hint_space.py
cls_option_tutor/one_hint_tutor/hint_planner.py
cls_option_tutor/one_hint_tutor/metrics.py
```

结果文件来源：

```text
result/utility_ablation_search_main_free_operator_20seed_summary.md
result/fast_transfer_frontier_norm_eta_08_10_16seed_summary.md
result/fast_transfer_frontier_norm_eta_05_07_16seed_summary.md
result/fast_transfer_frontier_norm_eta_00_16seed_summary.md
result/fast_transfer_frontier_beam_light_eta_00_07_8seed_summary.md
```

---

## 1. 当前系统一句话

当前 `one_hint_tutor` 是一个 **known-prelearn + observation-calibrated inverse tutor**：

```text
真实 learner 先用 prelearn examples 学习；
tutor 知道这批 prelearn curriculum，并用同一批 examples 初始化 shadow learner；
tutor 通过 observation 阶段看到 learner 的公开动作和 reveal history；
tutor 用这些公开 trace 更新 profile posterior 和 shadow learner state；
tutor 在 teach 阶段从候选 hint pool 中选择一个 hint；
最后比较 tutor、no tutor、random hint、oracle hint 的 teach success 和 eval transfer。
```

这不是 observation-only tutor。当前 tutor 不需要从零推断 learner prelearn 过什么；它知道 learner 的 prelearn examples。Observation 的作用是校准 learner 的行为 profile 和公开 reveal 后的 shadow state。

---

## 2. 实验流程

主流程在 `prepare_one_hint_experiment` 中：

```text
1. build task context
2. sample prelearn examples
3. build true base learner from prelearn examples
4. sample observation cases
5. run learner on observation cases
6. fit inverse posterior from observation traces
7. build rank-stratified teach case
8. build eval items
9. planner selects one hint
10. run tutor / no-tutor / random / oracle conditions
```

### 2.1 Prelearn

Prelearn 是 learner 的学习阶段。当前主线使用：

```text
prelearn_profile = 4
n_pre_easy = 1
n_pre_medium = 2
n_pre_hard = 1
```

真实 learner 用这些 examples 初始化 CLS semantic scorer。当前代码也把同一批 `prelearn_examples` 传给 inverse tutor，用于初始化 shadow learner。

这个设定可以解释为：

```text
课程系统知道自己之前给 learner 看过哪些 examples，
但不知道 learner 内部权重和行为风格。
```

### 2.2 Observation

Observation 阶段仍然存在，当前主线使用：

```text
n_obs = 4
obs_difficulty = hard
obs_menu_size = 6
```

tutor 在 obs 中看到的是公开 trace：

```text
target output
menu options
learner action
picked option index
是否正确
wrong pick 后 reveal 的 true rendered output
risk/damage/highlight/refresh 等公开状态
```

Observation 不负责从零发现 learner 的 prelearn data；它负责更新：

```text
P(profile psi | observation)
shadow learner state after public reveals
```

### 2.3 Teach

当前主 benchmark 使用 rank-stratified teach menu：

```text
teach_difficulty = hard
teach_menu_size K = 20
max_attempts_main T = 5
hint_count_budget H = 1
no_tutor_bonus_attempts = 1
```

rank-stratified 的目标是让 no tutor 不是 ceiling：

```text
target_initial_rank_min = 5
target_initial_rank_max = 12
teach_menu_build_trials = 24
```

也就是构造一个 learner 初始不容易、但仍可达的 hard menu。这样 tutor 才有展示空间。

### 2.4 Eval

默认 eval 从 teach menu 派生：

```text
eval_n_per_diff = 10
easy / medium / hard each 10
total ~= 30 eval items
```

主要指标：

```text
eval_exact = 完整 output 完全匹配比例
eval_cell = cell-level accuracy
```

frontier 机制线还会构造 exposure-sensitive eval，用于检查 fast tutor 是否因为少看 wrong reveals 而减少 transfer exposure。

---

## 3. Learner 机制

### 3.1 Semantic scorer

Learner 使用 CLS semantic scorer：

```text
use_cls = True
n_em = 1
tau_sem = 1.0
use_hpc = False
```

它学习 program words 到 rendered output 的关系。面对 teach menu 时，每个 option 会得到 semantic score：

```text
S_sem(j) = scorer(option_text_j, target_output)
```

score 越高，learner 越认为该 option 可能匹配 target output。

### 3.2 Pick policy

在 no-risk 主线中，learner 的 pick utility 可以近似理解为：

```text
U_pick(j) ≈ semantic_score(j) + memory / attention terms
```

然后通过 profile 相关参数转成 action probability：

```text
pi_psi(j | x_t)
  = policy under profile psi at current state x_t
```

可以直观理解为 softmax：

```text
pi(j) ∝ exp(beta_psi * U_pick(j))
```

不同 profile `psi` 会改变选择温度、attention、risk sensitivity 等行为参数。

### 3.3 Wrong reveal

当 learner 选错 option `a`：

```text
env reveals rendered_output(a)
```

也就是 learner 看到：

```text
wrong option text -> that option's true output
```

默认：

```text
reveal_learning_mode = cortex_em
```

因此 wrong reveal 会作为 semantic update 信号，让 learner 学到这个 wrong option 自己的 input/output mapping。它不是 correct answer 的 mapping，但可能帮助 learner 理解 operator 或排除相似结构。

当前 teach 统计：

```text
wrong_before_correct = correct pick 前选错次数
teach_updates = teach 阶段发生的 semantic update 次数
```

如果 tutor 让 learner 很快选对，learner 会看到更少 wrong reveals，teach-time update 也更少。这是 fast-vs-transfer 机制线关心的问题。

### 3.4 Correct pick

正确 pick 后 query 结束。当前：

```text
correct_pick_learning_mode = cortex_em
eta_correct_pick = 1.0
```

因此 correct pick 后也可做 positive semantic update。主线结果主要关注 hint 和 wrong reveal 如何影响 search / eval。

---

## 4. Inverse Tutor 机制

### 4.1 Shadow learner

tutor 不能读取真实 learner 内部状态，但它知道 learner 的算法族和 prelearn curriculum。它创建一个 shadow learner：

```text
shadow scorer = create_scorer(grammar, prelearn_examples, shadow_n_em=1)
shadow danger head
shadow attention state
```

重要设定：

```text
shadow_use_cls = True
shadow_n_em = 1
planning_lambda_neg = 2.0
eta_prof = 1.0
profile_top_mass = 0.90
profile_min_keep = 2
refine_profile_top_mass = 0.75
refine_profile_min_keep = 2
```

### 4.2 Profile psi

`profile psi` 是 learner 的一种行为假设，不是具体 hint，也不是具体 example。它表示 learner 如何把当前知识转成行动。

例子：

```text
psi_1: semantic scorer 权重高，选择更确定
psi_2: 温度高，选择更随机
psi_3: 更受 attention/highlight 影响
psi_4: 更 risk-sensitive
```

tutor 维护离散 profile 集合：

```text
Psi = {psi_1, psi_2, ..., psi_M}
```

每个 profile 给出一个 action model：

```text
pi_psi(a_t | x_t)
```

### 4.3 Observation posterior update

每个 observation step，tutor 用当前 shadow state 预测 observed action likelihood：

```text
L_t(psi) = pi_psi(a_t | x_t)
```

然后更新 profile posterior：

```text
w_t(psi) ∝ w_{t-1}(psi) * L_t(psi) ^ eta_prof
```

log-space：

```text
log w_t(psi)
  = log w_{t-1}(psi) + eta_prof * log pi_psi(a_t | x_t)
```

归一化后：

```text
sum_psi w_t(psi) = 1
```

如果某个 profile 很能解释 learner 行为，它的 posterior weight 上升。

### 4.4 Shadow state update

profile posterior 更新后，如果 observation step 中 learner 选错并看到 reveal，tutor 也用同样的公开 reveal 更新 shadow learner：

```text
shadow.update_from_reveal(
  wrong_text,
  revealed_output,
  danger_vec,
  damage
)
```

顺序很重要：

```text
1. 用 step t 前的 shadow state 预测 action likelihood
2. 用 observed action 更新 profile posterior
3. 再用 step t 的 reveal 更新 shadow state
```

这样避免用未来信息预测当前动作。

### 4.5 Teach-time prediction

到了 teach 阶段，tutor 用 profile posterior 的 mixture 做预测：

```text
P(action | state, hint h)
  = sum_psi P(psi | obs) * pi_psi(action | state, hint h)
```

hint utility 也是 profile mixture 下的期望：

```text
U(h)
  = E_{psi ~ posterior}[ outcome under shadow learner + profile psi + hint h ]
```

---

## 5. Hint Candidate Pool

### 5.1 主线 pool

主线 `search_main_free_operator` 使用：

```text
hint_families = ["free", "operator_probe"]
free_hint_pool_easy = 2
free_hint_pool_medium = 3
free_hint_pool_hard = 3
operator_probe_limit = 6
```

候选上限：

```text
free: 8
operator_probe: 6
total: 14 hint candidates
```

planner 内部还会加一个 `None` / no-hint baseline record：

```text
14 hint candidates + 1 no-hint baseline
```

### 5.2 Frontier pool

fast/transfer frontier 使用更大的机制分析 pool：

```text
free
operator_probe
answer_neighbor_nonanswer
target_neighborhood_robust_filtered
```

配置：

```text
free: 8
operator_probe: 8
answer_neighbor_nonanswer: 8
target_neighborhood_robust_filtered: 16
total max: 40
```

实际经过 grammar render、非法候选过滤、去重后，diagnostics 中大约是：

```text
28-29 candidates
```

进入 planner 阶段后：

```text
prefilter_top_k = 18
proxy evaluates = 18 hints + None
normal refine_top_k = 9 hints + None
beam-light refine_top_k = 5 hints + None
```

### 5.3 Candidate family 含义

| family | 含义 |
|---|---|
| `free` | 从 task example pools 中抽普通 non-answer example |
| `operator_probe` | 与 teach example 共享 operator 的 probe example，偏 operator 泛化 |
| `target_neighborhood_robust_filtered` | 从 teach example 附近生成结构相似但非答案的近邻，并做 rank/robust 过滤 |
| `answer_neighbor_nonanswer` | 基于 correct program 生成 non-answer 邻居，不直接给 correct words/output |

当前主 claim 只依赖 `free + operator_probe`。`answer_neighbor_nonanswer` 和 target robust 主要用于机制分析和 fast/transfer frontier。

---

## 6. Planner Cascade

当前 tutor 主要使用 cascade planner：

```text
planner_mode = cascade
prefilter_enabled = True
proxy_rollout_mode = mc
refine_enabled = True
refine_update_mode = first_reveal_cached_cls
```

流程：

```text
1. build hint candidates
2. build score tables for each candidate
3. prefilter by cheap static scores
4. proxy rollout estimates success/tau/exposure
5. optional transfer eval proxy
6. refine with first-reveal cached CLS dynamics
7. compute final utility
8. optionally abstain
9. selected hint applied to learner
```

主线参数：

```text
prefilter_top_k = 4 by default config
main grids may override planning parameters
refine_top_k = 2 by default config
```

frontier runner 显式设置：

```text
prefilter_top_k = 18
proxy_n_rollouts = 24
refine_top_k = 9
refine_n_rollouts = 12
first_reveal_top_b = 5
objective_bucketed_prefilter = True
objective_bucketed_refine = True
```

beam-light frontier 设置：

```text
refine_top_k = 5
refine_keep_fast = 1
refine_keep_transfer = 2
refine_keep_balanced = 1
transfer_eval_proxy_beam_top_b = 2
transfer_eval_proxy_beam_keep_l = 8
transfer_eval_proxy_max_items = 6
```

---

## 7. Utility 公式

### 7.1 Notation

```text
h = candidate hint
T = tutor teach attempt limit
H = hint budget
T0 = T + H = no-tutor bonus limit
tau = first correct attempt
A_eval = eval cell accuracy
```

主要概率来自 rollout：

```text
p_t(h) = P_h(tau = t)
P_h(tau <= T) = sum_{t=1..T} p_t(h)
P_h(tau > T) = 1 - P_h(tau <= T)
```

### 7.2 Main utility: advantage_delta

主线使用 `advantage_delta`，目标是相对 no-tutor bonus baseline 的增益：

```text
Delta U(h)
  = U_adv(h) - U_adv(no_tutor_T+H)
```

当前主线等价公式：

```text
U_adv(h)
  = 3.0 * P(tau <= T)
  + 2.0 * A_eval_cell(h)
  + 2.0 * R_tau(h)
  + 0.5 * E[safe_wrong_before_correct]
  - 8.0 * P(tau > T)
  - 3.0 * C_collapse(h)
```

因为：

```text
P(tau > T) = 1 - P(tau <= T)
```

在 delta utility 里：

```text
3 * success - 8 * fail
```

等价于一个更大的 success-margin 权重：

```text
(3 + 8) * Delta P(tau <= T)
```

所以主线里 success 是核心驱动力。

### 7.3 Soft timing reward

```text
R_tau(h)
  = sum_t P(tau=t | h) * exp(-(t - c)^2 / (2 sigma^2))
```

当前主线：

```text
soft_tau_center c = 4.5
soft_tau_sigma = 1.5
lambda_soft_tau = 2.0
```

含义：鼓励 learner 不要太早撞对，也不要太晚失败，而是在 teach window 中比较合适的时间学会。

### 7.4 Band success

```text
P_band(h) = P(target_tau_min <= tau <= T)
```

当前：

```text
target_tau_min = 3
T = 5 in main benchmark
```

band success 用于衡量“不是第 1-2 步直接撞对，而是在教学窗口内成功”。

### 7.5 Early success / early-no-transfer

```text
P_early(h) = P(tau < target_tau_min)
```

early-no-transfer 判定大致是：

```text
tau <= 2
and eval gain 不明显
```

主线里有 early gate / penalty：

```text
early_success_reject_prob = 0.40
early_success_eval_margin = 0.01
early_no_transfer_penalty = 4.0
```

但 utility ablation 显示，关闭 early gate 后 success 更高，说明 early guard 是风格/教学约束，不是纯 success 最大化项。

### 7.6 Exposure term

```text
E[safe_wrong_before_correct]
```

表示 correct 前 learner 看到的 safe wrong reveal 数。它鼓励 learner 在成功前获得一些非危险、可能有教学价值的 wrong reveal exposure。

注意：多 exposure 不一定等于有用 transfer。当前机制线已经看到，纯增加 wrong reveals 不一定提高 eval。

### 7.7 Collapse penalty

```text
C_collapse(h)
```

当前实现是 conservative reveal penalty，基于 attempt-level correct probability 的变化：

```text
pred_attempt_correct_prob_mean[t]
```

它惩罚“hint 让 learner 对 reveal 特别敏感/可能造成 post-reveal instability”的候选。核心参数：

```text
conservative_reveal_penalty_weight = 3.0
conservative_reveal_first_jump_weight = 2.0
conservative_reveal_monotone_margin = 0.01
```

直观解释：

```text
如果 hint 预测中 correct probability 的 post-reveal jump 过大或轨迹不稳，
planner 会更保守。
```

它不是实际 collapse rate，而是规划阶段的保守稳定性惩罚。

---

## 8. Fast / Transfer Eta-Mix Utility

机制线使用 normalized eta-mix：

```text
U_eta(h)
  = eta * z(U_fast(h)) + (1 - eta) * z(U_transfer(h))
```

其中：

```text
z(U) = (U - mean_candidates(U)) / std_candidates(U)
```

### 8.1 Fast utility

```text
U_fast(h)
  = 8.0 * P(tau <= T)
  + 10.0 * P(tau <= 2)
  - 4.0 * E[wrong_before_correct]
  + 2.0 * initial_correct_margin
  - 0.0 * C_collapse(h)
```

目标：

```text
尽快完成 teach
减少 wrong reveal / teach updates
不关心 eval
```

### 8.2 Transfer utility

```text
U_transfer(h)
  = 2.0 * P(tau <= T)
  + 6.0 * EvalCellProxy(h)
  + 4.0 * P_band(h)
  + 2.0 * E[safe_wrong_exposure]
  - 4.0 * P_early(h)
  - 3.0 * C_collapse(h)
```

目标：

```text
更偏 eval / transfer
保留一定 teach success floor
避免过早成功和 collapse
```

### 8.3 Transfer gate

transfer-like mode 使用 eval-delta gate：

```text
use hint if:
  predicted_eval_delta >= 0.005
  and predicted_success >= max(0.15, baseline_success - 0.05)
```

这替代了旧的 band/search gate。旧 gate 会导致 transfer mode 经常 abstain。

### 8.4 Eval proxy

当前有两类：

```text
static_subset:
  apply hint -> directly evaluate shadow learner on eval subset

beam_leaf_subset:
  rollout teach trajectory by beam approximation
  evaluate shadow learner at leaf states
```

当前 16-seed eta frontier 主结果使用 `static_subset`。`beam_leaf_subset` 已实现但较慢，并且 light beam 实验显示 predicted eval 可能过乐观。

---

## 9. Baselines and Controls

主线比较对象：

```text
no_tutor_T:
  no hint, T attempts

no_tutor_T+H:
  no hint, T+1 attempts
  用于补偿 hint 本身相当于额外示例

random_hard_hint_T:
  random hard hint, T attempts

random_same_pool_hint_T:
  从 tutor 同一 candidate pool 随机抽 hint, T attempts

tutor_T:
  planner selected hint, T attempts

oracle:
  用真实 outcome 后验选择 best candidate，作为 ceiling/headroom
```

final presentation 主线中：

```text
random_hard_n = 5
random_same_pool_n = 5
```

因此 summary 中有 random baseline std。

---

## 10. 当前主线实验设定

主线结果固定为：

```text
experiment = search_main_free_operator / utility_baseline_equiv
task = 000001
seeds = 0..19
prelearn = 4
obs = 4
teach difficulty = hard
menu = rank_stratified
K = 20
T_tutor = 5
T_no_tutor_bonus = 6
H = 1
hint_families = free + operator_probe
utility = advantage_delta
```

candidate pool：

```text
free 8 + operator_probe 6 = max 14 hint candidates
```

---

## 11. 主线结果

### 11.1 Main result table

| condition | success | eval cell |
|---|---:|---:|
| no_tutor_T | 0.20 | - |
| no_tutor_T+H | 0.25 | 0.2808 approx baseline |
| random_hard_hint_T | 0.33 | 0.2789 |
| random_same_pool_hint_T | 0.27 | 0.2807 |
| tutor_T | 0.55 | 0.2996 |
| oracle | 0.70 | 0.3493 |

Key deltas:

```text
Delta success vs no_tutor_T+H = +0.30
Delta success vs random_same_pool = +0.28
Delta eval cell vs no_tutor_T+H = +0.0188
Delta eval cell vs random_same_pool = +0.0189
```

Tutor trajectory:

```text
tutor band success = 0.45
tutor early success = 0.10
tutor early-no-transfer = 0.00
tutor soft tau score = 0.3997
```

Selected families:

```text
free: 13
operator_probe: 5
abstain: 2
```

Oracle headroom:

```text
oracle success headroom = 0.45
oracle band headroom = 0.40
oracle eval-cell headroom = 0.0685
```

Interpretation:

```text
当前主线足以支持：
在 rank-stratified hard option-search setting 下，
intentional one-hint tutor 明显优于 no-tutor bonus 和 random same-pool hint。
```

但仍有 oracle headroom，说明 planner 还远未最优。

---

## 12. Utility Ablation

主要 ablation 结果：

| variant | success | band | early-no-transfer | eval cell | interpretation |
|---|---:|---:|---:|---:|---|
| full / baseline | 0.55 | 0.45 | 0.00 | 0.2996 | balanced main utility |
| no_success | 0.20 | 0.15 | 0.05 | 0.2828 | success term 是核心 |
| no_eval | 0.55 | 0.45 | 0.00 | 0.2996 | eval 项在主 search benchmark 中不是主导 |
| no_soft_tau | 0.45 | 0.45 | 0.00 | 0.2976 | timing shaping 有贡献 |
| no_collapse | 0.55 | 0.45 | 0.00 | 0.2996 | collapse 项在当前主线影响小 |
| no_exposure | 0.40 | 0.40 | 0.00 | 0.2968 | exposure 项有一定作用 |
| no_early_gate | 0.65 | 0.45 | 0.00 | 0.3086 | 放松 early guard 可提高 success |
| strong_early_guard | 0.40 | 0.30 | 0.10 | 0.2767 | 过强 early guard 牺牲 success/eval |
| success_only | 0.50 | 0.40 | 0.00 | 0.2969 | 只追 success 不是最强 |
| search_minimal | 0.45 | 0.40 | 0.05 | 0.2877 | 简化 utility 会损失 |

Key conclusion:

```text
success term 是主线最关键项；
eval/collapse 在当前主 search benchmark 不是主要决策项；
soft_tau 和 exposure 是有用 shaping；
fail_prob 在 delta utility 下可并入 success margin；
early gate 控制教学风格，不是纯提升 success 的项。
```

---

## 13. Fast / Transfer Frontier

frontier 目标不是替代主线，而是分析：

```text
同一个 tutor 框架是否能通过 utility 参数 eta，
在 fast teach-success 和 eval transfer 之间移动。
```

frontier pool：

```text
free + operator_probe + answer_neighbor_nonanswer + target_neighborhood_robust_filtered
max 40, actual about 28-29 candidates
```

### 13.1 Static eval proxy frontier, 16 seeds

| eta | success | delta success vs no tutor | eval all | delta eval vs no tutor | wrong-before-correct | teach updates |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.5000 | +0.1875 | 0.3141 | +0.0199 | 3.5625 | 4.0625 |
| 0.9 | 0.5000 | +0.1875 | 0.3171 | +0.0228 | 3.5625 | 4.0625 |
| 0.8 | 0.5000 | +0.1875 | 0.3179 | +0.0236 | 3.5625 | 4.0625 |
| 0.7 | 0.4375 | +0.1250 | 0.3297 | +0.0355 | 3.6875 | 4.1250 |
| 0.6 | 0.3750 | +0.0625 | 0.3272 | +0.0330 | 3.8125 | 4.1875 |
| 0.5 | 0.3750 | +0.0625 | 0.3034 | +0.0091 | 3.7500 | 4.1250 |
| 0.0 | 0.3750 | +0.0625 | 0.3031 | +0.0089 | 3.8750 | 4.2500 |

Interpretation:

```text
eta=1.0..0.8:
  stronger teach/search, moderate eval

eta=0.7:
  empirical transfer sweet spot in current static proxy frontier

eta=0.0:
  pure transfer utility is not best eval because eval proxy is imperfect and gate/candidate flow matter
```

### 13.2 Beam-light proxy, 8 seeds

| mode | hint used | success | eval all | wrong-before-correct | teach updates |
|---|---:|---:|---:|---:|---:|
| eta=0.7 beam-light | 1.000 | 0.750 | 0.2974 | 3.000 | 3.750 |
| eta=0.0 beam-light | 0.625 | 0.750 | 0.2808 | 2.625 | 3.375 |

Beam-light conclusion:

```text
beam leaf proxy is implemented but expensive;
in light setting, predicted eval was over-optimistic;
it is not current main result.
```

---

## 14. Current Claims

### 14.1 Supported main claim

当前数据支持：

```text
In a rank-stratified hard option-search setting,
a one-shot inverse tutor that intentionally selects a hint from free+operator_probe candidates
improves bounded teach success over both no-tutor with one bonus attempt
and random same-pool hints.
```

中文：

```text
在 rank-stratified hard teach 设置下，
one-shot tutor 通过有意选择 hint，
比 no tutor 多一次尝试和 random same-pool hint 更稳定地帮助 learner 在有限尝试内完成 teach。
```

### 14.2 Secondary observation

```text
operator_probe 对 search 有帮助；
target/answer-neighbor family 对机制分析和 transfer frontier 有价值；
但 target/answer-neighbor 不应混入主 claim。
```

### 14.3 Frontier claim

当前可以谨慎说：

```text
normalized eta-mix utility shows an early search-vs-transfer direction,
but the frontier is non-monotonic because eval proxy and candidate cascade are imperfect.
```

---

## 15. Known Limitations

1. Tutor currently knows learner prelearn curriculum.

```text
This is curriculum-known inverse tutoring, not observation-only inverse tutoring.
```

如果要做 observation-only，需要把 learner prelearn knowledge 也放进 latent posterior。

2. Results are mainly on task `000001`.

当前主结果是：

```text
task = 000001
seeds = 0..19
```

论文级 claim 需要更多 tasks / held-out tasks。

3. Eval proxy mismatch remains.

尤其 transfer frontier 中：

```text
eta=0 不一定 eval 最强；
beam leaf predicted eval can be optimistic.
```

4. Oracle headroom remains large.

```text
oracle success = 0.70 vs tutor success = 0.55
oracle eval cell = 0.3493 vs tutor eval cell = 0.2996
```

说明候选池中还有更好的 hints，planner/reranker 仍有改进空间。

5. Exposure quantity is not equal to useful exposure.

更多 wrong reveals 不一定提升 eval。后续需要区分：

```text
operator-relevant exposure
target-relevant exposure
irrelevant safe exposure
collapse-inducing exposure
```

---

## 16. Presentation 建议

推荐把内容分成三条线：

```text
1. Main benchmark:
   advantage_delta + free/operator_probe
   claim: tutor improves bounded teach success

2. Utility ablation:
   success term is essential
   soft_tau/exposure shape pedagogical behavior
   fail term is redundant under delta

3. Mechanism frontier:
   eta-mix explores fast vs transfer
   eta=0.7 is empirical transfer sweet spot
   beam leaf proxy is future calibration work
```

最稳的主表：

| method | success |
|---|---:|
| no tutor T | 0.20 |
| no tutor T+1 | 0.25 |
| random hard hint | 0.33 |
| random same-pool hint | 0.27 |
| tutor selected hint | 0.55 |
| oracle hint | 0.70 |

一句话结论：

```text
当前主线已经能说明 tutor 的有意 hint selection 有实际 search/teach benefit；
transfer frontier 目前是机制分析和未来工作，不应替代主线结果。
```

