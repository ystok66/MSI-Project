# One-Hint Tutor 当前场景、机制与公式报告

生成时间：2026-04-22  
范围：`cls_option_tutor/one_hint_tutor` 当前 final presentation 相关实现与结果。

## 1. 当前要讲的两条线

现在系统里有两条需要分清的线：

| 线 | 目的 | 当前建议用途 |
|---|---|---|
| 主线 tutor：`advantage_delta` | 证明 tutor selected hint 是否真的优于 no tutor / random hint | Final presentation 主结果 |
| 机制线 tutor：normalized `eta-mix` | 解释同一个 tutor 能否因效用偏好不同而呈现 fast teach / eval transfer 风格差异 | Mechanism probe / analysis slide |

主线不要被机制线替代。主线已经比较稳定，机制线是为了说明效用项在控制 tutor 行为风格。

---

## 2. 当前实验场景

当前 one-hint tutor 协议是一个 option-search teaching 场景。

每个 task 给定一个 grammar。每个 example 有：

```text
program words -> rendered output
```

learner 先从少量 support examples 学语义映射，然后在 teach query 里看到一个 target output 和一个 option menu。menu 里只有一个 correct option，其余是 distractors。learner 的目标是在有限尝试内 pick correct option。

### 当前主 benchmark

主线固定为：

```text
task = 000001
prelearn = 4
obs = 4
teach difficulty = hard
menu = rank-stratified
K = 20
T_tutor = 5
H = 1
T_no_tutor_bonus = T + H = 6
hint families = free + operator_probe
utility = advantage_delta
```

其中 `rank-stratified` 是为了避免 no tutor ceiling。teach case 会反复采样 menu，并用 no-hint learner probe correct option 的初始 rank。当前目标区间是：

```text
target_initial_rank in [5, 12]
```

这样 no tutor 不是太容易，也不是完全不可达。

### 控制组

当前主结果同时比较：

```text
no_tutor_T
no_tutor_T+H
random_hard_hint_T
random_same_pool_hint_T
tutor_selected_hint_T
oracle_best_hint
```

关键控制是 `no_tutor_T+H` 和 `random_same_pool`。

`no_tutor_T+H` 给 no tutor 多一次尝试，补偿 one hint 本身是一个额外 example。  
`random_same_pool` 从 tutor 可用候选池里随机抽 hint，检验提升是否只是“多看一个 hint”，而不是 tutor 有意选择。

---

## 3. 整体流程

当前代码路径大致是：

```text
prepare_one_hint_experiment
  -> build_task_context
  -> sample_prelearn_examples
  -> build_base_learner
  -> build_observation_cases
  -> run_observation_case
  -> fit_inverse_posterior
  -> build rank-stratified teach_case
  -> build eval_items
  -> select_hint
  -> finalize_prepared_experiment
      -> run no_tutor / random / tutor conditions
```

### 3.1 Prelearn

learner 用少量 support examples 初始化 CLS semantic scorer：

```text
prelearn examples = easy + medium + hard
```

当前主线是 `prelearn_profile=4`，即总共 4 个 prelearn examples。

### 3.2 Observation

系统用 base learner 在 observation cases 上跑行为轨迹，但 observation learner 是 clone，并冻结 semantic / risk / memory 更新。tutor 只观察 learner 的 public choices，用这些行为拟合 inverse posterior。

### 3.3 Inverse posterior

tutor 不直接知道 learner 的真实 profile，而是在一组离散 learner profile hypotheses 上做 posterior inference。

profile 主要控制：

```text
alpha_sem
alpha_risk
alpha_unc
beta_L
epsilon
risk_scale
```

posterior 可以理解为：

```text
P(profile | obs) proportional P(profile) * P(obs | profile)
```

planning 时用 posterior 的 top-mass profiles 做加权预测。

### 3.4 Hint planning

tutor 从候选池生成 non-answer hints，然后用 shadow learner rollout 预测 hint 后 learner 的 teach trajectory 和 eval proxy，再按效用选 hint。

### 3.5 Teach

真实 teach 条件里，hint 被作为一个 supervised example 先写入 learner：

```text
apply_hint:
  learner.scorer.incremental_study([(hint_words, hint_output)])
```

然后 learner 开始 option search。每次 learner 选错时，环境会 reveal 被选 wrong option 的 rendered output，并可把该 wrong option 从 active menu 中移除。

### 3.6 Eval

teach 结束后，系统用当前 learner 状态跑 direct eval：

```text
eval_exact
eval_cell
eval group metrics
```

eval_cell 是主要 transfer 指标。

---

## 4. Learner 机制

learner 是一个 option-selection agent，核心包括：

```text
semantic scorer
risk/danger head
attention model
episodic / negative memory
softmax pick policy
```

当前主实验基本不使用 risk，因此语义和 reveal update 是主要因素。

### 4.1 语义 scorer

CLS semantic scorer 学习：

```text
program words -> rendered output
```

给定 target output 和 option text，scorer 给每个 option 一个 semantic score：

```text
S_j = score_option(target_output, option_text_j, attention)
```

直觉上，option 预测输出越接近 target output，`S_j` 越高。

### 4.2 Learner pick utility

真实 learner policy 和 shadow learner 的核心形式一致。对 active option `j`：

```text
U_pick(j)
  = alpha_sem * S_j
  - alpha_risk * mu_d(j)
  - alpha_unc  * u_d(j)
  + memory_penalty(j)
  + optional_negative_penalty(j)
```

其中：

```text
S_j       semantic score
mu_d(j)  predicted danger / damage mean
u_d(j)   danger uncertainty
```

当前 no-risk setting 下，主要是：

```text
U_pick(j) ~= alpha_sem * S_j + memory terms
```

action probability 是 softmax with lapse：

```text
pi(j)
  = (1 - epsilon) * softmax(beta_L * U_pick(j))
    + epsilon / |A|
```

`epsilon` 使 learner 有少量随机 lapse。

### 4.3 Wrong reveal 后发生什么

当 learner 选错 option `a`：

```text
env reveals rendered_output(a)
```

也就是给 learner 看到：

```text
wrong option text a -> its true rendered output
```

在默认 `reveal_learning_mode="cortex_em"` 下，这个 wrong reveal 会触发 semantic update，相当于 learner 学到了这个 wrong option 自己的 input/output mapping。它不是 correct answer 的 mapping，但可能帮助 learner理解 operator 或排除相似结构。

当前 teach 统计里：

```text
wrong_before_correct = correct 前选错次数
teach_updates = teach 阶段实际 semantic updates 数
```

因此如果一个 tutor 让 learner 很快做对，learner 会看到更少 wrong reveals，也会获得更少 teach-time update。这正是 fast-vs-transfer 机制线关心的问题。

### 4.4 Correct pick 后

正确 pick 后 query 结束，并可按 `correct_pick_learning_mode` 做 positive update。当前主要分析集中在 hint 和 wrong reveal 对 search/eval 的影响。

---

## 5. Tutor 机制

tutor 是 inverse tutor，不是 oracle tutor。它先拟合 learner posterior，然后用 shadow learner 预测每个候选 hint 的影响。

### 5.1 Hint 候选池

主线用：

```text
free
operator_probe
```

机制线 frontier 用更大的 pool：

```text
free
operator_probe
answer_neighbor_nonanswer
target_neighborhood_robust_filtered
```

各 family 含义：

| family | 含义 |
|---|---|
| `free` | 从 task example pools 里抽普通 non-answer example |
| `operator_probe` | 和 teach example 共享 operator 的 probe example，更偏算子泛化 |
| `target_neighborhood_robust_filtered` | 从 teach example 附近生成结构相似但非答案的邻近 example，并做 rank/robust 过滤 |
| `answer_neighbor_nonanswer` | 基于 correct program 做 non-answer 近邻，不直接泄露 correct words/output |

所有正式 deployable hints 都应满足：

```text
hint != correct answer
hint_output != target_output
```

`menu_correct_ceiling` 只作为 oracle/ceiling，不进入主比较。

### 5.2 Planner cascade

当前 planner 是 cascade：

```text
candidate generation
  -> prefilter
  -> proxy rollout
  -> refine
  -> utility / gate / abstain
```

#### Prefilter

用 score table 快速算：

```text
initial_correct_prob
initial_correct_rank
initial_correct_margin
prefilter_score
```

#### Proxy rollout

用 score-table MC rollout 估计：

```text
P(tau <= T)
P(tau <= 2)
pred_p_tau
safe_wrong_mean
wrong_before_correct_mean
collapse features
```

`P(tau<=2)` 有一部分使用 first-reveal cached dynamic 近似，不再只依赖静态 initial rank。

#### Refine

当前常用：

```text
refine_update_mode = first_reveal_cached_cls
```

它对最可能的 first wrong reveal 做 cached semantic update，用于更真实地预测 post-reveal dynamics，同时避免 full CLS rollout 太慢。

#### Transfer eval proxy

机制线里接入了：

```text
transfer_eval_proxy_mode = static_subset
```

即对每个候选 hint 后的 shadow learner，在一个小 eval subset 上直接估计：

```text
EvalCellProxy(h)
```

这让 transfer utility 不再面对全 0 的 eval proxy。

---

## 6. 主线效用：`advantage_delta`

主线 tutor 用于证明 tutor 有用。

实际选择基于相对 no-tutor bonus baseline 的 delta：

```text
Delta U(h) = U_adv(h) - U_adv(no_tutor_T+H)
```

其中：

```text
U_adv(h)
  = lambda_success * P(tau <= T)
  + lambda_eval_cell * EvalCell(h)
  + lambda_soft_tau * R_tau(h)
  + lambda_exposure * E[safe_wrong_before_correct]
  - lambda_fail * P(tau > T)
  - lambda_collapse * C_collapse(h)
```

当前主要权重：

```text
lambda_success   = 3
lambda_fail      = 8
lambda_eval_cell = 2
lambda_soft_tau  = 2
lambda_exposure  = 0.5
lambda_collapse  = 3
```

因为：

```text
P(fail) = 1 - P(success)
```

所以在 delta utility 中：

```text
lambda_success * P(success) - lambda_fail * P(fail)
```

等价于：

```text
(lambda_success + lambda_fail) * Delta P(success)
```

也就是当前主线的有效 success 权重大约是：

```text
lambda_success_effective = 11
```

### Soft tau reward

```text
R_tau(h)
  = sum_t P(tau=t) * exp(-(t - c)^2 / (2 sigma^2))
```

主线里：

```text
c = 4.5
sigma = 1.5
```

它鼓励不要太早也不要太晚成功。

### Collapse penalty

`C_collapse` 惩罚 wrong reveal 后 correct probability/rank 预测崩掉的候选。它的目标不是直接提高 success，而是避免 planner 选择那种“第一步看起来好，但 reveal 后 learner posterior 崩掉”的 hint。

---

## 7. 机制线效用：normalized eta-mix

机制线用于展示同一个 tutor 框架在不同效用偏好下的行为分化。

同一个 tutor 框架不变：

```text
same inverse posterior
same candidate pool
same planner cascade
same teach/eval protocol
```

只改 utility：

```text
U_eta(h)
  = eta * z(U_fast(h))
  + (1 - eta) * z(U_transfer(h))
```

其中 z-score 在同一个 seed 的候选池内计算：

```text
z(U) = (U - mean_candidates(U)) / std_candidates(U)
```

归一化是必要的，因为 `U_fast` 和 `U_transfer` 原始尺度不同。未归一化时，`eta=0.5` 不一定是真正中点。

### Fast component

```text
U_fast(h)
  = 8  * P(tau <= T)
  + 10 * P(tau <= 2)
  - 4  * E[wrong_before_correct]
  + 2  * initial_correct_margin
  - 0  * C_collapse
```

含义：

```text
P(tau <= T)             有限尝试内做对
P(tau <= 2)             更快做对
wrong_before_correct    correct 前看了多少 wrong reveal，越少越好
initial_correct_margin  correct option 相对最强错误项的初始概率领先幅度
```

因此 `eta` 高时，tutor 更偏：

```text
teach success
early success
less wrong reveal
```

### Transfer component

```text
U_transfer(h)
  = 2 * P(tau <= T)
  + 6 * EvalCellProxy(h)
  + 4 * P(3 <= tau <= T)
  + 2 * E[safe_wrong_exposure]
  - 4 * P(tau < 3)
  - 3 * C_collapse(h)
```

含义：

```text
EvalCellProxy        hint 后小 eval subset 的 cell accuracy proxy
P(3 <= tau <= T)     教学窗口内成功
safe_wrong_exposure  多看一些非危险 wrong reveal，可能帮助 transfer
P(tau < 3)           惩罚太早成功
C_collapse           避免 post-reveal collapse
```

### Transfer gate

transfer 侧不用 search/band gate，而用 eval-delta gate：

```text
use hint if:
  predicted_eval_delta >= 0.005
  and predicted_success >= max(0.15, baseline_success - 0.05)
```

这样 transfer mode 不再因为 `no_band_candidate` / `early_only_candidates` 被 search gate 卡死。

---

## 8. 主线结果摘要

来自：

```text
utility_ablation_search_main_free_operator_20seed_summary.md
```

主线 baseline equivalent：

| condition | success | eval cell |
|---|---:|---:|
| `no_tutor_T` | 0.20 | - |
| `no_tutor_T+H` | 0.25 | 0.2808 approx |
| `random_hard_hint_T` | 0.33 | 0.2789 |
| `random_same_pool` | 0.27 | 0.2807 |
| `tutor` | 0.55 | 0.2996 |
| `oracle` | 0.70 | 0.3493 |

关键 delta：

```text
Delta success vs no_tutor_T+H      = +0.30
Delta success vs random_same_pool  = +0.28
Delta eval cell vs no_tutor_T+H    = +0.0188
Delta eval cell vs random_same_pool= +0.0189
tutor band success                 = 0.45
early-no-transfer                  = 0.00
```

这支持主 claim：

```text
在 rank-stratified hard option-search setting 下，
one-shot tutor selected hint 显著提高 bounded teach success，
并且优于 no tutor bonus 和 random same-pool hint。
```

### Utility ablation 的要点

`utility_no_success` 使 tutor success 从 `0.55` 掉到 `0.20`，说明 success 项是核心项。

`utility_failprob_equiv_check` 与 baseline 等价，支持：

```text
fail_prob 可并入 success 项
```

`eval/collapse/exposure/soft_tau` 在主 search benchmark 中更多是 shaping，不是主驱动力。

---

## 9. Fast / Transfer frontier 结果摘要

来自：

```text
fast_transfer_frontier_norm_eta_08_10_16seed_summary.md
fast_transfer_frontier_norm_eta_05_07_16seed_summary.md
```

当前 normalized eta-mix 16-seed 结果：

| eta | success | Delta success vs no tutor | Delta success vs random | eval all | Delta eval vs no tutor | Delta eval vs random | wrong-before-correct | teach updates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.5000 | +0.1875 | +0.1500 | 0.3141 | +0.0199 | +0.0208 | 3.5625 | 4.0625 |
| 0.9 | 0.5000 | +0.1875 | +0.1500 | 0.3171 | +0.0228 | +0.0238 | 3.5625 | 4.0625 |
| 0.8 | 0.5000 | +0.1875 | +0.1500 | 0.3179 | +0.0236 | +0.0246 | 3.5625 | 4.0625 |
| 0.7 | 0.4375 | +0.1250 | +0.0875 | 0.3297 | +0.0355 | +0.0364 | 3.6875 | 4.1250 |
| 0.6 | 0.3750 | +0.0625 | +0.0250 | 0.3272 | +0.0330 | +0.0339 | 3.8125 | 4.1875 |
| 0.5 | 0.3750 | +0.0625 | +0.0250 | 0.3034 | +0.0091 | +0.0101 | 3.7500 | 4.1250 |

解释：

```text
eta high:
  search/teach success stronger

eta around 0.7:
  eval highest, but teach success lower

eta too low:
  transfer-side candidate ranking hits a plateau / degrades
```

最适合展示的对比：

```text
eta=1.0:
  success = 0.50
  eval all = 0.3141

eta=0.7:
  success = 0.4375
  eval all = 0.3297
```

这说明：

```text
同一个 tutor 框架，通过 normalized eta utility，
可以从 fast/search 风格推向更 eval/transfer 风格。
```

但这条机制线仍不是最终主 claim，因为它还没有完全单调 frontier，且 `eta=0.5` 表现较差。

---

## 10. 当前报告时建议的表述

### 主 claim

```text
In a rank-stratified hard option-search setting, a one-shot inverse tutor
improves bounded teach success over both no-tutor with one extra attempt
and random same-pool hints.
```

中文：

```text
在 rank-stratified hard option-search 设置下，one-shot inverse tutor
通过有意选择 hint，相比 no tutor 多一次尝试和 random same-pool hint，
都能显著提高有限尝试内的 teach success。
```

### 机制 claim

```text
Using the same tutor and candidate pool, normalized eta-mix utility can
shift behavior from fast/search-oriented hinting toward transfer-oriented
hinting.
```

中文：

```text
在同一个 tutor 框架和同一个候选池下，normalized eta-mix utility
可以改变 tutor 的教学风格：高 eta 更偏 teach/search success，
较低 eta 更偏 eval transfer。
```

### Caveat

```text
The eta frontier is non-monotonic. Eta=0.7 gives the best eval in the current
run, but lower eta does not keep improving eval. This indicates transfer-side
candidate ranking/proxy is improved but not final.
```

中文：

```text
eta frontier 不是严格单调。当前 eta=0.7 的 eval 最好，但继续降低 eta
不会继续提高 eval，说明 transfer-side eval proxy 和候选排序已有进展，
但还不是最终版本。
```

---

## 11. 当前代码中最重要的文件

| 文件 | 作用 |
|---|---|
| `protocol.py` | 实验准备流程：prelearn / obs / inverse posterior / teach case / eval |
| `learner_runner.py` | 真实 learner condition 执行、hint application、teach/eval metrics |
| `menu_builder.py` | context、prelearn、obs、teach menu、eval items |
| `hint_space.py` | hint candidate families |
| `hint_planner.py` | planner cascade、utility selection、eval proxy gate |
| `metrics.py` | tutor utility formulas 和 gates |
| `rollout.py` | score-table rollout、first-reveal cache、dynamic metrics、eval proxy |
| `baselines.py` | no tutor / random hard / random same-pool controls |
| `run_fast_transfer_frontier.py` | normalized eta-mix frontier diagnostic runner |

---

## 12. 最终建议

PPT 中建议按如下顺序讲：

1. 场景：learner few-shot 学 grammar，然后从 hard menu 里找 correct option。
2. 问题：one-shot tutor 能否选一个有用 hint？
3. 为什么 rank-stratified：避免 no tutor ceiling。
4. 主结果：`advantage_delta + free/operator_probe` 明显提升 bounded teach success。
5. Ablation：success 项是核心，fail 可并入 success，其它项是 shaping。
6. 机制线：normalized eta-mix 显示 fast/search 与 transfer 之间的效用控制雏形。
7. Future work：更好的 eval proxy、candidate reranking、risk extension。

