# Antigravity Next-Step Guide For Maze

日期：2026-04-20

这份文档是给 Antigravity 的下一阶段实现指南。它不是单纯的 brainstorming，也不是只记录当前结果；它的目标是把：

- 当前代码已经实现的机制
- 当前实验已经暴露出来的真实问题
- 下一步应修改哪些模块、哪些模块暂时不要动
- 需要补哪些 diagnostics、tests、experiments
- 应该如何尽量减少超参数和机制冗余

整理成一份可以直接照着实现、验证、迭代的规范。

这份指南默认以当前主线代码为准，主要对应这些模块：

- `risky_maze/env/fixed_loader.py`
- `risky_maze/env/objectives.py`
- `risky_maze/env/pomdp_episode.py`
- `risky_maze/learner/objective_agent.py`
- `risky_maze/tutor/candidates.py`
- `risky_maze/tutor/baselines.py`
- `risky_maze/tutor/rollout.py`
- `risky_maze/tutor/inverse_planner.py`
- `risky_maze/tutor/diagnostics.py`
- `risky_maze/runner/fixed_episode_runner.py`
- `risky_maze/runner/fixed_metrics.py`
- `risky_maze/runner/fixed_block_runner.py`
- `risky_maze/experiments/run_fixed_maze.py`
- `risky_maze/experiments/run_phase12_batch.py`
- `risky_maze/experiments/run_d4_fix_comparison.py`

## 1. 当前阶段判断

当前 maze 系统已经进入真正有研究价值的阶段。现在的瓶颈已经不是“fixed-map POMDP 能不能跑”，而是：

```text
1. fixed-map POMDP 已经能跑；
2. baselines 已经能区分 teach safety / immortal exploration；
3. inverse_plan_full 已经真实改变 tutor 行为；
4. 但当前结果还不能证明“完整 inverse planning 的安全探索优于简单策略”；
5. 主要瓶颈是：
   - utility 标度
   - 任务切片饱和
   - warning-only 退化
   - waypoint over-help
   - map memory / risk concept 的归因没有拆清
```

因此，下一步不应该继续只修 runtime，也不应该立刻做更复杂的大框架，而应该围绕下面三个问题收敛：

- `WAIT / WARNING / WAYPOINT` 的价值函数是否表达正确
- teach 中的“探索收益”到底来自 map memory 还是 risk concept
- full tutor 是否真的在允许安全、可复用、有信息量的探索，而不是在过度帮助

## 2. 代码现状与模块判断

### 2.1 当前不再是 blocker 的模块

这些模块已经进入“可以跑实验”的状态，原则上不做大重写，只做局部增量修改：

- `risky_maze/env/fixed_loader.py`
  - fixed spec loading 已稳定
  - `FixedRuntimeLayout` 已支持 `cell_features`
  - 当前主要保留，不做大改
- `risky_maze/env/pomdp_episode.py`
  - fixed-map POMDP runtime、teach/eval phase、baseline mode 已接主线
  - 当前主要做 instrumentation 和 config 开关，不做结构性重写
- `risky_maze/runner/fixed_block_runner.py`
  - teach/eval block 流程已稳定
  - 当前应主要补新 eval split 和 ablation mode
- `risky_maze/experiments/run_fixed_maze.py`
  - CSV/README/config 输出链已稳定
  - 当前应主要补新 condition、new-map probe、paired comparison 脚本

### 2.2 当前必须修改的模块

这些模块是下一阶段的核心：

- `risky_maze/learner/objective_agent.py`
  - 需要修改 `OnlineGaussianRiskBelief.warning_update`
  - 需要为 ablation 和 probe 提供更细粒度开关
  - 需要让 learner 在 warning 后的 replanning 更可观测
- `risky_maze/tutor/rollout.py`
  - 需要加入 safety shield / catastrophe constraint
  - 需要让 warning 的 effect 真正通过 replanning 进入 rollout value
  - 需要拆出更清晰的 Q components 和 diagnostics
- `risky_maze/tutor/inverse_planner.py`
  - 需要从“纯 weighted Q”向“安全约束 + VOI proxy”收敛
  - 需要显式支持 `warning_only_current / safety_shield / replanning_rollout / constrained_voi`
- `risky_maze/tutor/candidates.py`
  - 需要对 WAYPOINT 做 bandwidth control
  - 需要区分 frontier waypoint / known bottleneck waypoint / hidden oracle waypoint
- `risky_maze/runner/fixed_episode_runner.py`
  - 需要补 post-warning / post-waypoint diagnostics
  - 需要补 preventable death / warning actionability 等 per-step 记录
- `risky_maze/runner/fixed_metrics.py`
  - 需要补 attribution metrics、wait quality metrics、waypoint leakage metrics、hard risk probes

### 2.3 当前建议降级或只保留为 debug 的模块/条件

- `risk_threshold_warn`
  - 当前结果是 warnings 很多但 deaths 也高
  - 暂时不要把它当正式主 baseline
  - 保留为 debug baseline，等 timing / actionability 修完再恢复主基线地位
- `T01_NW_key_gem_NE_exit -> E01_WestGarden_to_NE_exit`
  - 当前适合作为 smoke / sanity slice
  - 不适合作为正式机制比较的主 slice，因为 `eval_success` 已饱和
- 当前过多依赖 reward weights 的 full tutor utility
  - 建议逐步改成 safety-constrained VOI，而不是继续堆更多 penalty weights

### 2.4 当前不建议优先修改的模块

这些模块目前不是首要矛盾：

- `risky_maze/env/layout.py`
- `risky_maze/env/episode.py`
- `risky_maze/runner/episode_runner.py`
- `risky_maze/runner/block_runner.py`
- `risky_maze/core/pathing.py`
- 旧 random-map smoke 原型

原因：

- 当前主要问题不是 pathing 底层错误
- 也不是 random maze runtime 缺失
- 而是 fixed-map inverse tutor 的 decision semantics 和 attribution 不够清晰

## 3. 当前结果说明了什么

### 3.1 Fixed runtime 已经不是问题

当前系统已经跨过：

```text
spec -> fixed_loader -> POMDP env -> learner/tutor -> runner -> csv
```

因此后续结论不应再写成“先把 runtime 做好”，而应该写成：

```text
当前 runtime 足够支持对 tutor strategy、exploration value、assist leakage 的研究比较。
```

### 3.2 当前已经说明“灾难成本压制探索”

已有结果表明：

- `no_tutor_mortal` 的 teach success / map coverage 明显差
- `no_tutor_immortal_warnlike` 和 `no_tutor_immortal_no_timeout` 的 teach coverage 更高、teach success 更高

这说明：

```text
如果移除死亡/超时等灾难成本，learner 会探索更多，而且这种探索具有潜在的后续价值。
```

这正是当前 maze 场景优于 `cls_color_selection` 的关键地方。

### 3.3 当前 `eval_success` 已饱和

在很多 slice 上：

```text
eval_success = 1.0
```

因此后面不能再主要看：

- `eval_success_rate`

而要主要看：

- `eval_regret_to_oracle_safe_path`
- `eval_mean_steps`
- `eval_mean_damage`
- `map_reuse_eval`
- `useful_exploration_rate`
- `risk_auc` on harder probes
- `assist_leakage`

## 4. 当前主要机制问题

### 4.1 `inverse_plan_warn_only` 退化成 WAIT

这是当前最大的机制 blocker。当前 warning-only tutor 的表现说明，它没有正确表达：

- WAIT 的风险
- WARNING 的 replanning 价值
- WARNING 的 learning value

最可能的问题源：

- utility 标度不统一
- warning update 没有转化成 replanning
- rollout 太短太 myopic
- 没有 safety constraint，只是普通 weighted Q

### 4.2 `risk_threshold_warn` 结果异常

它当前表现为：

- warnings 多
- deaths 仍然高

这更像：

- warning timing 错
- warning set 不 actionable
- warning 后 learner 没有真正改变 path

而不是“阈值 warning 理论上不行”。

### 4.3 D4 fix 已经真实改变 full tutor 行为

当前结果已经表明：

- `inverse_plan_warn_only` 对 D4 不敏感，更多只是内部 Q 微动
- `inverse_plan_full` 对 D4 已经开始有行为变化

但变化方向依赖任务结构：

- `T04 -> E05` 更像 over-waypoint diagnostic
- `T08 -> E07` 更像 positive long-horizon slice

这说明系统已经进入“真实策略问题”阶段，而不再只是 wiring 问题。

## 5. 下一阶段的数学重构方向

目标不是再加更多超参数，而是把当前 weighted reward tutor 收敛到：

```text
Safety constraint
+ no-progress constraint
+ Bayesian value of information / expected regret reduction
```

### 5.1 Learner belief 分解

定义 learner belief 为：

$$
b_t = \Big(q_t(M),\; q_t(Z),\; q_t(\phi),\; q_t(\psi)\Big)
$$

其中：

- $M$：map memory / known topology
- $Z$：cell latent danger labels
- $\phi$：risk concept parameters，例如 Gaussian prototypes
- $\psi$：learner profile / planning parameters

在当前代码中，近似对应为：

- `SimpleMapMemory` in `risky_maze/learner/objective_agent.py`
- `OnlineGaussianRiskBelief` in `risky_maze/learner/objective_agent.py`
- tutor 侧的 profile posterior in `risky_maze/tutor/inverse_planner.py`

### 5.2 Warning set Bayesian update

对 warning set $S$ 中每个 cell 的 feature $x_i$，先有：

$$
p_i = P(z_i=\text{danger}\mid x_i)
$$

WARNING 的语义是：

$$
F_S = \left[\exists i\in S: z_i=\text{danger}\right]
$$

集合条件化：

$$
P(Z_S \mid X_S, F_S)
\propto
\mathbb{1}\left[\exists i\in S: z_i=\text{danger}\right]
\prod_{i\in S}P(z_i\mid x_i)
$$

二元近似的边缘更新可写成：

$$
p_i'
\approx
\frac{p_i}{1-\prod_{j\in S}(1-p_j)}
$$

当前代码对应位置：

- `OnlineGaussianRiskBelief.warning_update()` in `risky_maze/learner/objective_agent.py`

问题不是公式完全错，而是 warning evidence 现在过强、且 effective sample size 固定。

建议改成：

$$
\eta_w(S)
=
\eta_0
\cdot
\frac{\mathrm{KL}(q'(Z_S)\|q(Z_S))}
{\log(1+|S|)+\epsilon}
$$

然后把 warning 的软更新写成：

$$
\kappa_k' = \kappa_k + \eta_w \sum_{i\in S} w_{ik}
$$

$$
\mu_k'
=
\mu_k
+
\frac{\eta_w W_k}{\kappa_k'}
(\bar{x}_k-\mu_k)
$$

其中：

$$
w_{ik}=P(z_i=k\mid X_S,F_S), \qquad
W_k=\sum_{i\in S}w_{ik}
$$

实现建议：

- 保留当前 binary belief 版本作为 `warning_update_literal`
- 新增 `warning_update_effective_sample`
- 用 config 开关对照比较，而不是一次性替换

### 5.3 Tutor 的 profile posterior

定义 tutor 对 learner profile 的后验：

$$
q_T(\psi\mid h_t)
\propto
q_T(\psi\mid h_{t-1})
\cdot
\pi_\psi(a_t^L \mid b_t^L)
$$

当前代码对应：

- `InversePlanningTutor._maybe_update_profile_belief()`
- `first_action_probability()`
- `predict_topk()`

这个部分当前可以保留，不建议重写。下一步主要增加 diagnostics：

- `profile_entropy`
- `profile_mass_shift`
- `action_prob_of_observed_action`

### 5.4 Tutor action 的 safety-constrained VOI

定义 tutor action：

$$
a\in\{\text{WAIT},\text{WARNING}(S),\text{WAYPOINT}(g)\}
$$

定义 catastrophe event：

$$
C_{\text{cat}}(\tau)
=
\mathbb{1}\left[
\text{death}
\lor
\text{HP}\le 0
\lor
\text{damage}\ge d_{\text{cat}}
\right]
$$

先做 safety filter：

$$
\mathcal{A}_{\text{safe}}
=
\left\{
a :
P(C_{\text{cat}}=1\mid a,h_t)\le \delta_{\text{safe}}
\right\}
$$

再在 safe actions 中比较学习价值：

$$
a^*
=
\arg\max_{a\in\mathcal{A}_{\text{safe}}}
\Big[
\mathrm{VOI}(a)
-
C_{\text{teach}}(a)
-
C_{\text{assist}}(a)
-
C_{\text{bore}}(a)
\Big]
$$

其中：

$$
\mathrm{VOI}(a)
=
\mathbb{E}_{o\sim P(o\mid a,b_t)}
\left[
\hat{C}_{\text{eval}}(b_t)
-
\hat{C}_{\text{eval}}(b_{t+1}^{a,o})
\right]
$$

实现建议：

- 不要一次性把当前 `TutorUtilityWeights` 删除
- 先在 `CounterfactualRolloutEvaluator` 里新增两层逻辑：
  - `catastrophe_probability(action, ...)`
  - `voi_proxy(action, ...)`
- 再在 `InversePlanningTutor._select_with_guardrails()` 里改为：
  - 先 safety filtering
  - 再做 utility compare

当前对应代码：

- `risky_maze/tutor/rollout.py`
- `risky_maze/tutor/inverse_planner.py`

### 5.5 WAYPOINT 的 assist leakage

当前 `assist_leakage` 主要还是简单 scalar，建议改成：

$$
C_{\text{assist}}(\text{WAYPOINT}(g))
=
\lambda_{\text{wp}}
\cdot
L(g)
\cdot
G(g)
$$

其中可见性泄露：

$$
L(g)=
\begin{cases}
0.25, & g\text{ is visible frontier}\\
0.50, & g\text{ is known but not visible}\\
1.00, & g\text{ is hidden oracle-only}
\end{cases}
$$

进度赠送量：

$$
G(g)=
\frac{
\max(0,\; d(s,o)-d(g,o))
}{
d(s,o)+\epsilon
}
$$

这里：

- $s$：current position
- $o$：current objective
- $g$：waypoint

实现建议：

- `generate_waypoint_candidates()` in `risky_maze/tutor/candidates.py`
  - 现在已经在做 known/visible 限制
  - 下一步把 `base_assist_leakage` 拆成：
    - `waypoint_visibility_leak`
    - `waypoint_progress_gift`
- `fixed_episode_runner.py`
  - 记录 per-decision `waypoint_visibility_leak`
  - 记录 `waypoint_progress_gift`

### 5.6 WAIT 的 pedagogical value

WAIT 不是“什么都不做”，它应该显式包含有用探索价值：

$$
V_{\text{WAIT}}
=
\mathbb{E}
\left[
\Delta I_{\text{map}}
+
\Delta I_{\text{risk}}
-
C_{\text{cat}}
-
C_{\text{bore}}
\right]
$$

其中：

$$
\Delta I_{\text{map}}
=
\sum_{c\in \text{newly observed}} P_{\text{reuse}}(c)
$$

$$
P_{\text{reuse}}(c)
=
\frac{1}{|\mathcal{E}|}
\sum_{e\in\mathcal{E}}
\mathbb{1}[c\in \mathrm{OraclePath}(e)]
$$

实现建议：

- 先不要做全量 Bayesian future task expectation
- 第一版在 rollout 里用 proxy：
  - newly observed cells count
  - whether those cells lie on current eval objective oracle route
- 第二版再扩展到全 task suite reuse estimate

## 6. 模块级实现建议

### 6.1 `risky_maze/learner/objective_agent.py`

#### 必改

- `OnlineGaussianRiskBelief.warning_update`
  - 新增 effective sample size 版本
  - 增加返回 diagnostics：
    - `warning_set_size`
    - `warning_eta`
    - `warning_kl`
    - `warning_mean_abs_delta`
    - `warning_sum_delta`
- `ObjectiveAwareLearner.apply_warning`
  - 记录 warning 后 planner 是否改变
  - 可通过比较 `last_plan` 前后 prefix 完成

#### 增加开关

- `warning_update_mode`
  - `literal`
  - `effective_sample`
- `warning_eta0`
- `ablate_risk_update`
- `ablate_warning_update`

#### 不建议现在做的事

- 不要把 `OnlineGaussianRiskBelief` 立刻升级成复杂 mixture model
- 不要在这一轮引入 neural learner

### 6.2 `risky_maze/tutor/rollout.py`

#### 必改

- 新增 `catastrophe_probability()` 或等价逻辑
- 新增 warning effect 后的 replanning diagnostics
- `_simulate_particle()` 里记录：
  - `path_before_warning`
  - `path_after_warning`
  - `predicted_damage_before`
  - `predicted_damage_after`
  - `warning_actionability`
- 把当前 `TutorActionValue` 拆得更明确：
  - `expected_catastrophe`
  - `expected_boredom`
  - `expected_map_gain`
  - `expected_risk_ig`
  - `expected_assist_cost`

#### 方案建议

方案 A：最小改动

- 保留当前 weighted Q
- 只加 safety shield
- 只加 warning replanning diagnostics

方案 B：中等改动

- 保留 weighted Q 外壳
- 把 WAIT/WARNING/WAYPOINT 的分项明确分解
- 将 `risk_ig` 与 `map_gain` 规范化到相近量纲

方案 C：主推方案

- 先 safety filter
- 再用 VOI proxy 比较
- 只保留极少数环境级 cost constants

建议先做 A 和 B，再和 C 比较，不要一开始就把旧逻辑全部删掉。

### 6.3 `risky_maze/tutor/inverse_planner.py`

#### 必改

- `TutorConfig` 新增：
  - `tutor_safety_shield_enabled`
  - `tutor_catastrophe_threshold`
  - `tutor_warning_requires_replan`
  - `tutor_waypoint_budget`
  - `tutor_waypoint_frontier_only`
  - `tutor_decision_mode`
- `_select_with_guardrails()`
  - 从“冷却 + min advantage”升级到：
    - safety filtering
    - waypoint budget filtering
    - then utility compare

#### 增加模式

- `warning_only_current`
- `warning_only_safety_shield`
- `warning_only_replanning_rollout`
- `warning_only_constrained_voi`
- `full_current`
- `full_bandwidth_controlled`
- `full_constrained_voi`

#### 保留

- `profile_belief` 的有限 profile 形式先保留
- 当前 `_build_profile_path_cache()` 的缓存结构先保留

### 6.4 `risky_maze/tutor/candidates.py`

#### 必改

- `generate_warning_candidates()`
  - 记录 candidate 是否覆盖真正 imminent trap
  - 记录 warning set size / path probability / predicted danger mass
- `generate_waypoint_candidates()`
  - 显式区分：
    - `frontier_waypoint`
    - `known_bottleneck_waypoint`
    - `known_safe_detour_waypoint`
  - 禁止 hidden oracle waypoint 进入主线 full tutor

#### 建议新增 diagnostics

- `waypoint_visibility_leak`
- `waypoint_progress_gift`
- `waypoint_kind`

### 6.5 `risky_maze/tutor/baselines.py`

#### 必改

- `RiskThresholdWarnTutor`
  - 标注为 debug baseline
  - 增加 diagnostics：
    - `true_prefix_risk`
    - `warning_was_actionable`
    - `death_followed_warning`
- `AlwaysWaypointTutor`
  - 保留为 leakage ceiling
  - 不作为 pedagogical main baseline

#### 建议新增

- `WaypointOnlyTutor`
- `AlwaysWarnStrictTutor`
  - 如果 need，可把 catastrophic prefix detection 单独做一个强安全 ceiling

### 6.6 `risky_maze/runner/fixed_episode_runner.py`

#### 必改

补这些 per-step / per-decision 记录：

- `before_warning_path_has_trap`
- `after_warning_path_has_trap`
- `post_warning_action_changed`
- `warning_voided_action`
- `warning_actionability`
- `warning_ig_pred_actual_gap`
- `waypoint_visibility_leak`
- `waypoint_progress_gift`
- `waypoint_aftereffect_new_cells`
- `waypoint_aftereffect_risk_ig`
- `wait_was_useful`
- `wait_was_bad`
- `preventable_death`

#### 建议

- 在 `_decision_row()` 中加入：
  - `predicted_catastrophe_wait`
  - `predicted_catastrophe_selected`
  - `safety_shield_triggered`

### 6.7 `risky_maze/runner/fixed_metrics.py`

#### 必改

`EpisodeMetrics` 和 `aggregate_block_metrics()` 需要新增：

- `warning_actionability_rate`
- `preventable_death_rate`
- `warning_ig_pred_actual_gap`
- `waypoint_progress_gift_mean`
- `waypoint_visibility_leak_mean`
- `useful_wait_rate`
- `bad_wait_rate`
- `coordinate_memory_gain`
- `risk_concept_gain`

#### 风险 probe

新增风险 probe 模式：

- `risk_eval_observed_cells_only`
- `risk_eval_unseen_cells_same_map`
- `risk_eval_new_map_same_prototypes`
- `risk_eval_near_boundary`

不要只保留当前全图、静态、易分的 `risk_auc`。

### 6.8 `risky_maze/experiments/*.py`

#### `run_fixed_maze.py`

需要支持更多 config overrides，并保留当前 CSV 输出。

#### `run_phase12_batch.py`

当前脚本应降级为：

- smoke + baseline feasibility batch

不要再把它当全部正式结果生成器。

#### `run_d4_fix_comparison.py`

应继续保留，因为 paired before/after 很有价值。

#### 建议新增

- `run_warn_debug_grid.py`
- `run_full_suite_baselines.py`
- `run_waypoint_ablation.py`
- `run_memory_risk_ablation.py`
- `run_risk_difficulty_sweep.py`

## 7. 需要新增的核心指标

### 7.1 Warning 相关

#### WarningActionability

定义：

$$
\text{WarningActionability}
=
P(
\text{post-warning planned path avoids true danger}
)
$$

需要记录：

- `before_warning_path_has_trap`
- `after_warning_path_has_trap`
- `post_warning_action_changed`

#### PreventableDeathRate

定义：

$$
\text{PreventableDeath}
=
\mathbb{1}
\left[
\text{death occurred}
\land
\exists S:
P(\text{death}\mid \text{WARNING}(S)) < P(\text{death}\mid \text{WAIT})
\right]
$$

#### WarningIGPredActualGap

定义：

$$
\Delta_{\text{IG}}
=
\widehat{IG}_{\text{rollout}}
-
IG_{\text{actual}}
$$

### 7.2 WAYPOINT 相关

#### WaypointProgressGift

$$
G(g)=
\frac{
d(s,o)-d(g,o)
}{
d(s,o)+\epsilon
}
$$

#### WaypointNoveltyLeak

建议离散分级：

- `0`: visible frontier
- `1`: known but not visible
- `2`: hidden / unknown-to-learner

主线 full tutor 应避免 `2`。

#### WaypointAfterEffect

记录 waypoint 后 $K$ 步：

- `new_cells_after_waypoint`
- `risk_ig_after_waypoint`
- `distance_progress_after_waypoint`
- `repeated_steps_after_waypoint`

### 7.3 WAIT 相关

#### UsefulWaitRate

$$
\text{UsefulWaitRate}
=
P(
\text{WAIT}
\land
\Delta I_{\text{map/risk}} > \epsilon
\land
\text{no catastrophe}
)
$$

#### BadWaitRate

$$
\text{BadWaitRate}
=
P(
\text{WAIT}
\land
[
\text{death}
\lor
\text{loop}
\lor
\text{no-info streak}
]
)
$$

### 7.4 Attribution 相关

#### CoordinateMemoryGain

same-map eval 中：

$$
\Delta C_{\text{map}}
=
\hat{C}_{\text{eval}}^{\text{no-map-memory}}
-
\hat{C}_{\text{eval}}^{\text{full}}
$$

#### RiskConceptGain

new-map same-risk eval 中：

$$
\Delta C_{\text{risk}}
=
\hat{C}_{\text{eval}}^{\text{no-risk-update}}
-
\hat{C}_{\text{eval}}^{\text{full}}
$$

## 8. 实验与验证计划

### Experiment 1：`inverse_plan_warn_only` debug grid

#### 目标

解决当前最大 blocker：

```text
为什么 rollout warning tutor 一直 WAIT
```

#### 任务切片

- `T01 -> E01`：smoke
- `T04 -> E05`：central diagnostic
- `T08 -> E07`：long-horizon diagnostic

#### 条件

- `always_warn`
- `inverse_warn`
- `inverse_plan_warn_only_current`
- `inverse_plan_warn_only + safety_shield`
- `inverse_plan_warn_only + replanning_rollout`
- `inverse_plan_warn_only + safety_shield + replanning_rollout`

#### 参数建议

先固定：

- `horizon=5`
- `top_k=3`
- `profile_count=5`
- `seeds=0..15`

#### 关注指标

- `TeachDeath`
- `TeachDamage`
- `WarningsPerEpisode`
- `WarningActionability`
- `PreventableDeathRate`
- `q_wait`
- `q_best_warning`
- `predicted_damage_wait`
- `predicted_damage_warning`
- `actual_damage_after_wait`
- `actual_damage_after_warning`
- `post_warning_action_changed`

#### 预期

- 加 `safety_shield` 后，teach death 应显著下降
- 加 `replanning_rollout` 后，warning Q 应提升，warning frequency 不应仍为 0

### Experiment 2：全 task suite Phase 1 baseline

#### 目标

确认：

- 探索收益是否稳定存在
- 收益来自 map memory 还是 risk learning
- 当前 `T01/E01` 是否太容易

#### 任务

- teach: `T01..T08`
- eval: `E01..E08`

#### 条件

- `no_tutor_mortal`
- `no_tutor_immortal_warnlike`
- `no_tutor_immortal_no_timeout`

#### seeds

- `0..31`

#### 关注指标

- `TeachSuccess`
- `TeachDeath`
- `TeachTimeout`
- `TeachDamage`
- `MapCoverageTeach`
- `RiskAUC_seen`
- `RiskAUC_unseen_same_map`
- `EvalRegret`
- `MapReuseEval`
- `UsefulExplorationRate`

#### 预期

- `immortal_no_timeout > immortal_warnlike > mortal`
  - 在 map coverage 和 eval regret 上应有清晰差异

### Experiment 3：D4 full tutor paired comparison 扩展

#### 目标

确认 D4 行为变化不是 8 seeds 的噪声。

#### 任务

- `T04 -> E05`
- `T08 -> E07`

#### 条件

- `inverse_plan_full_before_D4`
- `inverse_plan_full_after_D4`

#### seeds

- `0..31` 或 `0..63`

#### 方法

必须使用 paired seeds，并计算每个 seed 的差值：

- `Δeval_regret`
- `Δwarnings`
- `Δwaypoints`
- `Δwarning_IG`
- `Δassist_leakage`

#### 输出

- paired mean delta
- bootstrap 95% CI
- per-seed scatter

#### 预期

- `T08` 更可能是 positive slice
- `T04` 更可能暴露 over-waypoint

### Experiment 4：WAYPOINT 正式对照

#### 目标

判断 waypoint 是 teaching 还是 oracle navigation。

#### 条件

- `inverse_plan_warn_only_fixed`
- `waypoint_only`
- `always_waypoint`
- `inverse_plan_full`
- `inverse_plan_full + waypoint_budget`
- `inverse_plan_full + frontier_only_waypoint`

#### 指标

- `TeachSuccess`
- `TeachTimeout`
- `EvalRegret`
- `WaypointsPerEpisode`
- `AssistLeakage`
- `WaypointProgressGift`
- `WaypointNoveltyLeak`
- `MapGainAfterWaypoint`
- `RiskIGAfterWaypoint`
- `UsefulExplorationRate`

#### 预期

好的 full tutor 应满足：

- `TeachTimeout < warning_only`
- `AssistLeakage < always_waypoint`
- `EvalRegret <= always_waypoint` 或接近
- `UsefulExplorationRate >= always_waypoint`

### Experiment 5：map memory vs risk learning ablation

#### 目标

分清当前收益来源。

#### 条件

- `full learner`
- `no_map_memory_eval`
- `no_risk_update`
- `no_warning_update`
- `no_death_update`
- `map_memory_only`
- `risk_only_new_map`

#### 主要实现方式

- `no_map_memory_eval`
  - eval 时清空 `known_map` / `known_walkable`
  - 保留 risk belief
- `no_risk_update`
  - teach 阶段不更新 `OnlineGaussianRiskBelief`
- `no_warning_update`
  - warning 只影响 action，不更新 belief
- `risk_only_new_map`
  - 新地图，保留 risk prototypes

#### 指标

- `same_map_eval_regret`
- `new_map_same_risk_eval_regret`
- `risk_auc_unseen`
- `map_reuse_eval`
- `eval_damage`

### Experiment 6：risk difficulty sweep

#### 目标

解决 `risk_auc` 太完美的问题，找到 medium difficulty。

#### difficulty presets

- easy
  - `obs_noise=0.30`
  - `cluster_std=0.45`
- medium
  - `obs_noise=0.45`
  - `cluster_std=0.60`
- hard
  - `obs_noise=0.60`
  - `cluster_std=0.80`

#### 条件

- `no_tutor_mortal`
- `always_warn`
- `inverse_plan_warn_only_fixed`
- `inverse_plan_full`

#### 指标

- `RiskAUC_seen`
- `RiskAUC_unseen`
- `WarningIG`
- `EvalDamage`
- `EvalRegret`
- `TeachDeath`

#### 预期

- easy：no tutor 已经很好，tutor 空间小
- hard：warning 也学不动
- medium：最适合展示 tutor 权衡

## 9. Tests 与验收建议

### 9.1 Contract tests

建议新增：

- `tests/test_warning_actionability.py`
- `tests/test_waypoint_leakage.py`
- `tests/test_safety_shield.py`
- `tests/test_memory_risk_ablation.py`
- `tests/test_risk_probe_splits.py`

### 9.2 每个测试至少要覆盖

#### `test_safety_shield.py`

- WAIT catastrophic 且 WARNING safe 时，必须选 WARNING
- WAIT catastrophic 且 WARNING 不 safe、WAYPOINT safe 时，必须选 WAYPOINT
- eval phase 仍然强制 tutor-off

#### `test_warning_actionability.py`

- warning 后若 planner risk posterior 变化足够大，应触发 replanning
- replanning 后 path prefix 应不同于 warning 前 path prefix

#### `test_waypoint_leakage.py`

- hidden unknown coord 不得被 full tutor waypoint 主线选中
- frontier-only mode 只能选 visible/known frontier
- budget/cooldown 生效

#### `test_memory_risk_ablation.py`

- `no_map_memory_eval` 应清空 map memory 但保留 risk belief
- `no_risk_update` 应保留 map memory 但冻结 risk update

#### `test_risk_probe_splits.py`

- seen / unseen_same_map / new_map_same_risk / near_boundary 四类 probe 能独立构建

### 9.3 工程验收标准

完成下一阶段后，至少满足：

- `inverse_plan_warn_only` 不再在主诊断 slice 上长期全 WAIT
- `risk_threshold_warn` 的 warning actionability 不再异常低
- `inverse_plan_full` 有可解释的 `WAIT/WARNING/WAYPOINT` 分布
- same-map 和 new-map same-risk attribution 可以拆开报告
- `risk_auc` 不再在所有 setting 下接近 1.0

## 10. 多方案比较建议

不要只实现一个方案然后把它当最终答案。建议并行保留三条方案线：

### 方案 A：最小 patch 线

内容：

- safety shield
- warning replanning
- waypoint budget / cooldown

优点：

- 改动小
- 易于 debug
- 易于判断 wiring 是否正确

缺点：

- 仍然依赖现有 weighted reward

### 方案 B：结构化 weighted utility 线

内容：

- 保留 weighted Q
- 但拆出清晰分项：
  - catastrophe
  - damage
  - timeout
  - map gain
  - risk IG
  - assist
  - boredom

优点：

- 可解释性更高
- 可以保留当前大部分代码

缺点：

- 仍然会有不少权重

### 方案 C：constrained VOI 主线

内容：

- safe-action filtering
- VOI proxy
- minimal environment-level cost constants

优点：

- 最符合研究叙事
- 超参数最少
- 更稳健

缺点：

- 需要更多 diagnostics 支撑

建议执行方式：

- 先实现 A
- 再实现 B
- 最后实现 C
- 保留对照实验，不要一开始就删除旧逻辑

## 11. 超参数和机制冗余收敛原则

下一阶段必须尽量减少下面几类冗余：

- 用一堆 reward weights 同时表达 safety 和 assist
- 同时保留多个语义近似但不等价的 warning update
- 同时保留多个没有明确定位的 waypoint 类型
- 用单一 `risk_auc` 代表全部 risk learning 质量

建议最终保留的核心超参数尽量收敛到：

- `delta_safe`
- `warning_eta0`
- `waypoint_budget`
- `waypoint_cooldown`
- `rollout_horizon`
- `top_k_paths`
- `profile_count`
- 少量环境级 cost constants：
  - `c_damage`
  - `c_failure`

不建议长期保留太多手工 utility weights。

## 12. 推荐执行顺序

Antigravity 下一轮按这个顺序做：

1. 修 `inverse_plan_warn_only`
   - safety shield
   - warning replanning rollout
   - warning diagnostics
2. 把 `T01 -> E01` 降级为 smoke
   - 正式比较改用 `T04 -> E05` 和 `T08 -> E07`
3. 给 full tutor 加 waypoint bandwidth control
4. 改 warning update effective sample size
5. 做 risk probe 和 attribution ablation
6. 再做 full suite / difficulty sweep

## 13. 最终研究叙事建议

当前结果支持的叙事不是：

```text
inverse_plan_full 已经显著优于所有 baseline
```

当前更准确的叙事是：

```text
1. Maze POMDP 场景已经能显露出灾难成本压制探索；
2. 去掉灾难/timeout 后，learner 的 map coverage 和 map reuse 明显上升；
3. simple warning 能提供部分 safety，但 heuristic warning 与 always warning 在简单 slice 上难以区分；
4. full inverse tutor 已经能真实改变 WAIT/WARNING/WAYPOINT 行为；
5. 长程任务上已出现小幅正向信号，central maze 上已出现 over-help 信号；
6. 因此下一步不是盲目加复杂度，而是用 safety-constrained VOI 和 diagnostics 分解 tutor 行为。
```

如果这条线跑通，最终要证明的不是：

```text
exploration 本身有益
```

而是：

```text
tutor 通过 inverse planning 允许“安全、可复用、有信息量”的探索，
同时阻止 catastrophic risk 和 no-progress boredom。
```
