# pedagogical_ip 当前代码机制报告

本文档基于仓库当前代码与仓库内文档生成，目标是回答：

1. `pedagogical_ip` 当前主线场景是什么。
2. learner 当前到底在维护什么状态、如何观察、如何规划、如何学习。
3. tutor 当前到底在建模什么、如何做 intervention scoring。
4. 项目里冻结基线、主线 V2、历史归档之间是什么关系。
5. 当前代码的主要缺陷、冗余和研究风险是什么。

本文只依据以下代码和文档，不凭空补设定：

- `src/envs/lattice_v2_runner.py`
- `src/envs/scenario_families.py`
- `src/envs/dtmb_lattice.py`
- `src/envs/gtet_lattice.py`
- `src/envs/prs_session.py`
- `src/agents/feature_belief.py`
- `src/agents/observation_model.py`
- `src/agents/risk_model.py`
- `src/agents/cost_risk_model.py`
- `src/agents/planner_astar.py`
- `src/agents/belief_planning.py`
- `src/agents/warning_update.py`
- `src/teachers/robot_belief.py`
- `src/teachers/agent_predictor.py`
- `src/teachers/intervention_policy.py`
- `src/teachers/perceptual_model.py`
- `src/teachers/internalization_observer.py`
- `src/teachers/internalization_control_tutor_v4.py`
- `docs/architecture_5d_canonical.md`
- `docs/canonical_baseline_spec.md`
- `docs/DTMB_BENCHMARK_CONTRACT.md`
- `docs/regression_protocol.md`
- `configs/observer/a1_5d_kappa.yaml`
- `configs/tutor/bcictv4_2act.yaml`
- `scripts/eval_dtmb_main.py`
- `scripts/eval_prs_transfer.py`

## 1. 总结结论

当前 `pedagogical_ip` 不是单一的一条机制线，而是至少有两条需要明确区分的线：

1. 当前 V2 主线：
   - 入口是 `LatticeV2Runner`
   - learner 是 `FeatureBeliefMap + LatentCostRiskHead + belief-conditioned bounded planning`
   - tutor 是 `RobotBelief + AgentPredictor + InterventionPolicy`
   - 动作空间是 `{WAIT, WARN, UNLOCK, ITEM_DROP}`
   - 代表性场景是 `DTMB-L`、`GTET-L`、`PRS`

2. 冻结的 canonical internalization baseline：
   - observer 是 `A1MtObserver` / `A1MtObserverFrozen`
   - tutor 配置是 `BCICTv4(use_dose=False)` 的 2-act `{WAIT, WARN}`
   - 主要服务于内部化估计、κ̂ 宏观 bonus、稳定性回归
   - 文档是 `architecture_5d_canonical.md` 和 `canonical_baseline_spec.md`

3. 历史归档线：
   - `internalization_control_tutor_v4.py` 文件头已经明确标注为 archival
   - 它不是当前主线实验入口，但仍然是冻结基线文档的历史前身

最重要的结构判断是：

> 当前主线已经从“2-act internalization tutor”迁移到了“counterfactual intervention family tutor”，  
> 但仓库中仍保留了冻结基线文档和归档 tutor 代码，因此项目在概念上是“双中心”的。

这不是 bug，但它会显著增加理解和实验解释成本。

---

## 2. 当前主线系统总览

当前主线的 episode 语义源头是 `src/envs/lattice_v2_runner.py`。

每一步严格按以下顺序执行：

1. `observe(s)`：agent 观察当前局部 patch 的 4D feature。
2. `apply_tutor(s)`：tutor 可能执行 `WAIT / WARN / UNLOCK / ITEM_DROP`。
3. `plan_and_move(s)`：agent 基于 belief 重新规划，移动，结算 outcome，并更新在线模型。

也就是说，当前主线的 tutor 不是直接替 learner 决策，而是通过：

- 改变 belief-relevant evidence
- 改变 passability
- 改变 inventory
- 改变 warned extra cost

来间接改变 learner 的规划与结果。

这条主线的核心对象如下。

### 2.1 learner 侧

- `FeatureBeliefMap`
- `BayesianRiskHead` 或 `LatentCostRiskHead`
- `PlannerWeights`
- `plan_from_belief()` / `plan_with_alternatives_v2()`

### 2.2 tutor 侧

- `RobotBelief`
- `AgentPredictor`
- `InterventionPolicy`
- `PerceptualAccessState`

### 2.3 场景侧

- `baseline_v2`
- `DTMB-L`
- `GTET-L`
- `PRS` 会把多个 episode 组装成 session

---

## 3. 场景与任务设置

## 3.1 基础平台：Lattice V2

基础环境是格点图，但不是简单的 deterministic shortest path。每个 cell 同时有：

- 可通行性 `passable`
- 真 cost `true_cost`
- 真 risk `true_risk`
- 潜在 4D feature `z ∈ [0,1]^4`

learner 不能直接看到 `true_cost` 或 `true_risk`，只能通过 noisy feature observation 建立 belief。

项目当前大量代码都围绕“feature-as-latent”语义展开：

- 风险不是直接观察的 scalar
- learner 学的是 `feature -> cost/risk` 映射

## 3.2 Scenario registry

`src/envs/scenario_families.py` 当前暴露统一入口 `generate_scenario(...)`。

仓库内实际主打的 family 包括：

- `baseline_v2`
- `deep_tree_mixed_bottleneck_lattice`
- `goal_preference_temptation_entanglement_lattice`

此外 PRS 会把这些 family 封装成多 block session。

## 3.3 DTMB-L：当前 mixed-bottleneck 主 benchmark

`src/envs/dtmb_lattice.py` 将 episode 划成三段 bottleneck：

1. Stage 1：epistemic ambiguity
2. Stage 2：structural pressure
3. Stage 3：outcome bottleneck

对应 intervention leverage 设计为：

- Stage 1：`WARN / WAIT`
- Stage 2：`UNLOCK`
- Stage 3：`ITEM_DROP`

`docs/DTMB_BENCHMARK_CONTRACT.md` 明确规定了：

- medium 应主要检验 `WARN`
- hard 应主要检验 `UNLOCK` 和 `ITEM_DROP`

所以 DTMB-L 不是一般的 survival map，而是“多 intervention lever 切换” benchmark。

## 3.4 GTET-L：当前 latent ambiguity benchmark

`src/envs/gtet_lattice.py` 的核心不是更大地图，而是更深的 latent entanglement：

> 同一个 prefix behavior 可以同时被多个 `(goal, preference, temptation)` 组合解释。

它的设计目标是迫使系统维护更长时间的 joint ambiguity，而不是让地图单纯更难。

这使 GTET-L 更像“认知诊断 / posterior disambiguation” benchmark。

## 3.5 PRS：session-level transfer benchmark

`src/envs/prs_session.py` 把单 episode family 组织为四个 block：

- A：tutor-on training
- B：tutor-off IID
- C：tutor-off topology shift
- D：tutor-off semantic shift

其中 `LatentCostRiskHead` 可以跨 episode 持久化，这意味着 PRS 主要测的是：

> tutor 干预是否帮助 learner 学会了一个可迁移的 world model。

---

## 4. learner 机制

当前主线 learner 不是显式 symbolic student model，而是“belief + latent predictor + bounded planning”组合。

## 4.1 特征观察与 belief state

feature 观察函数在 `src/agents/observation_model.py`：

- `observe_features()`
- `observe_features_patch()`

观测对象是 4D feature，而不是 risk scalar。

默认噪声模型：

- self cell：`σ² = 0.01`
- 1-hop neighbor：`σ² = 0.08`
- far patch cell：`σ² = 0.20`

agent 对每个 cell 维护 Gaussian belief：

- `mean[r,c,:]`
- `var[r,c,:]`

在 `FeatureBeliefMap.update()` 中，更新是标准逐维 Kalman 形式：

```text
K = pv / (pv + ov)
μ' = μ + K (o - μ)
σ'^2 = pv (1 - K)
```

其中：

- `pv` 是 prior variance
- `ov` 是 observation variance
- `o` 是 noisy observed feature

此外，`FeatureBeliefMap` 还维护 provenance metadata：

- ever_seen
- ever_traversed
- best_view_quality
- reachable_since_t
- intervention_tags

这意味着当前 learner state 不是只有数值 belief，还有关于“这个 belief 是怎么来的”的元信息。

## 4.2 cost/risk world model

当前主线的 latent predictor 在 `src/agents/cost_risk_model.py`。

### 4.2.1 cost head

cost predictor 是 Bayesian linear head：

```text
ĉ(x) = w_c · x + b_c
```

在线更新采用 Gaussian likelihood + L2 prior 的 MAP/SGD 近似：

```text
error = y - (w_c · x + b_c)
grad_w = - error · x · weight + w_c / prior_var
grad_b = - error · weight
```

### 4.2.2 risk head

风险预测头在 `src/agents/risk_model.py`：

```text
r̂(x) = σ(w_r · x + b_r)
```

其中 `σ` 是 sigmoid。

在线更新同样是 MAP 风格 logistic head：

```text
p = σ(w_r · x + b_r)
error = y - p
grad_w = - error · x · weight + w_r / prior_var
grad_b = - error · weight
```

这里的 `y` 不是固定 0/1：

- fatal trap：`y = 1`
- visited risky but survived：
  - 若 `risk_supervision="oracle_visited"`，则 `y = true_risk`
  - 若 `risk_supervision="binary_outcome"`，则 `y = 0`
- safe cell：`y = 0`

### 4.2.3 joint latent predictor

当前 canonical latent predictor 是 `LatentCostRiskHead`：

```text
latent_predictor(x) = (ĉ(x), r̂(x), u_c(x), u_r(x))
```

其中 uncertainty 支持两种来源：

1. Hessian/Laplace 风格 predictor uncertainty
2. 更当前主线化的 directional uncertainty：

```text
u_c = w_c^T diag(var_x) w_c
u_r = w_r^T diag(var_x) w_r
```

也就是把 feature posterior variance 映射到 cost/risk prediction uncertainty。

## 4.3 learner 的规划目标

当前 planner 不是真正 POMDP，也不是 belief tree solver。

`src/agents/belief_planning.py` 文件头已经明确说：

> This module implements belief-conditioned bounded planning.  
> NOT a full belief-tree planner. NOT an exact POMDP solver.

实际搜索引擎是 `src/agents/planner_astar.py` 的 bounded-budget A*。

### 4.3.1 cell-level objective

当前 canonical latent cell objective 在 `cell_cost_v2_latent()` 中写得最清楚：

```text
J_i
= λ_c · ĉ_i
 + risk_penalty_i
 + λ_uc · (1 - n) · u^c_i
 + λ_ur · (1 - n) · u^r_i
 + warned_extra_i
```

其中：

- `ĉ_i`：predicted traversal cost
- `u^c_i, u^r_i`：directional uncertainty
- `n ∈ [0,1]`：route necessity
- `warned_extra_i`：warning 带来的额外规划惩罚

risk penalty 的完整形式是：

```text
risk_penalty_full = λ_r · [-log(1 - clip(r̂_i, ε, 1-ε))]
```

如果有 shield，则再乘：

```text
risk_penalty_full ← risk_penalty_full · (1 - shield_risk_reduction)
```

### 4.3.2 route necessity discount

Phase 10 的关键原则是：

> unknown ≠ dangerous

代码里通过 `route_necessity` 折扣 epistemic penalty：

```text
necessity_discount = 1 - route_necessity
```

并且对 risk penalty 做了“训练程度 × 必要性”的混合：

```text
learning_factor = min(1, n_updates / 10)
risk_penalty
= risk_penalty_full · [learning_factor + (1 - learning_factor) · necessity_discount]
```

直觉上：

- 如果 predictor 还很不成熟，很多“risk”其实只是 epistemic uncertainty 假装成 risk
- 当 route necessity 很高时，不应该把“未知”一律当成危险

这是当前主线 learner 里一个非常重要的设计判断。

### 4.3.3 bounded planning 输出

`plan_from_belief()` 输出 `BeliefPlan`：

- chosen action
- next position
- prefix
- full path
- expected cost
- expected risk
- uncertainty
- runner-up gap
- action confidence
- dominant reason

action confidence 形式为：

```text
conf = gap / (gap + temperature)
```

其中 `gap` 是最佳路径和 runner-up 路径的 path-level score gap。

## 4.4 learner 的在线更新链

当前 learner 的在线学习主要发生在 `LatticeV2Runner.plan_and_move()` 结算 outcome 时。

### 4.4.1 访问 risky cell

如果进入 risky cell：

- `risky_entered += 1`
- 真实生存风险是 `effective_risk`

若有 shield：

```text
effective_risk = true_risk · (1 - shield_risk_reduction)
```

如果死亡：

```text
latent_predictor.update_from_outcome(x_belief, cost_label=true_cost, risk_label=1.0, weight=4.0)
```

如果进入 risky 但存活：

```text
risk_label =
  true_risk,   if oracle_visited
  0.0,         if binary_outcome

latent_predictor.update_from_outcome(..., weight=1.5)
```

### 4.4.2 访问 safe cell

```text
latent_predictor.update_from_outcome(..., risk_label=0.0, weight=0.1)
```

这说明当前 learner 学习信号强度是高度不对称的：

- fatal risk：强更新
- visited risky survive：中等更新
- safe：弱更新

## 4.5 tutor 对 learner 的直接作用渠道

当前主线里 tutor 影响 learner 主要通过以下渠道。

### 4.5.1 WARN

WARN 在仓库里并不是一个完全统一的单机制，而是有多个分支：

- legacy bias 路径
- RSA 路径
- Phase 10 belief evidence 路径

在传统 `warning_update.py` 里，WARN 包含两个机制：

1. 对风险头做 pseudo-label injection
2. 对 lane / segment 增加 temporary warning bias

其匹配权重形式是：

```text
α(x, u) = exp(- ||x - proto_u||² / τ)
```

lane bias 是：

```text
b_warn(lane) = Σ_j α_j(u) · y_u
scaled_bias = λ_lane_warn · b_warn(lane)
```

而在当前 Phase 10 路径中，`FeatureBeliefMap.apply_warn_update()` 还会把 warning 作为 belief evidence factor：

```text
pseudo_obs = μ + warn_strength · warn_direction
pseudo_var = var · warn_confidence
μ' = μ + K (pseudo_obs - μ)
σ'^2 = var (1 - K)
```

所以严格说：

> 当前代码里的 WARN 不是单一语义动作，而是“planner bias / pseudo-label / belief evidence”三层机制的组合与分流。

### 4.5.2 UNLOCK

UNLOCK 直接改：

- `passable`
- `belief_cost`

并额外执行：

```text
feature_belief.apply_unlock_update(...)
```

其效果是不改均值，只降低该 cell 的 posterior uncertainty：

```text
var ← var · (1 - beta_unlock)
```

这代表：

> UNLOCK 表示“这个路径现在值得考虑了”，  
> 而不是“tutor 告诉你这格一定安全”。

### 4.5.3 ITEM_DROP

ITEM_DROP 只改 inventory，不直接改 belief：

```text
inventory.add_shield()
```

因此 ITEM_DROP 是 execution-time dynamics intervention，不是 epistemic intervention。

---

## 5. tutor 机制

## 5.1 RobotBelief：teacher 对 learner 的 surrogate

`src/teachers/robot_belief.py` 明确写了：

> NOT full Bayesian nested inference.

当前 tutor 对 learner 的建模不是精确 nested Bayes，而是一个 configurable surrogate copy。

它维护：

- `agent_belief_mean`
- `agent_belief_var`
- `agent_search_budget`
- `agent_planner_weights`
- `_predictor_snapshot`

支持的 mismatch 模式包括：

- `exact`
- `noisy`
- `stale`

因此当前 tutor 机制的准确表述应该是：

> tutor 不是在恢复 learner 的真 latent mind state，  
> 而是在维护一个可以做 counterfactual rollout 的近似 surrogate。

## 5.2 AgentPredictor：counterfactual surrogate rollout

`src/teachers/agent_predictor.py` 用 `RobotBelief` 构造 surrogate predictor，然后调用 learner 同一套 planner：

```text
predict_agent_prefix(...)
```

也就是说：

> tutor 预测 learner 行为，不是靠另一套 heuristic rule，  
> 而是直接在 surrogate 上运行同一个 belief planner。

这使当前 tutor 设计具备一个很重要的性质：

> WAIT / WARN / UNLOCK / ITEM_DROP 的比较，全部来自同一 counterfactual 预测引擎。

它会输出：

- predicted plan
- predicted failure modes
- candidate scores

并支持 counterfactual:

- after warn
- after unlock
- after item drop

## 5.3 InterventionPolicy：当前主线 tutor scorer

当前主线 tutor 决策核心在 `src/teachers/intervention_policy.py`。

这是当前多 action intervention family 的主 scorer。

### 5.3.1 WAIT score

先跑 baseline WAIT rollout：

```text
pred_wait = predict_agent_prefix(...)
```

然后：

```text
Q_WAIT
= w_LG · learning_gain
 - w_cat · wait_risk
 - w_deadline · deadline_miss
 - β_bore · boredom_penalty
```

其中 boredom penalty 为：

```text
avg_prefix_cost = expected_cost / prefix_len
boredom_penalty = avg_prefix_cost / (ε + learning_gain)
```

直觉上是：

> 如果继续 WAIT 只会继续花时间、但几乎学不到东西，那 WAIT 应该被惩罚。

### 5.3.2 WARN score

```text
catastrophe_reduction = max(0, wait_risk - warn_risk)
Q_WARN = w_warn · catastrophe_reduction - autonomy_penalty
```

### 5.3.3 UNLOCK score

```text
unlock_cat_reduction = max(0, wait_risk - unlock_risk)
topology_improvement = max(0, wait_path_len - unlock_path_len)

Q_UNLOCK
= w_unlock · (unlock_cat_reduction + 0.1 · topology_improvement)
 - autonomy_penalty
```

### 5.3.4 ITEM_DROP score

```text
item_cat_reduction = max(0, wait_risk - item_risk)
Q_ITEM = w_item · item_cat_reduction - item_drop_cost
```

## 5.4 Bottleneck diagnosis 与 perceptual model

Phase 10 在 intervention scorer 上又叠加了两层 tutor 诊断：

1. bottleneck diagnosis
2. tutor perceptual model

### 5.4.1 Tutor perceptual model

`src/teachers/perceptual_model.py` 维护 tutor 对“agent 已经有效看过哪些 cell”的估计：

```text
ρ_{i,t+1} = 1 - (1 - ρ_{i,t}) (1 - p_see_{i,t+1})
```

其中：

```text
p_see = exp(-λ_d · d) · q_obs
q_obs = 1 / (1 + obs_var)
```

这不是 learner 自己的 belief，而是 tutor 对 learner perceptual access 的估计。

WARN 冗余度定义为：

```text
R_warn = (1/|D|) Σ_i ρ_i · exp(-u_r_i / τ_u)
```

如果 tutor 认为 agent 已经看过这些 cell，而且风险不确定性也不高，那么 WARN 会被 redundancy penalty 压制。

### 5.4.2 Bottleneck matching

`diagnose_bottleneck(...)` 把当前状态分成：

- epistemic
- structural
- outcome

然后给 intervention 附加 matching bonus：

```text
bonus(a) = β_b · match_intervention_to_bottleneck(a, bn)
```

当前实现还包含：

- outcome-dominant WARN damping
- repeated WARN penalty
- repeated UNLOCK suppression

所以当前 tutor 已经不只是“预估生存收益”，而是带有：

- stage diagnosis
- perceptual access diagnosis
- anti-redundancy
- anti-over-intervention

这些 meta-control。

---

## 6. 冻结的 canonical internalization baseline

这一部分不是当前 V2 runner 主线，但在仓库里仍然是“冻结基准线”，不能忽略。

## 6.1 A1 5D observer

`src/teachers/internalization_observer.py` 当前最重要的冻结 observer 是：

- `A1MtObserver`
- `A1MtObserverFrozen`

它估计的 5D 状态是：

```text
m̂_t = (τ̂, ν̂, γ̂_gen, γ̂_spec, κ̂)
```

语义分别是：

- `τ̂`：trust / valid-advice uptake
- `ν̂`：dependence / blind obedience
- `γ̂_gen`：general suppression of exploration
- `γ̂_spec`：temptation-specific resistance state
- `κ̂`：risk calibration

### 6.1.1 τ̂ / ν̂ / γ̂_gen

A1 的更新形式是事件驱动 + 条件回归：

```text
τ̂' = τ̂
    + α_τ⁺ · e_trust⁺ · (1 - τ̂)
    - α_τ⁻ · e_trust⁻ · τ̂
    + λ_eff,τ · (τ₀ - ·)
```

```text
ν̂' = ν̂
    + α_ν⁺ · e_blind · (ν_max - ν̂)
    - α_ν⁻ · e_selfdisc · ν̂
    + λ_eff,ν · (ν₀ - ·)
```

```text
γ̂_gen' = γ̂_gen
       + α_γ⁺ · e_pressure · (γ_max - γ̂_gen)
       - α_γ⁻ · e_explore⁺ · γ̂_gen
       + λ_eff,γ · (γ₀ - ·)
```

其中条件回归的 effective λ 是：

```text
λ_eff = λ_base · (1 - confidence)
```

也就是说：

> 当最近没有 informative event，且该维 confidence 低时，才慢慢回归先验。

### 6.1.2 γ̂_spec

temptation-specific resistance 的更新是：

如果当前有 lure 且 learner 走了 oracle-safe 路线：

```text
e_resist = lure · (1 - γ̂_spec)
γ̂_spec ← γ̂_spec + α_resist · e_resist
```

如果 learner 跟随 lure：

```text
e_follow = lure · γ̂_spec
γ̂_spec ← γ̂_spec - α_follow · e_follow
```

### 6.1.3 κ̂

代码与 `docs/architecture_5d_canonical.md` 一致：

```text
δ_risk = risk_true - risk_hat

κ̂_{t+1}
= (1 - λ_κ) κ̂_t + λ_κ κ₀
 + α_κ⁺ · max(δ_risk, 0) · (κ_max - κ̂_t)
 + α_κ⁻ · min(δ_risk, 0) · (κ̂_t - κ_min)
```

且只在：

```text
risk ≥ risk_gate_threshold
and risk_hat is available
```

时更新。

### 6.1.4 confidence

A1 不是简单靠 event count 更新 confidence，而是结合：

- timing separation
- probe agreement
- predictive consistency

再通过 EMA 更新 `conf_tau / conf_nu / conf_gamma`。

## 6.2 冻结的 2-act canonical tutor

`docs/canonical_baseline_spec.md` 和 `configs/tutor/bcictv4_2act.yaml` 定义的 canonical baseline 是：

- 2-act `{WAIT, WARN}`
- `lambda_teach = 3.5`
- `lambda_over = 4.0`
- `lambda_sd = 1.5`
- `lambda_dep = 2.0`

但需要明确：

> 对应的代码主体 `internalization_control_tutor_v4.py` 文件头已经标为 archival。

因此更稳妥的说法是：

- 它仍是 repo 的冻结 canonical baseline 参考
- 但不是当前 V2 主线 intervention family 的执行入口

从 archival 代码本身看，它的旧打分结构大致是：

```text
V_full = V + λ_sd · p_sd - λ_dep · p_blind
Q = Q_online + λ_teach · V_full - λ_over · R
```

2-act canonical 情况下，相当于比较：

- `Q_WAIT`
- `Q_WARN`

加上 κ̂ 的宏观 lesson bonus。

## 6.3 κ̂ macro bonus

这部分在 `docs/architecture_5d_canonical.md` 中是冻结默认开启：

```text
S_teach^5d(ℓ)
= S_teach^base(ℓ)
 + β_κ · 1[ℓ ∈ L_risk] · |κ̂ - κ₀|
```

当前 canonical 文档给出的默认是：

```text
β_κ = 0.02
```

这部分更像 lesson ranking / macro diagnostic bonus，而不是当前 V2 step-wise intervention scorer 的一部分。

---

## 7. 当前代码现状

如果只从当前代码入口看，现状可以概括为以下几条。

## 7.1 当前主 benchmark 已经是 V2 intervention family

`scripts/eval_dtmb_main.py` 的 canonical policy 明确是：

- `robot_belief_mode = True`
- `intervention_family_mode = True`
- `item_drop_enabled = True`
- `belief_planning_mode = True`
- `latent_mode = True`

这说明当前主线评估已经不再是旧 2-act tutor，而是 4-action V2 family。

## 7.2 README 不是当前主线权威说明

仓库根 README 仍主要讲：

- v0
- v1a
- v1d

也就是更早期的 grid + oracle / particle teacher 叙述。

因此：

> 当前 repo 的真实权威说明已经从 README 转移到 `docs/* + src/*`。

## 7.3 当前仓库存在“冻结 canonical baseline”和“现行 V2 主线”双系统并存

这不是逻辑错误，但会带来一个解释风险：

- 讨论“canonical”时，有人指的是 A1 + BCICTv4 2-act
- 讨论“current mainline”时，实际跑的是 V2 intervention family

如果不显式区分，会非常容易把结论串错。

## 7.4 PRS 表明当前项目已进入 transfer / persistence 阶段

`scripts/eval_prs_transfer.py` 和 `src/envs/prs_session.py` 说明项目当前不是只看单 episode survival，而是在检验：

- tutor-on training 是否带来 tutor-off transfer
- 拓扑 shift / 语义 shift 下是否还能保持收益

这说明当前项目研究重点已经从“单次风险提示是否有效”推进到：

> tutor 是否帮助 learner 建立了可迁移的 latent world model。

---

## 8. 当前代码的主要缺陷与风险

下面这些不是凭空批评，而是直接从代码结构能观察到的系统性问题。

## 8.1 文档中心漂移：README 已经过时

问题：

- README 仍在讲旧版本故事
- 当前主线其实在 `LatticeV2Runner + intervention_policy + DTMB/GTET/PRS`

后果：

- 新读者极易误以为当前系统仍是“Oracle teacher + bounded rational agent”那条线
- 报告与实验口径容易错位

## 8.2 主线与冻结基线双中心并存，概念负担很高

仓库同时包含：

- V2 主线 intervention family
- 冻结的 A1 / BCICTv4 internalization baseline
- archival tutor v4 源码

后果：

- “canonical”一词并不唯一
- “最新 tutor”到底指 2-act BCICTv4 还是 4-action intervention family，不看代码很难判断

## 8.3 current WARN 语义并不唯一

当前 WARN 机制分成：

- legacy pseudo-label + lane bias
- RSA-only belief update
- hybrid RSA + Phase 10 belief evidence
- direct `apply_warn_update`

这导致：

> “WARN 有效”并不是一个单一机制结论，而是变体依赖结论。

这会增加消融实验和负结果解释的复杂度。

## 8.4 RobotBelief 不是 full nested Bayesian inference

这是代码注释自己就明确说的。

后果：

- tutor 对 learner 的建模是 surrogate，不是严格逆推断
- 如果研究问题写成“teacher 精确识别 learner latent state”，就会过度表述

更准确的 framing 应该是：

> 当前 tutor 维护的是一个用于 counterfactual intervention scoring 的近似 surrogate learner model。

## 8.5 belief planner 也不是 POMDP solver

`belief_planning.py` 文件头已经明确否定了 full belief-tree / exact POMDP。

所以当前 learner 规划的准确表述应该是：

> belief-conditioned bounded replanning

而不是：

> optimal belief-space planning

这在研究报告里必须说清楚。

## 8.6 主线复杂度高，runner 承担过多相位开关

`LatticeV2Runner` 同时承担：

- latent mode
- patch observation
- prefix prediction
- belief planning
- robot belief
- intervention family
- perceptual access
- RSA warning variants
- GTET factor modes
- PRS hooks

优点是统一入口强；
缺点是：

- mode interaction 很多
- 单次改动更容易出现隐式耦合
- 想做严格消融时，必须非常清楚自己打开了哪些开关

## 8.7 canonical dead-zone tie policy 不在主源码内

`docs/canonical_baseline_spec.md` 明确说：

> ε_Q = 0.05 的 dead-zone tie policy 是 external wrapper, not in source code

这意味着：

- 文档里的“推荐 canonical 行为”并没有 100% 内嵌在核心源码
- 如果未来复现实验，容易出现 wrapper 行为缺失

## 8.8 PRS 当前持久化的是 predictor，不是完整 agent cognition

在 `prs_session.py` 中，跨 episode 持久化的是：

- `LatentCostRiskHead`

而不是：

- 完整 `FeatureBeliefMap`
- 完整路径经验
- 完整 tutor-side observer

这使 PRS 的 transfer 含义更准确地说是：

> world-model parameter transfer

而不是完整意义上的：

> 全认知状态 transfer

---

## 9. 如何理解“risk / danger”在当前系统里的位置

当前系统没有把 “danger” 实现成一个完全独立于 risk 的新变量。

更准确地说：

1. 环境层：
   - 每个 cell 有 `true_risk`
   - risky cell 的 hazard 以 Bernoulli 方式结算

2. learner 层：
   - 维护 `r̂(x)` 作为 risk probability
   - 维护 `u_r(x)` 作为 risk uncertainty
   - 规划里使用的是 risk penalty transform `-log(1-r̂)`

3. tutor 层：
   - 主要关心 `catastrophe_reduction`
   - 以及 prefix 上 failure modes / bottleneck diagnosis

因此：

> 当前主线里，“danger”更多是 outcome-level catastrophe risk 的 operational interpretation，  
> 而不是一套与 `risk_hat` 平行存在的新数学头。

---

## 10. 推荐的阅读顺序

如果要真正理解当前代码，而不是只看表面文件名，建议按这个顺序读：

1. `src/envs/lattice_v2_runner.py`
2. `src/agents/feature_belief.py`
3. `src/agents/cost_risk_model.py`
4. `src/agents/planner_astar.py`
5. `src/agents/belief_planning.py`
6. `src/teachers/robot_belief.py`
7. `src/teachers/agent_predictor.py`
8. `src/teachers/intervention_policy.py`
9. `src/teachers/perceptual_model.py`
10. `src/envs/dtmb_lattice.py`
11. `src/envs/gtet_lattice.py`
12. `src/envs/prs_session.py`
13. `docs/architecture_5d_canonical.md`
14. `src/teachers/internalization_observer.py`
15. `configs/tutor/bcictv4_2act.yaml`
16. `src/teachers/internalization_control_tutor_v4.py` 仅作归档参照

---

## 11. 最后的总判断

如果一句话概括当前 `pedagogical_ip`：

> 它现在已经不是一个单纯的“teacher 提醒 learner 别踩坑”的项目，而是一个以 latent feature world-model、bounded belief planning、surrogate learner modeling 和多 intervention family 为核心的教学型交互系统。

但如果再补一句更现实的判断：

> 当前仓库最大的系统性问题，不是单个公式错，而是“冻结 canonical baseline”和“当前 V2 主线”并存，且 warning / tutor 语义分支较多，导致概念解释与实验口径都需要非常小心。

如果后续要继续做系统化研究，最值得先收敛的不是再叠机制，而是：

1. 把 README 升级为当前主线说明
2. 明确区分“frozen baseline”与“active mainline”
3. 收敛 WARN 语义分支
4. 给 V2 主线写一份单独的 canonical contract

