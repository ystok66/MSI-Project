# Tutor 机制全览报告

> 最后更新：Phase 2B 完成后（2026-04-07）
> 覆盖范围：`src/teachers/` 全部模块 (46 个 .py 文件)

---

## 目录

1. [Tutor 总体架构](#1-tutor-总体架构)
2. [干预决策主线（成熟）](#2-干预决策主线成熟)
3. [counterfactual Agent 预测器（成熟）](#3-counterfactual-agent-预测器成熟)
4. [Robot Belief 代理模型（成熟）](#4-robot-belief-代理模型成熟)
5. [瓶颈诊断系统（成熟）](#5-瓶颈诊断系统成熟)
6. [感知接入模型 TPM（成熟）](#6-感知接入模型-tpm成熟)
7. [干预语义系统（成熟）](#7-干预语义系统成熟)
8. [Boredom / Frustration 代理（成熟 / Promoted）](#8-boredom--frustration-代理成熟--promoted)
9. [Warning 子类型选择器（Shadow-only）](#9-warning-子类型选择器shadow-only)
10. [Internalization Observer（成熟 / 5D Canonical）](#10-internalization-observer成熟--5d-canonical)
11. [Shadow / Ablation 系统](#11-shadow--ablation-系统)
12. [残缺 / 未完成模块](#12-残缺--未完成模块)
13. [文件索引](#13-文件索引)

---

## 1. Tutor 总体架构

```
                    ┌───────────────────────────┐
                    │     Lattice V2 Runner      │
                    │  (每步调 tutor.decide)      │
                    └─────────┬─────────────────┘
                              │ agent state
                    ┌─────────▼─────────────────┐
                    │      RobotBelief           │
                    │  (surrogate of agent's     │
                    │   belief + competence)      │
                    └─────────┬─────────────────┘
                              │ surrogate plan
                    ┌─────────▼─────────────────┐
                    │   AgentPredictor           │
                    │  (counterfactual rollouts)  │
                    │  WAIT / WARN / UNLOCK /     │
                    │  ITEM_DROP scenarios        │
                    └─────────┬─────────────────┘
                              │ Q-values
                    ┌─────────▼─────────────────┐
                    │   InterventionPolicy       │
                    │  + BottleneckDiagnosis      │
                    │  + PerceptualAccess         │
                    │  + BoredomPenalty           │
                    └─────────┬─────────────────┘
                              │ best action
                    ┌─────────▼─────────────────┐
                    │  InterventionSemantics      │
                    │  (execution: apply action)  │
                    └───────────────────────────┘
```

**核心原则：**
- Tutor **不能访问** true trap 位置、true latent vectors、true future risk
- 所有预测通过 **surrogate counterfactual rollout** 得到
- 所有干预使用明确的 **语义分离**：WARN→belief / UNLOCK→topology / ITEM_DROP→traversal

---

## 2. 干预决策主线（成熟）

> 源文件：[intervention_policy.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/intervention_policy.py)

### 2.1 四种动作的 Q-value 公式

#### Q_WAIT

```
Q_WAIT = β_learn · LG − β_cat · R_wait − β_ddl · D_wait − β_bore · B_wait
```

| 符号 | 含义 | 默认值 |
|------|------|--------|
| β_learn | learning_gain_weight | 1.0 |
| β_cat | catastrophe_weight | 5.0 |
| β_ddl | deadline_weight | 2.0 |
| β_bore | boredom_weight | **0.3** (Phase 1B promoted) |
| LG | predict prefix 上的平均 uncertainty | 由 `estimate_learning_gain` 计算 |
| R_wait | WAIT rollout 的 expected_risk | 由 surrogate plan 得到 |
| D_wait | deadline miss probability | 由 `estimate_failure_modes` 得到 |
| B_wait | boredom penalty = avg_cost/(ε+LG) | 见 §8 |

#### Q_WARN

```
Q_WARN = β_warn · max(0, R_wait − R_warn) − β_auto
       + β_b · M(WARN, bottleneck)
       − β_R · Redundancy
       − warn_damping
       − warn_repeat_penalty
```

| 符号 | 含义 | 默认值 |
|------|------|--------|
| β_warn | warn_effect_weight | 3.0 |
| β_auto | autonomy_penalty | 1.0 |
| β_b | bottleneck_match_weight | 2.0 |
| β_R | redundancy_penalty_weight | 1.5 |
| R_warn | WARN counterfactual 的 expected_risk | |
| M(WARN, bn) | `match_intervention_to_bottleneck("WARN", bn)` = bn.epistemic | |
| Redundancy | `compute_redundancy(pa, warn_cells, risk_unc)` | |
| warn_damping | outcome_dominant 时 WARN 的抑制 | min(2.0, dominance×0.5) |
| warn_repeat | n_warns × 0.5 | |

#### Q_UNLOCK

```
Q_UNLOCK = β_unlock · (ΔR_cat + 0.1·ΔL_topo) − β_auto
         + β_b · M(UNLOCK, bottleneck)
         − unlock_memory_penalty
```

| 符号 | 含义 | 默认值 |
|------|------|--------|
| β_unlock | unlock_effect_weight | 3.0 |
| ΔR_cat | max(0, R_wait − R_unlock) | |
| ΔL_topo | max(0, L_wait − L_unlock) — path length gain | |
| M(UNLOCK, bn) | bn.structural | |
| unlock_memory_penalty | 已用且无 door 时 = 2×β_auto | |

若无可 unlock 的 door：`Q_UNLOCK = −2·β_auto`

#### Q_ITEM_DROP

```
Q_ITEM_DROP = β_item · max(0, R_wait − R_item) − C_item
            + β_b · M(ITEM_DROP, bottleneck)
```

| 符号 | 含义 | 默认值 |
|------|------|--------|
| β_item | item_drop_weight | 3.0 |
| C_item | item_drop_cost | 1.5 |
| M(ITEM, bn) | bn.outcome | |

若 shield 已持有或未启用：`Q_ITEM_DROP = −2·C_item`

### 2.2 决策规则

```python
best_action = argmax_a Q(a)
margin = Q_best − Q_2nd
```

### 2.3 InterventionConfig 全参数表

| 参数 | 默认 | 变种 | 状态 |
|------|------|------|------|
| catastrophe_weight | 5.0 | — | ✅ Frozen |
| learning_gain_weight | 1.0 | — | ✅ Frozen |
| warn_effect_weight | 3.0 | — | ✅ Frozen |
| unlock_effect_weight | 3.0 | — | ✅ Frozen |
| item_drop_weight | 3.0 | — | ✅ Frozen |
| autonomy_penalty | 1.0 | — | ✅ Frozen |
| deadline_weight | 2.0 | — | ✅ Frozen |
| item_drop_cost | 1.5 | — | ✅ Frozen |
| bottleneck_match_weight | 2.0 | — | ✅ Phase 10 |
| redundancy_penalty_weight | 1.5 | — | ✅ Phase 10 |
| boredom_weight | **0.3** | — | ✅ Phase 1B promoted |
| use_bottleneck_matching | True | False=ablation | ✅ |
| use_warn_damping | True | False=ablation | ✅ |
| use_unlock_memory | True | False=ablation | ✅ |
| use_perceptual_access | True | False=ablation | ✅ |

---

## 3. Counterfactual Agent 预测器（成熟）

> 源文件：[agent_predictor.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/agent_predictor.py)

### 3.1 核心机制

Tutor 通过 **surrogate counterfactual rollout** 预测 agent 行为。每种干预生成一个 "what-if" 场景：

| 函数 | 对什么做 counterfactual | 输入修改 |
|------|------------------------|---------|
| `predict_agent_prefix` | WAIT baseline | 无修改 |
| `predict_agent_prefix_after_warn` | WARN | 在 belief_cost 加 warned_cell_extra |
| `predict_agent_prefix_after_unlock` | UNLOCK | 在 passable 数组开门 |
| `predict_agent_prefix_after_item_drop` | ITEM_DROP | 在 inventory 加 shield |

### 3.2 Surrogate Rollout 过程

```
1. Build surrogate predictor = deepcopy(robot_belief.predictor_snapshot)
2. Call plan_from_belief(surrogate) → BeliefPlan
3. Call plan_with_alternatives_v2(surrogate) → cand_scores
4. Call estimate_failure_modes(plan, cand_scores) → FailureModeEstimate
5. Return AgentPrediction(plan, failure_modes, cand_scores)
```

> [!IMPORTANT]
> 所有 rollout 都是 **read-only**：deepcopy surrogate，永不修改真实 agent 或环境。

### 3.3 Learning Gain 估计

```
LG = (1/|prefix|) Σ_{(r,c) ∈ prefix} mean(belief_var[r,c])
```

直觉：predicted prefix 上的平均 uncertainty — agent 游走这些 cell 时能学到多少。

---

## 4. Robot Belief 代理模型（成熟）

> 源文件：[robot_belief.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/robot_belief.py)

### 4.1 数据结构

```
RobotBelief:
  agent_belief_mean: (H, W, d)    # surrogate feature belief
  agent_belief_var:  (H, W, d)    # surrogate uncertainty
  agent_search_budget: int = 30
  agent_risk_weight: float = 3.0
  agent_uncertainty_weight: float = 0.5
  agent_lambda_c, _uc, _ur        # planner weights
  _predictor_snapshot: deepcopy of latent predictor
```

### 4.2 Copy Modes

| 模式 | 含义 | Belief 更新 |
|------|------|------------|
| `exact` | 完全同步 | mean/var = agent copy |
| `noisy` | 有噪声同步 | mean += N(0, σ_noise) |
| `stale` | 延迟同步 | 每 N 步才同步一次 |

### 4.3 Competence Mismatch

Tutor 可以有和 agent 不同的搜索预算和风险权重估计：
```
rb.agent_search_budget = true_budget + budget_mismatch
rb.agent_risk_weight   = true_risk_weight + risk_weight_mismatch
```

### 4.4 Predictor Snapshot

Phase 3A 后支持 **任意 PredictorProtocol 头的 deepcopy**：
- LatentCostRiskHead (4D linear)
- StructuredBasisCostRiskHead (6D/7D basis)
- GenericSlowFastPredictor (dual-timescale wrapper)

---

## 5. 瓶颈诊断系统（成熟）

> 源文件：[bottleneck_diagnosis.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/bottleneck_diagnosis.py)

### 5.1 三维瓶颈分数

每个维度 = severity × intervention_gain。

#### Epistemic Score（→ WARN lever）

```
U_D = (1/|D|) Σ_{i∈D} [ω_ρ(1−ρ_i) + ω_u · u_r_i]
s_epi = U_D · (1 + max(0, Q_WARN − Q_WAIT))
```

| 符号 | 含义 |
|------|------|
| D | decision-relevant prefix cells |
| ρ_i | perceptual access probability (agent 已看到 cell i 的概率) |
| u_r_i | risk_uncertainty_map[r,c] from predictor |
| ω_ρ | unseen probability weight = 1.0 |
| ω_u | risk uncertainty weight = 1.0 |

#### Structural Score（→ UNLOCK lever）

```
g_t = 1.0                         if slack ≤ 0 or has_locked_doors
    = exp(−slack / τ_s)            otherwise

s_str = g_t · (1 + max(0, Q_UNLOCK − Q_WAIT))
```

| 符号 | 含义 | 默认 |
|------|------|------|
| slack | (t_max − t) − shortest_path_len | |
| τ_s | structural urgency temperature | 3.0 |

#### Outcome Score（→ ITEM_DROP lever）

```
s_out = min_path_risk · (1 + max(0, Q_ITEM − Q_WAIT))
```

### 5.2 Intervention-Bottleneck Matching

```
M(WARN,    bn) = bn.epistemic
M(UNLOCK,  bn) = bn.structural
M(ITEM,    bn) = bn.outcome
M(WAIT,    bn) = 0
```

最终 Q 加成：`Q_a += β_b · M(a, bn)`

---

## 6. 感知接入模型 TPM（成熟）

> 源文件：[perceptual_model.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/perceptual_model.py)

### 6.1 核心状态变量

```
PerceptualAccessState:
  seen_prob[H, W]:       ρ_i ∈ [0,1] — agent 已有效看到 cell i 的概率
  effective_obs_var[H, W]: 最佳观测方差
  intervention_memory:    {warn_count, unlock_count, ...}
```

### 6.2 更新公式

每步 agent 位移后：

```
对于 agent_pos 周围 patch_radius 内的每个可通行 cell (r,c):
    d = Manhattan(agent_pos, (r,c))
    if d ≤ patch_radius:
        obs_var = base_self_var           (d=0, default=0.01)
                  base_neighbor_var × d   (d>0, default=0.08×d)
        q_obs = 1 / (1 + obs_var)
        p_see = exp(−λ_d · d) · q_obs                 (λ_d = 0.8)
        ρ_new = 1 − (1 − ρ_old)(1 − p_see)           (单调递增)
```

### 6.3 Redundancy 计算

```
R_warn = (1/|D|) Σ_{i∈D} ρ_i · exp(−u_r_i / τ_u)
```

| 符号 | 含义 | 默认 |
|------|------|------|
| τ_u | uncertainty threshold | 0.3 |

高 redundancy → agent 已经知道这些 cell → warning 浪费。

---

## 7. 干预语义系统（成熟）

> 源文件：[intervention_semantics.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/intervention_semantics.py)

### 7.1 语义分离原则

| 干预 | 修改 belief | 修改 topology | 修改 traversal cost |
|------|------------|--------------|-------------------|
| WARN | ✅ | ✗ | ✗ |
| UNLOCK | ✗ | ✅ | ✗ |
| ITEM_DROP | ✗ | ✗ | ✅ |

### 7.2 WARN 语义

```
μ_i⁺ = μ_i + α_warn · v̂_warn      (α_warn = 0.3)
Σ_i⁺ = (1 − β_warn) · Σ_i         (β_warn = 0.2)
```

其中 `v̂_warn` = normalized direction toward risk awareness (aligned with risk-head gradient)。

### 7.3 UNLOCK 语义

```
s_{t+1}^world = Unlock(s_t^world)   (passable[r,c] = True)
b_{t+1}^{A,env} = AffordanceReveal  (var *= (1 − 0.3))
```

### 7.4 ITEM_DROP (Shield) 语义

```
TraversalCost^shield(i) = λ_r · (1 − γ_shield) · φ(r̂_i)
φ(r) = −ln(1 − r)                  (logarithmic risk cost)
γ_shield = 0.5                     (50% risk reduction)
```

---

## 8. Boredom / Frustration 代理（成熟 / Promoted）

> 源文件：[boredom_proxy.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/boredom_proxy.py)

### 8.1 核心公式

```
B_wait = avg_prefix_cost / (ε + max(0, LG))

avg_prefix_cost = expected_cost / max(1, prefix_len)
Q_WAIT_new = Q_WAIT_old − β_bore · B_wait
```

**唯一新超参**：β_bore = 0.3（Phase 1B 实验确认并 promoted）

### 8.2 设计直觉

| 状态 | LG | B_wait | 效果 |
|------|-----|--------|------|
| Agent 在学习 | 高 | 低 | WAIT 不受惩罚 |
| Agent 啥也没学但还在走 | ≈0 | 高 | WAIT 被惩罚 → 倾向干预 |
| Agent 几乎不动 | 低 | 中 | 轻度惩罚 |

---

## 9. Warning 子类型选择器（Shadow-only）

> 源文件：[warning_utterance_policy.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/warning_utterance_policy.py)

### 9.1 状态：🟡 Shadow-only — 仅记录，不影响决策

四种子类型：

| 子类型 | 条件 | 含义 |
|--------|------|------|
| `hint` | p_self>0.5, p_blind<0.3 | agent 自己还能发现 |
| `alert` | p_blind>0.5, time moderate | 紧急打断错误路径 |
| `explain` | p_blind<0.5, time ample | 有时间教推理 |
| `directive` | p_timeout>0.7 or catastrophe | 直接指令 |

### 9.2 评分公式

```
S_hint      = 0.3 + 1.0·p_self − 0.5·p_blind − 0.3·p_timeout + 0.3·t_rem − 0.2·ν̂
S_alert     = 0.5·p_blind + 0.3·p_timeout − 0.2·p_self + 0.1·(1−t_rem)
S_explain   = 0.3·t_rem + 0.2·(1−p_blind) + 0.2·(1−p_timeout) − 0.1·ν̂
S_directive = 0.8·max(p_timeout−0.7, 0) + 0.5·max(p_blind−0.5, 0) − 0.3·t_rem − 0.2·p_self
```

---

## 10. Internalization Observer（成熟 / 5D Canonical）

> 源文件：[internalization_observer.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/internalization_observer.py)

### 10.1 观测状态维度

系统追踪 agent 的 5 维内化状态估计 m̂_t：

| 维度 | 符号 | 含义 | 范围 | 版本 |
|------|------|------|------|------|
| 信任 | τ̂ | 对 tutor advice 的信任 | [0, 1] | A0+ |
| 依赖 | ν̂ | 对 tutor 的过度依赖 | [0, 0.8] | A0+ |
| 泛化抑制 | γ̂_gen | 探索泛化的抑制程度 | [0, 0.5] | A0+ |
| 诱惑特异性 | γ̂_spec | 抵抗特定诱惑的能力 | [0, 1] | P4-A (A1+) |
| 风险校准 | κ̂ | 风险估计校准状态 | [0, 1] | P5 (A1+) |

### 10.2 τ̂ 更新（信任）

```
τ̂_new = τ̂_old + α⁺_τ · e_trust⁺ · (1 − τ̂)     ← 正确 warn 被遵从 → 信任↑
       − α⁻_τ · e_trust⁻ · τ̂                    ← 错误 warn 被遵从 → 信任↓
       + λ_τ·(1−conf_τ)·(τ₀−τ̂)                  ← 条件 mean-reversion（无近期事件时）
```

| 参数 | A0 | A1/Frozen |
|------|-----|-----------|
| α⁺_τ | 0.22 | 0.22 |
| α⁻_τ | 0.10 | 0.10 |
| λ_τ | 0.02 | **0.005** (conditional) |
| τ₀ | 0.30 | 0.30 |
| β_τ_probe | 0.15 | **0.0** (gated OFF) |

### 10.3 ν̂ 更新（依赖）

```
ν̂_new = ν̂_old + α⁺_ν · e_blind · (ν_max − ν̂)   ← blind compliance → 依赖↑
       − α⁻_ν · e_selfdisc · ν̂                   ← self-discovery → 依赖↓
       + λ_ν·(1−conf_ν)·(ν₀−ν̂)                   ← 条件 mean-reversion
```

| 参数 | A0 | A1/Frozen | A2 |
|------|-----|-----------|-----|
| α⁺_ν | 0.18 | 0.18 | 0.18 |
| α⁻_ν | 0.13 | 0.13 | 0.13 |
| ν_max | 0.80 | 0.80 | 0.80 |
| e_blind | warn∧follow·(1−p_self) | same | **dose>0 ∧ comply·(1−p_self)** |

> [!NOTE]
> A2 的关键修复：e_blind 从 `(warned ∧ follow_warn)` 扩展到 `(dose>0 ∧ comply)`，捕获 SOFT 干预。

### 10.4 γ̂_gen 更新（泛化抑制）

```
γ̂_gen_new = γ̂_gen_old + α⁺_γ · e_pressure · (γ_max − γ̂_gen)
          − α⁻_γ · e_explore⁺ · γ̂_gen
          + λ_γ·(1−conf_γ)·(γ₀−γ̂_gen)
```

其中 `e_pressure = EMA(dose, α=0.3)`

### 10.5 γ̂_spec 更新（诱惑特异性, P4-A）

```
if lure ≥ 0.3:
    if agent chose safe:  γ̂_spec += α_resist · lure · (1 − γ̂_spec)
    else:                 γ̂_spec −= α_follow · lure · γ̂_spec
```

### 10.6 κ̂ 更新（风险校准, P5）

```
if risk ≥ 0.1 and risk_hat available:
    δ = risk − risk_hat                          (signed error)
    κ̂_new = (1−λ_κ)·κ̂ + λ_κ·κ₀
    if δ > 0: κ̂_new += α⁺_κ · δ · (κ_max − κ̂)  ← underestimate → more cautious
    if δ < 0: κ̂_new += α⁻_κ · δ · (κ̂ − κ_min)  ← overestimate → relax
```

### 10.7 Observer 版本谱系

| 版本 | 类名 | 关键改动 | 状态 |
|------|------|---------|------|
| A0 | `RuleBasedMtObserver` | 原始 rule-based 3D | 🟡 Legacy baseline |
| **A1** | `A1MtObserver` | probe OFF, conditional reversion, +γ̂_spec+κ̂ | ✅ **Canonical** |
| A1-Frozen | `A1MtObserverFrozen` | 参数锁定（2026-03-28 冻结）| ✅ Frozen baseline |
| A2 | `A2MtObserver` | expanded blind, action stability confidence | 🟡 Experimental |

---

## 11. Shadow / Ablation 系统

### 11.1 Shadow 模块列表

以下模块在 `src/teachers/` 中存在，但运行在 **shadow mode**（仅记录，不影响主线决策）：

| 模块 | 文件 | 状态 | 用途 |
|------|------|------|------|
| Probabilistic Shadow Observer | `a1mt_observer_shadow_prob.py` | 🟡 Shadow | Beta/Gaussian latent filter 诊断 |
| Shadow Bridge | `a1mt_observer_shadow_bridge.py` | 🟡 Shadow | 连接 shadow observer 到 tutor |
| Shadow Types | `a1mt_observer_shadow_types.py` | 🟡 Shadow | 类型定义 |
| MicroBayes Shadow v2/v2.1/v3 | `micro_bayes_shadow_v2*.py` | 🟡 Shadow | 微观 Bayesian 诊断 |
| Shadow Bridge (generic) | `shadow_bridge.py` | 🟡 Shadow | 通用 shadow 桥接 |
| P_Self Calibration | `p_self_calibration.py` | 🟡 Shadow | p_self 参数校准实验 |
| P_Self Posterior Shadow | `p_self_posterior_shadow.py` | 🟡 Shadow | p_self 后验估计 |

### 11.2 InterventionConfig Ablation Toggles

四个 ablation 开关（默认全 True = full TPM）：

| Toggle | 关闭效果 |
|--------|---------|
| `use_bottleneck_matching = False` | 去掉瓶颈-干预匹配加成 |
| `use_warn_damping = False` | 去掉 outcome-dominant 时 WARN 抑制 |
| `use_unlock_memory = False` | 不惩罚重复 UNLOCK |
| `use_perceptual_access = False` | 不用感知接入模型做 redundancy |

---

## 12. 残缺 / 未完成模块

### 12.1 存在但未接入主线的模块

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| `internalization_control_tutor_v4.py` | BCICTv4 | ⚪ 档案 | 早期 internalization control tutor，被 intervention_policy 替代 |
| `intervention_risk_head.py` | 独立 risk head | ⚪ 档案 | Tutor 独有的 risk head，不在主线 |
| `consequence_grounded_option_rollout.py` | CGOR | 🔲 设计 | 后果驱动的 option rollout, 未完全接入 |
| `option_intervention_controller.py` | OIC | 🔲 设计 | Option-level intervention controller |
| `gtet_factor_adapter.py` | GTET factor | ✅ 成熟 | GTET 场景的 `G_THETA` factor 模式适配 |
| `credit_correction.py` | | 🔲 实验性 | 归因校正 |
| `effort_latent_shadow.py` | | 🟡 Shadow | 努力潜变量追踪 |
| `cause_scoring.py` | | 🟡 Shadow | 原因打分系统 |
| `time_aware_door_tutor.py` | | 🟡 Legacy | 时间感知的门控 tutor |

### 12.2 Goal/Preference 系列（实验性）

以下模块用于 GTET 场景的目标/偏好推断，处于不同成熟阶段：

| 模块 | 状态 | 说明 |
|------|------|------|
| `goal_temptation_posterior.py` | 🟡 | (g,z) 联合后验 |
| `joint_goal_pref_posterior.py` | 🟡 | (g,θ,z) 联合后验 |
| `composite_goal_compatibility.py` | 🟡 | 组合目标兼容度 |
| `compositional_goal_hypotheses.py` | 🟡 | 组合目标假设 |
| `compositional_goal_prior.py` | 🟡 | 组合目标先验 |
| `compositional_goal_bridge.py` | 🟡 | 桥接到主策略 |
| `goal_conditional_curriculum_hook.py` | 🟡 | 目标条件课程钩子 |

### 12.3 Profile / Curriculum 系列

| 模块 | 状态 | 说明 |
|------|------|------|
| `profile_state.py` | 🟡 | learner profile 持久化 |
| `profile_manager.py` | 🟡 | 跨 session profile 管理 |
| `profile_bootstrap.py` | 🟡 | profile → observer 初始化 |
| `preference_aware_policy_v2.py` | 🟡 | preference-conditioned 策略 |
| `macro_predictive_hook.py` | 🟡 | 宏观预测钩子 |
| `bayesian_macro_objective_shadow.py` | 🟡 Shadow | Bayesian 宏观目标推断 |

### 12.4 Robot Belief Over Agent（实验性）

| 模块 | 状态 | 说明 |
|------|------|------|
| `robot_belief_over_agent.py` | 🟡 | "belief over agent's belief" 结构 |
| `action_predictor.py` | 🟡 | 独立的 agent action 预测器 |

---

## 13. 文件索引

### 主线（✅ 成熟 / Frozen）

| 文件 | 行数 | 角色 |
|------|------|------|
| [intervention_policy.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/intervention_policy.py) | 455 | **核心决策**：Q-value 计算 + 最终选择 |
| [agent_predictor.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/agent_predictor.py) | 209 | Counterfactual surrogate rollout |
| [robot_belief.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/robot_belief.py) | 145 | Surrogate belief + predictor snapshot |
| [bottleneck_diagnosis.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/bottleneck_diagnosis.py) | 146 | 三维瓶颈分数 |
| [perceptual_model.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/perceptual_model.py) | 143 | 感知接入模型 |
| [intervention_semantics.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/intervention_semantics.py) | 222 | 干预语义 (WARN/UNLOCK/ITEM) |
| [interventions.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/interventions.py) | 168 | 干预类型定义 + inventory |
| [boredom_proxy.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/boredom_proxy.py) | 124 | Q_WAIT boredom penalty |
| [internalization_observer.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/internalization_observer.py) | 720 | 5D 内化状态观测器 (A0/A1/A2) |
| [gtet_factor_adapter.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/gtet_factor_adapter.py) | ~300 | GTET G_THETA factor 适配 |

### Shadow / 实验性（🟡）

| 文件 | 行数 | 角色 |
|------|------|------|
| [warning_utterance_policy.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/warning_utterance_policy.py) | 148 | Warning 子类型 (shadow-only) |
| [a1mt_observer_shadow_prob.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/a1mt_observer_shadow_prob.py) | ~500 | Probabilistic shadow observer |
| [micro_bayes_shadow_v3.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/micro_bayes_shadow_v3.py) | ~250 | MicroBayes shadow v3 |
| 其余 goal/pref/profile 系列 | | |
