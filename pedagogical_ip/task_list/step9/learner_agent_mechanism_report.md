# Learner Agent 机制全览报告

> 最后更新：Phase 2B 完成后（2026-04-07）
> 覆盖范围：`src/agents/` 全部模块 (37 个 .py 文件)

---

## 目录

1. [Agent 总体架构](#1-agent-总体架构)
2. [观测模型（成熟）](#2-观测模型成熟)
3. [Feature Belief 维护（成熟）](#3-feature-belief-维护成熟)
4. [Cost-Risk 预测头（成熟 / 三代）](#4-cost-risk-预测头成熟--三代)
5. [Bounded A* Planner（成熟）](#5-bounded-a-planner成熟)
6. [RSA Warning Channel（成熟 / Canonical）](#6-rsa-warning-channel成熟--canonical)
7. [Internalization State（成熟 / 5D）](#7-internalization-state成熟--5d)
8. [Transfer 机制（成熟 / Phase 2B）](#8-transfer-机制成熟--phase-2b)
9. [PredictorProtocol 接口（成熟 / Frozen）](#9-predictorprotocol-接口成熟--frozen)
10. [Shadow / 实验性模块](#10-shadow--实验性模块)
11. [残缺 / 未完成模块](#11-残缺--未完成模块)
12. [文件索引](#12-文件索引)

---

## 1. Agent 总体架构

```
 ┌────────────────────────────────────────────────────────────────────┐
 │                       每一步 step loop                             │
 │                                                                    │
 │  1. Observe:   observe_features_patch(pos, features, patch_r)      │
 │           ↓    4D noisy feature vector per cell                    │
 │  2. Believe:   FeatureBeliefMap.update(obs_mean, obs_var)          │
 │           ↓    Kalman posterior (mean, var)                        │
 │  3. Predict:   LatentCostRiskHead.predict_{cost,risk}(x_belief)   │
 │           ↓    ĉ, r̂ per cell                                     │
 │  4. Plan:      plan_next_action_v2(belief, predictor, planner)    │
 │           ↓    bounded A* → action, next_pos, path                │
 │  5. Execute:   move to next_pos, observe outcome                   │
 │  6. Learn:     predictor.update_from_outcome(x, cost, risk)       │
 └────────────────────────────────────────────────────────────────────┘
```

**核心约束：**
- Agent **不能看到** true feature vector → 只看到 noisy observation
- Agent **不能看到** WorldWeights → 必须从 (feature → cost/risk) 在线学习
- Agent **不能预见** 未来 → bounded A* 只展开有限节点

---

## 2. 观测模型（成熟）

> 源文件：[observation_model.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/observation_model.py)

### 2.1 V2 Feature Observation（Canonical）

函数 `observe_features_patch()`。Agent 在每步看到局部 patch 内 cell 的 **noisy 4D feature**：

```
对于 agent_pos 周围 patch_radius 内的每个非墙 cell:
    z_obs = clip(z_true + N(0, σ²_obs · I_d), 0, 1)
```

### 2.2 三级噪声模型

| 距离 | σ²_obs | 含义 |
|------|--------|------|
| d = 0 (self) | **0.01** | 近乎精确：站在上面几乎无噪声 |
| d = 1 (neighbor) | **0.08** | 模糊但信息量大 |
| d ≥ 2 (far) | **0.20** | 非常模糊 |

### 2.3 Patch Radius

| 值 | 含义 | 状态 |
|----|------|------|
| `patch_radius = 1` | 只看自己和 4 邻居（legacy 兼容模式）| |
| `patch_radius = 2` | 看 Manhattan 距离 ≤ 2 的 cell（**canonical default**）| ✅ |

> [!NOTE]
> `patch_radius=1` 时自动委托给 `observe_features()` 以保证和旧版完全相同的 RNG 调用顺序。

---

## 3. Feature Belief 维护（成熟）

> 源文件：[feature_belief.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/feature_belief.py)

### 3.1 Per-Cell Gaussian Belief

```
FeatureBeliefMap:
  mean[H, W, d]  — 后验均值
  var[H, W, d]   — 后验方差（对角协方差）
  memory[H, W]   — CellMemoryMeta 出处追踪
```

初始先验：
```
mean = 0.5,  var = 0.25  (uninformative)
```

### 3.2 Kalman 更新

每次观测到 cell (r,c) 的 noisy feature `z_obs`，方差 `σ²_obs`：

```
K = var / (var + σ²_obs)              (Kalman gain, per-dim)
mean_new = mean + K · (z_obs - mean)   (posterior mean)
var_new  = var · (1 - K)               (posterior variance)
```

### 3.3 干预引起的 Belief 更新

#### WARN（belief bias）

```
pseudo_obs = mean + α_warn · v̂_warn     (α_warn = 0.15)
pseudo_var = var · γ_conf                (γ_conf = 2.0, 较弱证据)
→ 然后执行一次 Kalman 更新 with (pseudo_obs, pseudo_var)
```

v̂_warn = normalized direction vector（默认为均等方向 = 1/√d）

#### UNLOCK（uncertainty reduction）

```
var_new = var · (1 - β_unlock)           (β_unlock = 0.5)
```

仅降不确定性，**不改 mean**。

### 3.4 CellMemoryMeta 出处追踪

每个 cell 记录：
```
CellMemoryMeta:
  ever_seen / seen_count / ever_traversed / traversed_count
  last_seen_t / last_traversed_t / best_view_quality
  reachable_since_t                   # UNLOCK 使可达的时间
  intervention_tags: {"warned", "unlocked", "item_affected"}
```

---

## 4. Cost-Risk 预测头（成熟 / 三代）

### 4.1 Generation 1: `LatentCostRiskHead`（4D Linear, Canonical Default）

> 源文件：[cost_risk_model.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/cost_risk_model.py)

#### Cost Head（BayesianCostHead）

```
ĉ(x) = max(w_c · x + b_c, 0.1)

update:  w_c ← w_c − lr · [−error·x + w_c/σ²_prior]   (MAP with Gaussian prior)
         b_c ← b_c − lr · (−error)
         gradient clipping: |grad| ≤ 5.0
         weight clamping: |w| ≤ 10.0
```

| 参数 | 默认值 |
|------|--------|
| d | 4 |
| prior: w | N(0, σ²_prior·I), σ²_prior=1.0 |
| prior: b | 1.0 |
| lr | 0.1 |

#### Risk Head（BayesianRiskHead）

```
r̂(x) = σ(w_r · x + b_r)

update (oracle_visited):  同 cost head 但用 binary cross-entropy gradient
                          grad_w = −(y − r̂) · x + w_r/σ²_prior
update (binary_outcome):  用 0/1 hazard outcome
```

| 参数 | 默认值 |
|------|--------|
| lr | 0.3 |
| prior: w | N(0, 1.0·I) |
| prior: b | 0.0 → sigmoid(0)=0.5 |

#### Uncertainty 估计

```
u_cost(x) = x^T · H_cost^{−1} · x     (posterior Hessian inverse)
u_risk(x) = x^T · H_risk^{−1} · x
```

其中 `H = (1/N)·ΣxxT + I/σ²_prior`

### 4.2 Generation 2: `StructuredBasisCostRiskHead`（Phase 2A）

> 源文件：[structured_basis_head.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/structured_basis_head.py)

#### Basis Expansion

```
Cost basis:  φ_c(z) = [1, z₀, z₁, z₂+z₃, z₀z₁, (z₂+z₃)²]     → 6D
Risk basis:  φ_r(z) = [1, z₂, z₃, z₂z₃, |z₂−z₃|, z₁z₂, z₁z₃]  → 7D
```

Predictions：
```
ĉ(x) = max(w_c · φ_c(x) + b_c, 0.1)
r̂(x) = σ(w_r · φ_r(x) + b_r)
```

#### Jacobian Uncertainty（Phase 2B 升级）

```
u_cost(Σ_z) ≈ σ'(η_c) · √(g_c^T Σ_z g_c)
u_risk(Σ_z) ≈ σ'(η_r) · √(g_r^T Σ_z g_r)

where g = ∂φ/∂z (Jacobian of basis expansion)
```

替代了旧的 `w[:4]²` 粗暴代理。

#### 性能对比

| 场景 | Linear | Basis |
|------|--------|-------|
| baseline_v2 | 0.400 | **1.000** |
| harder_baseline_v2 | 0.540 | **0.960** |
| GTET | 1.000 | 1.000（无差异）|
| DTMB | 1.000 | 1.000（无差异）|

> [!IMPORTANT]
> Basis 的优势 **仅限于 feature-driven 场景**（baseline 系列）。在拓扑主导的 GTET/DTMB 上，Linear 和 Basis 表现相同。

### 4.3 Generation 3: `GenericSlowFastPredictor`（Phase 2B Transfer Wrapper）

> 源文件：[slow_fast_head.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/slow_fast_head.py)

#### 架构

```
GenericSlowFastPredictor:
  base_factory: callable → PredictorProtocol   (任意头类型)
  slow: PredictorProtocol                      (跨 episode 持久)
  fast: PredictorProtocol                      (每 episode 重置)
  alpha: float                                 (EMA 系数)
```

#### Episode 生命周期

```
begin_episode():
    fast = copy(slow)                   # 从慢头克隆到快头

predict_*(x):
    return fast.predict_*(x)            # 总是用快头预测

update_from_outcome(...):
    fast.update_from_outcome(...)       # 只训练快头

end_episode():
    θ_slow_new = (1−α)·θ_slow + α·θ_fast    # EMA 更新慢头
```

#### θ 分量提取（维度无关）

```
extract_theta_components(predictor) → {
    "cost_w", "cost_b", "risk_w", "risk_b"
}
```

支持 Linear（4D）和 Basis（6D/7D）统一提取。

#### α-Sweep 结果

| 配置 | α=0.1 | α=0.2 | α=0.3 | α=0.5 |
|------|-------|-------|-------|-------|
| basis_slowfast (harder) | 0.980 | **1.000** | 1.000 | 1.000 |
| basis_fresh (harder) | 0.960 | — | — | — |
| basis_persist (harder) | 0.680 | — | — | — |

**推荐 α = 0.2**（最小有效值）。

---

## 5. Bounded A* Planner（成熟）

> 源文件：[planner_astar.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/planner_astar.py)

### 5.1 核心 A* 算法

```
_astar_core(start, goal, cost_fn, H, W, budget, passable_mask)
```

- 最多展开 `budget` 个节点（bounded）
- 启发式：Manhattan 距离
- 若 budget 耗尽：返回离目标最近的 partial path

### 5.2 Search Budget 采样

```
budget_class ∈ {4, 8, 16}
candidates: {4→[3,4,5], 8→[6,8,12], 16→[14,16,20]}
P(budget) = [0.25, 0.50, 0.25]   (peaked at center)
```

### 5.3 V2 Latent 路径代价函数（Canonical）

```
J(i) = λ_c · ĉ_i
     + risk_penalty(r̂_i, learning_factor, necessity)
     + λ_uc · (1−n) · u_c_i
     + λ_ur · (1−n) · u_r_i
     + warned_extra_cost[i]              (if warned)
```

#### Risk Penalty 公式

```
φ(r̂) = −ln(1 − clip(r̂, ε, 1−ε))           (survival transform)

learning_factor = min(1, n_updates/10)       (0=untrained, 1=trained)
necessity_discount = 1 − route_necessity

risk_penalty = φ(r̂) · λ_r
             × [learning_factor + (1−learning_factor) · necessity_discount]
             × (1−γ_shield)                   (if has shield)
```

> [!NOTE]
> 设计原则：**unknown ≠ dangerous**。当 predictor 未训练时（`learning_factor≈0`），risk penalty 被 necessity 折扣。高 necessity（无安全替代路线）→ 不罚不确定性。

#### 默认 Planner 权重

| 参数 | 含义 | 默认值 | 状态 |
|------|------|--------|------|
| λ_c | cost weight | 1.0 | ✅ Frozen |
| λ_r | risk weight | 3.0~5.0 | ✅ |
| λ_uc | cost uncertainty weight | 0.1 | ✅ |
| λ_ur | risk uncertainty weight | 0.1 | ✅ |

### 5.4 Path-Level Candidate Scoring

`plan_with_alternatives_v2()` 为所有一步邻居计算完整路径得分：

```
对于每个合法 first-step neighbor n:
    sub_path = A*(n → goal, budget=budget/2)
    score[n] = cost_fn(n) + Σ cost_fn(sub_path[i])
```

→ 产出 `candidate_scores: dict[action → total_cost]`，用于 confidence 和 failure mode 估计。

### 5.5 Belief Plan 结构

```
BeliefPlan:
  action, next_pos, planned_prefix, full_path
  expected_cost, expected_risk, uncertainty
  runner_up_gap        = best−2nd 间的差
  action_confidence    = gap / (gap + temperature)
  dominant_reason      ∈ {lower_risk, lower_cost, lower_uncertainty, deadline_pressure, mixed}
```

---

## 6. RSA Warning Channel（成熟 / Canonical）

> 源文件：[rsa_warning_channel.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/rsa_warning_channel.py)

### 6.1 假设空间（4-way segment-level）

```
R = {LEFT_RISKY, RIGHT_RISKY, BOTH_SAFE, HAZARD_AHEAD}
```

### 6.2 话语库（4-way）

```
U = {WARN_LEFT, WARN_RIGHT, WARN_AHEAD, GENERIC_WARN}
C(u) = {0, 0, 0.2, 0.5}   (specific=free, generic=cheap)
```

### 6.3 L0 Literal Listener

```
L0(r|u,c) ∝ exp(λ_sem · match(u,r,c)) · P(r|c)
```

`match(u,r,c) ∈ [0,1]`：side alignment × specificity × branch availability

### 6.4 S1 Pragmatic Speaker

```
S1(u|r,c) ∝ exp(α_RSA · [ln L0(r|u,c) − λ_C · C(u)])
```

### 6.5 Agent Belief Update（三种变体）

| 变体 | 公式 | 状态 |
|------|------|------|
| `l0` | b⁺(r) ∝ L0(r\|u,c) · b⁻(r) | 🟡 Ablation |
| **`s1`** | b⁺(r) ∝ S1(u\|r,c) · b⁻(r) | ✅ **Canonical** (`rsa_obs_s1`) |
| `s1_trust` | b⁺(r) ∝ [S1(u\|r,c)]^η_τ · b⁻(r)，η_τ=clip(τ̂, 0.3, 2.0) | 🟡 Ablation |

### 6.6 Planner Adapter

```
Δρ = E_{r~b⁺}[ρ(r)] − E_{r~uniform}[ρ(r)]    (risk delta)
cell_penalty[(r,c)] = Δρ · λ_lane_warn          (λ_lane_warn = 5.0)
```

### 6.7 超参数

| 参数 | 含义 | 默认 | 状态 |
|------|------|------|------|
| λ_sem | semantic matching sharpness | 3.0 | ✅ |
| α_RSA | speaker rationality | 2.0 | ✅ |
| λ_C | utterance cost weight | 1.0 | ✅ |

---

## 7. Internalization State（成熟 / 5D）

> 源文件：[internalization_state_v3.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/internalization_state_v3.py)

### 7.1 状态向量

```
m_t = (κ, τ, ν, γ_spec, γ_gen)
```

| 维度 | 符号 | 含义 | 范围 | 初始值 |
|------|------|------|------|--------|
| 风险敏感度 | κ | 校准 risk penalty 的强度 | [0.3, 3.0] | 1.0 |
| 信任 | τ | 对 tutor 建议的信任水平 | [0, 1] | 0.3 |
| 依赖 | ν | 对 tutor 的过度依赖 | [0, 0.8] | 0.1 |
| 诱惑特异性 | γ_spec | 对诱惑分支的抑制 | [0, 0.7] | 0.0 |
| 泛化抑制 | γ_gen | 对新颖探索的抑制 | [0, 0.5] | 0.0 |

### 7.2 更新规则

#### κ — Risk Sensitivity（回归基线）

```
κ_new = (1−β_κ)·κ + β_κ·κ₀ + α_κ·(risk_real − risk_expected)
```

| 参数 | 值 |
|------|---|
| κ₀ | 1.0 |
| β_κ | 0.08 (mean reversion) |
| α_κ | 0.40 (shock multiplier) |

#### τ — Trust

```
if warn_helpful: τ += α⁺_τ · (1−τ)       (α⁺_τ = 0.25)
if warn_bad:     τ −= α⁻_τ · τ           (α⁻_τ = 0.12)
```

#### ν — Dependence

```
if blind_obey:     ν += α⁺_ν · (1−ν)     (α⁺_ν = 0.20)
if self_discovery: ν −= α⁻_ν · ν         (α⁻_ν = 0.15)
```

#### γ_spec — Temptation Suppression

```
if tempt_error:        γ_spec += α⁺_gs · (1−γ_spec)  (α⁺_gs = 0.22)
if false_suppression:  γ_spec −= α⁻_gs · γ_spec      (α⁻_gs = 0.10)
```

#### γ_gen — Generalization Suppression

```
if sustained_pressure:     γ_gen += α⁺_gg · (1−γ_gen)  (α⁺_gg = 0.08)
if successful_exploration: γ_gen −= α⁻_gg · γ_gen       (α⁻_gg = 0.12)
```

### 7.3 Branch Utility 公式

```
U(branch, θ, m) = λ_θ · R_pref(branch, θ)
                − κ² · risk_penalty(branch)
                − γ_spec · temptation_score
                − γ_gen · (0.3 if novel else 0)
                + τ · warn_bonus
                − ν · (0.2 if warned else 0)
```

### 7.4 B2 Epistemic Risk Shaping（可选扩展）

当 `use_epistemic_risk=True`：
```
ũ_r = clip(risk_unc / u_ref, 0, 1)           (u_ref = 0.5)
α = α_min + (1−α_min)·exp(−γ_epi·ũ_r)       (α_min=0.25, γ_epi=3.0)
risk_term = κ² · [ρ + (1−ρ)·α] · risk_penalty  (ρ=0.35 risk floor)
```

### 7.5 Softmax Choice

```
P(branch=i) = (1−ε)·softmax(β·U_i) + ε·uniform
```

| 参数 | 含义 | 典型值 |
|------|------|--------|
| β | inverse temperature | 3.0~5.0 |
| ε | exploration rate | 0.05~0.1 |

---

## 8. Transfer 机制（成熟 / Phase 2B）

### 8.1 每个 Episode 的 Predictor 生命周期

```
三种模式：
  fresh:   每 episode 全新初始化 → 无 transfer
  persist: 不 reset，继续上一 episode 权重 → 有 contamination
  slowfast: begin_episode → fast=copy(slow); end → EMA 更新 slow
```

### 8.2 GenericSlowFast 的 EMA 更新

```
θ_slow_new = (1−α)·θ_slow + α·θ_fast
```

通过 `extract_theta_components()` 实现维度无关的逐分量更新。

### 8.3 实验结论

| 配置 | harder_baseline_v2 survival |
|------|-----------------------------|
| linear_fresh | 0.540 |
| linear_persist | 0.760 |
| basis_fresh | 0.960 |
| **basis_slowfast_0.2** | **1.000** |
| basis_persist | 0.680（contamination 导致退化）|

---

## 9. PredictorProtocol 接口（成熟 / Frozen）

> 源文件：[predictor_protocol.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/predictor_protocol.py)

### 9.1 必须实现的方法

```python
class PredictorProtocol:
    def predict_cost(self, x: np.ndarray) -> float
    def predict_risk(self, x: np.ndarray) -> float
    def predict_cost_uncertainty(self, x: np.ndarray) -> float
    def predict_risk_uncertainty(self, x: np.ndarray) -> float
    def predict_cost_uncertainty_from_var(self, x_var: np.ndarray) -> float
    def predict_risk_uncertainty_from_var(self, x_var: np.ndarray) -> float
    def update_from_outcome(self, x, cost_label, risk_label, weight) -> None
```

### 9.2 工具函数

| 函数 | 用途 |
|------|------|
| `snapshot_predictor(p)` | deepcopy for read-only use |
| `restore_predictor(p, snap)` | restore from snapshot |
| `extract_theta(p)` | flat weight vector |
| `extract_theta_components(p)` | structured {cost_w, cost_b, risk_w, risk_b} |

---

## 10. Shadow / 实验性模块

### 10.1 Shadow 模块

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| Continuous Reward Shadow | `continuous_reward_shadow.py` | 🟡 Shadow | 连续 reward 信号诊断 |
| Planner Risk Shadow | `planner_risk_shadow.py` | 🟡 Shadow | planner 的 risk 消融诊断 |

### 10.2 实验性扩展

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| B2 Epistemic Risk | `internalization_state_v3.py` | 🟡 Optional | `use_epistemic_risk` flag |
| B2 Epistemic Bonus | `internalization_state_v3.py` | 🟡 Optional | `use_epistemic_bonus` flag |
| Route Necessity | `route_necessity.py` | ✅ | 计算 n ∈ [0,1]，已集成至 cell_cost_v2_latent |
| Necessity Gate Variants | `necessity_gate_variants.py` | 🟡 Ablation | 不同 gate function 的变种 |

### 10.3 Goal/Preference 后验模块

| 模块 | 状态 | 说明 |
|------|------|------|
| `goal_posterior_v1.py` | 🟡 | 简单 goal 后验 |
| `goal_factor_posterior.py` | 🟡 | G_THETA 因子化后验 |
| `joint_posterior_v2.py` | 🟡 | (g,θ) joint 后验 |
| `preference_posterior.py` / `v2` | 🟡 | θ 偏好后验 |

---

## 11. 残缺 / 未完成模块

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| `bounded_agent.py` | legacy agent wrapper | ⚪ Legacy | 旧版 agent，不在 V2 主线使用 |
| `belief.py` | legacy belief system | ⚪ Legacy | V0 scalar belief，被 FeatureBeliefMap 替代 |
| `agent_belief_state.py` | | ⚪ Legacy | |
| `world_state.py` | | ⚪ Legacy | |
| `internalization_agent.py` | | ⚪ Legacy | 早期 internalization agent |
| `familiarity.py` | | 🔲 实验性 | 场景熟悉度追踪 |
| `branch_concepts.py` | | 🟡 | 分支语义概念化 |
| `branch_scorer_probe.py` | | 🟡 | 分支打分 probe |
| `branch_summary.py` | | 🟡 | 分支信息聚合 |
| `behavior_bridge.py` | | 🟡 | 行为桥接 |
| `behavior_probes.py` | | 🟡 | 行为 probe（VA/IA/EP）|
| `trainable_bridge.py` | | 🟡 | 可训练的行为桥接 |
| `warning_update.py` | | 🟡 Legacy | 旧版 warning（pseudo-label + lane bias）|
| `warning_belief_delta.py` | | ✅ | WarningBeliefDelta 数据结构 |
| `stochastic_agent_policy.py` | | ✅ | softmax 分支选择 |
| `prefix_prediction.py` | | ✅ | 前缀预测 |

---

## 12. 文件索引

### 主线（✅ 成熟 / Frozen）

| 文件 | 行数 | 角色 |
|------|------|------|
| [observation_model.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/observation_model.py) | 229 | **观测模型**：noisy feature observation |
| [feature_belief.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/feature_belief.py) | 202 | **Belief 维护**：Kalman 更新 + CellMemory |
| [cost_risk_model.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/cost_risk_model.py) | 239 | **预测头 Gen1**：4D linear Bayesian |
| [structured_basis_head.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/structured_basis_head.py) | 393 | **预测头 Gen2**：6D/7D basis + Jacobian |
| [slow_fast_head.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/slow_fast_head.py) | ~210 | **预测头 Gen3**：dual-timescale wrapper |
| [predictor_protocol.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/predictor_protocol.py) | ~160 | **接口协议**：统一 head 接口 |
| [planner_astar.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/planner_astar.py) | 525 | **Planner**：bounded A* (V0+V2) |
| [belief_planning.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/belief_planning.py) | 291 | **Belief Plan**：结构化规划结果 |
| [rsa_warning_channel.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/rsa_warning_channel.py) | 631 | **RSA Warning**：L0/S1/S1_trust belief update |
| [internalization_state_v3.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/internalization_state_v3.py) | 217 | **5D 内化状态**：κ/τ/ν/γ_spec/γ_gen |
| [route_necessity.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/route_necessity.py) | ~110 | **路线必要性**：n ∈ [0,1] |
