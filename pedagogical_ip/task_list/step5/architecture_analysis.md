# Architecture Analysis: 超参数、冗余、复杂性与缺失模块

> 基于 Stage 5 全部源码的深度分析。
> 覆盖四个维度：超参数审计、冗余结构、过度复杂机制、缺失功能。

---

# 一、超参数全景审计

## 1.1 超参数总量

当前系统包含 **约 120 个显式超参数**，分布如下：

| 模块 | 参数数量 | 关键文件 |
|------|:-------:|---------|
| Agent 内化状态 (5D) | ~18 | `internalization_state_v3.py` |
| Agent 选择模型 | 3 | `stochastic_agent_policy.py` |
| Bayesian 风险/成本模型 | ~10 | `cost_risk_model.py` |
| 行为探针 + 区间 | ~15 | `behavior_probes.py` |
| 行为桥接 (Bridge) | ~70 权重 | `behavior_bridge.py` |
| Observer 3D (A0) | ~15 | `internalization_observer.py` |
| Observer γ̂_spec / κ̂ | ~10 | `internalization_observer.py` |
| Micro Tutor (BCICTv4) | ~10 | `internalization_control_tutor_v4.py` |
| Macro Controller (v13) | ~45 | `curriculum_controller_v13.py` |

## 1.2 Agent 内化状态参数

| 参数 | 值 | 作用 | 简化空间 |
|------|:--:|------|:--------:|
| κ_0 | 1.0 | 风险敏感度基线 | ✅ 可与 observer 统一 |
| β_κ | 0.08 | 回归速度 | ✅ 可合并为统一 λ |
| α_κ | 0.40 | 风险误差驱动强度 | 保持 |
| α_τ⁺ / α_τ⁻ | 0.25 / 0.12 | 信任增/减速率 | ⚠️ 可考虑统一为单一 α_τ + 方向系数 |
| α_ν⁺ / α_ν⁻ | 0.20 / 0.15 | 依赖增/减速率 | ⚠️ 同上 |
| α_gs⁺ / α_gs⁻ | 0.22 / 0.10 | 特异抑制增/减 | ⚠️ 同上 |
| α_gg⁺ / α_gg⁻ | 0.08 / 0.12 | 一般抑制增/减 | ⚠️ 同上 |
| ν_max | 0.8 | 依赖上限 | 保持 |
| γ_spec_max / γ_gen_max | 0.7 / 0.5 | 抑制上限 | 保持 |

**简化建议**：

> 目前每个维度都有独立的 α⁺ 和 α⁻，共 8 个 α 参数。可以简化为：每维一个基础学习率 α_d，加一个全局不对称比 r_asym（如 r=0.6 表示下降速率 = 0.6 × 上升速率）。这样 8 个参数 → 4+1 = 5 个参数，保持不对称性但减少自由度。

## 1.3 Observer 参数

| 参数 | A0 | A1 (Frozen) | 作用 |
|------|:--:|:-----------:|------|
| α_τ⁺/⁻ | 0.22/0.10 | 同 | 比 agent 略保守 |
| α_ν⁺/⁻ | 0.18/0.13 | 同 | 比 agent 略保守 |
| α_γ⁺/⁻ | 0.07/0.10 | 同 | 比 agent 略保守 |
| β_τ_probe | 0.15 | **0.0** | A1 关闭 |
| β_ν_probe | 0.10 | **0.0** | A1 关闭 |
| β_γ_probe | 0.10 | **0.0** | A1 关闭 |
| λ_τ/ν/γ | 0.02 | **0.005** | 均值回归 |
| α_gs_resist/follow | — | 0.03/0.025 | γ̂_spec |
| λ_κ, α_κ⁺/⁻, κ_0 | — | 0.02/0.015/0.012/0.3 | κ̂ |

**关键观察**：Observer 的 α 参数是 agent 参数的 85–90%。这不是独立校准的结果，而是一个经验规则：「observer 比 agent 保守一点」。

**简化建议**：

> 引入全局 damping 系数 d_obs (≈ 0.87)，observer 参数直接计算为 agent 参数 × d_obs。这样 observer 的 6 个 α 参数从独立变量变成 1 个系数自动推导。

## 1.4 Micro Tutor 参数

| 参数 | 值 | 作用 | 必要性 |
|------|:--:|------|:------:|
| λ_teach | 3.5 | Teaching value 权重 | ✅ 核心 |
| λ_over | 4.0 | Overteach penalty 权重 | ✅ 核心 |
| λ_sd | 1.5 | Self-discovery bonus | ⚠️ 可合并 |
| λ_dep | 2.0 | Blind-obey penalty | ⚠️ 可合并 |

**简化建议**：

> λ_sd 和 λ_dep 的作用分别是奖励自发发现和惩罚盲从。它们的比值 (1.5 / 2.0 = 0.75) 表达了系统对「盲从比自发更有害」的判断。可以简化为一个参数 λ_autonomy = 2.0 和一个 discovery-penalty 比 r_dp = 0.75。

## 1.5 Macro Controller 参数 (v13.3)

最大的参数池在宏观控制器中（~45 个）：

| 类别 | 参数数量 | 示例 |
|------|:-------:|------|
| STOP per-θ | 10 | eps_0_safe, a_s_nu_safe, eps_0_shiny, ... |
| Gated STOP | 5 | min_teach_safe/shiny, plateau_window, plateau_tau_safe/shiny |
| EVAL | 4 | lambda_info, lambda_var_gain, c_eval, max_eval |
| 风险约束 | 5 | eta_otr_0, eta_nu_0, beta_pessimism, ... |
| 排名/增益 | 6 | w_gain, lambda_unc_base, tau_n, tau_B, lambda_fid, lambda_rep |
| 家族先验 | ~6 | 饱和衰减 τ_fam_safe/shiny + per-θ per-family 先验 |
| Replay | 3 | replay_top_k, w_C, w_otr |
| 消融/研究 | ~6 | EIG (eig_beta/c_gain/c_mastery/mix), ZPD (alpha_zpd/beta_zpd) |

**简化建议**：

> 1. **per-θ STOP** (10 params) 可重写为基线参数 + θ-conditional 缩放因子，减少到 5+1 = 6 参数
> 2. **消融标志** (use_eig_uncertainty, use_zpd_feature 等) 及其关联参数（~12个）在 canonical 中全部 OFF，建议迁移到独立的 research config
> 3. **Legacy 参数** (eps_0, a_s_nu, b_s_gg 等旧 shared 系数) 仍存在于 config 中但已被 per-θ 版本取代。应当删除

## 1.6 Behavior Bridge 权重

`BRIDGE_WEIGHTS` 包含 5 个 probe × 13 维 = **65 个固定权重**。这些权重是从 BI-ICT-v3 rollout 拟合的，不是在线学习的。

**简化空间**：

> 65 个权重中有很多零值（约 30 个）。可以重构为稀疏表示，只存储非零权重。但更根本的问题是：这些权重是否应该在线学习（adaptive bridge）而不是预先固定？这是一个研究问题，不是简化问题。

---

# 二、冗余结构分析

## 2.1 已确认的冗余

| 冗余项 | 证据 | 当前状态 |
|--------|------|:--------:|
| SOFT 动作 (dose=0.5) | Exact-Q: V_SOFT ≈ 0 | 已移出 canonical，保留为 legacy |
| close-gap 机制 | 消融确认 dead code | 已从 v13 删除 |
| A0 probe 校正 (β_probe) | A1 关闭后性能不降 | 已设为 0 |
| Legacy shared STOP 参数 | 被 per-θ 版本取代 | ⚠️ 仍在 config 中，应删除 |

## 2.2 可能的冗余

### 2.2.1 InternalizationDynamicsV2 文件

`internalization_dynamics_v2.py` 定义了一个 3D 状态 `InternalizationStateV2(κ, η, γ)`。这与当前 canonical 的 `FactoredInternalizationState(κ, τ, ν, γ_spec, γ_gen)` 完全不同。

**判断**：这是早期版本的 internalization state，已经被 v3 完全取代。**建议归档或删除**。

### 2.2.2 两套 Behavior Loss 实现

- `behavior_probes.py` 中的 `behavior_loss()` — 使用原始 5 个探针函数直接评估
- `behavior_bridge.py` 中的 `bridge_behavior_loss()` — 使用预拟合权重的桥接预测

BCICTv4 默认使用 bridge 版本 (`use_bridge=True`)。原始版本仅在 `use_bridge=False` 时激活。

**判断**：原始版本实际上是一个 fallback。如果 bridge 已经是 canonical，原始版本可以标记为 `@deprecated`。但保留作为消融比较有价值。

### 2.2.3 两套 Zone 定义

- `behavior_probes.py` → `BEHAVIOR_ZONES`（硬编码，区分 safe/shiny）
- `behavior_bridge.py` → `EmpiricalZoneCalibrator`（在线校准）

BCICTv4 默认使用 calibrator (`use_calibrated_zones=True`)，但在数据不足时 fall back 到硬编码 zones。

**判断**：两套共存是合理的（cold start 需要 fallback）。但建议将硬编码 zones 明确标记为 prior。

### 2.2.4 BEHAVIOR_WEIGHTS 定义重复

- `behavior_probes.py:102` → `{RC: 1.0, TR: 1.2, EP: 2.0, VA: 1.5, IA: 2.5}`
- `behavior_bridge.py:98` → `{RC: 1.0, TR: 1.2, EP: 2.5, VA: 1.5, IA: 2.5}`

EP 的权重不一致（2.0 vs 2.5）。

**判断**：这是一个 bug 或未同步的更新。应统一到一处定义。

### 2.2.5 Branch Scorer Probe

`BranchScorerProbe` 被明确标注为 "NOT connected to planner — pure diagnostic"。它在每次实验中训练和评估，但结果不影响任何决策。

**判断**：如果长期不打算接入 planner，建议迁移到 `src/evals/` 或 `scripts/` 目录，从 agent 核心路径中移除。

## 2.3 公式简化机会

### 2.3.1 Factored Utility 简化

当前：

$$U = \lambda_\theta R_{pref} - \kappa^2 \rho - \gamma^{spec} \cdot tempt - \gamma^{gen} \cdot 0.3 \cdot \mathbf{1}[novel] + \tau \cdot warn - \nu \cdot 0.2 \cdot \mathbf{1}[warned]$$

观察：
- `γ_gen · 0.3 · 1[novel]` 中的 0.3 是硬编码系数，语义不清
- `ν · 0.2 · 1[warned]` 中的 0.2 也是硬编码

**建议简化**：将 0.3 和 0.2 提取为 FactoredInternalizationState 的参数（`novel_sensitivity=0.3`, `obey_cost=0.2`），或者直接用 γ_gen 和 ν 的尺度来吸收这些系数。

### 2.3.2 Observer 更新的统一形式

τ̂、ν̂、γ̂_gen 的更新式有相同的数学结构：

$$\hat{x}_{t+1} = (1-\lambda)\hat{x}_t + \lambda x_0 + \alpha^+ e^+ (x_{max} - \hat{x}_t) - \alpha^- e^- \hat{x}_t$$

这可以重构为一个通用函数：

```python
def bounded_ema_update(x, x_0, lam, alpha_up, alpha_down,
                       e_up, e_down, x_min, x_max):
    x_new = (1-lam)*x + lam*x_0 + alpha_up*e_up*(x_max-x) - alpha_down*e_down*x
    return clip(x_new, x_min, x_max)
```

目前每个维度的更新都是独立实现的，有 ~30 行重复代码。

### 2.3.3 Macro Score 公式

当前 lesson score：

$$J = w_{gain} \cdot G_{mean} + \lambda_{unc} \cdot \sqrt{var} - \lambda_{fid} \cdot 0.08 - \lambda_{rep} \cdot \frac{count}{5} + zpd + fam\_bonus$$

其中 `λ_fid · 0.08` 是一个常数项（不依赖于 lesson），对排名没有影响。

**判断**：`r_fid = λ_fid · 0.08` 可以直接移除，因为它是所有 lesson 共享的常数偏移，不影响 argmax。

---

# 三、复杂性分析

## 3.1 过度复杂的模块

### 3.1.1 CurriculumControllerV13 (635 行)

这是整个系统中最复杂的单个文件：

- 6 层决策流水线（Feasibility → Risk → Score → STOP → EVAL → TEACH）
- 2 种消融模式 (EIG, ZPD) 各有独立参数和逻辑分支
- Active counterfactual replay 嵌入在 update_response 中
- Actionability audit 嵌入在控制器中

**建议拆分**：

| 提取模块 | 功能 | 预估行数 |
|---------|------|:-------:|
| `stop_decision.py` | Gated STOP (warm-up + plateau + margin) | ~80 |
| `eval_decision.py` | EVAL value computation | ~40 |
| `risk_filter.py` | Risk constraint filtering | ~50 |
| `replay_engine.py` | Active counterfactual replay | ~60 |
| `audit.py` | Actionability audit + diagnostics | ~50 |

这样 v13 核心缩减到 ~300 行，职责清晰。

### 3.1.2 Observer (685 行)

`internalization_observer.py` 包含 4 个类（A0, A1, A1Frozen, A2），每个类都重写了 update/get_estimate 方法。

**建议**：使用组合模式而非继承。将 γ̂_spec 更新和 κ̂ 更新提取为可插拔的 `DimensionUpdater` 组件，observer 只负责协调。

### 3.1.3 PairwiseResponseModel

`pairwise_response_model.py` (8853 bytes) 同时处理 gain 预测、harm 预测、pairwise replay，以及 hierarchical/residual 分解。

**建议**：至少将 hierarchical empirical Bayes 和 residual 部分拆分为独立的 sub-model。

## 3.2 适当复杂的模块（不建议简化）

| 模块 | 行数 | 判断 |
|------|:----:|------|
| BCICTv4 | 297 | 合理——Q 函数 + dose control + shadow refactors |
| BranchConceptLibrary | 147 | 合理——3 个概念 + KL scoring + Welford 更新 |
| FactoredInternalizationState | 217 | 合理——5 个维度各有独立更新逻辑 |
| LatentCostRiskHead | 237 | 合理——两个 Bayesian 线性头 + uncertainty |

---

# 四、缺失模块与未来方向

## 4.1 明确缺失的功能

### 4.1.1 Persistent Profile（持久化学习者画像）

**当前状态**：每个 session 独立，m_t 不跨 session 保存。

**需要实现**：
- 跨 session 的 m 持久化存储
- Session 间 m_t 初始化：使用上次 session 的终态 m_T
- 长期 learner profile 漂移检测
- 5D observer 的跨 session 校准（κ̂ 和 γ̂_spec 的长期趋势）

**优先级**：🔴 高——这是 5D 架构的核心下游应用

### 4.1.2 Compositional Goals（组合目标）

**当前状态**：CGC-v2 (Compositional Goal Corridor) 已有基本框架，但只支持 coupled posterior。

**需要实现**：
- Goal + Preference + Temptation + Risk Calibration 四因素共存
- Goal-conditional curriculum selection
- 多因素后验推理（当前只有 θ 后验）

**优先级**：🟡 中——需要 persistent profile 作为前置

### 4.1.3 POMDP Solver（可选）

**当前状态**：系统明确定义为「非 RL、非 exact POMDP」，使用 constrained greedy planning。

**如果要实现**：
- Belief-space MDP 建模
- Point-based value iteration (PBVI) 或 POMCP
- State: (q_θ, m̂, u, h, B)
- Action: {TEACH_ℓ, EVAL, STOP}

**优先级**：🔵 低——当前 greedy planning 已经足够好，POMDP 的主要价值在于理论对比

### 4.1.4 Richer Intervention Semantics

**当前状态**：Micro 只有 {WAIT, WARN}（2-act canonical）。

**可扩展**：
- **Graduated Warning**：不是 dose 0/1，而是语义化的 "hint" / "explain" / "alert" / "directive"
- **UNLOCK**：已有 CellType.LOCKED_DOOR 但 UNLOCK action 未在 BCICTv4 中实现
- **ITEM_DROP**：Stage 4 定义了 action space 但 BCICTv4 未实现
- **信息性干预**：提供部分信息而非警告（"这个区域有隐藏风险"）

**优先级**：🟡 中——需要在 2-act baseline 稳定后才扩展

### 4.1.5 多 Learner Type 后验

**当前状态**：q_t(θ) 只在 {safe, shiny} 上定义。`PREF_REWARD` 还定义了 risky, shortcut, neutral，但未被 canonical 使用。

**需要实现**：
- 支持 >2 learner types 的后验更新
- 验证 per-θ 机制在 K>2 时的可扩展性
- 可能需要粒子滤波替代 exact posterior

**优先级**：🟡 中

### 4.1.6 Non-Stationary Learner

**当前状态**：θ 在 session 内固定。

**现实中**：学习者的偏好可能随时间变化（shiny → safe 的成熟过程）。

**需要实现**：
- θ 的缓慢漂移模型
- 后验更新中的遗忘因子
- 用 m_t 的变化来检测 θ 漂移

**优先级**：🔵 低——需要 persistent profile 先完成

## 4.2 已有框架但未完成的功能

| 功能 | 已有代码 | 缺失部分 |
|------|:--------:|---------|
| CGC-v2 场景 | `cgc_v2_family.py` | 未接入 5D observer 和 κ̂ |
| Epistemic Risk (B2) | `compute_factored_utility()` 中 | 默认 OFF，未做过 canonical 评估 |
| EIG Exploration (B1) | `_lambda_unc_hybrid()` 中 | 默认 OFF，参数未校准 |
| ZPD Feature (B3) | `_zpd_adjustment()` 中 | 默认 OFF，参数未校准 |
| Belief-Horizon p_self | BCICTv4 shadow | 默认 OFF，未验证 |
| EPU Shadow | BCICTv4 shadow | 默认 OFF，纯对比 |
| EIG Observation | BCICTv4 shadow | 默认 OFF，纯对比 |
| Branch Scorer | `branch_scorer_probe.py` | 诊断用途，未接入 planner |

## 4.3 建议的研究路线图

```
Phase 1 (当前)
─── 5D Canonical 已完成 ✅
└── κ̂ macro bonus 已验证 ✅

Phase 2 (推荐下一步)
├── Persistent Profile ← m_t 跨 session
├── γ̂_spec macro utility ← 需要 persistent profile 的数据
└── 代码重构：拆分 v13, 清理 legacy

Phase 3 (中期)
├── Compositional Goals + CGC-v2 接入 5D
├── Richer intervention ({hint, explain, alert, directive})
├── Multi-type posterior (K > 2)
└── B1/B2/B3 canonical 评估

Phase 4 (远期)
├── Non-stationary learner (θ drift)
├── POMDP baseline comparison
├── Multi-learner classroom
└── Real human deployment interface
```

---

# 五、总结：最高优先级 Action Items

| 项目 | 类型 | 估计工作量 | 理由 |
|------|:----:|:----------:|------|
| 统一 observer α 参数为 damping 系数 | 简化 | 小 | 减少 6 个自由参数 |
| 统一 BEHAVIOR_WEIGHTS（修复 EP 2.0 vs 2.5） | Bug | 极小 | 消除不一致 |
| 删除 legacy STOP 参数 | 清理 | 极小 | 减少混淆 |
| 提取 `r_fid` 常数项 | 简化 | 极小 | 移除无效排名偏移 |
| 归档 `internalization_dynamics_v2.py` | 清理 | 极小 | 不再使用的旧版本 |
| 拆分 CurriculumControllerV13 | 重构 | 中 | 635行 → ~300行核心 |
| 实现 Persistent Profile | 新功能 | 大 | Phase 2 核心 |
| Observer 更新的通用化 | 简化 | 中 | 减少 ~30 行重复代码 |
