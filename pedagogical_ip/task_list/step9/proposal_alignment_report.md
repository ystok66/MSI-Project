# Proposal × AGENT_BRIEF × 代码现状 — 对齐审计报告

> 审计日期：2026-04-07  
> 参考文档：`proposal.md`（原始研究提案）、`AGENT_BRIEF.md`（重构方向）  
> 审计范围：`src/` 全部代码 + `tests/` + `scripts/`

---

## 目录

1. [概要矩阵：Proposal 项 vs 代码现状](#1-概要矩阵)
2. [已完成 — 超越 Proposal 的部分](#2-已完成--超越-proposal-的部分)
3. [已完成 — 与 Proposal 一致的部分](#3-已完成--与-proposal-一致的部分)
4. [部分完成 — 有实现但有缺陷或未集成](#4-部分完成)
5. [未完成 — Proposal 要求但未实现](#5-未完成)
6. [AGENT_BRIEF 三个 Target 的对齐状态](#6-agent_brief-三个-target-的对齐状态)
7. [Proposal 时间线 vs 实际进度](#7-proposal-时间线-vs-实际进度)
8. [Identified Gaps — 影响 Paper 的关键缺口](#8-identified-gaps--影响-paper-的关键缺口)
9. [推荐行动](#9-推荐行动)

---

## 1. 概要矩阵

| Proposal 要求 | 对应代码 | 状态 | 缺陷 |
|--------------|---------|------|------|
| 网格环境 + 多维 feature → cost/risk | `lattice_v2.py`, `map_families.py`, `scenario_families.py` | ✅ 完成 | — |
| Agent noisy observation | `observation_model.py` | ✅ 完成 | — |
| Agent Bayesian learner (cost/risk) | `cost_risk_model.py`, `structured_basis_head.py` | ✅ 完成 (3 代头) | NaN guard 缺失 (BUG-3) |
| 干预-开捷径 (UNLOCK) | `interventions.py`, `intervention_semantics.py` | ✅ 完成 | — |
| 干预-告知 / RSA warning | `rsa_warning_channel.py`, RSA L0/S1/S1_trust | ✅ 完成 | — |
| 干预-提供道具 (ITEM_DROP / Shield) | `interventions.py`, `InventoryState` | ✅ 完成 | `active_duration` ghost attr |
| 有限时间评估 | `t_max`, deadline penalty in Q | ✅ 完成 | — |
| Boredom/Frustration 指标 | `boredom_proxy.py`, β_bore=0.3 | ✅ 完成 | — |
| Agent 表现提升 (Transfer) | `slow_fast_head.py`, `GenericSlowFastPredictor` | 🟡 架构完成但未集成 | Runner 不调用 lifecycle (BUG-1) |
| **POMDP Agent** | `FeatureBeliefMap` + bounded A* | 🟡 功能等价但非 POMDP 抽象 | — |
| **Belief over Agent's Belief** (nested ToM) | `RobotBelief` + `robot_belief_over_agent.py` | 🟡 单层 proxy 完成，nested 未集成 | `robot_belief_over_agent` 仍是 shadow |
| **Sequential Bayesian Inverse Planning** | `agent_predictor.py` counterfactual rollout | 🟡 一步 counterfactual，非序列 | 非 Bayesian inverse planning |
| **Bounded Rationality 建模** | bounded A* search budget | ✅ 完成 | — |
| Compositional Goals (Q1) | `compositional_goal_*.py` (6 个文件) | 🟡 存在但 shadow | 从未接入主线 |
| Temptation / Hidden Preference (Q2) | `γ_spec` + `temptation_strength` + `stochastic_agent_policy` | ✅ 完成 | — |
| 安全阈值约束 | `κ` risk sensitivity + shield + risk budget | ✅ 完成 | — |

---

## 2. 已完成 — 超越 Proposal 的部分

以下是 Proposal 没有明确要求，但代码中已经实现并成熟的功能：

### 2.1 Phase 10: Tutor Perceptual Model (TPM)

Proposal 只提及 "Robot monitors Agent's trajectory"，但代码实现了完整的感知接入模型：
- `perceptual_model.py` — 追踪 tutor 对 agent 已看到什么的估计 (ρ_{i,t})
- `compute_redundancy()` — 基于 ρ 的 warning 冗余度计算
- `compute_decision_relevant_unseen()` — 决策相关未见比例

### 2.2 5D Internalization State

Proposal 提到 agent 学习，但没有提到建模 agent 的 **过度依赖** 或 **探索抑制**：
- `internalization_state_v3.py` — 5D 状态: (κ, τ, ν, γ_spec, γ_gen)
- `internalization_observer.py` — 4 个 observer 版本 (A0/A1/A1-Frozen/A2)
- `a1mt_observer_shadow_prob.py` — probabilistic shadow observer

### 2.3 Bottleneck Diagnosis

三维瓶颈诊断（epistemic/structural/outcome），自动匹配干预类型：
- `bottleneck_diagnosis.py`
- 与 `intervention_policy.py` 集成

### 2.4 Warning 子类型系统

Proposal 只提到 "informing"，代码实现了四种 warning 子类型（hint/alert/explain/directive）：
- `warning_utterance_policy.py` —（shadow-only，但架构完整）

### 2.5 三代 Predictor Head + Transfer 机制

- Gen1: 4D Linear Bayesian
- Gen2: 6D/7D Structured Basis
- Gen3: GenericSlowFast dual-timescale wrapper
- `PredictorProtocol` — 统一接口

### 2.6 多场景家族

Proposal 只描述了一种网格环境，代码实现了 **三个场景家族**：
- `baseline_v2` — feature-driven risk learning
- `GTET` — goal/temptation/epistemic tradeoff
- `DTMB` — deep tree mixed bottleneck (多阶段决策)

---

## 3. 已完成 — 与 Proposal 一致的部分

| Proposal 项 | 实现文件 | 验证状态 |
|-------------|---------|---------|
| §2 Grid-world + latent cost/risk vectors | `lattice_v2.py`, `map_generator.py`, `pedagogical_grid.py` | ✅ 64 个测试覆盖 |
| §2 Noisy perceptual cues | `observation_model.py` 三级噪声: self=0.01, neighbor=0.08, far=0.20 | ✅ `test_patch_observation.py` |
| §2 True cost/risk revealed on visit | `lattice_v2_runner.py:520-555` 的 outcome learning | ✅ |
| §2 Bayesian learning mechanism | `BayesianCostHead`, `BayesianRiskHead` + MAP update | ✅ `test_cost_risk_model.py` |
| §3a Environmental Modification | `interventions.py:UNLOCK`, `passable[r,c] = True` | ✅ `test_interventions_api.py` |
| §3b RSA Pragmatic Communication | `rsa_warning_channel.py`: L0/S1 literal/pragmatic listener | ✅ `test_rsa_warning.py`, `_run_rsa_tests.py` |
| §3c Affordance Provision (Shield) | `InventoryState`, shield_risk_reduction=0.5 | ✅ `test_item_drop.py` |
| §4.1 Time-Bounded Success Rate | `s.survived`, `s.reached_goal`, `t_max` enforcement | ✅ 主 metric |
| §4.2 Boredom/Frustration | `boredom_proxy.py`: B_wait = avg_cost / (ε + LG) | ✅ β_bore=0.3 promoted |
| §4.3 Transfer Learning | `GenericSlowFastPredictor`, α-sweep 实验完成 | ✅ 实验，🔴 未集成到 runner |
| §5 Agent POMDP representation | `FeatureBeliefMap` Kalman posterior | ✅ 功能等价 |
| §5 Belief over Agent's Belief | `RobotBelief` surrogate snapshot | ✅ 单层 |
| §5 Bounded Rationality | bounded A* search budget sampling {4,8,16} | ✅ |
| §6 Q2: Temptations / hidden preference | `γ_spec` + `temptation_strength` + `latent_preference` | ✅ |

---

## 4. 部分完成

### 4.1 🟡 Transfer Learning — 架构完成但未集成

| 要素 | 状态 | 详情 |
|------|------|------|
| `GenericSlowFastPredictor` 类 | ✅ 完成 | `slow_fast_head.py` |
| `begin_episode()` / `end_episode()` | ✅ 实现 | 方法存在 |
| Runner 调用 lifecycle | 🔴 **缺失** | `lattice_v2_runner.py` 从不调用 (BUG-1) |
| α-sweep 实验 | ✅ 完成 | basis_slowfast α=0.2 → 1.000 survival |
| `end_episode()` 更新 slow xx_sum | 🔴 **缺失** | 只更新 w,b (BUG-2) |
| Transfer eval pipeline | 🟡 存在 | `transfer_eval.py` 但用 manual w-copy, 会和 basis 冲突 |

> [!WARNING]
> **这是 Proposal §4.3 最关键的缺口**。"Agent Performance Delta (Transfer Learning)" 作为 三大评估指标之一 在 Proposal 中被明确提出，但当前主 runner pipeline 不支持跨 episode transfer。实验只在独立 scripts 中手动实现。

### 4.2 🟡 Nested Theory of Mind — 单层 proxy

| 要素 | Proposal 要求 | 实现状态 |
|------|-------------|---------|
| Robot 知道 agent 的 partial observation | ✅ 需要 | `RobotBelief` 持有 `agent_belief_mean/var` |
| Robot 估计 agent 的 predictor 参数 | ✅ 需要 | `_predictor_snapshot = deepcopy(predictor)` |
| **Robot 维护 belief OVER agent belief** | ✅ 需要 | `robot_belief_over_agent.py` 🟡 shadow mode |
| **Sequential inverse planning** | ✅ 需要 | `agent_predictor.py` 🟡 一步 counterfactual only |

Proposal §5 明确要求 **"Belief over Agent's Belief"** 和 **"Sequential Bayesian Inverse Planning"**，但代码中：
- `RobotBelief` 是 **exact/noisy/stale copy**，而非 Bayesian posterior over belief space
- `agent_predictor.py` 是 **one-step counterfactual**，而非 sequential planning

### 4.3 🟡 Compositional Goals (Q1)

Proposal Q1 明确提出组合目标体系，代码有 6 个相关文件但全部 shadow:

| 文件 | 状态 |
|------|------|
| `compositional_goal_hypotheses.py` | 🟡 Shadow |
| `compositional_goal_prior.py` | 🟡 Shadow |
| `compositional_goal_bridge.py` | 🟡 Shadow |
| `composite_goal_compatibility.py` | 🟡 Shadow |
| `goal_conditional_curriculum_hook.py` | 🟡 Shadow |
| `test_compositional_goal_*.py` | 存在但无主线集成 |

---

## 5. 未完成

### 5.1 🔲 序列化 Bayesian Inverse Planning

Proposal：
> "The Robot utilizes a predictive model based on **sequential Bayesian inverse planning** to forecast the Agent's next-action distributions"

代码现状：
- 一步 counterfactual only（predict_agent_prefix → 单次 A* rollout）
- 没有 T-step 递归推理
- 没有 Bayesian posterior over agent's possible goals/intentions

### 5.2 🔲 Action Distribution Prediction

Proposal：
> "forecast the Agent's next-action **distributions**"

代码现状：
- `agent_predictor.py` 返回单一确定性 path（A* optimal path）
- `plan_with_alternatives_v2()` 计算 candidate_scores 但不输出 action probability distribution
- 无 softmax-rationality 模型来生成 P(a | state, θ)

### 5.3 🔲 真正的 Nested Belief Space

Proposal：
> "Belief over Agent's Belief" — a distribution, not a point estimate

代码现状：
- `RobotBelief` 是 agent belief 的 **point snapshot**（exact copy, noisy copy, 或 stale copy）
- `robot_belief_over_agent.py` 存在但是 shallow wrapper, shadow-only
- 没有 particle filter / distribution over possible agent beliefs

### 5.4 🔲 完整的 Epistemic-Cost Trade-off Metric (§4.2)

Proposal：
> "a state where the Agent's expected information gain approaches zero while physical/time costs continue to accumulate"

代码现状：
- `boredom_proxy.py` 实现了 B_wait = cost / (ε + LG)，这是 **tutor decision input**
- 但作为 **evaluation metric**（可以量化报告、数据驱动分析）目前不存在独立的 boredom 时间序列或统计接口
- `step_logger.py` 不记录 boredom score

---

## 6. AGENT_BRIEF 三个 Target 的对齐状态

### Target A — Environment/API cleanup

> "make environment, agent state, observation, and intervention interfaces cleaner and more modular"

| 子项 | 状态 | 说明 |
|------|------|------|
| 环境 API 清晰度 | ✅ | `lattice_v2.py` + `lattice_v2_env.py` 分离 |
| Agent state 模块化 | 🟡 | `FeatureBeliefMap` 干净，但 runner state (`V2EpisodeState`) 是巨大 dataclass（50+ 字段），mixing agent/env/tutor state |
| Observation 接口 | ✅ | `observation_model.py` 干净且有两级 API |
| Intervention 接口 | ✅ | `InterventionDecision` 语义分离 |
| **关键残留问题** | 🟡 | `V2EpisodeState` 同时包含 agent state, tutor state, env state，violates SoC |

### Target B — Belief / teacher refactor

> "Separate: world state, agent belief, robot belief about agent belief, predictive action model"

| 子项 | Proposal/Brief 要求 | 代码现状 | Gap |
|------|---------------------|---------|-----|
| World state | 独立 | `world_state.py` 存在 (⚪ legacy, 未用) | 未迁移到主线 |
| Agent belief | 独立 | `FeatureBeliefMap` ✅ 独立 | 无 gap |
| Robot belief about agent belief | Bayesian posterior | `RobotBelief` 🟡 point snapshot | 非 Bayesian |
| Predictive action model | Sequential inverse planning | `agent_predictor.py` 🟡 one-step | 非序列 |

> [!IMPORTANT]
> Target B 是 AGENT_BRIEF 和 Proposal 共同强调的核心差距。当前 `RobotBelief` 是 "copy-sync" 模式，不是 "Bayesian belief over belief"。`agent_predictor` 是 "one-shot counterfactual"，不是 "sequential inverse planning"。

### Target C — Warning semantics refactor

> "replace the current lane-warning logic with a simple modular pragmatic warning model"

| 子项 | 状态 | 说明 |
|------|------|------|
| RSA L0/S1 Speaker-Listener | ✅ 完成 | `rsa_warning_channel.py`, 631 行完整实现 |
| 4-way utterance space | ✅ | WARN_LEFT, WARN_RIGHT, WARN_AHEAD, GENERIC_WARN |
| Planner adapter | ✅ | `apply_planner_adapter()` 将 belief delta 转为 cell penalty |
| Legacy lane-warning | ⚪ 仍存在 | `warned_lane_bias` 和 `legacy_bias` 路径仍在，但已 demoted |
| 模块化程度 | ✅ | RSA channel 与 planner 完全解耦 |

**Target C 是三个 Target 中完成度最高的。** RSA warning 已经完全模块化（`rsa_warning_channel.py` + `warning_belief_delta.py` + planner adapter），且有 ablation variant（l0, s1, s1_trust）。残留的 legacy_bias 路径仅为向后兼容。

---

## 7. Proposal 时间线 vs 实际进度

| 时间段 | Proposal 计划 | 实际完成 | 评估 |
|--------|-------------|---------|------|
| 3.8-3.22 | Literature Review & Environment Setup | ✅ lattice_v2 + 3 场景家族 + scenario_families (120K 行) | 超额完成 |
| 3.23-3.31 | Agent Modeling & Baseline | ✅ 3 代 predictor + 5D internalization + RSA warning | 超额完成 |
| 4.1-4.7 | Robot ToM & RSA Integration, Intervention Logic | 🟡 intervention_policy + bottleneck + TPM，但 nested ToM 只有 shadow | 核心 intervention 完成，nested ToM 差距 |
| 4.8-4.20 | Evaluation & Metrics, Data Analysis, Ablation | 🟡 step_logger + boredom proxy + transfer 实验，但统一 eval pipeline 未就绪 | 进行中 |

---

## 8. Identified Gaps — 影响 Paper 的关键缺口

### Gap-1: ⭐ Transfer Learning 不可用于正式实验

**Proposal 重要性**：§4.3 三大指标之一  
**当前状态**：
- SlowFast 机制只在手动 scripts 中有效
- 主 runner 不调用 `begin_episode()/end_episode()`
- `transfer_eval.py` 用 manual w-copy，会和 basis head 冲突
- **无法产出 "with-tutor vs without-tutor transfer Δ" 数据**

**修复**：runner 增加 SlowFast lifecycle 调用 + 统一 transfer eval API

### Gap-2: ⭐ 缺少 "Tutor Off" 对照组管线

Proposal 要求度量 **agent 独立能力提升**：
> "improvement in the Agent's standalone planning efficiency in a zero-shot, robot-free environment"

当前 runner 总是带 tutor（`tutor_mode` 可以设为 WAIT-only，但无正式 "tutor-off probe" 管线）。需要一个标准的 "train with tutor → eval without tutor" protocol。

### Gap-3: Nested ToM 深度不足

Proposal 和 AGENT_BRIEF 都强调 "Belief over Agent's Belief"，但当前实现是 point-copy。对于 paper 故事来说，至少需要 demonstration：
- Robot 对 agent belief 的不确定性影响干预时机
- 与 exact-copy baseline 比较

`robot_belief_over_agent.py` 已存在但未接入。

### Gap-4: Boredom metric 只是决策输入，不是评估指标

Proposal 将 Boredom/Frustration 列为 **evaluation metric**（§4.2），但代码中 `boredom_proxy.py` 只是 Q_WAIT 的一个 penalty term，没有独立的 evaluation API。需要：
- 每步记录 boredom score 到 step_logger
- 在 eval summary 中输出 mean/max boredom

### Gap-5: NaN safety 在 risk heads 中缺失

- `BayesianRiskHead` 和 `BasisRiskHead` 缺少 `np.isfinite` gradient check
- 在真实跑大规模实验时可能导致 silent NaN crash

---

## 9. 推荐行动

### 9.1 P0 — 必须在实验阶段 (4.8-4.20) 前完成

| # | 行动 | 修复的 Gap | 难度 | 文件 |
|---|------|-----------|------|------|
| 1 | **Runner 增加 SlowFast lifecycle 调用** | Gap-1 | 🟢 约 10 行 | `lattice_v2_runner.py` |
| 2 | **NaN gradient guard** 加到两个 risk head | Gap-5 | 🟢 约 4 行 | `risk_model.py`, `structured_basis_head.py` |
| 3 | **统一 planner weights**：runner init_robot_belief 传 agent_risk_weight=5.0 | MISMATCH-1 | 🟢 约 2 行 | `lattice_v2_runner.py` |
| 4 | **修复 inventory_state 传递** | BUG-1,2 | 🟢 约 5 行 | `agent_predictor.py`, `planner_astar.py` |
| 5 | **`active_duration` → `has_shield()`** 修复 | BUG-4 | 🟢 1 行 | `dtmb_helpers.py` |

### 9.2 P1 — 实验阶段完善

| # | 行动 | 修复的 Gap | 难度 |
|---|------|-----------|------|
| 6 | **`step_logger` 增加 boredom score 记录** | Gap-4 | 🟡 |
| 7 | **Transfer eval protocol** — train-with-tutor → eval-without-tutor 标准管线 | Gap-2 | 🟡 |
| 8 | **`end_episode()` 更新 slow xx_sum/n_updates** | BUG-2 | 🟡 |
| 9 | **Compositional goal 至少 1 个 demo** 接入主线 | Proposal Q1 | 🟠 |

### 9.3 P2 — Paper 叙事增强（可选）

| # | 行动 | 修复的 Gap | 难度 |
|---|------|-----------|------|
| 10 | **Nested ToM demo** — `robot_belief_over_agent.py` 接入 + noisy-copy ablation | Gap-3 | 🟠 |
| 11 | **Action distribution prediction** — 从 A* scores 生成 softmax P(a) | §5 | 🟠 |
| 12 | **清理 V0 dead code** — 约 500 行可移除 | 维护性 | 🟢 |

---

## 附录 A: AGENT_BRIEF 与 Proposal 关系

AGENT_BRIEF 是基于 Proposal 写的 **实施指南**（而非替代文档）。两者关系：

```
Proposal              → what to build (research goals)
AGENT_BRIEF           → how to build (engineering targets)
step9/reports         → what was actually built (audit)
```

| AGENT_BRIEF 项 | Proposal 对应 | 一致性 |
|----------------|-------------|--------|
| Target A: Env/API cleanup | §2 Task Description | ✅ 一致 |
| Target B: Belief/teacher refactor | §5 Computational Modeling | ✅ 一致（但 B 更具体） |
| Target C: Warning semantics | §3b RSA Communication | ✅ 一致 |
| "Don't rewrite" 约束 | — | AGENT_BRIEF 独有但合理 |
| External refs (Minigrid, pomdp-py, pypragmods) | — | AGENT_BRIEF 独有，用于灵感 |

AGENT_BRIEF 的 "Current scientific gap" 诊断（§L88-94）**与本报告 §4-5 的发现完全一致**：

> 1. "Agent/robot mental-state modeling is not yet a clean POMDP + nested-belief abstraction."  
>    → 确认：`RobotBelief` 是 point-copy

> 2. "Planning/inverse-planning interface is still too custom and heuristic."  
>    → 确认：`agent_predictor` 是 one-step counterfactual, 不是 sequential inverse planning

> 3. "Warning semantics work, but are not yet a clean modular pragmatic listener/speaker design."  
>    → **已修复**：RSA channel 已完全模块化 (Target C ✅)

---

## 附录 B: 与其他 step9 报告的关系

| 报告 | 关注维度 | 与本报告的关系 |
|------|---------|--------------|
| `scenario_cell_mechanism_report.md` | 环境层面 | 对应 Proposal §2 |
| `tutor_mechanism_report.md` | Teacher 层面 | 对应 Proposal §3+§5 |
| `learner_agent_mechanism_report.md` | Agent 层面 | 对应 Proposal §2+§5 |
| `interface_dead_param_audit.md` | 工程质量 | 对应 AGENT_BRIEF constraints |
| `potential_bugs_and_temp_code_audit.md` | Bug/临时代码 | 对应 AGENT_BRIEF constraints |
| **本报告** | **研究目标对齐** | **跨文档交叉引用** |
