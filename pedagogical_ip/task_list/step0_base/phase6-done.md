# Phase 6 完成总结：Belief-Conditioned Bounded Planning

## 结果

- **177/177 tests pass**（+27 新增：8 belief_planning + 11 diagnostics + 7 integration + 1 mode隔离）
- **Legacy 基线不变**：no_tutor=9%, warn=80%, door_2=68%, door_3=99%, close=100%

---

### 核心变化

#### 新建 `src/agents/belief_planning.py`（~240 行）

| 类/函数 | 语义 |
|---------|------|
| `BeliefPlan` | 结构化规划结果：action + prefix + scores + confidence + dominant_reason |
| `ScoreBreakdown` | cost/risk/uncertainty 三项分解 |
| `FailureModeEstimate` | 5 项启发式故障模式分数（prefix-based） |
| `plan_from_belief()` | wrap A* + alternatives → BeliefPlan |
| `estimate_failure_modes()` | 启发式分析 prefix 风险 |

#### 修改 `src/agents/planner_astar.py`（+75 行）

| 函数 | 语义 |
|------|------|
| `plan_with_alternatives_v2()` | 对每个合法首步做 constrained A* → path-level 候选分数 |
| 返回 | `(action, next_pos, path, candidate_scores)` |

#### 修改 `src/envs/lattice_v2_runner.py`（+40 行）

- `V2EpisodeState` 新增：`belief_planning_mode`, `confidence_temperature`, `last_belief_plan`, `last_failure_modes`
- `plan_and_move()`：`belief_planning_mode=True` 时走 belief planning 路径
- `reset()`：传入 `belief_planning_mode`, `confidence_temperature`

---

### 设计要点

| 要点 | 实现 |
|------|------|
| Confidence | path-level runner-up gap (constrained A*), 非单格 neighbor cost |
| Temperature | 可配置 `confidence_temperature`，默认 1.0 |
| dominant_reason | cost/risk/uncertainty 三项比较，差 <20% 返回 `"mixed"` |
| Failure modes | 全部 prefix-based：cumulative_risk, uncertainty, deadline_miss, no_safe_route, warning_insufficient |
| score_breakdown | BeliefPlan 内含 cost_term / risk_term / uncertainty_term |
| Mode 隔离 | `belief_planning_mode=False` → 旧路径完全不变 |

---

### 累积进度

| Phase | 内容 | 测试 |
|-------|------|------|
| 0 | 基线冻结 | 77 |
| 1 | 协议层 | 91 |
| 2 | Runner 平台化 | 97 |
| 3 | 环境接口 | 105 |
| 4 | Latent world 语义 | 124 |
| 5 | Patch 观测 + Prefix 预测 | 150 |
| **6** | **Belief-conditioned bounded planning** | **177 (+27)** |
