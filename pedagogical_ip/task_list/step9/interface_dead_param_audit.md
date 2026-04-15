# 接口混乱 & 死参数审计报告

> 审计范围：`src/agents/` + `src/teachers/` + `src/envs/lattice_v2_runner.py`
> 审计日期：2026-04-07

---

## 目录

1. [🔴 BUG: 死参数 — 传入但因代码路径永远不起效](#1-bug-死参数)
2. [🟠 MISMATCH: 默认值不一致 — 名义相同但值不同](#2-mismatch-默认值不一致)
3. [🟡 NAMING: 接口命名混乱 — 可统一](#3-naming-接口命名混乱)
4. [⚪ LEGACY: 冗余传参 — 功能重叠](#4-legacy-冗余传参)
5. [修复建议优先级](#5-修复建议优先级)

---

## 1. 🔴 BUG: 死参数

### BUG-1: `inventory_state` 在 `plan_from_belief` 中未传递

| 位置 | 详情 |
|------|------|
| 文件 | [agent_predictor.py:62-75](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/agent_predictor.py#L62-L75) |
| 问题 | `predict_agent_prefix()` 接收 `inventory_state` 参数（L50），传给了 `plan_with_alternatives_v2`（L88），但 **没有传给** `plan_from_belief`（L62-75）。|
| 影响 | `plan_from_belief` 内部调用 `plan_with_alternatives_v2` 时 `inventory_state` 始终为 `None`，导致 shield 效果在 **belief 规划路径** 上被静默忽略。|
| 影响范围 | 当 `belief_planning_mode=True` + 有 shield 时，tutor 的 counterfactual 预测 **不考虑 shield 对 risk 的衰减**。|

```python
# agent_predictor.py L62-75 — 缺少 inventory_state
bp = plan_from_belief(
    agent_pos, goal, belief_cost, rb.agent_belief_mean,
    surrogate_lp.risk_head, passable,
    latent_predictor=surrogate_lp,
    warned_cell_extra=warned_cell_extra,
    search_budget=rb.agent_search_budget,
    ...
    # ❌ 没有: inventory_state=inventory_state
)
```

> [!CAUTION]
> 同一函数内的 `plan_with_alternatives_v2` 调用（L78-89）**正确传了** `inventory_state`。说明两次调用结果不一致：belief plan 评估的 risk 比 candidate scoring 评估的 risk 更高。

---

### BUG-2: `inventory_state` 在 `plan_next_action_v2` 中未传递

| 位置 | 详情 |
|------|------|
| 文件 | [planner_astar.py:417-429](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/planner_astar.py#L417-L429) |
| 问题 | `plan_next_action_v2()` 压根**不接受** `inventory_state` 参数。而同文件的 `plan_with_alternatives_v2()` 接受并正确传给 `cell_cost_v2_latent()`。|
| 影响 | runner 中 `plan_and_move()` 走 legacy 路径（`belief_planning_mode=False`）时总是用 `plan_next_action_v2`，shield **在规划中完全不起效**。|

```python
# planner_astar.py L390-407 — 不接受 inventory_state
def plan_next_action_v2(
    agent_pos, goal, belief_cost_mean, feature_belief_mean,
    risk_model, budget=30, lambda_risk=5.0, ...
    # ❌ 无 inventory_state 参数
) -> tuple[str, tuple[int, int], list[tuple[int, int]]]:
```

对比同文件 `plan_with_alternatives_v2`（L461）正确接受 `inventory_state`。

---

### BUG-3: `feature_belief_var` 在 tutor counterfactual 中未传递

| 位置 | 详情 |
|------|------|
| 文件 | [agent_predictor.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/agent_predictor.py) 全部 4 个函数 |
| 问题 | 4 个 `predict_agent_prefix_*` 函数都调用了 planner，但 **没有一个** 传入 `feature_belief_var`。|
| 影响 | tutor 的 counterfactual 规划在计算 uncertainty 时使用 Hessian-based 标量近似（`predict_cost_uncertainty(x)`），而 runner 的真实规划使用 posterior variance propagation（`predict_cost_uncertainty_from_var(x_var)`）。两种 uncertainty 计算方法不一致。|

```diff
 # agent_predictor.py → plan_from_belief 调用
 bp = plan_from_belief(
     agent_pos, goal, belief_cost, rb.agent_belief_mean,
     surrogate_lp.risk_head, passable,
     latent_predictor=surrogate_lp,
+    feature_belief_var=rb.agent_belief_var,   # ← 缺失
     ...
 )
```

---

### BUG-4: `route_necessity` 在 tutor counterfactual 中未传递

| 位置 | 详情 |
|------|------|
| 文件 | [agent_predictor.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/agent_predictor.py) 全部 4 个函数 |
| 问题 | runner 在 `plan_and_move()` 中计算 `route_necessity` 并传给 planner，但 tutor 的 counterfactual rollouts **始终用默认值 0.0**。|
| 影响 | tutor 高估 uncertainty penalty（因为 `1-n = 1.0`），导致在结构紧张场景中 WAIT 分数被错误压低。|

---

### BUG-5: `lambda_risk` / `lambda_uncertainty` 默认值在 runner legacy 路径中硬编码

| 位置 | 详情 |
|------|------|
| 文件 | [lattice_v2_runner.py:498-504](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/lattice_v2_runner.py#L498-L504) |
| 问题 | legacy 路径 `plan_next_action_v2(...)` 没有传入 planner weights，始终使用函数默认值 `lambda_risk=5.0, lambda_c=1.0, lambda_uc=0.1, lambda_ur=0.1`。runner state 中不存储这些参数。|
| 影响 | 无法从 scenario 配置控制 planner 权重，所有场景使用同一组硬编码默认值。|

---

## 2. 🟠 MISMATCH: 默认值不一致

### MISMATCH-1: Planner lambda_risk 在 runner/tutor 之间不一致

| 参数 | Runner（隐式默认） | RobotBelief 默认 | 差值 |
|------|-------------------|------------------|------|
| λ_risk | **5.0** | **3.0** | +2.0 |
| λ_uncertainty | **0.1** | **0.5** | −0.4 |

**根本原因：** runner 不显式传 planner weights，直接用函数默认值；`init_robot_belief()` 也不传 `agent_risk_weight/agent_uncertainty_weight`(L358-363), 直接用 `RobotBelief` dataclass 默认值。

```python
# lattice_v2_runner.py L358-363 — init_robot_belief 调用
state.robot_belief = init_robot_belief(
    fb.mean, fb.var, latent_predictor=lp,
    copy_mode=belief_copy_mode,
    budget_mismatch=budget_mismatch,
    rng=rng,
    # ❌ 没有传: agent_risk_weight=5.0, agent_uncertainty_weight=0.1
)
```

> [!WARNING]
> 效果：tutor 的 surrogate 比 agent 更保守（λ_risk=3.0 < 5.0），同时对 uncertainty 更敏感（0.5 > 0.1）。两者的规划行为系统性不一致。在大多数场景中这可能不影响 action ranking，但在边际情况下会导致 tutor 误判 agent 行为。

---

### MISMATCH-2: cell_cost_v2 vs cell_cost_v2_latent 的 lambda_risk 默认值不同

| 函数 | lambda_risk 默认 | 用途 |
|------|-----------------|------|
| `cell_cost_v2()` | 5.0 | legacy 路径（无 latent_predictor）|
| `cell_cost_v2_latent()` | 5.0 | latent 路径 |
| `cell_cost()` (V0) | 传入 | V0 路径 |
| `bounded_astar()` | 3.0 | V0 直接调用 |
| `plan_next_action()` | 3.0 | V0 直接调用 |

V0 的默认 3.0 和 V2 的默认 5.0 来自不同阶段的 design decision，但名字相同（`lambda_risk`）。

---

### MISMATCH-3: `belief_cost` 在 latent 路径中是冗余参数

| 位置 | 详情 |
|------|------|
| 多处 | runner → planner → cell_cost_v2_latent |
| 问题 | `cell_cost_v2_latent()` 的 cost 来自 `latent_predictor.predict_cost(x_belief)`，**完全不使用** `belief_cost` 数组。但 `plan_next_action_v2` 和所有上游都强制要求传入 `belief_cost`。|
| 影响 | `belief_cost` 只用于两个地方：①判断 wall（从 passable mask 已覆盖），②legacy fallback 路径。在 latent 路径中它被完全忽略。|
| 说明 | 初始化时 `belief_cost[wall]=100.0`，但 latent path 用 `passable[wall]=False` 直接跳过，所以 100.0 值永远不被读取。|

---

## 3. 🟡 NAMING: 接口命名混乱

### NAMING-1: `lambda_risk` vs `lambda_r`

同一个概念（risk 权重），在不同函数中使用不同参数名：

| 函数 | 参数名 | 文件 |
|------|--------|------|
| `bounded_astar()` | `lambda_risk` | planner_astar.py |
| `cell_cost()` | `lambda_risk` | planner_astar.py |
| `cell_cost_v2()` | `lambda_risk` | planner_astar.py |
| `cell_cost_v2_latent()` | **`lambda_r`** | planner_astar.py |
| `plan_next_action_v2()` | `lambda_risk` | planner_astar.py |
| `plan_from_belief()` | `lambda_risk` | belief_planning.py |
| `RobotBelief` | `agent_risk_weight` | robot_belief.py |

> 三个名字指同一个量：`lambda_risk` / `lambda_r` / `agent_risk_weight`

### NAMING-2: `lambda_uncertainty` vs `lambda_uc` / `lambda_ur`

V2-latent 路径把 uncertainty 分拆为 cost/risk 两个独立权重（`lambda_uc`, `lambda_ur`），但 V2-legacy 路径仍然用一个 `lambda_uncertainty`。同一个 `plan_next_action_v2` 函数同时接受两套：

```python
def plan_next_action_v2(
    ...
    lambda_risk: float = 5.0,          # 用于 legacy 路径
    lambda_uncertainty: float = 0.1,   # 用于 legacy 路径
    ...
    lambda_c: float = 1.0,            # 用于 latent 路径
    lambda_uc: float = 0.1,           # 用于 latent 路径
    lambda_ur: float = 0.1,           # 用于 latent 路径
    ...
):
```

调用者必须同时传两套权重，而 `if latent_predictor is not None` 分支只用一套。**另一套永远不被使用**。

### NAMING-3: `risk_model` vs `risk_head`

| 用法 | 指代 |
|------|------|
| `risk_model` (planner_astar.py) | BayesianRiskHead（只预测 risk）|
| `risk_head` (cost_risk_model.py) | BayesianRiskHead（内部 sub-head）|
| `risk_head` (runner state) | BayesianRiskHead（agent 的独立 risk head）|
| `latent_predictor` | LatentCostRiskHead（包含 cost_head + risk_head）|

Planner 在 V2 函数签名中同时接受 `risk_model`（用于 legacy path）和 `latent_predictor`（用于 latent path）。在 latent path 下 `risk_model` 完全不使用。

---

## 4. ⚪ LEGACY: 冗余传参

### LEGACY-1: V0 路径（cell_cost / bounded_astar / plan_next_action）

这些函数使用 `(belief_cost_mean, belief_risk_mean, belief_cost_var)` 三数组作为 belief。已被 V2 路径（FeatureBeliefMap + latent_predictor）完全替代。但仍保留在 planner_astar.py 中。

- `cell_cost()` — 234 行代码
- `bounded_astar()` — 没有调用者
- `plan_next_action()` — 没有调用者

### LEGACY-2: `observation_model.py` V0 路径

文件头部已标注 `DEPRECATED`。仍保留 88 行代码。

### LEGACY-3: `warned_lane_bias` vs `warned_cell_extra`

Runner 同时维护两个 warned 数据结构：
- `warned_lane_bias: dict` — 按 segment_index 存放 aggregate lane penalty
- `warned_cell_extra: dict` — 按 (row, col) 存放 per-cell extra cost

Legacy 路径：warning → `warned_lane_bias` → `_build_warned_cell_extra()` → `warned_cell_extra`
RSA 路径：warning → `delta.planner_cell_penalties` → `warned_cell_extra`

在 RSA 路径下 `warned_lane_bias` **永远为空**，但仍然在 state 中占位并在每步被传递。

---

## 5. 修复建议优先级

### P0：立即修复（真实 Bug）

| ID | 修复 | 难度 |
|----|------|------|
| BUG-1 | `plan_from_belief()` 增加 `inventory_state` 参数并传递 | 🟢 简单 |
| BUG-2 | `plan_next_action_v2()` 增加 `inventory_state` 参数并传递 | 🟢 简单 |
| BUG-3 | `predict_agent_prefix()` → `plan_from_belief` 传入 `feature_belief_var=rb.agent_belief_var` | 🟢 简单 |
| BUG-4 | `predict_agent_prefix()` 增加 `route_necessity` 参数 | 🟡 需要计算 |
| MISMATCH-1 | runner `init_robot_belief()` 调用时显式传入 `agent_risk_weight=5.0, agent_uncertainty_weight=0.1` | 🟢 简单 |

### P1：清理（不影响功能但阻碍维护）

| ID | 修复 | 难度 |
|----|------|------|
| NAMING-1 | 统一 `lambda_r` → `lambda_risk`（或反向） | 🟡 |
| NAMING-2 | `plan_next_action_v2` 去掉 legacy-only 的 `lambda_uncertainty`，始终用 `(lambda_uc, lambda_ur)` | 🟡 |
| MISMATCH-3 | Latent 路径的 `belief_cost` 改为 Optional，在 latent path 不要求 | 🟡 |
| LEGACY-3 | RSA 路径下去掉 `warned_lane_bias` 维护 | 🟡 |

### P2：清扫（低优先）

| ID | 修复 | 难度 |
|----|------|------|
| LEGACY-1 | 如果 V0 不再使用，标记 DEPRECATED 或移除 | 🟢 |
| LEGACY-2 | `observation_model.py` V0 部分可移除 | 🟢 |
| NAMING-3 | 重命名 `risk_model` → `legacy_risk_head`，消除歧义 | 🟢 |
| BUG-5 | 在 state 中存储 planner weights，消除硬编码默认 | 🟡 |
