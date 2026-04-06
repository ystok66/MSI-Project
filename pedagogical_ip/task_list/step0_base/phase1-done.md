# Phase 1 完成总结：协议层整理（不改科学内容）

> 详细任务说明见 [step0-phase1.md](./step0-phase1.md)

## 结果

- **91/91 tests pass**（原 77 + 新增 14）
- **V2 基线不变**：no_tutor=9%, warn=80%, door_2=68%, door_3=99%, close=100%
- 所有改动均为纯重构 / 纯新增，零行为变化

---

### 1. Planner core 统一

- 在 `planner_astar.py` 中提取共享 `_astar_core(cost_fn, ...)`
- 保留 `plan_next_action()` ＋ `plan_next_action_v2()` 作为 adapter
- 新增 `tests/test_planner_v2.py`（4 tests）

### 2. Warning protocol 统一

- 新建 `src/agents/pragmatic_warning.py` → `PragmaticWarner` 协议
- `rsa_warning.py` → `RSAWarner`，`warning_update.py` → `LaneWarner`
- **统一调用协议，不统一 speaker policy 本身**
- 新增 `tests/test_warning_protocol.py`（6 tests）

### 3. Belief protocol 统一

- 新建 `src/agents/belief_protocol.py` → `CellBelief` 协议
- `BeliefMap` 加 `H`/`W`/`get_belief()`/`reset()`
- **只统一 method surface，不统一内部语义**
- 新增 `tests/test_belief_protocol.py`（4 tests）
