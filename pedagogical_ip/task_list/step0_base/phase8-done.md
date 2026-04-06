# Phase 8 完成总结：Unified Intervention Family + Minimal Item-Drop

## 结果

- **245/245 tests pass**（+36 新增：8 interventions_api + 6 warn_unlock_regression + 9 item_drop + 14 integration - 1 旧测试更新）
- **Legacy 基线不变**：no_tutor=9%, warn=80%, door_2=68%, door_3=99%, close=100%

---

### 核心变化

#### 增强 `src/teachers/interventions.py`（+80 行）

| 新增 | 语义 |
|------|------|
| `ItemType` | 仅 `SHIELD`（Phase 8 scope） |
| `ItemEffect` | risk_reduction + auto_consume + location |
| `InventoryState` | binary shield {0,1}, clone(), consume/add |
| `MAIN_INTERVENTION_FAMILY` | WAIT/WARN/UNLOCK/DROP_SHIELD（BLOCK_PATH **不在**主比较内） |
| `SHIELD_DEFAULT_RISK_REDUCTION` | 单一 config source（0.5） |
| `item_drop()` 工厂 | 验证 item_type + location |
| `VALID_ITEM_LOCATIONS` | 仅 `current_cell` |

#### 增强 `src/agents/planner_astar.py`（+10 行）

- `cell_cost_v2_latent()`：新增 `inventory_state` 参数
- 有 shield 时 `risk_penalty *= (1 - shield_risk_reduction)`
- **同一个 reduction 因子**用于 planner / predictor / runner

#### 增强 `src/teachers/agent_predictor.py`（+35 行）

- `predict_agent_prefix()` 接受 `inventory_state`
- 新增 `predict_agent_prefix_after_item_drop()`：克隆 inventory → add shield → rollout

#### 重写 `src/teachers/intervention_policy.py`

- 四路反事实 rollout：WAIT / WARN / UNLOCK / **ITEM_DROP**
- **BLOCK_PATH 不参与主比较**
- `InterventionDecision` 新增 `expected_item_effect`
- `InterventionConfig` 新增 `item_drop_weight`, `item_drop_cost`, `item_drop_enabled`

#### 增强 `src/envs/lattice_v2_runner.py`（+30 行）

- 新增 state：`intervention_family_mode`, `item_drop_enabled`, `inventory`
- ITEM_DROP 执行：`inventory.add_shield()`
- Shield 消耗：risky traversal 时 `effective_risk *= (1 - reduction)` + `consume_shield()`

---

### 你的 7 条修改全部落地

| 修改 | 状态 |
|------|------|
| 1. `inventory_state` 而非 `has_shield` | ✅ |
| 2. Location 锁 `current_cell` | ✅ |
| 3. Planner/execution shield 语义一致 | ✅ 同一个 `(1 - shield_risk_reduction)` |
| 4. Counterfactual 用 clone 不改真实 state | ✅ `inventory.clone()` |
| 5. `InterventionDecision.expected_item_effect` | ✅ |
| 6. `BLOCK_PATH` 不进主比较 | ✅ |
| 7. `shield_risk_reduction` 单一 config source | ✅ `SHIELD_DEFAULT_RISK_REDUCTION` |

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
| 6 | Belief-conditioned bounded planning | 177 |
| 7 | Approximate robot belief | 209 |
| **8** | **Unified intervention family + shield item-drop** | **245 (+36)** |
