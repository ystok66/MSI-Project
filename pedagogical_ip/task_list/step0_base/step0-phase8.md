可以。基于你现在的状态，**Phase 8 最合理的定义**是：

# Phase 8：统一并补齐 intervention family

也就是先把 teacher 侧的干预动作真正统一成一个**可比较、可扩展、可测试**的动作空间，然后在这个统一接口下，把三类干预补齐：

* `WAIT`
* `WARN(payload)`
* `UNLOCK(target)`
* `ITEM_DROP(item_type, location)`

其中 **WARN** 和 **UNLOCK** 已经有基础，**ITEM_DROP / SHIELD** 才是这一步真正新增的内容。

我建议你这次继续让 Antigravity 走你前面一直在用的节奏：
**先出 Implementation Plan，再执行；执行时保留 Task List；终端保持 Request Review；完成后给 Walkthrough。** 这些流程在官方文档里都是现成支持的。 ([Google Antigravity][1])

---

# 一、Phase 8 的大致目的

到 Phase 7 为止，你已经有了：

* belief-conditioned agent
* robot-belief tutor
* prefix-based intervention reasoning
* 209 tests
* legacy baseline 一直锁住

所以现在最自然的下一步，不是继续加更深的 ToM，而是把 **teacher action space** 真正整理成 proposal 里的三类干预家族。

当前 gap 是：

* `WARN` 有逻辑，但还没成为统一 intervention API 的一个正式成员
* `UNLOCK` 有 door tutor 思路，但还是偏专用路径
* `ITEM_DROP / SHIELD` 还没进入 planner / env 的正式状态空间

Phase 8 的目标就是：

> **让三类干预在同一接口下可比较、可切换、可回归测试。**

---

# 二、Phase 8 的思路

## 1. 先统一接口，再补 item-drop

这一步最重要的，不是直接写 shield 逻辑，而是先把 teacher 侧的动作空间固定下来。

我建议以 `teachers/interventions.py` 为中心，统一成：

* `WAIT`
* `WARN(payload)`
* `UNLOCK(target)`
* `ITEM_DROP(item_type, location)`
* `BLOCK(target)` 可保留为 legacy / debug action，但不是主线

## 2. Phase 8 只做一个最小 item family

不要一开始就做多种 item。
我建议只支持：

* `item_type = "shield"`

而且先把 shield 定义成最小可行版本：

* agent inventory 最多持有 1 个 shield
* shield 是 consumable
* shield 在首次高风险 traversing 时自动消耗，或在 planner 明确决定使用时消耗
* location 可以先只支持：

  * 当前格
  * agent prefix 上的某个格
  * 或最小化为 `agent_current_pos`

这样 scope 最稳。

## 3. planner 只做“最小 inventory-aware”

Phase 8 不要重写 planner。
只要让 planner 在 item-drop 路径下，能考虑：

* inventory state（有没有 shield）
* shield 对某类风险格的 cost/risk 修正
* consumable 使用后的状态变化

也就是说：

> **先做单个 binary inventory state 就够了。**

## 4. intervention scoring 继续走 Phase 7 的 counterfactual rollout

这一点非常重要。
不要让 `ITEM_DROP` 变成另一个拍脑袋打分分支。
仍然建议统一成：

* `WAIT`：rollout surrogate
* `WARN`：在 surrogate 上施加 warning effect，再 rollout
* `UNLOCK`：在 surrogate topology 上施加 unlock，再 rollout
* `ITEM_DROP`：在 surrogate inventory / world 上施加 item-drop effect，再 rollout

这样所有 intervention 的比较才公平。

---

# 三、Phase 8 的方法

## 方法 A：新增统一 intervention schema

建议新增或扩展：

* `src/teachers/interventions.py`

让它成为**唯一动作定义层**。

## 方法 B：新增 item semantics，不重写旧 warn/unlock

我建议：

* `WARN` 保持你已有的 warning science
* `UNLOCK` 保持你已有的 topology-change science
* `ITEM_DROP` 新增最小 shield semantics

## 方法 C：继续 additive migration

建议加 config：

* `intervention_family_mode: false/true`
* `item_drop_enabled: false/true`
* `item_type: "shield"`
* `max_inventory: 1`
* `shield_risk_reduction`
* `shield_auto_consume: true/false`

这样：

* legacy tutor path 不动
* robot-belief tutor path 不动
* intervention family path 单独打开

---

# 四、我建议你先加的 8 个约束

## 1. 只做一个 item：`shield`

不要一开始做多个 item 类型。
Phase 8 的最小成功标准不是 inventory richness，而是 intervention family 闭环。

## 2. item-drop 一开始只支持单物品、单库存

建议：

* inventory ∈ {0,1}
* 不做 stack
* 不做多种 item 共存

## 3. shield 的语义要写死

建议明确：

* shield 对高风险 cell 的 risk penalty 下降，或 survival 提升
* 只生效一次
* 消耗后 inventory 清空

## 4. `ITEM_DROP` 先只做局部 location

不要支持任意地图任意 drop 策略。
建议最小化成：

* `current_cell`
* 或 `planned_prefix[0] / planned_prefix[1]`

## 5. intervention comparison 必须继续走 counterfactual rollout

这点和 Phase 7 一样重要。
不要让 item 分支用不同评分逻辑。

## 6. planner 必须是 inventory-aware，但不必 item-policy-aware

也就是 planner 能考虑“有 shield vs 没 shield”的路径代价差异，但不必在这一阶段自己做复杂 item scheduling。

## 7. `BLOCK` 继续留着，但标记为 legacy/debug

避免主线又被带回“强制封路”这个旧世界。

## 8. 所有 item 诊断都必须 prefix-based

也就是：

* shield 预期在哪个 prefix cell 发挥作用
* 预期减少多少 cumulative risk
* 如果不 drop，会在哪个 prefix 失败

---

# 下面是可以直接给 Antigravity 的任务单

```text
Project: pedagogical_ip
Phase: Phase 8 — unified intervention family with minimal item-drop support

Current status
Phase 0 complete: baselines frozen
Phase 1 complete: protocol cleanup
Phase 2 complete: runner platformization
Phase 3 complete: environment interface
Phase 4 complete: latent world semantics
Phase 5 complete: patch observation + prefix prediction
Phase 6 complete: belief-conditioned bounded planning
Phase 7 complete: approximate robot belief over agent belief
Current total tests: 209
Legacy V2 baseline remains unchanged

High-level goal
This phase unifies the intervention family and fills in the third intervention type.

The intervention family should become:
- WAIT
- WARN(payload)
- UNLOCK(target)
- ITEM_DROP(item_type, location)

Optional legacy/debug action:
- BLOCK(target)

Important clarification
This phase is NOT about adding many item types.
This phase should implement the smallest viable intervention family with one minimal item baseline.

Core design principle
First unify the intervention API.
Then add minimal item-drop support.
Do not redesign warning science or unlock science.

Scope for this phase
1. unify teacher-side intervention representation
2. keep WARN and UNLOCK as first-class actions under the same family
3. add a minimal ITEM_DROP baseline
4. make planner inventory-aware in the smallest possible way
5. compare WAIT / WARN / UNLOCK / ITEM_DROP under the same counterfactual rollout logic

What this phase should NOT do
- do NOT add multiple item families
- do NOT redesign the planner family
- do NOT add full inventory management
- do NOT add item crafting / stacking / trading
- do NOT introduce RL over intervention policies
- do NOT remove any legacy path
- do NOT break the current robot-belief tutor path

Minimal item design preference
Please implement only one item first:
- item_type = "shield"

Suggested semantics:
- binary inventory: 0 or 1
- consumable
- reduces risk penalty or improves survival on one risky traversal
- may be auto-consumed or explicitly consumed, but behavior must be simple and documented

Suggested location scope
Keep ITEM_DROP location minimal:
- current_cell
- or a near-prefix cell
Do not support arbitrary global drop policies unless clearly justified.

Counterfactual evaluation requirement
WAIT / WARN / UNLOCK / ITEM_DROP must all be scored using the same surrogate rollout framework.
Do NOT let ITEM_DROP use a separate ad hoc scoring path.

Strongly suggested files to inspect first
- src/teachers/intervention_policy.py
- src/teachers/robot_belief.py
- src/teachers/agent_predictor.py
- src/teachers/interventions.py
- src/agents/belief_planning.py
- src/agents/planner_astar.py
- src/envs/lattice_v2_env.py
- src/envs/lattice_v2_runner.py
- configs/teacher.yaml
- configs/agent.yaml
- configs/experiment.yaml
- tests/test_intervention_policy.py
- tests/test_v2_robot_belief_integration.py

Recommended implementation order
Priority 1:
Unify intervention action types under teachers/interventions.py.

Priority 2:
Refactor existing WARN and UNLOCK paths so they can be invoked through the unified intervention interface without changing behavior.

Priority 3:
Add a minimal shield-based ITEM_DROP action.

Priority 4:
Make planning/item evaluation inventory-aware in the smallest possible way.

Priority 5:
Extend intervention scoring so WAIT / WARN / UNLOCK / ITEM_DROP are compared under the same counterfactual surrogate rollout logic.

Priority 6:
Expose structured intervention-family diagnostics through env/runner info.

Suggested config additions
- intervention_family_mode: false/true
- item_drop_enabled: false/true
- item_type: shield
- max_inventory: 1
- shield_risk_reduction
- shield_auto_consume: true/false
- item_drop_cost
- intervention_compare_mode: unified_counterfactual

Suggested new structured outputs
Please consider a structured intervention decision that includes:
- chosen_action
- scores for WAIT / WARN / UNLOCK / ITEM_DROP
- predicted_prefix
- predicted_failure_modes
- counterfactual_scores
- decision_margin
- expected_item_effect (if item action is considered)

Required output format before implementation
1. concise diagnosis of current intervention fragmentation
2. proposed unified intervention API
3. proposed minimal shield semantics
4. proposed planner/inventory wiring
5. proposed counterfactual evaluation design
6. file-by-file change plan
7. config strategy
8. proposed tests
9. risks of over-engineering
10. minimal implementation order

Acceptance criteria
Phase 8 is successful only if:
- existing tests still pass unless intentionally updated
- new intervention-family tests pass
- legacy baseline path remains reproducible
- WARN / UNLOCK old behavior remains runnable
- ITEM_DROP path remains runnable
- planner can account for warning / topology change / simple inventory state
- all intervention types are comparable under one structured policy interface

Baseline preservation requirement
Legacy baseline values must remain reproducible:
- no_tutor = 9%
- warning_only (lambda=5) = 80%
- door_2 = 68%
- door_3 = 99%
- always_close = 100%
- lambda sweep: 1→9%, 3→46%, 5→80%, 7→100%

Please start with diagnosis and implementation plan only.
Do not code until the plan is written.
```

---

# 五、建议的测试内容

这一阶段的测试要证明四件事：

1. intervention API 真的统一了
2. WARN / UNLOCK 没被你统一接口时搞坏
3. ITEM_DROP / SHIELD 有最小可运行语义
4. planner / policy 能在同一框架下比较四类动作

---

## A. 旧测试先继续全过

先跑：

```bash
python -m pytest tests/ -v --tb=short
```

预期：

* 当前 **209 tests 全通过**

---

## B. 建议新增测试文件

我建议新增 4 个主测试文件。

---

### 1. `tests/test_interventions_api.py`

测统一 intervention schema。

建议至少 8 个：

#### `test_wait_action_constructible`

验证：

* `WAIT` 能通过统一接口构造

#### `test_warn_action_constructible`

验证：

* `WARN(payload)` 能通过统一接口构造

#### `test_unlock_action_constructible`

验证：

* `UNLOCK(target)` 能通过统一接口构造

#### `test_item_drop_action_constructible`

验证：

* `ITEM_DROP(item_type, location)` 能通过统一接口构造

#### `test_block_legacy_action_still_available`

验证：

* `BLOCK(target)` 若保留，仍可构造但标记为 legacy/debug

#### `test_intervention_enum_or_schema_stable`

验证：

* 统一 action schema 稳定

#### `test_invalid_item_type_rejected`

验证：

* 非法 item type 被拒绝

#### `test_invalid_location_rejected`

验证：

* 非法 location 被拒绝

---

### 2. `tests/test_warn_unlock_regression.py`

测 WARN / UNLOCK 在新接口下行为不漂移。

建议至少 6 个：

#### `test_warn_path_runs_under_unified_api`

验证：

* WARN 通过统一接口仍可运行

#### `test_unlock_path_runs_under_unified_api`

验证：

* UNLOCK 通过统一接口仍可运行

#### `test_warn_behavior_matches_previous_mode`

验证：

* 统一前后 WARN 行为一致或等价

#### `test_unlock_behavior_matches_previous_mode`

验证：

* 统一前后 UNLOCK 行为一致或等价

#### `test_unified_api_does_not_change_heuristic_tutor`

验证：

* heuristic tutor 旧路径不受影响

#### `test_unified_api_does_not_change_robot_belief_tutor_when_item_disabled`

验证：

* item 关闭时 robot-belief tutor 不漂

---

### 3. `tests/test_item_drop.py`

测 shield 的最小语义。

建议至少 9 个：

#### `test_item_drop_shield_constructible`

验证：

* shield item 构造正常

#### `test_inventory_binary_state_supported`

验证：

* inventory 只支持 0/1

#### `test_item_drop_adds_inventory`

验证：

* drop 后 inventory 从 0 到 1

#### `test_shield_consumes_once`

验证：

* shield 使用一次后清空

#### `test_shield_reduces_risk_effect`

验证：

* 有 shield 时 risky traversal 的风险/罚项下降

#### `test_shield_does_not_stack`

验证：

* 重复 drop 不会无限叠加

#### `test_item_drop_location_scope_restricted`

验证：

* 只允许最小 location 范围

#### `test_item_drop_is_counterfactually_evaluable`

验证：

* item action 能进入 counterfactual rollout 评分

#### `test_item_drop_read_only_in_counterfactual_mode`

验证：

* rollout 评分不会污染真实 inventory/env state

---

### 4. `tests/test_v2_intervention_family_integration.py`

测端到端集成。

建议至少 10 个：

#### `test_intervention_family_mode_episode_runs`

验证：

* intervention family mode 能完整跑一集

#### `test_env_info_contains_intervention_scores_for_all_actions`

验证：

* info 里有 WAIT/WARN/UNLOCK/ITEM_DROP 的评分

#### `test_env_info_contains_expected_item_effect`

验证：

* item action 的预期效果被暴露出来

#### `test_wait_warn_unlock_item_comparable`

验证：

* 四类动作都在同一 decision object 中可比较

#### `test_item_drop_changes_decision_in_relevant_case`

验证：

* 某些场景下 ITEM_DROP 真能成为最佳动作

#### `test_warn_still_best_in_warning_friendly_case`

验证：

* warning 有用的场景不会被 item 错抢

#### `test_unlock_still_best_in_topology_case`

验证：

* topology bottle-neck 场景仍倾向 UNLOCK

#### `test_wait_still_best_in_safe_case`

验证：

* 安全场景仍倾向 WAIT

#### `test_robot_belief_tutor_with_item_mode_runs`

验证：

* robot-belief tutor + item family mode 可运行

#### `test_legacy_mode_baseline_unchanged`

验证：

* legacy baseline 不变

---

## C. 我建议再补 3 个特别关键的测试

### `test_item_drop_scoring_uses_same_counterfactual_rollout_as_other_actions`

验证：

* ITEM_DROP 不走单独拍脑袋逻辑
* 而是走与 WAIT/WARN/UNLOCK 同一个 surrogate rollout 比较框架

### `test_planner_inventory_awareness_changes_prefix_score`

验证：

* 有无 shield 时，同一 prefix 的 score 真会变化

### `test_item_disabled_mode_is_equivalent_to_previous_phase7_behavior`

验证：

* `item_drop_enabled=False` 时，Phase 8 代码路径与 Phase 7 行为一致

这三条非常值。

---

# 六、建议的回归验证

这一阶段建议做四种回归：

## 1. legacy path 回归

```bash
python scripts/_diag_l2c1_sweep.py
```

预期：

* 仍然是 `9/80/68/99/100%`

## 2. robot-belief tutor 旧路径 smoke

保留 Phase 7 smoke
确保 item 关闭时行为不漂

## 3. intervention family smoke

建议新增：

```text
scripts/_diag_l2c1_intervention_family_smoke.py
```

只要求：

* 能跑
* 输出四类 action scores
* 输出 chosen action
* 不要求 item 一定最优

## 4. item-drop smoke

建议再加一个更小脚本：

```text
scripts/_diag_l2c1_item_drop_smoke.py
```

只要求：

* inventory 逻辑能跑
* shield 会消耗
* item counterfactual scoring 正常

---

# 七、这一阶段的预期结果

如果 Antigravity 做对了，Phase 8 结束后你应该得到：

## 代码层

* intervention family 有统一 schema
* WARN / UNLOCK 被纳入统一接口
* ITEM_DROP / SHIELD 有最小可运行 baseline
* planner 至少支持最小 inventory-aware scoring
* policy 能统一比较四类动作

## 测试层

* 旧 209 tests 继续通过
* 新增大约 26–36 个 intervention family / item tests
* legacy baseline 继续锁住

## 研究层

* 你终于可以说 teacher side 的 action space 是完整的：

  * communication
  * topology change
  * resource assistance
* 这会非常自然地连接 proposal 里“什么时候该 warn、unlock、还是给工具”这个问题

---

# 八、我对这一步的简短建议

你这次给 Antigravity 时，最值得放在最前面的四句是：

```text
Unify the intervention API first; do not start with item implementation.

Implement only one minimal item type in this phase: shield.

ITEM_DROP must be evaluated under the same counterfactual surrogate rollout framework as WAIT / WARN / UNLOCK.

Keep inventory binary and semantics simple.
```

这样它基本就不容易 scope 爆炸了。

[1]: https://antigravity.google/docs/implementation-plan?utm_source=chatgpt.com "Implementation Plan"
