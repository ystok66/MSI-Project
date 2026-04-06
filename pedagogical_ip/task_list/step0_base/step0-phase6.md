整体看，**这个 Phase 6 方向是对的，可以开工**。
但我建议你先把它再收紧成一句更准确的话：

> **Phase 6 不是 full belief-space search。**
> 它是：**agent 用 belief 生成的 `cost_hat / risk_hat / uncertainty` 来规划，并输出更丰富的决策诊断。**

也就是说，这一阶段最重要的是把 agent 从“看真值 / 看半真值”切到“看 belief 估计值”，而**不是**去实现完整 POMDP tree search。pomdp-py 本身最值得借鉴的也正是这种把 state、action、observation、belief 分开的接口思路，而不是要求你把项目迁成 exact solver。([GitHub][1])

另外，你前面一直采用的 antigravity 工作流在这一步仍然最合适：**先让它产出 implementation plan，再看 task list，再执行，最后看 walkthrough；终端继续设成 Request Review。** 这些都是官方文档支持的标准流程。([Google Antigravity][2])

---

# 我建议你先补的 6 个小修改

## 1. 把 “belief-space planner” 改成 “belief-conditioned planner”

这个名字更准确。
因为你现在真正要做的是：

* 从 belief 取 `cost_hat / risk_hat / uncertainty`
* 在 expected / MAP world 上做 bounded A*
* 输出 prefix 和 failure diagnostics

这不是完整意义上的 belief tree planning。
我建议文档里直接写：

> Phase 6 implements a **belief-conditioned bounded planner**, not a full belief tree planner.

---

## 2. `why_this_path` 不要返回自然语言，先返回结构化诊断

建议让 antigravity 输出一个结构化对象，而不是一句文字。

例如：

```python
{
  "chosen_prefix": ...,
  "expected_cost": ...,
  "expected_risk": ...,
  "uncertainty": ...,
  "deadline_slack": ...,
  "dominant_reason": "lower_risk" | "lower_cost" | "lower_uncertainty" | "deadline_pressure",
  "runner_up_gap": ...
}
```

这样以后：

* 好测试
* 好写日志
* 好喂给 teacher
* 不会绑定到某种说明文本格式

---

## 3. `action_confidence` 要先选一个简单、可解释的定义

不建议这一阶段做 learned calibration。
最稳的是定义成：

* 最优 action / path 与次优 action / path 的 score gap
* 或者 softmax gap / margin

比如：

```python
confidence = score_2nd_best - score_best
```

然后再做一个归一化版本也行。
重点是：**Phase 6 要的是可解释 confidence，不是“真正概率意义上的 confidence”。**

---

## 4. `estimate_failure_modes()` 先做启发式分解，不要做 fancy classifier

我建议这一阶段只输出一组简单 failure mode scores：

* `high_cumulative_risk`
* `high_uncertainty`
* `deadline_miss`
* `no_safe_route`
* `warning_ignored_or_insufficient`

本质上就是对当前 prefix 做规则化分析。
不要让 antigravity 把它做成复杂判别器。

---

## 5. boundedness 参数要显式入 config

这一阶段最该固定的是“agent competence knobs”。

我建议至少加这些 config：

* `search_budget`
* `heuristic_noise_std`
* `prefix_horizon`
* `confidence_temperature` 或 `confidence_mode`
* `risk_weight`
* `uncertainty_weight`
* `replan_every_step: true/false`

否则后面你说“分析不同 agent competence”时，会缺少干净控制量。

---

## 6. 尽量新建 `belief_planning.py`，不要把 `bounded_agent.py` 塞太满

我更推荐：

* `src/agents/belief_planning.py`
* `src/agents/bounded_agent.py` 只做薄适配

理由很简单：

* Phase 4 做了 latent semantics
* Phase 5 做了 patch/prefix diagnostics
* Phase 6 会明显增加 planning 逻辑复杂度

这时候再把所有东西继续堆进 `bounded_agent.py`，后面会很难收拾。

---

# Phase 6 的大致目的

Phase 4 已经统一了 latent world。
Phase 5 已经有 patch observation 和 prefix diagnostics。
所以 Phase 6 的真正目标是：

> **让 agent 的决策正式建立在自己的 belief 上，而不是建立在环境真值或半真值捷径上。**

也就是每一步：

1. 观测局部 patch
2. 更新 observed cells 的 latent belief
3. 用 joint heads 预测 `cost_hat / risk_hat / uncertainty`
4. 在 bounded search 下规划
5. 输出 action + prefix + confidence + failure diagnostics

---

# Phase 6 的思路

这一步建议坚持三条：

### 1. 不重写 planner

还是现有 bounded A* 框架。
变化的是它的输入：不再是旧 risk-only path，而是 belief 产生的 joint estimates。

### 2. 不做 full POMDP

不做 belief tree expansion，不做 observation branching。
只做 **belief-conditioned planning on the current posterior estimate**。

### 3. 继续 additive migration

建议加一个新开关，比如：

* `belief_planning_mode: false/true`

这样你可以同时保留：

* legacy path
* latent path
* latent + patch path
* latent + patch + belief-planning path

---

# Phase 6 的方法

## 方法 A：先让 agent 从 belief 规划，再补 diagnostics

先做核心接口：

* `plan_from_belief(...)`
* `predict_path_prefix(horizon=K)`
* `estimate_failure_modes(...)`

## 方法 B：所有新输出都走结构化 schema

不要直接返回松散 tuple。
建议增加新的 dataclass，比如：

* `BeliefPlan`
* `FailureModeEstimate`

## 方法 C：先做 deterministic diagnostics，再谈 stochasticity

confidence / failure modes 先都做 deterministic, explainable 版本。

---

# 下面是可以直接给 Antigravity 的任务单

```text
Project: pedagogical_ip
Phase: Phase 6 — belief-conditioned bounded planning

Current status
Phase 0 complete: baselines frozen
Phase 1 complete: protocol cleanup
Phase 2 complete: runner platformization
Phase 3 complete: environment interface
Phase 4 complete: latent world semantics
Phase 5 complete: patch observation + prefix prediction
Current total tests: 150
Legacy V2 baseline remains unchanged

High-level goal
This phase upgrades the agent from:
- planning mostly over environment-facing scores

to:
- planning explicitly from its own learned belief estimates

Important clarification
This phase should implement a belief-conditioned bounded planner.
It is NOT a full belief-tree planner and NOT a full exact POMDP solver.

Core behavior per step
At each step, the agent should:
1. observe a local patch
2. update latent belief for observed cells
3. compute predicted cost / predicted risk / uncertainty from belief
4. plan in the expected / MAP world using bounded A*
5. output:
   - next_action
   - planned_prefix
   - action_confidence
   - structured why_this_path diagnostics
   - structured failure mode estimates

What this phase should achieve
1. the agent should plan from belief-derived predictions, not from true hidden values
2. agent competence should become configurable through boundedness knobs
3. path-choice diagnostics should become explicit and structured
4. failure-mode estimates should be exposed for later teacher use

What this phase should NOT do
- do NOT implement full POMDP tree search
- do NOT branch on hypothetical future observations
- do NOT redesign the teacher yet
- do NOT add robot nested belief yet
- do NOT replace the existing planner algorithm unless clearly necessary
- do NOT remove legacy / latent / patch paths

Design preferences
- additive migration only
- keep current bounded A* search structure
- change planner inputs, not planner family
- use structured diagnostics, not natural-language explanations
- prefer a new file such as src/agents/belief_planning.py over bloating bounded_agent.py

Suggested new interfaces
Please strongly consider adding:
- plan_from_belief(...)
- predict_path_prefix(horizon=K)
- estimate_failure_modes(...)

Suggested structured outputs
For planning:
- chosen action
- chosen path prefix
- expected cost
- expected risk
- uncertainty
- deadline slack
- runner_up_gap
- dominant_reason

For failure modes:
- high_cumulative_risk
- high_uncertainty
- deadline_miss
- no_safe_route
- warning_insufficient

Suggested config additions
- belief_planning_mode: false/true
- search_budget
- heuristic_noise_std
- prefix_horizon
- confidence_mode or confidence_temperature
- risk_weight
- uncertainty_weight
- replan_every_step

Files to inspect first
- src/agents/bounded_agent.py
- src/agents/planner_astar.py
- src/agents/feature_belief.py
- src/agents/cost_risk_model.py
- src/agents/observation_model.py
- src/agents/warning_update.py
- src/envs/lattice_v2_env.py
- src/envs/lattice_v2_runner.py
- configs/agent.yaml
- configs/env.yaml
- configs/experiment.yaml
- tests/test_prefix_prediction.py
- tests/test_patch_belief_update.py
- tests/test_v2_patch_prefix_integration.py

Recommended implementation order
Priority 1:
Create a belief-planning layer that computes planning scores from belief-derived:
- cost_hat
- risk_hat
- uncertainty

Priority 2:
Add a structured planning result object.
Do not use loose tuples for the new path.

Priority 3:
Add action confidence based on score margin / runner-up gap.
Do not do learned calibration.

Priority 4:
Add structured failure-mode estimates.
Use simple heuristic decomposition first.

Priority 5:
Expose these outputs through env/runner info in the new path.

Required output format before implementation
1. concise diagnosis of remaining gap between patch-prefix agent and belief-conditioned planning
2. proposed belief-planning design
3. file-by-file change plan
4. config strategy
5. proposed tests
6. risks of over-engineering
7. minimal implementation order

Acceptance criteria
Phase 6 is successful only if:
- existing tests still pass unless intentionally updated
- new belief-planning tests pass
- legacy baseline path remains reproducible
- latent + patch path remains runnable
- belief-planning path remains runnable
- agent action selection demonstrably depends on belief-derived cost/risk/uncertainty
- agent diagnostics are structured and inspectable

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

# 建议的测试内容

这一阶段的测试要证明三件事：

1. agent 真的是从 belief 规划
2. boundedness knobs 真能改变行为
3. diagnostics 是结构化、稳定、可供 teacher 读取的

---

## A. 旧测试先继续全过

先跑：

```bash
python -m pytest tests/ -v --tb=short
```

预期：

* 当前 **150 tests 全通过**

---

## B. 建议新增测试文件

我建议新增 3 个主测试文件。

---

### 1. `tests/test_belief_planning.py`

测 planning 输入是否真的来自 belief。

建议至少 8 个：

#### `test_plan_from_belief_runs`

验证：

* belief-planning path 能正常运行并返回结构化结果

#### `test_plan_from_belief_uses_predicted_cost`

验证：

* 改变 `cost_hat` 会影响 action/path 选择

#### `test_plan_from_belief_uses_predicted_risk`

验证：

* 改变 `risk_hat` 会影响 action/path 选择

#### `test_plan_from_belief_uses_uncertainty`

验证：

* 提高 uncertainty penalty 会改变偏好

#### `test_belief_planning_differs_from_true_world_when_belief_is_wrong`

验证：

* 当 belief 偏差很大时，行为能与 oracle/true-world 规划不同
* 这是“从 belief 规划”的关键证据

#### `test_search_budget_affects_behavior`

验证：

* 改 `search_budget` 会影响计划质量或路径

#### `test_heuristic_noise_affects_behavior`

验证：

* 改 `heuristic_noise_std` 会影响路径或 confidence

#### `test_belief_planning_config_switch`

验证：

* `belief_planning_mode` 能切换新旧路径

---

### 2. `tests/test_planning_diagnostics.py`

测结构化输出。

建议至少 7 个：

#### `test_planning_result_contains_expected_fields`

验证：

* result 包含：

  * action
  * planned_prefix
  * expected_cost
  * expected_risk
  * uncertainty
  * dominant_reason
  * runner_up_gap

#### `test_action_confidence_computable`

验证：

* confidence 有定义且数值稳定

#### `test_action_confidence_changes_with_runner_up_gap`

验证：

* 最优/次优差距变大时 confidence 合理变化

#### `test_dominant_reason_is_valid_enum`

验证：

* `dominant_reason` 是可控的结构化取值，而不是自由文本

#### `test_failure_modes_computable`

验证：

* failure mode scores 能计算出来

#### `test_failure_modes_reflect_path_risk`

验证：

* 高 cumulative risk 时，`high_cumulative_risk` 分数更高

#### `test_failure_modes_reflect_deadline_pressure`

验证：

* 时间紧时，`deadline_miss` 或等价项上升

---

### 3. `tests/test_v2_belief_planning_integration.py`

测 env/runner 集成。

建议至少 6 个：

#### `test_belief_planning_episode_runs`

验证：

* latent + patch + belief_planning mode 能完整跑一集

#### `test_env_info_contains_planning_diagnostics`

验证：

* env/info 暴露结构化 planning result

#### `test_env_info_contains_failure_modes`

验证：

* env/info 暴露 failure mode estimates

#### `test_predict_path_prefix_exposed`

验证：

* teacher / caller 能读取 predicted path prefix

#### `test_warning_can_affect_belief_based_plan`

验证：

* warning 仍然能改变 belief-based path 选择

#### `test_legacy_mode_baseline_unchanged`

验证：

* legacy path baseline 不变

---

## C. 我建议再加两个很关键的测试

### `test_planning_diagnostics_are_read_only`

验证：

* 读取 planning result / failure modes 不会改 env / belief / planner state

### `test_same_belief_same_plan_fixed_seed`

验证：

* 相同 belief、相同 seed、相同 config 下结果可复现

这两条很适合防 Phase 6 最容易出现的 bug：

* diagnostics 带副作用
* 隐式随机性把行为搞漂

---

# 建议的回归验证

这一阶段建议做三种回归：

## 1. legacy path 回归

```bash
python scripts/_diag_l2c1_sweep.py
```

预期：

* 仍然是 `9/80/68/99/100%`

## 2. latent + patch path smoke

保留上一阶段 smoke

* 只要求可跑

## 3. belief-planning smoke

建议新增一个轻量脚本，比如：

```text
scripts/_diag_l2c1_belief_plan_smoke.py
```

只要求：

* 能跑
* 输出 structured planning diagnostics
* 输出 failure mode estimates
* 不要求和 legacy baseline 数值一样

---

# 这一阶段的预期结果

如果 antigravity 做对了，Phase 6 结束后你应该得到：

## 代码层

* agent 已经有独立的 belief-planning 层
* planner 还是原有 family，但输入正式来自 belief predictions
* planning result 和 failure modes 都是结构化对象
* env/runner 可以把这些结果暴露出来

## 测试层

* 旧 150 tests 继续通过
* 新增大约 20–23 个 belief-planning / diagnostics tests
* legacy baseline 继续锁住

## 研究层

* 你已经能真正说：

  * agent 是 partial-observation 的
  * agent 是 bounded-rational 的
  * agent 是按 learned belief 在规划的
* 这样下一阶段就非常自然能进入：

  * robot 读取 predicted path prefix
  * robot 建模 agent belief
  * teacher 做真正 pedagogical intervention policy

---

# 我对这一步的最终建议

你这次给 antigravity 时，最值得放在最前面的两句是：

```text
This phase implements belief-conditioned bounded planning, not full belief-tree planning.
Do not change the planner family unless clearly necessary.
```

再加一句也很好：

```text
Structured diagnostics are first-class outputs in this phase.
Do not return free-form explanations.
```

这样它基本就不容易走偏了。

[1]: https://github.com/h2r/pomdp-py?utm_source=chatgpt.com "GitHub - h2r/pomdp-py: A framework to build and solve ..."
[2]: https://antigravity.google/docs/implementation-plan?utm_source=chatgpt.com "Implementation Plan - Google Antigravity Documentation"
