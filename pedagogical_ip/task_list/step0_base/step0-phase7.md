可以。基于你现在的状态，**Phase 7 最合理的定义**是：

# Phase 7：最小可行的 robot belief over agent belief

也就是先做一个**近似 nested ToM**版本，而不是 full particle POMDP over beliefs。
这一步的目标不是“让 robot 完全懂 agent”，而是让 robot 开始基于：

* agent 现在大概知道什么
* agent 接下来大概会怎么走
* agent 大概会因为什么失败

来决定 **WAIT / WARN / UNLOCK**。

另外，这一步我仍然建议你让 Antigravity 走官方支持的复杂任务流程：
先让它产出 **Implementation Plan**，然后看 **Task List**，执行时保持 **Planning 模式 + Request Review**，最后要求它给 **Walkthrough**。这些流程本身就是 Antigravity 文档里明确支持的。([Google Antigravity][1])

---

# 一、Phase 7 的大致目的

你现在已经有：

* Phase 4：latent world semantics
* Phase 5：patch observation + prefix prediction
* Phase 6：belief-conditioned bounded planning
* 177 tests
* legacy baseline 一直锁住

所以现在最自然的下一步不是再继续扩 agent，而是开始让 **robot 显式建模 agent**。

当前 gap 很清楚：

* agent 已经能输出 `BeliefPlan`
* agent 已经有 `FailureModeEstimate`
* 但 robot 还没有一个明确模块来表示
  **“我认为 agent 的 belief 是什么、它会怎么走、为什么会失败”**

Phase 7 的目标就是补上这一层。

---

# 二、Phase 7 的思路

## 1. 不做 full nested inference

这一阶段不要做：

* full particle POMDP over beliefs
* robot 对未来 observation branching
* full goal uncertainty
* item-drop 干预

先做一个**最小可行近似**：

### Robot 维护

* agent feature belief 的近似副本
* agent cost/risk predictor 参数的近似副本
* agent boundedness 参数的近似副本

### Robot 预测

* `P(agent path prefix | robot's model of the agent)`
* `P(failure modes | predicted prefix)`
* `expected catastrophe reduction if intervene`
* `expected learning gain if wait`（先做启发式）

---

## 2. 先做“surrogate agent model”，再做 intervention policy

我建议 Phase 7 拆成两个层次：

### 7A. RobotBelief + AgentPredictor

先回答：

* robot 如何保存一个对 agent 的近似模型
* robot 如何用这个近似模型 rollout agent 的 prefix

### 7B. InterventionPolicy

再回答：

* robot 在 WAIT / WARN / UNLOCK 中选什么
* 依据是 predicted path、failure mode、learning gain

这比直接把“robot belief + intervention”混在一个 tutor 文件里稳很多。

---

## 3. 先只支持 WAIT / WARN / UNLOCK

这一步不要加 item-drop。
最小对比集就够了：

* `WAIT`
* `WARN`
* `UNLOCK`

然后与你已有的 baseline 对照：

* `oracle_teacher.py`：上界
* `time_aware_door_tutor.py`：heuristic baseline
* `rsa_warning.py`：communication baseline
* 新的 `robot-belief tutor`：最小 nested-belief version

---

## 4. 先让 robot 读 agent 的结构化输出，不要再造新信号

Phase 6 已经有：

* `BeliefPlan`
* `FailureModeEstimate`

所以 Phase 7 最应该做的是让 robot 读这些，而不是重新发明一套评估指标。

---

# 三、Phase 7 的方法

## 方法 A：新增 teacher-side 模块，不改 agent 主干

优先新建：

* `src/teachers/robot_belief.py`
* `src/teachers/agent_predictor.py`
* `src/teachers/intervention_policy.py`

尽量不要重改：

* `feature_belief.py`
* `cost_risk_model.py`
* `belief_planning.py`

## 方法 B：继续 additive migration

建议加新 mode，例如：

* `robot_belief_tutor_mode: false/true`

保留所有旧 tutor 路径不变。

## 方法 C：先做 deterministic, inspectable 近似

robot belief 一开始不要太 fancy。
建议先做：

* shallow copy + optional degradation/noise
* boundedness mismatch
* partial synchronization after each agent observation

这样更容易测，也更适合论文叙事。

---

# 四、我建议你先加的 6 个约束

## 1. 名字建议叫 `approx_robot_belief` 或 `robot_surrogate_belief`

这样更诚实。
因为你这一步不是严格 Bayesian nested belief inference。

## 2. 先不做 goal uncertainty

你原文里提到 “对 goal 的 belief（如果之后做 goal uncertainty）”。
我建议 Phase 7 完全不碰它。
否则 scope 会明显膨胀。

## 3. `expected learning gain if wait` 先做启发式

建议定义成：

* wait 后 predicted uncertainty reduction
* 或 wait 后 agent 自主避险概率提升
  不要做复杂信息论量。

## 4. intervention policy 先做 score-based，不做 RL

可以直接：

```text
intervene_score = catastrophe_reduction_weight * expected_catastrophe_reduction
                + deadline_weight * expected_deadline_improvement
                - autonomy_penalty * intervention_cost
                - pedagogy_penalty * over-helping
```

## 5. robot belief 必须允许“不完全匹配真实 agent”

这一步很关键。
否则 robot 只是把 agent 真状态抄一遍，不算真正 nested modeling。

建议至少支持 config：

* `belief_copy_mode: exact | noisy | stale`
* `boundedness_mismatch`
* `risk_head_mismatch`

## 6. 所有新诊断必须 prefix-based

和 Phase 6 一样，robot 的判断必须建立在 **predicted path prefix** 上，不要偷看全局真值。

---

# 下面是可以直接给 Antigravity 的任务单

```text
Project: pedagogical_ip
Phase: Phase 7 — approximate robot belief over agent belief

Current status
Phase 0 complete: baselines frozen
Phase 1 complete: protocol cleanup
Phase 2 complete: runner platformization
Phase 3 complete: environment interface
Phase 4 complete: latent world semantics
Phase 5 complete: patch observation + prefix prediction
Phase 6 complete: belief-conditioned bounded planning
Current total tests: 177
Legacy V2 baseline remains unchanged

High-level goal
This phase introduces a minimal nested-ToM layer on the teacher side.

The robot should no longer act only on world-facing risk.
Instead, it should maintain an approximate model of:
- what the agent currently believes
- how the agent is likely to plan
- how the agent is likely to fail

Important clarification
This phase should implement an approximate robot-belief-over-agent-belief system.
It is NOT a full particle POMDP over beliefs and NOT full exact nested Bayesian inference.

Core behavior
The robot should maintain an approximate surrogate of the agent and predict:
- the agent’s likely path prefix
- likely failure modes
- expected catastrophe / timeout if the robot waits
- expected effect of WARN or UNLOCK
- optional heuristic learning gain if the robot waits

Scope for this phase
Only support intervention choices among:
- WAIT
- WARN
- UNLOCK

Do NOT add item-drop in this phase.

What this phase should achieve
1. robot-side approximate belief over the agent’s belief state
2. robot-side prediction of the agent’s likely path prefix
3. robot-side prediction of failure causes from that prefix
4. a simple intervention policy that compares WAIT / WARN / UNLOCK
5. comparable baselines:
   - oracle teacher
   - heuristic V2 tutor
   - robot-belief tutor

What this phase should NOT do
- do NOT implement full nested exact inference
- do NOT implement goal uncertainty yet
- do NOT redesign the agent planner family
- do NOT redesign warning science
- do NOT add item-drop
- do NOT remove any legacy teacher path
- do NOT let the robot directly use hidden true world values except where explicitly allowed for oracle baselines

Design preferences
- additive migration only
- create new teacher-side modules rather than modifying agent internals
- use structured, inspectable outputs
- keep the robot’s model approximate and configurable
- use prefix-based diagnostics only

Strongly suggested new files
- src/teachers/robot_belief.py
- src/teachers/agent_predictor.py
- src/teachers/intervention_policy.py

Suggested file responsibilities

robot_belief.py
- maintain an approximate surrogate of agent belief
- support configurable mismatch / noise / staleness
- store approximate agent competence parameters

agent_predictor.py
- run the agent model forward from the robot’s surrogate
- produce predicted path prefix
- produce predicted BeliefPlan / failure-mode-like outputs

intervention_policy.py
- compare WAIT / WARN / UNLOCK
- use predicted catastrophe reduction, deadline pressure, and autonomy cost
- output structured intervention decisions and scores

Suggested config additions
- robot_belief_tutor_mode: false/true
- belief_copy_mode: exact | noisy | stale
- boundedness_mismatch
- predictor_prefix_horizon
- wait_value_weight
- catastrophe_weight
- autonomy_penalty
- warning_cost
- unlock_cost

Files to inspect first
- src/teachers/time_aware_door_tutor.py
- src/teachers/oracle_teacher.py
- src/teachers/particle_teacher.py
- src/teachers/rsa_warning.py
- src/agents/belief_planning.py
- src/agents/feature_belief.py
- src/agents/cost_risk_model.py
- src/envs/lattice_v2_env.py
- src/envs/lattice_v2_runner.py
- configs/teacher.yaml
- configs/agent.yaml
- configs/experiment.yaml
- tests/test_v2_belief_planning_integration.py

Recommended implementation order
Priority 1:
Create a RobotBelief representation that stores an approximate surrogate of:
- agent latent belief
- agent predictor parameters
- boundedness parameters

Priority 2:
Create an AgentPredictor that rolls out the agent’s likely prefix from the surrogate model.

Priority 3:
Create a simple InterventionPolicy that scores WAIT / WARN / UNLOCK.

Priority 4:
Expose structured robot-side diagnostics through env/runner info when this mode is enabled.

Required output format before implementation
1. concise diagnosis of current teacher limitations
2. proposed approximate robot-belief design
3. proposed agent-predictor design
4. proposed intervention scoring design
5. file-by-file change plan
6. config strategy
7. proposed tests
8. risks of over-engineering
9. minimal implementation order

Acceptance criteria
Phase 7 is successful only if:
- existing tests still pass unless intentionally updated
- new robot-belief tests pass
- legacy baseline path remains reproducible
- heuristic tutor path remains runnable
- robot-belief tutor path remains runnable
- robot decisions depend on predicted agent path prefix, not just current local risk
- oracle / heuristic / robot-belief tutors become directly comparable

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

1. robot 真的有一个近似 agent surrogate
2. robot 真的在预测 agent prefix
3. intervention 真的取决于 predicted prefix / failure mode
4. 旧 tutor 路径和 legacy baseline 都不漂移

---

## A. 旧测试先继续全过

先跑：

```bash
python -m pytest tests/ -v --tb=short
```

预期：

* 当前 **177 tests 全通过**

---

## B. 建议新增测试文件

我建议新增 4 个主测试文件。

---

### 1. `tests/test_robot_belief.py`

测 robot surrogate 本身。

建议至少 7 个：

#### `test_robot_belief_init_runs`

验证：

* RobotBelief 能正常初始化

#### `test_robot_belief_can_copy_agent_belief_exact`

验证：

* exact 模式下能正确复制 agent belief

#### `test_robot_belief_noisy_mode_differs_from_exact`

验证：

* noisy 模式下 surrogate 与 exact 不同

#### `test_robot_belief_stale_mode_updates_less_frequently`

验证：

* stale 模式不会每步完全同步

#### `test_robot_belief_tracks_boundedness_params`

验证：

* search_budget / heuristic_noise 等 agent competence 参数会进入 robot surrogate

#### `test_robot_belief_does_not_mutate_agent_state`

验证：

* robot surrogate 更新不会污染真实 agent belief

#### `test_robot_belief_config_switch`

验证：

* `robot_belief_tutor_mode` 或相关开关可切换路径

---

### 2. `tests/test_agent_predictor.py`

测 robot 对 agent 的 prefix 预测。

建议至少 7 个：

#### `test_agent_predictor_runs`

验证：

* AgentPredictor 能从 robot surrogate rollout 一个 prefix

#### `test_agent_predictor_returns_prefix`

验证：

* 输出 prefix，而不是单步动作

#### `test_agent_predictor_respects_prefix_horizon`

验证：

* `predictor_prefix_horizon` 生效

#### `test_agent_predictor_changes_with_belief_mismatch`

验证：

* surrogate belief 改变后，预测 prefix 会改变

#### `test_agent_predictor_changes_with_boundedness_mismatch`

验证：

* search_budget / heuristic noise 改变后，预测 prefix 会改变

#### `test_agent_predictor_uses_failure_mode_estimates`

验证：

* predictor 输出能带上 failure-mode-like 分析或兼容结构

#### `test_agent_predictor_is_read_only`

验证：

* rollout 不会改真实 env / agent state

---

### 3. `tests/test_intervention_policy.py`

测 WAIT / WARN / UNLOCK 评分。

建议至少 8 个：

#### `test_intervention_policy_scores_actions`

验证：

* 能给 WAIT / WARN / UNLOCK 都打分

#### `test_wait_preferred_when_predicted_path_is_safe`

验证：

* 预测 path 安全时倾向 WAIT

#### `test_warn_preferred_when_prefix_risk_is_fixable_by_warning`

验证：

* warning 有用时倾向 WARN

#### `test_unlock_preferred_when_topology_change_helps_more`

验证：

* 拓扑瓶颈场景下倾向 UNLOCK

#### `test_autonomy_penalty_discourages_overhelping`

验证：

* autonomy penalty 上升时，robot 更少干预

#### `test_deadline_pressure_changes_choice`

验证：

* deadline 紧时 intervention 选择会改变

#### `test_learning_gain_heuristic_affects_wait_score`

验证：

* heuristic learning gain 会影响 WAIT 评分

#### `test_policy_outputs_structured_decision`

验证：

* 输出结构化 decision，不是自由文本

---

### 4. `tests/test_v2_robot_belief_integration.py`

测 V2 集成。

建议至少 7 个：

#### `test_robot_belief_tutor_episode_runs`

验证：

* robot-belief tutor 模式能完整跑一集

#### `test_env_info_contains_robot_prediction`

验证：

* env/info 暴露 robot 侧 predicted prefix

#### `test_env_info_contains_robot_failure_estimate`

验证：

* env/info 暴露 robot 侧 failure estimates

#### `test_env_info_contains_intervention_scores`

验证：

* env/info 暴露 WAIT / WARN / UNLOCK 评分

#### `test_robot_belief_tutor_differs_from_heuristic_tutor`

验证：

* robot-belief tutor 的决策不只是 heuristic tutor 的简单复制

#### `test_oracle_heuristic_robotbelief_all_runnable`

验证：

* 三类 tutor 都可运行并可比较

#### `test_legacy_mode_baseline_unchanged`

验证：

* legacy baseline 不变

---

## C. 我建议再补两个特别关键的测试

### `test_robot_decision_depends_on_predicted_prefix_not_only_local_cell`

验证：

* 在当前局部风险相同、但未来 prefix 风险不同的场景下
* robot 会做不同决策

这是 Phase 7 最关键的科学测试之一。

### `test_robot_belief_mismatch_can_change_intervention_choice`

验证：

* 当 robot 对 agent 的能力估计不同（比如 budget 更低）时
* intervention choice 会改变

这个能证明你不是在“假装做 nested ToM”。

---

# 六、建议的回归验证

这一阶段建议做四种回归：

## 1. legacy path 回归

```bash
python scripts/_diag_l2c1_sweep.py
```

预期：

* 仍然是 `9/80/68/99/100%`

## 2. heuristic tutor smoke

保留你已有的 heuristic tutor 路径 smoke

## 3. belief-planning smoke

保留上一阶段 smoke

## 4. robot-belief smoke

建议新增一个轻量脚本，比如：

```text
scripts/_diag_l2c1_robot_belief_smoke.py
```

只要求：

* 能跑
* 输出 predicted prefix
* 输出 intervention scores
* 输出 chosen intervention
* 不要求数值立即优于 heuristic tutor

---

# 七、这一阶段的预期结果

如果 Antigravity 做对了，Phase 7 结束后你应该得到：

## 代码层

* teacher 侧新增 robot surrogate / predictor / policy 三层
* robot 的决策依据从“当前 risk”升级到“预测的 agent prefix”
* 旧 tutor 路径全部保留

## 测试层

* 旧 177 tests 继续通过
* 新增大约 24–31 个 robot-belief / intervention tests
* legacy baseline 继续锁住

## 研究层

* 你终于可以直接比较：

  * oracle teacher
  * heuristic tutor
  * robot-belief tutor
* 这会非常贴近你的 proposal 主线：
  **pedagogical intervention 不是只看世界风险，而是看 learner 将怎么走、将怎么错。**

---

# 八、我对这一步的简短建议

你这次给 Antigravity 时，最值得放在最前面的三句是：

```text
This phase implements an approximate robot belief over the agent’s belief, not full exact nested inference.

Robot decisions must depend on predicted agent path prefixes, not only current local risk.

Only support WAIT / WARN / UNLOCK in this phase.
```

这样它基本就不容易 scope 爆炸了。

[1]: https://antigravity.google/docs/implementation-plan?utm_source=chatgpt.com "Implementation Plan"
