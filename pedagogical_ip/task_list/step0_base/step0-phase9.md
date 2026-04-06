可以。基于你现在的状态，**Phase 9 不再是“做新算法”，而是“把论文级实验系统和指标系统做完整”**。

你已经有了：

- 统一 latent world
- patch observation + prefix prediction
- belief-conditioned planning
- approximate robot belief
- unified intervention family + shield
- 245 tests
- 且 legacy baseline 一直锁住

所以 **Phase 9 的唯一核心任务** 是：

> **把实验问题正式变成可回答、可对比、可画图、可写论文的 evaluation framework。**

也就是说，这一步最重要的不是再写一个 tutor，而是把下面三件事钉死：

1. **指标系统**：到底怎么定义“帮助完成任务”和“帮助学习”
2. **实验矩阵**：不同 learner / tutor / environment 条件怎么组合
3. **结果产物**：每个 run 输出什么、聚合成什么表、画什么图

另外，这一步很适合继续让 Antigravity 用你前面一直在用的流程：先出 **Implementation Plan**，再看 **Task List**，执行时用 **Planning** 模式和 **Request Review**，结束后要 **Walkthrough**。这些都是它官方文档里已经支持的复杂任务工作流。([Google Antigravity][1])

---

# Phase 9 的大致目的

Phase 9 的目标不是证明“代码能跑”，而是回答 proposal 里的核心问题：

**机器人是在帮 agent 完成任务，还是在帮 agent 学会自己完成任务？**

所以这一阶段需要把现有系统整理成三层 evaluation：

### 1. 任务表现

- success rate
- death rate
- timeout rate
- cumulative cost
- cumulative risk
- intervention count / intervention budget

### 2. 学习表现

- no-robot transfer performance
- risk prediction calibration
- cost prediction error
- uncertainty reduction（visited cells + nearby observed cells）

### 3. pedagogical 表现

- expected information gain
- boredom proxy
- frustration proxy
- intervention timing quality

这一步的主线不是“更强 agent”，而是“更清楚的 measurement”。

---

# Phase 9 的思路

## 1. 先统一 logging / metrics schema，再做 sweep

不要先大规模跑实验，再回头想日志字段不够。
最稳的顺序是：

- 统一 `episode summary`
- 统一 `step-level optional log`
- 统一 `aggregate metrics`
- 再做 experiment matrix

## 2. 先做最小完整矩阵，再扩

不要一下子把所有组合全开。
建议先定义一个**最小论文矩阵**：

### Agent

- weak
- medium
- strong

### Teacher

- no_tutor
- warning_only
- unlock_only
- item_only
- heuristic_mixed
- oracle
- robot_belief

### Environment

- easy / medium / hard

先把这个 3 × 7 × 3 的骨架跑通。
trap density、deadline pressure、cue strength 可以作为第二层扩展。

## 3. transfer evaluation 必须单独设计

“帮助学习”最关键的一点是：
**agent 在没有 tutor 时，后续表现有没有更好。**

所以 Phase 9 里一定要有两种评估：

- **with-tutor online performance**
- **post-training no-tutor transfer performance**

这是这一步最值得单独强调的一点。

## 4. boredom / frustration 只做启发式 proxy

不要这一步突然做复杂 affective model。
建议保持为 heuristic:

- boredom proxy：低信息增益 + 时间/代价持续增加
- frustration proxy：高不确定 + 高代价 + 连续失败/回退

---

# Phase 9 的方法

## 方法 A：新增 evaluation 层，不重写核心 agent/tutor

我建议主要新增：

- `src/metrics/phase9_metrics.py`
- `src/metrics/transfer_eval.py`
- `scripts/run_phase9_matrix.py`
- `scripts/plot_phase9_results.py`

以及必要时一个：

- `configs/phase9_eval.yaml`

## 方法 B：以 config 驱动矩阵，而不是硬编码

最好让 Antigravity 把矩阵定义放到 config 里，而不是脚本里写一堆 for loop。

## 方法 C：每类图表对应一个明确指标表

建议至少保证下面几类结果表能自动导出：

- online task metrics table
- transfer metrics table
- intervention usage table
- uncertainty / calibration table
- tradeoff summary table

---

# 下面是可以直接给 Antigravity 的任务单

```text
Project: pedagogical_ip
Phase: Phase 9 — evaluation system and experiment matrix

Current status
Phase 0 complete: baselines frozen
Phase 1 complete: protocol cleanup
Phase 2 complete: runner platformization
Phase 3 complete: environment interface
Phase 4 complete: latent world semantics
Phase 5 complete: patch observation + prefix prediction
Phase 6 complete: belief-conditioned bounded planning
Phase 7 complete: approximate robot belief
Phase 8 complete: unified intervention family + shield
Current total tests: 245
Legacy V2 baseline remains unchanged

High-level goal
This phase is not about adding new core algorithms.
It is about building a paper-ready evaluation system that can answer:

Is the robot helping task completion, helping learning, or both?

Core evaluation dimensions
1. Task performance
2. Learning / transfer performance
3. Pedagogical quality of intervention

What this phase should achieve
1. a unified metrics/logging layer for Phase 9 experiments
2. a configurable experiment matrix over agent / teacher / environment conditions
3. transfer evaluation without tutor after tutor-assisted experience
4. aggregation-ready outputs for tables and plots
5. minimal plotting/report scripts for tradeoff visualization

What this phase should NOT do
- do NOT redesign the agent
- do NOT redesign teacher policies
- do NOT add new intervention types
- do NOT add new planner families
- do NOT break legacy baselines
- do NOT over-engineer affective modeling

Priority metrics to implement

Task metrics
- success_rate
- death_rate
- timeout_rate
- cumulative_cost
- cumulative_risk
- intervention_count
- intervention_budget_used

Learning / transfer metrics
- no_tutor_transfer_success
- no_tutor_transfer_cost
- no_tutor_transfer_risk
- risk_prediction_calibration
- cost_prediction_error
- uncertainty_reduction_visited
- uncertainty_reduction_nearby

Pedagogical metrics
- expected_information_gain
- boredom_proxy
- frustration_proxy
- intervention_timing_quality

Minimum experiment matrix

Agent levels
- weak
- medium
- strong

Teacher conditions
- no_tutor
- warning_only
- unlock_only
- item_only
- heuristic_mixed
- oracle
- robot_belief

Environment conditions
- easy
- medium
- hard

Design preferences
- config-driven experiment matrix
- unified episode summary schema
- optional step-level logging
- separate online metrics from transfer metrics
- make outputs easy to aggregate into CSV/JSON tables and plots

Strongly suggested files to inspect first
- src/metrics/
- src/logging/episode_logger.py
- src/logging/visualize.py
- src/envs/lattice_v2_env.py
- src/envs/lattice_v2_runner.py
- src/teachers/intervention_policy.py
- src/agents/belief_planning.py
- configs/experiment.yaml
- configs/teacher.yaml
- configs/agent.yaml
- scripts/_diag_l2c1_sweep.py

Strongly suggested new files
- src/metrics/phase9_metrics.py
- src/metrics/transfer_eval.py
- scripts/run_phase9_matrix.py
- scripts/plot_phase9_results.py
- configs/phase9_eval.yaml

Required output format before implementation
1. concise diagnosis of current evaluation gaps
2. proposed metric schema
3. proposed experiment matrix schema
4. proposed episode summary / aggregate output format
5. file-by-file change plan
6. config strategy
7. proposed tests
8. risks of over-engineering
9. minimal implementation order

Acceptance criteria
Phase 9 is successful only if:
- existing tests still pass unless intentionally updated
- new metrics/matrix tests pass
- legacy baseline path remains reproducible
- experiment matrix runs end-to-end
- online vs transfer metrics are clearly separated
- results can support help-vs-learning tradeoff analysis
- at least one plotting/report pipeline is runnable

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

这一步的测试重点不是算法行为，而是：

- 指标是否定义清楚
- transfer 是否和 online 分开
- matrix 是否可运行
- 输出是否可聚合

## A. 旧测试先继续全过

先跑：

```bash
python -m pytest tests/ -v --tb=short
```

预期：

- 当前 **245 tests 全通过**

---

## B. 建议新增测试文件

我建议新增 4 个主测试文件。

---

### 1. `tests/test_phase9_metrics.py`

测指标定义本身。

建议至少 10 个：

#### `test_task_metrics_computable`

验证：

- success/death/timeout/cost/risk/intervention_count 可算

#### `test_learning_metrics_computable`

验证：

- calibration / cost error / uncertainty reduction 可算

#### `test_pedagogical_metrics_computable`

验证：

- information gain / boredom proxy / frustration proxy / timing quality 可算

#### `test_online_and_transfer_metrics_separated`

验证：

- online metrics 和 transfer metrics 不混

#### `test_uncertainty_reduction_visited_computable`

验证：

- visited cells uncertainty reduction 有定义

#### `test_uncertainty_reduction_nearby_computable`

验证：

- nearby observed cells uncertainty reduction 有定义

#### `test_boredom_proxy_monotonic_in_low_info_high_cost_case`

验证：

- 低信息增益+高代价时 boredom proxy 更高

#### `test_frustration_proxy_monotonic_in_high_uncertainty_failure_case`

验证：

- 高不确定+连续失败时 frustration proxy 更高

#### `test_timing_quality_prefers_timely_intervention`

验证：

- timing quality 对早/迟 intervention 有合理区分

#### `test_metrics_schema_stable`

验证：

- 输出字段 schema 稳定

---

### 2. `tests/test_transfer_eval.py`

测 no-tutor transfer 评估。

建议至少 8 个：

#### `test_transfer_eval_runs`

验证：

- transfer evaluation 能跑

#### `test_transfer_eval_uses_no_tutor_policy`

验证：

- transfer 阶段确实没有 tutor

#### `test_transfer_metrics_recorded_separately`

验证：

- transfer 指标单独记录

#### `test_transfer_after_training_differs_from_online_metrics`

验证：

- transfer 结果不会和 online metrics 混成同一条记录

#### `test_transfer_eval_respects_agent_checkpoint_or_state_copy`

验证：

- transfer 是从训练后的 agent 状态继续，而不是重新初始化成无学习状态

#### `test_transfer_eval_read_only_to_training_logs`

验证：

- transfer 不污染 online 训练日志

#### `test_transfer_eval_reproducible_with_seed`

验证：

- fixed seed 可复现

#### `test_transfer_eval_handles_multiple_agent_strengths`

验证：

- weak/medium/strong 都可跑

---

### 3. `tests/test_experiment_matrix.py`

测矩阵系统本身。

建议至少 9 个：

#### `test_phase9_matrix_config_loads`

验证：

- `phase9_eval.yaml` 可读

#### `test_matrix_expands_agent_teacher_env_grid`

验证：

- agent/teacher/env 组合能正确展开

#### `test_matrix_can_filter_subset`

验证：

- 可以跑子集，不必每次全矩阵

#### `test_matrix_job_schema_stable`

验证：

- 每个实验 job 的 schema 稳定

#### `test_matrix_runs_small_smoke_subset`

验证：

- 一个小子集能跑通端到端

#### `test_matrix_outputs_episode_summaries`

验证：

- 每个 job 都有标准化 episode summary

#### `test_matrix_outputs_aggregate_metrics`

验证：

- 有聚合结果输出

#### `test_matrix_keeps_online_and_transfer_outputs_distinct`

验证：

- online/transfer 输出分开

#### `test_matrix_legacy_path_still_reproducible`

验证：

- legacy baseline 可继续回归

---

### 4. `tests/test_phase9_reporting.py`

测结果产物。

建议至少 7 个：

#### `test_results_table_export_runs`

验证：

- 表格导出可运行

#### `test_tradeoff_plot_data_preparable`

验证：

- help-vs-learning plot 数据可准备出来

#### `test_intervention_usage_plot_data_preparable`

验证：

- intervention usage plot 数据可准备出来

#### `test_transfer_plot_data_preparable`

验证：

- transfer comparison plot 数据可准备出来

#### `test_report_schema_contains_required_columns`

验证：

- 报表字段齐全

#### `test_plotting_script_handles_empty_subset_gracefully`

验证：

- 子集为空时 plotting 不崩

#### `test_reporting_is_read_only`

验证：

- report/plot 阶段不修改原始结果

---

## C. 我建议再补 4 个特别关键的测试

### `test_help_vs_learning_tradeoff_computable`

验证：

- 你能从输出里同时拿到
  - online task gain
  - transfer learning gain
  - intervention cost

- 形成真正的 tradeoff 数据

### `test_item_only_warning_only_unlock_only_comparable`

验证：

- 三类单一 intervention condition 在同一 schema 下可比较

### `test_robot_belief_vs_heuristic_vs_oracle_comparable`

验证：

- 关键 tutor family 输出字段一致，能直接对比

### `test_phase9_does_not_change_phase8_behavior_when_reporting_disabled`

验证：

- 关闭 phase9 reporting/matrix 时，运行行为不被新 logging 污染

---

# 建议的回归验证

这一阶段建议做四种回归：

## 1. legacy baseline 回归

```bash
python scripts/_diag_l2c1_sweep.py
```

预期：

- 仍然是 `9/80/68/99/100%`

## 2. 小矩阵 smoke test

建议新增：

```text
scripts/run_phase9_matrix.py --smoke
```

只要求：

- 跑一个很小的矩阵子集
- 能输出 online + transfer summary

## 3. plotting/report smoke

建议新增：

```text
scripts/plot_phase9_results.py --smoke
```

只要求：

- 能从小矩阵结果生成至少一类表/图数据
- 不要求画得很漂亮

## 4. tutor comparability smoke

建议再做一个很小的对比脚本，确保：

- `no_tutor / warning_only / unlock_only / item_only / robot_belief / oracle`
  至少能在一个统一输出 schema 下比较

---

# 这一阶段的预期结果

如果 Antigravity 做对了，Phase 9 结束后你应该得到：

## 代码层

- 有统一 metrics schema
- 有 transfer eval 模块
- 有 config-driven experiment matrix
- 有最小 report / plotting pipeline

## 测试层

- 旧 245 tests 继续通过
- 新增大约 30–38 个 metrics / matrix / reporting tests
- legacy baseline 继续锁住

## 研究层

- 你终于能正式回答：
  - 哪种 tutor 最能帮 agent 完成任务
  - 哪种 tutor 最能帮 agent 学会自己完成任务
  - 不同 competence 下，最优 intervention family 是什么

- 也就是这一步之后，你的系统已经非常接近论文主实验框架了

---

# 最后给 Antigravity 的 4 句硬约束

```text
This phase is for evaluation infrastructure, not new core algorithms.

Keep online task metrics and no-tutor transfer metrics strictly separated.

Use a config-driven experiment matrix and standardized output schema.

Do not let new logging/reporting code change the behavior of existing agent or tutor paths.
```

这四句我建议放在最前面。

如果你愿意，我下一条可以继续把这份内容压成一版**更短、适合直接粘贴到 Antigravity 输入框的一屏版 Phase 9 prompt**。

[1]: https://antigravity.google/docs/implementation-plan?utm_source=chatgpt.com "Implementation Plan"
