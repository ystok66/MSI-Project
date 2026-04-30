# Safety Shield + Pedagogical Scaffolding Implementation Plan

日期：2026-04-21

这份文档是给下一轮实现用的正式 implementation plan。它基于：

- 当前主线代码
- 最新诊断实验 `runs/tutor_diagnostic_suite_4seed_20260421`
- [2026-04-21_tutor_diagnostic_suite_report.md](../notes/2026-04-21_tutor_diagnostic_suite_report.md)

它的目标不是继续把问题表述成：

```text
inverse tutor 要不要打败 always_warn
```

而是把系统重构成两个明确层次：

```text
Layer 0: Safety Shield
    truthful warning，负责防灾难

Layer 1: Pedagogical Scaffolding
    WAIT / WAYPOINT / HINT，负责探索质量、学习效率、泛化与 over-help 控制
```

这份 plan 会明确：

- 现有代码哪些模块已经够用
- 哪些模块必须重构
- 哪些条件应当降级为 debug baseline
- 下一步应当做哪些 tests / experiments
- 如何减少超参数和机制冗余
- 如何让系统更鲁棒、更可泛化，并和未来框架兼容

---

## 1. Current Conclusion

基于 `T04 -> E05` 和 `T08 -> E07` 的 4-seed 诊断套件，当前最重要的结论是：

1. `always_warn` 现在强，是合理现象，不是坏事。
   这说明 warning 通道和 set-level risk update 确实有作用。

2. 当前 inverse tutor 的核心问题不是 waypoint 太少，而是 warning 不是 route-repairing。

   当前最关键的诊断信号是：

   ```text
   warning_actionability ≈ 0
   ```

3. `T04` 和 `T08` 当前都不是理想的“teacher 必须帮助才能成功”的 slice。

   尤其 `T08` 里所有条件 `teach_success = 1.0`，说明 teach success 已经饱和。

因此下一步不应继续用一个统一 weighted-Q 让 warning 和 waypoint 在同一层里竞争，而应改成：

```text
Safety Shield:
    遇到 catastrophic risk 必须 warning

Scaffolding:
    在已经安全的前提下，
    再判断 WAIT / WAYPOINT / HINT
```

---

## 2. Current Code Mapping

### 2.1 Modules To Keep As Foundation

这些模块已经足够稳定，不应大改结构，只做增量式扩展：

- `risky_maze/env/fixed_loader.py`
  - fixed spec loading 和 `FixedRuntimeLayout` 已稳定
- `risky_maze/env/objectives.py`
  - `pickup / pass / collect_gem / exit` objective machine 已可用
- `risky_maze/env/pomdp_episode.py`
  - fixed-map POMDP runtime 已跑通
- `risky_maze/runner/fixed_episode_runner.py`
  - 已有 per-step tutor diagnostics 和真实 outcome logging
- `risky_maze/runner/fixed_block_runner.py`
  - teach/eval split 已稳定
- `risky_maze/experiments/run_fixed_maze.py`
  - CLI、summary/episodes/steps/tutor_decisions/risk_eval 输出已可用
- `risky_maze/experiments/run_formal_tutor_matrix.py`
  - 已支持 partial checkpoint、progress.log、代表性 slice 运行
- `risky_maze/scenarios/mini_*.py`
  - `MiniRiskGate_v0`
  - `MiniExploreLoop_v0`
  - `MiniWaypointBottleneck_v0`

### 2.2 Modules That Must Change

这些模块是下一阶段的核心改动点：

- `risky_maze/tutor/inverse_planner.py`
  - 当前仍是单层 action selection
  - 必须重构成 `Safety Shield -> Scaffolding` 两阶段决策

- `risky_maze/tutor/rollout.py`
  - 当前仍主要在用统一 weighted utility
  - 必须显式支持：
    - catastrophe check
    - no-progress / timeout risk
    - waypoint counterfactual value
    - warning actionability

- `risky_maze/tutor/candidates.py`
  - 当前 waypoint 没有明确层级分类
  - 必须区分：
    - frontier waypoint
    - landmark waypoint
    - bottleneck waypoint
    - oracle waypoint

- `risky_maze/learner/objective_agent.py`
  - 当前 warning suspicion / memory commit 机制还不够支持 over-help 研究
  - 必须加入：
    - assist-discounted consolidation
    - success-gated consolidation hooks
    - route graph / landmark relation 更新接口

- `risky_maze/runner/fixed_metrics.py`
  - 当前指标还不够区分：
    - 安全保护
    - 自主探索
    - scaffold 帮助
    - over-help 依赖
  - 必须补 autonomy / transfer / route-graph 类指标

### 2.3 Modules To Keep As Debug Or Ablation Only

这些条件或机制不应继续作为主线结论来源：

- `inverse_plan_warn_only`
  - 保留为 warning actionability debug baseline
  - 不再作为“最终主方法”使用

- `inverse_plan_full` 当前 weighted-Q 版本
  - 保留为 legacy ablation
  - 下一轮主线不要继续直接调这套大 utility

- `risk_threshold_warn`
  - 继续保留为 debug baseline
  - 不作为正式 baseline

- `inverse_plan_full_frontier_only`
  - 保留为 waypoint candidate restriction ablation
  - 不作为主方法本体

---

## 3. Target Architecture

目标架构：

```text
TutorV2
├─ Layer 0: Safety Shield
│   └─ WARNING(path prefix / selected risky set)
└─ Layer 1: Pedagogical Scaffolding
    ├─ WAIT
    ├─ WAYPOINT(frontier / landmark / bottleneck)
    └─ HINT(optional, later)
```

### 3.1 Layer 0: Safety Shield

warning 不再被视为“教学层的可选动作”，而是安全底线。

定义 catastrophe：

\[
C_{\text{cat}}(\tau)
=
\mathbb{1}\left[
\text{death} \lor \text{damage} \ge d_{\text{cat}}
\right]
\]

如果：

\[
P(C_{\text{cat}} = 1 \mid \text{WAIT}, b_t) > \delta_{\text{safe}}
\]

则直接：

\[
a_t = \text{WARNING}(S)
\]

其中 `S` 默认为 learner 当前计划路径前缀或其对应 risky set。

warning 语义继续沿用 set-level semantics：

\[
F_S = [\exists c \in S: z_c = \text{danger}]
\]

\[
P(Z_S \mid X_S, F_S)
\propto
\mathbb{1}[\exists c \in S: z_c = \text{danger}]
\prod_{c \in S} P(z_c \mid x_c)
\]

这一层的目标不是“少 warning”，而是：

```text
该 warning 的时候必须 warning
```

### 3.2 Layer 1: Pedagogical Scaffolding

只有在 Layer 0 判定当前 plan 安全后，才进入 pedagogical 决策。

Layer 1 的核心问题不是“要不要帮”，而是：

```text
当前 WAIT 是有用探索，还是无效绕圈？
如果 WAIT 不好，怎样给最小必要 scaffold？
```

---

## 4. Decision Rules

### 4.1 WAIT As Useful Exploration

定义 learner belief：

\[
b_t = (M_t, q_t(Z), q_t(\theta_{\text{risk}}), q_t(G), q_t(\psi))
\]

其中：

- `M_t`: map memory
- `q_t(Z)`: cell danger posterior
- `q_t(\theta_risk)`: risk concept parameters
- `q_t(G)`: route graph / landmark graph
- `q_t(\psi)`: learner profile posterior

定义 WAIT 的探索价值：

\[
V_{\text{explore}}(\text{WAIT})
=
\mathbb{E}[
\Delta I_{\text{map}}
+
\Delta I_{\text{risk}}
+
\Delta I_{\text{route}}
]
\]

其中：

\[
\Delta I_{\text{map}}
=
\sum_{c \in \text{new cells}} P_{\text{reuse}}(c)
\]

\[
P_{\text{reuse}}(c)
=
\frac{1}{|\mathcal{E}|}
\sum_{e \in \mathcal{E}} \mathbb{1}[c \in \pi^*_e]
\]

同时定义 WAIT 的失败风险：

\[
C_{\text{fail}}(\text{WAIT})
=
P(\text{timeout} \mid \text{WAIT})
+
P(\text{loop} \mid \text{WAIT})
+
P(\text{teach failure} \mid \text{WAIT})
\]

决策规则：

```text
if safe and V_explore high and C_fail low:
    return WAIT
```

### 4.2 Minimal-Sufficient Waypoint

waypoint 只在下面情形考虑：

```text
WAIT 安全，但会导致 timeout / loop / no-progress
```

Waypoint 不再当作“答案路径”，而当作 scaffold。

候选 waypoint 分类：

1. `frontier`
2. `landmark`
3. `bottleneck`
4. `oracle`

主线实验中只允许：

```text
frontier / landmark / bottleneck
```

`oracle` 仅保留为 over-help baseline。

选择规则改成 constraint form，而不是继续堆更多权重：

\[
g^* = \arg\min_g C_{\text{assist}}(g)
\]

subject to:

\[
P(\text{teach success} \mid g) \ge \rho_{\text{succ}}
\]

\[
P(\text{catastrophe} \mid g) \le \delta_{\text{safe}}
\]

\[
\text{NoProgressRisk}(g)
<
\text{NoProgressRisk}(\text{WAIT}) - \epsilon
\]

直觉：

```text
只要能把 learner 从失败/卡住状态拉出来，
选帮助最小、泄露最低的 waypoint
```

---

## 5. Over-Help And Generalization

### 5.1 Assist-Discounted Consolidation

这一块是当前代码里最缺失但最关键的机制。

teach 时 learner 完成一个 route segment：

\[
e = (u \rightarrow v)
\]

定义自主信用：

\[
A(e)
=
\exp(
-\lambda_h N_{\text{hint}}(e)
-\lambda_w G_{\text{wp}}(e)
-\lambda_o O_{\text{oracle}}(e)
)
\]

其中：

- `N_hint(e)`: segment 上 hint 次数
- `G_wp(e)`: waypoint progress gift
- `O_oracle(e)`: 是否使用 oracle waypoint

Waypoint progress gift：

\[
G_{\text{wp}}(g)
=
\frac{
\max(0, d(s,o)-d(g,o))
}{
d(s,o)+\epsilon
}
\]

长期 route memory 更新：

\[
\kappa'_e
=
\kappa_e + A(e)\cdot \mathbb{1}[\text{segment success}]
\]

这意味着：

```text
teacher 帮助越强，
learner 对这段 route 的长期置信度增加越少
```

### 5.2 Success-Gated Consolidation

teach success 后才 commit 长期知识：

\[
M^{LT}_{t+1}
=
\begin{cases}
M^{LT}_t \cup \text{Commit}(M^{episode}_t), & \text{teach success} \\
M^{LT}_t \cup \text{RiskOnly}(D_{\text{death}}), & \text{teach fail}
\end{cases}
\]

也就是：

```text
teach success:
    commit route graph / landmark relation / risk summary

teach fail:
    只保留局部 risk 教训，不 commit 完整 route principle
```

这一步是实现下面目标的核心：

```text
teach 不成功，eval 也难成功
```

---

## 6. Concrete Code Plan

### 6.1 `risky_maze/tutor/inverse_planner.py`

当前问题：

- 仍然把 warning / waypoint / wait 放在同一层里竞争
- 当前 `warning_actionability_threshold` 只是过滤器，不是架构级分层

必须修改：

1. 新增显式两阶段决策入口：

   - `act_safety_layer(context)`
   - `act_scaffold_layer(context)`
   - `act(context)` 先调 safety，再调 scaffold

2. 引入新的 tutor mode：

   - `safety_shield_only`
   - `shield_plus_inverse_minimal_waypoint`
   - `shield_plus_random_frontier_waypoint`
   - `shield_plus_oracle_waypoint`

3. `inverse_plan_warn_only` 改名/降级为 debug condition

4. 当前 `inverse_plan_full` 保留为 legacy / ablation

不建议做的事：

- 不要继续往当前 weighted-Q 里堆更多 penalty 项

### 6.2 `risky_maze/tutor/rollout.py`

当前问题：

- warning 的 Q 仍主要来自统一 utility
- 没有把 “是否真的修复危险路径” 作为一等公民

必须修改：

1. 新增 safety-only counterfactual：

   - `predicted_catastrophe_if_wait`
   - `predicted_catastrophe_if_warning`

2. 新增 warning route-repair diagnostics：

   - `pre_warning_true_trap_count`
   - `post_warning_true_trap_count`
   - `warning_route_repair_gain`

   建议定义：

   \[
   A_{\text{warn}}
   =
   \widehat{\text{Risk}}(\pi_{\text{wait}})
   -
   \widehat{\text{Risk}}(\pi_{\text{post-warning}})
   \]

3. waypoint rollout 改成最小充分 scaffold 评价：

   - `predicted_timeout_if_wait`
   - `predicted_timeout_if_waypoint`
   - `predicted_loop_if_wait`
   - `predicted_loop_if_waypoint`
   - `waypoint_ate`

4. 加 route-graph / landmark graph info gain proxy

### 6.3 `risky_maze/tutor/candidates.py`

当前问题：

- waypoint 候选还没有被清晰分层

必须修改：

1. 把当前 `generate_waypoint_candidates` 拆成：

   - `generate_frontier_waypoints`
   - `generate_landmark_waypoints`
   - `generate_bottleneck_waypoints`
   - `generate_oracle_waypoints`（只给 over-help baseline）

2. 所有主线方法禁止：

   - hidden oracle shortcut
   - next-step oracle path
   - 每步 route following

3. 给每个 waypoint 显式打标签：

   - `waypoint_type`
   - `novelty_leak_level`
   - `progress_gift`

### 6.4 `risky_maze/learner/objective_agent.py`

当前已接好：

- `warning_suspicion_mode`
- `clear_warning_suspicion`
- `replan_only`

仍需实现：

1. 默认主线把 `persistent` 降级为 ablation

   建议默认：

   - `learner_warning_suspicion_mode = replan_only`

2. route graph / landmark graph 数据结构

   可新增：

   - `risky_maze/learner/route_graph.py`
   - `risky_maze/learner/consolidation.py`

3. assist-discounted consolidation hook

4. success-gated commit hook

5. `ObjectiveAwareLearner.clone_for_eval` 支持：

   - `clear_route_graph`
   - `clear_landmark_graph`
   - `clear_autonomy_credit`

### 6.5 `risky_maze/env/objectives.py`

当前问题：

- `gem / door / key` 主要还是 objective token
- 还不是 learning event

必须修改：

1. gem 变成 consolidation event

2. door 变成 transferable bottleneck relation

3. key-door pass 成功后，生成 route-graph edge / landmark relation

### 6.6 `risky_maze/env/pomdp_episode.py`

必须增加：

- segment-level success events
- consolidation trigger events
- assist credit annotations

建议新增：

- `objective_segment_completed`
- `landmark_event`
- `consolidation_candidate`

### 6.7 `risky_maze/runner/fixed_episode_runner.py`

当前已经有较丰富 logging。

还需新增：

- `autonomy_credit_before`
- `autonomy_credit_after`
- `route_graph_commit`
- `landmark_relation_commit`
- `teach_success_commit`
- `teach_fail_no_commit`

### 6.8 `risky_maze/runner/fixed_metrics.py`

当前已补：

- `teach_safe_success_rate`
- `damage_per_100_steps`
- `trap_entries`
- `warning_suspicion_mass_*`

还需新增：

- `AutonomyCredit`
- `HintedDistanceFraction`
- `RouteGraphConfidence`
- `GeneralizationGapOverhelp`
- `P(eval_success | teach_success)`
- `P(eval_success | teach_fail)`
- `WaypointATE`

### 6.9 `risky_maze/runner/fixed_block_runner.py`

需要增加：

- success-gated consolidation mode
- assist-discounted consolidation mode
- route graph eval clone ablations

建议增加 config：

- `teach_consolidation_mode`
  - `always_commit`
  - `success_gated`
  - `success_gated_assist_discounted`

### 6.10 `risky_maze/experiments/*.py`

建议新增两个 runner：

- `run_safety_scaffold_suite.py`
- `run_overhelp_transfer_suite.py`

保留并继续使用：

- `run_tutor_diagnostic_suite.py`
  - 用于 warning / suspicion / route-repair debug
- `run_memory_risk_ablation.py`
  - 后续扩成 route-graph / autonomy ablation

---

## 7. Redundancy Reduction

当前系统最需要减少的不是代码量，而是机制冗余。

### 7.1 Remove From Mainline

这些不要继续作为主线：

- 单个巨大 weighted utility 统管一切
- 继续增加 warning / waypoint / boredom / assist 等大量手工权重
- 把 warning 同时当作 safety action 和 pedagogical action

### 7.2 Keep Only As Ablation

- `persistent warning_suspicion`
- `inverse_plan_full` current weighted-Q version
- `oracle waypoint`
- `risk_threshold_warn`

### 7.3 Prefer Constraint-Based Control

主线只保留最少阈值：

- `delta_safe`
- `rho_success`
- `epsilon_no_progress`
- `autonomy_lambda_hint`
- `autonomy_lambda_waypoint`
- `autonomy_lambda_oracle`

其中前三个是决策阈值，后三个是 consolidation 的最小必要参数。

不要再增加更多 feature-specific reward weights，除非实验明确显示必要。

---

## 8. Experimental Program

### Experiment 1: Safety Shield Fixed, Waypoint Only

目的：

```text
warning 固定成 safety shield
只测试 waypoint / scaffold 有没有真正教学价值
```

地图：

- `MiniWaypointBottleneck_v0`
- `MiniExploreLoop_v0`

条件：

- `no_tutor_mortal`
- `always_warn`
- `always_warn + random_frontier_waypoint`
- `always_warn + always_waypoint_objective`
- `always_warn + oracle_path_waypoint`
- `always_warn + inverse_minimal_waypoint`

核心指标：

- `TeachSteps`
- `TeachTimeout`
- `TeachDamage`
- `TeachSafeSuccess`
- `LoopRate`
- `NoProgressStepRate`
- `EvalRegret`
- `AssistLeakage`
- `AutonomyCredit`

预期：

- `oracle_path_waypoint` teach 最快但 generalization 差
- `inverse_minimal_waypoint` 在 teach / eval 之间最平衡

### Experiment 2: Teach-Success-Gated Consolidation

目的：

```text
验证 teach 不成功时 eval 也难成功
```

条件：

- `always_commit`
- `success_gated`
- `success_gated_assist_discounted`

指标：

- `P(eval_success | teach_success)`
- `P(eval_success | teach_fail)`
- `EvalRegret | teach_success`
- `EvalRegret | teach_fail`
- `RouteGraphConfidence`

### Experiment 3: Over-Help Generalization

目的：

```text
证明老师过度帮助会限制泛化
```

条件：

- `always_warn`
- `always_warn + sparse_frontier_waypoint`
- `always_warn + inverse_minimal_waypoint`
- `always_warn + dense_waypoint`
- `always_warn + oracle_path_waypoint`

核心指标：

- `AssistLeakage`
- `WaypointProgressGift`
- `AutonomyCredit`
- `MapReuseEval`
- `UsefulExplorationRate`
- `EvalRegret`

### Experiment 4: Safety-Hard Gate Slice

目的：

```text
让 no_tutor 真正危险，突出 warning 的必要性
```

推荐地图：

- `MiniRiskGate_v0`
- 或新增 safety-hard gate map

条件：

- `no_tutor_mortal`
- `no_tutor_immortal_warnlike`
- `always_warn`
- `always_warn + inverse_minimal_waypoint`

参数 sweep：

- `HP = 1, 2`
- tighter `time_limit`

---

## 9. Tests To Add

### 9.1 Safety / Scaffolding Contract

新增：

- `tests/test_safety_scaffold_split.py`

检查：

- catastrophic current plan 时一定返回 `WARNING`
- safe + useful exploration 时返回 `WAIT`
- safe 但 predicted timeout / no-progress 时返回 `WAYPOINT`

### 9.2 Waypoint Category Contract

新增：

- `tests/test_waypoint_categories.py`

检查：

- frontier / landmark / bottleneck / oracle 分类正确
- 主线方法不会发 oracle waypoint

### 9.3 Consolidation Contract

新增：

- `tests/test_success_gated_consolidation.py`

检查：

- teach success 时 commit route graph
- teach fail 时不 commit route graph
- assist leakage 越高，autonomy credit 越低

### 9.4 Overhelp Transfer Contract

新增：

- `tests/test_assist_discounted_memory.py`

检查：

- 同一路段，oracle-hinted credit < sparse-hinted credit < autonomous credit

---

## 10. Acceptance Criteria

下一轮实现完成后，至少满足：

1. warning 与 waypoint 决策已分层，不再共用一个混合 weighted-Q 主入口。
2. `always_warn` 被明确定位为 safety baseline，而非必须被主方法打败。
3. `inverse_minimal_waypoint` 可以在 `always_warn` 安全层之上运行。
4. route graph / landmark graph / success-gated consolidation 至少有最小闭环实现。
5. `oracle waypoint` 只作为 over-help baseline，不进入主方法。
6. 能输出：
   - `AutonomyCredit`
   - `RouteGraphConfidence`
   - `P(eval_success | teach_success)`
   - `P(eval_success | teach_fail)`
7. `MiniRiskGate_v0` 上 no tutor 真正表现出 safety-hard 现象。
8. `MiniWaypointBottleneck_v0` 上能清楚分离：
   - minimal scaffold
   - dense scaffold
   - oracle scaffold

---

## 11. Recommended Execution Order

按 Task Groups 执行：

### Group A: Refactor Tutor Layers

- `inverse_planner.py`
- `rollout.py`
- `factory.py`

目标：

```text
先把 Safety Shield 和 Pedagogical Scaffolding 分开
```

### Group B: Waypoint Taxonomy

- `candidates.py`
- `diagnostics.py`

目标：

```text
明确区分 frontier / landmark / bottleneck / oracle waypoint
```

### Group C: Consolidation

- `objective_agent.py`
- `objectives.py`
- `pomdp_episode.py`
- `fixed_episode_runner.py`

目标：

```text
把 teach success -> eval transfer 做成结构性机制
```

### Group D: Metrics / Tests

- `fixed_metrics.py`
- `tests/`

### Group E: Experiments

- `run_safety_scaffold_suite.py`
- `run_overhelp_transfer_suite.py`
- 扩展 `run_memory_risk_ablation.py`

---

## 12. Bottom Line

当前阶段不应继续追问：

```text
inverse tutor 要不要打败 always_warn
```

而应改成：

```text
危险时，warning 必须 truthful 且可靠；
在已经安全的前提下，
什么时候该 WAIT，
什么时候该给最小必要 waypoint / hint，
帮助 learner 高质量探索而不过度帮忙。
```

这才是下一轮实现的主线。
