有，整体方向是对的，但我建议你把 **Phase 5 再收紧一点**，这样更稳，也更容易让 antigravity 不走偏。

我先给结论：

**Phase 5 不要同时追求“完整 noisy patch observation + 完整 belief propagation + 完整新 planner”。**
最稳的做法是把它拆成一个**最小可行版本**：

1. **局部 patch 观测先做出来**
2. **patch 内多个格子的 belief 更新先做出来**
3. **planner 先输出 path prefix 预测和 prefix diagnostics**
4. **不要在这一阶段发明第二套规划算法**

也就是说，Phase 5 的核心不是“重新设计 planner”，而是：

> **让当前 planner 真正消费多个格子的 joint predictions，并且把这些预测显式输出出来。**

这样更符合你现在的阶段位置：
Phase 4 已经统一了 latent world 语义，Phase 5 最该做的是 **multi-cell observation + multi-cell prediction surface**，而不是直接上更大规模的 planning rewrite。

---

# 我对 Phase 5 的建议修改

## 1. 把目标写得更明确：先做“patch sensing + prefix prediction”

你现在写的是：

* 对局部 patch 观测
* 对未访问格子 belief propagation
* 对前方多个格子做预测性评估

这没错，但我建议你在文档里把 Phase 5 的主目标改成：

### Phase 5 主目标

* **局部 patch 观测是第一优先级**
* **多格 belief 更新是第二优先级**
* **path prefix 预测输出是第三优先级**
* **完整多步规划新算法不是本阶段主目标**

这样 antigravity 不容易一下子把 scope 做爆。

---

## 2. “belief propagation” 这个词太大，建议改成更保守的说法

`belief propagation` 容易让 agent 误会成：

* 图模型消息传递
* 邻域结构传播
* 更重的推断算法

你现在最需要的其实只是：

> **patch-based belief updates**
> 即：对当前可观测 patch 内的多个格子做并行 / 批量更新。

所以我建议你把措辞改成：

* **multi-cell patch update**
* 或 **patch-based belief update**

而不是 `belief propagation`。

---

## 3. covariance 先坚持 diagonal，不要上 full matrix

你原文里写：

* `Sigma[s] ∈ R^{d×d}` 或先用 diagonal

我建议这里直接写死：

> **Phase 5 uses diagonal covariance only.**

原因很简单：

* 你现在已经进入多个格子 + patch 更新
* 如果还同时做 full covariance，复杂度和 bug 面都会大增
* 对当前研究故事帮助不大

所以建议 Antigravity 这一阶段只做：

* `mu[s] ∈ R^d`
* `var[s] ∈ R^d` 或 `diag_cov[s] ∈ R^d`

这样就够了。

---

## 4. “多格预测”先定义成**path prefix diagnostics**

这是我最建议你加的一句。

你现在写的是：

* 输出 planned path prefix
* prefix expected cost
* prefix expected cumulative risk
* risky frontier cells

这很好。
我建议明确写成：

> **Phase 5 does not require a new planner search algorithm.
> It is acceptable to compute multi-cell prediction by taking the current planned path and producing prefix-level diagnostics over the first H cells.**

这句话非常重要。
因为它能明确告诉 antigravity：

* 不用重写 A*
* 不用做完整 belief-space planner
* 只要沿当前 plan 计算未来 3–8 格的 cost/risk/uncertainty 统计即可

这一步非常符合“最小可行版本”。

---

## 5. warning 在这一阶段只要求“影响 prefix belief / prefix score”

你写的是：

> warning 不只是改 next action，而是改 path segment belief

这个方向对。
但我建议补一句：

> **In Phase 5, warning only needs to influence patch-level or prefix-level predictions.
> It does not need a new communication model.**

也就是：

* 不要再改 Phase 1 已经做好的 warning 协议
* 只要让它的效果从“单格偏置”扩展到“影响 prefix 评估”就够了

---

## 6. 建议加两个 config 开关，继续 additive migration

Phase 4 做得最对的地方就是 additive migration。
Phase 5 也应该照这个来。

我建议至少加：

* `patch_observation_mode: false/true`
* `prefix_prediction_mode: false/true`

这样你就能保住：

* legacy path
* latent path
* latent+patch path
* latent+patch+prefix path

这对调试和 regression 非常重要。

---

# 我建议你给 antigravity 的 Phase 5 总体口径

一句话版：

> **Do not redesign the planner.
> Add patch-based multi-cell observation and multi-cell belief updates first, then expose prefix-level predictive diagnostics from the existing planner.**

这句话我非常建议放在任务单最前面。

---

# 下面是可以直接给 Antigravity 的任务单

```text
Project: pedagogical_ip
Phase: Phase 5 — patch-based multi-cell sensing and prefix prediction

Current status
Phase 0 complete: baselines frozen
Phase 1 complete: protocol cleanup
Phase 2 complete: V2 runner platformization
Phase 3 complete: V2 environment interface
Phase 4 complete: latent world semantics with unified cost+risk heads
Current total tests: 124
Legacy V2 baseline remains unchanged

High-level goal
This phase should make the agent meaningfully reason about multiple cells.

Move from:
- mostly current-cell / local single-cell updates

toward:
- noisy local patch observation
- patch-based multi-cell belief updates
- prefix-level multi-cell prediction over future path cells

This phase is NOT yet about a full new planner.
Do NOT redesign the planner search algorithm unless absolutely necessary.

Core design principle
Use the existing planner path and expose richer prediction over the first H future cells.
Patch sensing and patch-based belief updates come first.
Path-prefix diagnostics come second.

What this phase should achieve
1. local patch observation around the agent
2. noisy observation over 1-hop or 2-hop patch
3. multi-cell belief updates for all cells in the observed patch
4. planner-visible cost/risk/uncertainty predictions over several future cells
5. path-prefix outputs such as:
   - planned path prefix
   - prefix expected cost
   - prefix expected cumulative risk
   - risky frontier cells

What this phase should NOT do
- do NOT redesign the teacher science
- do NOT introduce a new full planner if avoidable
- do NOT implement full robot belief over agent belief yet
- do NOT add item-drop intervention yet
- do NOT introduce neural networks
- do NOT use full covariance matrices
- do NOT remove legacy or Phase 4 paths

Design preferences
- use diagonal covariance only
- prefer patch-based belief update over any heavy “belief propagation”
- prefer additive migration with config flags
- use the current planner path and compute prefix diagnostics over it

Suggested config strategy
Please consider adding config switches such as:
- latent_mode: false/true (existing)
- patch_observation_mode: false/true
- patch_radius: 0/1/2
- observation_noise_std
- prefix_prediction_mode: false/true
- prefix_horizon: e.g. 3–8

Files to inspect first
- src/envs/lattice_v2_env.py
- src/envs/lattice_v2_runner.py
- src/envs/lattice_v2.py
- src/agents/feature_belief.py
- src/agents/cost_risk_model.py
- src/agents/observation_model.py
- src/agents/planner_astar.py
- src/agents/warning_update.py
- configs/agent.yaml
- configs/env.yaml
- configs/experiment.yaml
- tests/test_v2_latent_path.py
- tests/test_cost_risk_model.py
- tests/test_latent_belief.py

Recommended implementation order
Priority 1:
Introduce noisy patch observation.
Observation should cover a configurable local patch around the agent.
Current cell may remain most reliable; nearby cells may be noisier.

Priority 2:
Extend FeatureBeliefMap to support patch-based updates over multiple observed cells.
For each cell store at least:
- latent mean
- diagonal latent variance
- visit_count
- last_observed_t

Priority 3:
Make planner diagnostics prefix-aware.
Do not replace the planner search unless needed.
Instead:
- compute path prefix over first H future cells
- compute prefix expected cost
- compute prefix cumulative risk
- compute uncertainty summary
- expose risky frontier cells

Priority 4:
Expose prefix predictions through env/runner info and state APIs.

Important constraint
Use an additive migration strategy.
Do not replace the existing latent path in-place.
New patch/prefix behavior should be guarded by config.

Required output format before implementation
1. concise diagnosis of current single-cell limitations
2. proposed patch observation schema
3. proposed patch update design
4. proposed prefix prediction outputs
5. file-by-file change plan
6. config plan
7. proposed tests
8. risks of over-engineering

Acceptance criteria
Phase 5 is successful only if:
- existing tests still pass unless intentionally updated
- new patch/prefix tests pass
- legacy baseline path remains reproducible
- latent path remains runnable
- patch path remains runnable
- planner exposes meaningful multi-cell prefix predictions

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

这一步的测试应该分成三层：

1. **patch observation**
2. **multi-cell belief update**
3. **prefix prediction / planner integration**

---

## A. 旧测试先继续全过

先跑全套：

```bash
python -m pytest tests/ -v --tb=short
```

预期：

* 当前 **124 tests 全通过**
* 新功能必须以 config 受控方式加进去，不能把旧路径搞坏

---

## B. 建议新增测试文件

我建议这一步新增 3 个主测试文件。

---

### 1. `tests/test_patch_observation.py`

测 observation schema 和 noise 行为。

建议至少包含这 6 个：

#### `test_patch_observation_radius0_matches_current_cell`

验证：

* `patch_radius=0` 时，行为退化为当前 cell 观测

#### `test_patch_observation_radius1_returns_local_patch`

验证：

* `patch_radius=1` 时返回正确邻域范围
* 不要求全图

#### `test_patch_observation_zero_noise_matches_truth`

验证：

* `observation_noise_std=0` 时，观测均值等于真实 latent/features

#### `test_patch_observation_nonzero_noise_changes_values`

验证：

* 非零噪声时，观测值偏离真值，但 schema 不变

#### `test_patch_observation_respects_bounds`

验证：

* 靠近边界时 patch 不越界，返回有效子区域

#### `test_patch_observation_schema_stable`

验证：

* observation 结构、字段、shape 是稳定的

---

### 2. `tests/test_patch_belief_update.py`

测 patch-based multi-cell update。

建议至少包含这 6 个：

#### `test_patch_update_changes_multiple_cells`

验证：

* 一次 patch 更新会影响多个格子的 belief，而不是只改当前位置

#### `test_unobserved_cells_unchanged`

验证：

* patch 外格子不会被错误修改

#### `test_visit_count_updated_for_observed_cells`

验证：

* 被观测格子的 `visit_count` 增加

#### `test_last_observed_time_updated`

验证：

* 被观测格子的 `last_observed_t` 正确更新

#### `test_diagonal_variance_decreases_with_repeated_observation`

验证：

* 多次观察同一格后，对应 latent variance 下降

#### `test_patch_update_still_satisfies_belief_protocol`

验证：

* 扩展后依然满足 `CellBelief` 协议

---

### 3. `tests/test_prefix_prediction.py`

测 planner/path prefix diagnostics。

建议至少包含这 8 个：

#### `test_prefix_prediction_returns_path_prefix`

验证：

* 返回 prefix，不只是 next action

#### `test_prefix_length_respects_horizon`

验证：

* `prefix_horizon=H` 时输出长度不超过 H

#### `test_prefix_expected_cost_computable`

验证：

* prefix expected cost 能算出来

#### `test_prefix_expected_risk_computable`

验证：

* prefix cumulative risk 能算出来

#### `test_prefix_uncertainty_computable`

验证：

* prefix uncertainty summary 能算出来

#### `test_risky_frontier_cells_reported`

验证：

* 能识别并输出高风险 frontier cells

#### `test_warning_changes_prefix_score_or_belief`

验证：

* warning 的作用不再只体现在 next action，也体现在 prefix 评估

#### `test_existing_planner_path_still_runs`

验证：

* 不开 prefix mode 时，原 planner 路径仍正常

---

## C. 建议新增一个集成测试文件

### 4. `tests/test_v2_patch_prefix_integration.py`

建议至少 4 个：

#### `test_latent_patch_mode_episode_runs`

验证：

* latent + patch mode 能完整跑一集

#### `test_latent_patch_prefix_mode_episode_runs`

验证：

* latent + patch + prefix mode 能完整跑一集

#### `test_env_info_contains_prefix_predictions`

验证：

* env / runner 的 info / state 已包含 prefix diagnostics

#### `test_legacy_mode_baseline_unchanged`

验证：

* legacy path baseline 不变

---

# 建议的回归验证

这一阶段我建议做三种回归：

## 1. legacy path 回归

```bash
python scripts/_diag_l2c1_sweep.py
```

预期：

* 仍然是 `9/80/68/99/100%`

## 2. latent path smoke

保留你上一阶段的 latent smoke

* 只要求能跑
* joint cost/risk diagnostics 正常

## 3. latent + patch smoke

建议新增一个轻量脚本，比如：

```text
scripts/_diag_l2c1_patch_smoke.py
```

只要求：

* 能跑
* 输出 patch observation / prefix diagnostics
* 不要求和 legacy baseline 一样

---

# 这一阶段的预期结果

如果 antigravity 做对了，Phase 5 结束后你应该得到：

## 代码层

* observation_model 真正支持 patch observation
* FeatureBeliefMap 支持 patch-based multi-cell updates
* planner / env / runner 能输出 prefix diagnostics
* 旧 planner 仍在，未被暴力重写

## 测试层

* 旧 124 tests 继续通过
* 新增约 20–24 个 patch/prefix 相关测试
* legacy baseline 继续锁住

## 研究层

* agent 已经开始“看多个格子、更新多个格子、预测未来多个格子”
* 但还没有进入更重的 robot nested belief
* 这样下一阶段就很自然能进入：

  * robot 读取 agent predicted path prefix
  * teacher 基于 prefix risk 做 intervention

---

# 最后，我建议你补给 antigravity 的两句硬约束

这两句很重要，我建议放在任务单最前面：

```text
Do not redesign the planner unless clearly necessary.
Use the current planner path and add prefix-level predictive diagnostics first.

Use patch-based multi-cell belief updates, not a heavy belief-propagation algorithm.
Keep covariance diagonal in this phase.
```

另外，antigravity 这次也建议继续按官方工作流来：先看 implementation plan，再执行；让它把任务拆进 task list，完成后给 walkthrough；终端保持 Request Review。官方文档里这几项都是现成支持的。([Google Antigravity][1])

如果你愿意，我下一条可以继续给你一版**更短、更适合直接粘贴到 antigravity 输入框的一屏版 Phase 5 prompt**。

[1]: https://antigravity.google/docs/implementation-plan?utm_source=chatgpt.com "Implementation Plan"
