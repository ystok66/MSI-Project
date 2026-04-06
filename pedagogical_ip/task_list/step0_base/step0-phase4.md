可以。基于你现在的状态，我建议把 **Phase 4** 明确定义成：

# Phase 4：统一 latent world 语义

把 V2 从“feature belief + risk head”推进到
**“每个格子有 latent vector，cost 和 risk 都由它决定，agent 通过观测近似学习它”**

这一步才真正进入 proposal 的核心，但仍然要**克制**：
**先做最小可行版本，不上神经网络，不改 teacher 科学设定，不做 multi-cell prediction。**

另外，这一轮我建议你继续要求 antigravity：**先出 implementation plan，再实现；保留 task list；终端权限保持 Request Review；完成后给 walkthrough。** 这些都是它文档里支持的标准工作流。([Google Antigravity][1])

---

## 一、Phase 4 的大致目的

你现在已经有：

* Phase 0：基线冻结
* Phase 1：协议层整理
* Phase 2：runner 平台化
* Phase 3：环境接口壳子
* 105 tests
* V2 baseline 始终不变

所以现在最自然的下一步，不是再做结构壳，而是开始统一 V2 的世界语义。

当前 V2 的核心限制是：

* `FeatureBeliefMap` 里有 feature belief
* `risk_model.py` 里主要围绕 risk
* cost 还是相对分离的
* 于是会出现“feature 是一套语义，risk 是一套语义，planner 代价又像另一套语义”的问题

Phase 4 的目标就是把这件事统一成：

> 对每个格子 (s)，存在潜在向量 (z_s\in \mathbb{R}^d)。
> 真值 cost 和真值 risk 都来自 (z_s)。
> agent 只能通过局部观测、带噪信息去近似学习 (z_s)。

---

## 二、Phase 4 的思路

### 1. 先统一“世界真相”，再统一 planner 输入

这一阶段的重点不是 planner 算法本身，而是 **planner 所依赖的 cell-level semantics**。

建议先把每个 cell 的语义改成：

* latent vector: `z_s`
* true cost: `c_s = f_cost(z_s)`
* true risk: `rho_s = sigmoid(f_risk(z_s))`

最开始用**线性头**就够了：

* `c_hat = w_c · z + b_c`
* `rho_hat = sigmoid(w_r · z + b_r)`

### 2. 不要急着上神经网络

这一阶段不要引入：

* MLP encoder
* 大规模训练循环
* 复杂 offline dataset
* end-to-end learned planner

先做：

* shared linear heads
* 可解释参数
* online / incremental update
* 不改变当前实验主故事

### 3. 先保留当前局部观测形式，噪声只做最小接入

Phase 4 可以开始给观测模型加一点明确噪声语义，但**不要直接进入 Phase 5 的 multi-cell noisy patch**。

这一轮更适合做：

* visited cell 提供强监督
* warning / outcome 提供弱监督
* optional small observation noise hook，但不要扩成完整 multi-cell observation framework

### 4. planner 先升级输入，不升级复杂度

这一阶段 planner 不需要大改。
只要让 planner 的 cell score 从“主要依赖 risk”变成：

* predicted cost
* predicted risk
* uncertainty penalty

即可。

---

## 三、Phase 4 的方法

### 方法 A：新增文件优先，不重写旧文件

我建议这一步优先：

* **新增** `cost_risk_model.py`
* 对 `feature_belief.py` 做**增量式扩展**
* 对 `lattice_v2_runner.py` / `lattice_v2_env.py` 做**最小 wiring**

而不是大改 `risk_model.py` 直到它同时承担旧逻辑和新逻辑。

### 方法 B：保留旧路径，新增 latent path

最好采用“双路径共存”的短期策略：

* 旧 risk-only path 暂时还在
* 新 latent cost+risk path 加进来
* 通过 config 切换
* baseline 不漂移

### 方法 C：先做“单格 latent 语义成立”，再谈多格预测

这一步的成功标准不是 multi-cell prediction，而是：

* 每个格子已经有统一 latent 语义
* cost 和 risk 都来自同一 latent belief
* planner 已经真正消费 `cost_hat + risk_hat + uncertainty`

---

# 下面是直接给 Antigravity 的任务单

```text
Project: pedagogical_ip
Phase: Phase 4 — latent world semantics for V2

Current status
Phase 0 complete: baselines frozen
Phase 1 complete: planner/warning/belief protocol cleanup
Phase 2 complete: runner platformization
Phase 3 complete: thin V2 environment interface
Current total tests: 105
V2 baselines remain unchanged across all phases

High-level goal
This phase is the first step into the core scientific model.

The goal is to move V2 from:
- feature belief + learned risk

toward:
- latent vector per cell
- true cost and true risk both derived from the same latent vector
- agent learns that latent vector approximately through observation and outcomes

Target semantics
For each cell s:
- latent vector z_s in R^d
- true cost c_s = f_cost(z_s)
- true risk rho_s = sigmoid(f_risk(z_s))

At first, do NOT use neural networks.
Use simple linear heads:
- c_hat = w_c · z + b_c
- rho_hat = sigmoid(w_r · z + b_r)

What this phase should achieve
1. give each cell a unified latent meaning
2. make both cost and risk come from that latent meaning
3. make planner consume predicted cost + predicted risk + uncertainty
4. preserve existing experimental behavior as much as possible
5. keep the system runnable and testable

What this phase should NOT do
- do NOT implement multi-cell prediction yet
- do NOT implement full noisy patch observation yet
- do NOT redesign teacher policies
- do NOT introduce neural networks
- do NOT rewrite the whole environment
- do NOT break the current V2 baseline path unless explicitly guarded by config
- do NOT remove old behavior until the new path is validated

Core design preference
Prefer adding a new latent cost+risk path, with configuration switches if needed, rather than replacing all old logic at once.

Files to inspect first
- src/envs/lattice_v2_env.py
- src/envs/lattice_v2_runner.py
- src/envs/lattice_v2.py
- src/agents/feature_belief.py
- src/agents/risk_model.py
- src/agents/planner_astar.py
- src/agents/observation_model.py
- configs/agent.yaml
- configs/env.yaml
- configs/experiment.yaml
- tests/test_v2_runner.py
- tests/test_v2_env_api.py

Suggested code direction
Strongly consider:
- keeping feature_belief.py, but extending it so each cell has an explicit latent-vector interpretation
- creating a new file:
  src/agents/cost_risk_model.py

This new file should likely contain:
- BayesianCostHead
- BayesianRiskHead
- optional shared utilities for linear latent heads

Avoid overloading risk_model.py too much unless there is a strong reason.

Recommended implementation scope
Priority 1:
Introduce explicit latent-vector semantics for each V2 cell.

Priority 2:
Add a joint cost+risk prediction module based on latent vectors.

Priority 3:
Wire planner cost computation so it can use:
- predicted cost
- predicted risk
- uncertainty penalty

Priority 4:
Make the new path configurable so old baseline behavior can still be regression-tested.

Required output format before implementation
1. concise diagnosis of the current feature/risk split
2. proposed latent-world design
3. file-by-file change plan
4. config strategy for old path vs new path
5. proposed tests
6. risks of over-engineering
7. minimal implementation order

Acceptance criteria
Phase 4 is successful only if:
- old tests still pass unless intentionally updated
- new latent/cost-risk tests pass
- the system remains runnable end-to-end
- planner can consume cost + risk + uncertainty from the new latent path
- V2 still has a stable regression path

Baseline preservation requirement
Current V2 baseline values must remain reproducible under the legacy path:
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

## 四、建议的测试内容

这一阶段的测试重点变了。
不是再测试接口壳，而是测试：

1. latent vector 语义是否成立
2. cost 和 risk 是否真的由同一 latent belief 导出
3. planner 是否消费了新的 joint signal
4. 旧路径是否还能保 baseline

---

## A. 旧测试必须继续通过

先跑全套：

```bash
python -m pytest tests/ -v --tb=short
```

预期：

* 当前 **105 tests 全通过**
* 如果某些旧测试必须更新，必须是因为 config/path 显式切换，而不是隐式行为变化

---

## B. 建议新增测试文件

我建议这一阶段新增 3 个主测试文件。

### 1. `tests/test_cost_risk_model.py`

重点测 joint latent heads。

建议至少包含这 6 个：

#### `test_cost_head_linear_response`

验证：

* cost head 对 latent vector 的响应符合线性预期

#### `test_risk_head_sigmoid_response`

验证：

* risk head 输出在 `(0,1)` 内
* 对 latent vector 的变化方向合理

#### `test_same_latent_affects_both_cost_and_risk`

验证：

* 同一个 latent vector 变化会同时影响 `cost_hat` 和 `risk_hat`
* 这是“统一世界语义”的核心测试

#### `test_uncertainty_propagates_to_predictions`

验证：

* latent belief 不确定性升高时，预测 uncertainty 指标升高
* 即使 cost/risk 的点估计不变，也能反映更不确定

#### `test_heads_can_use_shared_latent_dimension`

验证：

* cost/risk heads 可以共享 latent 维度，而不是各玩各的

#### `test_cost_risk_model_configurable`

验证：

* config 能切换新旧 path，或者切换 latent dim / weight setup

---

### 2. `tests/test_latent_belief.py`

重点测 `FeatureBeliefMap` 的 latent 语义扩展。

建议至少包含这 5 个：

#### `test_feature_belief_exposes_latent_belief`

验证：

* 每个 cell 都能取到 `mu_z` / `Sigma_z` 或等价表示

#### `test_latent_update_changes_belief`

验证：

* 新观测会改变 latent belief

#### `test_latent_copy_reset_still_work`

验证：

* copy / reset 在 latent path 下仍然正确

#### `test_visited_cell_stronger_supervision`

验证：

* visited cell 的更新强于未访问/弱观测更新
* 这能对应你“强监督 vs 弱监督”的阶段性设计

#### `test_belief_protocol_still_satisfied`

验证：

* 扩展后仍满足 `CellBelief` 协议

---

### 3. `tests/test_v2_latent_path.py`

重点测 runner/env/planner 的集成。

建议至少包含这 6 个：

#### `test_v2_latent_mode_reset_runs`

验证：

* 开启 latent mode 后，env/runner 仍能正常 reset

#### `test_v2_latent_mode_episode_runs`

验证：

* 开启 latent mode 后，单集可正常运行到终止

#### `test_planner_uses_cost_risk_uncertainty`

验证：

* planner 的 cell score 确实使用了：

  * predicted cost
  * predicted risk
  * uncertainty
* 不是只看 risk

#### `test_high_cost_low_risk_tradeoff_affects_action`

验证：

* 构造两个 candidate cells：

  * 高 cost / 低 risk
  * 低 cost / 高 risk
* planner 行为会随权重合理变化

#### `test_legacy_mode_baseline_unchanged`

验证：

* 旧 config/path 下 baseline 仍能复现

#### `test_latent_mode_info_contains_predictions`

验证：

* env/state/info 至少能暴露一部分 joint prediction 信息
* 例如当前 cell / frontier 的 `cost_hat`, `risk_hat`, `uncertainty`

---

## C. 可选测试

如果 antigravity 把 observation noise hook 一起最小接入，可以再补：

### `tests/test_latent_observation_noise.py`

建议 2–3 个测试：

* `test_zero_noise_matches_observation_mean`
* `test_nonzero_noise_changes_update_strength`
* `test_noise_config_is_respected`

但这组测试是**可选**，因为完整 noisy patch observation 更像下一阶段。

---

## 五、建议的回归验证

这一阶段一定要做**双回归**：

### 1. 旧路径回归

```bash
python scripts/_diag_l2c1_sweep.py
```

预期：

* legacy path 下 baseline 仍是

  * 9 / 80 / 68 / 99 / 100%

### 2. 新路径 smoke test

建议新增一个轻量脚本，例如：

```text
scripts/_diag_l2c1_latent_smoke.py
```

只要求：

* 能跑
* 不崩
* 输出 joint cost/risk diagnostics
* 不要求和 legacy baseline 完全一样

---

## 六、这一阶段的预期结果

如果 antigravity 做对了，Phase 4 结束时你应该得到：

### 代码层

* `FeatureBeliefMap` 已经有明确 latent vector 语义
* 新增 `cost_risk_model.py`
* planner 已能消费 `cost_hat + risk_hat + uncertainty`
* legacy path 仍保留

### 测试层

* 旧 105 tests 继续通过
* 新增大约 14–17 个 latent/joint model tests
* baseline 通过 legacy path 继续锁住

### 研究层

* 你终于真正进入 proposal 的世界建模核心
* 下一阶段才适合继续做：

  * noisy local patch observation
  * multi-cell prediction
  * robot belief over agent belief

也就是说：

**Phase 4 的成功，不是“效果暴涨”，而是“世界语义终于统一了”。**

---

## 七、我对这一阶段的一个关键建议

你这次给 antigravity 时，最好再明确一条：

> **不要把 Phase 4 做成“替换旧系统”的迁移。要把它做成“新增 latent path，并用 config 控制”的安全扩展。**

这是这一步最关键的风险控制。
否则它很容易为了“整洁”直接把旧 V2 行为改掉，那样你 baseline 锁就没意义了。

你可以在任务单最前面再加一句：

```text
Use an additive migration strategy.
Prefer a new latent path guarded by config over replacing the current legacy path in-place.
```

如果你愿意，我下一条可以继续给你一版**更短、更适合直接粘贴到 antigravity 输入框的一屏版 Phase 4 prompt**。

[1]: https://antigravity.google/docs/implementation-plan?utm_source=chatgpt.com "Implementation Plan"
