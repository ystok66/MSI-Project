下面我给你一份**基于当前状态、面向 Phase 1 的 Antigravity 指南**。
我会先用一小段说明**目的、思路、方法**，然后后面直接给出**可以贴给 Antigravity 的任务单**、**测试要求**、**预期结果**。

---

# Phase 1 的大致目的

当前项目已经完成了准备阶段：

* WSL / `pedip310` 环境稳定
* 现有测试 **77 passed**
* V2 baseline 已固定
* `external_refs` 已准备好，关键参考为：

  * `Minigrid`
  * `pomdp-py`
  * `pypragmods`

但当前代码仍然处于**两个平行系统并存**的状态：

* **System A**：v0-v1d，Gym 风格，`BeliefMap + planner + RSA warning + teacher`
* **System B**：Lattice V2，feature belief + risk head + time-aware door tutor，但还更像实验原型

**Phase 1 的目标不是重写系统，也不是马上做最终复杂模型。**
Phase 1 的唯一目标是：

> **把 Lattice V2 稳定成“正式主实验平台”的前半步，同时做最小量、低风险的结构整理。**

也就是说，这一阶段主要做三件事：

1. **把 V2 相关逻辑从“脚本拼接”往“可维护主系统”推进**
2. **统一重复度最高的基础组件接口**
3. **补足 V2 的测试与回归验证机制**

---

# Phase 1 的思路

整个思路非常明确：

### 1. 不迁框架

不把项目迁到 Minigrid，不引入 pomdp-py 作为核心依赖，不把 pypragmods 直接嵌进去。

外部仓库只作为**结构参考**：

* **Minigrid**：环境 API / wrapper 风格参考
* **pomdp-py**：belief / state / observation / planner 抽象参考
* **pypragmods**：warning 的 speaker/listener 分层参考

### 2. 不改科学结论

这一阶段不改变当前实验结论，不引入新的科学设定。
尤其不应该破坏当前已经固定的 V2 baseline：

* `no_tutor = 9%`
* `warning_only (λ=5) = 80%`
* `door_2 = 68%`
* `door_3 = 99%`
* `always_close = 100%`

### 3. 只做最小 diff

优先顺序固定为：

1. **Planner 去重**
2. **Warning 抽象**
3. **Belief 协议**
4. **V2 runner / env 封装（如果前面稳定）**

---

# Phase 1 的方法

这一阶段采用的方法是：

### 方法 A：先“纯重构”，后“新增薄适配层”

优先做不会改变行为的改动：

* 提取共享 `_astar_core()`
* 新建 `pragmatic_warning.py`
* 新建 `belief_protocol.py`

### 方法 B：所有改动都必须伴随测试

不是“写了代码再看”，而是每一步都要：

* 保住现有 tests
* 新增对应的 V2 tests
* 跑 baseline 回归

### 方法 C：V2 先成为可调用主平台，不急着完整 Gym 化

如果要做 wrapper，也应先做**轻量 runner / minimal env**，不要一上来强行改成通用 RL training loop。

---

# 下面是直接给 Antigravity 的任务单和说明

你可以把下面整段直接发给 Antigravity。

---

## 给 Antigravity 的任务单（可直接复制）

```text
Project: pedagogical_ip
Phase: Phase 1 — stabilize Lattice V2 as the main experiment platform without rewriting the system

High-level goal
The codebase already has:
- a stable Phase 0 environment,
- 77 passing tests,
- fixed V2 baselines,
- external reference repos (Minigrid, pomdp-py, pypragmods).

This phase is NOT a rewrite.
This phase is about minimal-diff structural cleanup so that Lattice V2 can become a proper main experiment platform.

What you should optimize for
- minimal code changes
- preserving existing experimental behavior
- preserving existing tests
- making V2 easier to extend later toward latent feature vectors, noisy observations, learned cost/risk, and multi-cell prediction

What you must NOT do
- do NOT migrate the project into Minigrid
- do NOT introduce pomdp-py as a core dependency
- do NOT rewrite the whole environment stack
- do NOT merge System A and System B into one forced abstraction
- do NOT implement deep recursive RSA
- do NOT rewrite sweep scripts into a generic RL training framework
- do NOT change scientific assumptions in this phase

Current project reality
System A:
- Gym-style pedagogical grid
- scalar BeliefMap over cost/risk
- bounded agent
- RSA warning
- particle/oracle teachers

System B:
- Lattice V2 forced-choice environment
- FeatureBeliefMap over 4D features
- BayesianRiskHead
- warning_update.py
- time_aware_door_tutor.py
- currently more script-driven than framework-driven

Your task in this phase
Focus on four subgoals, in order:

1) Planner deduplication
2) Warning abstraction
3) Belief protocol cleanup
4) If the first three are stable, propose a minimal V2 runner/env wrapper plan

You must first inspect:
- src/agents/planner_astar.py
- src/teachers/rsa_warning.py
- src/agents/warning_update.py
- src/agents/belief.py
- src/agents/feature_belief.py
- src/envs/lattice_v2.py
- src/teachers/time_aware_door_tutor.py
- tests/test_planner.py
- tests/test_rsa_warning.py
- tests/test_belief_update.py
- tests/test_env.py

External references
Use external_refs only as design inspiration:
- Minigrid → environment API / wrapper structure
- pomdp-py → belief/state/action/observation interface boundaries
- pypragmods → warning speaker/listener factoring

Do not port code from those repos blindly.

Required output format
Before editing code, produce:
1. A concise diagnosis of the current Phase 1 code situation
2. A ranked minimal-diff plan
3. Exact files to modify first
4. Expected risks
5. Exact tests to run after each step

Then only implement the smallest high-value changes first.

Step-by-step implementation priorities

Priority 1: planner deduplication
Goal:
- extract a shared _astar_core() from planner_astar.py
- keep both public APIs working:
  - plan_next_action()
  - plan_next_action_v2()

Constraint:
- behavior must not change
- existing test_planner.py must still pass

Priority 2: warning abstraction
Goal:
- create a new file src/agents/pragmatic_warning.py
- define a minimal PragmaticWarner interface with:
  - select_utterance(...)
  - listener_update(...)

Then adapt:
- src/teachers/rsa_warning.py
- src/agents/warning_update.py

Constraint:
- keep existing logic intact
- do not over-generalize
- do not implement deep multi-level RSA

Priority 3: belief protocol cleanup
Goal:
- create src/agents/belief_protocol.py
- define a minimal shared protocol for per-cell belief maps

Required minimal interface:
- update(...)
- get_belief(...)
- copy()
- reset()

Then adapt:
- BeliefMap
- FeatureBeliefMap

Constraint:
- only unify interface, not internal semantics
- scalar belief and feature belief are not the same thing

Priority 4: V2 platformization proposal
Goal:
- only after priorities 1–3 are stable
- propose the smallest viable way to make V2 a first-class runnable platform
- prefer a light runner or minimal env wrapper
- do not do a large Gym refactor unless clearly justified

Acceptance criteria
Phase 1 is successful only if:
- all existing tests still pass
- new V2 tests pass
- V2 baseline behavior remains unchanged
- code becomes more modular without becoming more abstract than needed

Current baseline that must be preserved
V2 baseline:
- no_tutor = 9%
- warning_only (lambda=5) = 80%
- door_2 = 68%
- door_3 = 99%
- always_close = 100%
- lambda sweep: 1→9%, 3→46%, 5→80%, 7→100%

Please keep changes minimal and explain why each change is scientifically useful and structurally safe.
```

---

# 建议 Antigravity 执行的测试内容

下面这部分你也可以一起发给它，或者自己保留做验收标准。

---

## Phase 1 测试总原则

每完成一个子步骤，都必须做两类验证：

1. **单元测试不破**
2. **V2 baseline 不漂移**

---

## Step 1：Planner 去重后的测试

### 必须通过的旧测试

```bash
python -m pytest tests/test_planner.py -v
```

### 建议新增测试

新建：

```text
tests/test_planner_v2.py
```

建议至少包含这 4 个：

```text
test_v2_astar_finds_path
- 验证 plan_next_action_v2() 在 lattice_v2 上能找到合法路径

test_v2_cost_respects_risk_head
- 验证 cell_cost_v2() 会随 risk head 输出变化而变化

test_v2_warned_extra_cost
- 验证 warned_cell_extra_cost 或等价 bias 机制会真实改变路径代价

test_astar_core_equivalence
- 在相同 cost_fn / passable / budget 下，新 _astar_core() 与旧行为一致
```

### 回归验证

```bash
python scripts/_diag_l2c1_sweep.py
```

### 预期

* 所有旧 planner tests 仍过
* V2 sweep 结果不变
* planner 文件内部重复度下降，但外部 API 不变

---

## Step 2：Warning 抽象后的测试

### 必须通过的旧测试

```bash
python -m pytest tests/test_rsa_warning.py -v
```

### 建议新增测试

新建：

```text
tests/test_warning_protocol.py
```

建议包含这 6 个：

```text
test_rsa_warner_implements_protocol
- RSAWarner 满足 PragmaticWarner 协议

test_lane_warner_implements_protocol
- LaneWarner 满足 PragmaticWarner 协议

test_rsa_select_returns_valid_utterance
- RSAWarner.select_utterance() 返回合法 utterance

test_lane_select_returns_valid_utterance
- LaneWarner.select_utterance() 返回合法 utterance

test_lane_listener_update_changes_belief
- LaneWarner.listener_update() 会改变 feature belief 或对应 bias

test_lane_bias_computable
- lane-bias / warning score 可计算且数值合理
```

### 回归验证

```bash
python -m pytest tests/test_rsa_warning.py tests/test_warning_protocol.py -v
python scripts/_diag_l2c1_sweep.py
```

### 预期

* RSA warning 旧行为不变
* V2 warning 行为不变
* 两套 warning 都能通过统一接口调用
* 未来如果替换 speaker/listener，不需要重写调用链

---

## Step 3：Belief 协议后的测试

### 必须通过的旧测试

```bash
python -m pytest tests/test_belief_update.py -v
```

### 建议新增测试

新建：

```text
tests/test_belief_protocol.py
```

建议包含这 4 个：

```text
test_belief_map_satisfies_protocol
- BeliefMap 满足 CellBelief 协议

test_feature_belief_satisfies_protocol
- FeatureBeliefMap 满足 CellBelief 协议

test_protocol_methods_exist
- update / get_belief / copy / reset 都存在且可调用

test_protocol_copy_reset_behavior
- copy 与 reset 的基本行为合理
```

### 回归验证

```bash
python -m pytest tests/test_belief_update.py tests/test_belief_protocol.py -v
python scripts/_diag_l2c1_sweep.py
```

### 预期

* belief 调用边界更清晰
* 但不把 scalar belief 和 feature belief 强行合并
* V2 逻辑不变，旧系统逻辑也不变

---

## Step 4：V2 runner / wrapper 提案阶段

这一阶段**先不要直接大改**。
要求 Antigravity 先给出设计提案，再决定是否实现。

### 希望它输出的内容

```text
1. 推荐做 minimal runner 还是 minimal Gym wrapper
2. teacher action 的 step 频率怎么定义
3. 哪些状态要进入 observation / info
4. 怎样保证不破坏现有 sweep 逻辑
5. 最小新增文件集
```

### 如果它建议实现

建议优先做：

```text
src/envs/lattice_v2_runner.py
```

而不是一上来做一个非常重的 `gym.Env` 封装。

### 预期

* V2 运行逻辑从脚本里稍微抽离
* 但不强行产品化
* 为后续 latent vector / multi-cell prediction 留接口

---

# 建议 Antigravity 的执行顺序

你可以再补一句，让它按这个顺序来：

```text
Please execute in this exact order:
1. inspect and diagnose
2. planner deduplication
3. planner tests + V2 regression
4. warning abstraction
5. warning tests + V2 regression
6. belief protocol
7. belief tests + V2 regression
8. only then propose V2 runner/env design
```

---

# 你可以要求它修改的文件范围

为了防止它乱动，建议明确写上：

```text
You may edit only these files in the first pass:
- src/agents/planner_astar.py
- src/teachers/rsa_warning.py
- src/agents/warning_update.py
- src/agents/belief.py
- src/agents/feature_belief.py

You may add only these new files in the first pass:
- src/agents/pragmatic_warning.py
- src/agents/belief_protocol.py
- tests/test_planner_v2.py
- tests/test_warning_protocol.py
- tests/test_belief_protocol.py

Do not edit env files yet unless necessary for testability.
Do not change scripts unless required for regression verification.
```

---

# 这一阶段的预期结果

如果 Phase 1 做得对，到最后你应该看到的是：

### 代码层

* `planner_astar.py` 不再有两套高度重复的 A* 核心
* `rsa_warning.py` 与 `warning_update.py` 有统一入口
* `BeliefMap` 与 `FeatureBeliefMap` 至少有共同协议
* V2 还是现在的 V2，但以后更容易升级

### 实验层

* 当前 77 tests 继续通过
* 新增 10–15 个 V2 / protocol tests
* V2 baseline 不漂移

### 研究层

* Phase 1 结束后，你就能进入下一阶段：

  * latent vector jointly determines cost + risk
  * noisy multi-cell observation
  * multi-cell prediction in planning
  * robot model over agent belief

也就是说，**Phase 1 的意义不是功能大增，而是把后面真正重要的复杂能力变成“可接上去”的状态。**

---

# 最后给你的简短建议

你现在发给 Antigravity 时，最好不要只发一句“帮我改代码”。
最稳的方式是：

1. 先发上面那段 **任务单**
2. 再发 **测试要求**
3. 明确要求它 **先出诊断和计划，再改代码**
4. 明确要求它 **每一步都跑测试和 V2 baseline 回归**

这样它就更不容易直接把项目往“大重构”方向带偏。

你要的话，我下一条可以继续直接给你一版**更短、更适合直接粘贴到 Antigravity 输入框里的精简版 prompt**。
