可以。先校准一下你现在的阶段定义：

**按你当前进度，Phase 2 已经完成了“runner 平台化”。**
所以现在我建议把 **Phase 3** 定义为：

> **V2 environment API / thin wrapper 阶段**
> 在不改变科学内容的前提下，把已经存在的 `LatticeV2Runner` 再往上抬一层，做成一个清晰、稳定、可测试、可扩展的环境接口。

这一步不是去加 latent vector、noisy observation 或 multi-cell prediction；而是先把 **“未来复杂能力要挂在哪里”** 这件事定下来。

另外，给 antigravity 的工作流上，我建议你这次明确要求它：**先只产出 Implementation Plan，再实现；执行时保留 task list；完成后给 walkthrough；终端权限保持 Request Review。** 这些都是它官方文档里已经明确支持的工作流组件。([Google Antigravity][1])

---

# Phase 3 的大致目的

现在 V2 已经有：

* `lattice_v2_runner.py`
* 明确的 `V2EpisodeState`
* 明确 teacher cadence
* 薄化后的 sweep script
* 稳定 baseline

所以 Phase 3 不该重复做 runner，而应该做：

1. **给 V2 加一个清晰的环境接口层**
2. **把 runner 变成内部执行核心**
3. **把外部调用方只面向 env API，而不是直接拼 runner/state/script**
4. **给后面 Phase 4+ 留下扩展挂点**

一句话说：

**Phase 3 是“从 first-class runnable platform，走向 first-class environment interface”。**

---

# Phase 3 的思路

## 1. 不做重型 Gym 化，先做 thin env

你之前那句“先 A 后 B”在现在这个状态下仍然成立。
因为 runner 已经有了，所以现在最合理的做法是：

* **先做一个 Python 环境接口层**
* 不要求立刻完整兼容通用 RL `gym.Env`
* 但接口风格可以尽量接近标准 env 习惯

我建议新增：

* `src/envs/lattice_v2_env.py`

它本质上是一个 **thin facade over runner**，不是第二套执行逻辑。

## 2. runner 是 engine，env 是 facade

这一步最关键的设计边界是：

* `runner` 负责真正 episode 推进
* `env` 负责对外暴露稳定 API

不要让 `lattice_v2_env.py` 自己再复制一遍 episode loop。

## 3. 明确“teacher step”和“agent step”

这一步最该固定的是 step semantics。

因为你后面会做：

* latent vector
* noisy local patch observation
* multi-cell prediction
* robot belief over agent belief

这些东西都需要你现在先把下面几个接口固定下来：

* `reset(seed)`
* `get_state()`
* `observe_agent()`
* `step_teacher(action)`
* `step_agent()`
* `get_metrics()`

可选再加一个：

* `step_cycle(teacher_action=None)`

## 4. 这一步仍然不改科学行为

Phase 3 只是把“接口层”立起来，不该做：

* latent feature vector
* learned cost+risk head
* multi-cell sensing
* item drop
* nested belief
* 新 teacher policy 设计

---

# Phase 3 的方法

## 方法 A：最小新增文件，少改旧核心

建议这一步只允许小范围编辑：

可编辑：

* `src/envs/lattice_v2_runner.py`
* `src/envs/lattice_v2.py`
* `src/teachers/time_aware_door_tutor.py`
* 必要时少量 `configs/*.yaml`

可新增：

* `src/envs/lattice_v2_env.py`
* `tests/test_v2_env_api.py`

## 方法 B：先固定 schema，再写逻辑

比起先写代码，更重要的是让 antigravity 先定：

* env 对外 observation schema
* state snapshot schema
* info schema
* teacher action schema
* terminal / truncation / success / failure 的标志语义

## 方法 C：只做一层 facade，不复制逻辑

如果它的方案里出现：

* 再写一套 reset
* 再写一套 step
* 再维护一套 episode state

那就说明方向错了。

---

# 下面是可以直接给 Antigravity 的任务单

```text
Project: pedagogical_ip
Phase: Phase 3 — V2 environment interface (thin wrapper over runner)

Current status
Phase 0 complete.
Phase 1 complete:
- planner deduplication
- warning abstraction
- belief protocol
Phase 2 complete:
- LatticeV2Runner exists
- V2EpisodeState exists
- runner-based V2 execution works
- V2 baselines remain unchanged
- tests increased to 97

High-level goal
This phase is NOT about new science yet.
This phase is about making V2 expose a clean environment API on top of the existing runner.

We want Lattice V2 to become a first-class environment interface, not just a first-class runner.

Core design principle
- runner remains the internal execution engine
- env becomes a thin facade over runner
- do not duplicate episode logic
- do not change scientific behavior

What this phase should prepare for later
This phase should prepare the codebase for future phases:
- latent vector per cell
- noisy local patch observation
- learned cost + learned risk
- multi-cell prediction
- robot belief over agent belief

But none of those should be implemented now.

What you must NOT do
- do NOT rewrite V2 into a heavy Gym refactor
- do NOT create a second episode engine separate from the runner
- do NOT redesign the teacher science
- do NOT add latent vector / noisy observation / cost+risk learning yet
- do NOT touch planner/warning/belief internals except for wiring if absolutely necessary
- do NOT change baseline behavior

Primary goal of this phase
Create a thin V2 environment API layer, likely:

- reset(seed=None, config=None)
- observe_agent()
- step_teacher(action)
- step_agent()
- get_state()
- get_metrics()

Optional:
- step_cycle(teacher_action=None)

The environment wrapper should delegate to LatticeV2Runner internally.

Files to inspect first
- src/envs/lattice_v2_runner.py
- src/envs/lattice_v2.py
- src/teachers/time_aware_door_tutor.py
- scripts/_diag_l2c1_sweep.py
- tests/test_v2_runner.py
- configs/env.yaml
- configs/teacher.yaml
- configs/experiment.yaml

Before editing code, produce a design proposal that answers:
1. What should the public V2 env API be?
2. What should reset() return?
3. What should observe_agent() return?
4. What exactly is the semantic difference between step_teacher() and step_agent()?
5. Should there be a convenience step_cycle()?
6. What fields belong in get_state() vs get_metrics() vs info?
7. What is the smallest safe file-change set?

Expected design preference
Prefer a thin Python env facade first.
Only propose full Gym compatibility if it is almost free and does not distort teacher cadence semantics.

Implementation priorities

Priority 1:
Add a new file:
- src/envs/lattice_v2_env.py

This file should:
- wrap the existing runner
- expose a stable public API
- avoid reimplementing episode logic

Priority 2:
Make observation/state/info schemas explicit and documented.

Priority 3:
If useful, add a very small compatibility layer for scripts so future code can call env instead of manipulating runner/state directly.

Allowed edits in first implementation pass
You may edit:
- src/envs/lattice_v2_runner.py
- src/envs/lattice_v2.py
- src/teachers/time_aware_door_tutor.py
- scripts/_diag_l2c1_sweep.py (only if needed for thin wiring)
- configs/env.yaml
- configs/teacher.yaml
- configs/experiment.yaml

You may add:
- src/envs/lattice_v2_env.py
- tests/test_v2_env_api.py

Avoid editing any other files unless absolutely necessary.

Required output format before implementation
1. concise diagnosis
2. proposed public env API
3. observation/state/info schema
4. teacher-step / agent-step semantics
5. file-by-file change plan
6. proposed tests
7. over-engineering risks

Acceptance criteria
Phase 3 is successful only if:
- all 97 existing tests still pass
- new env API tests pass
- V2 baselines remain exactly unchanged
- env wrapper is thin and delegates to runner
- future phases can plug into a stable environment API

Current V2 baseline that must remain unchanged
- no_tutor = 9%
- warning_only (lambda=5) = 80%
- door_2 = 68%
- door_3 = 99%
- always_close = 100%
- lambda sweep: 1→9%, 3→46%, 5→80%, 7→100%

Please start with diagnosis + design proposal only.
Do not implement until the plan is written.
```

---

# 建议的测试内容

这一步的测试重点不是算法正确性，而是：

> **环境接口是否清晰、runner 是否仍是唯一执行核心、行为是否完全不漂移。**

## 必须继续通过的旧测试

先全跑：

```bash
python -m pytest tests/ -v --tb=short
```

预期：

* 现有 **97 tests 全通过**

---

## 建议新增测试

建议新增：

* `tests/test_v2_env_api.py`

我建议至少写这 7 个。

### 1. `test_env_reset_reproducible`

验证：

* 相同 seed 下 `env.reset(seed=...)` 的关键初始状态一致
* 至少检查：

  * agent 初始位置
  * 地图/segment 布局
  * tutor mode 相关状态
  * step/time 为 0

### 2. `test_env_observe_agent_schema`

验证：

* `observe_agent()` 返回结构固定
* 至少包含你后面会继续用到的字段
* 不要求复杂，只要 schema 清楚且稳定

### 3. `test_env_get_state_schema`

验证：

* `get_state()` 返回完整状态快照
* 至少包括：

  * agent position
  * current step / time
  * visited cells
  * warning memory
  * gate state
  * terminal flags

### 4. `test_env_step_teacher_semantics`

验证：

* `step_teacher(action)` 不会偷偷执行 agent step
* teacher cadence 语义明确
* 非触发位置时 teacher action 被忽略或安全处理的行为是确定的

### 5. `test_env_step_agent_progresses_episode`

验证：

* `step_agent()` 真正推进 episode
* step 数、位置、累计指标至少有一项正确更新

### 6. `test_env_cycle_matches_runner_fixed_seed`

验证：

* 在固定 seed + 固定 teacher action 序列下
* `lattice_v2_env.py` 路径与 `runner` 路径结果一致
* 这是最关键的一条：证明 env 只是 facade，不是第二套逻辑

### 7. `test_env_terminal_and_metrics_consistent`

验证：

* 到达 terminal 后
* `get_metrics()` 与 terminal flags 一致
* success / failure / truncation 语义不冲突

---

## 可选新增测试

如果 antigravity 提议加一个 `step_cycle()`，再补 2 个：

### 8. `test_env_step_cycle_runs_one_teacher_agent_cycle`

验证：

* `step_cycle()` 的语义等于

  * `step_teacher(...)`
  * 再 `step_agent()`

### 9. `test_env_invalid_teacher_action_handled`

验证：

* 非法 teacher action 不会让环境状态坏掉
* 要么报清晰错误，要么稳定 no-op

---

# 建议的回归验证

除了 pytest，这一步一定要保留 sweep 回归：

```bash
python scripts/_diag_l2c1_sweep.py
```

预期结果必须仍然是：

* `no_tutor = 9%`
* `warning_only (λ=5) = 80%`
* `door_2 = 68%`
* `door_3 = 99%`
* `always_close = 100%`
* `lambda sweep: 1→9%, 3→46%, 5→80%, 7→100%`

这个是 Phase 3 最重要的验收标准。

---

# 这一阶段的预期结果

如果它做对了，你最后应该得到的是：

## 代码层

* 新增 `src/envs/lattice_v2_env.py`
* `runner` 仍是唯一执行核心
* `env` 只是稳定对外 API
* `lattice_v2.py` / scripts 不再承担接口设计职责

## 测试层

* 97 个旧测试仍全过
* 新增约 7–9 个 env API 测试
* baseline 完全不漂移

## 研究层

* 后面 Phase 4 可以自然接入：

  * latent vector per cell
  * noisy local patch observation
  * learned cost+risk heads
  * multi-cell prediction

也就是：

**Phase 3 完成后，你终于有了一个“可扩展的环境壳”，后面复杂科学内容可以往里面挂。**

---

# 我对你这一步的简短建议

你这次给 antigravity，最好分两轮：

第一轮只发任务单，让它先出 **diagnosis + design proposal + file plan**。
第二轮你看完没问题，再允许它实现。

这样最稳，也最符合它官方支持的“先 plan、再执行、再 walkthrough”的工作流。([Google Antigravity][1])

要是你愿意，我下一条可以继续给你一版**更短、适合直接粘贴到 antigravity 输入框的一屏版 prompt**。

[1]: https://antigravity.google/docs/implementation-plan?utm_source=chatgpt.com "Implementation Plan"
