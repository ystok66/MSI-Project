# Phase 3 完成总结：V2 环境接口

## 结果

- **105/105 tests pass**（+8 env API tests）
- **V2 基线不变**：no_tutor=9%, warn=80%, door_2=68%, door_3=99%, close=100%

---

### 改了什么

#### 新建 `src/envs/lattice_v2_env.py`

薄 facade，零重复逻辑：

| 方法 | 语义 |
|------|------|
| `reset(seed, **config)` | 新 episode → 初始 Observation |
| `observe_agent()` | agent 当前可见信息 |
| `step_teacher()` | teacher 决策（door/warn），不动 agent |
| `step_agent()` | plan→move→outcome |
| `step_full()` | **语义基准** = observe→teacher→agent |
| `get_state()` | 完整诊断快照 |
| `get_metrics()` | 兼容 sweep 输出 |

4 个 schema dataclass：`Observation`、`TeacherInfo`、`StepResult`、`StateSnapshot`

#### 修改 `src/envs/lattice_v2_runner.py`（~15 行）

- `step()` 拆成 `observe()` + `apply_tutor()` + `plan_and_move()`
- `step()` 改为组合调用这三个，行为不变
- `_apply_tutor` → `_apply_tutor_dispatch`

#### 新建 `tests/test_v2_env_api.py`（8 tests）

- reset 可复现 / observe_agent schema / get_state schema
- step_teacher 不动 agent / step_agent 推进 episode
- step_full 与 runner 严格等价（固定 seed）
- terminal + metrics 一致 / teacher trigger region 外 no-op

### 设计原则

- env 只持有 runner state 的引用，无重复 state
- `step_full()` 是语义唯一基准
- `step_teacher()` + `step_agent()` 为后续 latent vector / robot belief 留钩子
