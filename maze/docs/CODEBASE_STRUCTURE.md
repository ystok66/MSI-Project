# 代码结构规范

这份文档定义 `maze/` 工作区后续的默认代码组织方式。目标不是做“特别复杂的企业目录”，而是确保：

- 每个文件职责单一
- 新机制能按层次插入
- runner、policy、env、belief 不再混在同一个文件里
- 以后加 `WAYPOINT`、更复杂 eval、真正 inverse rollout 时不需要再次大搬家

## 1. 顶层目录约定

```text
maze/
├─ risky_maze/   # 主包
├─ docs/         # 稳定文档与规范
├─ tests/        # 测试
└─ README.md     # 入口说明
```

## 2. 主包分层

```text
risky_maze/
├─ core/
├─ env/
├─ learner/
├─ scenarios/
├─ tutor/
├─ runner/
├─ config.py
├─ demo.py
└─ __init__.py
```

### `core/`

只放跨模块共享、且不带具体业务状态的内容：

- 枚举、dataclass、共享类型
- 动作与坐标定义
- 通用路径搜索函数

不应该放：

- learner 参数更新
- tutor 策略
- env rollout 逻辑

### `env/`

只负责世界本身：

- prototype bank
- maze layout
- layout 生成器
- episode step / observation dynamics

不应该放：

- learner belief
- tutor utility
- teach/eval orchestration

### `learner/`

只放 learner 侧内容：

- risk belief
- map memory
- planner / action policy

如果以后增加：

- RSA listener
- memory variants
- multiple learner profiles

也应该优先放这里。

### `tutor/`

只放 tutor 侧内容：

- tutor base protocol
- warning / waypoint / mixed intervention policies
- future inverse-planning rollout code

不要把 tutor 的 episode loop 写进这里。episode loop 属于 `runner/`。

### `scenarios/`

只放固定场景资产和相关校验逻辑：

- hand-authored map specs
- task suites
- spec validator
- future `v1 / v2` map variants

不要把 learner / tutor 行为逻辑写进这里。

### `runner/`

只放实验执行流程：

- 单 episode 执行
- teach/eval block 执行
- metrics 聚合

这里是“把 env、learner、tutor 接起来”的地方。

## 3. 文件粒度规则

默认遵守下面的拆分原则：

- 一个 `.py` 文件只承载一类主要责任。
- 如果一个文件同时在做“数据结构 + 更新算法 + 运行流程”，就该拆。
- 如果一个文件超过约 200-300 行，优先检查是不是已经跨了职责边界。
- 新增机制优先加新模块，不优先把旧文件继续堆大。

推荐模式：

- `risk_belief.py` 放 belief 更新
- `memory.py` 放 memory state
- `agent.py` 放把 belief + memory 组合成 policy 的 agent

不推荐模式：

- `learner.py` 里同时放 belief、memory、policy、analysis helper

## 4. import 方向规则

默认 import 方向应尽量单向：

```text
core -> env / learner / tutor / runner
env -> core
learner -> core
tutor -> core + learner + env
runner -> env + learner + tutor
```

尽量避免：

- `env` 反向 import `runner`
- `learner` 反向 import `tutor`
- 跨层循环依赖

如果某个工具被多层共用，就把它下沉到 `core/`。

## 5. 命名规范

- 包名和模块名用小写加下划线。
- 类名用 `CamelCase`。
- 函数和变量用 `snake_case`。
- 配置类统一放在 `config.py`，使用 dataclass。
- 结果聚合函数使用明确动词，例如 `run_block`、`merge_episode_metrics`。

策略命名建议：

- `NoTutor`
- `AlwaysWarnTutor`
- `InverseWarnTutor`

而不是：

- `TutorV2`
- `BetterTutor`
- `NewPolicy`

## 6. 测试组织规范

测试放在 `tests/` 下，并按“机制类别”而不是“日期”命名。

当前推荐：

- `test_smoke.py`
- `test_env_generation.py`
- `test_warning_update.py`
- `test_inverse_warn_tutor.py`

如果以后测试增多，可以再拆子目录，但先保持简单。

## 7. 后续扩展时的落点

### 加 `WAYPOINT`

放在：

- `risky_maze/tutor/waypoint_policies.py`

如果需要新的 action schema，再更新：

- `risky_maze/core/types.py`

### 加 richer metrics

放在：

- `risky_maze/runner/metrics.py`

### 加新的 eval split

放在：

- `risky_maze/runner/block_runner.py`

必要时把 split 配置抽成单独模块。

### 加 learner profile / inverse model

放在：

- `risky_maze/learner/`
- `risky_maze/tutor/`

不要把 shadow learner 或 rollout 逻辑塞回 `runner/`。

## 8. 本次重构后的原则

后续如果要继续整理，优先级如下：

1. 先保持分层边界清楚
2. 再优化命名和小工具抽取
3. 最后才考虑更细的目录颗粒度

也就是说：

- 先避免“大杂烩文件”
- 不急着把目录拆到过度复杂
