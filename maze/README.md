# Risky Maze

`Risky Maze` 是一个面向研究的 gridworld / maze 教学环境，用来研究下面这个核心问题：

> tutor 的价值是否不仅仅是“阻止失败”，而是在
> `允许有价值探索`、`阻止灾难`、`减少无效重复`、`帮助形成可复用知识`
> 之间做动态权衡。

这个仓库当前包含两条并行主线：

- `random-maze prototype`
  用于快速验证环境、risk learner、warning 机制和基础 runner。
- `fixed-map POMDP runtime`
  用于在固定大图 `HugeRiskyGemMaze_v0` 上稳定比较
  `no tutor / warning tutor / inverse-planning tutor / waypoint tutor`
  的行为差异。

当前重点已经从早期的单文件原型，转向一个分层、可扩展、可做实验的代码结构：

- `env` 负责世界与隐藏状态
- `learner` 负责记忆、risk belief 与规划
- `tutor` 负责干预策略与 inverse planning
- `runner` 负责 teach / eval 实验编排

## 1. 项目想解决什么问题

这个项目不是普通的 shortest-path 迷宫。

我们关心的是一种更教学化的设置：

- 环境里有墙、门、宝石、出口、陷阱。
- 陷阱不是直接可见标签，而是隐藏在 noisy continuous feature vector 背后。
- learner 通过探索获得两种长期收益：
  - 地图记忆收益：记住哪里通、哪里堵、哪里有门、哪里有捷径。
  - risk concept 收益：记住什么样的向量模式更像 danger，并在新区域上泛化。
- tutor 不是每次都给最优答案，而是要判断：
  - 现在应该 `WAIT` 让 learner 继续学吗？
  - 现在应该 `WARNING` 阻止高风险前缀吗？
  - 现在应该 `WAYPOINT` 避免超时、死循环或无效游走吗？

项目的核心论点是：

> “探索”只有在它形成可复用知识时才有价值。
> 所以我们不只看 teach 当下是否成功，而是看这些探索是否在后续 eval 中被复用。

## 2. 当前实现到什么程度

当前仓库已经具备以下能力。

### 已经实现

- 固定地图 POMDP runtime：
  - 隐状态、局部观测、HP / time dynamics
  - objective sequence
  - baseline modes
- 固定大图 `HugeRiskyGemMaze_v0`：
  - hand-authored map
  - teach / eval task suite
  - spec validator
- learner：
  - map memory
  - online Gaussian risk belief
  - objective-aware A* planner
  - warning-based posterior update
  - temporary waypoint support
- tutor：
  - `no_tutor`
  - `always_warn`
  - `risk_threshold_warn`
  - heuristic `inverse_warn`
  - finite-profile short-horizon `inverse_plan_warn_only`
  - finite-profile short-horizon `inverse_plan_full`
- runner / metrics：
  - teach + eval_same_map execution
  - per-step logging
  - risk / map / assist metrics
  - CSV output for analysis
- tests：
  - fixed runtime smoke tests
  - inverse tutor overlay tests
  - experiment runner tests

### 当前仍在推进

- `inverse_plan_warn_only` 的策略质量还需要继续调试
- `WAYPOINT` 的带宽约束和过强问题还需要系统实验
- `eval_new_map_same_risk` 这一条还没有完全接入固定大图主实验流程
- richer RSA semantics 还没有进入主线实现
- full I-POMDP / nested belief solver 还没有做，这里仍然是近似 inverse planning

## 3. 核心理念

这个项目的设计不是“让 tutor 尽可能帮 learner 完成任务”，而是下面这些原则。

### 3.1 探索必须产生长期价值

我们不把“多走了很多格子”当成学习成功。

我们更关心：

- teach 中发现的新区域，eval 里是否真的被使用了
- warning 是否真的降低了 risk uncertainty
- learner 是否学会了 risk vector，而不只是记住几个 trap 坐标

### 3.2 tutor 不能退化成 oracle navigator

如果 tutor 每一步都把最优路告诉 learner，那么 teach success 可能很高，但 learner 自己并没有学会。

所以本项目默认要求：

- warning 不能退化成“具体哪个格子是 trap”的直接标签
- waypoint 不能逐步给完整 oracle path
- eval phase tutor 必须关闭

### 3.3 先做近似可运行，再逐步上更强模型

仓库当前刻意避免一开始就上 full POMDP solver 或 nested I-POMDP。

当前的 tutor 采用的是：

- 有限 learner profile posterior
- shadow learner snapshot
- top-K path prediction
- short-horizon counterfactual rollout

这是一个工程上可控、调试友好、能先跑实验的版本。

## 4. 世界设定与任务形式

### 4.1 基本世界

世界是一个部分可观测的 grid maze。

元素包括：

- `wall`
- `walkable floor`
- `trap`
- `key`
- `door`
- `gem`
- `exit`

固定地图主场景是：

- [`HugeRiskyGemMaze_v0`](./docs/specs/HUGE_RISKY_GEM_MAZE_V0.md)
- 资产位置：
  - [`risky_maze/scenarios/huge_risky_gem_maze_v0.py`](./risky_maze/scenarios/huge_risky_gem_maze_v0.py)
  - [`risky_maze/scenarios/huge_risky_gem_maze_v0.json`](./risky_maze/scenarios/huge_risky_gem_maze_v0.json)
  - [`risky_maze/scenarios/validation.py`](./risky_maze/scenarios/validation.py)

### 4.2 坐标约定

这里有一个很重要的约定：

- 场景 spec 对外使用 `(x, y)`，原点在左上角。
- runtime 内部主要使用 `(row, col)`，也就是 `(y, x)`。

这个转换由 [`risky_maze/env/fixed_loader.py`](./risky_maze/env/fixed_loader.py) 负责。

### 4.3 objective sequence

当前固定地图 runtime 支持顺序目标：

- `pickup`
- `pass`
- `collect_gem`
- `exit`

也就是说，一个任务不再只是 “从起点到终点”，而是：

```text
start -> objective_1 -> objective_2 -> ... -> final exit
```

这一层定义在：

- [`risky_maze/env/objectives.py`](./risky_maze/env/objectives.py)

## 5. 形式化：为什么这是 POMDP

对 learner 来说，这个环境天然是 POMDP。

### 5.1 隐状态

可以把真实状态写成：

```text
s_t = (
  agent_pos,
  hp,
  time_remaining,
  full_layout,
  objective_progress,
  trap_types,
  latent_risk_features,
  inventory
)
```

其中 learner 不能直接看到：

- 完整地图
- `r / m / q` 这类 true trap label
- 每个格子的 latent trap type

### 5.2 观测

learner 在每个时间步看到的是：

```text
o_t = (
  local visible cells,
  learner-facing visible_kind,
  noisy risk vectors for visible walkable cells,
  hp,
  current objective,
  current position
)
```

当前 runtime 中最关键的约束是：

> observation 只暴露 `wall / key / door / gem / exit / walkable`
> 这类 learner-facing visible kind，
> 永远不会把 true `r / m / q` trap label 直接泄露给 learner。

对应代码：

- [`risky_maze/env/pomdp_episode.py`](./risky_maze/env/pomdp_episode.py)
- [`risky_maze/env/fixed_loader.py`](./risky_maze/env/fixed_loader.py)

## 6. risk vector 与 concept learning

### 6.1 latent type 和 noisy observation

对每个 walkable cell `c`，我们假设存在一个 latent class `z_c`。

安全与风险 cell 由不同 prototype 生成：

```text
z_c in {safe_1, ..., safe_K, danger_1, ..., danger_M}
v_c ~ N(mu_{z_c}, sigma_cluster^2 I)
x_c = v_c + epsilon, epsilon ~ N(0, sigma_obs^2 I)
```

其中：

- `v_c` 是 cell 的 latent feature
- `x_c` 是 learner 实际观测到的 noisy vector

在固定地图 spec 里，`r / m / q` 只是 oracle 使用的 latent trap class 标记；
learner 实际接收到的是 noisy feature vector。

### 6.2 当前 learner 的 risk belief

固定地图主线目前使用一个简化版的 binary Gaussian risk belief：

- 一个 `safe_mean`
- 一个 `danger_mean`
- 一个 online-updated `prior_danger`

对于某个 feature `x`，当前 danger 概率近似为：

```text
P(danger | x)
  propto P(x | danger) P(danger)
```

实现位置：

- [`risky_maze/learner/objective_agent.py`](./risky_maze/learner/objective_agent.py)

对应到代码里的直觉是：

- 如果 `x` 更接近 `danger_mean`，`danger_probability(x)` 就更高
- 如果 learner 在 trap 上踩到 damage，它会对 danger prototype 做 supervised update
- 如果 learner 安全通过某格子，也会对 safe prototype 做较弱更新

### 6.3 warning 的语义

当前 warning 不是：

```text
"cell (x, y) is danger"
```

而是更接近：

```text
"你当前计划经过的一组 cells 里，至少有一个危险"
```

这和旧 `cls_color_selection` 系统里的 set-level warning 是一致的。

### 6.4 warning update 的当前公式

如果 warned set 中每个 cell 的先验危险概率是 `p_i`，
当前实现用一个简化的集合条件化：

```text
P(warning) = 1 - Prod_i (1 - p_i)
p_i' = p_i / max(eps, P(warning))
```

然后对这些更新后的 `p_i'` 做 soft prototype update。

这部分逻辑在：

- [`OnlineGaussianRiskBelief.warning_update`](./risky_maze/learner/objective_agent.py)

这个版本仍然是 literal Bayesian set update，不是完整 RSA。

## 7. learner 机制

固定地图主线 learner 由三部分组成。

### 7.1 Map memory

learner 维护：

- 已知墙 `known_walls`
- 已知可通行格 `known_walkable`
- 每格见过的 `visible_kind`
- 每格累计观测到的 risk vectors
- 已确认 trap / safe cells
- 访问次数
- warning suspicion

实现位置：

- [`SimpleMapMemory`](./risky_maze/learner/objective_agent.py)

### 7.2 Risk belief

learner 维护一个在线更新的 `OnlineGaussianRiskBelief`：

- `safe_mean`
- `danger_mean`
- `safe_count`
- `danger_count`
- `prior_danger`

### 7.3 Planner

当前 fixed-map learner 使用 objective-aware A*。

对每个候选 cell，planner 的 step cost 近似为：

```text
cost(cell) =
  1
  + risk_weight * P(danger | x_cell)
  + revisit_penalty * visit_count(cell)
  + unknown_penalty * 1[cell unknown]
  + warning_suspicion_weight * suspicion(cell)
```

当前默认参数大致是：

- `risk_weight = 4.0`
- `revisit_penalty = 0.15`
- `unknown_penalty = 0.20`
- `warning_suspicion_weight = 2.0`

这意味着 learner 会自然权衡：

- 更短路径
- 更低风险
- 更少重复
- 更少踩 warning 怀疑区域

实现位置：

- [`ObjectiveAwareLearner`](./risky_maze/learner/objective_agent.py)

## 8. tutor 机制

### 8.1 action space

当前 tutor action space 是：

- `WAIT`
- `WARNING`
- `WAYPOINT`

但不同 tutor mode 不一定都允许全部动作。

### 8.2 baseline tutors

已经实现的 baseline 包括：

- `no_tutor`
- `always_warn`
- `risk_threshold_warn`
- `inverse_warn`

其中：

- `always_warn` 只要预测前缀里有明显风险就警告
- `risk_threshold_warn` 更依赖固定阈值
- `inverse_warn` 是早期启发式 inverse tutor，不是完整 rollout tutor

相关位置：

- [`risky_maze/tutor/warning_policies.py`](./risky_maze/tutor/warning_policies.py)
- [`risky_maze/tutor/baselines.py`](./risky_maze/tutor/baselines.py)

### 8.3 当前 inverse planning tutor 不是 full I-POMDP

当前主线的 inverse tutor 是一个近似实现：

1. 维护有限 learner profile posterior
2. 从 learner memory / risk snapshots 构造 shadow learner
3. 预测 top-K candidate paths
4. 对 `WAIT / WARNING / WAYPOINT` 做短 horizon counterfactual rollout
5. 选择 utility 最高的干预

实现位置：

- [`risky_maze/tutor/profiles.py`](./risky_maze/tutor/profiles.py)
- [`risky_maze/tutor/shadow.py`](./risky_maze/tutor/shadow.py)
- [`risky_maze/tutor/path_predictor.py`](./risky_maze/tutor/path_predictor.py)
- [`risky_maze/tutor/candidates.py`](./risky_maze/tutor/candidates.py)
- [`risky_maze/tutor/rollout.py`](./risky_maze/tutor/rollout.py)
- [`risky_maze/tutor/inverse_planner.py`](./risky_maze/tutor/inverse_planner.py)

### 8.4 profile posterior

tutor 当前不做无限复杂的 belief inference，而是维护一个有限 learner profile 分布。

直觉上，它在问：

- learner 更像 risk-averse 还是 curious?
- learner 当前动作更符合哪个 profile 预测出来的 path?

然后用 tempered Bayesian update 去更新 profile belief。

### 8.5 tutor utility

当前 rollout evaluator 的 utility 是一个线性组合，大体形式为：

```text
Q(a) =
  + w_success * P(success)
  - w_death   * P(death)
  - w_damage  * E[damage]
  - w_timeout * P(timeout)
  + w_map     * E[map_gain]
  + w_riskIG  * E[risk_info_gain]
  + w_eval    * E[eval_gain_proxy]
  - w_cost    * intervention_cost(a)
  - w_assist  * assist_leakage(a)
  - w_bore    * boredom_cost(a)
```

当前实现里的一组默认权重大致是：

- `success = 4.0`
- `death = 30.0`
- `damage = 5.0`
- `timeout = 8.0`
- `map_gain = 0.35`
- `risk_ig = 1.5`
- `eval_gain = 0.5`
- `cost = 1.0`
- `assist = 2.0`
- `boredom = 1.0`

同时 action 还有带宽 / 泄露代价：

- `warning_cost = 0.2`
- `waypoint_cost = 0.8`

这正是本项目最关键的研究理念：

> tutor 不是只为了当前一步更安全，
> 也不是只为了当前任务更快结束，
> 而是在 safety、learning value 和 assistance leakage 之间做权衡。

### 8.6 guardrails

为了防止 tutor 过于激进，当前还有一些 guardrails：

- `waypoint_cooldown_steps`
- `waypoint_min_advantage_over_wait`
- `warning_min_advantage_over_wait`
- eval phase tutor disabled

这些逻辑主要在：

- [`risky_maze/tutor/inverse_planner.py`](./risky_maze/tutor/inverse_planner.py)

## 9. baseline modes

固定地图 runtime 已实现 3 种 baseline mode。

### `mortal`

- trap 会造成 damage
- HP 降到 0 会死亡
- timeout 正常生效

### `immortal_warnlike`

- teach 中踩到 trap 不会真正死亡
- 会记录 `immortal_danger_event`
- learner 仍然能从这次 danger encounter 中学 risk

这个 baseline 的意义是：

> 估计“去掉灾难终止成本后，探索最多能学到多少”。

### `immortal_no_timeout`

- teach 中忽略 death / timeout
- eval 中恢复正常 death / timeout

这个 baseline 更接近 exploration upper bound。

实现位置：

- [`risky_maze/env/pomdp_episode.py`](./risky_maze/env/pomdp_episode.py)

## 10. teach / eval 实验流程

当前主实验流程是：

1. 载入固定地图 spec
2. 运行 teach tasks
3. teach 阶段允许 tutor 干预
4. learner 保留 teach 中形成的 memory / risk belief
5. 运行 `eval_same_map_no_tutor`
6. eval 阶段 tutor 强制关闭
7. 记录 teach / eval 各类指标

这保证了一个关键实验原则：

> eval 测的是 learner 学到了什么，
> 不是 tutor 在 eval 时又帮了多少。

主 runner 入口：

- [`risky_maze/runner/fixed_episode_runner.py`](./risky_maze/runner/fixed_episode_runner.py)
- [`risky_maze/runner/fixed_block_runner.py`](./risky_maze/runner/fixed_block_runner.py)
- [`risky_maze/experiments/run_fixed_maze.py`](./risky_maze/experiments/run_fixed_maze.py)

## 11. 核心指标

### 11.1 成功 / 安全指标

- `teach_success_rate`
- `eval_success_rate`
- `teach_death_rate`
- `eval_death_rate`
- `teach_timeout_rate`
- `eval_timeout_rate`
- `teach_mean_damage`
- `eval_mean_damage`

### 11.2 地图记忆指标

- `map_coverage_teach`
- `map_reuse_eval`
- `useful_exploration_rate`

其中当前实现里：

```text
teach_new_cells = teach 中首次发现的 walkable cells
eval_used_cells = eval 中实际走过的 cells

useful_exploration_rate =
  |teach_new_cells ∩ eval_used_cells| / max(1, |teach_new_cells|)

map_reuse_eval =
  |eval_used_cells ∩ teach_new_cells| / max(1, |eval_used_cells|)
```

### 11.3 risk 学习指标

- `risk_auc`
- `risk_nll`
- `risk_calibration_ece`
- `warning_information_gain`
- `posterior_shift_after_warning`

### 11.4 行为质量指标

- `eval_regret_to_oracle_safe_path`
- `loop_rate`
- `repeated_known_cell_rate`
- `no_info_step_rate`
- `frontier_progress_rate`

### 11.5 tutor 指标

- `warnings`
- `waypoints`
- `assist_leakage`
- `tutor_decisions.csv` 中的 counterfactual diagnostics

实现位置：

- [`risky_maze/runner/fixed_metrics.py`](./risky_maze/runner/fixed_metrics.py)

## 12. 目录结构

当前建议把这个仓库理解成下面几层。

```text
maze/
|-- risky_maze/
|   |-- core/          # 通用类型、pathing、共享基础件
|   |-- env/           # random runtime + fixed POMDP runtime + objectives + loaders
|   |-- learner/       # map memory, risk belief, planners
|   |-- tutor/         # warning baselines + inverse planning overlay
|   |-- runner/        # episode/block runner, metrics
|   |-- scenarios/     # fixed map specs and validators
|   |-- experiments/   # experiment CLIs
|   |-- config.py
|   `-- demo.py
|-- docs/
|   |-- README.md
|   |-- CODEBASE_STRUCTURE.md
|   |-- DOCUMENTATION_STANDARD.md
|   |-- RUN_MAZE_EXPERIMENTS.md
|   |-- architecture/
|   |-- specs/
|   |-- adr/
|   `-- notes/
|-- tests/
|-- risky_maze_pomdp_runtime_patch/
|-- tutor_inverse_planning_overlay/
`-- runs/
```

### 12.1 `risky_maze_pomdp_runtime_patch/`

这是 fixed-map POMDP runtime 的补丁来源目录，保留作为参考。
主实现已经合并到 `risky_maze/` 主包。

### 12.2 `tutor_inverse_planning_overlay/`

这是 inverse planning tutor overlay 的补丁来源目录，同样保留作为参考。
主实现也已经合并到 `risky_maze/tutor/`。

## 13. 从代码入口如何理解项目

如果你第一次读代码，推荐顺序如下。

1. 看固定地图 spec
   - [`docs/specs/HUGE_RISKY_GEM_MAZE_V0.md`](./docs/specs/HUGE_RISKY_GEM_MAZE_V0.md)
2. 看 loader 和 runtime
   - [`risky_maze/env/fixed_loader.py`](./risky_maze/env/fixed_loader.py)
   - [`risky_maze/env/pomdp_episode.py`](./risky_maze/env/pomdp_episode.py)
3. 看 learner
   - [`risky_maze/learner/objective_agent.py`](./risky_maze/learner/objective_agent.py)
4. 看 tutor baselines 与 inverse planner
   - [`risky_maze/tutor/warning_policies.py`](./risky_maze/tutor/warning_policies.py)
   - [`risky_maze/tutor/inverse_planner.py`](./risky_maze/tutor/inverse_planner.py)
5. 看 runner
   - [`risky_maze/runner/fixed_episode_runner.py`](./risky_maze/runner/fixed_episode_runner.py)
   - [`risky_maze/runner/fixed_block_runner.py`](./risky_maze/runner/fixed_block_runner.py)
6. 最后看 experiments CLI
   - [`risky_maze/experiments/run_fixed_maze.py`](./risky_maze/experiments/run_fixed_maze.py)
   - [`risky_maze/experiments/run_phase12_batch.py`](./risky_maze/experiments/run_phase12_batch.py)

## 14. 如何运行

### 14.1 跑测试

```powershell
python -m unittest discover -s tests -v
```

### 14.2 跑一个 fixed-map smoke

```powershell
python -m risky_maze.experiments.run_fixed_maze `
  --spec HugeRiskyGemMaze_v0 `
  --tutor no_tutor `
  --baseline mortal `
  --seeds 0 1 `
  --workers 2 `
  --teach-task-ids T01_NW_key_gem_NE_exit `
  --eval-task-ids E01_WestGarden_to_NE_exit `
  --out runs\smoke\fixed_no_tutor_mortal
```

### 14.3 跑 warning-only inverse tutor smoke

```powershell
python -m risky_maze.experiments.run_fixed_maze `
  --spec HugeRiskyGemMaze_v0 `
  --tutor inverse_plan_warn_only `
  --baseline mortal `
  --seeds 0 `
  --workers 1 `
  --tutor-rollout-horizon 3 `
  --tutor-top-k-paths 1 `
  --tutor-max-candidates 4 `
  --tutor-profile-count 2 `
  --out runs\smoke\inverse_warn_only
```

### 14.4 跑 Phase 1/2 批处理

```powershell
python -m risky_maze.experiments.run_phase12_batch `
  --spec HugeRiskyGemMaze_v0 `
  --workers 16 `
  --out runs\phase12_batch
```

更完整的命令说明见：

- [`docs/RUN_MAZE_EXPERIMENTS.md`](./docs/RUN_MAZE_EXPERIMENTS.md)

## 15. 实验输出结构

固定地图实验默认输出：

```text
runs/<name>/
|-- config.json
|-- summary.csv
|-- seed_summary.csv
|-- episodes.csv
|-- steps.csv
|-- tutor_decisions.csv
|-- risk_eval.csv
|-- map_reuse.csv
|-- objective_progress.csv
|-- trajectories/
`-- README.md
```

其中：

- `summary.csv`
  给出条件级别的聚合结果
- `episodes.csv`
  给出每个 task / phase 的 episode 结果
- `steps.csv`
  给出逐步行为记录
- `tutor_decisions.csv`
  给出 tutor 的 counterfactual diagnostics

## 16. 当前实验状态

截至目前，固定地图主线已经能稳定跑通一组 Phase 1/2 对照。

最近一轮批处理结果位于：

- [`runs/phase12_batch/phase12_summary.csv`](./runs/phase12_batch/phase12_summary.csv)

从这组结果看，至少有几个事实已经比较清楚：

- fixed-map runtime、task suite、runner、logging 链路已经接通
- `no_tutor_mortal` 在当前 teach task 上仍然会明显失败
- `always_warn` 和当前 heuristic `inverse_warn` 在这组任务上表现接近
- `risk_threshold_warn` 目前表现偏弱
- `inverse_plan_warn_only` 已经接线，但策略质量还需要继续 debug

这些结果说明：

> 当前项目已经从“概念设计”进入“可做条件对照实验”的阶段，
> 但真正有研究价值的部分，接下来在于调通 inverse planning tutor，
> 并把 map memory、risk learning、warning value、waypoint leakage 明确拆开分析。

## 17. 代码与文档索引

### 项目入口

- [`README.md`](./README.md)
- [`docs/README.md`](./docs/README.md)

### 结构规范

- [`docs/CODEBASE_STRUCTURE.md`](./docs/CODEBASE_STRUCTURE.md)
- [`docs/DOCUMENTATION_STANDARD.md`](./docs/DOCUMENTATION_STANDARD.md)

### 实验说明

- [`docs/RUN_MAZE_EXPERIMENTS.md`](./docs/RUN_MAZE_EXPERIMENTS.md)

### 场景与规格

- [`docs/specs/HUGE_RISKY_GEM_MAZE_V0.md`](./docs/specs/HUGE_RISKY_GEM_MAZE_V0.md)
- [`docs/architecture/SCENARIO_OVERVIEW.md`](./docs/architecture/SCENARIO_OVERVIEW.md)

## 18. 已知限制

当前版本仍有这些边界：

- fixed-map 主线的 risk belief 还是 binary safe-vs-danger，尚未 fully exploit 多 trap subtype
- current warning update 是 literal set conditioning，不是完整 RSA speaker-listener 推理
- inverse planner 是短 horizon 近似，不是 full nested solver
- `WAYPOINT` 已接入框架，但“多强算过强”仍需实验验证
- same-map eval 已经主线可用，但 cross-map risk generalization 还需要继续接上

## 19. 最后一句话概括这个项目

这个仓库不是为了做一个“会走迷宫的 agent”而存在的。

它更像一个研究平台，用来测试下面这个命题：

> 在部分可观测、有风险概念学习的任务里，
> 一个好的 tutor 应该知道何时放手让 learner 学，
> 何时警告，
> 何时最小化地指路，
> 以及这些帮助究竟有没有转化成真正可复用的能力。
