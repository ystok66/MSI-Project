# Current Progress And Experiment Report

日期：2026-04-20

这份报告基于当前仓库里的真实代码、测试结果和最近一次 fixed-map 实验输出整理，目标是明确四件事：

- 现在到底已经实现了什么
- 最近实验实际说明了什么
- 哪些内容只是临时近似或调试性实现
- 哪些关键研究能力仍然没有完成

## 1. 结论先行

截至当前版本，`maze/` 已经不再是“只有场景想法”的阶段，而是一个可以运行固定大图 teach/eval 对照实验的研究原型。

当前最重要的判断是：

- fixed-map POMDP runtime 已经接入主线代码，不再只是 spec + validator。
- `HugeRiskyGemMaze_v0` 已经可以被 loader、runtime、runner 和 experiment CLI 真正调用。
- Phase 1 核心 baselines 已经接上：
  - `no_tutor_mortal`
  - `no_tutor_immortal_warnlike`
  - `no_tutor_immortal_no_timeout`
- Phase 2 warning baselines 已经接上：
  - `always_warn`
  - `risk_threshold_warn`
  - `inverse_warn`
  - `inverse_plan_warn_only`
- Phase 3 的 `inverse_plan_full` 和 `WAYPOINT` 框架已经接上，但还只是 smoke / pilot 水平，远不能算稳定结论。
- 当前最明确的主问题，已经不是“fixed runtime 有没有接上”，而是：
  - `inverse_plan_warn_only` 为什么基本退化成一直 `WAIT`
  - `WAYPOINT` 在 full tutor 里是否过强
  - 当前 same-map 实验里看到的收益，到底来自 map memory，还是来自 risk concept learning

一句话概括：

```text
当前已经完成“可运行实验系统”的主体骨架，
但还没有完成“可支撑最终研究结论”的稳定实验体系。
```

## 2. 本报告依据

这份报告主要依据以下证据源：

- 代码目录 `risky_maze/`
- 测试目录 `tests/`
- 固定地图 spec：
  - `risky_maze/scenarios/huge_risky_gem_maze_v0.py`
  - `risky_maze/scenarios/huge_risky_gem_maze_v0.json`
- 最近实验输出：
  - `runs/phase12_batch/`
  - `runs/smoke/fixed_no_tutor_mortal/`
  - `runs/smoke/inverse_plan_warn_only/`
  - `runs/smoke/inverse_plan_full/`

本次重新验证过的自动化测试命令：

```powershell
python -m unittest discover -s tests -v
```

当前结果：

- `16/16` tests pass

## 3. 当前代码进度

### 3.1 已经完成并接入主线的部分

#### A. 固定地图 POMDP runtime

以下模块已经合并进主线，且不是补丁目录里的“待应用文件”状态：

- `risky_maze/env/objectives.py`
- `risky_maze/env/fixed_loader.py`
- `risky_maze/env/pomdp_episode.py`

它们当前已经支持：

- fixed spec loading
- `(x, y)` 外部坐标到 runtime `(row, col)` 的转换
- objective sequence
- local partial observation
- hidden trap labels + learner-visible risk vectors
- baseline mode 切换
- teach / eval 阶段差异化处理

这意味着当前环境不再只是早期的：

```text
start -> one gem -> one exit
```

而是已经支持 fixed-map task suite 所需的：

- `pickup`
- `pass`
- `collect_gem`
- `exit`

#### B. 固定地图 learner

以下模块已经接入主线：

- `risky_maze/learner/objective_agent.py`

它当前包含：

- `SimpleMapMemory`
- `OnlineGaussianRiskBelief`
- `ObjectiveAwareLearner`

目前 learner 已经具备：

- 局部地图记忆
- 风险原型在线更新
- warning set Bayesian-style update
- objective-aware A* 规划
- temporary waypoint 支持

当前 planner 的 cell cost 已经是多因素的：

```text
1
+ risk_weight * P(danger | x)
+ revisit_penalty * visits
+ unknown_penalty * unknown
+ warning_suspicion_weight * suspicion
```

所以这部分已经不是“空的接口”，而是真正在跑的 learner。

#### C. 固定地图 runner 与 metrics

以下模块已接入主线：

- `risky_maze/runner/fixed_metrics.py`
- `risky_maze/runner/fixed_episode_runner.py`
- `risky_maze/runner/fixed_block_runner.py`

当前已经支持：

- teach tasks 顺序运行
- eval_same_map_no_tutor 顺序运行
- per-step logging
- per-episode logging
- per-condition summary 输出

这意味着 fixed-map 实验已经不是只有一份 spec，而是已经有完整数据流：

```text
spec -> loader -> env -> learner/tutor -> runner -> csv outputs
```

#### D. 实验脚本

以下实验入口已接入主线：

- `risky_maze/experiments/run_fixed_maze.py`
- `risky_maze/experiments/run_phase12_batch.py`

当前已经能自动输出：

- `config.json`
- `summary.csv`
- `seed_summary.csv`
- `episodes.csv`
- `steps.csv`
- `tutor_decisions.csv`
- `risk_eval.csv`
- `map_reuse.csv`
- `objective_progress.csv`
- `trajectories/`

#### E. inverse planning overlay

以下模块已经接入主线：

- `risky_maze/tutor/profiles.py`
- `risky_maze/tutor/shadow.py`
- `risky_maze/tutor/path_predictor.py`
- `risky_maze/tutor/candidates.py`
- `risky_maze/tutor/rollout.py`
- `risky_maze/tutor/diagnostics.py`
- `risky_maze/tutor/inverse_planner.py`
- `risky_maze/tutor/baselines.py`
- `risky_maze/tutor/factory.py`

这意味着当前 tutor 不再只有旧版 heuristic warning tutor。
仓库里现在确实已经有一个“有限 learner profiles + short rollout”的 inverse tutor 版本。

### 3.2 当前已经具备但仍然偏早期的能力

这些功能现在已经存在并可运行，但还不能视为“稳定研究实现”。

#### A. `inverse_plan_warn_only`

它已经接入，也能产出 `tutor_decisions.csv`。

但从当前 pilot 结果看，它在主要试验配置上基本退化为：

```text
一直 WAIT
```

所以它目前更像：

- 已接线的真实算法骨架
- 但尚未调通的策略

而不是可用的主实验方法。

#### B. `inverse_plan_full`

它已经能在 smoke run 里发出 `WAYPOINT`，并完成任务。

但当前它还只是：

- 1 个 seed
- 1 组任务对
- 明显较高 assist leakage

因此它目前只能说明：

```text
WAYPOINT 机制已经接上
```

还不能说明：

```text
WAYPOINT 的研究行为已经合理
```

#### C. risk metrics

当前已经有：

- `risk_auc`
- `risk_nll`
- `risk_calibration_ece`

但目前这些值在现有批次里非常“漂亮”，很多条件下接近完美。
这提示我们它们虽然已经接上了，但还需要继续确认：

- 评估数据是否过于容易
- 当前 binary danger setup 是否让这个 probe 过乐观
- 是否需要更难的 out-of-distribution risk probe

所以这些指标当前是：

- 已实现
- 但解释时仍要谨慎

### 3.3 当前已经过时的旧判断

仓库里原有的旧版 `Current Implementation Report` 曾经判断很多内容“还没实现”，但那份判断现在已经过时。

现在已经过时的旧说法包括：

- “fixed-map runtime 还没接上”
- “baseline modes 还没实现”
- “inverse overlay 还只是补丁”
- “warning / waypoint / rollout 还没有进主线代码”

这些在当前代码状态下都已经不准确了。

## 4. 当前测试状态

本次重新跑过的测试结果：

- `16/16` pass

当前测试实际覆盖到的内容包括：

### 4.1 fixed spec 与 validator

- `HugeRiskyGemMaze_v0` 没有 validation error
- summary counts 与预期一致
- 能检测到 `E04` 起点落在 `g` tile 上这个 warning

### 4.2 fixed runtime

- 所有 fixed tasks 都能 reset
- 所有 fixed tasks 在 `no_tutor` 下能运行
- 所有 fixed tasks 在 `immortal_warnlike` 下能运行
- 所有 fixed tasks 在 `immortal_no_timeout` 下能运行
- objective sequence 会按坐标推进
- learner observation 不暴露 true trap symbols

### 4.3 tutor overlay

- eval 阶段 tutor 会被禁用
- `risk_threshold_warn` baseline 至少会触发警告
- warning-only inverse tutor 在构造性的 deadly prefix 条件下会发出 warning

### 4.4 experiment runner

- `run_fixed_experiment()` 会生成完整输出文件

当前测试层面能支持的结论是：

```text
fixed-map 主链路是接通的
```

还不能支持的结论是：

```text
当前 inverse planning 行为已经合理
```

## 5. 当前实验整理

## 5.1 已跑过的实验

当前仓库里已经有三类可直接引用的实验输出。

### A. fixed-map smoke

- `runs/smoke/fixed_no_tutor_mortal/`
- `runs/smoke/inverse_plan_warn_only/`
- `runs/smoke/inverse_plan_full/`

### B. Phase 1/2 batch

- `runs/phase12_batch/`

当前这批次的配置是：

- spec: `HugeRiskyGemMaze_v0`
- teach task: `T01_NW_key_gem_NE_exit`
- eval task: `E01_WestGarden_to_NE_exit`
- workers: `16`

条件如下：

- `no_tutor_mortal`
- `no_tutor_immortal_warnlike`
- `no_tutor_immortal_no_timeout`
- `always_warn_mortal`
- `risk_threshold_warn_mortal`
- `inverse_warn_mortal`
- `inverse_plan_warn_only_pilot`

其中 `inverse_plan_warn_only_pilot` 只跑了 `2` 个 seeds，而且 tutor rollout 参数被降到了调试配置：

- `tutor_rollout_horizon = 3`
- `tutor_top_k_paths = 1`
- `tutor_max_candidates = 4`
- `tutor_profile_count = 2`

所以这一项必须视为：

```text
pilot debug run
```

不能和其他 8-seed 条件等价比较。

## 5.2 Phase 1/2 batch 关键结果

下面先列最核心结果。

| condition | seeds | teach_success | teach_death | eval_success | warnings | map_coverage_teach | map_reuse_eval | useful_exploration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_tutor_mortal` | 8 | 0.000 | 1.000 | 1.000 | 0.0 | 0.162 | 0.203 | 0.106 |
| `no_tutor_immortal_warnlike` | 8 | 1.000 | 0.000 | 1.000 | 0.0 | 0.240 | 0.356 | 0.114 |
| `no_tutor_immortal_no_timeout` | 8 | 1.000 | 0.000 | 1.000 | 0.0 | 0.240 | 0.356 | 0.114 |
| `always_warn_mortal` | 8 | 1.000 | 0.000 | 1.000 | 6.5 | 0.257 | 0.310 | 0.095 |
| `risk_threshold_warn_mortal` | 8 | 0.125 | 0.875 | 1.000 | 9.5 | 0.172 | 0.171 | 0.084 |
| `inverse_warn_mortal` | 8 | 1.000 | 0.000 | 1.000 | 6.5 | 0.257 | 0.310 | 0.095 |
| `inverse_plan_warn_only_pilot` | 2 | 0.000 | 1.000 | 1.000 | 0.0 | 0.163 | 0.180 | 0.087 |

## 5.3 对这些结果的解释

### A. `no_tutor_mortal`

这是当前最重要的基线之一。

结果显示：

- teach 成功率 `0.0`
- teach 死亡率 `1.0`
- eval 成功率 `1.0`

这说明在当前这组 teach task 上：

- 没有 tutor 且允许死亡时，learner 经常在 teach 里直接失败
- 这正说明 tutor 的 safety 价值空间是真实存在的

但也要注意：

- eval task `E01` 对当前 learner 来说并不难
- 所以目前这个批次更像是在测 teach safety，不足以说明“泛化是否更好”

### B. `no_tutor_immortal_warnlike` 与 `no_tutor_immortal_no_timeout`

这两个条件当前结果几乎一样：

- teach success `1.0`
- eval success `1.0`
- map coverage 比 `no_tutor_mortal` 高很多
- map reuse / useful exploration 也更高

这说明至少在当前任务切片上：

```text
如果移除 teach 中的灾难终止成本，
learner 确实能探索更多，并把一部分探索转成后续可复用信息。
```

这是当前最接近你研究命题核心的一条正面信号。

但它仍然只是 same-map 单任务对结果，不能过度外推。

### C. `always_warn_mortal`

结果：

- teach success `1.0`
- teach death `0.0`
- warnings `6.5`
- eval success `1.0`

这说明：

- warning 干预链路已经真实工作
- warning 的 timing 至少能在这组任务里有效降低灾难

### D. `inverse_warn_mortal`

结果与 `always_warn_mortal` 在这一批里几乎完全一致：

- teach success `1.0`
- teach death `0.0`
- warnings `6.5`
- map coverage / map reuse / useful exploration 基本相同

这意味着当前 heuristic `inverse_warn` 在这组任务切片里并没有展示出比 `always_warn` 更多的价值。

最合理的解释是：

- 这组任务太简单，heuristic 就足够
- 或者当前 heuristic inverse policy 实际上退化到了和 `always_warn` 近似相同的行为

### E. `risk_threshold_warn_mortal`

这是当前最差的 warning baseline。

结果：

- teach success 只有 `0.125`
- teach death 高达 `0.875`
- warnings 反而最多，为 `9.5`

这说明它不是“warning 太少”，而是：

- warning 的策略本身不合理
- 或者 warning timing / threshold 设计有问题

它当前更像一个：

```text
存在但效果很差的对照条件
```

### F. `inverse_plan_warn_only_pilot`

这是当前最需要 debug 的条件。

结果：

- teach success `0.0`
- teach death `1.0`
- warnings `0.0`

结合 `tutor_decisions.csv` 可以看到：

- 它几乎一直选择 `WAIT`
- `q_wait` 常年固定在 `-1.0`
- `q_best_warning` 通常更低
- 所以从当前 utility 设定看，warning 很难被选中

这不是“方法失败”的最终结论，而更像是：

```text
rollout utility / candidate scoring / guardrail 还没有调通
```

## 5.4 `inverse_plan_full` smoke 的当前意义

`runs/smoke/inverse_plan_full/summary.csv` 当前显示：

- `seed_count = 1`
- `teach_success_rate = 1.0`
- `eval_success_rate = 1.0`
- `teach_mean_waypoints = 8.0`
- `teach_assist_leakage = 4.4`

`tutor_decisions.csv` 里已经能看到：

- tutor 真实发出了多个 `WAYPOINT`
- 主要 reason 是 `nearest_known_frontier`

这说明：

- full tutor 的 `WAYPOINT` action schema 已经真实接线
- candidate generation 和 rollout selection 至少能走通

但当前这组结果只能说明：

```text
WAYPOINT 系统能跑
```

不能说明：

```text
WAYPOINT 已经合理
```

原因很简单：

- 只有 1 个 seed
- 只测了 `T07` teach 和 `E08` eval
- assist leakage 很高
- 还没有与 `always_waypoint` 做正式多 seed 对照

因此这块当前必须归类为：

```text
临时 smoke 能力
```

而不是稳定实验结论。

## 6. 当前哪些东西还没有完成

### 6.1 实验覆盖面还很窄

尽管 fixed runtime 已经接上，但目前真正跑过成规模对照的，只是：

- 1 个 teach task
- 1 个 eval task
- 1 组 same-map Phase 1/2 条件

还没有完成的关键实验包括：

- 全 task suite 批量跑
- `eval_new_map_same_risk`
- cross-map risk generalization probe
- waypoint 正式 batch
- map memory vs risk learning ablation

### 6.2 `inverse_plan_warn_only` 还没有可用

虽然框架已经有了，但目前从行为上看，它还不能算一个“工作中的方法”。

当前还没有完成的包括：

- utility 标度调优
- boredom / map_gain / risk_ig 的平衡
- candidate generation 的有效性验证
- WAIT 和 WARNING 之间的可分性调通

### 6.3 `WAYPOINT` 研究问题还没有真正展开

虽然 full tutor 已经会发 waypoint，但下面这些关键问题还没有系统回答：

- `always_waypoint` 是否过强
- `waypoint_only` 是否比 warning-only 更容易 over-help
- `inverse_plan_full` 是否只是“更会指路”，而不是“更会教学”
- assist leakage 应该如何规范化定义

### 6.4 风险概念泛化还没有被充分证明

当前很多 summary 里 `risk_auc = 1.0`，这并不能直接证明 risk learning 问题已经解决了。

更准确的说法是：

- 当前 risk probe 已经接上
- 但现有 probe 仍偏简单
- 还没有在更难的 new-map same-risk 设置下证明泛化

### 6.5 文档和工程整理还没完全收尾

虽然主 README 已经更新，但工程里仍然还有一些“阶段性痕迹”：

- `risky_maze_pomdp_runtime_patch/`
- `tutor_inverse_planning_overlay/`

这两个目录当前是：

- 作为补丁来源保留
- 不是主线运行入口

它们现在属于：

```text
过渡性工程遗留
```

后面可以考虑继续保留作审计参考，或者在确认不再需要后归档。

## 7. 当前哪些内容只是临时实现

下面这些内容现在“能跑”，但不应当被当成最终研究版本。

### A. heuristic `inverse_warn`

它当前更像：

```text
oracle-risk-aware heuristic baseline
```

而不是完整 inverse planning tutor。

它的价值主要是：

- 提供一个强于 `always_warn` 的潜在方向
- 作为 full inverse tutor 的过渡基线

### B. `inverse_plan_warn_only_pilot`

它当前是：

- 真正的 rollout tutor 骨架
- 但行为上仍是 pilot / debug 状态

### C. `inverse_plan_full` smoke

它当前是：

- true waypoint-capable tutor skeleton
- 但还没有经过正式条件对照验证

### D. 部分 utility 权重

当前 tutor rollout 的权重，例如：

- `success = 4.0`
- `death = 30.0`
- `damage = 5.0`
- `timeout = 8.0`
- `warning_cost = 0.2`
- `waypoint_cost = 0.8`

目前更接近：

```text
工程上合理的初始权重
```

而不是：

```text
经过系统 sweep 证明稳健的研究设置
```

### E. 当前批处理结论

当前 `runs/phase12_batch/` 的结果是有价值的，但它们仍然是：

- 单一 task-pair
- same-map
- 主要验证链路是否工作

因此当前更应该把它理解为：

```text
phase sanity check
```

而不是：

```text
最终论文级结果
```

## 8. 当前最重要的下一步

如果目标是尽快把项目推进到“可以认真解释结果”的阶段，优先级建议如下。

### 第一优先级

调通 `inverse_plan_warn_only`。

这是当前最大的主线 blocker，因为它直接关系到：

- inverse planning 是否真的优于 heuristic
- warning-only 阶段能否成立

### 第二优先级

扩展 fixed-map 实验覆盖面。

至少应继续跑：

- 更多 teach tasks
- 更多 eval tasks
- 多 seed full batch

### 第三优先级

正式做 Phase 3 对照：

- `always_waypoint`
- `waypoint_only`
- `inverse_plan_full`

### 第四优先级

把 map memory 与 risk learning 的贡献拆开。

否则即使结果变好，也很难回答：

```text
到底是学到了地图，还是学到了 risk concept
```

## 9. 当前总体完成度判断

如果按“工程骨架是否搭好”来算：

```text
大约 70% - 80%
```

因为主线环境、固定图、baselines、inverse overlay、metrics、runner、tests 都已经存在了。

如果按“是否已经能稳定支撑研究结论”来算：

```text
大约 45% - 55%
```

原因是：

- 关键 full-method 还没调通
- 实验覆盖面还窄
- Phase 3 还没有正式比较
- 风险泛化还没有被充分验证

最准确的一句话总结是：

```text
当前项目已经完成了“可运行研究原型”的主体建设，
但还没有完成“可稳定解释研究命题”的实验验证阶段。
```
