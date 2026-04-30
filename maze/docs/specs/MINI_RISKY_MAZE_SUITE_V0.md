# MiniRiskyMazeSuite_v0

这份文档把三个小固定图正式接入仓库，作为 `HugeRiskyGemMaze_v0` 的快速诊断补集。

代码入口：

- [suite Python spec](../../risky_maze/scenarios/mini_risky_maze_suite_v0.py)
- [MiniRiskGate JSON](../../risky_maze/scenarios/mini_risk_gate_v0.json)
- [MiniExploreLoop JSON](../../risky_maze/scenarios/mini_explore_loop_v0.json)
- [MiniWaypointBottleneck JSON](../../risky_maze/scenarios/mini_waypoint_bottleneck_v0.json)
- [validator](../../risky_maze/scenarios/validation.py)

## 总体定位

- 原大图 `HugeRiskyGemMaze_v0` 尺寸为 `61 x 39 = 2379` cells。
- 这里每张 mini map 为 `31 x 19 = 589` cells，大约是大图面积的 `24.8%`。
- 目的不是替代大图，而是让我们更快地定位三类核心机制：
  - `WARNING` 是否真的阻止明显灾难。
  - `WAIT` 是否真的允许有价值探索，而不是单纯不作为。
  - `WAYPOINT` 是否过强，是否退化成 oracle navigation。

共享设定：

- legend、任务格式、`K / D / g / E` objective 结构延续大图。
- `r / m / q` 仍然只是 oracle latent type。
- learner 不应直接看到 `r / m / q`，而是看到 noisy vector。
- risk vector 学习仍然应兼容 Gaussian / moment-matching 的连续概念学习框架。

## 1. MiniRiskGate_v0

用途：`warning / safety gate` 诊断图。

这个图故意把一条较短但危险的中部 corridor 和一条较长但安全的绕路并列放出来。它主要回答：

- `inverse_plan_warn_only` 到底会不会在“很明显该提醒”的地方发 warning。
- `always_warn` 是否确实降低 teach death。
- `inverse_plan_full` 是否会在本来只需要 warning 的地方过度使用 waypoint。

如果这张图上 `inverse_plan_warn_only` 仍然长期 `WAIT`，那优先怀疑：

- safety shield 没真正进入决策。
- warning candidate 没覆盖 learner 当前 path prefix。
- warning posterior update 没让 shadow learner 重新规划。
- rollout 里的 `Q(wait)` / `Q(warning)` 量纲仍然失真。

建议主比较：

- `no_tutor_mortal`
- `always_warn`
- `heuristic_inverse_warn`
- `inverse_plan_warn_only`
- `inverse_plan_full`

建议重点指标：

- `TeachDeath`
- `PreventableDeathRate`
- `WarningActionability`
- `post_warning_action_changed`
- `WarningsPerEpisode`
- `AssistLeakage`

## 2. MiniExploreLoop_v0

用途：`map memory / useful exploration` 诊断图。

这张图的 west loop、central loop、east room 是故意做成可复用结构的。它主要回答：

- tutor 的 `WAIT` 是不是 pedagogical waiting。
- immortal exploration 产生的地图信息是否能在 eval 里复用。
- `always_warn` 是否因为过早干预压制了有价值探索。

如果这张图上：

- `immortal_no_timeout` 的 `MapCoverageTeach` 明显更高，
- 并且 `MapReuseEval` / `UsefulExplorationRate` 也更高，

那说明“探索确实能转成能力”。

如果 `inverse_plan_full` 相比 `always_warn` 没有更高的 useful exploration，那么说明 tutor 还没有真正学会放手。

建议主比较：

- `no_tutor_mortal`
- `no_tutor_immortal_warnlike`
- `no_tutor_immortal_no_timeout`
- `always_warn`
- `inverse_plan_warn_only`
- `inverse_plan_full`

建议重点指标：

- `MapCoverageTeach`
- `MapReuseEval`
- `UsefulExplorationRate`
- `UsefulWaitRate`
- `BadWaitRate`
- `EvalRegretToOracleSafePath`
- `LoopRate`
- `NoInfoStepRate`

## 3. MiniWaypointBottleneck_v0

用途：`waypoint / assist leakage` 诊断图。

这张图有 central bottleneck、上下两个子区、dead-end 和 lure 区域。它主要回答：

- `WAYPOINT` 是否真的只是减少 loop / timeout。
- `WAYPOINT` 是否已经泄露过多目标进度。
- `frontier_only_waypoint` 和 budget / cooldown 是否能压住 over-help。

如果 `always_waypoint` 在这里 teach 和 eval 都全面占优，就要非常警惕：我们可能已经在做导航器，而不是 tutor。

建议主比较：

- `inverse_plan_warn_only`
- `waypoint_only`
- `always_waypoint`
- `inverse_plan_full`
- `inverse_plan_full_frontier_only`

建议重点指标：

- `WaypointsPerEpisode`
- `AssistLeakage`
- `WaypointProgressGift`
- `WaypointNoveltyLeak`
- `MapGainAfterWaypoint`
- `RiskIGAfterWaypoint`
- `EvalRegret`

## 推荐实验顺序

### Stage A: smoke

先确认三个图都能被 loader / runtime 调起来：

```cmd
python -m risky_maze.experiments.run_fixed_maze ^
  --spec MiniRiskGate_v0 ^
  --tutor no_tutor ^
  --baseline mortal ^
  --seeds 0 ^
  --out runs\mini_smoke\mini_risk_gate
```

同理再测 `MiniExploreLoop_v0` 和 `MiniWaypointBottleneck_v0`。

### Stage B: warning-only debug

先用 `MiniRiskGate_v0` 确认 `inverse_plan_warn_only` 不再退化成全 `WAIT`。

### Stage C: exploration / memory

再用 `MiniExploreLoop_v0` 看 `WAIT` 是否真的允许 useful exploration。

### Stage D: waypoint bandwidth

最后用 `MiniWaypointBottleneck_v0` 压测 `WAYPOINT` 是否 over-help。

## 为什么要保留这三个小图

不建议只做一个 mini map。因为如果只有一张小图，我们很容易把三种完全不同的问题混在一起：

- safety gate
- exploration value
- waypoint leakage

这三个图的价值就在于：它们让我们可以在保持相同 runtime / learner / tutor 框架的前提下，快速切换“哪一类机制是今天要调试的核心对象”。

## 当前 validator 结果

当前三个 mini spec 都已经通过 validator，没有 error。

当前有两个需要明确记录的 warning：

- `MiniExploreLoop_v0 / B_E01_reverse_east_to_west_gem` 的 start `(28, 16)` 位于 `E` tile 上。
- `MiniWaypointBottleneck_v0 / C_E03_central_to_east_gem_exit` 的 start `(15, 9)` 位于 `D` tile 上。

这两条现在不视为错误，因为它们可以用于测试“从交互节点出发”的路径重组行为；但如果后续 runtime 改成“站上交互 tile 立即自动触发”，就需要重新审查这两个 task 的起点设计。
