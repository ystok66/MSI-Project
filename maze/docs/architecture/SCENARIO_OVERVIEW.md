# Scenario Overview

`risky_maze` 当前是一个最小可运行版本，用来验证：

- 地图探索是否能在 tutor-free eval 中复用
- risk vector 是否能跨地图泛化
- tutor 是否需要在“允许探索”和“阻止高风险前缀”之间做权衡

## 当前最小机制

- 环境：partial-observation maze
- 目标：`start -> gem -> exit`
- 风险：cell 有 hidden trap type 和 noisy vector
- learner：map memory + Gaussian risk belief + risk-aware A*
- tutor：`WAIT / WARNING(path-prefix)`
- eval：
  - `eval_same_map`
  - `eval_new_map`

## 当前不做的事情

- full POMDP solver
- nested belief inference
- `WAYPOINT`
- richer object system
- key / door multi-stage puzzle

## 代码落点

- env: [risky_maze/env](../../risky_maze/env)
- learner: [risky_maze/learner](../../risky_maze/learner)
- tutor: [risky_maze/tutor](../../risky_maze/tutor)
- runner: [risky_maze/runner](../../risky_maze/runner)

## 后续最自然的扩展

1. 加 `WAYPOINT`
2. 加更细的 eval metrics
3. 把 heuristic inverse warning 升级成短 horizon rollout tutor
4. 再考虑 gem-door-key 的更复杂任务结构
