# Run Maze Experiments

固定大图实验入口：

```powershell
python -m risky_maze.experiments.run_fixed_maze `
  --spec HugeRiskyGemMaze_v0 `
  --tutor no_tutor `
  --baseline mortal `
  --seeds 0 1 2 `
  --workers 3 `
  --out runs\\smoke\\fixed_no_tutor_mortal
```

warning-only inverse tutor：

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
  --out runs\\smoke\\inverse_warn_only
```

full inverse tutor：

```powershell
python -m risky_maze.experiments.run_fixed_maze `
  --spec HugeRiskyGemMaze_v0 `
  --tutor inverse_plan_full `
  --baseline mortal `
  --seeds 0 `
  --workers 1 `
  --tutor-rollout-horizon 3 `
  --tutor-top-k-paths 1 `
  --tutor-max-candidates 4 `
  --tutor-profile-count 2 `
  --out runs\\smoke\\inverse_full
```

常用 tutor 名称别名：

- `inverse_plan_warn_only`
- `warning_only_inverse`
- `warning_only_safety_shield`
- `inverse_plan_warn_only_safety_shield`
- `inverse_plan_full`
- `full_inverse`
- `inverse_plan_full_frontier_only`

一些常用控制开关：

- `--tutor-safety-shield-enabled`
- `--tutor-frontier-only-waypoint`
- `--tutor-max-waypoints-per-episode 2`
- `--warning-update-mode effective_sample`
- `--ablate-eval-clear-map-memory`
- `--ablate-eval-clear-risk-belief`

输出目录结构：

```text
runs/<name>/
├─ config.json
├─ summary.csv
├─ seed_summary.csv
├─ episodes.csv
├─ steps.csv
├─ tutor_decisions.csv
├─ risk_eval.csv
├─ map_reuse.csv
├─ objective_progress.csv
├─ trajectories/
└─ README.md
```
