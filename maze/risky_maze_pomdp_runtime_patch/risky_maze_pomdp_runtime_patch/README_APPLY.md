# RiskyGemMaze POMDP runtime patch

This patch bundle adds a fixed-map POMDP runtime without deleting the existing random-maze prototype.

## Files added

```text
risky_maze/env/objectives.py
risky_maze/env/fixed_loader.py
risky_maze/env/pomdp_episode.py
risky_maze/learner/objective_agent.py
risky_maze/runner/fixed_metrics.py
risky_maze/runner/fixed_episode_runner.py
risky_maze/runner/fixed_block_runner.py
tests/test_fixed_runtime_smoke.py
```

## Apply

From the repository root:

```bash
cp -r risky_maze_pomdp_runtime_patch/risky_maze/* risky_maze/
cp -r risky_maze_pomdp_runtime_patch/tests/* tests/
python -m unittest discover -s tests -v
```

If the project uses `pytest`, this also works:

```bash
pytest -q
```

## Smoke usage

```python
from risky_maze.runner.fixed_block_runner import run_fixed_block

result = run_fixed_block(
    spec_name="HugeRiskyGemMaze_v0",
    tutor_name="no_tutor",
    baseline_mode="mortal",
    seed=0,
)

print(result["aggregate"]["useful_exploration_rate"])
print(result["aggregate"]["map_reuse_eval"])
```

## Notes

- Observations expose `wall / walkable / key / door / gem / exit` plus noisy risk vectors, but never expose true `r/m/q` labels.
- `D` is implemented as a pass/bottleneck objective, not a locked door.
- `eval_same_map` is forced tutor-off inside `run_fixed_block`.
- `immortal_warnlike` and `immortal_no_timeout` are supported in the environment/runner layer.
- Metrics include `useful_exploration_rate`, `map_reuse_eval`, risk AUC/NLL/ECE, warning posterior-shift proxies, oracle safe-path regret, and loop/no-info/frontier progress rates.

Because this bundle was produced from the implementation report rather than the full repository tree, you may need minor import-name adjustments if the existing project uses different class/module names.
