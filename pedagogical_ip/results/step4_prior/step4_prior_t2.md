# Step 4: Prior Refactor Results (Θ2)

**Seeds**: 30 | **Steps**: 10 | **Elapsed**: 7.2s

## Headline Metrics

| Variant | Scenario | NLL | SM_Acc | GoalAcc | H | FCI |
|---------|----------|-----|--------|---------|---|-----|
| legacy_bonus | goal_aligned | 0.296 | 0.534 | 0.067 | 2.405 | 0.468 |
| legacy_bonus | goal_conflict | 0.555 | 0.423 | 0.000 | 2.362 | 0.396 |
| legacy_bonus | temptation_hard | 0.656 | 0.462 | 0.000 | 1.876 | 0.177 |
| legacy_bonus | shortcut | 0.730 | 0.385 | 0.267 | 2.012 | 0.335 |
| no_bonus | goal_aligned | 0.296 | 0.534 | 0.067 | 2.405 | 0.468 |
| no_bonus | goal_conflict | 0.555 | 0.423 | 0.000 | 2.362 | 0.396 |
| no_bonus | temptation_hard | 0.656 | 0.462 | 0.000 | 1.876 | 0.177 |
| no_bonus | shortcut | 0.730 | 0.385 | 0.267 | 2.012 | 0.335 |
| structural | goal_aligned | 0.295 | 0.414 | 0.067 | 2.290 | 0.237 |
| structural | goal_conflict | 0.557 | 0.336 | 0.000 | 2.246 | 0.209 |
| structural | temptation_hard | 0.648 | 0.362 | 0.000 | 1.749 | 0.105 |
| structural | shortcut | 0.731 | 0.299 | 0.000 | 1.909 | 0.181 |
| pcfg | goal_aligned | 0.290 | 0.343 | 0.067 | 2.043 | 0.079 |
| pcfg | goal_conflict | 0.558 | 0.261 | 0.000 | 1.963 | 0.074 |
| pcfg | temptation_hard | 0.640 | 0.318 | 0.000 | 1.547 | 0.035 |
| pcfg | shortcut | 0.733 | 0.233 | 0.000 | 1.673 | 0.078 |

## Promotion Analysis

### goal_aligned

- NLL: 0.295 vs 0.296 BETTER
- SM_Acc: 0.414 vs 0.534 WORSE

### goal_conflict

- NLL: 0.557 vs 0.555 WORSE
- SM_Acc: 0.336 vs 0.423 WORSE

### temptation_hard

- NLL: 0.648 vs 0.656 BETTER
- SM_Acc: 0.362 vs 0.462 WORSE

### shortcut

- NLL: 0.731 vs 0.730 WORSE
- SM_Acc: 0.299 vs 0.385 WORSE

