# Step 4: Prior Refactor Results (Θk)

**Seeds**: 30 | **Steps**: 10 | **Elapsed**: 14.5s

## Headline Metrics

| Variant | Scenario | NLL | SM_Acc | GoalAcc | H | FCI |
|---------|----------|-----|--------|---------|---|-----|
| legacy_bonus | goal_aligned | 0.277 | 0.592 | 0.000 | 3.366 | 0.530 |
| legacy_bonus | goal_conflict | 0.547 | 0.416 | 0.000 | 3.357 | 0.417 |
| legacy_bonus | temptation_hard | 0.663 | 0.381 | 0.000 | 2.630 | 0.363 |
| legacy_bonus | shortcut | 0.737 | 0.341 | 0.000 | 2.822 | 0.491 |
| no_bonus | goal_aligned | 0.277 | 0.592 | 0.000 | 3.366 | 0.530 |
| no_bonus | goal_conflict | 0.547 | 0.416 | 0.000 | 3.357 | 0.417 |
| no_bonus | temptation_hard | 0.663 | 0.381 | 0.000 | 2.630 | 0.363 |
| no_bonus | shortcut | 0.737 | 0.341 | 0.000 | 2.822 | 0.491 |
| structural | goal_aligned | 0.287 | 0.483 | 0.000 | 3.238 | 0.287 |
| structural | goal_conflict | 0.550 | 0.329 | 0.000 | 3.231 | 0.219 |
| structural | temptation_hard | 0.658 | 0.281 | 0.000 | 2.511 | 0.206 |
| structural | shortcut | 0.743 | 0.282 | 0.000 | 2.729 | 0.293 |
| pcfg | goal_aligned | 0.290 | 0.410 | 0.000 | 2.928 | 0.095 |
| pcfg | goal_conflict | 0.551 | 0.255 | 0.000 | 2.934 | 0.077 |
| pcfg | temptation_hard | 0.655 | 0.216 | 0.000 | 2.221 | 0.060 |
| pcfg | shortcut | 0.749 | 0.227 | 0.000 | 2.402 | 0.111 |

## Promotion Analysis

### goal_aligned

- NLL: 0.287 vs 0.277 WORSE
- SM_Acc: 0.483 vs 0.592 WORSE

### goal_conflict

- NLL: 0.550 vs 0.547 WORSE
- SM_Acc: 0.329 vs 0.416 WORSE

### temptation_hard

- NLL: 0.658 vs 0.663 BETTER
- SM_Acc: 0.281 vs 0.381 WORSE

### shortcut

- NLL: 0.743 vs 0.737 WORSE
- SM_Acc: 0.282 vs 0.341 WORSE

