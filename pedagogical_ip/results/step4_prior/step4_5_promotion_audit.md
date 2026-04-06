# Step 4.5: Promotion & Audit Report

**Seeds**: 30 | **Elapsed**: 95.3s

## Exp A: Longer-Horizon Recovery

| Variant | Scenario | t | SM_Acc | FCI | Entropy |
|---------|----------|---|--------|-----|--------|
| legacy_bonus | goal_aligned | 5 | 0.426 | 0.216 | 2.265 |
| legacy_bonus | goal_aligned | 10 | 0.362 | 0.089 | 2.035 |
| legacy_bonus | goal_aligned | 20 | 0.328 | 0.025 | 1.787 |
| legacy_bonus | goal_aligned | 30 | 0.331 | 0.023 | 1.718 |
| legacy_bonus | goal_aligned | 50 | 0.310 | 0.019 | 1.595 |
| legacy_bonus | goal_conflict | 5 | 0.337 | 0.214 | 2.323 |
| legacy_bonus | goal_conflict | 10 | 0.276 | 0.106 | 2.023 |
| legacy_bonus | goal_conflict | 20 | 0.219 | 0.035 | 1.639 |
| legacy_bonus | goal_conflict | 30 | 0.194 | 0.022 | 1.441 |
| legacy_bonus | goal_conflict | 50 | 0.156 | 0.020 | 1.293 |
| legacy_bonus | temptation_hard | 5 | 0.127 | 0.096 | 1.379 |
| legacy_bonus | temptation_hard | 10 | 0.056 | 0.032 | 0.863 |
| legacy_bonus | temptation_hard | 20 | 0.027 | 0.014 | 0.594 |
| legacy_bonus | temptation_hard | 30 | 0.056 | 0.020 | 0.689 |
| legacy_bonus | temptation_hard | 50 | 0.079 | 0.018 | 0.739 |
| legacy_bonus | shortcut | 5 | 0.290 | 0.196 | 2.054 |
| legacy_bonus | shortcut | 10 | 0.238 | 0.086 | 1.650 |
| legacy_bonus | shortcut | 20 | 0.175 | 0.025 | 0.965 |
| legacy_bonus | shortcut | 30 | 0.175 | 0.015 | 0.685 |
| legacy_bonus | shortcut | 50 | 0.200 | 0.013 | 0.566 |
| structural | goal_aligned | 5 | 0.414 | 0.227 | 2.289 |
| structural | goal_aligned | 10 | 0.414 | 0.237 | 2.290 |
| structural | goal_aligned | 20 | 0.415 | 0.255 | 2.267 |
| structural | goal_aligned | 30 | 0.419 | 0.272 | 2.266 |
| structural | goal_aligned | 50 | 0.423 | 0.313 | 2.266 |
| structural | goal_conflict | 5 | 0.332 | 0.201 | 2.341 |
| structural | goal_conflict | 10 | 0.336 | 0.209 | 2.246 |
| structural | goal_conflict | 20 | 0.342 | 0.226 | 2.142 |
| structural | goal_conflict | 30 | 0.347 | 0.243 | 2.083 |
| structural | goal_conflict | 50 | 0.365 | 0.294 | 2.050 |
| structural | temptation_hard | 5 | 0.322 | 0.140 | 1.941 |
| structural | temptation_hard | 10 | 0.362 | 0.105 | 1.749 |
| structural | temptation_hard | 20 | 0.446 | 0.093 | 1.622 |
| structural | temptation_hard | 30 | 0.494 | 0.089 | 1.575 |
| structural | temptation_hard | 50 | 0.540 | 0.045 | 1.385 |
| structural | shortcut | 5 | 0.289 | 0.203 | 2.119 |
| structural | shortcut | 10 | 0.299 | 0.181 | 1.909 |
| structural | shortcut | 20 | 0.340 | 0.128 | 1.758 |
| structural | shortcut | 30 | 0.365 | 0.121 | 1.671 |
| structural | shortcut | 50 | 0.415 | 0.120 | 1.644 |
| pcfg | goal_aligned | 5 | 0.342 | 0.063 | 2.018 |
| pcfg | goal_aligned | 10 | 0.343 | 0.079 | 2.043 |
| pcfg | goal_aligned | 20 | 0.345 | 0.105 | 2.057 |
| pcfg | goal_aligned | 30 | 0.354 | 0.134 | 2.089 |
| pcfg | goal_aligned | 50 | 0.361 | 0.183 | 2.131 |
| pcfg | goal_conflict | 5 | 0.256 | 0.059 | 2.020 |
| pcfg | goal_conflict | 10 | 0.261 | 0.074 | 1.963 |
| pcfg | goal_conflict | 20 | 0.265 | 0.102 | 1.912 |
| pcfg | goal_conflict | 30 | 0.269 | 0.126 | 1.885 |
| pcfg | goal_conflict | 50 | 0.274 | 0.189 | 1.898 |
| pcfg | temptation_hard | 5 | 0.269 | 0.038 | 1.687 |
| pcfg | temptation_hard | 10 | 0.318 | 0.035 | 1.547 |
| pcfg | temptation_hard | 20 | 0.404 | 0.047 | 1.459 |
| pcfg | temptation_hard | 30 | 0.460 | 0.062 | 1.448 |
| pcfg | temptation_hard | 50 | 0.508 | 0.040 | 1.288 |
| pcfg | shortcut | 5 | 0.231 | 0.068 | 1.808 |
| pcfg | shortcut | 10 | 0.233 | 0.078 | 1.673 |
| pcfg | shortcut | 20 | 0.267 | 0.079 | 1.567 |
| pcfg | shortcut | 30 | 0.291 | 0.076 | 1.508 |
| pcfg | shortcut | 50 | 0.341 | 0.085 | 1.500 |

## Exp B: Leave-One-Feature-Out Prior Ablation

| Config | Scenario | NLL | SM_Acc | FCI | Entropy |
|--------|----------|-----|--------|-----|--------|
| complexity_only | goal_aligned | 0.293 | 0.433 | 0.263 | 2.320 |
| complexity_only | goal_conflict | 0.556 | 0.341 | 0.232 | 2.270 |
| complexity_only | temptation_hard | 0.647 | 0.382 | 0.102 | 1.776 |
| complexity_only | shortcut | 0.730 | 0.305 | 0.208 | 1.937 |
| redundancy_only | goal_aligned | 0.298 | 0.503 | 0.432 | 2.393 |
| redundancy_only | goal_conflict | 0.556 | 0.418 | 0.362 | 2.353 |
| redundancy_only | temptation_hard | 0.659 | 0.425 | 0.187 | 1.877 |
| redundancy_only | shortcut | 0.732 | 0.378 | 0.291 | 2.004 |
| complexity+redund | goal_aligned | 0.295 | 0.414 | 0.237 | 2.290 |
| complexity+redund | goal_conflict | 0.557 | 0.336 | 0.209 | 2.246 |
| complexity+redund | temptation_hard | 0.648 | 0.362 | 0.105 | 1.749 |
| complexity+redund | shortcut | 0.731 | 0.299 | 0.181 | 1.909 |
| neither (uniform) | goal_aligned | 0.296 | 0.534 | 0.468 | 2.405 |
| neither (uniform) | goal_conflict | 0.555 | 0.423 | 0.396 | 2.362 |
| neither (uniform) | temptation_hard | 0.656 | 0.462 | 0.177 | 1.876 |
| neither (uniform) | shortcut | 0.730 | 0.385 | 0.335 | 2.012 |

## Exp C: Full CGC-v2 Integration Test

| Variant | Scenario | SM_Acc | FCI | Bridge_stable |
|---------|----------|--------|-----|---------------|
| legacy_bonus | goal_aligned | 0.362 | 0.089 | YES |
| legacy_bonus | goal_conflict | 0.276 | 0.106 | YES |
| legacy_bonus | temptation_hard | 0.056 | 0.032 | YES |
| legacy_bonus | shortcut | 0.238 | 0.086 | YES |
| structural | goal_aligned | 0.414 | 0.237 | YES |
| structural | goal_conflict | 0.336 | 0.209 | YES |
| structural | temptation_hard | 0.362 | 0.105 | YES |
| structural | shortcut | 0.299 | 0.181 | YES |

## Exp D: Θ₂ vs Θ_K Promotion Test (structural prior)

| Θ-mode | Scenario | NLL | SM_Acc | FCI | Entropy |
|--------|----------|-----|--------|-----|--------|
| Θ2 | goal_aligned | 0.295 | 0.414 | 0.237 | 2.290 |
| Θ2 | goal_conflict | 0.557 | 0.336 | 0.209 | 2.246 |
| Θ2 | temptation_hard | 0.648 | 0.362 | 0.105 | 1.749 |
| Θ2 | shortcut | 0.731 | 0.299 | 0.181 | 1.909 |
| Θk | goal_aligned | 0.287 | 0.483 | 0.287 | 3.238 |
| Θk | goal_conflict | 0.550 | 0.329 | 0.219 | 3.231 |
| Θk | temptation_hard | 0.658 | 0.281 | 0.206 | 2.511 |
| Θk | shortcut | 0.743 | 0.282 | 0.293 | 2.729 |

## Exp E: Subgoal Calibration Audit

### legacy_bonus

**Brier**: 0.3761 | **ECE**: 0.3256

| Bin | Avg Pred | Avg Correct | N |
|-----|----------|-------------|---|
| [0.0,0.1) | 0.042 | 0.672 | 131 |
| [0.1,0.2) | 0.157 | 0.296 | 71 |
| [0.2,0.3) | 0.240 | 0.167 | 72 |
| [0.3,0.4) | 0.361 | 0.536 | 112 |
| [0.4,0.5) | 0.441 | 0.556 | 45 |
| [0.5,0.6) | 0.524 | 0.143 | 14 |
| [0.6,0.7) | 0.620 | 0.250 | 4 |
| [0.7,0.8) | 0.777 | 0.500 | 2 |
| [0.8,0.9) | 0.863 | 0.000 | 10 |
| [0.9,1.0) | 0.941 | 0.000 | 19 |

### structural

**Brier**: 0.2482 | **ECE**: 0.2027

| Bin | Avg Pred | Avg Correct | N |
|-----|----------|-------------|---|
| [0.0,0.1) | 0.045 | 0.417 | 24 |
| [0.1,0.2) | 0.152 | 0.554 | 65 |
| [0.2,0.3) | 0.248 | 0.142 | 127 |
| [0.3,0.4) | 0.354 | 0.328 | 119 |
| [0.4,0.5) | 0.426 | 0.756 | 127 |
| [0.5,0.6) | 0.547 | 0.688 | 16 |
| [0.7,0.8) | 0.729 | 0.000 | 2 |

### pcfg

**Brier**: 0.2798 | **ECE**: 0.1738

| Bin | Avg Pred | Avg Correct | N |
|-----|----------|-------------|---|
| [0.0,0.1) | 0.045 | 0.691 | 55 |
| [0.1,0.2) | 0.157 | 0.239 | 67 |
| [0.2,0.3) | 0.247 | 0.244 | 119 |
| [0.3,0.4) | 0.351 | 0.522 | 209 |
| [0.4,0.5) | 0.443 | 0.667 | 21 |
| [0.5,0.6) | 0.528 | 0.571 | 7 |
| [0.7,0.8) | 0.745 | 0.000 | 2 |

