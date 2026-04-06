# Unified Robustness Suite

## 1. Mirror Invariance
| θ | Tutor | SelGap(L) | SelGap(R) | |Δ| |
|---|-------|-----------|-----------|-----|
| safe | v1_1 | 0.895 | 0.871 | 0.024 |
| safe | joint_v2 | 0.588 | 0.644 | 0.056 |
| shiny | v1_1 | 0.781 | 0.781 | 0.000 |
| shiny | joint_v2 | 0.429 | 0.429 | 0.000 |

## 2. Parameter Shift
Tutor tuned on mid-range, tested on boundary values.

| θ | Tutor | Param | Train-range | Test-value | SelGap |
|---|-------|-------|-------------|------------|--------|
| safe | v1_1 | baseline | [2,5] | mid | 0.871 |
| safe | v1_1 | ε=0.3 | ε=0.1 | 0.3 | 0.741 |
| safe | v1_1 | β=2.0 | β=4.0 | 2.0 | 0.574 |
| safe | v1_1 | β=8.0 | β=4.0 | 8.0 | 0.820 |
| safe | joint_v2 | baseline | [2,5] | mid | 0.533 |
| safe | joint_v2 | ε=0.3 | ε=0.1 | 0.3 | 0.466 |
| safe | joint_v2 | β=2.0 | β=4.0 | 2.0 | 0.410 |
| safe | joint_v2 | β=8.0 | β=4.0 | 8.0 | 0.509 |

## 3. Noise Sweep (ε)
| θ | Tutor | ε=0.05 | ε=0.10 | ε=0.20 | ε=0.30 | ε=0.40 |
|---|-------|--------|--------|--------|--------|--------|
| safe | v1_1 | 1.000 | 1.000 | 0.917 | 0.854 | 0.604 |
| safe | joint_v2 | 0.708 | 0.542 | 0.458 | 0.542 | 0.375 |
| shiny | v1_1 | 0.857 | 0.690 | 0.690 | 0.690 | 0.643 |
| shiny | joint_v2 | 0.429 | 0.548 | 0.548 | 0.595 | 0.548 |

## 4. Session-Order Shuffle
Subtype ordering randomized vs fixed. Should be similar.

| θ | Tutor | SelGap(fixed) | SelGap(shuffled) | |Δ| |
|---|-------|--------------|-----------------|-----|
| safe | v1_1 | 0.871 | 0.500 | 0.371 |
| safe | joint_v2 | 0.533 | 0.250 | 0.283 |
| shiny | v1_1 | 0.781 | 0.958 | 0.177 |
| shiny | joint_v2 | 0.429 | 0.708 | 0.279 |

## 5. Posterior Calibration
Predicted top-1 prob vs actual correctness (ECE proxy).

| θ | Tutor | PredTop1 | ActualCorrect | |Gap| |
|---|-------|----------|---------------|--------|
| safe | v1_1 | 0.681 | 1.000 | 0.319 |
| safe | joint_v2 | 0.241 | 1.000 | 0.759 |
| shiny | v1_1 | 0.720 | 0.833 | 0.113 |
| shiny | joint_v2 | 0.138 | 0.333 | 0.195 |

## 6. Wrong-Memory Regression
| θ | Condition | SelGap | WR | Ent(1st) | Ent(2nd) |
|---|-----------|--------|-----|----------|----------|
| safe | correct_prior | 0.871 | 60% | 1.3720 | 1.0600 |
| safe | mild_wrong | 0.886 | 56% | 1.3360 | 1.0620 |
| safe | adversarial_wrong | 0.967 | 47% | 1.0760 | 1.0730 |
| shiny | correct_prior | 0.781 | 60% | 1.1250 | 0.8420 |
| shiny | mild_wrong | 0.781 | 57% | 1.0850 | 0.8700 |
| shiny | adversarial_wrong | 0.900 | 46% | 0.7130 | 0.8220 |

## 7. Cross-Family Transfer Matrix (SelGap)
| Family | v4 | v1.1 | joint_v2 |
|--------|-----|------|----------|
| PP-MRB (safe) | 0.292 | 1.000 | 0.542 |
| PP-MRB (shiny) | 0.325 | 0.690 | 0.548 |
| delayed_corridor | 100% | 100% | 100% |
| distractor_cue | 100% | 100% | 100% |
| elcb_po | 100% | 100% | 100% |
| temptation_corridor | 100% | 100% | 100% |

