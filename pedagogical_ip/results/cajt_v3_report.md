# CAJT-v3: Calibrated Adaptive Joint Tutor

## A. PP-MRB Main Results

| θ | Strategy | SBCR | WR | WR(wc) | WR(wt) | **SelGap** | C_t | CalTop1 | D_t |
|---|----------|------|----|--------|--------|-----------|-----|---------|-----|
| safe | v1_1 | 89% | 68% | 26% | 100% | **0.739** | — | — | — |
| safe | joint_v2 | 89% | 83% | 51% | 100% | **0.489** | — | — | — |
| safe | cajt_v3_no_cal | 89% | 82% | 43% | 100% | **0.572** | 0.0600 | 0.1520 | 0.3330 |
| safe | cajt_v3_no_adapt | 89% | 67% | 32% | 100% | **0.683** | 0.1340 | 0.2720 | 0.3290 |
| safe | cajt_v3_full | 89% | 71% | 34% | 100% | **0.656** | 0.1090 | 0.2380 | 0.3300 |
| safe | oracle | 89% | 31% | 0% | 100% | **1.000** | — | — | — |
| shiny | v1_1 | 29% | 62% | 5% | 100% | **0.950** | — | — | — |
| shiny | joint_v2 | 29% | 94% | 76% | 100% | **0.240** | — | — | — |
| shiny | cajt_v3_no_cal | 29% | 94% | 77% | 100% | **0.230** | 0.0680 | 0.1590 | 0.3530 |
| shiny | cajt_v3_no_adapt | 29% | 83% | 37% | 100% | **0.630** | 0.1750 | 0.2890 | 0.3530 |
| shiny | cajt_v3_full | 29% | 86% | 47% | 100% | **0.530** | 0.1360 | 0.2490 | 0.3540 |
| shiny | oracle | 29% | 39% | 0% | 100% | **1.000** | — | — | — |

### SelGap Comparison

| θ | v1.1 | joint_v2 | v3_no_cal | v3_no_adapt | **v3_full** | oracle |
|---|------|----------|-----------|-------------|------------|--------|
| safe | 0.739 | 0.489 | 0.572 | 0.683 | **0.656** | 1.000 |
| shiny | 0.950 | 0.240 | 0.230 | 0.630 | **0.530** | 1.000 |

## B. Mild Latent Drift (θ switches at episode 8)

| θ→ | Strategy | SelGap(stable) | SelGap(drift) | |Δ| |
|-----|----------|----------------|---------------|---------|
| safe→neutral | v1_1 | 0.739 | 0.739 | 0.000 |
| safe→neutral | joint_v2 | 0.489 | 0.322 | 0.167 |
| safe→neutral | cajt_v3_full | 0.656 | 0.489 | 0.167 |

## C. Wrong-Memory Recovery

| θ | Strategy | SG(correct) | SG(adversarial) | Recovery |
|---|----------|-------------|-----------------|----------|
| safe | v1_1 | 0.739 | 0.794 | ✅ |
| safe | cajt_v3_full | 0.656 | 0.689 | ✅ |
| shiny | v1_1 | 0.950 | 0.950 | ✅ |
| shiny | cajt_v3_full | 0.530 | 1.000 | ✅ |

## D. Calibration Gap (PredTop1 vs ActualCorrect)

| θ | Strategy | PredTop1 | ActualCorr | |Gap| |
|---|----------|----------|------------|--------|
| safe | v1_1 | 0.575 | 1.000 | 0.425 |
| safe | joint_v2 | 0.192 | 1.000 | 0.808 |
| safe | cajt_v3_full | 0.242 | 1.000 | 0.758 |
| shiny | v1_1 | 0.774 | 0.833 | 0.060 |
| shiny | joint_v2 | 0.158 | 0.333 | 0.175 |
| shiny | cajt_v3_full | 0.180 | 0.167 | 0.014 |
