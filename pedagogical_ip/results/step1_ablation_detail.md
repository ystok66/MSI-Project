# Step 1: Ablation Detail

## Ablation Results

| Ablation | TBSR | WarnRate | SD_Rate | Brier | p_undecided |
|:---------|:----:|:-------:|:-------:|:-----:|:----------:|
| LG1_behavior_loss | 0.4875 | 0.2500 | 0.3292 | 0.3383 | 0.0000 |
| LG2_entropy_reduction | 0.4875 | 0.2542 | 0.3292 | 0.3383 | 0.0000 |
| DC_simple | 0.4875 | 0.2500 | 0.3292 | 0.3383 | 0.0000 |
| DC_full | 0.4875 | 0.2500 | 0.3292 | 0.3383 | 0.0000 |
| PS_baseline | 0.4875 | 0.2500 | 0.3292 | 0.3383 | 0.0000 |
| PS_A_fusion | 0.4875 | 0.3083 | 0.2667 | 0.2284 | 0.0000 |
| PS_B_predictive | 0.4875 | 0.3083 | 0.2667 | 0.2284 | 0.0000 |
| PS_C_three_outcome | 0.4875 | 0.3083 | 0.2667 | 0.2284 | 0.1719 |
| kappa_aware | 0.4875 | 0.2500 | 0.3292 | 0.3383 | 0.0000 |
| kappa_off | 0.4875 | 0.2500 | 0.3292 | 0.3383 | 0.0000 |

## Verdicts

> **LearnGain**: behavior_loss TBSR=0.4875 vs entropy_reduction TBSR=0.4875
> → Approximately equivalent. Keep behavior_loss (simpler).

> **DepCost**: simple WR=0.2500 vs full WR=0.2500
> → ν̂ increment adds no discriminative value. Use simple.

> **p_self variants** Brier: baseline=0.3383, A=0.2284, B=0.2284, C=0.2284
> → Best calibration: **A** (Brier=0.2284)

> **Three-outcome model**: mean p_undecided = 0.1719
> → p_fail = 1-p_self is **too coarse** — significant undecided mass.

> **κ̂-aware**: TBSR=0.4875 vs 3D-only=0.4875
> → κ̂ does not help. Keep 3D-only as default.
