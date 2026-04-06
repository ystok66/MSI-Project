# T1 Exp-4: Stability Retest (ε_Q=0.05, β_κ=0.02)

## Part 1: Micro Metrics (with dead-zone)

| θ | DivAll | Div@Act | OWR | WarnNecRecall | Success |
|:-:|:------:|:-------:|:---:|:-------------:|:-------:|
| safe | 0.0167 | 0.3846 | 0.0167 | 1.0000 | 0.480 |
| shiny | 0.0033 | 0.1111 | 0.0000 | 0.8889 | 0.497 |

## Part 2: Macro Stability

| θ | Top-1 Same | Kendall | Risk Shift | STOP | κ̂ |
|:-:|:----------:|:-------:|:----------:|:----:|:--:|
| safe | ✅ | 0.9868 | +0.7 | 0.3075 | 0.2982 |
| shiny | ✅ | 0.9868 | +0.7 | 0.3189 | 0.2993 |

## Part 3: OOD Robustness

| Condition | θ | DivAll | OWR | Top-1 | κ̂ |
|-----------|:-:|:------:|:---:|:-----:|:--:|
| Canonical | safe | 0.0167 | 0.0167 | ✅ | 0.2982 |
| Canonical | shiny | 0.0033 | 0.0000 | ✅ | 0.2993 |
| Tempt-rich | safe | 0.0167 | 0.0167 | ✅ | 0.2982 |
| Tempt-rich | shiny | 0.0033 | 0.0000 | ✅ | 0.2993 |
| Balanced-Act | safe | 0.0767 | 0.0667 | ✅ | 0.3056 |
| Balanced-Act | shiny | 0.0733 | 0.0733 | ✅ | 0.3088 |
| High-risk×1.5 | safe | 0.0167 | 0.0167 | ✅ | 0.3204 |
| High-risk×1.5 | shiny | 0.0067 | 0.0033 | ✅ | 0.3216 |
| Low-risk×0.5 | safe | 0.0133 | 0.0133 | ✅ | 0.2881 |
| Low-risk×0.5 | shiny | 0.0033 | 0.0000 | ✅ | 0.2876 |

## Part 4: Held-Out Family Prediction

| Held-Out | MAE(4D) | MAE(5D) | Δ |
|----------|:-------:|:-------:|:-:|
| beneficial_novelty | 0.2531 | 0.2149 | +0.0383 |
| blind_activation_corridor | 0.1607 | 0.1460 | +0.0147 |
| false_suppression | 0.2237 | 0.1692 | +0.0544 |
| ppmrb_self_discovery | 0.0604 | 0.0652 | -0.0048 |
| ppmrb_standard | 0.1020 | 0.0962 | +0.0057 |
| soft_boundary_tradeoff | 0.1321 | 0.1438 | -0.0117 |
| sparse_invalid_advice | 0.1987 | 0.1786 | +0.0201 |
| sparse_valid_advice | 0.0657 | 0.0723 | -0.0066 |
| tic_rescue_heavy | 0.1921 | 0.1670 | +0.0251 |
| tic_self_discovery | 0.0732 | 0.0906 | -0.0175 |
| tic_temptation | 0.1082 | 0.0859 | +0.0224 |
| verified_warn | 0.1027 | 0.1415 | -0.0388 |
| warn_symmetric_rescue | 0.1780 | 0.2141 | -0.0361 |

## Final Verdict

> OOD pass rate: 10/10
> Held-out 5D wins: 7/13
> ✅ OOD Top-1 all pass
> ✅ Held-out 5D >= 50% wins

> **✅ STABILITY RETEST: PASS — Canonical is locked.**
