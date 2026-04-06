# T1 Exp-1: κ̂ Bonus Default-On Regression

## Part 1: Micro Metrics (Canonical baseline)

| θ | DivAll | Div@Act | OverWarnRate | Success |
|:-:|:------:|:-------:|:-----------:|:-------:|
| safe | 0.0083 | 0.2000 | 0.0083 | 0.537 |
| shiny | 0.0000 | 0.0000 | 0.0000 | 0.492 |

## Part 2: Macro — β_κ=0 vs β_κ=0.02

| θ | β_κ | Top-1 Same | Kendall | Risk Shift | STOP | κ̂ |
|:-:|:---:|:----------:|:-------:|:----------:|:----:|:--:|
| safe | 0.00 | ✅ | 1.0000 | +0.0 | 0.3340 | 0.2980 |
| safe | 0.02 | ✅ | 0.9868 | +0.7 | 0.3340 | 0.2980 |
| shiny | 0.00 | ✅ | 1.0000 | +0.0 | 0.3045 | 0.2980 |
| shiny | 0.02 | ✅ | 0.9868 | +0.7 | 0.3045 | 0.2980 |

## Part 3: OOD Robustness (β_κ=0.02)

| Condition | θ | DivAll | OWR | Top-1 | Kendall | κ̂ |
|-----------|:-:|:------:|:---:|:-----:|:-------:|:--:|
| Canonical | safe | 0.0083 | 0.0083 | ✅ | 0.9868 | 0.2980 |
| Canonical | shiny | 0.0000 | 0.0000 | ✅ | 0.9868 | 0.2980 |
| Tempt-rich | safe | 0.0083 | 0.0083 | ✅ | 0.9868 | 0.2980 |
| Tempt-rich | shiny | 0.0000 | 0.0000 | ✅ | 0.9868 | 0.2980 |
| Balanced-Act | safe | 0.0875 | 0.0875 | ✅ | 0.9868 | 0.3097 |
| Balanced-Act | shiny | 0.0708 | 0.0708 | ✅ | 0.9864 | 0.3120 |
| High-risk×1.5 | safe | 0.0083 | 0.0083 | ✅ | 0.9868 | 0.3201 |
| High-risk×1.5 | shiny | 0.0000 | 0.0000 | ✅ | 0.9868 | 0.3180 |
| Low-risk×0.5 | safe | 0.0083 | 0.0083 | ✅ | 0.9868 | 0.2870 |
| Low-risk×0.5 | shiny | 0.0000 | 0.0000 | ✅ | 0.9868 | 0.2881 |

## Part 4: Per-Family Check (no regression)

| Family | n | R²(4D) | R²(5D) | ΔR² |
|--------|:-:|:------:|:------:|:---:|
| beneficial_novelty | 24 | 0.1479 | 0.1481 | +0.0002 |
| blind_activation_corridor | 24 | 0.0353 | 0.0391 | +0.0037 |
| false_suppression | 24 | 0.0196 | 0.0371 | +0.0175 |
| ppmrb_self_discovery | 48 | 0.1131 | 0.1157 | +0.0026 |
| ppmrb_standard | 48 | 0.0747 | 0.2203 | +0.1456 |
| soft_boundary_tradeoff | 24 | 0.1639 | 0.2987 | +0.1347 |
| sparse_invalid_advice | 48 | 0.0255 | 0.0282 | +0.0027 |
| sparse_valid_advice | 48 | 0.1686 | 0.3823 | +0.2138 |
| tic_rescue_heavy | 48 | 0.0912 | 0.4504 | +0.3592 |
| tic_self_discovery | 48 | 0.0548 | 0.0665 | +0.0117 |
| tic_temptation | 48 | 0.2112 | 0.2577 | +0.0465 |
| verified_warn | 24 | 0.0389 | 0.0923 | +0.0534 |
| warn_symmetric_rescue | 24 | 0.0455 | 0.3497 | +0.3042 |

## Verdict

> OOD pass rate: 10/10
> Families with significant ΔR² regression: 0/13
> **✅ κ̂ bonus (β=0.02) default-on: PASS**
