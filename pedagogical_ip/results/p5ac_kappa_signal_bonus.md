# P5-A: κ Family Signal Audit + P5-C: Macro Bonus

## Part 1: Per-Family κ̂ Signal

| Family | n | Mean δ_risk | Mean Δκ̂ | Sign Acc | Corr(κ̂,δ) | κ̂ range |
|--------|:-:|:----------:|:-------:|:--------:|:----------:|:--------:|
| beneficial_novelty | 30 | -0.3526 | -0.000609 | 100.0% | -0.2726 | [0.2949,0.3004] |
| blind_activation_corridor | 30 | 0.0657 | 0.000751 | 96.3% | 0.9921 | [0.2959,0.3034] |
| false_suppression | 30 | -0.3389 | -0.000847 | 100.0% | 0.0772 | [0.2941,0.3003] |
| ppmrb_self_discovery | 60 | -0.1324 | -0.000460 | 100.0% | 0.9942 | [0.2954,0.3032] |
| ppmrb_standard | 60 | -0.1152 | -0.000346 | 100.0% | 0.9728 | [0.2958,0.3035] |
| soft_boundary_tradeoff | 30 | 0.0042 | 0.000282 | 100.0% | 0.9609 | [0.2950,0.3026] |
| sparse_invalid_advice | 60 | -0.2389 | -0.000837 | 100.0% | 0.9879 | [0.2945,0.3015] |
| sparse_valid_advice | 60 | -0.1296 | -0.000447 | 100.0% | 0.9913 | [0.2955,0.3023] |
| tic_rescue_heavy | 60 | 0.0858 | 0.000978 | 100.0% | 0.9926 | [0.2954,0.3032] |
| tic_self_discovery | 60 | -0.1394 | -0.000501 | 100.0% | 0.9910 | [0.2956,0.3027] |
| tic_temptation | 60 | -0.0128 | 0.000113 | 100.0% | 0.9550 | [0.2962,0.3029] |
| verified_warn | 30 | -0.0021 | 0.000280 | 96.6% | 0.9693 | [0.2946,0.3012] |
| warn_symmetric_rescue | 30 | 0.0943 | 0.001066 | 100.0% | 0.9964 | [0.2951,0.3026] |

**Overall: Sign Acc = 99.6%, Corr(κ̂,δ_risk) = 0.8761**

### Collinearity (5D)

| Dim | Corr(κ̂, dim) |
|-----|:------------:|
| tau | 0.0000 |
| nu | -0.1290 |
| gamma_gen | -0.1330 |
| gamma_spec | -0.1577 |

## Part 2: Macro κ Bonus — Risk Lesson Priority

### Lesson Ranking: Baseline vs κ-Bonus

| θ | Baseline Top-3 | κ-Bonus Top-3 | Risk-lesson Rank Shift |
|:-:|:--------------:|:-------------:|:----------------------:|
| safe | tic_rescue_heavy, tic_self_discovery, ppmrb_self_discovery | tic_rescue_heavy, warn_symmetric_rescue, blind_activation_corridor | avg +2.3 |
| shiny | tic_rescue_heavy, tic_self_discovery, ppmrb_self_discovery | tic_rescue_heavy, warn_symmetric_rescue, blind_activation_corridor | avg +2.3 |

### TEACH Top-1 Stability Under κ-Bonus

| θ | Baseline Top-1 Agree | κ-Bonus Top-1 Agree | Kendall(base) | Kendall(bonus) |
|:-:|:--------------------:|:-------------------:|:-------------:|:--------------:|
| safe | 100% | 100% | 1.0000 | 0.8531 |
| shiny | 100% | 100% | 1.0000 | 0.8523 |

## Verdict

> **κ̂ signal audit**: Overall sign accuracy = 99.6%, Corr(κ̂,δ_risk) = 0.8761

> **κ̂ is a responsive risk-calibration state.** It moves in the correct direction relative to risk prediction error.
