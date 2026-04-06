# T2 Exp-2C: TIC-v4 Longitudinal Audit — 3-Arm


## θ = safe

### Success, SelGap, WarnRate

| Sess | Mode | Success | SelGap | WR_nec | WR_unnec |
|:----:|:----:|:-------:|:------:|:------:|:--------:|
| 0 | reset | 0.595 | 0.054 | 0.054 | 0.000 |
| 0 | persistent_nohook | 0.595 | 0.054 | 0.054 | 0.000 |
| 0 | persistent_needhook | 0.595 | 0.054 | 0.054 | 0.000 |
| 1 | reset | 0.520 | 0.107 | 0.107 | 0.000 |
| 1 | persistent_nohook | 0.515 | 0.089 | 0.089 | 0.000 |
| 1 | persistent_needhook | 0.515 | 0.089 | 0.089 | 0.000 |
| 2 | reset | 0.510 | 0.125 | 0.125 | 0.000 |
| 2 | persistent_nohook | 0.500 | 0.107 | 0.107 | 0.000 |
| 2 | persistent_needhook | 0.500 | 0.107 | 0.107 | 0.000 |
| 3 | reset | 0.545 | 0.071 | 0.071 | 0.000 |
| 3 | persistent_nohook | 0.545 | 0.054 | 0.054 | 0.000 |
| 3 | persistent_needhook | 0.545 | 0.054 | 0.054 | 0.000 |
| 4 | reset | 0.495 | 0.071 | 0.071 | 0.000 |
| 4 | persistent_nohook | 0.490 | 0.054 | 0.054 | 0.000 |
| 4 | persistent_needhook | 0.490 | 0.054 | 0.054 | 0.000 |

## θ = shiny

### Success, SelGap, WarnRate

| Sess | Mode | Success | SelGap | WR_nec | WR_unnec |
|:----:|:----:|:-------:|:------:|:------:|:--------:|
| 0 | reset | 0.445 | 0.036 | 0.036 | 0.000 |
| 0 | persistent_nohook | 0.445 | 0.036 | 0.036 | 0.000 |
| 0 | persistent_needhook | 0.445 | 0.036 | 0.036 | 0.000 |
| 1 | reset | 0.480 | 0.107 | 0.107 | 0.000 |
| 1 | persistent_nohook | 0.475 | 0.089 | 0.089 | 0.000 |
| 1 | persistent_needhook | 0.475 | 0.089 | 0.089 | 0.000 |
| 2 | reset | 0.525 | 0.125 | 0.125 | 0.000 |
| 2 | persistent_nohook | 0.540 | 0.125 | 0.125 | 0.000 |
| 2 | persistent_needhook | 0.540 | 0.125 | 0.125 | 0.000 |
| 3 | reset | 0.505 | 0.071 | 0.071 | 0.000 |
| 3 | persistent_nohook | 0.520 | 0.054 | 0.054 | 0.000 |
| 3 | persistent_needhook | 0.520 | 0.054 | 0.054 | 0.000 |
| 4 | reset | 0.575 | 0.071 | 0.071 | 0.000 |
| 4 | persistent_nohook | 0.575 | 0.054 | 0.054 | 0.000 |
| 4 | persistent_needhook | 0.575 | 0.054 | 0.054 | 0.000 |

## Per-Subtype WarnRate (Final Session)

| θ | Subtype | reset | nohook | needhook |
|:-:|:-------:|:-----:|:------:|:--------:|
| safe | verified_warn | 0.000 | 0.000 | 0.000 |
| safe | warn_rescue | 0.125 | 0.094 | 0.094 |
| safe | sparse_valid_advice | 0.000 | 0.000 | 0.000 |
| safe | sparse_invalid_advice | 0.000 | 0.000 | 0.000 |
| safe | beneficial_novelty | 0.000 | 0.000 | 0.000 |
| safe | false_suppression_cost | 0.000 | 0.000 | 0.000 |
| safe | self_discovery_needed | 0.000 | 0.000 | 0.000 |
| safe | temptation_repeat | 0.000 | 0.000 | 0.000 |
| shiny | verified_warn | 0.000 | 0.000 | 0.000 |
| shiny | warn_rescue | 0.125 | 0.094 | 0.094 |
| shiny | sparse_valid_advice | 0.000 | 0.000 | 0.000 |
| shiny | sparse_invalid_advice | 0.000 | 0.000 | 0.000 |
| shiny | beneficial_novelty | 0.000 | 0.000 | 0.000 |
| shiny | false_suppression_cost | 0.000 | 0.000 | 0.000 |
| shiny | self_discovery_needed | 0.000 | 0.000 | 0.000 |
| shiny | temptation_repeat | 0.000 | 0.000 | 0.000 |

## Calibration Error & Drift Tracking

| θ | Sess | Mode | E_calib | E_drift |
|:-:|:----:|:----:|:-------:|:-------:|
| safe | 0 | reset | 0.0213 | 0.0000 |
| safe | 0 | persistent_nohook | 0.0213 | 0.0000 |
| safe | 0 | persistent_needhook | 0.0213 | 0.0000 |
| safe | 1 | reset | 0.0574 | 0.0550 |
| safe | 1 | persistent_nohook | 0.0631 | 0.0456 |
| safe | 1 | persistent_needhook | 0.0631 | 0.0456 |
| safe | 2 | reset | 0.0616 | 0.0566 |
| safe | 2 | persistent_nohook | 0.0898 | 0.0475 |
| safe | 2 | persistent_needhook | 0.0898 | 0.0475 |
| safe | 3 | reset | 0.0317 | 0.0515 |
| safe | 3 | persistent_nohook | 0.0900 | 0.0358 |
| safe | 3 | persistent_needhook | 0.0900 | 0.0358 |
| safe | 4 | reset | 0.0303 | 0.0421 |
| safe | 4 | persistent_nohook | 0.1011 | 0.0259 |
| safe | 4 | persistent_needhook | 0.1011 | 0.0259 |
| shiny | 0 | reset | 0.0086 | 0.0000 |
| shiny | 0 | persistent_nohook | 0.0086 | 0.0000 |
| shiny | 0 | persistent_needhook | 0.0086 | 0.0000 |
| shiny | 1 | reset | 0.0493 | 0.0549 |
| shiny | 1 | persistent_nohook | 0.0477 | 0.0520 |
| shiny | 1 | persistent_needhook | 0.0477 | 0.0520 |
| shiny | 2 | reset | 0.0412 | 0.0587 |
| shiny | 2 | persistent_nohook | 0.0648 | 0.0339 |
| shiny | 2 | persistent_needhook | 0.0648 | 0.0339 |
| shiny | 3 | reset | 0.0257 | 0.0414 |
| shiny | 3 | persistent_nohook | 0.0507 | 0.0252 |
| shiny | 3 | persistent_needhook | 0.0507 | 0.0252 |
| shiny | 4 | reset | 0.0328 | 0.0483 |
| shiny | 4 | persistent_nohook | 0.0591 | 0.0374 |
| shiny | 4 | persistent_needhook | 0.0591 | 0.0374 |

## Saturation Rate (Lock/Wash Detection)

| θ | Mode | SatRate_τ | SatRate_ν | SatRate_γ |
|:-:|:----:|:---------:|:---------:|:---------:|
| safe | reset | 0.000 | 0.550 | 1.000 |
| safe | persistent_nohook | 0.000 | 0.600 | 0.975 |
| safe | persistent_needhook | 0.000 | 0.600 | 0.975 |
| shiny | reset | 0.000 | 0.850 | 1.000 |
| shiny | persistent_nohook | 0.000 | 0.875 | 0.975 |
| shiny | persistent_needhook | 0.000 | 0.875 | 0.975 |

## Verdict

> Success maintained: persistent ≥ reset in 2/2 θ
> Warn-unnecessary no spike: 2/2 θ
> Calibration error growing: ⚠️ YES
> Saturation alert: ⚠️ YES
> needhook > nohook differentiation: 0/2 θ
> **⚠️ Issues found — investigate before closing Task 2**
