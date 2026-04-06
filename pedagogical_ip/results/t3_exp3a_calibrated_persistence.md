# T3 Exp-3A: Calibrated Persistence

## Calibration Error (E_calib) by Session

| θ | Sess | reset | raw | calibrated | cal < raw? |
|:-:|:----:|:-----:|:---:|:----------:|:----------:|
| safe | 0 | 0.0213 | 0.0213 | 0.0213 | ≈ |
| safe | 1 | 0.0574 | 0.0631 | 0.0631 | ≈ |
| safe | 2 | 0.0616 | 0.0898 | 0.0900 | ≈ |
| safe | 3 | 0.0317 | 0.0900 | 0.0907 | ≈ |
| safe | 4 | 0.0303 | 0.1011 | 0.1017 | ≈ |
| safe | 5 | 0.0538 | 0.1179 | 0.1185 | ≈ |
| shiny | 0 | 0.0086 | 0.0086 | 0.0086 | ≈ |
| shiny | 1 | 0.0493 | 0.0477 | 0.0477 | ≈ |
| shiny | 2 | 0.0412 | 0.0648 | 0.0649 | ≈ |
| shiny | 3 | 0.0257 | 0.0507 | 0.0504 | ≈ |
| shiny | 4 | 0.0328 | 0.0591 | 0.0587 | ≈ |
| shiny | 5 | 0.0449 | 0.0824 | 0.0821 | ≈ |

## WarnRate by Session

| θ | Sess | reset | raw | calibrated |
|:-:|:----:|:-----:|:---:|:----------:|
| safe | 0 | 0.015 | 0.015 | 0.015 |
| safe | 1 | 0.030 | 0.025 | 0.025 |
| safe | 2 | 0.035 | 0.030 | 0.030 |
| safe | 3 | 0.020 | 0.015 | 0.015 |
| safe | 4 | 0.020 | 0.015 | 0.015 |
| safe | 5 | 0.030 | 0.025 | 0.025 |
| shiny | 0 | 0.010 | 0.010 | 0.010 |
| shiny | 1 | 0.030 | 0.025 | 0.025 |
| shiny | 2 | 0.035 | 0.035 | 0.035 |
| shiny | 3 | 0.020 | 0.015 | 0.015 |
| shiny | 4 | 0.020 | 0.015 | 0.015 |
| shiny | 5 | 0.030 | 0.025 | 0.025 |

## Per-Dimension Calibration (Final Session)

| θ | Mode | E_τ | E_ν | E_γ |
|:-:|:----:|:---:|:---:|:---:|
| safe | reset | 0.0984 | 0.0336 | 0.0293 |
| safe | persistent_raw | 0.2951 | 0.0313 | 0.0273 |
| safe | persistent_calibrated | 0.2951 | 0.0317 | 0.0287 |
| shiny | reset | 0.0219 | 0.0849 | 0.0279 |
| shiny | persistent_raw | 0.1367 | 0.0825 | 0.0281 |
| shiny | persistent_calibrated | 0.1367 | 0.0805 | 0.0292 |

## Verdict

> E_calib reduction: calibrated < raw in 0/2 θ
> WarnRate preserved: calibrated ≤ reset in 2/2 θ
> E_calib slope flattened: 0/2 θ
> **⚠️ Calibration needs further investigation**
