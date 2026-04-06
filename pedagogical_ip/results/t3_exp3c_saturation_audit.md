# T3 Exp-3C: ν/γ_gen Saturation Audit

## SatRate by Curriculum (All Sessions Aggregated)

| θ | Curriculum | SatRate_ν | SatRate_γ | SatRate_τ |
|:-:|:----------:|:---------:|:---------:|:---------:|
| safe | no_tutor | 0.925 | 1.000 | 0.000 |
| safe | rescue_heavy | 0.000 | 0.125 | 0.000 |
| safe | self_disc | 1.000 | 1.000 | 0.000 |
| safe | mixed | 0.575 | 0.950 | 0.000 |
| shiny | no_tutor | 0.950 | 1.000 | 0.000 |
| shiny | rescue_heavy | 0.000 | 0.075 | 0.000 |
| shiny | self_disc | 1.000 | 1.000 | 0.000 |
| shiny | mixed | 0.725 | 0.975 | 0.000 |

## Terminal State Means (Final Session)

| θ | Curriculum | ν_T | γ_gen_T | ν̂_T | γ̂_gen_T |
|:-:|:----------:|:---:|:-------:|:---:|:-------:|
| safe | no_tutor | 0.020 | 0.000 | 0.030 | 0.000 |
| safe | rescue_heavy | 0.424 | 0.154 | 0.248 | 0.061 |
| safe | self_disc | 0.015 | 0.000 | 0.023 | 0.000 |
| safe | mixed | 0.111 | 0.048 | 0.064 | 0.024 |
| shiny | no_tutor | 0.022 | 0.000 | 0.031 | 0.000 |
| shiny | rescue_heavy | 0.459 | 0.174 | 0.200 | 0.069 |
| shiny | self_disc | 0.016 | 0.000 | 0.024 | 0.000 |
| shiny | mixed | 0.104 | 0.043 | 0.069 | 0.022 |

## Mid-Session Trajectory (Steps 10-20, Final Session)

| θ | Curriculum | ν_mid | γ_gen_mid |
|:-:|:----------:|:-----:|:---------:|
| safe | no_tutor | 0.046 | 0.000 |
| safe | rescue_heavy | 0.395 | 0.139 |
| safe | self_disc | 0.035 | 0.000 |
| safe | mixed | 0.142 | 0.045 |
| shiny | no_tutor | 0.047 | 0.000 |
| shiny | rescue_heavy | 0.415 | 0.150 |
| shiny | self_disc | 0.039 | 0.000 |
| shiny | mixed | 0.125 | 0.037 |

## Verdict

> rescue_heavy SatRate_ν > self_disc: 0/2 θ
> no_tutor SatRate_γ < mixed: 0/2 θ
> self_disc ν_T < mixed: 2/2 θ
> **⚠️ Saturation appears state-dynamics intrinsic, not curriculum-modulated**
