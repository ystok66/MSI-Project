# Online Infer-Only + Active Benchmark + Macro Replay

## Exp 1: Infer-Only vs Oracle (Canonical + Temptation)

| Config | θ | Success | Dose Rate | Diverge All | Diverge@Active | n_active | Transfer |
|--------|:-:|:-------:|:---------:|:-----------:|:--------------:|:--------:|:--------:|
| canonical | safe | 0.5708 | 0.0208 | 0.0 | 0.0 | 5 | 0.5333 |
| canonical | shiny | 0.525 | 0.0083 | 0.0 | 0.0 | 2 | 0.5833 |
| tempt=0.6 | safe | 0.7917 | 0.0208 | 0.0 | 0.0 | 5 | 0.8167 |
| tempt=0.6 | shiny | 0.0542 | 0.0083 | 0.0 | 0.0 | 2 | 0.1667 |
| tempt=1.0 | safe | 0.8917 | 0.0208 | 0.0 | 0.0 | 5 | 0.9167 |
| tempt=1.0 | shiny | 0.0417 | 0.0083 | 0.0 | 0.0 | 2 | 0.1 |

## Exp 2: Active Benchmark (Forced Intervention Scenarios)

| Regime | θ | Diverge All | Diverge@Active | n_active | Diverge@Hard | n_hard |
|--------|:-:|:-----------:|:--------------:|:--------:|:------------:|:------:|
| natural | safe | 0.0 | 0.0 | 5 | 0.0 | 120 |
| natural | shiny | 0.0 | 0.0 | 2 | 0.0 | 120 |
| active_0.5 | safe | 0.0125 | 0.6 | 5 | 0.025 | 120 |
| active_0.5 | shiny | 0.0042 | 0.5 | 2 | 0.0083 | 120 |
| active_1.0 | safe | 0.0125 | 0.6 | 5 | 0.025 | 120 |
| active_1.0 | shiny | 0.0042 | 0.5 | 2 | 0.0083 | 120 |

## Exp 3: Q-Margin Analysis

| θ | Mean Q_oracle | Mean Q_infer | Mean |ΔQ| | Corr(Q_o, Q_i) |
|:-:|:-------------:|:------------:|:----------:|:--------------:|
| safe | 3.7441 | 3.7456 | 0.0024 | 1.0000 |
| shiny | 3.7982 | 3.7992 | 0.0013 | 1.0000 |

## Exp 4: Macro STOP/EVAL Offline Replay

| α | θ | ε_stop oracle | ε_stop infer | Δε_stop | STOP agree |
|:-:|:-:|:---:|:---:|:---:|:---:|
| 0.0 | safe | 0.3136 | 0.3136 | 0.0000 | 1.000 |
| 0.0 | shiny | 0.3113 | 0.3113 | 0.0000 | 1.000 |
| 0.5 | safe | 0.3136 | 0.3123 | 0.0021 | 0.992 |
| 0.5 | shiny | 0.3113 | 0.3109 | 0.0013 | 1.000 |
| 1.0 | safe | 0.3136 | 0.3111 | 0.0042 | 0.992 |
| 1.0 | shiny | 0.3113 | 0.3104 | 0.0026 | 1.000 |

## Exp 5: Policy Coverage Benchmark

| θ | warn_rate | dose>0 | blind>0 | selfdisc>0 | trust>0 |
|:-:|:---------:|:------:|:-------:|:----------:|:-------:|
| safe | 0.0208 | 0.0208 | 0.0208 | 0.5625 | 0.0208 |
| shiny | 0.0083 | 0.0083 | 0.0083 | 0.5250 | 0.0083 |
