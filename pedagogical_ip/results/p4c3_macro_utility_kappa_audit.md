# P4-C.3: 3D vs 4D Macro Utility + κ Observability

## Part 1: 3D vs 4D Macro Utility

### STOP Agreement

| θ | tempt | STOP(3D) | STOP(4D) | STOP(oracle) | Agree(3D) | Agree(4D) |
|:-:|:-----:|:--------:|:--------:|:------------:|:---------:|:---------:|
| safe | none | 0.305 | 0.305 | 0.303 | 93.3% | 93.3% |
| safe | al=0.6 | 0.303 | 0.303 | 0.302 | 100.0% | 100.0% |
| safe | cf=1.0 | 0.302 | 0.302 | 0.301 | 100.0% | 100.0% |
| shiny | none | 0.308 | 0.308 | 0.316 | 100.0% | 100.0% |
| shiny | al=0.6 | 0.317 | 0.317 | 0.343 | 100.0% | 100.0% |
| shiny | cf=1.0 | 0.318 | 0.318 | 0.350 | 100.0% | 100.0% |

### TEACH Ranking (Kendall τ + Top-1)

| θ | tempt | Kendall(3D) | Kendall(4D) | Top1(3D) | Top1(4D) |
|:-:|:-----:|:-----------:|:-----------:|:--------:|:--------:|
| safe | none | 0.9964 | 0.9964 | 100% | 100% |
| safe | al=0.6 | 0.9991 | 0.9991 | 100% | 100% |
| safe | cf=1.0 | 1.0000 | 1.0000 | 100% | 100% |
| shiny | none | 0.9973 | 0.9973 | 100% | 100% |
| shiny | al=0.6 | 0.9964 | 0.9964 | 100% | 100% |
| shiny | cf=1.0 | 0.9964 | 0.9964 | 100% | 100% |

## Part 2: κ Observability Audit

### Risk Error Signal

| θ | Mean |e_risk| | Std | Range | Non-zero% |
|:-:|:-------------:|:---:|:-----:|:---------:|
| safe | 0.1450 | 0.1082 | [0.0002, 0.4542] | 97.0% |
| shiny | 0.1465 | 0.1081 | [0.0003, 0.4385] | 94.3% |

### Collinearity: κ-proxy vs Existing Dims

| θ | Corr(e_risk, τ̂) | Corr(e_risk, ν̂) | Corr(e_risk, γ̂_gen) | Corr(e_risk, γ̂_spec) |
|:-:|:---------------:|:---------------:|:-------------------:|:--------------------:|
| safe | 0.0000 | 0.0144 | 0.0945 | -0.0361 |
| shiny | 0.0000 | 0.0112 | -0.0033 | -0.0407 |

### Simulated κ Trajectory (EMA of 1−e_risk)

| θ | κ(final) | κ stability (last 5 std) |
|:-:|:--------:|:-----------------------:|
| safe | 0.4002 | 0.0050 |
| shiny | 0.3995 | 0.0056 |

## Verdict

### Macro Utility

> See STOP agreement and Kendall τ above for 3D vs 4D comparison.

### κ Observability

> **κ signal present.** Mean |e_risk| = 0.1458. Risk error is detectable and could support a 5th dimension.
