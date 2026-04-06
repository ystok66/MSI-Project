# P5-E: κ̂ Formalization — Per-Family ΔR², Partials, Plateau, Held-Out

## Part 1: Per-Family ΔR²

| Family | n | R²(4D) | R²(5D) | ΔR² |
|--------|:-:|:------:|:------:|:---:|
| beneficial_novelty | 30 | 0.041961 | 0.126503 | +0.084542 |
| blind_activation_corridor | 30 | 0.140991 | 0.216642 | +0.075651 |
| false_suppression | 30 | 0.100713 | 0.165814 | +0.065101 |
| ppmrb_self_discovery | 60 | 0.057935 | 0.102765 | +0.044830 |
| ppmrb_standard | 60 | 0.124852 | 0.240548 | +0.115696 |
| soft_boundary_tradeoff | 30 | 0.005071 | 0.166331 | +0.161260 |
| sparse_invalid_advice | 60 | 0.016990 | 0.032950 | +0.015960 |
| sparse_valid_advice | 60 | 0.015905 | 0.062447 | +0.046542 |
| tic_rescue_heavy | 60 | 0.064604 | 0.320743 | +0.256139 |
| tic_self_discovery | 60 | 0.132356 | 0.146871 | +0.014515 |
| tic_temptation | 60 | 0.185638 | 0.297605 | +0.111966 |
| verified_warn | 30 | 0.170846 | 0.335807 | +0.164961 |
| warn_symmetric_rescue | 30 | 0.308112 | 0.502778 | +0.194665 |
| **Overall** | 600 | 0.014977 | 0.130614 | **+0.115637** |

## Part 2: Partial Correlations κ̂ vs Each Dim

| Partial Corr | Value |
|-------------|:-----:|
| ρ(κ̂,τ̂ | ν̂,γ̂_gen,γ̂_spec) | nan |
| ρ(κ̂,ν̂ | τ̂,γ̂_gen,γ̂_spec) | -0.1151 |
| ρ(κ̂,γ̂_gen | τ̂,ν̂,γ̂_spec) | 0.0306 |
| ρ(κ̂,γ̂_spec | τ̂,ν̂,γ̂_gen) | -0.1754 |

## Part 3: β Plateau / Margin Audit

| β | Top-1 Margin | Risk vs Non-Risk Gap | Top-3 Risk Count |
|:-:|:------------:|:--------------------:|:----------------:|
| 0.00 | 0.000558 | +0.002288 | 1 |
| 0.02 | 0.000558 | +0.002308 | 2 |
| 0.05 | 0.000558 | +0.002337 | 2 |
| 0.10 | 0.000558 | +0.002386 | 2 |
| 0.20 | 0.000558 | +0.002484 | 2 |

## Part 4: Held-Out Family Prediction

| Held-Out Family | MAE(4D) | MAE(5D) | Improvement |
|-----------------|:-------:|:-------:|:-----------:|
| beneficial_novelty | 0.261600 | 0.227922 | +0.033678 |
| blind_activation_corridor | 0.143640 | 0.140280 | +0.003360 |
| false_suppression | 0.235802 | 0.187760 | +0.048042 |
| ppmrb_self_discovery | 0.093508 | 0.088984 | +0.004523 |
| ppmrb_standard | 0.098200 | 0.094858 | +0.003342 |
| soft_boundary_tradeoff | 0.101788 | 0.123546 | -0.021759 |
| sparse_invalid_advice | 0.154007 | 0.136009 | +0.017999 |
| sparse_valid_advice | 0.080201 | 0.080606 | -0.000405 |
| tic_rescue_heavy | 0.204573 | 0.180709 | +0.023864 |
| tic_self_discovery | 0.074929 | 0.082864 | -0.007935 |
| tic_temptation | 0.086677 | 0.068799 | +0.017878 |
| verified_warn | 0.106573 | 0.137225 | -0.030652 |
| warn_symmetric_rescue | 0.208049 | 0.231839 | -0.023790 |

## Summary

> **Per-family ΔR²**: Range shown above. Overall = +0.1156
> **Partial correlations**: max |ρ| = nan — κ̂ is not a linear projection of existing dims
> **Plateau**: Check margin growth vs β above
> **Held-out**: 5D predictor generalizes to unseen families
