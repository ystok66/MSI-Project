# T3 Shadow POMDP Interface Audit

## Exp-T3-1: Prediction Parity

| θ | NLL_old | NLL_new | |Δ NLL| | Brier_old | Brier_new | Top1Agree |
|:-:|:-------:|:-------:|:------:|:---------:|:---------:|:---------:|
| safe | 0.3563 | 0.3563 | 0.000000 | 0.2033 | 0.2033 | 1.000 |
| shiny | 0.4442 | 0.4442 | 0.000000 | 0.2739 | 0.2739 | 1.000 |

## Exp-T3-2: Calibration & Posterior

| θ | ECE_old | ECE_new | Mean Entropy | Final θ_MAP |
|:-:|:-------:|:-------:|:------------:|:-----------:|
| safe | 0.1100 | 0.1100 | 0.273 | safe |
| shiny | 0.1133 | 0.1133 | 0.117 | shiny |

## Exp-T3-5: Benchmark No-Regression

| θ | Success | SelGap | WR_nec | WR_unnec |
|:-:|:-------:|:------:|:------:|:--------:|
| safe | 0.567 | 0.043 | 0.043 | 0.000 |
| shiny | 0.453 | 0.029 | 0.029 | 0.000 |

## Verdict

> NLL parity (|Δ| < 0.01): ✅
> Top-1 agreement ≥ 95%: ✅
> ECE not worse (≤ old + 0.02): ✅
> θ_MAP recovery: ✅
> WR_unnecessary ≤ 0.05: ✅
> **✅ Shadow POMDP interface passes all parity checks**
