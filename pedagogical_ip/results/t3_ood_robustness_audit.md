# T3 Exp-T3-6: OOD Robustness Audit

## Prediction Quality Under OOD Conditions

| θ | Condition | NLL | Brier | ECE | |Δ NLL| | Top1Agree | Entropy |
|:-:|:---------:|:---:|:-----:|:---:|:------:|:---------:|:-------:|
| safe | baseline | 0.3066 | 0.1620 | 0.0375 | 0.000000 | 1.000 | 1.041 |
| safe | noisy_obs | 0.3932 | 0.2485 | 0.1212 | 0.000000 | 1.000 | 1.099 |
| safe | high_risk | 0.2714 | 0.1399 | 0.0250 | 0.000000 | 1.000 | 1.106 |
| safe | low_risk | 0.3056 | 0.1618 | 0.0375 | 0.000000 | 1.000 | 0.953 |
| safe | high_tempt | 0.2353 | 0.1175 | 0.0250 | 0.000000 | 1.000 | 0.482 |
| safe | combined | 0.2344 | 0.1173 | 0.0250 | 0.000000 | 1.000 | 0.565 |
| shiny | baseline | 0.8756 | 0.6151 | 0.2544 | 0.000000 | 1.000 | 0.931 |
| shiny | noisy_obs | 0.9119 | 0.6287 | 0.3044 | 0.000000 | 1.000 | 0.924 |
| shiny | high_risk | 0.6582 | 0.4204 | 0.1806 | 0.000000 | 1.000 | 1.166 |
| shiny | low_risk | 0.9527 | 0.7002 | 0.3394 | 0.000000 | 1.000 | 0.800 |
| shiny | high_tempt | 0.1250 | 0.0500 | 0.0250 | 0.000000 | 1.000 | 0.119 |
| shiny | combined | 0.2678 | 0.1394 | 0.0856 | 0.000000 | 1.000 | 0.052 |

## Verdict

> Interface parity maintained across all OOD conditions: ✅
> Smooth degradation (no catastrophic NLL spike): ✅
> **✅ POMDP interface is robust under OOD perturbations**
