# Shadow Observer Results

> Seeds: 15

## Exp A: Targeted Identification

| Target | Family | MAE_τ | Corr_τ | MAE_ν | Corr_ν | MAE_γ | Corr_γ | ADR |
|--------|--------|:-----:|:------:|:-----:|:------:|:-----:|:------:|:---:|
| tau | tic_rescue_heavy+sparse_valid_advice | 0.0772 | 0.7758 | 0.0991 | 0.5369 | 0.0626 | 0.398 | 0.011 |
| nu | sparse_invalid_advice+self_discovery_needed | 0.0792 | 0.0 | 0.1197 | -0.0948 | 0.074 | 0.0 | 0.0 |
| gamma_gen | beneficial_novelty+false_suppression_cost | 0.078 | 0.0 | 0.1188 | 0.0472 | 0.0736 | 0.0 | 0.0 |

## Exp B: Mixed-Family Generalization

| θ | MAE_τ | Corr_τ | MAE_ν | Corr_ν | MAE_γ | Corr_γ | ADR |
|---|:-----:|:------:|:-----:|:------:|:-----:|:------:|:---:|
| safe | 0.1131 | 0.6211 | 0.1273 | 0.3977 | 0.0759 | 0.3876 | 0.0 |
| shiny | 0.0465 | 0.876 | 0.0956 | 0.6318 | 0.0645 | 0.3327 | 0.011 |

## Exp C: Robustness Sweeps

### C1: Noise Sweep

| Noise | MAE_τ | MAE_ν | MAE_γ | ADR |
|:-----:|:-----:|:-----:|:-----:|:---:|
| 0.0 | 0.1159 | 0.1331 | 0.0802 | 0.0 |
| 0.3 | 0.1127 | 0.145 | 0.0833 | 0.0 |
| 0.5 | 0.1124 | 0.1436 | 0.0825 | 0.0 |

### C2: θ Sweep

| θ | MAE_τ | MAE_ν | MAE_γ | ADR |
|---|:-----:|:-----:|:-----:|:---:|
| safe | 0.1131 | 0.1273 | 0.0759 | 0.0 |
| shiny | 0.0465 | 0.0956 | 0.0645 | 0.011 |
