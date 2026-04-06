# T1 Exp-3: Over-Warn Fix — WAIT-Preferred Dead-Zone

## ε_Q Sweep Results

| ε_Q | DivAll_raw | DivAll_dz | OWR_raw | OWR_dz | WarnNecRecall_raw | WarnNecRecall_dz |
|:---:|:----------:|:---------:|:-------:|:------:|:-----------------:|:----------------:|
| 0.00 | 0.0083 | 0.0083 | 0.0083 | 0.0083 | 1.0000 | 1.0000 |
| 0.05 | 0.0083 | 0.0050 | 0.0083 | 0.0050 | 1.0000 | 1.0000 |
| 0.08 | 0.0083 | 0.0017 | 0.0083 | 0.0000 | 1.0000 | 0.9444 |
| 0.10 | 0.0083 | 0.0033 | 0.0083 | 0.0000 | 1.0000 | 0.8889 |
| 0.12 | 0.0083 | 0.0033 | 0.0083 | 0.0000 | 1.0000 | 0.8889 |
| 0.15 | 0.0083 | 0.0033 | 0.0083 | 0.0000 | 1.0000 | 0.8889 |

## Per-Family Detail at ε_Q = 0.10

| Family | OWR_raw | OWR_dz | WNR_raw | WNR_dz |
|--------|:-------:|:------:|:-------:|:------:|
| beneficial_novelty | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| blind_activation_corridor **←** | 0.1000 | 0.0000 | 1.0000 | 0.8571 |
| false_suppression | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| ppmrb_self_discovery | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| ppmrb_standard | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| soft_boundary_tradeoff | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| sparse_invalid_advice | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| sparse_valid_advice | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| tic_rescue_heavy **←** | 0.0000 | 0.0000 | 1.0000 | 0.8750 |
| tic_self_discovery | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| tic_temptation | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| verified_warn | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| warn_symmetric_rescue **←** | 0.0667 | 0.0000 | 1.0000 | 1.0000 |

## Verdict

> OWR reduction at ε=0.10: 5 → 0
> WarnNecRecall: 1.0000 → 0.8889
> **⚠️ Check WarnNecRecall constraint**
