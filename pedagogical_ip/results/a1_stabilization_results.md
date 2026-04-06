# A1 Stabilization & Blind Channel Audit

## Exp A: p_self Activation Audit (A1)

| Metric | Value |
|--------|------:|
| warn_rate | 0.0174 |
| follow_warn_rate | 0.0104 |
| dose>0_rate | 0.0174 |
| mean_blind (A1) | 0.009923 |
| mean_selfdisc | 0.3675 |
| mean_p_self | 0.8021 |
| n_blind>0 | 3 / 288 |

## Exp B: Blind Definition — A1 vs A2

| Observer | MAE_ν | Corr_ν | Blind>0 | Mean blind | ADR |
|----------|:-----:|:------:|:-------:|:----------:|:---:|
| A1 | 0.0178 | 0.7903 | 3 / 288 | 0.009923 | 0.0 |
| A2 | 0.0178 | 0.7903 | 3 / 288 | 0.009923 | 0.0 |

## Exp C: Confidence Calibration (A0 vs A1 vs A2)

| Observer | Dim | Corr(conf, −|err|) |
|----------|-----|:---:|
| A0 | tau | -0.6012 |
| A0 | nu | -0.4887 |
| A0 | gamma_gen | -0.4895 |
| A1 | tau | -0.2346 |
| A1 | nu | 0.0443 |
| A1 | gamma_gen | -0.0577 |
| A2 | tau | -0.2422 |
| A2 | nu | 0.0113 |
| A2 | gamma_gen | -0.0882 |

## Exp D: Hybrid Dry-Run (A1)

| α | MAE_τ | MAE_ν | MAE_γ | micro_ADR | Δε_stop |
|:-:|:-----:|:-----:|:-----:|:---------:|:-------:|
| 0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0000 |
| 0.25 | 0.0006 | 0.0045 | 0.0018 | 0.0 | 0.0008 |
| 0.5 | 0.0012 | 0.0089 | 0.0036 | 0.0 | 0.0017 |
| 0.75 | 0.0018 | 0.0134 | 0.0055 | 0.0 | 0.0025 |
| 1.0 | 0.0024 | 0.0178 | 0.0073 | 0.0 | 0.0034 |
