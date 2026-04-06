# T2 Exp-2A: Persistent vs Reset — PP-MRB


## θ = safe

### WarnRate by Session

| Session | Mode | Overall | wait_clean | wait_lure | boundary | warn_trap | SelGap |
|:-------:|:----:|:-------:|:----------:|:---------:|:--------:|:---------:|:------:|
| 0 | reset | 0.045 | 0.000 | 0.000 | 0.000 | 0.145 | 0.145 |
| 0 | persistent | 0.045 | 0.000 | 0.000 | 0.000 | 0.145 | 0.145 |
| 1 | reset | 0.035 | 0.000 | 0.000 | 0.000 | 0.092 | 0.092 |
| 1 | persistent | 0.035 | 0.000 | 0.000 | 0.000 | 0.092 | 0.092 |
| 2 | reset | 0.020 | 0.000 | 0.000 | 0.000 | 0.065 | 0.065 |
| 2 | persistent | 0.015 | 0.000 | 0.000 | 0.000 | 0.065 | 0.065 |
| 3 | reset | 0.025 | 0.000 | 0.000 | 0.000 | 0.072 | 0.072 |
| 3 | persistent | 0.020 | 0.000 | 0.000 | 0.000 | 0.072 | 0.072 |
| 4 | reset | 0.050 | 0.000 | 0.000 | 0.000 | 0.139 | 0.139 |
| 4 | persistent | 0.040 | 0.000 | 0.000 | 0.000 | 0.139 | 0.139 |

## θ = shiny

### WarnRate by Session

| Session | Mode | Overall | wait_clean | wait_lure | boundary | warn_trap | SelGap |
|:-------:|:----:|:-------:|:----------:|:---------:|:--------:|:---------:|:------:|
| 0 | reset | 0.045 | 0.000 | 0.000 | 0.000 | 0.145 | 0.145 |
| 0 | persistent | 0.045 | 0.000 | 0.000 | 0.000 | 0.145 | 0.145 |
| 1 | reset | 0.035 | 0.000 | 0.000 | 0.000 | 0.092 | 0.092 |
| 1 | persistent | 0.035 | 0.000 | 0.000 | 0.000 | 0.092 | 0.092 |
| 2 | reset | 0.020 | 0.000 | 0.000 | 0.000 | 0.065 | 0.065 |
| 2 | persistent | 0.015 | 0.000 | 0.000 | 0.000 | 0.065 | 0.065 |
| 3 | reset | 0.020 | 0.000 | 0.000 | 0.000 | 0.072 | 0.072 |
| 3 | persistent | 0.020 | 0.000 | 0.000 | 0.000 | 0.072 | 0.072 |
| 4 | reset | 0.045 | 0.000 | 0.000 | 0.000 | 0.139 | 0.139 |
| 4 | persistent | 0.040 | 0.000 | 0.000 | 0.000 | 0.139 | 0.139 |

## SelGap Trend Summary

| θ | Mode | Session 0 | Session 2 | Session 4 | Trend |
|:-:|:----:|:---------:|:---------:|:---------:|:-----:|
| safe | reset | 0.145 | 0.065 | 0.139 | → |
| safe | persistent | 0.145 | 0.065 | 0.139 | → |
| shiny | reset | 0.145 | 0.065 | 0.139 | → |
| shiny | persistent | 0.145 | 0.065 | 0.139 | → |

## Talk Less After Learning

Δ WarnRate(wait_clean) from Session 0 → Session 4:

| θ | Reset Δ | Persistent Δ | Persistent Better? |
|:-:|:-------:|:------------:|:------------------:|
| safe | +0.000 | +0.000 | ≈ |
| shiny | +0.000 | +0.000 | ≈ |

## Verdict

> SelGap maintained or improved: 2/2 θ
> warn_trap collapse: ⚠️ YES
> **⚠️ Persistent advantage not clear — investigate**
