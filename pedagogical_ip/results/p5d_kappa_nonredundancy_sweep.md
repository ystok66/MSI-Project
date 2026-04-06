# P5-D: κ̂ Non-Redundancy Proof + β Sweep

## Part 1: ΔR² — Incremental Explanatory Power

| Model | R² | ΔR² |
|-------|:--:|:---:|
| 4D (τ̂,ν̂,γ̂_gen,γ̂_spec) | 0.013646 | — |
| 5D (+κ̂) | 0.139337 | **+0.125690** |

### Per-Condition ΔR²

| Condition | R²(4D) | R²(5D) | ΔR² |
|-----------|:------:|:------:|:---:|
| safe_none | 0.029441 | 0.163845 | +0.134404 |
| safe_tempt | 0.028397 | 0.155835 | +0.127438 |
| shiny_none | 0.019199 | 0.141678 | +0.122480 |
| shiny_tempt | 0.004226 | 0.123379 | +0.119153 |

## Part 2: Conditional Corr(κ̂, γ̂_spec)

| Condition | Raw Corr | Partial Corr (residualized) |
|-----------|:--------:|:---------------------------:|
| safe_none | -0.3805 | -0.3577 |
| safe_tempt | -0.2813 | -0.2300 |
| shiny_none | 0.0255 | -0.0282 |
| shiny_tempt | -0.2721 | -0.2629 |

## Part 3: Macro κ-Bonus β Sweep

| β | Risk Rank Shift | Top-1 Agree | Kendall τ | STOP Agree |
|:-:|:---------------:|:-----------:|:---------:|:----------:|
| 0.00 | +0.0 | 100% | 1.0000 | 100% |
| 0.02 | +2.3 | 100% | 0.8531 | 100% |
| 0.05 | +2.3 | 100% | 0.8531 | 100% |
| 0.10 | +2.3 | 100% | 0.8531 | 100% |
| 0.20 | +2.3 | 100% | 0.8531 | 100% |

## Verdict

> **ΔR² = +0.125690**: κ̂ adds meaningful incremental explanatory power beyond 4D.

> **β sweep**: See table above for optimal β range.
