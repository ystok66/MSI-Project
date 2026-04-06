# P5-B: 5D No-Score Integration Evaluation

**Observer: 5D (τ̂,ν̂,γ̂_gen,γ̂_spec_state,κ̂) | Micro: 3D view | Tutor: 2-act**

## Exp 1: 5D Micro No-Score Stability

| Suite | θ | DivAll | Div@Act | n_act | Success |
|-------|:-:|:------:|:-------:|:-----:|:-------:|
| Canonical | safe | 0.0033 | 0.1111 | 9 | 0.467 |
| Canonical | shiny | 0.0167 | 0.3571 | 14 | 0.503 |
| Active | safe | 0.0733 | 0.4783 | 46 | 0.483 |
| Active | shiny | 0.0867 | 0.5306 | 49 | 0.500 |

## Exp 2: 5D Macro No-Score Stability

| θ | STOP Agree | Top-1 | Kendall τ | κ̂(final) |
|:-:|:----------:|:-----:|:---------:|:---------:|
| safe | 100.0% | 1 | 1.0000 | 0.2974 |
| shiny | 100.0% | 1 | 1.0000 | 0.2967 |

## Exp 3: κ Observability Replication

### Signal & Orthogonality

| θ | κ̂(final) | σ(κ̂) | Corr(κ̂,τ̂) | Corr(κ̂,ν̂) | Corr(κ̂,γ̂_gen) | Corr(κ̂,γ̂_spec) |
|:-:|:--------:|:----:|:----------:|:----------:|:--------------:|:---------------:|
| safe | 0.2985 | 0.0019 | 0.0000 | -0.2295 | -0.0396 | -0.4396 |
| shiny | 0.2984 | 0.0017 | 0.0000 | 0.1890 | -0.0616 | -0.0816 |

## Verdict

> 5D no-score integration complete. κ̂ is in Layer 1; micro uses 3D view; macro reports full 5D.
