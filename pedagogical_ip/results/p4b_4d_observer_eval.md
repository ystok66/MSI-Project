# P4-B: 4D Observer Formal Evaluation

**Observer: 4D (τ̂, ν̂, γ̂_gen, γ̂_spec_state) | Tutor: 2-act canonical**

## Exp 1: 4D Micro Infer-Only

| Suite | θ | tempt | Div All | Div@Active | n_active | Success |
|-------|:-:|:-----:|:-------:|:----------:|:--------:|:-------:|
| Canonical | safe | none | 0.0133 | 0.0000 | 5 | 0.473 |
| Canonical | shiny | none | 0.0133 | 0.0000 | 5 | 0.553 |
| Balanced Active | safe | none | 0.0700 | 0.0000 | 26 | 0.523 |
| Balanced Active | shiny | none | 0.0867 | 0.0000 | 25 | 0.447 |
| Temptation | safe | none | 0.0133 | 0.0000 | 5 | 0.473 |
| Temptation | safe | aligned=0.6 | 0.0033 | 0.0000 | 6 | 0.760 |
| Temptation | safe | conflict=1.0 | 0.0033 | 0.1250 | 8 | 0.887 |
| Temptation | shiny | none | 0.0133 | 0.0000 | 5 | 0.553 |
| Temptation | shiny | aligned=0.6 | 0.0167 | 0.0000 | 4 | 0.067 |
| Temptation | shiny | conflict=1.0 | 0.0167 | 0.0000 | 4 | 0.053 |

## Exp 2: 4D Macro Hybrid

| θ | α | STOP Agree | Top-1 | Kendall τ |
|:-:|:-:|:----------:|:-----:|:---------:|
| safe | 0.5 | 0.960 | 1 | 0.9864 |
| safe | 1.0 | 0.960 | 1 | 0.9864 |
| shiny | 0.5 | 1.000 | 1 | 1.0000 |
| shiny | 1.0 | 1.000 | 1 | 1.0000 |

## Exp 3: ν Contamination (4D)

| θ | ν̂(t=0) | ν̂(t=0.6) | ν̂(t=1.0) | Δν̂(0→1) |
|:-:|:------:|:--------:|:--------:|:--------:|
| safe | 0.0590 | 0.0348 | 0.0473 | -0.0117 |
| shiny | 0.0559 | 0.0992 | 0.1019 | 0.0460 |

## Exp 4: State Semantics (γ̂_spec vs Resist Rate)

| θ | tempt | γ̂_spec(final) | r_resist | per-tempt Corr |
|:-:|:-----:|:-------------:|:--------:|:--------------:|
| safe | 0.0 | 0.1590 | 0.4733 | 0.8625 |
| safe | 0.3 | 0.2619 | 0.5933 | 0.9058 |
| safe | 0.6 | 0.4098 | 0.7600 | 0.9306 |
| safe | 1.0 | 0.5733 | 0.8867 | 0.9247 |

**θ=safe overall: Corr = 0.9592 (p=0.0000)**

| shiny | 0.0 | 0.1802 | 0.5533 | 0.9612 |
| shiny | 0.3 | 0.1035 | 0.2233 | 0.9902 |
| shiny | 0.6 | 0.0384 | 0.0667 | 0.9712 |
| shiny | 1.0 | 0.0380 | 0.0533 | 0.9449 |

**θ=shiny overall: Corr = 0.9654 (p=0.0000)**


## Verdict

> 4D observer formal evaluation complete. See metrics above for pass/fail on each experiment.
