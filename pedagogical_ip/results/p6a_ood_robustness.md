# P6-A: κ̂ Macro Bonus OOD Robustness (β=0.02)

## Micro Protocol A

| Condition | θ | DivAll | Div@Act | OWR | Success |
|-----------|:-:|:------:|:-------:|:---:|:-------:|
| Canonical | safe | 0.0000 | 0.0000 | 0.0000 | 0.533 |
| Canonical | shiny | 0.0125 | 0.2727 | 0.0125 | 0.492 |
| Temptation-rich | safe | 0.0000 | 0.0000 | 0.0000 | 0.533 |
| Temptation-rich | shiny | 0.0125 | 0.2727 | 0.0125 | 0.492 |
| Balanced-Active | safe | 0.0792 | 0.4524 | 0.0792 | 0.475 |
| Balanced-Active | shiny | 0.0667 | 0.4103 | 0.0667 | 0.454 |
| High-risk (×1.5) | safe | 0.0042 | 0.1250 | 0.0042 | 0.512 |
| High-risk (×1.5) | shiny | 0.0125 | 0.2727 | 0.0125 | 0.458 |
| Low-risk (×0.5) | safe | 0.0000 | 0.0000 | 0.0000 | 0.521 |
| Low-risk (×0.5) | shiny | 0.0083 | 0.1818 | 0.0083 | 0.500 |

## Macro Protocol B (κ-Bonus)

| Condition | θ | Risk Shift | Top-1 Same | Kendall | κ̂(final) |
|-----------|:-:|:----------:|:----------:|:-------:|:---------:|
| Canonical | safe | +0.7 | ✅ | 0.9864 | 0.2962 |
| Canonical | shiny | +0.7 | ✅ | 0.9864 | 0.2957 |
| Temptation-rich | safe | +0.7 | ✅ | 0.9864 | 0.2962 |
| Temptation-rich | shiny | +0.7 | ✅ | 0.9864 | 0.2957 |
| Balanced-Active | safe | +0.7 | ✅ | 0.9868 | 0.3180 |
| Balanced-Active | shiny | +0.7 | ✅ | 0.9868 | 0.3113 |
| High-risk (×1.5) | safe | +0.7 | ✅ | 0.9864 | 0.3201 |
| High-risk (×1.5) | shiny | +0.7 | ✅ | 0.9868 | 0.3143 |
| Low-risk (×0.5) | safe | +0.7 | ✅ | 0.9868 | 0.2869 |
| Low-risk (×0.5) | shiny | +0.7 | ✅ | 0.9864 | 0.2862 |

## Verdict

> **Top-1 stable across all OOD**: ✅
> **Risk shifts non-negative**: ✅
> **κ̂ macro bonus (β=0.02) passes OOD robustness.**
