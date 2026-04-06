# T5 Exp-T5-1: Hidden Temptation Audit

## Action NLL by True Temptation Level

| θ | True z | Mean NLL | Entropy Δ | E[z]_final | MAP z |
|:-:|:------:|:--------:|:---------:|:----------:|:-----:|
| safe | 0.0 | 0.125 | -0.001 | 0.300 | 0.0 |
| safe | 0.3 | 0.110 | -0.001 | 0.301 | 0.0 |
| safe | 0.6 | 0.110 | -0.001 | 0.301 | 0.0 |
| safe | 0.9 | 0.110 | -0.001 | 0.301 | 0.0 |
| shiny | 0.0 | 0.804 | +0.984 | 0.086 | 0.0 |
| shiny | 0.3 | 0.946 | +0.413 | 0.477 | 0.3 |
| shiny | 0.6 | 0.968 | +0.322 | 0.521 | 0.3 |
| shiny | 0.9 | 0.968 | +0.322 | 0.521 | 0.3 |

## Posterior Calibration: P(MAP = true z)

| θ | True z | P(MAP=true) | P(MAP=±0.3) |
|:-:|:------:|:-----------:|:-----------:|
| safe | 0.0 | 1.00 | 1.00 |
| safe | 0.3 | 0.00 | 1.00 |
| safe | 0.6 | 0.00 | 0.00 |
| safe | 0.9 | 0.00 | 0.00 |
| shiny | 0.0 | 0.70 | 1.00 |
| shiny | 0.3 | 0.70 | 1.00 |
| shiny | 0.6 | 0.50 | 1.00 |
| shiny | 0.9 | 0.00 | 0.50 |

## Verdict

> NLL worsens smoothly with z_tempt (shiny): ✅
> Entropy reduces for safe θ, z=0.6: ❌
> Entropy reduces for shiny θ, z=0.6: ✅
