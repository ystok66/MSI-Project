# T4 Exp-T4-3: Inflation / Transfer Audit

## Δν̂ and Δγ̂_gen by Strategy

| θ | Family | Strategy | SR | Δν̂ | Δγ̂_gen | Final ν̂ | Final γ̂_gen |
|:-:|:------:|:--------:|:--:|:---:|:------:|:------:|:----------:|
| safe | fork_trap | NONE | 0.960 | -0.0980 | +0.0000 | 0.002 | 0.000 |
| safe | fork_trap | WARN | 0.960 | +0.7000 | +0.5000 | 0.800 | 0.500 |
| safe | fork_trap | UNLOCK | 0.960 | +0.0000 | +0.0000 | 0.100 | 0.000 |
| safe | fork_trap | ITEM_DROP | 0.960 | +0.7000 | +0.0000 | 0.800 | 0.000 |
| safe | fork_trap | **option_ctrl** | 0.960 | +0.2514 | +0.2104 | 0.351 | 0.210 |
| safe | hazard_belt | NONE | 0.956 | -0.0979 | +0.0000 | 0.002 | 0.000 |
| safe | hazard_belt | WARN | 0.956 | +0.7000 | +0.5000 | 0.800 | 0.500 |
| safe | hazard_belt | UNLOCK | 0.956 | +0.0000 | +0.0000 | 0.100 | 0.000 |
| safe | hazard_belt | ITEM_DROP | 0.956 | +0.7000 | +0.0000 | 0.800 | 0.000 |
| safe | hazard_belt | **option_ctrl** | 0.956 | +0.2910 | +0.0000 | 0.391 | 0.000 |
| safe | deadline_gate | NONE | 0.956 | -0.0979 | +0.0000 | 0.002 | 0.000 |
| safe | deadline_gate | WARN | 0.956 | +0.7000 | +0.5000 | 0.800 | 0.500 |
| safe | deadline_gate | UNLOCK | 0.956 | +0.0000 | +0.0000 | 0.100 | 0.000 |
| safe | deadline_gate | ITEM_DROP | 0.956 | +0.7000 | +0.0000 | 0.800 | 0.000 |
| safe | deadline_gate | **option_ctrl** | 0.956 | +0.0023 | +0.0512 | 0.102 | 0.051 |
| shiny | fork_trap | NONE | 0.300 | -0.0699 | +0.0000 | 0.030 | 0.000 |
| shiny | fork_trap | WARN | 0.300 | +0.7000 | +0.5000 | 0.800 | 0.500 |
| shiny | fork_trap | UNLOCK | 0.300 | +0.0000 | +0.0000 | 0.100 | 0.000 |
| shiny | fork_trap | ITEM_DROP | 0.300 | +0.7000 | +0.0000 | 0.800 | 0.000 |
| shiny | fork_trap | **option_ctrl** | 0.300 | +0.3484 | +0.2386 | 0.448 | 0.239 |
| shiny | hazard_belt | NONE | 0.516 | -0.0872 | +0.0000 | 0.013 | 0.000 |
| shiny | hazard_belt | WARN | 0.516 | +0.7000 | +0.5000 | 0.800 | 0.500 |
| shiny | hazard_belt | UNLOCK | 0.516 | +0.0000 | +0.0000 | 0.100 | 0.000 |
| shiny | hazard_belt | ITEM_DROP | 0.516 | +0.7000 | +0.0000 | 0.800 | 0.000 |
| shiny | hazard_belt | **option_ctrl** | 0.516 | +0.3041 | +0.0000 | 0.404 | 0.000 |
| shiny | deadline_gate | NONE | 0.536 | -0.0878 | +0.0000 | 0.012 | 0.000 |
| shiny | deadline_gate | WARN | 0.536 | +0.7000 | +0.5000 | 0.800 | 0.500 |
| shiny | deadline_gate | UNLOCK | 0.536 | +0.0000 | +0.0000 | 0.100 | 0.000 |
| shiny | deadline_gate | ITEM_DROP | 0.536 | +0.7000 | +0.0000 | 0.800 | 0.000 |
| shiny | deadline_gate | **option_ctrl** | 0.536 | -0.0298 | +0.0292 | 0.070 | 0.029 |

## Verdict

> Δν̂ < 0.15: ❌
> Δγ̂_gen < 0.10: ❌
> **⚠️ Inflation check failed — needs weight tuning**
