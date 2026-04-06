# T5 Exp-T5-3+4: Grounded Mixed-Family + Inflation Decomposition

## Exp-T5-3: Mixed-Family Success Rate (Grounded)

| θ | Strategy | fork_trap SR | hazard_belt SR | deadline_gate SR | Mean SR |
|:-:|:--------:|:----------:|:-------------:|:---------------:|:-------:|
| safe | NONE | 0.960 | 0.956 | 0.956 | 0.957 |
| safe | WARN | 0.964 | 0.960 | 0.956 | 0.960 |
| safe | UNLOCK | 0.960 | 0.956 | 0.956 | 0.957 |
| safe | ITEM_DROP | 0.960 | 0.956 | 0.948 | 0.955 |
| safe | **option_ctrl** | 0.960 | 0.956 | 0.956 | **0.957** |
| shiny | NONE | 0.300 | 0.516 | 0.536 | 0.451 |
| shiny | WARN | 0.424 | 0.608 | 0.536 | 0.523 |
| shiny | UNLOCK | 0.300 | 0.516 | 0.536 | 0.451 |
| shiny | ITEM_DROP | 0.164 | 0.416 | 0.528 | 0.369 |
| shiny | **option_ctrl** | 0.324 | 0.500 | 0.536 | **0.453** |

## Exp-T5-4: Inflation Decomposition

| θ | Family | Strategy | Δν̂ | n_int | Δν̂/n_int | Δγ̂_gen | Δγ̂/n_int |
|:-:|:------:|:--------:|:---:|:----:|:--------:|:------:|:--------:|
| safe | fork_trap | NONE | -0.098 | 0.0 | -0.0980 | +0.000 | +0.0000 |
| safe | fork_trap | WARN | +0.700 | 25.0 | +0.0280 | +0.500 | +0.0200 |
| safe | fork_trap | ITEM_DROP | +0.700 | 25.0 | +0.0280 | +0.000 | +0.0000 |
| safe | fork_trap | **option_ctrl** | +0.251 | 9.0 | +0.0279 | +0.210 | +0.0234 |
| safe | hazard_belt | NONE | -0.098 | 0.0 | -0.0979 | +0.000 | +0.0000 |
| safe | hazard_belt | WARN | +0.700 | 25.0 | +0.0280 | +0.500 | +0.0200 |
| safe | hazard_belt | ITEM_DROP | +0.700 | 25.0 | +0.0280 | +0.000 | +0.0000 |
| safe | hazard_belt | **option_ctrl** | +0.291 | 8.5 | +0.0339 | +0.000 | +0.0000 |
| safe | deadline_gate | NONE | -0.098 | 0.0 | -0.0979 | +0.000 | +0.0000 |
| safe | deadline_gate | WARN | +0.700 | 25.0 | +0.0280 | +0.500 | +0.0200 |
| safe | deadline_gate | ITEM_DROP | +0.700 | 25.0 | +0.0280 | +0.000 | +0.0000 |
| safe | deadline_gate | **option_ctrl** | +0.002 | 5.0 | +0.0005 | +0.051 | +0.0102 |
| shiny | fork_trap | NONE | -0.070 | 0.0 | -0.0699 | +0.000 | +0.0000 |
| shiny | fork_trap | WARN | +0.700 | 25.0 | +0.0280 | +0.500 | +0.0200 |
| shiny | fork_trap | ITEM_DROP | +0.700 | 25.0 | +0.0280 | +0.000 | +0.0000 |
| shiny | fork_trap | **option_ctrl** | +0.343 | 5.6 | +0.0620 | +0.234 | +0.0423 |
| shiny | hazard_belt | NONE | -0.087 | 0.0 | -0.0872 | +0.000 | +0.0000 |
| shiny | hazard_belt | WARN | +0.700 | 25.0 | +0.0280 | +0.500 | +0.0200 |
| shiny | hazard_belt | ITEM_DROP | +0.700 | 25.0 | +0.0280 | +0.000 | +0.0000 |
| shiny | hazard_belt | **option_ctrl** | +0.282 | 6.1 | +0.0464 | +0.000 | +0.0000 |
| shiny | deadline_gate | NONE | -0.088 | 0.0 | -0.0878 | +0.000 | +0.0000 |
| shiny | deadline_gate | WARN | +0.700 | 25.0 | +0.0280 | +0.500 | +0.0200 |
| shiny | deadline_gate | ITEM_DROP | +0.700 | 25.0 | +0.0280 | +0.000 | +0.0000 |
| shiny | deadline_gate | **option_ctrl** | -0.030 | 4.0 | -0.0074 | +0.029 | +0.0073 |

## Verdict

> option_ctrl SR (shiny): 0.453
> Best baseline: WARN = 0.523
> Worst baseline: ITEM_DROP = 0.369
> Beats worst baseline: ✅
> Beats best baseline: ⚠️ (within tolerance)
