# T4 Exp-T4-1/2: Family Selectivity + Mixed Audit

## Exp-T4-1: Family-Selective Lever

| θ | Family | Primary | P(primary) | P(WARN) | P(UNLOCK) | P(ITEM) | SelGap |
|:-:|:------:|:-------:|:----------:|:-------:|:---------:|:-------:|:------:|
| safe | fork_trap | WARN | 0.630 | 0.630 | 0.000 | 0.000 | +0.580 |
| safe | hazard_belt | ITEM_DROP | 0.645 | 0.000 | 0.000 | 0.645 | +0.645 |
| safe | deadline_gate | UNLOCK | 0.150 | 0.100 | 0.150 | 0.000 | +0.150 |
| shiny | fork_trap | WARN | 0.455 | 0.455 | 0.000 | 0.000 | +0.415 |
| shiny | hazard_belt | ITEM_DROP | 0.505 | 0.000 | 0.000 | 0.505 | +0.505 |
| shiny | deadline_gate | UNLOCK | 0.150 | 0.080 | 0.150 | 0.000 | +0.150 |

## Exp-T4-2: Mixed-Family Comparison

| θ | Strategy | fork_trap SR | hazard_belt SR | deadline_gate SR | Mean SR |
|:-:|:--------:|:----------:|:-------------:|:---------------:|:-------:|
| safe | always_WARN | 0.975 | 0.970 | 0.970 | 0.972 |
| safe | always_UNLOCK | 0.975 | 0.970 | 0.970 | 0.972 |
| safe | always_ITEM_DROP | 0.975 | 0.970 | 0.970 | 0.972 |
| safe | always_NONE | 0.975 | 0.970 | 0.970 | 0.972 |
| safe | **option_ctrl** | 0.975 | 0.970 | 0.970 | **0.972** |
| shiny | always_WARN | 0.275 | 0.500 | 0.525 | 0.433 |
| shiny | always_UNLOCK | 0.275 | 0.500 | 0.525 | 0.433 |
| shiny | always_ITEM_DROP | 0.275 | 0.500 | 0.525 | 0.433 |
| shiny | always_NONE | 0.275 | 0.500 | 0.525 | 0.433 |
| shiny | **option_ctrl** | 0.275 | 0.500 | 0.525 | **0.433** |

## Warning Subtype Distribution (shadow)

| θ | Family | hint | alert | explain | directive |
|:-:|:------:|:----:|:-----:|:-------:|:---------:|
| safe | fork_trap | 0.000 | 0.921 | 0.079 | 0.000 |
| safe | hazard_belt | 0.000 | 0.000 | 0.000 | 0.000 |
| safe | deadline_gate | 1.000 | 0.000 | 0.000 | 0.000 |
| shiny | fork_trap | 0.000 | 0.890 | 0.110 | 0.000 |
| shiny | hazard_belt | 0.000 | 0.000 | 0.000 | 0.000 |
| shiny | deadline_gate | 1.000 | 0.000 | 0.000 | 0.000 |

## Verdict

