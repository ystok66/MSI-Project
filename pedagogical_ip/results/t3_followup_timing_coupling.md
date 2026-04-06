# T3-Followup: Intervention Timing + Planning Coupling

## Exp-T3-F1: Intervention Timing

| θ | IRH Precision | IRH Recall | Heur Precision | Heur Recall | IRH Lead |
|:-:|:------------:|:----------:|:--------------:|:-----------:|:--------:|
| safe | 0.256 | 1.000 | 0.294 | 0.312 | 14.7 |
| shiny | 0.276 | 1.000 | 0.309 | 0.304 | 13.2 |

## Risk Scores by Family

| θ | Family | Mean p_timeout | Mean p_blind | Mean u_int | Flag Rate |
|:-:|:------:|:--------------:|:------------:|:----------:|:---------:|
| safe | TIC | 0.500 | 0.395 | 0.895 | 1.000 |
| safe | TIC-v4 | 0.500 | 0.255 | 0.755 | 1.000 |
| shiny | TIC | 0.500 | 0.455 | 0.955 | 1.000 |
| shiny | TIC-v4 | 0.500 | 0.273 | 0.773 | 1.000 |

## Exp-T3-F2: Planning Coupling

| θ | Mean Gain | Rank Change Rate | Selective? |
|:-:|:---------:|:----------------:|:----------:|
| safe | 0.0500 | 0.000 | ✅ |
| shiny | 0.0500 | 0.000 | ✅ |

## Verdict

> IRH recall ≥ heuristic: 2/2 θ
> Planning coupling selective: 2/2 θ
> **✅ T3-Followup validates utility of POMDP interfaces**
