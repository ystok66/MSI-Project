# T7-B Exp-T7-B1: Composite Recovery with Compatibility Prior

## Exact Goal Recovery + Subgoal Marginal

| Goal | θ | Config | Goal Acc | Subgoal Acc | NLL | Entropy Δ |
|:----:|:-:|:------:|:--------:|:----------:|:---:|:---------:|
| collect_red | safe | baseline | 0.73 | 0.67 | 0.446 | +0.608 |
| collect_red | safe | compat | 1.00 | 1.00 | 0.443 | +1.488 |
| collect_red | safe | **compat+penalty** | 1.00 | 1.00 | 0.437 | +2.028 |
| collect_red | safe | **strong_compat** | 1.00 | 1.00 | 0.430 | +2.230 |
| collect_red | shiny | baseline | 1.00 | 1.00 | 0.254 | +0.649 |
| collect_red | shiny | compat | 1.00 | 1.00 | 0.248 | +1.251 |
| collect_red | shiny | **compat+penalty** | 1.00 | 1.00 | 0.245 | +1.873 |
| collect_red | shiny | **strong_compat** | 1.00 | 1.00 | 0.226 | +2.152 |
| use_safe | safe | baseline | 0.60 | 1.00 | 0.203 | +0.180 |
| use_safe | safe | compat | 0.63 | 1.00 | 0.199 | +0.289 |
| use_safe | safe | **compat+penalty** | 0.61 | 1.00 | 0.218 | +0.889 |
| use_safe | safe | **strong_compat** | 0.60 | 1.00 | 0.221 | +0.940 |
| use_safe | shiny | baseline | 0.60 | 1.00 | 0.203 | +0.180 |
| use_safe | shiny | compat | 0.63 | 1.00 | 0.199 | +0.289 |
| use_safe | shiny | **compat+penalty** | 0.61 | 1.00 | 0.218 | +0.889 |
| use_safe | shiny | **strong_compat** | 0.60 | 1.00 | 0.221 | +0.940 |
| collect_red+avoid_blue | safe | baseline | 0.00 | 0.50 | 0.226 | +0.244 |
| collect_red+avoid_blue | safe | compat | 0.00 | 0.50 | 0.224 | +0.307 |
| collect_red+avoid_blue | safe | **compat+penalty** | 0.00 | 0.50 | 0.228 | +0.967 |
| collect_red+avoid_blue | safe | **strong_compat** | 0.00 | 0.50 | 0.229 | +1.090 |
| collect_red+avoid_blue | shiny | baseline | 0.32 | 0.82 | 0.455 | +0.499 |
| collect_red+avoid_blue | shiny | compat | 0.37 | 0.81 | 0.460 | +0.623 |
| collect_red+avoid_blue | shiny | **compat+penalty** | 0.00 | 0.72 | 0.464 | +1.167 |
| collect_red+avoid_blue | shiny | **strong_compat** | 0.00 | 0.71 | 0.470 | +1.420 |
| collect_red+use_safe | safe | baseline | 0.13 | 0.50 | 0.186 | +0.131 |
| collect_red+use_safe | safe | compat | 0.03 | 0.50 | 0.185 | +0.152 |
| collect_red+use_safe | safe | **compat+penalty** | 0.00 | 0.47 | 0.190 | +0.856 |
| collect_red+use_safe | safe | **strong_compat** | 0.00 | 0.47 | 0.190 | +0.955 |
| collect_red+use_safe | shiny | baseline | 0.03 | 0.50 | 0.243 | +0.118 |
| collect_red+use_safe | shiny | compat | 0.00 | 0.50 | 0.243 | +0.143 |
| collect_red+use_safe | shiny | **compat+penalty** | 0.00 | 0.47 | 0.248 | +0.817 |
| collect_red+use_safe | shiny | **strong_compat** | 0.00 | 0.47 | 0.250 | +0.948 |
| avoid_blue+use_safe | safe | baseline | 0.00 | 1.00 | 0.213 | +0.273 |
| avoid_blue+use_safe | safe | compat | 0.00 | 1.00 | 0.208 | +0.369 |
| avoid_blue+use_safe | safe | **compat+penalty** | 0.00 | 1.00 | 0.227 | +0.900 |
| avoid_blue+use_safe | safe | **strong_compat** | 0.00 | 1.00 | 0.229 | +0.941 |
| avoid_blue+use_safe | shiny | baseline | 0.00 | 1.00 | 0.222 | +0.267 |
| avoid_blue+use_safe | shiny | compat | 0.00 | 1.00 | 0.213 | +0.411 |
| avoid_blue+use_safe | shiny | **compat+penalty** | 0.00 | 1.00 | 0.231 | +0.928 |
| avoid_blue+use_safe | shiny | **strong_compat** | 0.00 | 1.00 | 0.233 | +0.968 |
| reach_fast+avoid_blue | safe | baseline | 0.00 | 0.50 | 0.225 | +0.490 |
| reach_fast+avoid_blue | safe | compat | 0.00 | 0.50 | 0.223 | +0.533 |
| reach_fast+avoid_blue | safe | **compat+penalty** | 0.00 | 0.50 | 0.226 | +1.121 |
| reach_fast+avoid_blue | safe | **strong_compat** | 0.00 | 0.50 | 0.220 | +1.286 |
| reach_fast+avoid_blue | shiny | baseline | 0.64 | 0.80 | 0.621 | +0.920 |
| reach_fast+avoid_blue | shiny | compat | 0.43 | 0.81 | 0.630 | +1.234 |
| reach_fast+avoid_blue | shiny | **compat+penalty** | 0.00 | 0.73 | 0.635 | +1.671 |
| reach_fast+avoid_blue | shiny | **strong_compat** | 0.00 | 0.74 | 0.646 | +1.944 |

## Verdict

> baseline composite goal_acc: 0.033
> compat+penalty composite goal_acc: 0.000
> baseline composite subgoal_acc: 0.625
> compat+penalty composite subgoal_acc: 0.617
