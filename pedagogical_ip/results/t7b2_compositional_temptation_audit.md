# T7-B2: Compositional + Temptation Posterior

## Exp-T7-B2.1: q(g,θ,z) Stability

| Goal | θ | z_true | NLL | Subgoal Acc | Goal Acc | E[z] | Entropy Δ |
|:----:|:-:|:------:|:---:|:----------:|:--------:|:----:|:---------:|
| collect_red | safe | 0.0 | 0.444 | 0.93 | 0.93 | 0.04 | +3.016 |
| collect_red | safe | 0.6 | 0.862 | 0.00 | 0.00 | 0.28 | +2.175 |
| collect_red | shiny | 0.0 | 0.269 | 0.93 | 0.93 | 0.03 | +3.173 |
| collect_red | shiny | 0.6 | 1.069 | 0.00 | 0.00 | 0.55 | +2.258 |
| use_safe | safe | 0.0 | 0.214 | 1.00 | 0.95 | 0.27 | +0.994 |
| use_safe | safe | 0.6 | 0.214 | 1.00 | 0.95 | 0.27 | +0.994 |
| use_safe | shiny | 0.0 | 0.214 | 1.00 | 0.95 | 0.27 | +0.994 |
| use_safe | shiny | 0.6 | 0.251 | 1.00 | 0.65 | 0.28 | +1.055 |
| collect_red+avoid_blue | safe | 0.0 | 0.221 | 0.50 | 0.00 | 0.23 | +1.324 |
| collect_red+avoid_blue | safe | 0.6 | 0.205 | 0.50 | 0.00 | 0.23 | +1.337 |
| collect_red+avoid_blue | shiny | 0.0 | 0.459 | 0.67 | 0.00 | 0.17 | +1.711 |
| collect_red+avoid_blue | shiny | 0.6 | 0.806 | 0.50 | 0.00 | 0.51 | +2.923 |
| avoid_blue+use_safe | safe | 0.0 | 0.218 | 1.00 | 0.00 | 0.25 | +1.072 |
| avoid_blue+use_safe | safe | 0.6 | 0.218 | 1.00 | 0.00 | 0.25 | +1.072 |
| avoid_blue+use_safe | shiny | 0.0 | 0.221 | 1.00 | 0.00 | 0.25 | +1.099 |
| avoid_blue+use_safe | shiny | 0.6 | 0.536 | 0.99 | 0.00 | 0.42 | +1.872 |
| reach_fast+avoid_blue | safe | 0.0 | 0.204 | 0.50 | 0.00 | 0.28 | +1.365 |
| reach_fast+avoid_blue | safe | 0.6 | 0.204 | 0.50 | 0.00 | 0.28 | +1.365 |
| reach_fast+avoid_blue | shiny | 0.0 | 0.640 | 0.65 | 0.00 | 0.06 | +2.558 |
| reach_fast+avoid_blue | shiny | 0.6 | 0.463 | 0.50 | 0.00 | 0.43 | +2.520 |

## Exp-T7-B2.2: q(g,θ) vs q(g,θ,z) Ablation

| Goal | θ | z_true | Tempt? | NLL | Subgoal Acc | Goal Acc |
|:----:|:-:|:------:|:------:|:---:|:----------:|:--------:|
| collect_red+avoid_blue | shiny | 0.0 | No | 0.464 | 0.72 | 0.00 |
| collect_red+avoid_blue | shiny | 0.0 | Yes | 0.459 | 0.67 | 0.00 |
| collect_red+avoid_blue | shiny | 0.6 | No | 0.717 | 0.50 | 0.00 |
| collect_red+avoid_blue | shiny | 0.6 | Yes | 0.806 | 0.50 | 0.00 |
| collect_red+avoid_blue | shiny | 0.9 | No | 0.717 | 0.50 | 0.00 |
| collect_red+avoid_blue | shiny | 0.9 | Yes | 0.806 | 0.50 | 0.00 |
| reach_fast+avoid_blue | shiny | 0.0 | No | 0.635 | 0.73 | 0.00 |
| reach_fast+avoid_blue | shiny | 0.0 | Yes | 0.640 | 0.65 | 0.00 |
| reach_fast+avoid_blue | shiny | 0.6 | No | 0.343 | 0.50 | 0.00 |
| reach_fast+avoid_blue | shiny | 0.6 | Yes | 0.463 | 0.50 | 0.00 |
| reach_fast+avoid_blue | shiny | 0.9 | No | 0.312 | 0.50 | 0.00 |
| reach_fast+avoid_blue | shiny | 0.9 | Yes | 0.438 | 0.50 | 0.00 |

## Exp-T7-B2.3: Θ₂ vs Θ_K under Compositional + Temptation

| Goal | θ | z | Types | NLL | Subgoal | Goal | E[z] |
|:----:|:-:|:-:|:-----:|:---:|:------:|:----:|:----:|
| collect_red | safe | 0.3 | Θ₂ | 0.712 | 0.73 | 0.73 | 0.24 |
| collect_red | safe | 0.3 | Θ_K | 0.712 | 0.80 | 0.72 | 0.24 |
| collect_red | shiny | 0.3 | Θ₂ | 0.876 | 0.35 | 0.33 | 0.30 |
| collect_red | shiny | 0.3 | Θ_K | 0.777 | 0.45 | 0.36 | 0.28 |
| collect_red+avoid_blue | safe | 0.3 | Θ₂ | 0.205 | 0.50 | 0.00 | 0.23 |
| collect_red+avoid_blue | safe | 0.3 | Θ_K | 0.222 | 0.50 | 0.00 | 0.29 |
| collect_red+avoid_blue | shiny | 0.3 | Θ₂ | 0.816 | 0.50 | 0.00 | 0.33 |
| collect_red+avoid_blue | shiny | 0.3 | Θ_K | 0.666 | 0.50 | 0.00 | 0.37 |

## Verdict

> z=0.0: q(g,θ) NLL=0.464 | q(g,θ,z) NLL=0.459 | Δ=-0.005
> z=0.6: q(g,θ) NLL=0.717 | q(g,θ,z) NLL=0.806 | Δ=+0.088
> Overall subgoal marginal acc (q(g,θ,z)): 0.747
> Mean final entropy (should be >0): 2.722
> Collapse check: ✅ No collapse
