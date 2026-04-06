# T7 Exp-T7-1+2: Compositional Goal + Multi-Type Posterior

## Exp-T7-1: Single-Goal vs Compositional Posterior

| True Goal | θ | Space | Mean NLL | Entropy Δ | Goal Acc (last 5) |
|:---------:|:-:|:-----:|:--------:|:---------:|:-----------------:|
| use_safe | safe | atomic | 0.258 | +0.276 | 0.52 |
| use_safe | safe | full | 0.235 | +0.160 | 0.52 |
| use_safe | shiny | atomic | 0.258 | +0.276 | 0.52 |
| use_safe | shiny | full | 0.235 | +0.160 | 0.52 |
| collect_red | safe | atomic | 0.439 | +0.555 | 0.80 |
| collect_red | safe | full | 0.444 | +0.580 | 0.74 |
| collect_red | shiny | atomic | 0.312 | +0.518 | 1.00 |
| collect_red | shiny | full | 0.302 | +0.585 | 1.00 |
| collect_red+avoid_blue | safe | full | 0.254 | +0.207 | 0.00 |
| collect_red+avoid_blue | shiny | full | 0.474 | +0.454 | 0.28 |
| reach_fast+avoid_blue | safe | full | 0.267 | +0.419 | 0.00 |
| reach_fast+avoid_blue | shiny | full | 0.609 | +0.720 | 0.56 |

## Exp-T7-2: 2-Type vs K-Type Posterior

| True Goal | θ | Types | Mean NLL | Entropy | Goal Acc | Pref Acc |
|:---------:|:-:|:-----:|:--------:|:-------:|:--------:|:--------:|
| use_safe | safe | Θ₂ | 0.235 | 2.612 | 0.52 | 1.00 |
| use_safe | safe | Θ_K | 0.235 | 3.473 | 0.36 | 1.00 |
| use_safe | shiny | Θ₂ | 0.235 | 2.612 | 0.52 | 0.00 |
| use_safe | shiny | Θ_K | 0.235 | 3.473 | 0.36 | 0.00 |
| collect_red | safe | Θ₂ | 0.444 | 2.193 | 0.74 | 0.00 |
| collect_red | safe | Θ_K | 0.469 | 2.860 | 0.66 | 0.00 |
| collect_red | shiny | Θ₂ | 0.302 | 2.187 | 1.00 | 1.00 |
| collect_red | shiny | Θ_K | 0.349 | 2.742 | 1.00 | 1.00 |

## Exp-T7-3: Held-Out Compositional Generalization

| True Goal (held-out) | θ | Mean NLL | Entropy Δ | Goal Acc |
|:-------------------:|:-:|:--------:|:---------:|:--------:|
| collect_red+use_safe | safe | 0.222 | +0.109 | 0.06 |
| collect_red+use_safe | shiny | 0.270 | +0.100 | 0.00 |
| avoid_blue+use_safe | safe | 0.246 | +0.239 | 0.00 |
| avoid_blue+use_safe | shiny | 0.263 | +0.229 | 0.00 |

## Verdict

> Composite goal accuracy (full space): 0.00
> K-type goal accuracy (use_safe, safe): 0.36
> K-type calibration: ✅
