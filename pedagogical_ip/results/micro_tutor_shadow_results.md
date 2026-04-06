# Tutor Shadow Comparison & Edge-Case Tests

> Seeds: 20 | Families: 5

## Exp 1: Active-Family Shadow Comparison

### Dose Distribution by Arm

| Arm | WAIT | SOFT | WARN | Mean Dose |
|-----|:----:|:----:|:----:|:---------:|
| base | 97% | 2% | 1% | 0.022 |
| EPU | 97% | 2% | 1% | 0.022 |
| belief | 94% | 4% | 2% | 0.044 |
| EIG | 97% | 2% | 1% | 0.022 |
| all3 | 94% | 4% | 2% | 0.044 |

### EPU Shadow Agreement

| Subtype | EPU agrees with base | EPU action | Base action |
|---------|:----:|:------:|:------:|
| self_discovery_needed | 0% | WARN | WAIT |
| warn_rescue | 5% | WARN | WAIT |
| beneficial_novelty | 5% | WARN | WAIT |
| false_suppression_cost | 2% | WARN | WAIT |

### Belief-Horizon p_self Comparison

| Subtype | p_geom (mean) | p_hybrid (mean) | Δ (mean) |
|---------|:---:|:---:|:---:|
| self_discovery_needed | 0.9825 | 0.8149 | -0.1677 |
| warn_rescue | 0.2712 | 0.1356 | -0.1356 |
| beneficial_novelty | 0.8770 | 0.6395 | -0.2375 |
| false_suppression_cost | 0.7566 | 0.5238 | -0.2328 |

### EIG Observation Values

| Subtype | I(A;θ) mean | Wait boost mean |
|---------|:---:|:---:|
| self_discovery_needed | 0.243575 | 0.2436 |
| warn_rescue | 0.434607 | 0.4346 |
| beneficial_novelty | 0.249408 | 0.2494 |
| false_suppression_cost | 0.420128 | 0.4201 |

## Exp 2: Edge-Case Necessity Tests

### 2a: Noisy Self-Discovery (varied κ, ν)

| Condition | p_geom | p_hybrid(η=0.25) | p_hybrid(η=0.5) | p_hybrid(η=0.75) |
|-----------|:------:|:---:|:---:|:---:|
| standard (κ=0.5, ν=0.3) | 0.9779 | 0.8583 | 0.7386 | 0.6189 |
| risk-aware (κ=1.5, ν=0.2) | 0.9779 | 0.8761 | 0.7743 | 0.6724 |
| stubborn (κ=0.3, ν=0.7) | 0.9779 | 0.7655 | 0.5532 | 0.3408 |
| dependent (κ=0.8, ν=0.9) | 0.9779 | 0.7513 | 0.5246 | 0.2980 |
| fresh (κ=0.5, ν=0.0) | 0.9779 | 0.9118 | 0.8456 | 0.7794 |

### 2b: EIG Observation — θ Disambiguation

| θ_true | q(safe) | I(A;θ) | Interpretation |
|--------|:-------:|:------:|----------------|
| safe | 0.3 | 0.386949 | high info |
| safe | 0.5 | 0.386949 | high info |
| safe | 0.7 | 0.386949 | high info |
| safe | 0.9 | 0.386949 | high info |
| shiny | 0.3 | 0.395973 | high info |
| shiny | 0.5 | 0.395973 | high info |
| shiny | 0.7 | 0.395973 | high info |
| shiny | 0.9 | 0.395973 | high info |

## Summary & Verdicts

_Verdicts to be determined after reviewing results._
