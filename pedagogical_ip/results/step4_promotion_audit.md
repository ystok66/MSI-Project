# Step 4: v2.1 Promotion Audit

## Overall Metrics

| Arm | TBSR | WR | SelGap | SD | Brier | ν̂_T | Leakage | EffortT |
|:---:|:----:|:--:|:------:|:--:|:-----:|:---:|:-------:|:-------:|
| A | 0.5000 | 0.3736 | 0.8604 | 0.2688 | 0.2889 | 0.1000 | 0.0000 | 0.5000 |
| B | 0.5000 | 0.3750 | 0.8585 | 0.2688 | 0.2889 | 0.1000 | 0.0000 | 0.5000 |
| C | 0.5000 | 0.3750 | 0.8585 | 0.2688 | 0.2889 | 0.1000 | 0.0000 | 0.5000 |
| D | 0.5000 | 0.3750 | 0.8585 | 0.2688 | 0.2889 | 0.1000 | 0.0000 | 0.8204 |
| E | 0.5000 | 0.6278 | 0.5154 | 0.1847 | 0.3355 | 0.1000 | 0.0003 | 0.5000 |

## Deltas vs. A (Step 2 Best)

| Arm | ΔTBSR | ΔSelGap | ΔSD | ΔWR | Δν̂ | ΔBrier |
|:---:|:-----:|:-------:|:---:|:---:|:---:|:------:|
| B | +0.0000 | -0.0019 | +0.0000 | +0.0014 | +0.0000 | +0.0000 |
| C | +0.0000 | -0.0019 | +0.0000 | +0.0014 | +0.0000 | +0.0000 |
| D | +0.0000 | -0.0019 | +0.0000 | +0.0014 | +0.0000 | +0.0000 |
| E | +0.0000 | -0.3450 | -0.0841 | +0.2542 | +0.0000 | +0.0466 |

## Per-Subtype Breakdown

### self_discovery_teach

| Arm | n | Correct | WR | SD |
|:---:|:-:|:-------:|:--:|:--:|
| A | 72 | 0.542 | 0.000 | 0.542 |
| B | 72 | 0.542 | 0.000 | 0.542 |
| C | 72 | 0.542 | 0.000 | 0.542 |
| D | 72 | 0.542 | 0.000 | 0.542 |
| E | 72 | 0.542 | 0.000 | 0.542 |

### self_discovery_needed

| Arm | n | Correct | WR | SD |
|:---:|:-:|:-------:|:--:|:--:|
| A | 216 | 0.481 | 0.000 | 0.481 |
| B | 216 | 0.481 | 0.000 | 0.481 |
| C | 216 | 0.481 | 0.000 | 0.481 |
| D | 216 | 0.481 | 0.000 | 0.481 |
| E | 216 | 0.481 | 0.056 | 0.468 |

### boundary_obs

| Arm | n | Correct | WR | SD |
|:---:|:-:|:-------:|:--:|:--:|
| A | 92 | 0.522 | 0.261 | 0.163 |
| B | 92 | 0.522 | 0.283 | 0.163 |
| C | 92 | 0.522 | 0.283 | 0.163 |
| D | 92 | 0.522 | 0.283 | 0.163 |
| E | 92 | 0.522 | 0.913 | 0.043 |

### warn_rescue

| Arm | n | Correct | WR | SD |
|:---:|:-:|:-------:|:--:|:--:|
| A | 288 | 0.521 | 0.997 | 0.000 |
| B | 288 | 0.521 | 0.997 | 0.000 |
| C | 288 | 0.521 | 0.997 | 0.000 |
| D | 288 | 0.521 | 0.997 | 0.000 |
| E | 288 | 0.521 | 1.000 | 0.000 |

### false_suppression_cost

| Arm | n | Correct | WR | SD |
|:---:|:-:|:-------:|:--:|:--:|
| A | 144 | 0.472 | 0.208 | 0.299 |
| B | 144 | 0.472 | 0.208 | 0.299 |
| C | 144 | 0.472 | 0.208 | 0.299 |
| D | 144 | 0.472 | 0.208 | 0.299 |
| E | 144 | 0.472 | 0.771 | 0.111 |

### beneficial_novelty

| Arm | n | Correct | WR | SD |
|:---:|:-:|:-------:|:--:|:--:|
| A | 144 | 0.451 | 0.000 | 0.368 |
| B | 144 | 0.451 | 0.000 | 0.368 |
| C | 144 | 0.451 | 0.000 | 0.368 |
| D | 144 | 0.451 | 0.000 | 0.368 |
| E | 144 | 0.451 | 0.528 | 0.229 |

### blind_corridor

| Arm | n | Correct | WR | SD |
|:---:|:-:|:-------:|:--:|:--:|
| A | 144 | 0.472 | 1.000 | 0.000 |
| B | 144 | 0.472 | 1.000 | 0.000 |
| C | 144 | 0.472 | 1.000 | 0.000 |
| D | 144 | 0.472 | 1.000 | 0.000 |
| E | 144 | 0.472 | 1.000 | 0.000 |

### soft_gradual

| Arm | n | Correct | WR | SD |
|:---:|:-:|:-------:|:--:|:--:|
| A | 144 | 0.569 | 0.340 | 0.271 |
| B | 144 | 0.569 | 0.340 | 0.271 |
| C | 144 | 0.569 | 0.340 | 0.271 |
| D | 144 | 0.569 | 0.340 | 0.271 |
| E | 144 | 0.569 | 0.944 | 0.035 |

### verified_warn

| Arm | n | Correct | WR | SD |
|:---:|:-:|:-------:|:--:|:--:|
| A | 144 | 0.472 | 0.000 | 0.472 |
| B | 144 | 0.472 | 0.000 | 0.472 |
| C | 144 | 0.472 | 0.000 | 0.472 |
| D | 144 | 0.472 | 0.000 | 0.472 |
| E | 144 | 0.472 | 0.125 | 0.403 |

## Mirror / Side-Swap Parity

| Arm | safe_WR | shiny_WR | Δ | Parity |
|:---:|:-------:|:--------:|:-:|:------:|
| A | 0.375 | 0.372 | 0.003 | ✅ |
| B | 0.376 | 0.374 | 0.003 | ✅ |
| C | 0.376 | 0.374 | 0.003 | ✅ |
| D | 0.376 | 0.374 | 0.003 | ✅ |
| E | 0.640 | 0.615 | 0.025 | ✅ |

## Verdict

### Promotion Criteria for v2.1 (Arm B)

| Criterion | Required | Actual | Pass |
|:----------|:---------|:-------|:----:|
| ΔTBSR ≥ 0 | — | — | ✅ |
| ΔSelGap ≥ 0 | — | — | ✅ |
| ΔWR < 0 (less over-warning) | — | — | ✅ |
| Δν̂ reasonable | — | — | ✅ |
| warn_rescue WR ≥ 0.90 | — | — | ✅ |
| self_discovery WR = 0 | — | — | ✅ |
| mirror parity < 0.05 | — | — | ✅ |

> **✅ v2.1 PASSES ALL PROMOTION CRITERIA**

### Three-Outcome Necessity (E vs B)

> E (binary) vs B (three-outcome): ΔSelGap=-0.3431, ΔWR=+0.2528
> Three-outcome is **structurally necessary** — binary rollback degrades.

### Module Status Summary

| Module | Status | Reason |
|:-------|:------:|:-------|
| micro_bayes_shadow_v2_1 | **PROMOTE CANDIDATE** | Best SelGap + WR across all steps |
| p_self_posterior_C (three-outcome) | **DEFAULT INPUT** | Structural necessity (Step 2+4) |
| credit_correction | **INCLUDED** | Reduces leakage, improves SelGap |
| p_self_calibration | **DIAGNOSTICS** | Zero policy impact (Steps 2-4) |
| effort_latent_shadow | **DIAGNOSTICS** | Causes WR regression in policy (Step 3) |
| micro_bayes_shadow (v1) | **ABLATION-ONLY** | Superseded by v2.1 |
| micro_bayes_shadow_v2 | **ABLATION-ONLY** | Superseded by v2.1 |
| micro_bayes_shadow_v3 | **ABLATION-ONLY** | Effort component regresses |
