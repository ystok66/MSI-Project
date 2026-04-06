# Step 5: Contract Closure + Held-out Promotion Audit

## A. Held-Out Family Audit

| Lesson | n | TBSR | WR | SelGap | SD | Fallback |
|:-------|:-:|:----:|:--:|:------:|:--:|:--------:|
| beneficial_novelty | 240 | 0.492 | 0.000 | 0.000 | 0.442 | 0.000 |
| blind_activation_corridor | 240 | 0.550 | 1.000 | 1.000 | 0.000 | 0.000 |
| false_suppression | 240 | 0.487 | 0.175 | -0.175 | 0.338 | 0.000 |
| ppmrb_self_discovery | 240 | 0.529 | 0.000 | 0.000 | 0.529 | 0.000 |
| ppmrb_standard | 240 | 0.517 | 0.192 | 0.145 | 0.325 | 0.000 |
| soft_boundary_tradeoff | 240 | 0.533 | 0.250 | 0.865 | 0.271 | 0.000 |
| sparse_invalid_advice | 240 | 0.496 | 0.167 | -0.167 | 0.350 | 0.000 |
| sparse_valid_advice | 240 | 0.500 | 0.000 | 0.000 | 0.500 | 0.000 |
| tic_rescue_heavy | 240 | 0.537 | 1.000 | 0.000 | 0.000 | 0.000 |
| tic_self_discovery | 240 | 0.504 | 0.000 | 0.000 | 0.504 | 0.000 |
| tic_temptation | 240 | 0.521 | 0.083 | -0.083 | 0.479 | 0.000 |
| verified_warn | 240 | 0.529 | 0.000 | 0.000 | 0.529 | 0.000 |
| warn_symmetric_rescue | 240 | 0.537 | 1.000 | 0.000 | 0.000 | 0.000 |

## B. Parameter Sensitivity (Plateau Audit)

| Param | Value | TBSR | WR | SelGap | SD |
|:------|:-----:|:----:|:--:|:------:|:--:|
| beta_task | 0.5 | 0.511 | 0.000 | 0.000 | 0.300 |
| beta_task | 1.0 | 0.511 | 0.156 | 0.500 | 0.300 |
| beta_task | 1.5 ✦ | 0.511 | 0.422 | 0.839 | 0.300 |
| beta_task | 2.0 | 0.511 | 0.478 | 0.758 | 0.267 |
| beta_task | 2.5 | 0.511 | 0.522 | 0.694 | 0.256 |
| — | — | — | — | — | — |
| beta_learn | 1.0 | 0.511 | 0.422 | 0.839 | 0.300 |
| beta_learn | 1.5 | 0.511 | 0.422 | 0.839 | 0.300 |
| beta_learn | 2.5 ✦ | 0.511 | 0.422 | 0.839 | 0.300 |
| beta_learn | 3.5 | 0.511 | 0.422 | 0.839 | 0.300 |
| beta_learn | 4.0 | 0.511 | 0.422 | 0.839 | 0.300 |
| — | — | — | — | — | — |
| beta_dep | 0.5 | 0.511 | 0.422 | 0.839 | 0.300 |
| beta_dep | 1.0 | 0.511 | 0.422 | 0.839 | 0.300 |
| beta_dep | 2.0 ✦ | 0.511 | 0.422 | 0.839 | 0.300 |
| beta_dep | 3.0 | 0.511 | 0.422 | 0.839 | 0.300 |
| beta_dep | 4.0 | 0.511 | 0.378 | 0.903 | 0.300 |
| — | — | — | — | — | — |
| beta_cost | 0.5 | 0.511 | 0.522 | 0.694 | 0.256 |
| beta_cost | 1.0 | 0.511 | 0.433 | 0.823 | 0.289 |
| beta_cost | 1.5 ✦ | 0.511 | 0.422 | 0.839 | 0.300 |
| beta_cost | 2.0 | 0.511 | 0.356 | 0.935 | 0.300 |
| beta_cost | 3.0 | 0.511 | 0.000 | 0.000 | 0.300 |
| — | — | — | — | — | — |
| delta_threshold | 0.0 | 0.511 | 0.433 | 0.823 | 0.289 |
| delta_threshold | 0.25 | 0.511 | 0.422 | 0.839 | 0.300 |
| delta_threshold | 0.5 ✦ | 0.511 | 0.422 | 0.839 | 0.300 |
| delta_threshold | 0.75 | 0.511 | 0.422 | 0.839 | 0.300 |
| delta_threshold | 1.0 | 0.511 | 0.356 | 0.935 | 0.300 |
| — | — | — | — | — | — |
| tau_necessity | 0.0 | 0.511 | 0.422 | 0.839 | 0.300 |
| tau_necessity | 0.1 | 0.511 | 0.422 | 0.839 | 0.300 |
| tau_necessity | 0.2 ✦ | 0.511 | 0.422 | 0.839 | 0.300 |
| tau_necessity | 0.4 | 0.511 | 0.422 | 0.839 | 0.300 |
| tau_necessity | 0.6 | 0.511 | 0.422 | 0.839 | 0.300 |
| — | — | — | — | — | — |
| λ_cc | Value | TBSR | WR | SelGap | SD |
|:-----|:-----:|:----:|:--:|:------:|:--:|
| λ_cc | 0.0 | 0.511 | 0.422 | 0.839 | 0.300 |
| λ_cc | 0.3 | 0.511 | 0.422 | 0.839 | 0.300 |
| λ_cc | 0.6 ✦ | 0.511 | 0.422 | 0.839 | 0.300 |
| λ_cc | 0.9 | 0.511 | 0.422 | 0.839 | 0.300 |
| λ_cc | 1.0 | 0.511 | 0.422 | 0.839 | 0.300 |

## C. Ternary vs Binary Closure

### ternary (default)

TBSR=0.4667, WR=0.3250, SelGap=0.8804, SD=0.3125

| Subtype | n | WR | SD |
|:--------|:-:|:--:|:--:|
| self_discovery_needed | 36 | 0.000 | 0.583 |
| boundary_obs | 16 | 0.375 | 0.250 |
| warn_rescue | 36 | 1.000 | 0.000 |
| false_suppression_cost | 36 | 0.167 | 0.306 |
| beneficial_novelty | 36 | 0.000 | 0.472 |
| blind_corridor | 24 | 1.000 | 0.000 |
| soft_gradual | 24 | 0.250 | 0.333 |
| verified_warn | 24 | 0.000 | 0.375 |

### binary rollback

TBSR=0.4667, WR=0.5625, SelGap=0.5707, SD=0.2167

| Subtype | n | WR | SD |
|:--------|:-:|:--:|:--:|
| self_discovery_needed | 36 | 0.028 | 0.556 |
| boundary_obs | 16 | 0.875 | 0.062 |
| warn_rescue | 36 | 1.000 | 0.000 |
| false_suppression_cost | 36 | 0.639 | 0.139 |
| beneficial_novelty | 36 | 0.333 | 0.361 |
| blind_corridor | 24 | 1.000 | 0.000 |
| soft_gradual | 24 | 0.792 | 0.083 |
| verified_warn | 24 | 0.083 | 0.333 |

## D. Fallback Rate by Variant

- **posterior_C**: fallback rate = 0.0000 (0/120)
- **posterior_A**: fallback rate = 0.0000 (0/120)
- **posterior_B**: fallback rate = 1.0000 (120/120)

## E. Mirror / Side-Swap Parity

| Theta | WR | SelGap | SD |
|:------|:--:|:------:|:--:|
| safe | 0.2417 | 0.9192 | 0.3667 |
| shiny | 0.2417 | 0.9192 | 0.3083 |

