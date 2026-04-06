# P3-A: Balanced Active Coverage Suite

**Total lessons: 13** (10 original + 3 new ACTIVE families)

## Exp 1: Coverage Verification

| Family | n | selfdisc | **warned** | **active** | **dose>0** |
|--------|:-:|:--------:|:----------:|:----------:|:----------:|
| beneficial_novelty | 30 | 0.400 | 0.000 | 0.000 | 0.000 |
| blind_activation_corridor | 30 | 0.000 | 0.233 | 0.233 | 0.233 |
| false_suppression | 30 | 0.500 | 0.000 | 0.000 | 0.000 |
| ppmrb_self_discovery | 60 | 0.467 | 0.000 | 0.000 | 0.000 |
| ppmrb_standard | 60 | 0.267 | 0.000 | 0.000 | 0.000 |
| soft_boundary_tradeoff | 30 | 0.333 | 0.000 | 0.000 | 0.000 |
| sparse_invalid_advice | 60 | 0.300 | 0.000 | 0.000 | 0.000 |
| sparse_valid_advice | 60 | 0.483 | 0.000 | 0.000 | 0.000 |
| tic_rescue_heavy | 60 | 0.000 | 0.067 | 0.067 | 0.067 |
| tic_self_discovery | 60 | 0.417 | 0.000 | 0.000 | 0.000 |
| tic_temptation | 60 | 0.400 | 0.000 | 0.000 | 0.000 |
| verified_warn | 30 | 0.467 | 0.000 | 0.000 | 0.000 |
| warn_symmetric_rescue | 30 | 0.000 | 0.100 | 0.100 | 0.100 |

**Families with active>0: 3**

## Exp 2: Infer-Only on Full Catalog (incl. Active Families)

| θ | Div All | Div@Active | n_active | R_active | Div@Hard | Success |
|:-:|:-------:|:----------:|:--------:|:--------:|:--------:|:-------:|
| safe | 0.0000 | 0.0000 | 8 | 0.0000 | 0.0000 | 0.4633 |
| shiny | 0.0000 | 0.0000 | 6 | 0.0000 | 0.0000 | 0.4800 |

## Exp 3: Per-Family Divergence Forensics

| Family | n | Div All | n_active | Div@Active | R_active |
|--------|:-:|:-------:|:--------:|:----------:|:--------:|
| beneficial_novelty | 30 | 0.0000 | 0 | 0.0000 | 0.0000 |
| blind_activation_corridor | 30 | 0.0000 | 7 | 0.0000 | 0.0000 |
| false_suppression | 30 | 0.0000 | 0 | 0.0000 | 0.0000 |
| ppmrb_self_discovery | 60 | 0.0000 | 0 | 0.0000 | 0.0000 |
| ppmrb_standard | 60 | 0.0000 | 0 | 0.0000 | 0.0000 |
| soft_boundary_tradeoff | 30 | 0.0000 | 0 | 0.0000 | 0.0000 |
| sparse_invalid_advice | 60 | 0.0000 | 0 | 0.0000 | 0.0000 |
| sparse_valid_advice | 60 | 0.0000 | 0 | 0.0000 | 0.0000 |
| tic_rescue_heavy | 60 | 0.0000 | 4 | 0.0000 | 0.0000 |
| tic_self_discovery | 60 | 0.0000 | 0 | 0.0000 | 0.0000 |
| tic_temptation | 60 | 0.0000 | 0 | 0.0000 | 0.0000 |
| verified_warn | 30 | 0.0000 | 0 | 0.0000 | 0.0000 |
| warn_symmetric_rescue | 30 | 0.0000 | 3 | 0.0000 | 0.0000 |

**No divergences found.**

## Exp 4: Macro Pilot on Balanced Suite (α=1.0)

| θ | STOP Agree | Top-1 | Kendall τ | Δε_stop |
|:-:|:----------:|:-----:|:---------:|:-------:|
| safe | 0.983 | 1.000 | 0.9864 | 0.0050 |
| shiny | 1.000 | 1.000 | 1.0000 | 0.0030 |
