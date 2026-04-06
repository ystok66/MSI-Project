# FICA-v1: Framework Integrity & Causal Audit

## Block 1: Invariant Regression Matrix

### PP-MRB Family

| θ | Tutor | WR | SBCR | τ-ν | γg | OTR |
|---|-------|----|----|-----|-------|-----|
| safe | cajt_v3 | 0.875 | 0.281 | +0.805 | 0.441 | 0.639 |
| safe | ict_v2 | 0.156 | 0.281 | +0.397 | 0.097 | 0.000 |
| safe | bc_v4 | 0.125 | 0.281 | +0.353 | 0.077 | 0.000 |
| shiny | cajt_v3 | 0.875 | 0.438 | +0.805 | 0.441 | 0.639 |
| shiny | ict_v2 | 0.156 | 0.562 | +0.397 | 0.097 | 0.000 |
| shiny | bc_v4 | 0.125 | 0.562 | +0.353 | 0.077 | 0.000 |

### TIC-v4 Family

| θ | Tutor | WR | B | C | D | E | MCA_E | τ-ν | γg | OTR |
|---|-------|----|----|----|----|----|----|-----|-------|-----|
| safe | no_tutor | 0.000 | 0.375 | 0.250 | 0.375 | 0.625 | 0.625 | +0.253 | 0.000 | 0.500 |
| safe | cajt_v3 | 0.850 | 0.500 | 0.375 | 0.375 | 0.750 | 0.000 | +0.183 | 0.467 | 0.908 |
| safe | ict_v2 | 0.000 | 0.375 | 0.250 | 0.375 | 0.625 | 0.625 | +0.253 | 0.000 | 0.500 |
| safe | bc_v4 | 0.000 | 0.375 | 0.250 | 0.375 | 0.625 | 0.625 | +0.253 | 0.000 | 0.500 |
| shiny | no_tutor | 0.000 | 0.438 | 0.625 | 0.562 | 0.500 | 0.500 | +0.250 | 0.000 | 0.120 |
| shiny | cajt_v3 | 0.850 | 0.562 | 0.625 | 0.562 | 0.500 | 0.000 | +0.139 | 0.493 | 0.917 |
| shiny | ict_v2 | 0.025 | 0.438 | 0.625 | 0.562 | 0.500 | 0.500 | +0.269 | 0.012 | 0.120 |
| shiny | bc_v4 | 0.025 | 0.438 | 0.625 | 0.562 | 0.500 | 0.500 | +0.280 | 0.006 | 0.120 |

## Block 2: Mechanism-Consistent Accuracy

| θ | Tutor | E(raw) | **MCA_E** | Δ(raw-MCA) |
|---|-------|--------|---------|------------|
| safe | no_tutor | 0.625 | **0.625** | +0.000 |
| safe | cajt_v3 | 0.750 | **0.000** | — |
| safe | ict_v2 | 0.625 | **0.625** | +0.000 |
| safe | bc_v4 | 0.625 | **0.625** | +0.000 |
| shiny | no_tutor | 0.500 | **0.500** | +0.000 |
| shiny | cajt_v3 | 0.500 | **0.000** | — |
| shiny | ict_v2 | 0.500 | **0.500** | +0.000 |
| shiny | bc_v4 | 0.500 | **0.500** | +0.000 |

## Block 3: State→Behavior Jacobian

| | kappa | tau | nu | gamma_spec | gamma_gen |
|---|---|---|---|---|---|
| RC | +0.0400 | +0.0000 | +0.0000 | +0.0000 | -0.0100 |
| TR | +0.0300 | +0.0000 | +0.0000 | +0.1400 | -0.0100 |
| EP | -0.0400 | +0.0000 | -0.0700 | +0.0000 | -0.7000 |
| VA | +0.0000 | +0.4300 | -0.3500 | +0.0000 | +0.0000 |
| IA | +0.0000 | -0.1100 | +0.6800 | +0.0000 | +0.0900 |

Sparsity (on-diag / total): **0.737**

✅ All expected signs correct.

## Block 4: Dose-Curriculum Audit

| θ | Dose | B | C | D | E | ν | γg | OTR |
|---|------|---|---|---|---|---|-------|-----|
| safe | none | 0.375 | 0.250 | 0.375 | 0.625 | 0.047 | 0.000 | 0.500 |
| safe | soft_only | 0.500 | 0.375 | 0.375 | 0.688 | 0.686 | 0.335 | 0.889 |
| safe | hard_only | 0.500 | 0.375 | 0.375 | 0.750 | 0.800 | 0.500 | 0.926 |
| safe | mixed | 0.500 | 0.375 | 0.375 | 0.750 | 0.789 | 0.448 | 0.917 |
| shiny | none | 0.438 | 0.625 | 0.562 | 0.500 | 0.050 | 0.000 | 0.120 |
| shiny | soft_only | 0.500 | 0.688 | 0.500 | 0.500 | 0.686 | 0.335 | 0.889 |
| shiny | hard_only | 0.562 | 0.625 | 0.562 | 0.500 | 0.800 | 0.500 | 0.926 |
| shiny | mixed | 0.500 | 0.625 | 0.562 | 0.500 | 0.793 | 0.462 | 0.908 |

## Assertion Summary

**19/21 passed** (2 failed)

| # | Assertion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | cajt_v3 high WR on PP-MRB | ✅ | WR=0.875 |
| 2 | cajt_v3 τ-ν negative | ❌ | gap=0.805 |
| 3 | ict_v2 low OTR on PP-MRB | ✅ | OTR=0.0 |
| 4 | ict_v2 τ-ν positive | ✅ | gap=0.397 |
| 5 | bc_v4 low OTR on PP-MRB | ✅ | OTR=0.0 |
| 6 | bc_v4 τ-ν positive | ✅ | gap=0.353 |
| 7 | cajt_v3 high WR on PP-MRB | ✅ | WR=0.875 |
| 8 | cajt_v3 τ-ν negative | ❌ | gap=0.805 |
| 9 | ict_v2 low OTR on PP-MRB | ✅ | OTR=0.0 |
| 10 | ict_v2 τ-ν positive | ✅ | gap=0.397 |
| 11 | bc_v4 low OTR on PP-MRB | ✅ | OTR=0.0 |
| 12 | bc_v4 τ-ν positive | ✅ | gap=0.353 |
| 13 | cajt_v3 high γg on TIC-v4 | ✅ | γg=0.467 |
| 14 | ict_v2 low γg on TIC-v4 | ✅ | γg=0.0 |
| 15 | bc_v4 low γg on TIC-v4 | ✅ | γg=0.0 |
| 16 | cajt_v3 high γg on TIC-v4 | ✅ | γg=0.493 |
| 17 | ict_v2 low γg on TIC-v4 | ✅ | γg=0.012 |
| 18 | bc_v4 low γg on TIC-v4 | ✅ | γg=0.006 |
| 19 | Jacobian sign violations | ✅ | 0 violations |
| 20 | hard_only pushes ν higher (safe) | ✅ | none=0.047, hard=0.8 |
| 21 | hard_only pushes ν higher (shiny) | ✅ | none=0.05, hard=0.8 |
