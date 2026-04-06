# P3-B: SOFT-Optimality Geometry Audit

## Line 1: Synthetic Grid Sweep

**Total grid points: 9600**

| Action | Optimality Volume V_d | Count |
|--------|:---------------------:|:-----:|
| WAIT | 1.0000 | 9600 |
| **SOFT** | **0.0000** | **0** |
| WARN | 0.0000 | 0 |

**SOFT is NEVER optimal in the synthetic grid.**

## Line 2: Real Trajectory Empirical Volumes

| Family | n | WAIT | SOFT | WARN |
|--------|:-:|:----:|:----:|:----:|
| beneficial_novelty | 30 | 1.000 | 0.000 | 0.000 |
| blind_activation_corridor | 30 | 0.800 | 0.000 | 0.200 |
| false_suppression | 30 | 1.000 | 0.000 | 0.000 |
| ppmrb_self_discovery | 60 | 1.000 | 0.000 | 0.000 |
| ppmrb_standard | 60 | 1.000 | 0.000 | 0.000 |
| soft_boundary_tradeoff | 30 | 1.000 | 0.000 | 0.000 |
| sparse_invalid_advice | 60 | 1.000 | 0.000 | 0.000 |
| sparse_valid_advice | 60 | 1.000 | 0.000 | 0.000 |
| tic_rescue_heavy | 60 | 0.867 | 0.050 | 0.083 |
| tic_self_discovery | 60 | 1.000 | 0.000 | 0.000 |
| tic_temptation | 60 | 1.000 | 0.000 | 0.000 |
| verified_warn | 30 | 1.000 | 0.000 | 0.000 |
| warn_symmetric_rescue | 30 | 0.933 | 0.000 | 0.067 |

**Total: WAIT=584 SOFT=3 WARN=13 (n=600)**
- V_WAIT = 0.9733
- V_SOFT = 0.0050
- V_WARN = 0.0217

## Verdict

> **SOFT has marginal optimality region.** V_soft=0.0000 in synthetic, 3/600 in real. Consider simplifying to WAIT/WARN.
