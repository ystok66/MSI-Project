# P3-C: Exact-Q Geometry Audit + Action-Space Ablation

## Line 1: Exact-Q Per-Dose (Real Scenarios)

**Total decision points: 600**

| Action | V_d | Count |
|--------|:---:|:-----:|
| WAIT | 0.9733 | 584 |
| **SOFT** | **0.0050** | **3** |
| WARN | 0.0217 | 13 |

### Q Margins

| Comparison | Mean | Med | Min | Max | Frac>0 |
|------------|:----:|:---:|:---:|:---:|:------:|
| Q_soft − Q_wait | -1.9720 | -2.0907 | -3.7002 | 0.1980 | 0.0267 |
| Q_warn − Q_wait | -3.9550 | -4.1953 | -7.3606 | 0.2746 | 0.0267 |
| Q_soft − Q_warn | 1.9830 | 2.0997 | -0.0804 | 3.6604 | 0.9783 |

### By Family: Exact-Q Optimal Action

| Family | n | WAIT | SOFT | WARN | Q_soft−Q_wait(mean) |
|--------|:-:|:----:|:----:|:----:|:-------------------:|
| beneficial_novelty | 30 | 1.000 | 0.000 | 0.000 | -2.2303 |
| blind_activation_corridor | 30 | 0.833 | 0.000 | 0.167 | -0.1583 |
| false_suppression | 30 | 1.000 | 0.000 | 0.000 | -1.8527 |
| ppmrb_self_discovery | 60 | 1.000 | 0.000 | 0.000 | -3.4696 |
| ppmrb_standard | 60 | 1.000 | 0.000 | 0.000 | -2.0230 |
| soft_boundary_tradeoff | 30 | 1.000 | 0.000 | 0.000 | -1.6554 |
| sparse_invalid_advice | 60 | 1.000 | 0.000 | 0.000 | -1.9223 |
| sparse_valid_advice | 60 | 1.000 | 0.000 | 0.000 | -2.3830 |
| tic_rescue_heavy | 60 | 0.867 | 0.050 | 0.083 | -0.3428 |
| tic_self_discovery | 60 | 1.000 | 0.000 | 0.000 | -3.4351 |
| tic_temptation | 60 | 1.000 | 0.000 | 0.000 | -1.7814 |
| verified_warn | 30 | 1.000 | 0.000 | 0.000 | -2.2960 |
| warn_symmetric_rescue | 30 | 0.900 | 0.000 | 0.100 | -0.5330 |

## Line 2: Action-Space Ablation (3-act vs 2-act)

| θ | Config | Success | Dose Rate | Warn Rate | STOP Agree | Top-1 |
|:-:|--------|:-------:|:---------:|:---------:|:----------:|:-----:|
| safe | 3-act | 0.4767 | 0.0333 | 0.0300 | 0.973 | 1.0 |
| safe | 2-act | 0.4767 | 0.0333 | 0.0333 | 0.973 | 1.0 |
| shiny | 3-act | 0.5467 | 0.0200 | 0.0133 | 0.977 | 1.0 |
| shiny | 2-act | 0.5467 | 0.0200 | 0.0200 | 0.977 | 1.0 |

### 3-act vs 2-act Decision Comparison

**Decision disagreements: 3/600 (0.50%)**

| Family | Disagree | Total | Rate |
|--------|:--------:|:-----:|:----:|
| tic_rescue_heavy | 3 | 60 | 5.0% |

## Verdict

> **SOFT is near-redundant.** V_soft=3/600, 3v2 disagreements=3/600. Removing SOFT would have negligible impact.
