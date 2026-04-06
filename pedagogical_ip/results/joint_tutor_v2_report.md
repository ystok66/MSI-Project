# Joint Tutor v2 + Regression Report

## Phase 1: Old-Family Regression

Verifying v1.1 and joint_v2 don't break selectivity on established families.

| Family | Tutor | SBCR | WarnRate |
|--------|-------|------|----------|
| delayed_corridor | always_wait | 77% | 0% |
| delayed_corridor | always_warn | 77% | 100% |
| delayed_corridor | v4 | 77% | 100% |
| delayed_corridor | v1_1 | 77% | 100% |
| delayed_corridor | joint_v2 | 77% | 100% |
| distractor_cue | always_wait | 77% | 0% |
| distractor_cue | always_warn | 77% | 100% |
| distractor_cue | v4 | 77% | 100% |
| distractor_cue | v1_1 | 77% | 100% |
| distractor_cue | joint_v2 | 77% | 100% |
| elcb_po | always_wait | 77% | 0% |
| elcb_po | always_warn | 77% | 100% |
| elcb_po | v4 | 77% | 100% |
| elcb_po | v1_1 | 77% | 100% |
| elcb_po | joint_v2 | 77% | 100% |
| temptation_corridor | always_wait | 73% | 0% |
| temptation_corridor | always_warn | 73% | 100% |
| temptation_corridor | v4 | 73% | 100% |
| temptation_corridor | v1_1 | 73% | 100% |
| temptation_corridor | joint_v2 | 73% | 100% |

### Regression: v4 vs v1.1 vs joint_v2 WarnRate

| Family | v4 | v1.1 | joint_v2 | Δ(v1.1-v4) | Δ(jv2-v4) |
|--------|-----|------|----------|------------|----------|
| delayed_corridor | 100% | 100% | 100% | +0.000 | +0.000 |
| distractor_cue | 100% | 100% | 100% | +0.000 | +0.000 |
| elcb_po | 100% | 100% | 100% | +0.000 | +0.000 |
| temptation_corridor | 100% | 100% | 100% | +0.000 | +0.000 |

## Phase 2: Joint Conflict Corridor

| Strategy | SBCR | WarnRate | WR(wc) | WR(wt) | **SelGap** | AvgObsVal | AvgBR | AvgConflict |
|----------|------|---------|--------|--------|-----------|-----------|-------|-------------|
| always_wait | 49% | 0% | 0% | 0% | **0.000** | — | — | — |
| always_warn | 49% | 100% | 100% | 100% | **0.000** | — | — | — |
| v4_reset | 49% | 87% | 60% | 100% | **0.405** | — | — | — |
| v1_1_persistent | 49% | 67% | 27% | 100% | **0.730** | — | — | — |
| joint_v1 | 49% | 96% | 84% | 100% | **0.162** | — | — | — |
| joint_v2_coupled | 49% | 88% | 54% | 100% | **0.459** | 0.0172 | 0.1960 | 0.2070 |
| joint_v2_factorized_abl | 49% | 100% | 100% | 100% | **0.000** | 0.0832 | 0.2921 | 0.1785 |
| oracle_joint | 49% | 30% | 0% | 100% | **1.000** | — | — | — |

### Key: SelGap Comparison

| v4_reset | v1.1 | joint_v1 | **joint_v2** | fact_abl | oracle |
|----------|------|----------|-------------|----------|--------|
| 0.405 | 0.730 | 0.162 | **0.459** | 0.000 | 1.000 |
