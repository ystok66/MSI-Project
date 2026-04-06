# Step 5C: Bayesian Macro Objective Shadow Results

**Seeds**: 30 | **Elapsed**: 3.1s

## Agreement with Baseline

| Ablation | Scenario | Baseline | Shadow | Agrees |
|----------|----------|----------|--------|--------|
| task_only | low_risk | WARN | WARN | ✓ |
| task+info | low_risk | WARN | WARN | ✓ |
| task+info-dep | low_risk | WARN | WARN | ✓ |
| full | low_risk | WARN | WARN | ✓ |
| full_no_kappa | low_risk | WARN | WARN | ✓ |
| task_only | high_risk | WARN | WARN | ✓ |
| task+info | high_risk | WARN | WARN | ✓ |
| task+info-dep | high_risk | WARN | WARN | ✓ |
| full | high_risk | WARN | WARN | ✓ |
| full_no_kappa | high_risk | WARN | WARN | ✓ |
| task_only | dependent_agent | WARN | WARN | ✓ |
| task+info | dependent_agent | WARN | WARN | ✓ |
| task+info-dep | dependent_agent | WARN | WARN | ✓ |
| full | dependent_agent | WARN | WARN | ✓ |
| full_no_kappa | dependent_agent | WARN | WARN | ✓ |
| task_only | well_calibrated | WARN | WARN | ✓ |
| task+info | well_calibrated | WARN | WARN | ✓ |
| task+info-dep | well_calibrated | WARN | WARN | ✓ |
| full | well_calibrated | WARN | WARN | ✓ |
| full_no_kappa | well_calibrated | WARN | WARN | ✓ |

## Agreement Summary

| Ablation | Agreement Rate |
|----------|----------------|
| task_only | 1.000 (120/120) |
| task+info | 1.000 (120/120) |
| task+info-dep | 1.000 (120/120) |
| full | 1.000 (120/120) |
| full_no_kappa | 1.000 (120/120) |

## Component Breakdown (first seed)

### low_risk

| Option | Task | Info | Dep | κ | Res | Total |
|--------|------|------|-----|---|-----|-------|
| NONE | 0.000 | 0.000 | 0.000 | 0.003 | 0.00 | 0.003 |
| WARN | 0.017 | 0.823 | 0.059 | 0.006 | 0.20 | 0.355 |
| UNLOCK | -0.000 | 0.823 | 0.118 | 0.006 | 0.50 | 0.250 |
| ITEM_DROP | -0.009 | 0.823 | 0.176 | 0.006 | 0.80 | 0.152 |

**Baseline chose**: WARN | **Shadow chose**: WARN

### high_risk

| Option | Task | Info | Dep | κ | Res | Total |
|--------|------|------|-----|---|-----|-------|
| NONE | 0.000 | 0.000 | 0.000 | 0.001 | 0.00 | 0.001 |
| WARN | 0.047 | 0.601 | 0.084 | 0.002 | 0.20 | 0.246 |
| UNLOCK | 0.000 | 0.601 | 0.168 | 0.002 | 0.50 | 0.085 |
| ITEM_DROP | -0.042 | 0.601 | 0.251 | 0.002 | 0.80 | -0.071 |

**Baseline chose**: WARN | **Shadow chose**: WARN

### dependent_agent

| Option | Task | Info | Dep | κ | Res | Total |
|--------|------|------|-----|---|-----|-------|
| NONE | 0.000 | 0.000 | 0.000 | 0.005 | 0.00 | 0.005 |
| WARN | 0.007 | 0.674 | 0.100 | 0.010 | 0.20 | 0.234 |
| UNLOCK | 0.000 | 0.674 | 0.200 | 0.010 | 0.50 | 0.097 |
| ITEM_DROP | -0.004 | 0.674 | 0.300 | 0.010 | 0.80 | -0.037 |

**Baseline chose**: WARN | **Shadow chose**: WARN

### well_calibrated

| Option | Task | Info | Dep | κ | Res | Total |
|--------|------|------|-----|---|-----|-------|
| NONE | 0.000 | 0.000 | 0.000 | 0.006 | 0.00 | 0.006 |
| WARN | 0.031 | 0.762 | 0.062 | 0.012 | 0.20 | 0.342 |
| UNLOCK | 0.000 | 0.762 | 0.125 | 0.012 | 0.50 | 0.218 |
| ITEM_DROP | -0.008 | 0.762 | 0.188 | 0.012 | 0.80 | 0.118 |

**Baseline chose**: WARN | **Shadow chose**: WARN

