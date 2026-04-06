# ELCB Experiment Report

## SC1: Oracle Semantic Flip Curve

| Risk Offset | Safe Chosen Rate | Mean Margin |
|-------------|-----------------|-------------|
| 0.00 | 80% | 0.169 |
| 0.05 | 80% | 0.336 |
| 0.10 | 80% | 0.515 |
| 0.15 | 80% | 0.706 |
| 0.20 | 100% | 0.909 |
| 0.30 | 100% | 1.242 |
| 0.40 | 100% | 1.520 |
| 0.60 | 100% | 1.761 |

## SC2: Length-Neutrality

- Branch A chosen: 40%
- Branch B chosen: 60%
- Bias (|rate - 0.5|): 0.100
- ✅ PASS

## SC3: Prediction-Planning Coupling (PPCR)

- PPCR: 10/10 (100%)
| Seed | pred_r_A | pred_r_B | pred_pref | planner | coupled | safe |
|------|----------|----------|-----------|---------|---------|------|
| 0 | 0.6427 | 0.6334 | B | B | 1 | 0 |
| 1 | 0.6415 | 0.6347 | B | B | 1 | 1 |
| 2 | 0.6427 | 0.6335 | B | B | 1 | 0 |
| 3 | 0.6427 | 0.6335 | B | B | 1 | 0 |
| 4 | 0.6427 | 0.6335 | B | B | 1 | 0 |
| 5 | 0.6427 | 0.6335 | B | B | 1 | 0 |
| 6 | 0.6415 | 0.6346 | B | B | 1 | 1 |
| 7 | 0.6416 | 0.6347 | B | B | 1 | 1 |
| 8 | 0.6416 | 0.6347 | B | B | 1 | 1 |
| 9 | 0.6416 | 0.6346 | B | B | 1 | 1 |

## SC4: Passability Audit

- Violations: 0 / 1500
- ✅ PASS (zero violations)

## Online Sweep

| Condition | SBCR | SR |
|-----------|------|----|
| no_tutor | 55% | 100% |
| warn_only | 55% | 100% |
| robot_belief_post | 55% | 100% |

## Transfer (SBCR after k training episodes)

| Condition | k=0 | k=1 | k=3 | k=10 | k=30 |
|-----------|-----|-----|-----|------|------|
| no_tutor | 55% | 30% | 30% | 30% | 30% |
| robot_belief_post | 60% | 30% | 30% | 30% | 30% |

## Interpretation

### SC1: Does planner flip with oracle risk?
If SBCR rises with risk_offset → planner IS risk-sensitive.
The flip threshold tells us how much prediction difference is needed.

### SC2: Is topology neutral?
If bias < 0.15 → no systematic branch preference from geometry.

### SC3: Do predictions control planning?
If PPCR > 80% → predictions successfully drive route choice.
If PPCR < 50% → other terms (uncertainty, cost) dominate.

### Transfer: Does SBCR improve with training?
If SBCR rises with k → learner can transfer semantic knowledge.
If flat → same bottleneck as original families.
