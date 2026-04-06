# Learner Bottleneck Micro-Suite Report

## D1: Exposure Scaling

Does more training produce transfer?

| Family | Condition | k=0 | k=1 | k=3 | k=10 | k=30 | Trend |
|--------|-----------|-----|-----|-----|------|------|-------|
| fork_trap | no_tutor | 30% | 30% | 30% | 30% | 30% | → |
| fork_trap | robot_belief_post | 30% | 30% | 30% | 30% | 30% | → |
| | | | | | | | |
| hazard_belt | no_tutor | 50% | 50% | 50% | 50% | 50% | → |
| hazard_belt | robot_belief_post | 50% | 50% | 50% | 50% | 50% | → |
| | | | | | | | |
| deadline_gate | no_tutor | 100% | 100% | 100% | 100% | 100% | → |
| deadline_gate | robot_belief_post | 100% | 100% | 100% | 100% | 100% | → |
| | | | | | | | |

## D2: Transfer Gradient

Generalization radius: same-map → same-family → held-out

| Source Family | Tier | SR Before | SR After | Δ |
|-------------|------|-----------|----------|---|
| fork_trap | T1_same_map | 0% | 0% | +0% |
| fork_trap | T2_same_family | 0% | 30% | +30% |
| fork_trap→hazard_belt | T3_held_out | 0% | 50% | +50% |
| fork_trap→deadline_gate | T3_held_out | 0% | 100% | +100% |
| hazard_belt | T1_same_map | 0% | 0% | +0% |
| hazard_belt | T2_same_family | 0% | 50% | +50% |
| hazard_belt→fork_trap | T3_held_out | 0% | 30% | +30% |
| hazard_belt→deadline_gate | T3_held_out | 0% | 100% | +100% |
| deadline_gate | T1_same_map | 60% | 60% | +0% |
| deadline_gate | T2_same_family | 60% | 100% | +40% |
| deadline_gate→fork_trap | T3_held_out | 60% | 30% | -30% |
| deadline_gate→hazard_belt | T3_held_out | 60% | 50% | -10% |

## D3: Oracle Supervision Upper Bound

Can the linear head learn the mapping with perfect labels?

| Family | n_labels | Probe SR | MSE_cost | MSE_risk |
|--------|----------|----------|----------|----------|
| fork_trap | 10 | 30% | 0.011495 | 0.076916 |
| fork_trap | 50 | 30% | 0.005846 | 0.069535 |
| fork_trap | 200 | 30% | 0.008124 | 0.084060 |
| fork_trap | 1000 | 30% | 0.013536 | 0.078569 |
| hazard_belt | 10 | 50% | 0.079370 | 0.127189 |
| hazard_belt | 50 | 50% | 0.036530 | 0.098558 |
| hazard_belt | 200 | 50% | 0.020715 | 0.095024 |
| hazard_belt | 1000 | 50% | 0.015190 | 0.077725 |
| deadline_gate | 10 | 100% | 0.069708 | 0.021695 |
| deadline_gate | 50 | 100% | 0.027804 | 0.028962 |
| deadline_gate | 200 | 100% | 0.049193 | 0.027932 |
| deadline_gate | 1000 | 100% | 0.088628 | 0.029131 |

## Interpretation

### Key Questions Answered

1. **Does more exposure help?** (D1)
   - If SR flat at all k → learner capacity issue
   - If SR rises slowly → learning rate / sample efficiency issue

2. **Where does generalization break?** (D2)
   - T1 positive, T2/T3 zero → map-specific learning
   - T1+T2 positive, T3 zero → family-specific learning
   - All zero → no generalization at all

3. **Is the capacity sufficient?** (D3)
   - High SR with oracle → capacity OK, supervision is the bottleneck
   - Low SR even with oracle → linear head can't represent the mapping
