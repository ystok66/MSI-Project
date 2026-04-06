# Planning Sensitivity Audit v2

## Route Flip & Cost Gap

| Family | Flip Rate | Δθ | Gap_pre | Gap_post | Δ(gap) | risk_gap Δ | uc_gap Δ |
|--------|-----------|-----|---------|----------|--------|-----------|----------|
| fork_trap | 0% | 0.689 | -inf | -inf | +nan | +1.821 | +0.000 |
| hazard_belt | 0% | 0.208 | -inf | -inf | +nan | -0.409 | +0.000 |

## Example Risk Predictions (first risky vs safe cell)

| Family | risky_r̂_pre | risky_r̂_post | safe_r̂_pre | safe_r̂_post |
|--------|-----------|------------|----------|----------|
| fork_trap | 0.5000 | 0.6534 | 0.0000 | 0.0000 |
| hazard_belt | 0.5000 | 0.4576 | 0.0000 | 0.0000 |

## A* Branch Selection (from fork entry)

| Family | seed | branch_pre | branch_post | flip |
|--------|------|-----------|------------|------|
| fork_trap | 0 | unknown | unknown | 0 |
| fork_trap | 1 | risky | risky | 0 |
| fork_trap | 2 | unknown | unknown | 0 |
| fork_trap | 3 | unknown | unknown | 0 |
| fork_trap | 4 | unknown | unknown | 0 |
| hazard_belt | 0 | risky | risky | 0 |
| hazard_belt | 1 | risky | risky | 0 |
| hazard_belt | 2 | risky | risky | 0 |
| hazard_belt | 3 | risky | risky | 0 |
| hazard_belt | 4 | risky | risky | 0 |

## Diagnosis

### If gap_pre ≈ gap_post ≈ 0:
→ Risky and safe cells have IDENTICAL planner costs.
The latent predictor produces the SAME cost/risk for both because:
- Prior predicts uniform risk (sigmoid(0)=0.5) for unvisited cells.
- Features differ but the learned weights don't discriminate.

### If gap_post > gap_pre but no flip:
→ Training widens the gap but not enough to overcome the topology/uncertainty terms.

### If gap_post > gap_pre AND flips occur:
→ Training successfully influences route selection! Transfer should follow.
