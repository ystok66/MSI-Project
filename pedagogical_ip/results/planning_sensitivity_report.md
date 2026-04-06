# Planning Sensitivity Audit

10 seeds × 3 families, 10 training episodes each

## Route Flip Rate & Margin Change

| Family | Flip Rate | Mean Δθ | Mean M_pre | Mean M_post | Mean ΔM |
|--------|-----------|---------|-----------|------------|--------|
| fork_trap | 0% | 0.6890 | 0.00 | 0.00 | 0.0000 |
| hazard_belt | 0% | 0.2075 | 0.00 | 0.00 | 0.0000 |
| deadline_gate | 0% | nan | 0.00 | 0.00 | 0.0000 |

## Cell Cost: Risky vs Safe Branch

| Family | Gap_pre (risky-safe) | Gap_post | Δ(gap) |
|--------|---------------------|----------|--------|
| fork_trap | 0.0000 | 0.0000 | +0.0000 |
| hazard_belt | 0.0000 | 0.0000 | +0.0000 |
| deadline_gate | 0.0000 | 0.0000 | +0.0000 |

## Branch Ranking Accuracy

| Family | Rank Correct (risky > safe cost) |
|--------|---------|
| fork_trap | N/A |
| hazard_belt | N/A |
| deadline_gate | N/A |

## Key Diagnostic

If **Flip Rate ≈ 0%** AND **Δθ > 0** → confirms prediction→planning disconnect.

If **Gap_post > Gap_pre** → training does increase cost gap but not enough to flip.

If **Rank Correct ≈ 100%** → learner correctly ranks danger, but planner's other terms (uncertainty, necessity) dominate.
