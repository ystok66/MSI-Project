# Step-Level Diagnostic Report

5 seeds × 3 families × 3-4 conditions (medium difficulty)

## fork_trap

| Condition | SR | mean Δθ | Δθ_c | Δθ_r | ΔB_lat | ΔB_pred | BAR |
|-----------|----|---------|----|------|--------|---------|-----|
| no_tutor | 0% | 0.291161 | 0.025838 | 0.206112 | 0.000000 | 0.000000 | 0.00 |
| warning_only | 0% | 0.291161 | 0.025838 | 0.206112 | 0.000000 | 0.000000 | 0.00 |
| robot_belief_pre | 40% | 0.232057 | 0.019791 | 0.146192 | 0.000000 | 0.000000 | 1.00 |
| robot_belief_post | 40% | 0.232057 | 0.019791 | 0.146192 | 0.000000 | 0.000000 | 0.68 |

## hazard_belt

| Condition | SR | mean Δθ | Δθ_c | Δθ_r | ΔB_lat | ΔB_pred | BAR |
|-----------|----|---------|----|------|--------|---------|-----|
| no_tutor | 0% | 0.178267 | 0.005487 | 0.099447 | 0.000000 | 0.000000 | 0.00 |
| item_only | 60% | 0.120294 | 0.004057 | 0.058586 | 0.000000 | 0.000000 | 0.00 |
| robot_belief_pre | 20% | 0.207815 | 0.005279 | 0.112261 | 0.000000 | 0.000000 | 1.00 |
| robot_belief_post | 60% | 0.157270 | 0.003883 | 0.076055 | 0.000000 | 0.000000 | 0.69 |

## deadline_gate

| Condition | SR | mean Δθ | Δθ_c | Δθ_r | ΔB_lat | ΔB_pred | BAR |
|-----------|----|---------|----|------|--------|---------|-----|
| no_tutor | 60% | 0.108455 | 0.001294 | 0.047509 | 0.000000 | 0.000000 | 0.00 |
| unlock_only | 100% | nan | nan | 0.004621 | 0.000000 | 0.000000 | 0.04 |
| robot_belief_pre | 100% | nan | nan | 0.004621 | 0.000000 | 0.000000 | 1.00 |
| robot_belief_post | 100% | nan | nan | 0.004621 | 0.000000 | 0.000000 | 0.84 |

## Key Questions This Should Answer

1. Is Δθ near zero? → learner barely updates → null transfer explained
2. Is Δθ_r >> Δθ_c or vice versa? → which head is updating
3. Is ΔB_pred > 0 but Δθ ≈ 0? → belief changes but weights don't learn
4. Is BAR higher for post_tpm vs pre_tpm? → TPM improves intervention targeting
