# Transfer Evaluation Results

Protocol: k training episodes (with tutor) -> 10 no-tutor probes (held-out seeds 100-109)

## fork_trap

| Condition | k=0 SR | k=1 SR | k=2 SR | k=3 SR | k=0 AE | k=3 AE | LG(3) |
|-----------|--------|--------|--------|--------|--------|--------|-------|
| no_tutor | 30% | 30% | 30% | 30% | 22.32 | 22.32 | +0% |
| warning_only | 30% | 30% | 30% | 30% | 22.32 | 22.32 | +0% |
| robot_belief_pre | 30% | 30% | 30% | 30% | 22.32 | 22.32 | +0% |
| robot_belief_post | 30% | 30% | 30% | 30% | 22.32 | 22.32 | +0% |

## hazard_belt

| Condition | k=0 SR | k=1 SR | k=2 SR | k=3 SR | k=0 AE | k=3 AE | LG(3) |
|-----------|--------|--------|--------|--------|--------|--------|-------|
| no_tutor | 50% | 50% | 50% | 50% | 0.59 | 0.59 | +0% |
| warning_only | 50% | 50% | 50% | 50% | 0.59 | 0.59 | +0% |
| item_only | 50% | 50% | 50% | 50% | 0.59 | 0.59 | +0% |
| robot_belief_pre | 50% | 50% | 50% | 50% | 0.59 | 0.59 | +0% |
| robot_belief_post | 50% | 50% | 50% | 50% | 0.59 | 0.59 | +0% |

## deadline_gate

| Condition | k=0 SR | k=1 SR | k=2 SR | k=3 SR | k=0 AE | k=3 AE | LG(3) |
|-----------|--------|--------|--------|--------|--------|--------|-------|
| no_tutor | 100% | 100% | 100% | 100% | 1.00 | 1.00 | +0% |
| warning_only | 100% | 100% | 100% | 100% | 1.00 | 1.00 | +0% |
| robot_belief_pre | 100% | 100% | 100% | 100% | 1.00 | 1.00 | +0% |
| robot_belief_post | 100% | 100% | 100% | 100% | 1.00 | 1.00 | +0% |

## Pedagogical Efficiency

PE = LG / OHG where OHG = assisted_SR - no_tutor_SR, LG = probe_SR(k=3) - baseline

| Family | Condition | OHG | LG(3) | PE |
|--------|-----------|-----|-------|----|
| fork_trap | no_tutor | 0% | +0% | 0.00 |
| fork_trap | warning_only | 0% | +0% | 0.00 |
| fork_trap | robot_belief_pre | 60% | +0% | 0.00 |
| fork_trap | robot_belief_post | 65% | +0% | 0.00 |

| hazard_belt | no_tutor | 0% | +0% | 0.00 |
| hazard_belt | warning_only | 0% | +0% | 0.00 |
| hazard_belt | item_only | 30% | +0% | 0.00 |
| hazard_belt | robot_belief_pre | 10% | +0% | 0.00 |
| hazard_belt | robot_belief_post | 35% | +0% | 0.00 |

| deadline_gate | no_tutor | 0% | +0% | 0.00 |
| deadline_gate | warning_only | 0% | +0% | 0.00 |
| deadline_gate | robot_belief_pre | 30% | +0% | 0.00 |
| deadline_gate | robot_belief_post | 30% | +0% | 0.00 |

