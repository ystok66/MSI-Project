# Funnel Trap Sweep Results

20 seeds × 3 difficulties × 4 conditions

## easy

| Condition | SR | WBCR | TQ | PRCR | warns |
|-----------|----|----|----|----|-------|
| no_tutor | 60% | 1.00 | 0.00 | 0.10 | 0 |
| warning_only | 60% | 1.00 | 0.00 | 0.10 | 0 |
| robot_belief_pre | 70% | 1.00 | 0.27 | 0.00 | 20 |
| robot_belief_post | 95% | 1.00 | 0.31 | 0.00 | 20 |

## medium

| Condition | SR | WBCR | TQ | PRCR | warns |
|-----------|----|----|----|----|-------|
| no_tutor | 40% | 1.00 | 0.00 | 0.35 | 0 |
| warning_only | 40% | 1.00 | 0.00 | 0.35 | 0 |
| robot_belief_pre | 15% | 1.00 | 0.27 | 0.20 | 20 |
| robot_belief_post | 45% | 1.00 | 0.30 | 0.10 | 20 |

## hard

| Condition | SR | WBCR | TQ | PRCR | warns |
|-----------|----|----|----|----|-------|
| no_tutor | 5% | 1.00 | 0.00 | 0.40 | 0 |
| warning_only | 5% | 1.00 | 0.00 | 0.40 | 0 |
| robot_belief_pre | 20% | 1.00 | 0.25 | 0.20 | 20 |
| robot_belief_post | 45% | 1.00 | 0.28 | 0.20 | 20 |

## Metric Definitions

- **SR**: Success rate (reached goal & survived)
- **WBCR**: Wrong-Branch Commitment Rate (fraction of episodes where agent entered trap branch)
- **TQ**: Timing Quality of first WARN (1.0 = perfectly timed at decision point 2, decay with distance)
- **PRCR**: Prefix-Risk Correction Rate (fraction of trap entries corrected before commitment point)

## Expected Patterns

- no_tutor: high WBCR (agent takes shorter trap path)
- warning_only: lower WBCR, but TQ may be poor (warns too late)
- robot_belief_pre: may issue WARN but timing not optimized
- robot_belief_post (TPM): lowest WBCR, highest TQ, best SR
