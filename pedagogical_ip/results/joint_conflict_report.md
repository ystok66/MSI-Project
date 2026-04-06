# Joint Latent Conflict Report

## L1: Coupled Joint vs Factorized (25 obs, 50 trials)

| Condition | θ | Goal | Coupled Pref | Coupled Goal | Coupled Joint | Fact Pref | Fact Goal |
|-----------|---|------|-------------|-------------|--------------|----------|----------|
| aligned | shiny | goal_collect | 90% | 88% | 88% | 100% | 88% |
| aligned | safe | goal_safe_long | 94% | 88% | 88% | 70% | 70% |
| conflict | shiny | goal_safe_short | 74% | 68% | 68% | 68% | 0% |
| conflict | safe | goal_collect | 56% | 56% | 56% | 0% | 4% |

## L2+L3: Joint Conflict Corridor

| Conflict | Strategy | SBCR | CI | WarnRate | PrefAcc | GoalAcc | JointAcc |
|----------|----------|------|----|---------|---------|---------|---------|
| low | always_wait | 45% | [33%,57%] | 0% | — | — | — |
| low | always_warn | 55% | [43%,67%] | 100% | — | — | — |
| low | v4 | 55% | [43%,67%] | 100% | — | — | — |
| low | joint_v1 | 55% | [43%,67%] | 100% | 6% | 22% | 4% |
| low | oracle | 55% | [43%,67%] | 100% | — | — | — |
| med | always_wait | 45% | [33%,57%] | 0% | — | — | — |
| med | always_warn | 55% | [43%,67%] | 100% | — | — | — |
| med | v4 | 55% | [43%,67%] | 100% | — | — | — |
| med | joint_v1 | 55% | [43%,67%] | 100% | 20% | 32% | 12% |
| med | oracle | 55% | [43%,67%] | 100% | — | — | — |
| high | always_wait | 45% | [33%,57%] | 0% | — | — | — |
| high | always_warn | 55% | [43%,67%] | 100% | — | — | — |
| high | v4 | 55% | [43%,67%] | 100% | — | — | — |
| high | joint_v1 | 55% | [43%,67%] | 100% | 18% | 26% | 8% |
| high | oracle | 55% | [43%,67%] | 100% | — | — | — |

## L4: Persistent Agent Profile (θ=shiny, 8 episodes)

| Mode | Episode | WarnRate | PrefAcc | SBCR |
|------|---------|---------|---------|------|
| fresh | 0 | 100% | 100% | 47% |
| fresh | 1 | 100% | 100% | 47% |
| fresh | 2 | 100% | 100% | 47% |
| fresh | 3 | 100% | 100% | 47% |
| fresh | 4 | 100% | 100% | 47% |
| fresh | 5 | 100% | 100% | 47% |
| fresh | 6 | 100% | 100% | 47% |
| fresh | 7 | 100% | 100% | 47% |
| persistent | 0 | 100% | 100% | 47% |
| persistent | 1 | 100% | 100% | 47% |
| persistent | 2 | 100% | 100% | 47% |
| persistent | 3 | 100% | 100% | 47% |
| persistent | 4 | 100% | 100% | 47% |
| persistent | 5 | 100% | 100% | 47% |
| persistent | 6 | 100% | 100% | 47% |
| persistent | 7 | 100% | 100% | 47% |

### Persistent vs Fresh Summary

| Metric | Fresh (avg) | Persistent (avg) |
|--------|------------|------------------|
| WarnRate | 100% | 100% |
| PrefAcc | 100% | 100% |
| SBCR | 47% | 47% |
