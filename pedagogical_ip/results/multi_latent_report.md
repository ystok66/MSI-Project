# Multi-Latent Framework Report

## K1: Goal Posterior Convergence (20 obs, 50 trials)

| True Goal | GoalAcc |
|-----------|--------|
| goal_safe_short | 34% |
| goal_safe_long | 72% |
| goal_collect | 84% |
| goal_explore | 80% |
| goal_direct | 100% |

**Mean GoalAcc: 74.0%** (chance = 20.0%)

## K2: Joint Latent Convergence (25 obs, 50 trials)

| θ | Goal | PrefAcc | GoalAcc |
|---|------|---------|--------|
| safe | goal_safe_short | 60% | 24% |
| safe | goal_collect | 0% | 10% |
| shiny | goal_safe_short | 0% | 0% |
| shiny | goal_collect | 100% | 62% |

## K3+K6: Cross-Family Multi-Latent Robustness

| Name | Family | Tutor | SBCR | CI | WarnRate | PrefAcc | GoalAcc | Agent%Safe |
|---|---|---|---|---|---|---|---|---|
| elcb_po/always_wait | elcb_po | always_wait | 45% | [33%,57%] | 0% | — | — | — |
| elcb_po/v4 | elcb_po | v4 | 55% | [43%,67%] | 100% | — | — | — |
| elcb_po/pref_v2 | elcb_po | pref_v2 | 55% | [43%,67%] | 100% | 0% | — | — |
| elcb_po/goal_v1 | elcb_po | goal_v1 | 55% | [43%,67%] | 100% | — | 0% | — |
| elcb_po/oracle | elcb_po | oracle | 55% | [43%,67%] | 100% | — | — | — |
| delayed_Δ=-2/always_wait | delayed_corridor | always_wait | 45% | [33%,57%] | 0% | — | — | — |
| delayed_Δ=-2/v4 | delayed_corridor | v4 | 55% | [43%,67%] | 100% | — | — | — |
| delayed_Δ=-2/pref_v2 | delayed_corridor | pref_v2 | 55% | [43%,67%] | 100% | 0% | — | — |
| delayed_Δ=-2/goal_v1 | delayed_corridor | goal_v1 | 55% | [43%,67%] | 100% | — | 0% | — |
| delayed_Δ=-2/oracle | delayed_corridor | oracle | 55% | [43%,67%] | 100% | — | — | — |
| delayed_Δ=2/always_wait | delayed_corridor | always_wait | 100% | [100%,100%] | 0% | — | — | — |
| delayed_Δ=2/v4 | delayed_corridor | v4 | 100% | [100%,100%] | 20% | — | — | — |
| delayed_Δ=2/pref_v2 | delayed_corridor | pref_v2 | 100% | [100%,100%] | 14% | 0% | — | — |
| delayed_Δ=2/goal_v1 | delayed_corridor | goal_v1 | 100% | [100%,100%] | 4% | — | 0% | — |
| delayed_Δ=2/oracle | delayed_corridor | oracle | 100% | [100%,100%] | 100% | — | — | — |
| tempt_high/always_wait | temptation_corridor | always_wait | 100% | [100%,100%] | 0% | — | — | 62% |
| tempt_high/v4 | temptation_corridor | v4 | 55% | [43%,67%] | 100% | — | — | 62% |
| tempt_high/pref_v2 | temptation_corridor | pref_v2 | 55% | [43%,67%] | 100% | 10% | — | 62% |
| tempt_high/goal_v1 | temptation_corridor | goal_v1 | 55% | [43%,67%] | 100% | — | 0% | 62% |
| tempt_high/oracle | temptation_corridor | oracle | 55% | [43%,67%] | 100% | — | — | 62% |

### Tutor Comparison Summary

| Family | v4 SBCR | pref_v2 | goal_v1 | Oracle |
|--------|---------|---------|---------|--------|
| elcb_po | 55% | 55% | 55% | 55% |
| delayed_Δ=-2 | 55% | 55% | 55% | 55% |
| delayed_Δ=2 | 100% | 100% | 100% | 100% |
| tempt_high | 55% | 55% | 55% | 55% |
