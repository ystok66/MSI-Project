# Stochastic Agent + Preference Inference Report

## J1: Stochastic Agent Behavior Validation

| β | θ | P(safe) | P(tempt) |
|---|---|---------|----------|
| 2.0 | safe | 0.946 | 0.054 |
| 2.0 | risky | 0.61 | 0.39 |
| 2.0 | shiny | 0.093 | 0.907 |
| 2.0 | shortcut | 0.822 | 0.178 |
| 2.0 | neutral | 0.852 | 0.148 |
| 4.0 | safe | 0.95 | 0.05 |
| 4.0 | risky | 0.708 | 0.292 |
| 4.0 | shiny | 0.052 | 0.948 |
| 4.0 | shortcut | 0.926 | 0.074 |
| 4.0 | neutral | 0.937 | 0.063 |
| 8.0 | safe | 0.95 | 0.05 |
| 8.0 | risky | 0.843 | 0.157 |
| 8.0 | shiny | 0.05 | 0.95 |
| 8.0 | shortcut | 0.949 | 0.051 |
| 8.0 | neutral | 0.95 | 0.05 |

## J2: Preference Posterior Convergence (20 obs, 50 trials)

| True θ | PrefAcc |
|--------|--------|
| safe | 58% |
| risky | 84% |
| shiny | 100% |
| shortcut | 36% |
| neutral | 22% |

**Mean PrefAcc: 60.0%** (chance = 20.0%)

## J3: Temptation Corridor with Stochastic Agent

| Tempt | Strategy | SBCR | CI | WarnRate | PrefAcc | AgentSafe% |
|-------|----------|------|----|---------|---------|-----------|
| low | always_wait | 45% | [33%,57%] | 0% | — | 64% |
| low | always_warn | 55% | [43%,67%] | 100% | — | 64% |
| low | v4 | 55% | [43%,67%] | 100% | — | 64% |
| low | pref_v2 | 55% | [43%,67%] | 100% | 18% | 64% |
| low | oracle | 55% | [43%,67%] | 100% | — | 64% |
| med | always_wait | 95% | [88%,100%] | 0% | — | 62% |
| med | always_warn | 55% | [43%,67%] | 100% | — | 62% |
| med | v4 | 55% | [43%,67%] | 100% | — | 62% |
| med | pref_v2 | 55% | [43%,67%] | 100% | 6% | 62% |
| med | oracle | 55% | [43%,67%] | 100% | — | 62% |
| high | always_wait | 100% | [100%,100%] | 0% | — | 62% |
| high | always_warn | 55% | [43%,67%] | 100% | — | 62% |
| high | v4 | 55% | [43%,67%] | 100% | — | 62% |
| high | pref_v2 | 55% | [43%,67%] | 100% | 10% | 62% |
| high | oracle | 55% | [43%,67%] | 100% | — | 62% |

### pref_v2 vs v4 vs oracle

| Tempt | v4 WR | v4 SBCR | pref_v2 WR | pref_v2 SBCR | pref_v2 Acc | Oracle |
|-------|-------|---------|-----------|-------------|------------|--------|
| low | 100% | 55% | 100% | 55% | 18% | 55% |
| med | 100% | 55% | 100% | 55% | 6% | 55% |
| high | 100% | 55% | 100% | 55% | 10% | 55% |
