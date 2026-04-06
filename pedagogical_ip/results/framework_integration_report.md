# Framework Integration Report

## I2: Temptation Corridor — Preference-Aware Tutor

| Temptation | Strategy | SBCR | CI | WarnRate | PrefAcc |
|------------|----------|------|----|---------|---------|
| low | always_wait | 44% | [32%,58%] | 0% | — |
| low | always_warn | 60% | [46%,74%] | 100% | — |
| low | v4 | 60% | [46%,74%] | 100% | — |
| low | pref_v1 | 60% | [46%,74%] | 100% | 0% |
| low | oracle | 60% | [46%,74%] | 100% | — |
| med | always_wait | 100% | [100%,100%] | 0% | — |
| med | always_warn | 60% | [46%,74%] | 100% | — |
| med | v4 | 60% | [46%,74%] | 100% | — |
| med | pref_v1 | 60% | [46%,74%] | 100% | 0% |
| med | oracle | 60% | [46%,74%] | 100% | — |
| high | always_wait | 100% | [100%,100%] | 0% | — |
| high | always_warn | 60% | [46%,74%] | 100% | — |
| high | v4 | 60% | [46%,74%] | 100% | — |
| high | pref_v1 | 60% | [46%,74%] | 100% | 0% |
| high | oracle | 60% | [46%,74%] | 100% | — |

### pref_v1 vs v4 Comparison

| Tempt | v4 WR | v4 SBCR | pref WR | pref SBCR | pref Acc | Oracle |
|-------|-------|---------|---------|-----------|---------|--------|
| low | 100% | 60% | 100% | 60% | 0% | 60% |
| med | 100% | 60% | 100% | 60% | 0% | 60% |
| high | 100% | 60% | 100% | 60% | 0% | 60% |

## I4: Cross-Family Robustness (4 families)

| Family | Strategy | SBCR | CI | WarnRate | PrefAcc |
|--------|----------|------|----|---------|---------|
| elcb_po | always_wait | 40% | [26%,54%] | 0% | — |
| elcb_po | v4 | 60% | [46%,74%] | 100% | — |
| elcb_po | pref_v1 | 60% | [46%,74%] | 100% | — |
| elcb_po | oracle | 60% | [46%,74%] | 100% | — |
| delayed_Δ=-2 | always_wait | 40% | [26%,54%] | 0% | — |
| delayed_Δ=-2 | v4 | 60% | [46%,74%] | 100% | — |
| delayed_Δ=-2 | pref_v1 | 60% | [46%,74%] | 100% | — |
| delayed_Δ=-2 | oracle | 60% | [46%,74%] | 100% | — |
| delayed_Δ=2 | always_wait | 100% | [100%,100%] | 0% | — |
| delayed_Δ=2 | v4 | 100% | [100%,100%] | 25% | — |
| delayed_Δ=2 | pref_v1 | 100% | [100%,100%] | 0% | — |
| delayed_Δ=2 | oracle | 100% | [100%,100%] | 100% | — |
| distractor_high | always_wait | 50% | [36%,64%] | 0% | — |
| distractor_high | v4 | 60% | [46%,74%] | 100% | — |
| distractor_high | pref_v1 | 60% | [46%,74%] | 100% | — |
| distractor_high | oracle | 60% | [46%,74%] | 100% | — |
| tempt_high | always_wait | 100% | [100%,100%] | 0% | — |
| tempt_high | v4 | 60% | [46%,74%] | 100% | — |
| tempt_high | pref_v1 | 60% | [46%,74%] | 100% | 0% |
| tempt_high | oracle | 60% | [46%,74%] | 100% | — |

### Framework Summary: v4 vs pref_v1

| Family | v4 SBCR | pref SBCR | Oracle | v4=Oracle? | pref=Oracle? |
|--------|---------|-----------|--------|-----------|-------------|
| elcb_po | 60% | 60% | 60% | ✅ | ✅ |
| delayed_Δ=-2 | 60% | 60% | 60% | ✅ | ✅ |
| delayed_Δ=2 | 100% | 100% | 100% | ✅ | ✅ |
| distractor_high | 60% | 60% | 60% | ✅ | ✅ |
| tempt_high | 60% | 60% | 60% | ✅ | ✅ |
