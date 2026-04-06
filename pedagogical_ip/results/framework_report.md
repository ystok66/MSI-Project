# Pedagogical Decision Framework Report

## H2: Selectivity Law Calibration

### p_self vs Empirical Self-Discovery

| Δ | p_self | emp_self | WarnRate | SBCR |
|---|-------|---------|---------|------|
| -4 | 0.02 | 0.00 | 100% | 60% |
| -3 | 0.05 | 0.00 | 100% | 60% |
| -2 | 0.12 | 0.00 | 100% | 60% |
| -1 | 0.27 | 0.00 | 100% | 60% |
| 0 | 0.50 | 0.00 | 100% | 60% |
| 1 | 0.73 | 0.14 | 100% | 60% |
| 2 | 0.88 | 0.29 | 25% | 100% |
| 3 | 0.95 | 0.43 | 18% | 100% |
| 4 | 0.98 | 0.57 | 12% | 100% |

### Calibration Metrics

- **ECE(p_self)**: 0.3413
- **WarnRate curve monotonic**: True
- **Transition zone**: (2, 4)
- **Slope at Δ=0**: 0.0

## H4: Cross-Family Robustness

| Family | Strategy | SBCR | CI | WarnRate | DVOI |
|--------|----------|------|----|---------|----- |
| elcb_po | always_wait | 40% | [26%,54%] | 0% | 0.0037 |
| elcb_po | v4 | 60% | [46%,74%] | 100% | 0.0034 |
| elcb_po | oracle | 60% | [46%,74%] | 100% | 0.0034 |
| delayed_Δ=-2 | always_wait | 40% | [26%,54%] | 0% | 0.0038 |
| delayed_Δ=-2 | v4 | 60% | [46%,74%] | 100% | 0.0035 |
| delayed_Δ=-2 | oracle | 60% | [46%,74%] | 100% | 0.0035 |
| delayed_Δ=0 | always_wait | 40% | [26%,54%] | 0% | 0.0037 |
| delayed_Δ=0 | v4 | 60% | [46%,74%] | 100% | 0.0034 |
| delayed_Δ=0 | oracle | 60% | [46%,74%] | 100% | 0.0034 |
| delayed_Δ=2 | always_wait | 100% | [100%,100%] | 0% | 0.0024 |
| delayed_Δ=2 | v4 | 100% | [100%,100%] | 25% | 0.0024 |
| delayed_Δ=2 | oracle | 100% | [100%,100%] | 100% | 0.0023 |
| distractor_low | always_wait | 40% | [26%,54%] | 0% | 0.0036 |
| distractor_low | v4 | 60% | [46%,74%] | 100% | 0.0033 |
| distractor_low | oracle | 60% | [46%,74%] | 100% | 0.0033 |
| distractor_high | always_wait | 50% | [36%,64%] | 0% | 0.0014 |
| distractor_high | v4 | 60% | [46%,74%] | 100% | 0.0013 |
| distractor_high | oracle | 60% | [46%,74%] | 100% | 0.0013 |

### v4 Selectivity Summary

| Family | v4 WR | v4 SBCR | Oracle SBCR | Match |
|--------|-------|---------|-------------|-------|
| elcb_po | 100% | 60% | 60% | ✅ |
| delayed_Δ=-2 | 100% | 60% | 60% | ✅ |
| delayed_Δ=0 | 100% | 60% | 60% | ✅ |
| delayed_Δ=2 | 25% | 100% | 100% | ✅ |
| distractor_low | 100% | 60% | 60% | ✅ |
| distractor_high | 100% | 60% | 60% | ✅ |
