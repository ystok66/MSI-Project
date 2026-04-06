# Cross-Family Phase Diagram Report

## D1: Delayed Commitment — Selectivity Phase Diagram

Δ = commit_depth - reveal_depth (base reveal=3)

| Δ | Strategy | SBCR | CI | WarnRate |
|---|----------|------|----|---------|
| -2 | always_wait | 40% | [26%,54%] | 0% |
| -2 | always_warn | 60% | [46%,74%] | 100% |
| -2 | v3 | 60% | [46%,74%] | 100% |
| -2 | oracle | 60% | [46%,74%] | 100% |
| -1 | always_wait | 40% | [26%,54%] | 0% |
| -1 | always_warn | 60% | [46%,74%] | 100% |
| -1 | v3 | 60% | [46%,74%] | 100% |
| -1 | oracle | 60% | [46%,74%] | 100% |
| 0 | always_wait | 40% | [26%,54%] | 0% |
| 0 | always_warn | 60% | [46%,74%] | 100% |
| 0 | v3 | 40% | [26%,54%] | 0% |
| 0 | oracle | 60% | [46%,74%] | 100% |
| 1 | always_wait | 40% | [26%,54%] | 0% |
| 1 | always_warn | 60% | [46%,74%] | 100% |
| 1 | v3 | 40% | [26%,54%] | 0% |
| 1 | oracle | 60% | [46%,74%] | 100% |
| 2 | always_wait | 100% | [100%,100%] | 0% |
| 2 | always_warn | 100% | [100%,100%] | 100% |
| 2 | v3 | 100% | [100%,100%] | 0% |
| 2 | oracle | 100% | [100%,100%] | 100% |

### v3 Selectivity Summary

| Δ | v3 WarnRate | v3 SBCR | Oracle SBCR | Match? |
|---|-----------|---------|-------------|--------|
| -2 | 100% | 60% | 60% | ≈ |
| -1 | 100% | 60% | 60% | ≈ |
| 0 | 0% | 40% | 60% | ≠ |
| 1 | 0% | 40% | 60% | ≠ |
| 2 | 0% | 100% | 100% | ≈ |

## D2: Distractor Cue — Robustness

| Salience | Strategy | SBCR | CI | WarnRate |
|----------|----------|------|----|---------|
| low | always_wait | 40% | [26%,54%] | 0% |
| low | always_warn | 60% | [46%,74%] | 100% |
| low | v3 | 40% | [26%,54%] | 0% |
| low | oracle | 60% | [46%,74%] | 100% |
| med | always_wait | 48% | [32%,62%] | 0% |
| med | always_warn | 60% | [46%,74%] | 100% |
| med | v3 | 48% | [32%,62%] | 0% |
| med | oracle | 60% | [46%,74%] | 100% |
| high | always_wait | 50% | [36%,64%] | 0% |
| high | always_warn | 60% | [46%,74%] | 100% |
| high | v3 | 50% | [36%,64%] | 0% |
| high | oracle | 60% | [46%,74%] | 100% |
