# Ablation Enhancement Diagnostic Experiment

> Seeds: 8 | Fixed-dose track: T_fix=6

## Phase A: B2 Integration Verification

- **B2 Usage Rate**: 100.0%
- **B2 Active Episodes**: 2/2

B2 sample trace:

| Step | Lesson | risk_unc_risky | ΔU_risky | ΔU_safe |
|------|--------|:-:|:-:|:-:|
| 0 | sparse_invalid_advice | 1.0000 | 0.116270 | 0.046323 |
| 4 | verified_warn | 1.0000 | 0.235664 | 0.050141 |

## Natural STOP Track (canonical regime)

| θ | Arm | #T | **C** | **E** | OTR | H_fam |
|---|-----|---|---|---|---|---|
| safe | canonical | 2 | **56%** | **50%** | 0.033 | 0.000 |
| safe | eig | 2 | **56%** | **47%** | 0.033 | 0.000 |
| safe | epi | 2 | **56%** | **50%** | 0.033 | 0.000 |
| safe | zpd | 2 | **56%** | **50%** | 0.033 | 0.000 |
| safe | eig+epi | 2 | **56%** | **47%** | 0.033 | 0.000 |
| safe | all | 2 | **56%** | **50%** | 0.033 | 0.000 |
| shiny | canonical | 2 | **62%** | **44%** | 0.250 | 0.693 |
| shiny | eig | 2 | **62%** | **44%** | 0.250 | 0.693 |
| shiny | epi | 2 | **62%** | **44%** | 0.250 | 0.693 |
| shiny | zpd | 2 | **62%** | **44%** | 0.250 | 0.693 |
| shiny | eig+epi | 2 | **62%** | **44%** | 0.250 | 0.693 |
| shiny | all | 2 | **62%** | **44%** | 0.250 | 0.693 |

## Selection Divergence Rate (SDR)

| θ | Arm vs Canonical | SDR_macro |
|---|---|:-:|
| safe | eig | 1.000 |
| safe | epi | 0.000 |
| safe | zpd | 1.000 |
| safe | eig+epi | 1.000 |
| safe | all | 1.000 |
| shiny | eig | 0.500 |
| shiny | epi | 0.000 |
| shiny | zpd | 0.500 |
| shiny | eig+epi | 0.500 |
| shiny | all | 0.500 |

## Fixed-Dose Diagnostic Track (T_fix=6)

| θ | Arm | #T | **C** | **E** | OTR | H_fam |
|---|-----|---|---|---|---|---|
| safe | canonical | 6 | **38%** | **53%** | 0.174 | 0.893 |
| safe | eig | 6 | **41%** | **53%** | 0.174 | 0.893 |
| safe | epi | 6 | **41%** | **53%** | 0.174 | 0.893 |
| safe | zpd | 6 | **41%** | **53%** | 0.174 | 0.893 |
| safe | eig+epi | 6 | **41%** | **53%** | 0.174 | 0.893 |
| safe | all | 6 | **41%** | **53%** | 0.174 | 0.893 |
| shiny | canonical | 6 | **44%** | **44%** | 0.136 | 0.918 |
| shiny | eig | 6 | **44%** | **44%** | 0.136 | 0.918 |
| shiny | epi | 6 | **44%** | **44%** | 0.136 | 0.918 |
| shiny | zpd | 6 | **44%** | **44%** | 0.136 | 0.918 |
| shiny | eig+epi | 6 | **44%** | **44%** | 0.136 | 0.918 |
| shiny | all | 6 | **53%** | **38%** | 0.201 | 0.613 |

### Fixed-Dose SDR

| θ | Arm vs Canonical | SDR_macro |
|---|---|:-:|
| safe | eig | 0.333 |
| safe | epi | 0.000 |
| safe | zpd | 0.333 |
| safe | eig+epi | 0.333 |
| safe | all | 0.333 |
| shiny | eig | 0.167 |
| shiny | epi | 0.000 |
| shiny | zpd | 0.167 |
| shiny | eig+epi | 0.167 |
| shiny | all | 0.479 |
