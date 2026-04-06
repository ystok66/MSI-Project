# Stage-6.8: v13.3 STOP Coefficient Calibration

## Exp A: STOP Coefficient Ablation

| θ | STOP Mode | #T | **C** | **E** | OTR |
|---|----------|---|---|---|---|
| safe | shared | 2 | **41%** | **56%** | 0.116 |
| safe | per-θ_intercept | 2 | **59%** | **41%** | 0.125 |
| safe | per-θ_coeff | 2 | **59%** | **41%** | 0.125 |
| safe | hybrid | 2 | **59%** | **41%** | 0.125 |
| shiny | shared | 3 | **44%** | **53%** | 0.099 |
| shiny | per-θ_intercept | 2 | **50%** | **50%** | 0.097 |
| shiny | per-θ_coeff | 2 | **50%** | **50%** | 0.097 |
| shiny | hybrid | 2 | **50%** | **50%** | 0.097 |

## Exp B: v13 vs v13.2 vs v13.3

| θ | Config | #T | #E | **C** | **E** | OTR | G_pw PCR |
|---|--------|---|---|---|---|---|---|
| safe | v13 | 2 | 3 | **41%** | **56%** | 0.116 | 66% |
| safe | v13.2(int) | 2 | 3 | **59%** | **41%** | 0.125 | 67% |
| safe | v13.3 | 2 | 3 | **59%** | **41%** | 0.125 | 67% |
| shiny | v13 | 3 | 3 | **44%** | **53%** | 0.099 | 62% |
| shiny | v13.2(int) | 2 | 3 | **50%** | **50%** | 0.097 | 66% |
| shiny | v13.3 | 2 | 3 | **50%** | **50%** | 0.097 | 66% |

## Exp C: Same-Dose Fair Comparison (#T=4, #E=3)

| θ | Config | **C** | **E** |
|---|--------|---|---|
| safe | v13 | **41%** | **56%** |
| safe | v13.3 | **59%** | **41%** |
| shiny | v13 | **44%** | **53%** |
| shiny | v13.3 | **50%** | **50%** |

## Exp D: v13.3 Credibility Regression

| θ | OOD | **C** | **E** | OTR |
|---|-----|---|---|---|
| safe | none | **59%** | **41%** | 0.125 |
| safe | sign_flip | **56%** | **50%** | 0.066 |
| safe | noise_heavy | **75%** | **56%** | 0.118 |
| shiny | none | **50%** | **50%** | 0.097 |
| shiny | sign_flip | **53%** | **50%** | 0.000 |
| shiny | noise_heavy | **53%** | **53%** | 0.102 |

### Held-Out Family (v13.3)

| θ | Held-Out | **C** | **E** |
|---|----------|---|---|
| safe | none | **59%** | **41%** |
| safe | PP-MRB | **59%** | **41%** |
| safe | TIC | **59%** | **41%** |
| safe | TIC-v4 | **53%** | **50%** |
| shiny | none | **50%** | **50%** |
| shiny | PP-MRB | **50%** | **50%** |
| shiny | TIC | **50%** | **50%** |
| shiny | TIC-v4 | **38%** | **50%** |
