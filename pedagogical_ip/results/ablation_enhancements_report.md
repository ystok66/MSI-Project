# Ablation Enhancement Experiment

> 6-arm comparison: canonical baseline vs three flag-controlled enhancements

> Seeds: 8 per arm

## Main Results

| θ | Arm | #T | #E | **C** | **E** | OTR | H_fam |
|---|-----|---|---|---|---|---|---|
| safe | canonical | 2 | 3 | **50%** | **44%** | 0.132 | 0.000 |
| safe | eig | 2 | 3 | **50%** | **44%** | 0.132 | 0.000 |
| safe | epi | 2 | 3 | **50%** | **44%** | 0.132 | 0.000 |
| safe | zpd | 2 | 3 | **50%** | **44%** | 0.132 | 0.000 |
| safe | eig+epi | 2 | 3 | **50%** | **44%** | 0.132 | 0.000 |
| safe | all | 2 | 3 | **50%** | **44%** | 0.132 | 0.000 |
| shiny | canonical | 2 | 3 | **53%** | **50%** | 0.132 | 0.677 |
| shiny | eig | 2 | 3 | **53%** | **50%** | 0.132 | 0.677 |
| shiny | epi | 2 | 3 | **53%** | **50%** | 0.132 | 0.677 |
| shiny | zpd | 2 | 3 | **53%** | **50%** | 0.132 | 0.677 |
| shiny | eig+epi | 2 | 3 | **53%** | **50%** | 0.132 | 0.677 |
| shiny | all | 2 | 3 | **53%** | **50%** | 0.132 | 0.677 |

## OOD Robustness (Selected Arms)

| θ | Arm | OOD | **C** | **E** |
|---|-----|-----|---|---|
| safe | canonical | none | **75%** | **25%** |
| safe | canonical | sign_flip | **75%** | **25%** |
| safe | canonical | noise_heavy | **75%** | **25%** |
| safe | eig | none | **75%** | **25%** |
| safe | eig | sign_flip | **75%** | **25%** |
| safe | eig | noise_heavy | **75%** | **25%** |
| safe | eig+epi | none | **75%** | **25%** |
| safe | eig+epi | sign_flip | **75%** | **25%** |
| safe | eig+epi | noise_heavy | **75%** | **25%** |
| shiny | canonical | none | **50%** | **75%** |
| shiny | canonical | sign_flip | **50%** | **75%** |
| shiny | canonical | noise_heavy | **50%** | **75%** |
| shiny | eig | none | **50%** | **75%** |
| shiny | eig | sign_flip | **50%** | **75%** |
| shiny | eig | noise_heavy | **50%** | **75%** |
| shiny | eig+epi | none | **50%** | **75%** |
| shiny | eig+epi | sign_flip | **50%** | **75%** |
| shiny | eig+epi | noise_heavy | **50%** | **75%** |
