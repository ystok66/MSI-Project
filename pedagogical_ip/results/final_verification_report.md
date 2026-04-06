# Final Verification: Gated STOP + Saturated Family Prior

## Exp A: Gated STOP Canonicalization

| θ | Mode | #T | #E | **C** | **E** | OTR | Fam Entropy |
|---|------|---|---|---|---|---|---|
| safe | v13 | 3 | 3 | **19%** | **34%** | 0.112 | 0.000 |
| safe | gated+sat | 2 | 3 | **34%** | **41%** | 0.234 | 0.000 |
| shiny | v13 | 3 | 3 | **69%** | **62%** | 0.080 | 0.230 |
| shiny | gated+sat | 5 | 3 | **62%** | **56%** | 0.161 | 0.524 |

## Exp B: Family Prior Saturation Ablation

| θ | Mode | **C** | **E** | Fam Entropy |
|---|------|---|---|---|
| safe | raw_FP | **34%** | **41%** | 0.000 |
| safe | +decay | **34%** | **41%** | 0.000 |
| safe | +decay+rep | **34%** | **41%** | 0.000 |
| shiny | raw_FP | **44%** | **38%** | 0.737 |
| shiny | +decay | **66%** | **53%** | 0.467 |
| shiny | +decay+rep | **62%** | **56%** | 0.524 |

## Exp C: Held-Out Family (Gated+Sat+Rep)

| θ | Held-Out | **C** | **E** | #T |
|---|----------|---|---|---|
| safe | none | **34%** | **41%** | 2 |
| safe | PP-MRB | **34%** | **41%** | 2 |
| safe | TIC | **34%** | **41%** | 2 |
| safe | TIC-v4 | **62%** | **62%** | 3 |
| shiny | none | **62%** | **56%** | 5 |
| shiny | PP-MRB | **59%** | **34%** | 4 |
| shiny | TIC | **66%** | **66%** | 5 |
| shiny | TIC-v4 | **47%** | **38%** | 4 |

### OOD Credibility

| θ | OOD | **C** | **E** | OTR |
|---|-----|---|---|---|
| safe | none | **34%** | **41%** | 0.234 |
| safe | sign_flip | **62%** | **41%** | 0.210 |
| safe | noise_heavy | **69%** | **47%** | 0.066 |
| shiny | none | **62%** | **56%** | 0.161 |
| shiny | sign_flip | **41%** | **50%** | 0.246 |
| shiny | noise_heavy | **44%** | **44%** | 0.214 |

## Summary

| θ | v13 C | gated+sat C | Δ | PP-MRB drop |
|---|------|-----------|---|------------|
| safe | 19% | 34% | +16% | — |
| shiny | 69% | 62% | -6% | +3% |
