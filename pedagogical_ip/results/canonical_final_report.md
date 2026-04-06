# Canonical Controller Final Verification

> Canonical = gated STOP + saturated FP (decay only, no rep penalty)

## Exp A: Canonical vs v13

| θ | Config | #T | #E | **C** | **E** | OTR | H_fam |
|---|--------|---|---|---|---|---|---|
| safe | v13 | 3 | 3 | **47%** | **53%** | 0.215 | 0.000 |
| safe | canonical | 2 | 3 | **47%** | **50%** | 0.175 | 0.000 |
| shiny | v13 | 3 | 3 | **41%** | **66%** | 0.150 | 0.369 |
| shiny | canonical | 4 | 3 | **47%** | **41%** | 0.129 | 0.336 |

## Exp B: Saturation Ablation

| θ | Mode | **C** | **E** | H_fam |
|---|------|---|---|---|
| safe | raw_FP | **47%** | **50%** | 0.000 |
| safe | +decay | **47%** | **50%** | 0.000 |
| safe | +decay+rep | **47%** | **50%** | 0.000 |
| shiny | raw_FP | **34%** | **50%** | 0.700 |
| shiny | +decay | **47%** | **41%** | 0.336 |
| shiny | +decay+rep | **50%** | **44%** | 0.462 |

## Exp C: Held-Out Family (Canonical)

| θ | Held-Out | **C** | **E** | #T | Δ_heldout |
|---|----------|---|---|---|---|
| safe | none | **47%** | **50%** | 2 | — |
| safe | PP-MRB | **47%** | **50%** | 2 | — |
| safe | TIC | **47%** | **50%** | 2 | — |
| safe | TIC-v4 | **56%** | **44%** | 4 | +9% |
| shiny | none | **47%** | **41%** | 4 | — |
| shiny | PP-MRB | **59%** | **44%** | 4 | +12% |
| shiny | TIC | **47%** | **41%** | 4 | — |
| shiny | TIC-v4 | **50%** | **59%** | 4 | +3% |

## Exp D: OOD Robustness (Canonical)

| θ | OOD | **C** | **E** | OTR |
|---|-----|---|---|---|
| safe | none | **47%** | **50%** | 0.175 |
| safe | sign_flip | **56%** | **50%** | 0.178 |
| safe | noise_heavy | **75%** | **56%** | 0.240 |
| shiny | none | **47%** | **41%** | 0.129 |
| shiny | sign_flip | **34%** | **56%** | 0.147 |
| shiny | noise_heavy | **44%** | **41%** | 0.196 |

## Exp E: STOP Trace (Representative Episodes)

### safe

| step | action | #T | M_base | G_warm | G_plateau | Δu |
|------|--------|---|--------|--------|-----------|----|
| 0 | TEACH | 0 | — | — | — | — |
| 1 | EVAL | 1 | — | — | — | — |
| 2 | EVAL | 1 | — | — | — | — |
| 3 | EVAL | 1 | — | — | — | — |
| 4 | TEACH | 1 | — | — | — | — |
| 5 | STOP | 2 | — | — | — | — |
### shiny

| step | action | #T | M_base | G_warm | G_plateau | Δu |
|------|--------|---|--------|--------|-----------|----|
| 0 | TEACH | 0 | — | — | — | — |
| 1 | EVAL | 1 | — | — | — | — |
| 2 | EVAL | 1 | — | — | — | — |
| 3 | EVAL | 1 | — | — | — | — |
| 4 | TEACH | 1 | — | — | — | — |
| 5 | STOP | 2 | — | — | — | — |

## Exp F: Family Usage Distribution

| θ | Mode | PP-MRB | TIC | TIC-v4 | CGC-v2 | H_fam |
|---|------|--------|-----|--------|--------|---|
| safe | raw_FP | 0% | 0% | 100% | 0% | — |
| safe | +decay | 0% | 0% | 100% | 0% | — |
| safe | +decay+rep | 0% | 0% | 100% | 0% | — |
| shiny | raw_FP | 50% | 3% | 47% | 0% | — |
| shiny | +decay | 31% | 0% | 69% | 0% | — |
| shiny | +decay+rep | 37% | 0% | 63% | 0% | — |
