# Gated STOP: Exp B + C + Summary

## Exp B: Post-STOP Diagnostic

| θ | Mode | #T@STOP | margin | m_base | g_warm | g_plateau | Δu |
|---|------|---------|--------|--------|--------|-----------|----|
| safe | single | 2.8 | 0.7881 | 100% | 100% | 100% | — |
| safe | gated | 2.0 | 0.0 | 100% | 100% | 100% | — |
| shiny | single | 3.6 | 0.2057 | 100% | 100% | 100% | — |
| shiny | gated | 4.2 | 0.0 | 100% | 100% | 100% | — |

## Exp C: Cross-Family Robustness (Gated STOP)

| θ | Held-Out | **C** | **E** | #T |
|---|----------|---|---|---|
| safe | none | **53%** | **50%** | 2 |
| safe | PP-MRB | **53%** | **50%** | 2 |
| safe | TIC | **53%** | **50%** | 2 |
| safe | TIC-v4 | **34%** | **44%** | 3 |
| shiny | none | **59%** | **41%** | 4 |
| shiny | PP-MRB | **31%** | **44%** | 5 |
| shiny | TIC | **59%** | **41%** | 4 |
| shiny | TIC-v4 | **56%** | **59%** | 4 |

## OOD Credibility (Gated STOP)

| θ | OOD | **C** | **E** | OTR |
|---|-----|---|---|---|
| safe | none | **53%** | **50%** | 0.105 |
| safe | sign_flip | **50%** | **47%** | 0.159 |
| safe | noise_heavy | **50%** | **50%** | 0.040 |
| shiny | none | **59%** | **41%** | 0.205 |
| shiny | sign_flip | **56%** | **47%** | 0.163 |
| shiny | noise_heavy | **41%** | **50%** | 0.168 |

## Final Summary: v13 → Gated STOP

| θ | v13 C | gated C | Δ | v13 #T | gated #T |
|---|------|--------|---|--------|----------|
| safe | 44% | 53% | +9% | 3 | 2 |
| shiny | 59% | 59% | — | 4 | 4 |
