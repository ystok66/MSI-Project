# Phase 6.9 — 20-Seed Confirmation

> Seeds: 20 | Arms: canonical / B1 / B3 / all

## Exp A: Natural STOP Track

| θ | Arm | #T | C (mean±std) | C 95%CI | E | OTR | H_fam |
|---|-----|---|---|---|---|---|---|
| safe | canonical | 2.0 | 45%±0.187 | ±0.082 | 54% | 0.155 | 0.000 |
| safe | B1 | 2.0 | 45%±0.187 | ±0.082 | 55% | 0.208 | 0.000 |
| safe | B3 | 2.0 | 45%±0.187 | ±0.082 | 55% | 0.179 | 0.000 |
| safe | all | 2.0 | 45%±0.187 | ±0.082 | 55% | 0.176 | 0.000 |
| shiny | canonical | 2.7 | 49%±0.216 | ±0.095 | 39% | 0.057 | 0.647 |
| shiny | B1 | 2.25 | 54%±0.253 | ±0.111 | 44% | 0.072 | 0.685 |
| shiny | B3 | 2.25 | 54%±0.253 | ±0.111 | 44% | 0.072 | 0.685 |
| shiny | all | 2.25 | 54%±0.253 | ±0.111 | 44% | 0.072 | 0.685 |

### SDR (Natural)

| θ | Arm | SDR |
|---|---|:-:|
| safe | B1 | 1.0 |
| safe | B3 | 1.0 |
| safe | all | 1.0 |
| shiny | B1 | 0.455 |
| shiny | B3 | 0.455 |
| shiny | all | 0.455 |

## Exp A: Fixed-Dose Track (T_fix=6)

| θ | Arm | #T | C (mean±std) | C 95%CI | E | OTR | H_fam |
|---|-----|---|---|---|---|---|---|
| safe | canonical | 6.0 | 59%±0.213 | ±0.093 | 51% | 0.200 | 0.703 |
| safe | B1 | 6.0 | 59%±0.213 | ±0.093 | 51% | 0.232 | 0.703 |
| safe | B3 | 6.0 | 59%±0.213 | ±0.093 | 51% | 0.206 | 0.703 |
| safe | all | 6.0 | 59%±0.213 | ±0.093 | 51% | 0.198 | 0.703 |
| shiny | canonical | 6.0 | 54%±0.227 | ±0.100 | 51% | 0.163 | 0.878 |
| shiny | B1 | 6.0 | 54%±0.227 | ±0.100 | 51% | 0.167 | 0.878 |
| shiny | B3 | 6.0 | 54%±0.227 | ±0.100 | 51% | 0.167 | 0.878 |
| shiny | all | 6.0 | 52%±0.208 | ±0.091 | 51% | 0.180 | 0.816 |

### SDR (Fixed-Dose)

| θ | Arm | SDR |
|---|---|:-:|
| safe | B1 | 0.333 |
| safe | B3 | 0.333 |
| safe | all | 0.333 |
| shiny | B1 | 0.167 |
| shiny | B3 | 0.167 |
| shiny | all | 0.292 |

## Exp B: Pareto Analysis

| θ | Arm | C | OTR | ΔC | ΔOTR | ΔC/ΔOTR | S_0.25 | S_0.5 | S_1.0 |
|---|-----|---|-----|---|------|---------|--------|-------|-------|
| safe | canonical | 59% | 0.200 | +0.000 | +0.000 | 0.0 | 0.538 | 0.488 | 0.388 |
| safe | B1 | 59% | 0.232 | +0.000 | +0.032 | 0.0 | 0.530 | 0.472 | 0.356 |
| safe | B3 | 59% | 0.206 | +0.000 | +0.006 | 0.0 | 0.536 | 0.485 | 0.382 |
| safe | all | 59% | 0.198 | +0.000 | -0.002 | 0.0 | 0.538 | 0.489 | 0.390 |
| shiny | canonical | 54% | 0.163 | +0.000 | +0.000 | 0.0 | 0.497 | 0.456 | 0.375 |
| shiny | B1 | 54% | 0.167 | +0.000 | +0.004 | 0.0 | 0.496 | 0.454 | 0.371 |
| shiny | B3 | 54% | 0.167 | +0.000 | +0.004 | 0.0 | 0.496 | 0.454 | 0.371 |
| shiny | all | 52% | 0.180 | -0.013 | +0.017 | -0.72 | 0.480 | 0.435 | 0.345 |

## Exp C: OOD Robustness (Fixed-Dose)

| θ | Arm | OOD | C | E |
|---|-----|-----|---|---|
| safe | canonical | none | 62% | 66% |
| safe | canonical | sign_flip | 38% | 59% |
| safe | canonical | noise_heavy | 50% | 56% |
| safe | all | none | 62% | 66% |
| safe | all | sign_flip | 41% | 59% |
| safe | all | noise_heavy | 50% | 56% |
| shiny | canonical | none | 56% | 56% |
| shiny | canonical | sign_flip | 44% | 56% |
| shiny | canonical | noise_heavy | 53% | 50% |
| shiny | all | none | 56% | 59% |
| shiny | all | sign_flip | 44% | 53% |
| shiny | all | noise_heavy | 53% | 50% |

## Exp E: STOP Counterfactual Audit

| θ | Arm | Mean Regret | Max Regret | #STOP events |
|---|-----|:-:|:-:|:-:|
| safe | canonical | — | — | 0 |
| safe | all | — | — | 0 |
| shiny | canonical | — | — | 0 |
| shiny | all | — | — | 0 |
