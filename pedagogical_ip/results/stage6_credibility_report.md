# Stage-6: Credibility Closure

## Exp A: Adversarial OOD

| θ | OOD Mode | **C** | **E** | OTR | ν | #PW |
|---|----------|---|---|---|---|---|
| safe | none | **59%** | **53%** | 0.125 | 0.231 | 21 |
| safe | sign_flip | **50%** | **31%** | 0.112 | 0.297 | 21 |
| safe | noise_heavy | **41%** | **44%** | 0.129 | 0.298 | 22 |
| safe | scale_shift | **53%** | **44%** | 0.162 | 0.335 | 22 |
| shiny | none | **69%** | **47%** | 0.079 | 0.205 | 28 |
| shiny | sign_flip | **50%** | **56%** | 0.102 | 0.261 | 31 |
| shiny | noise_heavy | **50%** | **44%** | 0.075 | 0.254 | 29 |
| shiny | scale_shift | **50%** | **56%** | 0.102 | 0.261 | 31 |

### Held-Out Family

| θ | Held-Out | **C** | **E** | OTR |
|---|----------|---|---|---|
| safe | none | **59%** | **53%** | 0.125 |
| safe | PP-MRB | **59%** | **47%** | 0.224 |
| safe | TIC | **56%** | **56%** | 0.158 |
| safe | TIC-v4 | **59%** | **53%** | 0.125 |
| shiny | none | **69%** | **47%** | 0.079 |
| shiny | PP-MRB | **41%** | **56%** | 0.174 |
| shiny | TIC | **56%** | **50%** | 0.040 |
| shiny | TIC-v4 | **69%** | **47%** | 0.079 |

## Exp B: STOP Calibration Audit

| θ | Avg Margin | Monotonicity | #Stops |
|---|-----------|-------------|-------|
| safe | 0.0000 | 0.0000 | 1.0 |
| shiny | 0.4046 | 0.0000 | 1.0 |

## Exp C: EVAL Calibration Audit

| θ | Rank-Change Rate | #Evals | Trigger |
|---|-----------------|--------|--------|
| safe | 0.0% | 3.0 | uncertainty:24 |
| shiny | 0.0% | 3.0 | uncertainty:24 |

## Exp D: OTR Decomposition

| θ | OTR Total | OTR Teach | OTR Eval | Fam Repeats |
|---|----------|----------|---------|------------|
| safe | 0.125 | 0.050 | 0.075 | 0 |
| shiny | 0.079 | 0.032 | 0.047 | 1 |

## Exp E: Actionability Regression (θ-adaptive)

| θ | Term | AM | PCR |
|---|------|----|-----|
| safe | G | 0.058915 | 66.7% |
| safe | G_hier | 0.000766 | 0.0% |
| safe | G_res | 0.000461 | 0.0% |
| safe | G_pw | 0.047141 | 66.7% |
| safe | U | 0.000295 | 16.7% |
| safe | H | 0.006178 | 0.0% |
| shiny | G | 0.059003 | 60.7% |
| shiny | G_hier | 0.000766 | 0.0% |
| shiny | G_res | 0.000461 | 0.0% |
| shiny | G_pw | 0.047240 | 60.7% |
| shiny | U | 0.000309 | 15.2% |
| shiny | H | 0.006185 | 0.0% |
