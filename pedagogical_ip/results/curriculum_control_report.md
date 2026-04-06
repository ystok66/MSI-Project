# CCT-v1: Curriculum-Control Tutor

## Experiment A: 5-Phase Transfer

| θ | Curriculum | WR | **B** | **C** | MCA_C | **D** | **E** | MCA_E |
|---|-----------|----|----|----|----|----|----|----|
| safe | ppmrb_only | 2% | **38%** | **44%** | 44% | **38%** | **44%** | 44% |
| safe | tic_heavy | 2% | **38%** | **44%** | 44% | **38%** | **44%** | 44% |
| safe | mixed_random | 2% | **38%** | **44%** | 44% | **38%** | **44%** | 44% |
| safe | self_disc_heavy | 2% | **38%** | **44%** | 44% | **38%** | **44%** | 44% |
| safe | cct_v1 | 2% | **38%** | **44%** | 44% | **38%** | **44%** | 44% |
| shiny | ppmrb_only | 0% | **50%** | **66%** | 66% | **53%** | **47%** | 47% |
| shiny | tic_heavy | 0% | **50%** | **66%** | 66% | **53%** | **47%** | 47% |
| shiny | mixed_random | 0% | **50%** | **66%** | 66% | **53%** | **47%** | 47% |
| shiny | self_disc_heavy | 0% | **50%** | **66%** | 66% | **53%** | **47%** | 47% |
| shiny | cct_v1 | 0% | **50%** | **66%** | 66% | **53%** | **47%** | 47% |

## State + Curriculum

| θ | Curriculum | τ | ν | **τ-ν** | γg | OTR | #Unique |
|---|-----------|---|---|---------|----|----|--------|
| safe | ppmrb_only | 0.426 | 0.356 | **+0.070** | 0.007 | 0.375 | 2 |
| safe | tic_heavy | 0.426 | 0.361 | **+0.065** | 0.010 | 0.389 | 2 |
| safe | mixed_random | 0.426 | 0.361 | **+0.065** | 0.010 | 0.389 | 6 |
| safe | self_disc_heavy | 0.426 | 0.356 | **+0.070** | 0.007 | 0.375 | 4 |
| safe | cct_v1 | 0.426 | 0.356 | **+0.070** | 0.007 | 0.375 | 2 |
| shiny | ppmrb_only | 0.556 | 0.214 | **+0.342** | 0.000 | 0.255 | 2 |
| shiny | tic_heavy | 0.556 | 0.214 | **+0.342** | 0.000 | 0.255 | 2 |
| shiny | mixed_random | 0.556 | 0.214 | **+0.342** | 0.000 | 0.255 | 6 |
| shiny | self_disc_heavy | 0.556 | 0.214 | **+0.342** | 0.000 | 0.255 | 4 |
| shiny | cct_v1 | 0.556 | 0.214 | **+0.342** | 0.000 | 0.255 | 1 |

## Probes

| θ | Curriculum | EP | VA | IA |
|---|-----------|----|----|----|
| safe | ppmrb_only | 0.502 | 0.741 | 0.072 |
| safe | tic_heavy | 0.501 | 0.741 | 0.071 |
| safe | mixed_random | 0.501 | 0.741 | 0.071 |
| safe | self_disc_heavy | 0.502 | 0.741 | 0.072 |
| safe | cct_v1 | 0.502 | 0.741 | 0.072 |
| shiny | ppmrb_only | 0.598 | 0.585 | 0.468 |
| shiny | tic_heavy | 0.598 | 0.585 | 0.468 |
| shiny | mixed_random | 0.598 | 0.585 | 0.468 |
| shiny | self_disc_heavy | 0.598 | 0.585 | 0.468 |
| shiny | cct_v1 | 0.598 | 0.585 | 0.468 |

## Experiment B: Actionability Audit

| θ | Comparison | micro_PCR | curriculum_CR |
|---|-----------|-----------|---------------|
| safe | cct_v1 vs mixed | 0% | 95% |
| safe | cct_v1 vs tic_heavy | 0% | 100% |
| shiny | cct_v1 vs mixed | 0% | 96% |
| shiny | cct_v1 vs tic_heavy | 0% | 100% |
