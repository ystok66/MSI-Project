# CCT-v3: Mastery-Aware Closed-Loop Curriculum

## 5-Phase Transfer

| θ | Curriculum | WR | LF | #Teach | #Eval | Stopped | **B** | **C** | MCA_C | **D** | **E** | MCA_E |
|---|-----------|---|---|--------|-------|---------|----|----|----|----|----|----|----|
| safe | ppmrb_only | 0% | 0.778 | 12 | 0 | 0% | **50%** | **0%** | 0% | **75%** | **25%** | 25% |
| safe | tic_heavy | 3% | 1.000 | 12 | 0 | 0% | **50%** | **0%** | 0% | **75%** | **25%** | 25% |
| safe | mixed_random | 1% | 0.928 | 12 | 0 | 0% | **50%** | **0%** | 0% | **75%** | **25%** | 25% |
| safe | self_disc_heavy | 0% | 0.953 | 12 | 0 | 0% | **50%** | **0%** | 0% | **75%** | **25%** | 25% |
| safe | cct_v2_style | 1% | 0.928 | 12 | 0 | 0% | **50%** | **0%** | 0% | **75%** | **25%** | 25% |
| safe | cct_v3 | 0% | 0.932 | 5 | 1 | 100% | **59%** | **75%** | 75% | **0%** | **75%** | 75% |
| shiny | ppmrb_only | 0% | 0.781 | 12 | 0 | 0% | **97%** | **44%** | 44% | **25%** | **50%** | 50% |
| shiny | tic_heavy | 5% | 1.000 | 12 | 0 | 0% | **97%** | **28%** | 28% | **25%** | **50%** | 50% |
| shiny | mixed_random | 1% | 0.981 | 12 | 0 | 0% | **97%** | **47%** | 47% | **25%** | **50%** | 50% |
| shiny | self_disc_heavy | 0% | 0.975 | 12 | 0 | 0% | **100%** | **50%** | 50% | **25%** | **50%** | 50% |
| shiny | cct_v2_style | 1% | 0.981 | 12 | 0 | 0% | **97%** | **47%** | 47% | **25%** | **50%** | 50% |
| shiny | cct_v3 | 5% | 0.902 | 5 | 1 | 100% | **50%** | **75%** | 75% | **75%** | **50%** | 50% |

## State + Mastery

| θ | Curriculum | τ | ν | **τ-ν** | γg | OTR | MPG | BdgBlk |
|---|-----------|---|---|---------|----|----|-----|-------|
| safe | ppmrb_only | 0.300 | 0.222 | **+0.078** | 0.000 | 0.060 | 0.0000 | 0 |
| safe | tic_heavy | 0.366 | 0.258 | **+0.108** | 0.021 | 0.746 | 0.0000 | 0 |
| safe | mixed_random | 0.322 | 0.229 | **+0.093** | 0.003 | 0.384 | 0.0000 | 0 |
| safe | self_disc_heavy | 0.300 | 0.226 | **+0.074** | 0.000 | 0.000 | 0.0000 | 0 |
| safe | cct_v2_style | 0.322 | 0.229 | **+0.093** | 0.003 | 0.384 | 0.0000 | 0 |
| safe | cct_v3 | 0.705 | 0.399 | **+0.306** | 0.000 | 0.307 | 0.3630 | 0 |
| shiny | ppmrb_only | 0.431 | 0.393 | **+0.038** | 0.000 | 0.315 | 0.0000 | 0 |
| shiny | tic_heavy | 0.465 | 0.427 | **+0.038** | 0.030 | 0.716 | 0.0000 | 0 |
| shiny | mixed_random | 0.469 | 0.400 | **+0.069** | 0.009 | 0.315 | 0.0000 | 0 |
| shiny | self_disc_heavy | 0.475 | 0.387 | **+0.088** | 0.000 | 0.263 | 0.0000 | 0 |
| shiny | cct_v2_style | 0.469 | 0.400 | **+0.069** | 0.009 | 0.315 | 0.0000 | 0 |
| shiny | cct_v3 | 0.631 | 0.287 | **+0.344** | 0.014 | 0.080 | 0.2370 | 0 |

## Closed-Loop Actionability

| θ | vs | ERCR | micro_PCR |
|---|---|------|----------|
| safe | cct_v3 vs mixed | 85% | 85% |
| safe | cct_v3 vs tic_heavy | 96% | 90% |
| shiny | cct_v3 vs mixed | 92% | 92% |
| shiny | cct_v3 vs tic_heavy | 88% | 75% |

## Phase A Subtype Distribution

| θ | Curriculum | Subtypes (top 3) |
|---|-----------|------------------|
| safe | ppmrb_only | temptation_repeat(26), boundary_obs(25), self_discovery_teach(25) |
| safe | tic_heavy | warn_rescue(53), temptation_repeat(43) |
| safe | mixed_random | self_discovery_needed(18), temptation_repeat(15), sparse_valid_advice(12) |
| safe | self_disc_heavy | self_discovery_needed(41), beneficial_novelty(23), false_suppression_cost(19) |
| safe | cct_v2_style | self_discovery_needed(18), temptation_repeat(15), sparse_valid_advice(12) |
| safe | cct_v3 | self_discovery_needed(9), beneficial_novelty(8), false_suppression_cost(8) |
| shiny | ppmrb_only | self_discovery_needed(30), self_discovery_teach(26), boundary_obs(21) |
| shiny | tic_heavy | temptation_repeat(54), warn_rescue(42) |
| shiny | mixed_random | beneficial_novelty(19), self_discovery_needed(15), temptation_repeat(14) |
| shiny | self_disc_heavy | self_discovery_needed(33), false_suppression_cost(28), beneficial_novelty(24) |
| shiny | cct_v2_style | beneficial_novelty(19), self_discovery_needed(15), temptation_repeat(14) |
| shiny | cct_v3 | temptation_repeat(12), self_discovery_needed(11), warn_rescue(8) |
