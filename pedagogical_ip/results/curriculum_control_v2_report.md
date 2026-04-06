# CCT-v2 + AEG-v1: Closed-Loop Curriculum

## 5-Phase Transfer

| θ | Curriculum | WR | LF | **B** | **C** | MCA_C | **D** | **E** | MCA_E |
|---|-----------|---|----|----|----|----|----|----|----|
| safe | ppmrb_only | 0% | 0.764 | **47%** | **41%** | 41% | **44%** | **50%** | 50% |
| safe | tic_heavy | 6% | 1.000 | **59%** | **50%** | 50% | **28%** | **56%** | 56% |
| safe | mixed_random | 0% | 0.948 | **47%** | **44%** | 44% | **53%** | **50%** | 50% |
| safe | self_disc_heavy | 0% | 0.970 | **53%** | **47%** | 47% | **53%** | **50%** | 50% |
| safe | cct_v2 | 0% | 0.899 | **47%** | **59%** | 59% | **41%** | **62%** | 62% |
| shiny | ppmrb_only | 0% | 0.768 | **56%** | **56%** | 56% | **34%** | **56%** | 56% |
| shiny | tic_heavy | 6% | 1.000 | **62%** | **53%** | 53% | **50%** | **56%** | 56% |
| shiny | mixed_random | 2% | 0.962 | **62%** | **59%** | 59% | **38%** | **47%** | 47% |
| shiny | self_disc_heavy | 0% | 0.959 | **38%** | **62%** | 62% | **44%** | **41%** | 41% |
| shiny | cct_v2 | 0% | 0.857 | **53%** | **50%** | 50% | **47%** | **56%** | 56% |

## State

| θ | Curriculum | τ | ν | **τ-ν** | γg | OTR | BdgBlk |
|---|-----------|---|---|---------|----|----|-------|
| safe | ppmrb_only | 0.464 | 0.235 | **+0.229** | 0.000 | 0.282 | 0 |
| safe | tic_heavy | 0.515 | 0.340 | **+0.175** | 0.032 | 0.616 | 0 |
| safe | mixed_random | 0.409 | 0.246 | **+0.163** | 0.000 | 0.190 | 0 |
| safe | self_disc_heavy | 0.464 | 0.249 | **+0.215** | 0.000 | 0.130 | 0 |
| safe | cct_v2 | 0.522 | 0.274 | **+0.248** | 0.000 | 0.153 | 0 |
| shiny | ppmrb_only | 0.493 | 0.249 | **+0.244** | 0.000 | 0.218 | 0 |
| shiny | tic_heavy | 0.547 | 0.266 | **+0.281** | 0.035 | 0.630 | 0 |
| shiny | mixed_random | 0.569 | 0.271 | **+0.298** | 0.020 | 0.222 | 0 |
| shiny | self_disc_heavy | 0.519 | 0.175 | **+0.344** | 0.000 | 0.116 | 0 |
| shiny | cct_v2 | 0.476 | 0.202 | **+0.274** | 0.000 | 0.185 | 0 |

## Closed-Loop Actionability

| θ | vs | ERCR | micro_PCR |
|---|---|------|----------|
| safe | cct_v2 vs mixed | 88% | 88% |
| safe | cct_v2 vs tic_heavy | 100% | 100% |
| shiny | cct_v2 vs mixed | 88% | 88% |
| shiny | cct_v2 vs tic_heavy | 100% | 100% |

## Phase A Subtype Distribution

| θ | Curriculum | Subtypes (top 3) |
|---|-----------|------------------|
| safe | ppmrb_only | boundary_obs(24), self_discovery_needed(20), temptation_repeat(19) |
| safe | tic_heavy | temptation_repeat(44), warn_rescue(36) |
| safe | mixed_random | self_discovery_needed(14), beneficial_novelty(12), temptation_repeat(12) |
| safe | self_disc_heavy | self_discovery_needed(27), false_suppression_cost(25), beneficial_novelty(16) |
| safe | cct_v2 | self_discovery_teach(37), self_discovery_needed(27), beneficial_novelty(16) |
| shiny | ppmrb_only | self_discovery_needed(24), temptation_repeat(21), self_discovery_teach(18) |
| shiny | tic_heavy | warn_rescue(42), temptation_repeat(38) |
| shiny | mixed_random | verified_warn(13), self_discovery_needed(11), warn_rescue(10) |
| shiny | self_disc_heavy | self_discovery_needed(28), false_suppression_cost(22), beneficial_novelty(16) |
| shiny | cct_v2 | self_discovery_teach(42), self_discovery_needed(38) |
