# CGC-v2: Factor-Vector Goal Experiment

## Train Compositions (12 vectors)

| θ | Strategy | SBCR | WR | WR(aln) | WR(cnf) | **SelGap** | FactorAcc | ExactAcc |
|---|----------|------|----|---------|---------|-----------|-----------|----------|
| safe | v1_1 | 65% | 94% | 83% | 100% | **0.172** | — | — |
| safe | joint_v2 | 65% | 82% | 57% | 92% | **0.344** | — | — |
| safe | cajt_v3 | 65% | 71% | 34% | 67% | **0.322** | — | — |
| safe | factor_exact | 65% | 62% | 44% | 58% | **0.139** | 0.302 | 0.000 |
| safe | factor_cajt | 65% | 62% | 44% | 58% | **0.139** | 0.298 | 0.000 |
| safe | oracle | 65% | 6% | 0% | 0% | **0.000** | — | — |
| shiny | v1_1 | 78% | 94% | 78% | 100% | **0.222** | — | — |
| shiny | joint_v2 | 78% | 96% | 83% | 100% | **0.167** | — | — |
| shiny | cajt_v3 | 78% | 81% | 47% | 89% | **0.417** | — | — |
| shiny | factor_exact | 78% | 65% | 78% | 43% | **-0.344** | 0.298 | 0.000 |
| shiny | factor_cajt | 78% | 65% | 78% | 43% | **-0.344** | 0.302 | 0.000 |
| shiny | oracle | 78% | 7% | 0% | 0% | **0.000** | — | — |

### SelGap Comparison

| θ | v1.1 | joint_v2 | cajt_v3 | factor_exact | **factor_cajt** | oracle |
|---|------|----------|---------|-------------|----------------|--------|
| safe | 0.172 | 0.344 | 0.322 | 0.139 | **0.139** | 0.000 |
| shiny | 0.222 | 0.167 | 0.417 | -0.344 | **-0.344** | 0.000 |

## Held-Out Compositions (6 novel vectors)

| θ | Strategy | SBCR | SelGap | FactorAcc | ExactAcc |
|---|----------|------|--------|-----------|----------|
| safe | factor_cajt | 47% | -0.063 | 0.378 | 0.014 |
| safe | cajt_v3 | 47% | 0.570 | — | — |
| safe | v1_1 | 47% | 0.067 | — | — |
| safe | oracle | 47% | 0.000 | — | — |
| shiny | factor_cajt | 83% | 0.520 | 0.392 | 0.014 |
| shiny | cajt_v3 | 83% | 0.653 | — | — |
| shiny | v1_1 | 83% | 0.180 | — | — |
| shiny | oracle | 83% | 0.000 | — | — |
