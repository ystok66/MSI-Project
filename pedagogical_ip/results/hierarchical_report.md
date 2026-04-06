# Hierarchical vs Exact Goal-Factor Posterior

## Train Compositions

| θ | Strategy | SBCR | WR | **SelGap** | FactorAcc | ExactAcc | AvgFC |
|---|----------|------|----|-----------|-----------|----------|-------|
| safe | v1_1 | 53% | 94% | **0.327** | — | — | — |
| safe | cajt_v3 | 53% | 80% | **0.521** | — | — | — |
| safe | factor_exact | 53% | 68% | **0.348** | 0.273 | 0.000 | — |
| safe | factor_hier | 53% | 67% | **0.373** | 0.305 | 0.000 | 0.456 |
| safe | hier_cajt | 53% | 66% | **0.348** | 0.305 | 0.000 | 0.456 |
| safe | oracle | 53% | 8% | **0.000** | — | — | — |
| shiny | v1_1 | 79% | 100% | **0.000** | — | — | — |
| shiny | cajt_v3 | 79% | 77% | **0.415** | — | — | — |
| shiny | factor_exact | 79% | 65% | **-0.325** | 0.276 | 0.000 | — |
| shiny | factor_hier | 79% | 59% | **-0.221** | 0.292 | 0.000 | 0.444 |
| shiny | hier_cajt | 79% | 57% | **-0.200** | 0.292 | 0.000 | 0.444 |
| shiny | oracle | 79% | 14% | **0.000** | — | — | — |

### SelGap + FactorAcc Comparison

| θ | Metric | exact | **hier** | hier_cajt |
|---|--------|-------|---------|--------|
| safe | SelGap | 0.348 | **0.373** | 0.348 |
| safe | FactorAcc | 0.273 | **0.305** | 0.305 |
| safe | AvgFC | — | **0.456** | 0.456 |
| shiny | SelGap | -0.325 | **-0.221** | -0.200 |
| shiny | FactorAcc | 0.276 | **0.292** | 0.292 |
| shiny | AvgFC | — | **0.444** | 0.444 |

## Held-Out Compositions

| θ | Strategy | SelGap | FactorAcc | ExactAcc | AvgFC |
|---|----------|--------|-----------|----------|-------|
| safe | factor_exact | 0.071 | 0.383 | 0.021 | — |
| safe | factor_hier | 0.119 | 0.390 | 0.031 | 0.443 |
| safe | hier_cajt | 0.155 | 0.390 | 0.031 | 0.443 |
| safe | cajt_v3 | 0.286 | — | — | — |
| safe | oracle | 0.000 | — | — | — |
| shiny | factor_exact | 0.287 | 0.360 | 0.000 | — |
| shiny | factor_hier | 0.290 | 0.367 | 0.021 | 0.442 |
| shiny | hier_cajt | 0.332 | 0.367 | 0.021 | 0.442 |
| shiny | cajt_v3 | 0.763 | — | — | — |
| shiny | oracle | 0.000 | — | — | — |
