# Priority 3: DCE + Strong Drift + Cross-Family Transfer

## A. Decision Calibration Error (DCE)

DCE = E[|C_t − 1[oracle-compatible]|]

| θ | Strategy | DCE | OAA |
|---|----------|-----|-----|
| safe | v1_1 | 0.500 | 0.667 |
| safe | cajt_v3 | 0.612 | 0.635 |
| shiny | v1_1 | 0.500 | 0.750 |
| shiny | cajt_v3 | 0.507 | 0.521 |

## B. Strong Drift Robustness

### Abrupt Drift

| Strategy | SG(pre) | SG(post) | Δ(recovery) | SBCR(pre) | SBCR(post) |
|----------|---------|----------|-------------|-----------|------------|
| v1_1 | 0.733 | 0.786 | -0.042 | 89% | 9% |
| cajt_v3 | 0.733 | 0.429 | -0.292 | 89% | 9% |
### Gradual Drift

| Strategy | SG(pre) | SG(post) | Δ(recovery) | SBCR(pre) | SBCR(post) |
|----------|---------|----------|-------------|-----------|------------|
| v1_1 | 0.733 | 0.714 | -0.042 | 70% | 27% |
| cajt_v3 | 0.633 | 0.357 | -0.417 | 70% | 27% |
### Intra Drift

| Strategy | SG(pre) | SG(post) | Δ(recovery) | SBCR(pre) | SBCR(post) |
|----------|---------|----------|-------------|-----------|------------|
| v1_1 | 0.733 | 0.714 | -0.167 | 75% | 62% |
| cajt_v3 | 0.733 | 0.500 | -0.417 | 75% | 62% |

## C. Cross-Family Teaching Transfer

Train on source family (8 ep) → test on TIC (8 ep, no tutor)

| θ | Train Family | Strategy | Test SBCR | LG_total | κ_f | η_f | γ_f |
|---|-------------|----------|-----------|----------|-----|-----|-----|
| safe | ppmrb | no_tutor | 50% | 1.7930 | 1.656 | 0.001 | 0.625 |
| safe | ppmrb | v1_1 | 50% | 1.6700 | 1.581 | 0.024 | 0.600 |
| safe | ppmrb | cajt_v3 | 50% | 1.6890 | 1.600 | 0.023 | 0.600 |
| safe | tic | no_tutor | 48% | 1.1930 | 1.547 | 0.001 | 0.075 |
| safe | tic | v1_1 | 48% | 1.1710 | 1.547 | 0.022 | 0.075 |
| safe | tic | cajt_v3 | 48% | 1.1710 | 1.547 | 0.022 | 0.075 |
| shiny | ppmrb | no_tutor | 38% | 1.8990 | 1.649 | 0.001 | 0.750 |
| shiny | ppmrb | v1_1 | 38% | 1.9280 | 1.703 | 0.025 | 0.750 |
| shiny | ppmrb | cajt_v3 | 38% | 1.9220 | 1.703 | 0.031 | 0.750 |
| shiny | tic | no_tutor | 36% | 2.0710 | 1.771 | 0.001 | 0.800 |
| shiny | tic | v1_1 | 34% | 2.1550 | 1.877 | 0.022 | 0.800 |
| shiny | tic | cajt_v3 | 36% | 2.1490 | 1.876 | 0.027 | 0.800 |

### Transfer Improvement: PP-MRB→TIC vs TIC→TIC

| θ | Strategy | SBCR(ppmrb→tic) | SBCR(tic→tic) | Δ |
|---|----------|-----------------|---------------|---|
| safe | v1_1 | 34% | 34% | +0.000 |
| safe | cajt_v3 | 34% | 34% | +0.000 |
| shiny | v1_1 | 38% | 31% | +0.063 |
| shiny | cajt_v3 | 38% | 34% | +0.031 |
