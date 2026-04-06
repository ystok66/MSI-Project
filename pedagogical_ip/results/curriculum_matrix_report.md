# Curriculum-to-Transfer Matrix (CTM)

Train on source (8 ep, ICT-v1) → test on TIC (8 ep, no tutor)

| θ | Source | SBCR(same) | SBCR(shift) | κ_f | γ_f | ZHR | OTR |
|---|--------|-----------|------------|-----|-----|-----|-----|
| safe | none | 52% | 47% | 1.166 | 0.453 | 0.695 | 0.069 |
| safe | ppmrb | 50% | 48% | 1.256 | 0.663 | 0.301 | 0.272 |
| safe | tic_rescue | 48% | 48% | 1.278 | 0.734 | 0.279 | 0.448 |
| safe | tic_mixed | 48% | 48% | 1.278 | 0.734 | 0.279 | 0.448 |
| shiny | none | 45% | 47% | 1.179 | 0.511 | 0.444 | 0.208 |
| shiny | ppmrb | 47% | 38% | 1.229 | 0.685 | 0.176 | 0.382 |
| shiny | tic_rescue | 41% | 42% | 1.310 | 0.786 | 0.103 | 0.596 |
| shiny | tic_mixed | 41% | 42% | 1.310 | 0.786 | 0.103 | 0.596 |

## Transfer Improvement vs No-Source Baseline

| θ | Source | Δ(same) | Δ(shift) |
|---|--------|---------|----------|
| safe | ppmrb | -0.031 | +0.000 |
| safe | tic_rescue | -0.063 | +0.000 |
| safe | tic_mixed | -0.063 | +0.000 |
| shiny | ppmrb | +0.031 | +0.000 |
| shiny | tic_rescue | -0.063 | +0.094 |
| shiny | tic_mixed | -0.063 | +0.094 |
