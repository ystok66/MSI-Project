# TPM Ablation Results (medium difficulty, 20 seeds)

## fork_trap

| Ablation | SR | DR | Steps | RepeatUL | LMR | FalseWarnOC |
|----------|----|----|-------|----------|-----|-------------|
| full_tpm | 70% | 30% | 8.9 | 0.0 | 0.71 | 0.0 |
| no_bottleneck_match | 75% | 25% | 9.1 | 0.0 | 0.50 | 0.0 |
| no_warn_damping | 65% | 35% | 8.7 | 0.0 | 1.00 | 0.0 |
| no_unlock_memory | 70% | 30% | 8.9 | 0.0 | 0.71 | 0.0 |
| no_perceptual_access | 65% | 35% | 8.7 | 0.0 | 1.00 | 0.0 |
| cf_only | 65% | 35% | 8.7 | 0.0 | 1.00 | 0.0 |

## hazard_belt

| Ablation | SR | DR | Steps | RepeatUL | LMR | FalseWarnOC |
|----------|----|----|-------|----------|-----|-------------|
| full_tpm | 65% | 35% | 19.4 | 0.0 | 0.71 | 0.0 |
| no_bottleneck_match | 65% | 35% | 19.4 | 0.0 | 0.30 | 0.0 |
| no_warn_damping | 40% | 60% | 15.6 | 0.0 | 1.00 | 0.0 |
| no_unlock_memory | 65% | 35% | 19.4 | 0.0 | 0.71 | 0.0 |
| no_perceptual_access | 40% | 60% | 15.6 | 0.0 | 1.00 | 0.0 |
| cf_only | 40% | 60% | 15.6 | 0.0 | 1.00 | 0.0 |

## deadline_gate

| Ablation | SR | DR | Steps | RepeatUL | LMR | FalseWarnOC |
|----------|----|----|-------|----------|-----|-------------|
| full_tpm | 100% | 0% | 26.0 | 0.0 | 0.84 | 0.0 |
| no_bottleneck_match | 100% | 0% | 26.0 | 0.0 | 0.24 | 0.0 |
| no_warn_damping | 100% | 0% | 26.0 | 0.0 | 1.00 | 0.0 |
| no_unlock_memory | 100% | 0% | 26.0 | 0.0 | 0.77 | 0.0 |
| no_perceptual_access | 100% | 0% | 26.0 | 0.0 | 1.00 | 0.0 |
| cf_only | 100% | 0% | 26.0 | 0.0 | 1.00 | 0.0 |

## Delta SR from full_tpm

| Ablation | fork_trap | hazard_belt | deadline_gate |
|----------|-----------|-------------|---------------|
| full_tpm | +0% | +0% | +0% |
| no_bottleneck_match | +5% | +0% | +0% |
| no_warn_damping | -5% | -25% | +0% |
| no_unlock_memory | +0% | +0% | +0% |
| no_perceptual_access | -5% | -25% | +0% |
| cf_only | -5% | -25% | +0% |
