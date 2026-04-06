# T1 Exp-2: Q-Margin Audit

> Total steps: 600 | Active: 26 | Diverge: 3 | Over-warn: 3

## Table 1: |ΔQ| Distribution — All Steps

| |ΔQ| Range | Steps | Over-Warn | OWR% | Oracle=WARN | Disagree% |
|:----------:|:-----:|:---------:|:----:|:-----------:|:---------:|
| [0.00, 0.02) | 1 | 0 | 0.0% | 0 | 0.0% |
| [0.02, 0.05) | 2 | 0 | 0.0% | 1 | 0.0% |
| [0.05, 0.10) | 8 | 3 | 37.5% | 2 | 37.5% |
| [0.10, 0.15) | 11 | 0 | 0.0% | 6 | 0.0% |
| [0.15, ∞) | 578 | 0 | 0.0% | 14 | 0.0% |

## Table 2: NearTieCoverage(ε_Q)

| ε_Q | OWR in near-tie | Total OWR | Coverage |
|:---:|:---------------:|:---------:|:--------:|
| 0.02 | 0 | 3 | 0.0% |
| 0.05 | 0 | 3 | 0.0% |
| 0.10 | 3 | 3 | 100.0% |
| 0.15 | 3 | 3 | 100.0% |
| 0.20 | 3 | 3 | 100.0% |
| 0.50 | 3 | 3 | 100.0% |

## Table 3: Per-Family Over-Warn Distribution

| Family | Steps | OWR Count | OWR% | Mean |ΔQ| | Median |ΔQ| |
|--------|:-----:|:---------:|:----:|:----------:|:-----------:|
| beneficial_novelty | 30 | 0 | 0.0% | 4.4603 | 4.3458 |
| blind_activation_corridor **←** | 30 | 2 | 6.7% | 0.3689 | 0.2201 |
| false_suppression | 30 | 0 | 0.0% | 3.9074 | 3.5113 |
| ppmrb_self_discovery | 60 | 0 | 0.0% | 6.8918 | 6.9331 |
| ppmrb_standard | 60 | 0 | 0.0% | 3.7692 | 3.8214 |
| soft_boundary_tradeoff | 30 | 0 | 0.0% | 3.0651 | 3.3035 |
| sparse_invalid_advice | 60 | 0 | 0.0% | 3.8836 | 4.1081 |
| sparse_valid_advice | 60 | 0 | 0.0% | 4.6560 | 4.7952 |
| tic_rescue_heavy **←** | 60 | 0 | 0.0% | 0.8068 | 0.6245 |
| tic_self_discovery | 60 | 0 | 0.0% | 6.8542 | 6.9020 |
| tic_temptation | 60 | 0 | 0.0% | 3.7401 | 3.8657 |
| verified_warn | 30 | 0 | 0.0% | 4.4072 | 4.4937 |
| warn_symmetric_rescue **←** | 30 | 1 | 3.3% | 0.7674 | 0.3437 |

## Table 4: Q Decomposition at Over-Warn Points

| Metric | Mean | Median | Min | Max |
|--------|:----:|:------:|:---:|:---:|
| delta_Q | 0.0694 | 0.0606 | 0.0535 | 0.0941 |
| delta_Q_online | 1.4443 | 1.4451 | 1.4017 | 1.4861 |
| delta_V_full_raw | -0.4092 | -0.4145 | -0.4148 | -0.3983 |
| delta_V_full_weighted | -1.4322 | -1.4508 | -1.4517 | -1.3940 |
| delta_R_over_raw | -0.0143 | -0.0148 | -0.0149 | -0.0132 |
| delta_R_over_weighted | -0.0573 | -0.0592 | -0.0597 | -0.0529 |
| p_self | 0.0376 | 0.0474 | 0.0180 | 0.0474 |
| p_fail | 0.8989 | 0.8808 | 0.8808 | 0.9350 |
| delta_s | 0.0000 | 0.0000 | 0.0000 | 0.0001 |
| dvoi | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tempt | 0.7023 | 0.7450 | 0.5760 | 0.7860 |
| risk | 0.5103 | 0.5250 | 0.4690 | 0.5370 |
| dc_minus_dr | -3.3333 | -3.0000 | -4.0000 | -3.0000 |

### Dominant Flip Component at Over-Warn Points

For each over-warn step, which component makes ΔQ positive?

| Component | Count | Fraction |
|-----------|:-----:|:--------:|
| online | 3 | 100.0% |
| V_full | 0 | 0.0% |
| R_over | 0 | 0.0% |

## Table 5: Per-Step Over-Warn Detail

| Family | θ | ΔQ | ΔQ_online | ΔV_full_w | ΔR_over_w | p_self | tempt | risk | dc-dr |
|--------|:-:|:--:|:---------:|:---------:|:----------:|:-----:|:-----:|:----:|:----:|
| warn_symmetric_rescue | shiny | 0.0535 | 1.4451 | -1.4508 | -0.0592 | 0.05 | 0.74 | 0.54 | -3 |
| blind_activation_corridor | shiny | 0.0941 | 1.4861 | -1.4517 | -0.0597 | 0.05 | 0.79 | 0.47 | -3 |
| blind_activation_corridor | shiny | 0.0606 | 1.4017 | -1.3940 | -0.0529 | 0.02 | 0.58 | 0.53 | -4 |

## Summary

> Over-warn count: 3/600 (0.50%)
> NearTieCoverage(ε=0.10): 100.0%
> **Conclusion: Over-warn is predominantly a near-tie Q-margin phenomenon. Dead-zone fix is justified.**
