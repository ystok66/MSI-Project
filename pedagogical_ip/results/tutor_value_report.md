# Tutor Value Audit Report

## A1: Timing Curve

| Timing | Depth | SBCR | Entropy |
|--------|-------|------|---------|
| no_warn | 0 | 40% | 0.5265 |
| 1_cell_early | 1 | 40% | 0.6178 |
| weak_only | 3 | 40% | 0.6557 |
| partial_strong | 5 | 60% | 0.6007 |
| full_oracle | 10 | 60% | 0.4725 |

## A2: Tutor Value Decomposition

| Condition | SBCR | Entropy |
|-----------|------|---------|
| no_tutor+old | 100% | 0.0168 |
| no_tutor+branch | 40% | 0.5265 |
| rsa_warn+branch | 60% | 0.7136 |
| oracle_warn+branch | 60% | 0.4725 |

### Decomposition
- **OHG (RSA)**: +20% (SBCR lift from RSA warning)
- **OHG (Oracle)**: +20% (SBCR lift from oracle)
- **IG (RSA)**: -0.1871 (entropy reduction)
- **IG (Oracle)**: 0.0540 (entropy reduction)
- **Branch vs Old**: -60%

## A3: Training-Probe Loop (50 seeds, bootstrap 95% CI)

| Condition | k=0 | k=1 | k=3 | k=10 | k=30 |
|-----------|-----|-----|-----|------|------|
| no_tutor+branch | 52% [40%,64%] | 40% [28%,52%] | 40% [28%,54%] | 40% [28%,54%] | 40% [26%,52%] |
| rsa_warn+branch | 52% [40%,64%] | 60% [48%,72%] | 60% [46%,72%] | 40% [28%,54%] | 60% [48%,74%] |
| oracle_warn+branch | 52% [40%,64%] | 60% [48%,72%] | 60% [46%,72%] | 60% [46%,72%] | 60% [48%,74%] |

### Learning Gain LG(k) = SBCR(k) - SBCR(0)

- **no_tutor+branch**: k=1:-12%, k=3:-12%, k=10:-12%, k=30:-12%
- **rsa_warn+branch**: k=1:+8%, k=3:+8%, k=10:-12%, k=30:+8%
- **oracle_warn+branch**: k=1:+8%, k=3:+8%, k=10:+8%, k=30:+8%
