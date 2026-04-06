# PO Leakage Audit + Tutor Sensitivity Report

## Exp A: Leakage Audit

| Layer | ID Probe Acc | Early Gap | Full Gap | Ratio |
|-------|-------------|-----------|----------|-------|
| original | 100% | 0.0142 | 0.0180 | 1.3× |
| lane_neutral | 50% | 0.0001 | 0.0046 | 50.6× |
| semantic_subspace | 50% | 0.0001 | 0.0076 | 55.2× |
| full_fix | 50% | 0.0001 | 0.0076 | 55.2× |

## Exp B: Tutor Sensitivity (full_fix layer)

| Condition | SBCR |
|-----------|------|
| no_tutor+old | 100% |
| no_tutor+branch | 40% |
| rsa_warn+branch | 60% |
| oracle_warn+branch | 60% |

### Key Comparisons
- Oracle lift: +20%
- RSA lift: +20%

## Exp C: Training-Probe Loop

| Condition | k=0 | k=1 | k=3 | k=10 | k=30 |
|-----------|-----|-----|-----|------|------|
| no_tutor+old | 60% | 100% | 100% | 100% | 100% |
| no_tutor+branch | 60% | 40% | 40% | 40% | 40% |
| rsa_warn+branch | 60% | 60% | 60% | 40% | 60% |
| oracle_warn+branch | 60% | 60% | 60% | 60% | 60% |

### Learning Gain LG(k)

- **no_tutor+branch**: k=1:-20%, k=3:-20%, k=10:-20%, k=30:-20%
- **rsa_warn+branch**: k=1:+0%, k=3:+0%, k=10:-20%, k=30:+0%
- **oracle_warn+branch**: k=1:+0%, k=3:+0%, k=10:+0%, k=30:+0%
