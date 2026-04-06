# B1: Learning-Aware WAIT vs WARN Report

## Intervention Statistics

| Strategy | Warn Count | Wait Count | Warn Rate |
|----------|-----------|------------|----------|
| always_wait | 0 | 30 | 0% |
| always_warn | 30 | 0 | 100% |
| learning_aware | 1 | 29 | 3% |
| oracle | 30 | 0 | 100% |

## Training-Probe Results (50 seeds, 95% CI)

| Strategy | k=0 | k=1 | k=3 | k=10 | k=30 |
|----------|-----|-----|-----|------|------|
| always_wait | 52% [40%,66%] | 40% [26%,54%] | 40% [28%,52%] | 40% [26%,52%] | 40% [28%,52%] |
| always_warn | 52% [40%,66%] | 60% [46%,74%] | 60% [48%,72%] | 60% [48%,74%] | 60% [48%,72%] |
| learning_aware | 52% [40%,66%] | 40% [26%,54%] | 40% [28%,52%] | 40% [26%,52%] | 60% [48%,72%] |
| oracle | 52% [40%,66%] | 60% [46%,74%] | 60% [48%,72%] | 60% [48%,74%] | 60% [48%,72%] |

### Learning Gain LG(k) = SBCR(k) - SBCR(0)

- **always_wait**: k=1:-12%, k=3:-12%, k=10:-12%, k=30:-12%
- **always_warn**: k=1:+8%, k=3:+8%, k=10:+8%, k=30:+8%
- **learning_aware**: k=1:-12%, k=3:-12%, k=10:-12%, k=30:+8%
- **oracle**: k=1:+8%, k=3:+8%, k=10:+8%, k=30:+8%

### Pedagogical Efficiency PE(k) = LG(k) / warn_rate

- **always_warn** (WR=100%): k=1:0.08, k=3:0.08, k=10:0.08, k=30:0.08
- **learning_aware** (WR=3%): k=1:-3.64, k=3:-3.64, k=10:-3.64, k=30:2.42
- **oracle** (WR=100%): k=1:0.08, k=3:0.08, k=10:0.08, k=30:0.08
