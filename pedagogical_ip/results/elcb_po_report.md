# ELCB-PO Results (v2 — proper partial observability)

## Step 2: Sanity

- **Visible-only mean risk gap**: 0.0156
- **Full-branch mean risk gap**: 0.0161
- **Full/Visible ratio**: 1.0×

## Step 3: Tutor Value Audit (40 train, 30 probe)

| Condition | SBCR |
|-----------|------|
| no_tutor+old | 43% |
| rsa_warn+old | 43% |
| no_tutor+branch | 57% |
| rsa_warn+branch | 57% |
| oracle_warn+branch | 57% |

### Decomposition
- OHG (old planner): +0%
- OHG (branch planner): +0%
- Oracle lift: +0%

## Step 4: Training-Probe Loop

| Condition | k=0 | k=1 | k=3 | k=10 | k=30 |
|-----------|-----|-----|-----|------|------|
| no_tutor+old | 55% | 45% | 45% | 55% | 45% |
| rsa_warn+old | 55% | 45% | 45% | 55% | 45% |
| no_tutor+branch | 60% | 55% | 55% | 45% | 45% |
| rsa_warn+branch | 60% | 45% | 55% | 45% | 45% |
| oracle_warn+branch | 60% | 45% | 45% | 45% | 45% |

### Learning Gain LG(k)

- **no_tutor+branch**: k=1:-5%, k=3:-5%, k=10:-15%, k=30:-15%
- **rsa_warn+branch**: k=1:-15%, k=3:-5%, k=10:-15%, k=30:-15%
- **oracle_warn+branch**: k=1:-15%, k=3:-15%, k=10:-15%, k=30:-15%
