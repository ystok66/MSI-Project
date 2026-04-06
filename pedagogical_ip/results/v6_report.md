# V6 Branch-Aware Planner Results

## Exp 1: Planner Interface Ablation

| Condition | SBCR | n |
|-----------|------|---|
| old_planner | 67% | 30 |
| pointwise_only | 67% | 30 |
| concept_only | 100% | 30 |
| scorer_only | 100% | 30 |
| hybrid | 100% | 30 |
| hybrid_strong | 100% | 30 |

## Exp 2: ELCB Transfer

| Condition | k=0 | k=1 | k=3 | k=10 | k=30 |
|-----------|-----|-----|-----|------|------|
| old_planner | 60% | 70% | 30% | 30% | 70% |
| branch_aware | 100% | 100% | 100% | 100% | 100% |

## Exp 3: Side-Swap Robustness

- Overall SBCR: 100%
- SBCR when safe=A: 100% (n=30)
- SBCR when safe=B: 100% (n=20)
- SideBias: 0.000
- SemanticConsistency: 100%
