# Phase 2A — Experiment 1 Results: Basis-Only Shadow

**Date**: 2026-04-06  
**Experiment**: 15 episodes × 5 seeds × 4 conditions on baseline_v2

---

## Summary

| Condition | Surv | Goal | Steps | N |
|-----------|------|------|-------|---|
| linear_fresh | 0.347 | 0.347 | 13.9 | 75 |
| linear_persist | 0.573 | 0.573 | 21.3 | 75 |
| **basis_fresh** | **1.000** | **1.000** | 33.1 | 75 |
| **basis_persist** | **1.000** | **1.000** | 33.1 | 75 |

---

## Key Findings

### Q1: basis_fresh ≥ linear_fresh? → **OVERWHELMING YES**

basis_fresh achieves 100% survival/goal vs 34.7% for linear_fresh. The structured basis completely solves baseline_v2 within a single episode.

The basis captures what linear cannot: the cross-term `z₂z₃` (texture interaction) and `|z₂-z₃|` (texture dissimilarity) directly model the WorldWeights risk generation, which was designed with texture-dominant risk drivers.

### Q2: Does basis make transfer MORE valuable? → **CEILING EFFECT**

Transfer bonus = 0.000 for basis because basis_fresh is already 100%. This is NOT "basis reduces transfer value" — it's "basis solves the task so completely that transfer has nothing to add on this difficulty."

This means **baseline_v2 medium is too easy for the basis head**. The task difficulty needs to increase to see transfer effects.

### risk_w_norm behavior

- linear_fresh: risk_w ≈ 0.30-0.50 (unstable, high variance)
- linear_persist: risk_w converges: 0.31 → 0.015 (shrinking as model learns)
- basis_fresh: risk_w ≈ 0.041 (immediately stable!)
- basis_persist: risk_w: 0.041 → 0.007 (slow decay, already near-optimal)

The basis head's near-zero risk weights from episode 1 suggest the intercept term and raw texture features already provide most of the needed structure.

---

## Implications for Phase 2B

1. **baseline_v2 is dominated by basis** — need harder scenarios for transfer eval
2. **Next experiment should use GTET or DTMB** where agent faces structured temptation/decision trees
3. **The α sweep should wait** until we find a scenario where basis_fresh < 1.0
4. **The basis design is validated** — the semantic alignment (texture cross-terms) works

---

## Next Steps

1. Run Phase 2A on GTET and DTMB families (harder scenarios)
2. If basis_fresh < 1.0 on those, then α sweep becomes meaningful
3. Consider "hard" difficulty instead of "medium"
