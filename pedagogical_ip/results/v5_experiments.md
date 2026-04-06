# V5 Integration Experiments

## Exp 1: Semantic vs Position Bias Audit

### standard
- SBCR: 100%
- Side A rate: 50%
- SideBias = |rate - 0.5| = 0.000
- SemanticConsistency (SBCR): 100%

### mixed_effects
- SBCR: 100%
- Side A rate: 50%
- SideBias = |rate - 0.5| = 0.000
- SemanticConsistency (SBCR): 100%

## Exp 2: Warning Semantics Audit

| Seed | risk_A | risk_B | State | Utterance | P(risky) | pre_safe | teach_A | teach_B |
|------|--------|--------|-------|-----------|----------|----------|---------|----------|
| 0 | 0.421 | 0.446 | ambiguous | silence | 0.076 | 1 | rescue | rescue |
| 1 | 0.367 | 0.342 | ambiguous | silence | 0.076 | 1 | teach | teach |
| 2 | 0.246 | 0.231 | ambiguous | silence | 0.076 | 0 | teach | teach |
| 3 | 0.354 | 0.363 | ambiguous | silence | 0.076 | 1 | teach | teach |
| 4 | 0.361 | 0.372 | ambiguous | silence | 0.076 | 1 | teach | teach |
| 5 | 0.379 | 0.383 | ambiguous | silence | 0.076 | 1 | teach | teach |
| 6 | 0.324 | 0.299 | ambiguous | silence | 0.076 | 1 | teach | teach |
| 7 | 0.382 | 0.359 | ambiguous | silence | 0.076 | 1 | teach | teach |
| 8 | 0.275 | 0.248 | ambiguous | silence | 0.076 | 1 | teach | teach |
| 9 | 0.326 | 0.304 | ambiguous | silence | 0.076 | 1 | teach | teach |
| 10 | 0.286 | 0.280 | ambiguous | silence | 0.076 | 0 | teach | teach |
| 11 | 0.371 | 0.352 | ambiguous | silence | 0.076 | 1 | teach | teach |
| 12 | 0.351 | 0.326 | ambiguous | silence | 0.076 | 1 | teach | teach |
| 13 | 0.411 | 0.389 | ambiguous | silence | 0.076 | 1 | rescue | teach |
| 14 | 0.403 | 0.373 | ambiguous | silence | 0.076 | 1 | rescue | teach |
| 15 | 0.414 | 0.423 | ambiguous | silence | 0.076 | 1 | rescue | rescue |
| 16 | 0.388 | 0.392 | ambiguous | silence | 0.076 | 0 | teach | teach |
| 17 | 0.303 | 0.273 | ambiguous | silence | 0.076 | 1 | teach | teach |
| 18 | 0.240 | 0.216 | ambiguous | silence | 0.076 | 1 | teach | teach |
| 19 | 0.322 | 0.313 | ambiguous | silence | 0.076 | 0 | teach | teach |

- Pre-warning safe choice rate: 80%
- Utterance distribution: {'silence': 20}

## Exp 3: ELCB Transfer Re-run

| Condition | k=0 | k=1 | k=3 | k=10 | k=30 |
|-----------|-----|-----|-----|------|------|
| baseline | 60% | 70% | 30% | 30% | 70% |
| v5_concepts | 60% | 70% | 30% | 30% | 70% |

### Concept Accuracy (v5_concepts only)

- k=0: concept_acc=1.0
- k=1: concept_acc=1.0
- k=3: concept_acc=1.0
- k=10: concept_acc=1.0
- k=30: concept_acc=1.0

## Exp 4: Shared vs Residual Attribution

| λ_δ | |w_shared| | max |δ| | mean |δ| | discrimination |
|-----|---------|---------|---------|----------------|
| 0.5 | 0.0780 | 0.1608 | 0.1452 | 0.0059 |
| 1.0 | 0.0823 | 0.0873 | 0.0815 | 0.0063 |
| 2.0 | 0.0849 | 0.0484 | 0.0455 | 0.0065 |
| 5.0 | 0.0888 | 0.0278 | 0.0277 | 0.0069 |

## Exp 5: Branch Scorer Feasibility

- **Branch Scorer accuracy**: 100%
- **Pointwise baseline accuracy**: 67%
- Scorer updates: 80
- Concept κ safe: 41.0, risky: 41.0
- **Winner: Branch Scorer**

## Summary & Interpretation

### Key Questions Answered

1. **Does mixed-effects reduce side bias?** → Check Exp 1 SideBias values
2. **Does RSA warning identify correct risk state?** → Check Exp 2 utterance distribution
3. **Does V5 concept library improve transfer SBCR?** → Check Exp 3 v5_concepts vs baseline
4. **Does shrinkage separate shared from context?** → Check Exp 4 |w_shared| vs |δ|
5. **Is branch scorer better than pointwise?** → Check Exp 5 accuracy comparison
