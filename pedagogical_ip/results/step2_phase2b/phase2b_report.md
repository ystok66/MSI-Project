# Step 2 Phase 2B: Closed-Loop Warning Experiment

**Seeds**: 50 | **Elapsed**: 123.7s

## Headline Metrics

| Family | Variant | SBCR | TBSR | SelGap | WR_nec | WR_unnec | Warns | dM | dNLL |
|--------|---------|------|------|--------|--------|----------|-------|----|------|
| fork_trap | legacy_bias | 0.380 | 0.620 | 1.000 | 1.000 | 0.000 | 1.0 | 0.0000 | -0.6380 |
| fork_trap | rsa_obs_s1 | 0.380 | 0.500 | 1.000 | 1.000 | 0.000 | 1.0 | 0.5195 | -0.5547 |
| elcb_po | legacy_bias | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.0 | 0.0000 | -0.6419 |
| elcb_po | rsa_obs_s1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.0 | 0.5195 | -0.5547 |
| baseline_v2 | legacy_bias | 0.500 | 0.760 | -0.031 | 0.969 | 1.000 | 2.9 | 0.0000 | -0.6361 |
| baseline_v2 | rsa_obs_s1 | 0.520 | 0.740 | 0.000 | 1.000 | 1.000 | 3.0 | 0.5195 | -0.5547 |

## Promotion Decision

### fork_trap

- SBCR: 0.380 vs 0.380 PASS
- TBSR: 0.500 vs 0.620 FAIL
- SelGap: 1.000 vs 1.000 PASS
- dM_true: 0.5195 vs 0.0000 PASS

### elcb_po

- SBCR: 1.000 vs 1.000 PASS
- TBSR: 1.000 vs 1.000 PASS
- SelGap: 1.000 vs 1.000 PASS
- dM_true: 0.5195 vs 0.0000 PASS

### baseline_v2

- SBCR: 0.520 vs 0.500 PASS
- TBSR: 0.740 vs 0.760 PASS
- SelGap: 0.000 vs -0.031 PASS
- dM_true: 0.5195 vs 0.0000 PASS

