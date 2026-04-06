# Enhancement Final Classification Report

> Phase 7 Track A: close B1/B2/B3 story

# B1 / B3 Redundancy Analysis

> Compare Δ^B1(x,ℓ) vs Δ^B3(x,ℓ) across mastery states and lessons

## Score Deltas by Mastery State

### Mastery = low

| Lesson | J_base | J_B1 | J_B3 | Δ^B1 | Δ^B3 | argmax_changed? |
|--------|:------:|:----:|:----:|:----:|:----:|:---:|
| ppmrb_standard | 2.9597 | 2.9597 | 2.9590 | -0.0000 | -0.0007 | ✓ |
| ppmrb_self_discovery | 2.9506 | 2.9506 | 2.9500 | -0.0000 | -0.0006 | ✓ |
| tic_rescue_heavy | 3.0212 | 3.0212 | 3.0208 | -0.0000 | -0.0004 | ✓ |
| tic_temptation | 2.9856 | 2.9856 | 2.9853 | -0.0000 | -0.0003 | ✓ |
| tic_self_discovery | 2.9714 | 2.9714 | 2.9706 | -0.0000 | -0.0008 | ✓ |
| sparse_valid_advice | 2.9597 | 2.9597 | 2.9597 | -0.0000 | -0.0000 | — |
| sparse_invalid_advice | 2.9714 | 2.9714 | 2.9713 | -0.0000 | -0.0001 | — |
| beneficial_novelty | 2.9597 | 2.9597 | 2.9593 | -0.0000 | -0.0004 | ✓ |
| verified_warn | 2.9714 | 2.9714 | 2.9713 | -0.0000 | -0.0001 | ✓ |
| false_suppression | 2.9714 | 2.9714 | 2.9709 | -0.0000 | -0.0005 | ✓ |
### Mastery = mid

| Lesson | J_base | J_B1 | J_B3 | Δ^B1 | Δ^B3 | argmax_changed? |
|--------|:------:|:----:|:----:|:----:|:----:|:---:|
| ppmrb_standard | 3.0918 | 3.0918 | 3.0848 | -0.0000 | -0.0070 | ✓ |
| ppmrb_self_discovery | 3.0832 | 3.0832 | 3.0765 | -0.0000 | -0.0068 | ✓ |
| tic_rescue_heavy | 3.1495 | 3.1495 | 3.1456 | -0.0000 | -0.0039 | ✓ |
| tic_temptation | 3.1160 | 3.1160 | 3.1148 | -0.0000 | -0.0012 | ✓ |
| tic_self_discovery | 3.1027 | 3.1027 | 3.0954 | -0.0000 | -0.0073 | ✓ |
| sparse_valid_advice | 3.0918 | 3.0918 | 3.0906 | -0.0000 | -0.0012 | ✓ |
| sparse_invalid_advice | 3.1027 | 3.1027 | 3.1007 | -0.0000 | -0.0020 | ✓ |
| beneficial_novelty | 3.0918 | 3.0918 | 3.0869 | -0.0000 | -0.0049 | ✓ |
| verified_warn | 3.1027 | 3.1027 | 3.1010 | -0.0000 | -0.0018 | ✓ |
| false_suppression | 3.1027 | 3.1027 | 3.0989 | -0.0000 | -0.0039 | ✓ |
### Mastery = high

| Lesson | J_base | J_B1 | J_B3 | Δ^B1 | Δ^B3 | argmax_changed? |
|--------|:------:|:----:|:----:|:----:|:----:|:---:|
| ppmrb_standard | 3.3176 | 3.3176 | 3.2978 | -0.0000 | -0.0197 | ✓ |
| ppmrb_self_discovery | 3.3098 | 3.3098 | 3.2902 | -0.0000 | -0.0196 | ✓ |
| tic_rescue_heavy | 3.3699 | 3.3699 | 3.3587 | -0.0000 | -0.0112 | ✓ |
| tic_temptation | 3.3395 | 3.3395 | 3.3350 | -0.0000 | -0.0045 | ✓ |
| tic_self_discovery | 3.3275 | 3.3275 | 3.3058 | -0.0000 | -0.0217 | ✓ |
| sparse_valid_advice | 3.3176 | 3.3176 | 3.3136 | -0.0000 | -0.0039 | ✓ |
| sparse_invalid_advice | 3.3275 | 3.3275 | 3.3209 | -0.0000 | -0.0065 | ✓ |
| beneficial_novelty | 3.3176 | 3.3176 | 3.3030 | -0.0000 | -0.0146 | ✓ |
| verified_warn | 3.3275 | 3.3275 | 3.3214 | -0.0000 | -0.0060 | ✓ |
| false_suppression | 3.3275 | 3.3275 | 3.3150 | -0.0000 | -0.0125 | ✓ |
### Mastery = mixed

| Lesson | J_base | J_B1 | J_B3 | Δ^B1 | Δ^B3 | argmax_changed? |
|--------|:------:|:----:|:----:|:----:|:----:|:---:|
| ppmrb_standard | 3.0894 | 3.0894 | 3.0813 | -0.0000 | -0.0081 | ✓ |
| ppmrb_self_discovery | 3.0808 | 3.0808 | 3.0758 | -0.0000 | -0.0050 | ✓ |
| tic_rescue_heavy | 3.1471 | 3.1471 | 3.1423 | -0.0000 | -0.0048 | ✓ |
| tic_temptation | 3.1136 | 3.1136 | 3.1126 | -0.0000 | -0.0010 | ✓ |
| tic_self_discovery | 3.1003 | 3.1003 | 3.0941 | -0.0000 | -0.0062 | ✓ |
| sparse_valid_advice | 3.0894 | 3.0894 | 3.0874 | -0.0000 | -0.0019 | ✓ |
| sparse_invalid_advice | 3.1003 | 3.1003 | 3.1002 | -0.0000 | -0.0001 | — |
| beneficial_novelty | 3.0894 | 3.0894 | 3.0871 | -0.0000 | -0.0022 | ✓ |
| verified_warn | 3.1003 | 3.1003 | 3.0969 | -0.0000 | -0.0034 | ✓ |
| false_suppression | 3.1003 | 3.1003 | 3.0991 | -0.0000 | -0.0012 | ✓ |

## Summary

- **Corr(Δ^B1, Δ^B3)** = -0.4987
- B1 only switches argmax: 0/8
- B3 only switches argmax: 0/8
- Both switch to same lesson: 0/8
- Both switch to different lessons: 0/8

> **Verdict**: B1 and B3 have low correlation → complementary → **keep both as distinct optional modules**


---

# B2 Micro-Family Validation

> U-Safe: unknown path actually safe | U-Danger: unknown path actually dangerous

| Scenario | Metric | Base | B2 | Δ |
|----------|--------|:----:|:--:|:-:|
| U-Safe | P(enter unknown) | 0.00 | 0.00 | +0.00 |
| U-Safe | mean ΔU_unknown | — | — | +0.0383 |
| U-Danger | P(enter unknown) | 0.00 | 0.00 | +0.00 |
| U-Danger | mean ΔU_unknown | — | — | +0.4424 |

- **SDR_micro** = 0.000 (0/100)

## Second-Visit Correction (U-Danger)

| Visit | P(avoid danger) Base | P(avoid danger) B2 |
|-------|:-:|:-:|
| 1st (unc=0.8) | 1.00 | 1.00 |
| 2nd (unc=0.15) | 1.00 | 1.00 |

## Verdict

