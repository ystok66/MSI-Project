# PP-MRB v2.1 Robustness Suite

## Exp A: Mirror Swap

| θ | Mirror | WR(wait_fav) | WR(warn_nec) | SelGap |
|---|--------|:------------:|:------------:|:------:|
| shiny | left=safe | 41% | 100% | 0.588 |
| shiny | right=safe | 78% | 100% | 0.219 |
| shiny | mixed | 62% | 100% | 0.376 |
| safe | left=safe | 85% | 100% | 0.147 |
| safe | right=safe | 97% | 100% | 0.028 |
| safe | mixed | 91% | 100% | 0.089 |

## Exp B: Cue-Noise Sweep

| θ | Noise | WR(wait_fav) | WR(warn_nec) | SelGap |
|---|-------|:------------:|:------------:|:------:|
| shiny | σ=0.00 | 62% | 100% | 0.376 |
| shiny | σ=0.05 | 64% | 100% | 0.361 |
| shiny | σ=0.10 | 67% | 100% | 0.330 |
| shiny | σ=0.20 | 62% | 100% | 0.381 |
| safe | σ=0.00 | 91% | 100% | 0.089 |
| safe | σ=0.05 | 92% | 100% | 0.085 |
| safe | σ=0.10 | 92% | 100% | 0.080 |
| safe | σ=0.20 | 92% | 100% | 0.081 |

## Exp C: Δ = d_commit − d_reveal Sweep

| θ | Δ | d_commit | d_reveal | WarnRate | c_t(final) |
|---|---|----------|----------|---------|:-----------:|
| shiny | -4 | 1 | 5 | 100% | 0.831 |
| shiny | -2 | 2 | 4 | 100% | 0.830 |
| shiny | +0 | 3 | 3 | 100% | 0.830 |
| shiny | +2 | 4 | 2 | 100% | 0.829 |
| shiny | +4 | 5 | 1 | 42% | 0.828 |
| shiny | +5 | 6 | 1 | 26% | 0.828 |
| safe | -4 | 1 | 5 | 100% | 0.229 |
| safe | -2 | 2 | 4 | 100% | 0.230 |
| safe | +0 | 3 | 3 | 100% | 0.231 |
| safe | +2 | 4 | 2 | 100% | 0.232 |
| safe | +4 | 5 | 1 | 98% | 0.232 |
| safe | +5 | 6 | 1 | 94% | 0.232 |

## Exp D: Learner-Type Sweep

| θ | Cond | WR(wait_fav) | WR(warn_nec) | SelGap | SBCR |
|---|------|:------------:|:------------:|:------:|:----:|
| safe | persistent | 91% | 100% | 0.089 | 85% |
| safe | reset | 100% | 100% | 0.004 | 85% |
| shiny | persistent | 62% | 100% | 0.376 | 18% |
| shiny | reset | 98% | 100% | 0.015 | 18% |
| shortcut | persistent | 97% | 100% | 0.028 | 74% |
| shortcut | reset | 100% | 100% | 0.000 | 74% |
| risky | persistent | 90% | 100% | 0.098 | 56% |
| risky | reset | 99% | 100% | 0.013 | 56% |
| neutral | persistent | 93% | 100% | 0.072 | 76% |
| neutral | reset | 98% | 100% | 0.020 | 76% |

## Go/No-Go Assessment

| Gate | Criterion | Result | ✓/✗ |
|------|-----------|--------|-----|
| Mirror | SelGap > 0 on both sides | left=0.588, right=0.219 (shiny) | ✅ |
| Noise | Direction preserved at σ=0.20 | SG=0.381 at σ=0.20 vs 0.376 baseline | ✅ |
| Δ law | WR monotone w.r.t. Δ | 100%→100%→42%→26% (shiny) | ✅ |
| Multi-θ | ≥2 learner types with SG>0 | all 5 positive (0.028–0.376) | ✅ |
| SBCR | No regression vs reset | identical for all θ | ✅ |
| WR(warn_nec) | ≥80% everywhere | **100%** in all conditions | ✅ |

**Verdict: ALL GATES PASS → Stage 2 READY.**
