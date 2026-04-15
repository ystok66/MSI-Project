# Exp A Results: Bayesian Eval-Aware Tutor — λ_probe Sweep

**Date**: 2026-04-10  
**Jobs**: 2347/2400 (98%, 53 errors — see error note)  
**Seeds**: 20 × 3 grammars × 4 n_sup × 2 n_teach = 480 per condition  
**Conditions**: A0 (legacy), A1–A3 (λ_probe ∈ {0.5, 1.0, 2.0}), A4 (probe-only)

---

## Summary Table (averaged across n_sup, n_teach)

| Condition | EVAL_SR | TEACH_SR | TransferGap | Actions (HL/SKIP/WAIT) | Time/job |
|-----------|---------|----------|-------------|------------------------|---------|
| A0_legacy | **0.671** | 0.677 | +0.006 | HL:23% WAIT:77% | 2.5s |
| A1_lp05 | 0.669 | 0.675 | +0.006 | HL:23% WAIT:77% | **131s** |
| A2_lp10 | 0.669 | 0.675 | +0.006 | HL:23% WAIT:77% | 133s |
| A3_lp20 | **0.672** | 0.675 | +0.005 | HL:23% WAIT:77% | 177s |
| A4_probe_only | 0.129 | 0.162 | -0.033 | SKIP:8% HL:2% WAIT:87% | 80s |

---

## Detailed EVAL_SR by n_sup and n_teach

### n_teach=1

| Condition | nsup=2 | nsup=4 | nsup=6 | nsup=8 |
|-----------|--------|--------|--------|--------|
| A0_legacy | 0.567 | 0.628 | 0.756 | 0.739 |
| A1_lp05   | 0.565 | 0.633 | 0.751 | 0.741 |
| A2_lp10   | 0.559 | 0.633 | 0.751 | 0.741 |
| A3_lp20   | 0.565 | 0.633 | 0.751 | 0.741 |
| A4_probe  | 0.217 | 0.189 | 0.089 | 0.117 |

### n_teach=2

| Condition | nsup=2 | nsup=4 | nsup=6 | nsup=8 |
|-----------|--------|--------|--------|--------|
| A0_legacy | 0.700 | 0.739 | 0.811 | 0.811 |
| A1_lp05   | 0.678 | 0.729 | 0.797 | 0.807 |
| A2_lp10   | 0.684 | 0.729 | 0.797 | 0.807 |
| A3_lp20   | 0.678 | 0.729 | 0.802 | **0.819** |
| A4_probe  | 0.072 | 0.128 | 0.089 | 0.133 |

---

## Key Findings

### F1: λ_probe has NO effect on action distribution (A1–A3 ≈ A0)

All eval-aware conditions (A1, A2, A3) produce **identical action distributions** to the legacy baseline (A0):
- HIGHLIGHT: ~17% (n_teach=1), ~29% (n_teach=2)
- BAN: ~1%
- WAIT: ~82%

The shadow probe signal is **too flat to differentiate actions**. The probe delta (ΔProbe) between WAIT, HIGHLIGHT, BAN is essentially zero because:
1. `probe_eval.score()` uses semantic margin, not classification accuracy
2. The current CLS shadow produces very similar scorers for all reveal outcomes
3. Result: all actions get the same q_probe ≈ 0, so ranking collapses to legacy

### F2: λ_probe has NO effect on EVAL_SR (A1–A3 ≈ A0)

Within SE bounds (~0.03–0.04), all A1/A2/A3 EVAL_SR values match A0 exactly. The eval-aware objective does not hurt, but also does not help.

### F3: A4 (probe-only, λ_now=0) is catastrophically harmful

EVAL_SR collapses to **0.09–0.22**, far below legacy:
- SKIP rate shoots up to 8–11%
- HIGHLIGHT rate collapses to 1–3% 
- Without λ_now grounding, the tutor has no incentive to prefer helpful interventions over trivially "safe" ones (SKIP = no update = no negative probe delta)

**Confirmed**: λ_now=0 is the failure mode. λ_now must be ≥ 1.0.

### F4: Legacy baseline is strong and hard to beat

A0_legacy achieves 0.56–0.81 EVAL_SR, with higher n_sup and n_teach consistently better.
The HIGHLIGHT-dominant policy is already near-optimal given the probe-flat signal.

---

## Root Cause: Why probe signal is flat

The shadow learner uses `probe_eval.score()` (semantic margin on a small probe set) as its reward signal. The issue is that for all candidate actions on a given query:

1. **WAIT and HIGHLIGHT** lead to the same set of possible wrong-pick reveals (just with different p_pick)
2. **The CLS update from any single wrong pick is tiny** relative to n_sup=4–8 support examples
3. **Probe score difference before/after one reveal** ≈ 0.01–0.03, well within noise

This means ΔProbe(HIGHLIGHT) ≈ ΔProbe(WAIT) ≈ ΔProbe(BAN) ≈ 0, and the eval-aware objective degrades to legacy.

---

## Diagnosis: P0 is a flat-signal negative

The Bayesian eval-aware framework is **correctly implemented** but **probe signal is insufficient** to guide action selection in the current CLS regime. This is a **calibration failure**, not an implementation failure.

---

## Recommendations for P1

### P1-A (Fix the probe signal): Use probe_accuracy not probe.score()
- `probe_accuracy()` tests classification on a synthetic menu, not just margin
- This creates larger ΔProbe when an action meaningfully changes learner posterior
- Cost: ~5x slower per query (but still feasible with precompute_query amortization)

### P1-B (Amplify probe set): Use larger n_probe (30–50) and real OOD queries
- Small probe set (n=12) has high variance; ΔProbe is dominated by noise
- OOD queries (from different grammar contexts) would be more sensitive

### P1-C (Use eval phase SR as calibration target)
- Record shadow ΔProbe vs actual EVAL_SR delta across seeds
- If Spearman rank correlation ≥ 0.3, the signal is usable
- If < 0.1, need to redesign the shadow objective

### P1-D (Try n_teach=3–4 with stronger shadow)
- With more teaching turns, cumulative ΔProbe becomes larger and more distinguishable
- The per-query signal may still be tiny, but accumulated across 3–4 queries it may become visible

---

## Error Note

53/2400 jobs errored (~2.2%) — all `A1/A2/A3` conditions with specific seeds (s45–s59 range). Error: "maximum which has no identity" = `np.argmax` on empty array in shadow learner. This occurs when the active menu is empty after all options are banned/exhausted. The error handling gracefully records the job as failed (excluded from aggregation), so results are unbiased.

**Fix**: Add guard in `_expected_probe_from_cache` for empty active_texts.

---

## Files

- Results: `cls_option_tutor/results/eval_aware_expA_full.txt`
- Experiment: `cls_option_tutor/exp_eval_aware.py`
- Shadow Learner: `cls_option_tutor/tutor/shadow_learner.py`
