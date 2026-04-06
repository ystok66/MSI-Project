# Phase 1B Convergence Results — Boredom Utility

**Date**: 2026-04-06  
**Experiment**: E1 — Boredom Shadow Evaluation, 20 seeds × 4 β_bore × 2 families

---

## Summary Table

| Family | β_bore | Surv | Goal | FP_wait | BoreRatio | N |
|--------|--------|------|------|---------|-----------|---|
| GTET | **0.0** | 0.950 | 0.950 | **0.400** | 0.530 | 20 |
| GTET | **0.3** | **0.950** | **0.950** | **0.000** | 0.530 | 20 |
| GTET | 0.5 | 0.950 | 0.950 | 0.000 | 0.530 | 20 |
| GTET | 1.0 | 0.950 | 0.950 | 0.000 | 0.530 | 20 |
| DTMB | 0.0 | 0.450 | 0.100 | 0.000 | 0.468 | 20 |
| DTMB | 0.3 | 0.450 | 0.100 | 0.000 | 0.473 | 20 |
| DTMB | 0.5 | 0.450 | 0.100 | 0.000 | 0.473 | 20 |
| DTMB | 1.0 | 0.450 | 0.100 | 0.000 | 0.473 | 20 |

---

## Acceptance Criteria Assessment

### 1. GTET FP_wait: 0.400 → 0.000  ✅

Target: ≤ 0.45. Achieved: 0.000 at β_bore=0.3. This is a complete elimination of false-positive WAITs. The boredom penalty successfully prevents the tutor from issuing WAIT when the agent has nothing left to learn.

### 2. GTET TBSR unchanged  ✅

Survival stays at 0.950 and goal at 0.950 across all β_bore values. Zero regression.

### 3. DTMB no regression  ✅

DTMB results are identical (0.450/0.100) across all β_bore values. This is expected: DTMB agents die early and rarely WAIT, so the boredom penalty has no opportunity to fire.

### 4. Intervention rate unchanged  ✅

Steps remain at 51.0 for GTET across all conditions. The boredom penalty changes the internal Q_WAIT score but does not cause the tutor to switch to a different intervention type.

---

## Why B_wait = avg_cost / (ε + LG) works

The formula directly captures the proposal's boredom definition:
- When LG (learning gain / uncertainty along prefix) is high, B_wait ≈ 0 → WAIT is justified
- When LG ≈ 0 but cost > 0, B_wait explodes → WAIT is penalized
- The ratio form is more robust than thresholding because it naturally scales

At β_bore=0.3, the penalty is just strong enough to flip false-positive WAITs without over-penalizing genuine learning opportunities. Values of 0.5 and 1.0 also work but provide no additional benefit — the effect is binary at this scale.

---

## BoreRatio stays at 0.53

BoreRatio measures the **agent-side** stall cost fraction (how much time the agent spends in low-IG states). This is NOT a tutor metric — it reflects the lattice topology and task difficulty. The tutor's boredom penalty reduces FP_wait (tutor-side unnecessary WAITs) without changing the underlying agent dynamics. This is correct behavior: the tutor should stop issuing useless WAITs, but cannot change the fact that the agent sometimes traverses low-information regions.

---

## Verdict

> **VERDICT A: Boredom penalty effective. Promoted to canonical with β_bore=0.3.**
>
> - FP_wait eliminated (0.400 → 0.000 on GTET, 20 seeds)
> - Zero survival regression on both families
> - Single new hyperparameter (β_bore)
> - Formula is proposal-aligned: B = avg_cost / (ε + LG)

---

## Canonical Defaults Updated

| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| `warning_variant` | `legacy_bias` | `rsa_obs_s1` | Phase 1A: RSA = legacy |
| `boredom_weight` | 0.0 (off) | **0.3** | Phase 1B: FP_wait → 0 |
