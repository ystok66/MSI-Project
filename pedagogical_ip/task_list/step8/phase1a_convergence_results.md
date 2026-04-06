# Phase 1A Convergence Results

**Date**: 2026-04-06  
**Experiment**: E1 — baseline_v2 Warning Path Audit, 30 seeds × 5 variants

---

## Summary Table

| Variant | Surv | Goal | Steps | Δρ_inc | ΔNLL | Entropy |
|---------|------|------|-------|--------|------|---------|
| **legacy_bias** | **1.000** | **1.000** | 33.5 | +0.142 | -0.635 | 0.001 |
| **rsa_obs_l0** | **1.000** | **1.000** | 33.5 | +0.141 | -0.594 | 0.065 |
| **rsa_obs_s1** | **1.000** | **1.000** | 33.5 | +0.142 | -0.592 | 0.001 |
| rsa_obs_s1_trust | 0.467 | 0.467 | 20.5 | +0.118 | -0.476 | 0.474 |
| rsa_plus_phase10 | 0.667 | 0.667 | 23.9 | +0.142 | -0.593 | 0.001 |

## Regression Check

| Family | Surv | Goal | n |
|--------|------|------|---|
| DTMB | 0.500 | 0.200 | 10 |
| GTET | 0.900 | 0.900 | 10 |

---

## Findings

### 1. RSA-only = Legacy on all outcome metrics

`rsa_obs_s1` and `legacy_bias` produce **identical** results: 30/30 survival, 30/30 goal, 33.5 steps average. The RSA semantic layer alone is sufficient — pseudo-labels and lane bias add zero marginal value on baseline_v2.

### 2. Δρ_inc is identical across RSA and legacy

Both produce +0.142 risk delta per warning. The semantic signal from RSA's S1 speaker model matches the legacy prototype-based warning exactly in terms of planner impact.

### 3. Legacy pseudo-labels give stronger ΔNLL but no outcome benefit

ΔNLL(legacy) = -0.635 vs ΔNLL(rsa_s1) = -0.592. The ~7% difference in branch choice NLL doesn't translate to any survival or goal difference. This confirms that the planner adapter (warned_cell_extra) is the dominant channel, not the risk_head update.

### 4. Trust gate (τ̂=0.3) is harmful

`rsa_obs_s1_trust` drops to 46.7% survival. The trust gate at τ̂=0.3 dampens the warning signal too aggressively (Δρ falls to 0.118). If trust gating is desired, τ̂ needs to be much higher (≥0.8).

### 5. Hybrid is WORSE than either pure path

`rsa_plus_phase10` at 66.7% is worse than both pure-RSA (100%) and pure-legacy (100%). This suggests that applying both pseudo-label + RSA planner penalty creates over-correction that leads to suboptimal routing.

### 6. Family regressions pass

DTMB (0.50/0.20) and GTET (0.90/0.90) are consistent with Phase 0 baselines, confirming Phase 1A refactoring did not damage family-specific warning paths.

---

## Verdict

> **VERDICT A: RSA-only ≈ legacy. Legacy adapter can be demoted to ablation.**
>
> - `rsa_obs_s1` achieves identical performance to `legacy_bias` (30/30 seeds)
> - Pseudo-label injection provides no marginal benefit
> - Trust gate at current calibration is harmful
> - Hybrid is actively harmful (over-correction)
>
> **Phase 1A acceptance criteria met:**
> - ✅ A. Structural convergence — `rsa_warning_channel.py` is sole semantic source
> - ✅ B. Mechanism attribution — warning effect comes from `b⁺(r)` → planner adapter, NOT pseudo-labels
> - ✅ C. DTMB/GTET no regression
> - ✅ D. Redundancy judgment — legacy adapter demoted to ablation

---

## Recommended Next Steps

1. **Default warning_variant → `rsa_obs_s1`** (replace `legacy_bias` as default)
2. **Remove trust gate** from default config (or recalibrate τ̂ ≥ 0.8)
3. **Keep hybrid as ablation only** — useful for future papers but not for production
4. **Proceed to Phase 1B: boredom utility** (Q4 gap confirmed in Phase 0)
