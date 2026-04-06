# Mandatory Regression Protocol

**Authority: Task 1 Closure | Version: 1.0**

Any modification to canonical source code MUST pass this protocol before merge.

---

## Step 1: Unit Tests (Gate)

```bash
# WSL Ubuntu-24.04 + conda pedip310
python -m pytest tests/test_internalization_observer.py -v --tb=short
```

**Pass threshold**: 55/55 (zero failures allowed)

If this fails, **stop immediately** — do not proceed to experiments.

---

## Step 2: Behavioral Identity (Post-Modification)

If BCICTv4 or A1MtObserver source was modified:

```bash
python scripts/run_t1_smoke_check.py
```

**Must verify**: 26/26 steps, action identity, Q decomposition consistency.

---

## Step 3: κ̂ Default-On Regression (Exp-1 pattern)

```bash
python scripts/run_t1_exp1_kappa_regression.py
```

### Pass Criteria

| Metric | Threshold |
|--------|-----------|
| OOD Top-1 pass rate | 10/10 |
| Per-family ΔR² regression | 0/13 (none below −0.05) |
| Micro DivAll | ≤ 0.02 (canonical θ) |
| Macro Kendall τ | ≥ 0.95 |

---

## Step 4: Q-Margin Audit (optional, for boundary changes)

```bash
python scripts/run_t1_exp2_q_margin_audit.py
```

Only required if modification touches decision boundary logic.

### Report Requirements
- |ΔQ| binning table
- NearTieCoverage at ε=0.10
- Per-family over-warn distribution

---

## Step 5: Over-Warn Check

### Metrics to Report

| Metric | Formula | Threshold |
|--------|---------|-----------|
| OverWarnRate | Σ1[oracle=WAIT ∧ infer=WARN] / Σ1[oracle=WAIT] | ≤ 0.02 |
| WarnNecRecall | Σ1[oracle=WARN ∧ infer=WARN] / Σ1[oracle=WARN] | ≥ 0.85 |

---

## Step 6: OOD Robustness

5 conditions must all show Top-1 stability:

1. Canonical (baseline)
2. Temptation-rich (hidden_tempt=0.6)
3. Balanced-Active suite
4. High-risk (×1.5)
5. Low-risk (×0.5)

**Pass threshold**: 10/10 Top-1 same

---

## Failure Triage Order

If any step fails, investigate in this order:

1. **Unit test failure** → Check for syntax/import errors, then parameter conflicts
2. **Action identity failure** → Diff the Q logic, check for unintended branch changes
3. **OOD regression** → Check κ̂ direction (higher risk → higher κ̂), then check STOP formula
4. **Over-warn spike** → Run Q-margin audit, check if new near-tie families appeared
5. **Held-out MAE degradation** → Check if the modification affected observer estimation quality

---

## Script Reference

| Script | Purpose | Required? |
|--------|---------|:---------:|
| `pytest tests/test_internalization_observer.py` | Unit tests | Always |
| `run_t1_smoke_check.py` | Behavioral identity | After source mod |
| `run_t1_exp1_kappa_regression.py` | Full regression | Always |
| `run_t1_exp2_q_margin_audit.py` | Q boundary audit | If boundary changed |
| `run_t1_exp3_overwarn_fix.py` | Dead-zone sweep | If OWR increased |
| `run_t1_exp4_stability_retest.py` | Final lockdown | After any fix |
