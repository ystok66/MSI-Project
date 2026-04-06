# Task 1 Closure Note

**Date**: 2026-03-29 | **Status**: ✅ COMPLETE

---

## Final Status

| Question | Answer |
|----------|--------|
| Canonical locked? | **Yes** — 5D three-layer architecture with frozen parameters |
| κ̂ default-on? | **Yes** — β_κ=0.02, all OOD/regression checks pass |
| Over-warn acceptable? | **Yes** — 3/600 (0.5%) at baseline, 100% near-tie covered |
| Evaluation protocol fixed? | **Yes** — 6-step mandatory regression protocol |

---

## Experiment Summary

### Exp-1: κ̂ Default-On Regression ✅
- OOD pass rate: 10/10
- Per-family ΔR² regression: 0/13 (all positive, max +0.36 at tic_rescue_heavy)
- Top-1 stable across all conditions
- STOP stable (0.304–0.334)

### Exp-2: Q-Margin Audit ✅
- Over-warn: 3/600 steps (0.50%)
- NearTieCoverage(ε=0.10): 100%
- All 3 OWR events in |ΔQ| ∈ [0.054, 0.094]
- Dominant flip component: online Q (100%), V_full/R_over (0%)
- Focus families: blind_corridor (2), warn_rescue (1)
- Root cause: p_self very low (0.018–0.047), dc < dr geometry

### Exp-3: Dead-Zone Fix ✅
- ε_Q=0.05 selected: OWR reduced 40%, WarnNecRecall=1.0000 (zero loss)
- ε_Q=0.10 eliminates all OWR but drops WNR to 0.889
- Decision: ε_Q=0.05 is optimal (conservative, no WNR trade-off)
- Implementation: external wrapper, not in source code

### Exp-4: Stability Retest ✅
- OOD pass rate: 10/10
- Held-out 5D wins: 7/13
- Top-1 / STOP / Kendall all stable
- 55/55 unit tests pass

---

## Source Code Changes

Only one source file modified:

### `src/teachers/internalization_control_tutor_v4.py`
- Added `q_components` dict to cache per-dose Q breakdown in decide() loop
- Added `info["q_detail"]` with raw + weighted Q decomposition
- **Zero decision logic changes** — verified by 55/55 tests + 26/26 smoke check

---

## New Files Created

### Scripts (4)
- `scripts/run_t1_smoke_check.py` — Behavioral identity verifier
- `scripts/run_t1_exp1_kappa_regression.py` — Full family regression
- `scripts/run_t1_exp2_q_margin_audit.py` — Q-margin audit with binning
- `scripts/run_t1_exp3_overwarn_fix.py` — Dead-zone sweep
- `scripts/run_t1_exp4_stability_retest.py` — Final lockdown retest

### Documents (2)
- `docs/canonical_baseline_spec.md` — Frozen canonical configuration
- `docs/regression_protocol.md` — Mandatory regression suite

### Results (4)
- `results/t1_exp1_kappa_regression.md`
- `results/t1_exp2_q_margin_audit.md`
- `results/t1_exp3_overwarn_fix.md`
- `results/t1_exp4_stability_retest.md`

---

## Deferred to Future Tasks

| Item | Priority | Rationale |
|------|:--------:|-----------|
| Observer damping simplification | Medium | Clean-up, not baseline-critical |
| CurriculumControllerV13 refactoring | Medium | 634 lines, needs split |
| Legacy parameter cleanup | Low | Non-functional redundancy |
| Persistent learner profiles | High | Core κ̂ application |
| EPU / Belief-Horizon / EIG | Research | Shadow-mode only |
| Compositional goals (CGC-v2) | Research | Requires new scenario infra |
