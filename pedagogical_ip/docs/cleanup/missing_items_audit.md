# Missing Items Audit — Post Step 5 Cleanup

Generated: 2026-04-01

## 1. Code Residuals Found

### ⚠️ FIXED: prior_mode default was still "legacy_bonus"

| File | Line | Issue | Fix |
|:-----|:----:|:------|:----|
| `joint_goal_pref_posterior.py` | 77 | `prior_mode: str = "legacy_bonus"` | Changed to `"structural"` |

**Impact**: Any code instantiating `JointGoalPrefPosterior()` without explicit `prior_mode` was silently using the deprecated legacy bonus instead of the Step 4 canonical structural prior.

### ✓ CLEAN: No compatibility bonus in main update path

Searched for `exp.*beta_C`, `compatibility_bonus`, `compat.*bonus` in `src/` — **zero matches**. The legacy bonus is properly gated behind `prior_mode == "legacy_bonus"` in the update method.

### ✓ CLEAN: No deprecated imports in mainline src/

`composite_goal_compatibility` is imported only from:
- `tests/test_composite_goal_compatibility.py` (test file)
- `scripts/run_step4_5_promotion_audit.py` (experiment script)
- `scripts/run_t7b2_*.py`, `run_t7b_*.py` (old experiment scripts)

None of these are mainline `src/` imports. Safe.

### ✓ CLEAN: Shadow modules properly lazy-imported

In `internalization_control_tutor_v4.py`, shadow imports (`micro_bayes_shadow`, `v2`, `v2_1`, `v3`) are inside conditional blocks (line 370+), NOT at module level. They only activate when explicitly configured.

### ⚠️ OBSERVATION: Θ_K defined but not default

`THETA_K = ("safe", "shiny", "risky", "shortcut", "neutral")` is defined in `joint_goal_pref_posterior.py` line 39 but never used as default. Default is `THETA_2 = ("safe", "shiny")`. This is correct — Θ_K stays as research branch.

### ⚠️ OBSERVATION: Necessity gate imports in experiment scripts only

`necessity_gate_variants` is imported only from:
- `run_step5a1_necessity_gate_audit.py`
- `run_step5a2_cgc_promotion_audit.py`

NOT from any `src/` module. Safe — gate is experiment-only.

## 2. Documentation Residuals Found

### ⚠️ NEEDS UPDATE: Docstring in joint_goal_pref_posterior.py

Lines 6-9 still list `legacy_bonus` first in the prior mode documentation:
```
Step 4 prior modes:
  legacy_bonus:  original exp(β_C · C_t(g)) bonus in update (backward compat)
  structural:    P₀(g|c₀) at init, pure action-likelihood update
  pcfg:          PCFG-based P₀(g) at init, pure action-likelihood update
```

Should be reordered to list `structural` first as canonical.

### ✓ CLEAN: No documentation claiming "compatibility bonus is mainline"

No `.md` files in `docs/` claim legacy bonus is canonical.

## 3. Script Residuals Found

### ⚠️ OBSERVATION: 86 scripts in scripts/ directory

Many are from pre-Step-1 through Step 5. Current-step scripts (15):
- `run_step1_*.py` (2)
- `run_step2_*.py` (3)
- `run_step3_*.py` (2)
- `run_step4_*.py` (3)
- `run_step5*.py` (5)

Legacy/superseded scripts (71): These include `run_stage*`, `run_t*`, `run_p*`, `run_a*`, `run_final*`, etc. Most are from earlier experimental phases and should be archived.

### ⚠️ NEEDS CLEANUP: Debug/analysis scripts in scripts/

Files starting with `_` (4):
- `_analyze_compact.py`
- `_analyze_final.py`
- `_analyze_phase2a.py`
- `_debug_option_scores.py`
- `_diag_output.txt`
- `_diagnose_rsa.py`

These should move to `archive/legacy_runners/`.

## 4. Results Residuals Found

### ⚠️ OBSERVATION: 199 files in results/ root

- 10 structured subdirectories (`step2_phase2a/`, `step5a_planner/`, etc.) — KEEP
- ~40 `.txt` debug files — ARCHIVE
- ~30 `.csv` raw data files — ARCHIVE (reports supersede them)
- ~129 `.md` reports — Most are valuable; only pre-Step-1 reports could be archived

## 5. Configuration Residuals

### ✓ CLEAN: configs/agent.yaml

No legacy defaults found in active configs.

## 6. Summary

| Category | Found | Fixed | Pending |
|:---------|:-----:|:-----:|:-------:|
| Code residuals | 1 | 1 (prior_mode default) | 0 |
| Documentation | 1 | 0 | 1 (docstring order) |
| Script clutter | 75+ | 0 | Move to archive |
| Results clutter | 40+ | 0 | Move to archive |
| Import safety | 0 | — | — |
| Config safety | 0 | — | — |
