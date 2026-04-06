# Archive Manifest — Post Step 5 Cleanup

Generated: 2026-04-01 | **EXECUTED**

## Completed Archive Copies

| Original Path | Archive Path | Status | Notes |
|:--------------|:-------------|:------:|:------|
| `src/teachers/composite_goal_compatibility.py` | `archive/deprecated/composite_goal_compatibility.py` | ✅ DONE | Original kept in-place with [DEPRECATED] header |
| `src/agents/continuous_reward_shadow.py` | `archive/paper_baselines/continuous_reward_shadow.py` | ✅ DONE | Original kept with [FROZEN] header |
| `src/teachers/bayesian_macro_objective_shadow.py` | `archive/paper_baselines/bayesian_macro_objective_shadow.py` | ✅ DONE | Original kept with [NARRATIVE-ONLY] header |
| `src/agents/necessity_gate_variants.py` | `archive/ablations/necessity_gate_variants.py` | ✅ DONE | Original kept with [NOT PROMOTING] header |

## Status Headers Added

| File | Header Tag |
|:-----|:-----------|
| `composite_goal_compatibility.py` | `[STATUS: DEPRECATED]` |
| `continuous_reward_shadow.py` | `[STATUS: FROZEN]` |
| `bayesian_macro_objective_shadow.py` | `[STATUS: NARRATIVE-ONLY]` |
| `necessity_gate_variants.py` | `[STATUS: NOT PROMOTING]` |
| `joint_goal_pref_posterior.py` | Docstring reordered: structural first as [CANONICAL DEFAULT] |

## Decision: Keep Originals In-Place

All four modules are kept in `src/` (not physically deleted) because:
1. Scripts and tests import them via relative paths
2. Shim approach breaks due to relative imports inside archived modules
3. Status headers + archive copies provide the necessary **cognitive separation**
4. Import reverse audit confirmed: zero `src/` mainline imports — only scripts/tests

## Not Moving (confirmed per user request)

| File | Category | Reason |
|:-----|:---------|:-------|
| `src/teachers/micro_bayes_shadow*.py` | Shadow active | Step 1 — consumed by tutor config |
| `src/teachers/a1mt_observer_shadow*.py` | Shadow active | Step 3 — still research-active |
| `src/agents/planner_risk_shadow.py` | Shadow active | Step 5A — promotion candidate |
| `src/teachers/effort_latent_shadow.py` | Shadow active | Step 3 dependency |
| `scripts/run_step*.py` | Current scripts | Still actively used |

## Pending (user approval needed for future)

| Pattern | Target | Count | Notes |
|:--------|:-------|:-----:|:------|
| `scripts/_analyze_*.py`, `_debug_*.py` | `archive/legacy_runners/` | ~6 | Debug one-offs |
| `scripts/run_stage*.py`, `run_t*.py`, `run_p*.py` | `archive/legacy_runners/` | ~71 | Pre-Step-1 scripts |
| `results/*.txt` debug files | `archive/old_reports/` | ~40 | Superseded by .md |
