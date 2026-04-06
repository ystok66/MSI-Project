# Regression Report — Post Step 5 Cleanup

Generated: 2026-04-01

## Test Results

| # | Check | Status |
|:-:|:------|:------:|
| 1 | Posterior default = `structural` | ✅ PASS |
| 2 | Structural prior normalized (sums to 1.0) | ✅ PASS |
| 3 | A2 shadow planner imports | ✅ PASS |
| 4 | Warning utterance policy imports | ✅ PASS |
| 5 | Frozen observer (`RuleBasedMtObserver`) imports | ✅ PASS |
| 6 | Frozen micro tutor (`BCICTv4`) imports | ✅ PASS |
| 7 | 8-goal hypothesis space | ✅ PASS |
| 8 | `subgoal_marginals()` returns valid dict | ✅ PASS |
| 9 | CGC-v2 episode generation | ✅ PASS |
| 10 | Deprecated compat module importable (backward compat) | ✅ PASS |

**ALL 10 CHECKS PASSED**

## What Was Fixed

| Issue | File | Fix |
|:------|:-----|:----|
| `prior_mode` default was `"legacy_bonus"` | `joint_goal_pref_posterior.py:77` | Changed to `"structural"` |

## What Was Validated

1. **Step 4 promotional decision is now code-operative**: `JointGoalPrefPosterior()` defaults to structural prior
2. **Frozen modules import clean**: Observer and micro tutor have no broken dependencies
3. **Shadow modules import clean**: A2 planner shadow works independently
4. **CGC-v2 generation works**: Real episode generation is unaffected by cleanup
5. **Backward compatibility maintained**: Deprecated `composite_goal_compatibility.py` still importable

## Regression Script

Located at: `scripts/_regression_check.py`

Run: `python scripts/_regression_check.py`
