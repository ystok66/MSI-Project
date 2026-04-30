# Tutor Diagnostic Suite Report

- Run directory: `runs/tutor_diagnostic_suite_4seed_20260421`
- Spec: `HugeRiskyGemMaze_v0`
- Seeds: `0 1 2 3`
- Workers: `16`
- Slices:
  - `t04_e05`: central diagnostic / over-waypoint slice
  - `t08_e07`: long-horizon positive slice

## What Changed

This run was the first pass after wiring in the most important diagnosis hooks for the current inverse tutor failure mode.

Implemented before the run:

- `warning_suspicion` modes in `risky_maze/learner/objective_agent.py`
  - `persistent` (current behavior)
  - `replan_only`
  - `none`
  - `episode_decay` plumbing for future sweeps
- Eval-time suspicion ablation
  - `clone_for_eval(..., clear_warning_suspicion=True)`
- Tutor guardrails in `risky_maze/tutor/inverse_planner.py`
  - `warning_actionability_threshold`
  - `waypoint_damage_veto_margin`
- New metrics in `risky_maze/runner/fixed_metrics.py`
  - `teach_safe_success_rate`
  - `damage_per_100_steps`
  - `trap_entries`
  - `warning_suspicion_mass_end_teach`
  - `warning_suspicion_mass_eval_start`
  - `warning_suspicion_mass_on_eval_path`
- New experiment runner
  - `python -m risky_maze.experiments.run_tutor_diagnostic_suite`

The goal of this suite was not to prove the full tutor already wins. It was to separate three hypotheses:

1. `warning_suspicion` carryover is the main reason `inverse_plan_full` loses.
2. Warning actions are being selected even when they are not actionable.
3. The full tutor still gets value from waypoint-only behavior after bad warnings are removed.

## Conditions

The diagnostic suite compared:

- `no_tutor_mortal`
- `always_warn_mortal`
- `inverse_plan_warn_only`
- `inverse_plan_full`
- `inverse_plan_full_clear_suspicion_eval`
- `inverse_plan_full_replan_only_suspicion`
- `inverse_plan_full_actionability_gated`
- `inverse_plan_full_combined_fix`

`inverse_plan_full_combined_fix` used:

- `ablate_eval_clear_warning_suspicion=True`
- `learner_warning_suspicion_mode=replan_only`
- `tutor_warning_actionability_threshold=0.05`
- `tutor_waypoint_damage_veto_margin=0.0`

## Metrics Used

The most informative additions in this run were:

- `TeachSafeSuccess`

  `success and damage == 0 and not died and not timeout`

- `DamagePer100Steps`

  `100 * damage / max(1, steps)`

- `WarningSuspicionMassEndTeach`

  `sum_c suspicion(c)` after teach

- `WarningSuspicionMassOnEvalPath`

  `sum_{c in eval_path} suspicion(c)` measured from the eval-start learner snapshot

These were more useful than raw `teach_success` alone, because both `T04` and `T08` were teach-success saturated under the current learner.

## Results

### T04 -> E05

| condition | teach safe success | teach damage | eval success | eval regret | eval damage | warnings | waypoints | suspicion end teach |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_tutor_mortal` | 0.25 | 1.50 | 1.00 | 21.00 | 0.50 | 0.00 | 0.00 | 0.00 |
| `always_warn_mortal` | 1.00 | 0.00 | 1.00 | 15.50 | 0.00 | 1.00 | 0.00 | 5.00 |
| `inverse_plan_warn_only` | 0.25 | 1.50 | 1.00 | 20.00 | 0.25 | 4.00 | 0.00 | 4.50 |
| `inverse_plan_full` | 0.00 | 2.00 | 1.00 | 26.00 | 1.00 | 3.25 | 2.00 | 3.25 |
| `full_clear_susp_eval` | 0.00 | 2.00 | 1.00 | 26.00 | 1.00 | 3.25 | 2.00 | 3.25 |
| `full_replan_only_susp` | 0.00 | 2.00 | 1.00 | 26.00 | 1.00 | 3.25 | 2.00 | 0.00 |
| `full_actionability_gated` | 0.00 | 2.00 | 1.00 | 21.00 | 0.50 | 0.00 | 2.00 | 0.00 |
| `full_combined_fix` | 0.00 | 2.00 | 1.00 | 21.00 | 0.50 | 0.00 | 2.00 | 0.00 |

Main takeaways for `T04`:

- `always_warn_mortal` is still clearly best on this slice.
- `inverse_plan_full` is worse than `no_tutor` on both `eval_regret` and `eval_damage`.
- Clearing suspicion for eval does nothing on `T04`.
- Switching suspicion to `replan_only` also does nothing on `T04`.
- The actionability gate is what matters here:
  - it removes warnings entirely
  - keeps the 2 waypoint interventions
  - brings `eval_regret` from `26.0` down to `21.0`
  - brings `eval_damage` from `1.0` down to `0.5`

Interpretation:

- `T04` is an over-intervention slice, not a suspicion-carryover slice.
- Harmful warning behavior is the main failure mode here.
- Once warnings are removed, the remaining waypoint behavior roughly collapses to `no_tutor`-level performance.
- This means current warning scoring is adding noise, not pedagogically useful pressure.

Supporting tutor-behavior evidence:

- `inverse_plan_full`
  - `warning_selected_rate = 0.0489`
  - `warning_actionability = 0.0`
- `inverse_plan_full_actionability_gated`
  - `warning_selected_rate = 0.0`
  - `waypoint_selected_rate ≈ 0.0305`

So on `T04`, current full tutor is selecting warnings that are not actually route-repairing.

### T08 -> E07

| condition | teach safe success | teach damage | eval success | eval regret | eval damage | warnings | waypoints | suspicion end teach | suspicion on eval path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_tutor_mortal` | 0.25 | 0.75 | 1.00 | 68.00 | 2.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `always_warn_mortal` | 0.00 | 1.00 | 1.00 | 39.00 | 1.50 | 3.25 | 0.00 | 16.25 | 0.00 |
| `inverse_plan_warn_only` | 0.00 | 1.00 | 0.50 | 44.75 | 2.50 | 7.75 | 0.00 | 10.75 | 0.00 |
| `inverse_plan_full` | 0.00 | 1.00 | 1.00 | 75.50 | 1.50 | 8.25 | 2.00 | 11.00 | 0.75 |
| `full_clear_susp_eval` | 0.00 | 1.00 | 0.75 | 65.00 | 2.00 | 8.25 | 2.00 | 11.00 | 0.00 |
| `full_replan_only_susp` | 0.25 | 0.75 | 0.75 | 57.75 | 2.50 | 8.00 | 2.00 | 0.00 | 0.00 |
| `full_actionability_gated` | 0.25 | 0.75 | 1.00 | 71.50 | 2.00 | 0.00 | 2.00 | 0.00 | 0.00 |
| `full_combined_fix` | 0.25 | 0.75 | 1.00 | 71.50 | 2.00 | 0.00 | 2.00 | 0.00 | 0.00 |

Main takeaways for `T08`:

- `teach_success_rate = 1.0` for every condition.
  - This confirms `T08` is not teach-safety-hard under the current HP / time-limit / learner setting.
  - `teach_success` is not a useful headline metric on this slice.
- `always_warn_mortal` is still the strongest result:
  - `eval_success = 1.0`
  - `eval_regret = 39.0`
  - `eval_damage = 1.5`
- `inverse_plan_warn_only` is no longer all-WAIT, but it is unstable:
  - `warnings = 7.75`
  - `warning_actionability = 0.0`
  - `eval_success = 0.5`
  - `eval_damage = 2.5`
- Current `inverse_plan_full` is the worst regret case among successful policies:
  - `eval_success = 1.0`
  - `eval_regret = 75.5`
  - `warnings = 8.25`
  - `waypoints = 2.0`

Interpretation:

- Suspicion carryover is part of the story on `T08`, but not the whole story.
- Clearing eval suspicion helps regret:
  - `75.5 -> 65.0`
  - but also drops `eval_success` from `1.0 -> 0.75`
- Replan-only suspicion helps regret more:
  - `75.5 -> 57.75`
  - but again drops `eval_success` to `0.75`
  - and increases `eval_damage` to `2.5`
  - while also weakening risk quality:
    - `risk_auc = 0.958`
    - `risk_calibration_ece = 0.256`
- Actionability gating removes warnings entirely:
  - `warnings = 0.0`
  - `waypoints = 2.0`
  - `eval_success = 1.0`
  - but `eval_regret = 71.5`, still not better than `no_tutor` (`68.0`)

This means:

1. Bad warnings are hurting the current full tutor.
2. Suspicion carryover is not the dominant failure mode by itself.
3. After removing warnings, waypoint-only behavior is still not enough to beat `no_tutor`.

Supporting tutor-behavior evidence:

- `inverse_plan_full`
  - `wait_selected_rate = 0.9232`
  - `warning_selected_rate = 0.0618`
  - `waypoint_selected_rate = 0.0150`
  - `mean_q_wait = 9.5653`
  - `mean_q_best_warning = 9.3804`
- `inverse_plan_full_actionability_gated`
  - `warning_selected_rate = 0.0`
  - `waypoint_selected_rate = 0.0157`
  - `mean_q_best_warning = 10.4916`
  - but blocked by the actionability threshold

That last point is important: the rollout still assigns large value to warning candidates, but their measured actionability remains near zero. So the problem is not just "the tutor never wants to warn". The bigger problem is that the current rollout utility overvalues warnings that do not actually remove downstream trap exposure.

## Cross-Slice Conclusions

### 1. `always_warn_mortal` is still the best simple policy on both slices

This is the strongest practical result from the suite.

- On `T04`, it beats all inverse variants on both safety and regret.
- On `T08`, it gives the best regret by a wide margin.

So current inverse tutors are not yet delivering a real advantage over a simple warning policy.

### 2. `warning_actionability` is the key blocker

Observed:

- `t04_e05 / inverse_plan_full`: `warning_actionability = 0.0`
- `t08_e07 / inverse_plan_full`: `warning_actionability = 0.0208`
- `t08_e07 / inverse_plan_warn_only`: `warning_actionability = 0.0`

This confirms the core mechanism problem:

- warning candidates are being chosen
- warning updates are changing belief and path
- but those changes are not reliably converting a dangerous prefix into a safer one

That is exactly why removing warnings on `T04` helps so much.

### 3. Suspicion carryover matters on `T08`, but is not the main explanation

Evidence:

- `inverse_plan_full`
  - `warning_suspicion_mass_eval_start = 11.0`
  - `warning_suspicion_mass_on_eval_path = 0.75`
- `full_clear_susp_eval`
  - eval suspicion drops to `0.0`
  - regret improves `75.5 -> 65.0`
  - success drops `1.0 -> 0.75`
- `full_replan_only_susp`
  - teach suspicion mass drops to `0.0`
  - regret improves further to `57.75`
  - success still drops to `0.75`

Conclusion:

- Suspicion carryover is a real contributor to excess regret.
- But removing it does not restore robust performance.
- The tutor still needs better warning selection, not just less persistent suspicion.

### 4. The current actionability gate is useful but too blunt

Current gate behavior:

- It cleanly removes low-actionability warnings.
- On `T04`, this is good.
- On `T08`, it deletes warnings entirely and leaves only waypoint behavior.

So the gate is helpful as a diagnostic instrument, but not yet a full solution.

## Practical Next Steps

Priority order after this run:

1. Improve warning scoring, not just warning filtering.

   The rollout already predicts large warning value, but realized actionability stays near zero. The next fix should be to score warnings using a stronger counterfactual:

   - compare `pre-warning true trap count on predicted prefix`
   - compare `post-warning true trap count on predicted prefix`
   - require a positive margin before warning enters the candidate set

2. Keep `warning_suspicion` out of long-term eval memory by default.

   The current suite suggests `persistent` should not remain the default for the main paper path.

   Recommended next default:

   - `replan_only` or another narrowly scoped non-persistent mode

3. Treat `T04` as an over-intervention diagnostic, not as a headline win slice.

   On `T04`, the key question is:

   - can we avoid harmful warnings and over-pointing?

   It is not the place to claim long-horizon pedagogical advantage.

4. Treat `T08` as a positive candidate slice, but not yet a positive result.

   The run shows:

   - there is room for improvement
   - suspicion changes matter
   - warning choice matters

   But current inverse tutors still do not beat `always_warn`.

5. Use `MiniRiskGate_v0` for the next warning-only debugging pass.

   This suite confirms that `T08` is not teach-safety-hard and `T04` is mainly an over-intervention slice. The next warning-actionability debug pass should move to the compact gate map, where obvious catastrophic prefixes exist and the failure signal is easier to isolate.

## Bottom Line

This run does not support the claim that `inverse_plan_full` is already better than simple baselines.

It does support three narrower and very useful claims:

1. Current full-tutor failures are driven more by bad warning behavior than by missing waypoint bandwidth.
2. Suspicion carryover contributes to regret on long-horizon tasks, but removing it alone does not fix the tutor.
3. A minimal actionability gate is directionally correct: on the over-intervention slice it removes the harmful part of the policy almost immediately.

So the next engineering target is now much sharper:

- build warnings that are actually route-repairing
- keep suspicion local unless there is strong evidence
- only then revisit whether waypoint adds pedagogical value beyond `always_warn`
