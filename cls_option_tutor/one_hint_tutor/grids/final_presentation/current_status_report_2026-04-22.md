# One-Hint Tutor Current Status Report

Date: `2026-04-22`

This note consolidates the current state of the `cls_option_tutor/one_hint_tutor` project for presentation use. It separates:

- what is already implemented and validated,
- what is currently used as the main benchmark,
- what has been probed experimentally but is not yet part of the stable story,
- and what remains a proposed next step rather than a result.

## 1. Project Goal

The project studies a one-shot tutor that observes a learner, infers a posterior over learner profiles, then selects a single pedagogical hint before a hard option-search teach episode.

The main research question is:

> Can an intentional one-shot hint help the learner complete a hard bounded teach task better than no tutor or random hints, and can the tutor's utility control the trade-off between immediate teach success and downstream transfer?

## 2. Protocol and Scenario

The current protocol is:

1. `prelearn`
2. `observation`
3. `inverse posterior fitting`
4. `one hint`
5. `teach`
6. `eval`

The learner first studies a small support set (`prelearn`), then is observed on several held-out observation tasks. The tutor fits an inverse posterior over learner profiles from those observations. At teach time, the tutor may provide one hint, then the learner attempts a bounded multiple-choice search problem with reveal-style feedback. After teach, the learner is evaluated on derived generalization items.

### Main controls

All main experiments compare against:

- `no_tutor_T`
- `no_tutor_T+H`
- `random_hard_hint_T`
- `random_same_pool_hint_T`
- `tutor`
- `oracle`

Here `H=1`, so `no_tutor_T+H` gives no-tutor one extra attempt as a fair baseline against the cost of one hint.

## 3. Learner and Tutor Mechanism

### Learner

The learner is CLS-based and updates on examples during prelearn and teach. Wrong reveals can change the learner's internal concept state, which is why post-reveal dynamics matter.

### Tutor

The tutor:

1. builds a hint candidate pool,
2. evaluates candidates under a posterior over learner profiles,
3. uses cascade planning:
   - prefilter,
   - proxy rollout,
   - optional refine,
4. selects the best hint according to a utility.

### Important planner approximation

The deployed planner still relies heavily on cheap static or short-horizon proxies such as:

- `initial_correct_prob_mean`
- `initial_correct_rank_mean`
- short rollout `success_prob`
- first-reveal cached CLS refinement

These are useful, but they are not equivalent to true bounded teach success.

### 3.1 Main code path map

The core implementation path currently used by the stable benchmark is:

1. experiment setup:
   - `protocol.py`
   - `experiment_presets.py`
2. teach/eval execution:
   - `learner_runner.py`
   - `baselines.py`
3. hint generation:
   - `hint_space.py`
4. planning and selection:
   - `hint_planner.py`
   - `metrics.py`
   - `rollout.py`
5. grid orchestration and summaries:
   - `experiment_matrix.py`
   - `run_one_hint_grid.py`

The most relevant code-level notions behind the current story are:

- protocol construction:
  - prelearn, obs, rank-stratified teach selection
- candidate generation:
  - `free`
  - `operator_probe`
  - `target_neighborhood_robust_filtered`
- planner utility:
  - `advantage_delta`
- fair controls:
  - `no_tutor_T`
  - `no_tutor_T+H`
  - `random_hard_hint_T`
  - `random_same_pool_hint_T`
- diagnostics:
  - `run_fast_utility_diagnostics.py`

## 4. Current Main Benchmark

The current fixed main benchmark is:

```text
task = 000001
prelearn = 4
obs = 4
teach difficulty = hard
menu difficulty = rank-stratified
K = 20
tutor_T = 5
no_tutor_bonus = T + 1 = 6
hint_count = 1
common_randomness = true
```

The most important rank-stratified benchmark variants are:

- `search_main_free`
- `search_main_free_operator`
- `search_main_free_operator_target_robust`

### Why rank-stratified challenge was needed

Earlier settings often made `no_tutor` too strong. The rank-stratified menu construction fixed this by choosing teach menus whose no-hint initial correct rank falls in a target range, so the task is reachable but nontrivial.

## 5. Current Utility Definitions

### 5.1 Main deployable utility: `advantage_delta`

In code, the deployable benchmark currently uses `utility_mode = "advantage_delta"`.

The implemented absolute utility is:

```text
U_abs(h) =
  lambda_success * success_prob
  + lambda_eval_cell * eval_cell_acc
  + lambda_soft_tau * soft_tau_score
  + lambda_exposure * safe_wrong_mean
  - lambda_fail * fail_prob
  - lambda_collapse * collapse_penalty
```

The deployed planner compares this against a fair no-tutor bonus baseline:

```text
U_delta(h) = U_abs(h) - U_abs(no_tutor_T+H)
```

With the current mainline weights:

```text
lambda_success = 3.0
lambda_fail = 8.0
lambda_eval_cell = 2.0
lambda_soft_tau = 2.0
lambda_exposure = 0.5
lambda_collapse = 3.0
soft_tau_center = 4.5
soft_tau_sigma = 1.5
```

Because `fail_prob = 1 - success_prob`, the `success` and `fail` terms are redundant under delta utility. On the main benchmark this simplifies mathematically to:

```text
11 * Delta(success_prob)
+ 2 * Delta(eval_cell)
+ 2 * Delta(soft_tau_score)
+ 0.5 * Delta(safe_wrong_mean)
- 3 * Delta(collapse_penalty)
```

plus an early-no-transfer gate/penalty when enabled.

### 5.2 Other probed utilities

These were explored, but are not part of the fixed mainline:

- `fast_success`
- `min_updates`
- `success_only`
- `search_minimal`
- various weight-drop ablations

## 6. Fixed Main Story

### 6.1 Search-oriented main result

The main result to present is:

```text
search_main_free_operator
```

This is:

```text
families = free + operator_probe
utility = advantage_delta
benchmark = rank-stratified / pre4 / obs4 / K20 / T5 / H1
```

Key 20-seed result:

| Condition | Success |
| --- | ---: |
| `no_tutor_T` | 0.20 |
| `no_tutor_T+H` | 0.25 |
| `random_hard_hint_T` | 0.33 |
| `random_same_pool` | 0.27 |
| `tutor` | **0.55** |
| `oracle` | 0.70 |

Other important metrics:

- tutor band success: `0.45`
- tutor early success: `0.10`
- tutor early-no-transfer: `0.00`
- tutor eval cell: `0.2996`

Main interpretation:

> In the rank-stratified challenge setting, a one-shot tutor with `free + operator_probe` hints improves bounded teach/search success over both no-tutor-with-bonus and random same-pool hints.

### 6.2 Transfer-oriented secondary result

The best transfer-oriented variant is:

```text
search_main_free_operator_target_robust
```

This keeps the same benchmark but expands the candidate pool to:

```text
free + operator_probe + target_neighborhood_robust_filtered
```

Key 20-seed result:

- tutor success: `0.55`
- tutor eval exact: `0.0467`
- tutor eval cell: `0.3153`
- delta eval cell vs `no_tutor_T+H`: `+0.0345`
- delta eval cell vs `random_same_pool`: `+0.0358`

But the trajectory is less clean:

- band success: `0.35`
- early success: `0.20`
- early-no-transfer: `0.20`

Interpretation:

> The target-robust pool improves transfer signal, but not cleanly enough to replace the search-oriented mainline.

## 7. Tutor/Learner Mechanism in Practice

### 7.1 Candidate families currently available

The implemented candidate families include:

- `free`
- `operator_probe`
- `target_neighborhood`
- `target_neighborhood_loose`
- `target_neighborhood_rank_filtered`
- `target_neighborhood_robust_filtered`
- `menu_wrong`
- `menu_correct_ceiling`

#### `free`

Examples sampled from easy/medium/hard pools.

#### `operator_probe`

Examples constructed by instantiating operators that overlap with the teach example's operators.

#### `target_neighborhood*`

Examples structurally close to the teach example, including:

- contiguous subexpressions,
- atom replacements,
- operator swaps,
- overlap-based examples from the pool.

The robust-filtered version is then constrained by rank and post-reveal robustness filters inside the planner.

### 7.2 Diagnostic-only answer-neighbor family

A new diagnostic-only family was added locally in the fast diagnostic runner:

```text
answer_neighbor_nonanswer
```

It is a non-answer neighborhood around the correct teach expression:

- not identical to the correct teach program,
- grammar-valid,
- not a direct answer reveal,
- intended only as a diagnostic probe.

It is not part of the fixed mainline benchmark.

## 8. Recent Experimental Arc

### 8.1 Regime discovery

Regime discovery established that:

- ordinary settings were often too easy,
- rank-stratified `pre4 / K20 / T5` is the first clean challenge regime where tutor advantage appears.

### 8.2 Search main benchmark

`free + operator_probe` became the fixed search-oriented mainline because:

- it improves bounded success strongly,
- it beats both `no_tutor_T+H` and `random_same_pool`,
- it has low early-no-transfer.

### 8.3 Transfer benchmark

`free + operator_probe + target_robust` was retained as a secondary transfer-oriented variant because it improves eval cell more strongly, even though it is less clean in trajectory.

### 8.4 Utility ablation

The `utility_ablation_search_main_free_operator_20seed` results showed:

- `success` is essential,
- `fail_prob` is redundant under delta utility,
- `eval_cell` is not the main driver of selection on the main benchmark,
- `soft_tau` has real shaping effect,
- `collapse` has little visible effect on the current stabilized benchmark,
- `exposure` is a weak shaping term,
- removing the early gate can increase success and eval together on this benchmark.

The most important ablation takeaways are:

1. `utility_no_success` collapses tutor success from `0.55` to `0.20`.
2. `utility_failprob_equiv_check` matches the baseline, confirming redundancy.
3. `utility_no_eval` is almost unchanged, so eval is not the main selection driver on this benchmark.
4. `utility_no_soft_tau` reduces tutor success from `0.55` to `0.45`.
5. `utility_no_early_gate` raises success from `0.55` to `0.65` and eval cell from `0.2996` to `0.3086`.

Interpretation:

> The current full utility is not yet sitting on a clean search-vs-transfer Pareto frontier. On this benchmark, the early gate is conservative enough that removing it helps both search and eval.

### 8.5 Fast-success side probes

The `fast_success_tradeoff_sideprobe_20seed` experiments compared:

- `balanced_full_utility`
- `success_fast_deployable`
- `success_fast_oracle`
- `eval_cell_oracle`

Key findings:

- `balanced_full_utility`: success `0.55`, eval cell `0.2996`
- `success_fast_deployable`: success `0.30`, eval cell `0.2971`
- `success_fast_oracle`: success `0.70`, eval cell `0.3057`
- `eval_cell_oracle`: success `0.30`, eval cell `0.3493`

Interpretation:

> A clean search-vs-transfer frontier exists at the oracle level, but the deployable fast-success utility does not yet find it.

### 8.6 `min_updates` follow-up

`success_fast_deployable` was rerun with the exploratory `min_updates` objective:

- success improved from `0.30` to `0.35`,
- still much worse than the balanced mainline.

Interpretation:

> Explicitly rewarding fewer updates/wrong reveals helps a bit, but does not close the gap to the balanced utility.

### 8.7 Soft-tau center probes

For `free + operator_probe`, no early gate:

- `center = 4.5`: success `0.65`, eval cell `0.3086`
- `center = 2.0`: success `0.50`, eval cell `0.3070`

For `free + operator_probe + target_robust`, low-rollout:

- `center = 4.5`: success `0.45`, eval cell `0.2795`
- `center = 2.0`: success `0.50`, eval cell `0.2709`

But after rerunning the target-robust comparison at higher rollout:

- `center = 4.5`: success `0.50`, eval cell `0.2962`
- `center = 2.0`: success `0.45`, eval cell `0.2919`

Interpretation:

> The low-rollout success-vs-eval trade-off from `soft_tau_center=2` was not stable. High-rollout reruns suggest the earlier effect was driven largely by rollout noise.

## 9. Current Core Problem

The desired claim is:

> The same inverse tutor, with different utility preferences, should be able to shift between:
> - teach-strong / eval-weaker behavior, and
> - eval-strong / teach-weaker behavior.

At the moment, this is **not yet cleanly demonstrated**.

### What is true now

1. We do have a search-oriented mainline (`free + operator_probe`).
2. We do have a more transfer-oriented secondary variant (`free + operator_probe + target_robust`).
3. But those differ in both:
   - utility behavior,
   - and candidate pool.

So the current results do **not** yet prove:

> the same tutor, holding the candidate pool fixed, can be smoothly steered along a teach-vs-transfer frontier purely by utility choice.

### Why not

The recent evidence suggests three coupled bottlenecks:

1. `soft_tau_center` is only weak shaping, not a true fast-success objective.
2. Fast-success prediction is still too static/noisy.
3. Candidate-family availability matters as much as utility.

## 10. Seed-4 Fast Diagnostic

For presentation debugging, an isolated seed-4 diagnostic runner was added. It does not alter the main benchmark path.

The runner compared:

- `balanced_full`
- `fast_soft_tau_center2`
- `fast_explicit_dynamic`
- `fast_explicit_dynamic_answer_neighbor`

### Important note

The markdown summary generated by the current seed-4 diagnostic script still contains stale `selected_*` aggregation fields. The correct interpretation should be taken from the JSON row payload rather than the generated markdown summary.

### JSON-grounded seed-4 findings

For `seed=4`:

- candidate count: `28`
- true `tau<=2` candidates: `13`
- true `tau<=5` candidates: `16`

So this seed does **not** support the story "there are no fast hints in the pool."

#### What happened

`balanced_full`, `fast_soft_tau_center2`, and `fast_explicit_dynamic` all selected the same `free` hint:

```text
['blicket', 'tufa', 'blicket', 'fep', 'kiki', 'fep', 'dax']
```

and all failed with:

- success `0`
- wrong-before-correct `5`
- eval cell `0.4661`

This means:

> simply changing the utility was not enough to move the planner to a different fast-success hint on this seed.

However, `fast_explicit_dynamic_answer_neighbor` selected a different hint:

```text
['kiki', 'fep', 'lug', 'blicket']
```

with family:

```text
operator_probe
```

and achieved:

- success `1`
- `tau = 2`
- wrong-before-correct `1`
- eval cell `0.4436`

Interpretation:

1. The bottleneck is not just "no fast hints exist."
2. Utility changes alone still failed to surface a fast hint.
3. Once the candidate landscape was perturbed by answer-neighbor diagnostics, a real `tau=2` solution became selectable.
4. The selected hint was still `operator_probe`, not the new answer-neighbor family itself.

So the cleanest interpretation is:

> candidate-pool/ordering sensitivity remains a major bottleneck, and utility alone is not yet sufficient to steer fast-success behavior.

## 11. Current Interpretation of `initial_correct_rank`

`initial_correct_rank` remains useful, but only as a cheap reachability proxy.

### What it measures

It measures:

> where the correct option ranks in the learner's initial option distribution before the teach interaction unfolds.

### Why it is optimistic

It only reflects step-0 static reachability. It ignores:

- wrong reveals,
- reveal-conditioned learner updates,
- bounded interaction dynamics,
- narrow probability margins among top options.

Thus:

> `initial_correct_rank` is useful for challenge-set construction and cheap prefiltering, but it is not a fast-success proxy.

## 12. Beam-SMC and Exposure-Sensitive Eval Split

These are discussed directions, not current results.

### 12.1 Dynamic fast metrics

The current diagnostic runner already computes an exact-ish two-step fast metric:

```text
P(tau<=2)
= p0(correct)
+ sum_a p0(a) * p1(correct | first wrong reveal on a)
```

This is now available in the isolated diagnostic path.

### 12.2 Beam-SMC

The proposed next step for bounded fast success is to add a beam-SMC style dynamic estimate for:

- `P(tau<=5)`
- `E[tau]`
- `E[wrong reveals before correct]`
- collapse risk

Status:

- discussed,
- not implemented as the current main metric path,
- not part of the fixed benchmark claim.

### 12.3 Exposure-sensitive eval split

The proposed idea is to split eval into:

- `eval_cell_all`
- `eval_cell_exposure_sensitive`
- `eval_cell_general`

to better measure whether fast tutors lose transfer because they encounter fewer informative wrong reveals.

Status:

- discussed,
- not implemented yet.

## 13. Fixed Mainline for Presentation

### Main benchmark

Use:

```text
search_main_free_operator
```

with the story:

> In a rank-stratified hard option-search setting, one-shot tutor hint selection improves bounded teach success over both no-tutor-with-bonus and random same-pool hints.

### Secondary transfer variant

Use:

```text
search_main_free_operator_target_robust
```

with the story:

> Adding target-robust neighborhood candidates improves transfer signal, but the teach trajectory becomes less clean.

### Do not use as the main claim

- reranker benchmarks,
- fast-success deployable utility,
- seed-4 answer-neighbor diagnostic,
- any beam-SMC or exposure-sensitive eval claims.

These are still diagnostic or exploratory.

## 14. Bottom Line

The project already supports one clear, defensible claim:

> In a rank-stratified challenge benchmark, the tutor's intentional hint selection improves bounded teach/search success over fair no-tutor and random same-pool controls.

What the project does **not** yet cleanly support is:

> that the same tutor can be smoothly steered, using utility alone, along a clean teach-success vs eval-transfer frontier.

The evidence so far says:

- the search-oriented tutor is real,
- the transfer-oriented signal is real,
- oracle headroom remains large,
- utility shaping matters,
- but candidate-family availability and dynamic fastness estimation still bottleneck the desired trade-off control.

## 15. Source Artifacts

The main quantitative sources behind this report are:

- `grids/search_main_full_summary.md`
- `grids/transfer_main_full_summary.md`
- `grids/final_presentation/utility_ablation_search_main_free_operator_20seed_summary.md`
- `grids/final_presentation/fast_success_tradeoff_sideprobe_20seed_summary.md`
- `grids/final_presentation/fast_success_deployable_min_updates_20seed_summary.md`
- `grids/final_presentation/soft_tau_center_no_early_gate_20seed_summary.md`
- `grids/final_presentation/soft_tau_center_no_early_gate_target_robust_20seed_summary.md`
- `grids/final_presentation/soft_tau_center_no_early_gate_target_robust_highrollout_20seed_summary.md`
- `grids/final_presentation/fast_utility_diagnostic_seed4_results.json`
