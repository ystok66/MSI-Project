# Goal B Route A-support / Route B Implementation Spec

## 0. Purpose

This spec defines two concrete implementation routes for Goal B:

- `Route A-support`: query-local wrong history changes the final correct-step support set
- `Route B`: weak long-term wrong update + query-local history + final correct commit

The target question is:

> How can `wrong × k + correct` outperform `direct-correct` on eval,
> while also making the learned grammar subjectively closer to the real grammar?

This document is intentionally implementation-oriented. It avoids treating unverified mechanism hypotheses as facts, and it tries to minimize redundant hyperparameters and moving parts.

---

## 1. Current Constraint

Current code already supports:

- wrong-side grammar write:
  - [feedback_update.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/feedback_update.py:22>)
- direct correct constrained positive update:
  - [correct_update.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/correct_update.py:22>)
- query-local Goal B state:
  - [goal_b_state.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/goal_b_state.py:30>)
- grammar / probe diagnostics:
  - [goal_b_metrics.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/goal_b_metrics.py:32>)

Current negative results imply:

1. `correct × N` beats `wrong × k + correct` under matched budget.
2. Reweight-only local history on a fixed `Π_GT` does not beat `direct-correct`.
3. To make Goal B plausible, wrong history must provide extra structure not already contained in direct GT supervision.

The most important practical consequence is:

> Wrong history must either change the final candidate support, or change the long-term learner state in a weak and structured way. Pure posterior sharpening on a fixed `Π_GT` is not enough.

---

## 2. Success Criteria

We are not looking for any local improvement. We want a protocol-level win.

### 2.1 Primary success criterion

For some small `k ∈ {1, 2, 3}`:

```text
Eval(wrong × k + correct) > Eval(direct-correct)
```

where eval includes both:

- query-local grammar metrics:
  - `gt_mass`
  - `gt_rank`
  - `H_beam`
  - `margin`
- held-out generalization metrics:
  - `probe_acc`
  - `probe_ll`

### 2.2 Stronger success criterion

Under matched budget:

```text
Eval(wrong × k + correct) >= Eval(correct × (k+1))
```

This is hard. Do not assume it will happen early.

### 2.3 Mechanism criterion

The marginal gain of the final correct step should increase or at least stop decreasing:

```text
Gain_correct(Sk) >= Gain_correct(S0)
```

for at least one `k`.

If the final correct gain keeps shrinking with more prior wrongs, Goal B is still failing mechanistically.

---

## 3. Shared Formalization

Let:

- `x` be the input words
- `GT` be the correct output sequence
- `D^- = {(ŷ_t, m_t)}_{t=1..k}` be the wrong-confirm history
- `π` be a latent trace
- `s(π; x, θ)` be the trace score under grammar parameters `θ`

Base posterior:

```text
q(π | x, θ) ∝ exp(s(π; x, θ))
```

Direct-correct posterior:

```text
q_dir(π | x, GT, θ)
∝ exp(s(π; x, θ)) · 1[π ∈ Π_GT]
```

where:

```text
Π_GT = {π : Y(π) = GT}
```

Goal B requires a different final posterior:

```text
q_goalB(π | x, GT, D^-, θ)
∝ exp(s(π; x, θ)) · 1[π ∈ Π_goalB(GT, D^-)] · C(π; D^-)
```

Two important levers appear here:

- `Π_goalB(GT, D^-)`: support set
- `C(π; D^-)`: history-dependent trace weighting

Current Route A v2 only touched `C`, and only inside a fixed `Π_GT`. That was not enough.

---

## 4. Route A-support

## 4.1 Goal

Do not write wrong feedback directly into long-term grammar. Instead:

1. accumulate wrong history in a query-local state
2. use that history to change which traces are visible to the final correct step
3. commit once at the end

This keeps the protocol clean while giving wrong history a chance to provide genuinely new structural information.

## 4.2 Core idea

Instead of:

```text
Π_goalB = Π_GT
```

use:

```text
Π_goalB = Π_GT ∪ Π_rescue(D^-)
```

where `Π_rescue(D^-)` is a small set of traces generated from wrong history that are:

- structurally relevant to the wrong attempts
- still useful for discriminating GT-compatible families
- not already contained in the direct-correct support

The final posterior becomes:

```text
q_A(π)
∝ exp(s(π)) · 1[π ∈ Π_GT ∪ Π_rescue(D^-)] · C_A(π; D^-)
```

with:

- `C_A(π; D^-) = 1` for the first implementation
- optional history weighting added only after support intervention works

This is intentional. Support change is the main experiment. Weighting is secondary.

## 4.3 Minimal implementation

### Step A1. Extend protocol-local state

Keep [goal_b_state.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/goal_b_state.py:30>) as the state owner.

Add fields to `WrongObservation`:

```python
beam_top_outputs: Optional[list] = None
beam_top_scores: Optional[list] = None
trace_features_before: Optional[list] = None
```

Add helper methods:

```python
def add_wrong_from_beam(...)
def summarize_wrong_constraints(...)
def build_support_hints(...)
```

Do not move this into `QueryMemory` yet.

### Step A2. Add support-builder functions

Create a new module:

- `cls_color_selection/cls_color_selection/learner/goal_b_support.py`

Add one primary function:

```python
def build_goal_b_support(
    predictor,
    words,
    ground_truth,
    history_state,
    *,
    max_gt_traces=16,
    max_rescue_traces=8,
    support_mode="gt_plus_repair",
):
    ...
```

Return:

```python
{
    "gt_traces": [...],
    "rescue_traces": [...],
    "merged_traces": [...],
    "diag": {...},
}
```

### Step A3. How to build `Π_rescue(D^-)`

Keep this minimal. Do not invent a new search engine.

Use one of these two variants.

#### Variant A3.1 `repair_from_wrong_outputs`

For each wrong submission `ŷ_t`:

1. run constrained search against `ŷ_t`
2. collect top traces `Π_wrong_t`
3. keep only traces that are close to GT under a simple structural filter

Define a structural closeness score:

```text
R(π; GT, ŷ_t)
= α_len · sim_len(Y(π), GT)
+ α_pos · pos_match(Y(π), GT)
+ α_role · role_overlap(π, GT-family prototypes)
```

For the minimal version:

- set `α_len = 1`
- set `α_pos = 1`
- drop `α_role`

So:

```text
R(π; GT, ŷ_t) = sim_len(Y(π), GT) + pos_match(Y(π), GT)
```

Keep only top `M` rescue traces by `R`.

#### Variant A3.2 `local_repair_from_gt_traces`

Start from each `π ∈ Π_GT`, and derive local alternatives by perturbing:

- repeat count
- single word role

Then keep alternatives that:

- become compatible with one of the wrong outputs
- are structurally close to their GT parent trace

This variant is more work. Use only if A3.1 is too weak.

### Step A4. Merge and deduplicate support

Define:

```text
Π_goalB = unique(Π_GT ∪ Π_rescue)
```

Deduplication key should be trace-structural, not only output-level.

Recommended key:

```python
(tuple((step.word, step.role, step.repeat_k_or_none) for step in trace_steps), tuple(output))
```

### Step A5. Final posterior and commit

Add a new mode to [apply_correct_answer](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/correct_update.py:22>):

```python
mode="direct" | "history_reweight" | "route_a_support" | "route_b"
```

For `route_a_support`:

```text
q_A(π) ∝ exp(s(π)) · 1[π ∈ Π_goalB]
```

Optionally later:

```text
q_A(π) ∝ exp(s(π)) · 1[π ∈ Π_goalB] · C_A(π; D^-)
```

But do not start with both support intervention and extra weighting unless needed.

Then commit with the existing weighted M-step:

```text
Δθ_A ∝ η_correct · Σ_π q_A(π) · stats(π)
```

## 4.4 Why this route could beat direct-correct

Direct-correct only sees `Π_GT`.
Route A-support can win only if:

1. `Π_rescue(D^-)` introduces structurally informative alternatives
2. those alternatives help disambiguate GT-compatible families
3. the resulting M-step improves grammar in a way that transfers beyond the current query

If support changes but the posterior still collapses back onto the same top GT trace family, this route has failed.

## 4.5 Files to modify

Modify:

- [goal_b_state.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/goal_b_state.py:30>)
- [correct_update.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/correct_update.py:22>)
- `cls_color_selection/cls_color_selection/learner/goal_b_support.py` new
- current Goal B runner, ideally moved out of `tmp/`

Do not modify yet:

- [feedback_update.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/feedback_update.py:22>)
- `QueryMemory`
- tutor interaction logic

## 4.6 Route A-support experiments

### Experiment A1. Support-only vs direct-correct

Conditions:

- `correct_only`
- `hist1_support_correct`
- `hist2_support_correct`
- `hist3_support_correct`

Acceptance:

- at least one `histk_support_correct` has `gt_mass > correct_only + 0.003`
- no meaningful drop in `probe_acc`

### Experiment A2. Support-only vs history-reweight

Conditions:

- `histk_reweight_correct`
- `histk_support_correct`

Acceptance:

- support route beats reweight route on `gt_mass`
- `KL(q_routeA || q_dir)` is meaningfully larger than reweight-only route

### Experiment A3. Support diagnostics

Log:

- support size
- support overlap with `Π_GT`
- new-trace fraction
- ESS before and after merge
- top-trace family change rate

Acceptance:

- route actually changes support on at least 30% of episodes
- top family differs from direct-correct on a non-trivial subset

---

## 5. Route B

## 5.1 Goal

Preserve the small narrowing benefit that real wrong updates already seem to provide, but reduce harmful drift.

This route combines:

1. weak long-term wrong writes
2. query-local wrong history
3. final correct commit using the accumulated history

## 5.2 Core idea

Let wrong updates modify long-term grammar, but with a reduced learning rate:

```text
θ_t = θ_{t-1} + η_wrong_long · Δ_wrong_t
```

where:

```text
0 < η_wrong_long << η_fb
```

Then do a final correct step on the updated state:

```text
q_B(π)
∝ exp(s(π; θ_k)) · 1[π ∈ Π_goalB(GT, D^-)]
```

Commit:

```text
Δθ_correct,B ∝ η_correct · Σ_π q_B(π) · stats(π)
```

This route is trying to keep enough wrong-side narrowing to help, while avoiding the full anti-synergy of current wrong writes.

## 5.3 Minimal implementation

### Step B1. Add scaled wrong write

Modify [FeedbackUpdater](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/feedback_update.py:22>) to support an override:

```python
def apply_feedback(..., eta_scale: float = 1.0):
    ...
```

and in `differential_m_step(...)` use:

```text
η_eff = eta_fb · eta_scale
```

This is the only wrong-side hyperparameter needed at first.

### Step B2. Keep local history too

While applying the weak long-term write, also record each wrong in `GoalBProtocolState`.

That gives:

- long-term small shift
- local structural memory for final commit

### Step B3. Final correct mode

Add `mode="route_b"` in `apply_correct_answer(...)`.

For the first Route B version:

```text
Π_goalB = Π_GT
```

and:

```text
q_B(π) ∝ exp(s(π; θ_k)) · 1[π ∈ Π_GT]
```

This isolates the effect of weak wrong-side state change.

If this helps, then combine with Route A-support:

```text
Π_goalB = Π_GT ∪ Π_rescue(D^-)
```

## 5.4 Hyperparameters

Keep this tiny.

Use:

- `eta_wrong_scale ∈ {0.05, 0.1, 0.2}`
- `eta_correct = 1.0`
- no additional history weight for the first Route B run

Do not sweep more than this until one setting shows promise.

## 5.5 Files to modify

Modify:

- [feedback_update.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/feedback_update.py:22>)
- [goal_b_state.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/goal_b_state.py:30>)
- [correct_update.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/correct_update.py:22>)
- formal Goal B runner

Do not modify yet:

- support-time learner training
- online episode manager
- assist / hint pipeline

## 5.6 Route B experiments

### Experiment B1. Weak-wrong vs current wrong

Conditions:

- `wrong1_correct`
- `wrong1_weak0.05_correct`
- `wrong1_weak0.1_correct`
- `wrong1_weak0.2_correct`

Acceptance:

- at least one weak setting has higher final `gt_mass` than current wrong baseline
- marginal correct gain no longer shrinks as sharply

### Experiment B2. Matched-budget comparison

Conditions:

- `wrongk_weakη + correct`
- `correct × (k+1)`

Acceptance:

- gap to `correct × (k+1)` shrinks materially
- target threshold: absolute `gt_mass` gap reduced by at least 50% vs current baseline

### Experiment B3. Route B + support

Conditions:

- `wrongk_weakη + correct_routeB`
- `wrongk_weakη + support_correct`

Acceptance:

- at least one combined condition beats both pure weak-wrong and pure support-only route

---

## 6. Diagnostics Required for Both Routes

Extend current metrics logging with:

1. `corr_score_logC`
   - correlation between raw trace score and history factor
2. `ess_posterior`
   - posterior effective sample size
3. `kl_to_direct`
   - `KL(q_goalB || q_dir)`
4. `support_size`
5. `support_new_fraction`
6. `top_family_changed`
   - whether the dominant trace family differs from direct-correct
7. `correct_gain_gt_mass`
8. `correct_gain_probe_acc`

Recommended location:

- eventually move `goal_b_metrics.py` out of `tutor_api`
- but this migration is not a blocker for implementation

---

## 7. Tests

Add or extend these tests.

### 7.1 `tests/test_goal_b_support.py`

Test cases:

1. `build_goal_b_support(...)` returns non-empty GT support
2. merged support contains deduplicated traces only
3. support route can add new traces beyond pure `Π_GT`
4. support builder is deterministic under fixed seed

### 7.2 `tests/test_correct_update.py`

Add cases:

1. `mode="direct"` reproduces current behavior
2. `mode="route_a_support"` changes posterior support when rescue traces exist
3. `mode="route_b"` works when weak wrong updates were applied first
4. if rescue support is empty, route falls back cleanly to direct mode

### 7.3 `tests/test_feedback_update.py`

Add cases:

1. `eta_scale < 1` reduces update magnitude monotonically
2. `eta_scale=0` leaves grammar unchanged
3. scaled wrong update preserves sign of the original differential update

### 7.4 `tests/test_goal_b_metrics.py`

Add cases:

1. ESS calculation is stable
2. support-overlap logging handles empty rescue support
3. KL-to-direct is zero when posterior matches direct-correct

---

## 8. Experiment Matrix

Run in this order.

### Phase 1. Route A-support minimal

Conditions:

- `correct_only`
- `hist1_support_correct`
- `hist2_support_correct`
- `hist3_support_correct`

Stop rule:

- if support never changes or no metric moves at all, do not tune weights yet

### Phase 2. Route B minimal

Conditions:

- `wrong1_weak0.05_correct`
- `wrong1_weak0.1_correct`
- `wrong1_weak0.2_correct`
- `wrong2_weak0.1_correct`

Stop rule:

- if even weak wrong does not reduce anti-synergy, do not add replay yet

### Phase 3. Combined route

Conditions:

- `wrong1_weak0.1_support_correct`
- `wrong2_weak0.1_support_correct`

This is the first route with a real chance of positive Goal B.

### Phase 4. Only if needed: replay

Only add replay if:

- query-local grammar metrics improve
- but `probe_acc` / `probe_ll` remain flat

Minimal replay:

```text
Δθ_replay ∝ η_replay · Σ_{(x,GT)∈B_recent} Σ_π q_goalB(π|x,GT,D^-) · stats(π)
```

with:

- `η_replay = 0.25`
- `B_recent` tiny, such as the last 2 to 4 successful correct-commit episodes

Replay is not part of the initial Route A-support or Route B implementation.

---

## 9. Redundancy Control

To avoid mechanism sprawl:

1. Do not introduce both support change and history weighting in the first Route A-support run.
2. Do not introduce more than one new wrong-side hyperparameter.
3. Do not add near-GT soft targets before exact GT routes are understood.
4. Do not move Goal B state into `QueryMemory` yet.
5. Do not add replay until there is evidence of query-local benefit.

Recommended initial configuration:

```text
Route A-support:
  support_mode = "gt_plus_repair"
  max_gt_traces = 16
  max_rescue_traces = 8
  no extra history weighting

Route B:
  eta_wrong_scale ∈ {0.05, 0.1, 0.2}
  route_b final step uses exact GT support first
```

---

## 10. Interpretation Rules

### If Route A-support wins locally but probe is flat

Interpretation:

- wrong history is adding useful query-local structure
- but the final commit is not yet producing transferable grammar change

Next step:

- minimal replay

### If Route B shrinks the anti-synergy gap but does not surpass direct-correct

Interpretation:

- weak wrong-side state change is less harmful than the current wrong update
- but local history still is not adding enough extra structure

Next step:

- combine Route B with support intervention

### If neither route changes support or posterior family meaningfully

Interpretation:

- current grammar/search family may not contain enough structural diversity to support Goal B in this environment

Next step:

- stop adding protocol complexity
- reconsider whether this task family can express the target phenomenon

This is the clean failure condition for the current scene.

---

## 11. Final Recommendation

Implement in this order:

1. Route A-support minimal
2. Route B minimal
3. Route B + support
4. Replay only if generalization still lags

The most likely win condition is not:

```text
wrong history reweights the same GT traces better
```

It is:

```text
wrong history changes which explanations the final correct step can see,
and weak wrong-side learning preserves enough narrowing to make that useful.
```

If the codebase cannot produce that effect even after these routes, then this particular scene probably does not support Goal B in a robust way.
