# Hint Modeling Error / Over-Help Guide

## 0. Goal

This guide is for the following research question:

> If the tutor cannot model the learner well, can tutor hints become problematic?
> If hints are given too often, can they reduce learner autonomy or longer-term ability?

This document assumes:

- we **do not** prioritize query-level learning discount as the main mechanism
- we **do** want hint to affect learner inference and autonomy
- we want experiments that can distinguish:
  - bad hint content
  - bad hint timing / gating
  - over-help / dependency

---

## 1. What current code already supports

Current code already gives us three strong footholds.

### 1.1 BehavioralTutor can already generate imperfect hints

File:

- [tutor_behavioral.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/tutor_behavioral.py:115>)

Key property:

- `BehavioralTutor` uses its **own** grammar prediction for hint content
- so its hints may already be wrong if its own understanding is wrong

Most useful method:

- [on_confirm_fail](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/tutor_behavioral.py:216>)

This is the cleanest existing source of naturally bad hints.

### 1.2 InverseTutor already supports “good content, bad modeling”

File:

- [tutor_inverse.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/tutor_inverse.py:64>)

Key property:

- hint content comes from `TutorTaskModel` and is therefore correct
- but the decision to hint is gated by the tutor's learner model

This gives exactly the condition:

> tutor content is correct, but tutor modeling of learner may be poor

Also important:

- `registry_phase4.py` already has degraded learner-model init modes:
  - `cold`
  - `teacher_prior`

Files:

- [registry_phase4.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/experiments/registry_phase4.py:105>)
- [run_phase4.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/experiments/run_phase4.py:455>)

### 1.3 hint_policy already encodes over-help logic

File:

- [hint_policy.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/hint_policy.py:66>)

Useful pieces:

- coarse hint utility:
  - [compute_coarse_hint_utility](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/hint_policy.py:66>)
- position selection:
  - [select_hint_positions_greedy](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/hint_policy.py:163>)
- learning-loss proxy:
  - [compute_learning_loss](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/hint_policy.py:237>)
- full decision:
  - [decide_hint](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/hint_policy.py:271>)

This means we do **not** need to invent tutor-side over-help machinery from scratch.

---

## 2. Main design choice

We want bad hints / too many hints to matter because they change:

- how the learner infers the answer
- how much the learner relies on tutor help

We do **not** want to start by making hint a strong pseudo-label that directly rewrites grammar.

Therefore the recommended route is:

1. **Hint-aware target prediction**
2. **Hint-induced autonomy mode**
3. **Persistent trust / reliance state**

Do not start with direct grammar poisoning.

---

## 3. Step 1: Hint-aware target prediction

## 3.1 Why this is the right first step

Current hint only changes:

- `completion`
- `assist_mask`

See:

- [action_generators.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/action_generators.py:101>)

But target prediction remains hint-blind:

- [target_predictor.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/target_predictor.py:14>)
- [cls_wrapper.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/cls_wrapper.py:57>)

So the learner does **not** currently enter a tutor-conditioned guessing mode.

## 3.2 New semantics

After hint, the learner should behave like:

> “Tutor has revealed partial evidence about the target. I now infer the remaining target under that evidence.”

## 3.3 Recommended implementation

Modify:

- [target_predictor.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/target_predictor.py:14>)

Add:

```python
def beam_posterior_with_hint(
    self,
    words: List[str],
    hint_positions: List[Tuple[int, str]],
    *,
    beta_hint: float = 2.0,
    mode: str = "hard",
) -> list:
    ...

def predict_target_with_hint(
    self,
    words: List[str],
    hint_positions: List[Tuple[int, str]],
    *,
    beta_hint: float = 2.0,
    mode: str = "hard",
) -> List[str]:
    ...
```

## 3.4 Suggested formula

Let `h = {(i, c_i)}` be hint payload.

Hard mode:

```text
q_hint(π | x, h)
∝ exp(score(π; x, θ)) · 1[Y(π) matches all hinted positions]
```

Soft mode:

```text
Compat(π; h) = Σ_{(i,c_i)∈h} 1[Y_i(π)=c_i]
q_hint(π | x, h)
∝ exp(score(π; x, θ) + β_hint · Compat(π; h))
```

Start with:

- `mode = "hard"`
- `beta_hint = 2.0` only if soft mode is tested later

## 3.5 Where to call it

After `apply_hint_to_state(...)`, immediately recompute target.

Phase 2:

- [run_phase2.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/experiments/run_phase2.py:137>)

Phase 3:

- [run_phase3.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/experiments/run_phase3.py:136>)

Phase 4:

- [run_phase4.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/experiments/run_phase4.py:134>)

Pseudo-flow:

```python
state = apply_hint_to_state(state, hint_action)
new_target = target_pred.predict_target_with_hint(
    state.query_words,
    hint_action.hint_positions,
    mode=cfg.learner.hint_infer_mode,
    beta_hint=cfg.learner.beta_hint,
)
if new_target != list(state.target_output):
    state.target_output = new_target
    state.completion = merge_completion_after_target_flip(
        completion=state.completion,
        assist_mask=state.assist_mask,
        new_target=new_target,
    )
```

Add helper function:

```python
def merge_completion_after_target_flip(...):
    ...
```

Important:

- do not blindly zero out completion
- preserve tutor-filled positions if they are still aligned
- if a hinted position becomes inconsistent with the new target, keep the hinted visible state but log the inconsistency

## 3.6 New config

Add to `LearnerConfig`:

```python
hint_infer_mode: str = "hard"     # "hard" | "soft"
beta_hint: float = 2.0
enable_hint_bias: bool = False
```

The new logic should be off by default.

---

## 4. Step 2: Hint-induced autonomy mode

## 4.1 Why this is needed

Hint-aware target prediction changes what the learner thinks.
But to show over-help harms autonomy, we also need hint to change how the learner behaves.

Current selection/confirm policy is here:

- [policy.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/policy.py:31>)

Current state container is here:

- [state.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/state.py:17>)

## 4.2 New semantics

Once the learner has been hinted, it should become more likely to:

- confirm earlier
- explore less
- retry less
- rely on the hinted target rather than continuing broad search

This is the dependence mechanism.

## 4.3 State changes

Modify:

- [state.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/state.py:17>)

Add fields to `QueryState`:

```python
hinted_this_query: bool = False
hint_count: int = 0
hint_payload: List[Tuple[int, str]] = field(default_factory=list)
hint_target_flipped: bool = False
```

Update [apply_hint_to_state](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/action_generators.py:101>) to populate:

```python
state.hinted_this_query = True
state.hint_count += 1
state.hint_payload.extend(hint_action.hint_positions or [])
```

## 4.4 Policy changes

Modify:

- [policy.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/policy.py:31>)

### Change 1: earlier confirm after hint

Current:

```python
if state.is_complete: return True
if state.fill_ratio >= self.cfg.confirm_fill_threshold: return True
```

Replace with effective threshold:

```text
τ_confirm_eff =
  τ_confirm_base - γ_confirm_hint · 1[hinted_this_query]
```

Minimal implementation:

```python
threshold = self.cfg.confirm_fill_threshold
if getattr(state, "hinted_this_query", False):
    threshold = max(0.0, threshold - self.cfg.hint_confirm_bonus)
```

Recommended config:

```python
hint_confirm_bonus: float = 0.25
enable_hint_autonomy_shift: bool = False
```

### Change 2: lower exploration after hint

Current policy has epsilon exploration and greedy utility over candidate balls.

Add a hinted-mode scaling:

```text
ε_eff = ε_base · (1 - γ_eps_hint)
```

and optionally a stricter stopping rule:

```text
util_stop_threshold_eff = util_stop_threshold + γ_stop_hint
```

Minimal implementation:

- if hinted, reduce epsilon exploration
- if hinted, stop set expansion a little earlier

Pseudo-change inside `select_set(...)`:

```python
eps = self.cfg.epsilon_policy
if hinted:
    eps = eps * (1.0 - self.cfg.hint_exploration_drop)
```

and:

```python
if best_idx < 0 or best_util < (-2.0 + stop_shift):
    break
```

Recommended config:

```python
hint_exploration_drop: float = 0.8
hint_stop_shift: float = 0.5
```

## 4.5 Expected effect

Good hints:

- better teaching success
- fewer retries
- fewer confirms

Bad hints:

- learner commits earlier to the wrong hypothesis
- learner gathers less disconfirming evidence
- autonomy drops when tutor is removed

---

## 5. Optional Step 3: Persistent trust / reliance

This is not required for the first implementation, but it is the correct third step if A/B already show signal.

## 5.1 Purpose

To let over-help accumulate across queries rather than only inside one query.

## 5.2 New module

Create:

- `cls_color_selection/cls_color_selection/learner/hint_reliance.py`

Suggested state:

```python
@dataclass
class HintRelianceState:
    trust: float = 0.5
    n_hints_seen: int = 0
    n_hints_helpful: int = 0
    n_hints_harmful: int = 0
```

## 5.3 Use

Make:

```text
β_hint_eff = β_hint_base · trust
```

and optionally:

```text
hint_confirm_bonus_eff = hint_confirm_bonus · trust
```

This makes learner reliance adaptive.

Do not implement this before Step 1 and Step 2 are working.

---

## 6. Tutor-side modeling-error directions

There are two tutor-failure modes that current code can already express.

## 6.1 Correct hint content, bad learner modeling

Use `InverseTutor`.

Files:

- [tutor_inverse.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/tutor_inverse.py:64>)
- [registry_phase4.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/experiments/registry_phase4.py:105>)

Mechanism:

- hint content is correct (`TutorTaskModel`)
- but hint timing / amount depends on learner model quality

Use these init modes:

- oracle support init
- cold start
- teacher prior

This answers:

> if tutor models learner badly, does it over-hint / under-hint / hint at the wrong times?

## 6.2 Bad hint content from tutor's own imperfect task understanding

Use `BehavioralTutor`.

File:

- [tutor_behavioral.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/tutor_behavioral.py:216>)

Mechanism:

- tutor predicts hints from its own grammar
- hint content can be wrong

This answers:

> if hint content is wrong, does hint-aware learner inference amplify the damage?

## 6.3 Recommended comparison

Compare:

1. `oracle_hint_source`
   - correct hint content
2. `behavioral_hint_source`
   - hint content may be wrong
3. `inverse_bad_model_gate`
   - content correct, but gating model degraded

This separates:

- content error
- gating / timing error

---

## 7. Experiment plan

## A. Learner reaction model

### A1. `current_hint`

Current system.

### A2. `hint_bias_only`

Changes:

- Step 1 on
- Step 2 off

Goal:

- verify that hints now change target inference

### A3. `hint_bias + autonomy_shift`

Changes:

- Step 1 on
- Step 2 on

Goal:

- verify that over-help can now reduce autonomous behavior

### A4. `hint_bias + autonomy_shift + trust`

Changes:

- Step 1 on
- Step 2 on
- Step 3 on

Goal:

- verify cross-query dependence / accumulated over-help effect

### Metrics for A

Log:

- `TeachSuccessRate`
- `TeachTimeoutRate`
- `EvalSuccessRate`
- grammar metrics if available:
  - `gt_mass`
  - `probe_acc`
  - `probe_ll`
- `post_hint_target_flip_rate`
- `post_hint_target_to_gt_rate`
- `autonomy_gap`
  - performance with tutor on vs tutor removed

Add new diagnostics:

- `n_hint_conditioned_flips`
- `n_hint_conditioned_wrong_flips`
- `mean_hint_count_per_query`
- `mean_confirm_after_hint`
- `mean_retry_after_hint`

## B. Tutor modeling error

### B1. `oracle_hint_source`

Tutor content correct.

### B2. `behavioral_hint_source`

Tutor content may be wrong.

### B3. `inverse_bad_model_gate`

Use:

- `cold`
- `teacher_prior`
- optional partial observation mode later

### Metrics for B

Log:

- `bad_hint_rate`
  - fraction of hints whose payload disagrees with GT
- `over_hint_rate`
  - hints given when `current_hint` / oracle would not hint
- `under_hint_rate`
  - no hint when oracle would hint
- `autonomy_gap`
- `EvalSuccessRate`
- grammar probes if available

Acceptance:

- degraded tutor models should create measurable differences in hint quality or hint frequency
- these differences should propagate into learner autonomy metrics once Step 1 and Step 2 are active

## C. Over-hint experiment

Only meaningful after Step 1 and Step 2 exist.

Conditions:

- `never`
- `current`
- `early`
- `always_hint_after_fail`

Goal:

- determine whether too much tutor help reduces learner autonomy or later performance

Metrics:

- `TeachSuccessRate`
- `TeachTimeoutRate`
- `EvalSuccessRate`
- `autonomy_gap`
- `post_hint_target_flip_rate`
- if Step 3 exists:
  - `trust`

Interpretation:

- if teach improves monotonically but autonomy / eval declines, that is the over-help signature

---

## 8. Minimal file-change plan

### Required now

Modify:

- [target_predictor.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/target_predictor.py:14>)
- [state.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/state.py:17>)
- [policy.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/policy.py:31>)
- [action_generators.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/action_generators.py:101>)
- hint-handling points in:
  - [run_phase2.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/experiments/run_phase2.py:137>)
  - [run_phase3.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/experiments/run_phase3.py:136>)
  - [run_phase4.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/experiments/run_phase4.py:134>)

### Useful but optional in first round

- `hint_reliance.py` new
- experiment registry additions in phase 4

### Do not prioritize

- direct hint pseudo-supervision into grammar
- further `rho_assist` tuning
- query-level learning discount as the main mechanism

---

## 9. Recommended implementation order

1. Implement `hint_bias_only`
2. Implement `hint_bias + autonomy_shift`
3. Run learner-reaction experiment set A
4. Reuse existing `BehavioralTutor` and `InverseTutor` for experiment set B
5. Only then add persistent trust for set C

This order gives the cleanest causal story:

- hint changes inference
- then hint changes autonomy
- then tutor error and over-help effects become measurable

---

## 10. Final recommendation

For the question you care about, the most promising direction is:

> **Do not make hint stronger as supervision. Make hint stronger as a learner-facing inference cue and autonomy shaper.**

That is the most natural way to show:

- if tutor models learner badly, hint decisions can become problematic
- if tutor hints too often, learner may rely on tutor and lose autonomy
- the damage appears not only in current query outcomes, but in how the learner infers and behaves
