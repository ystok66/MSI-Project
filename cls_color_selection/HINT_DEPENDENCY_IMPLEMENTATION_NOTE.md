# Hint-Induced Dependency / Guessing Implementation Note

## 0. Goal

This note focuses on one specific design question:

> Can tutor hints make the learner behave more like “I infer the answer from the tutor's clue”
> rather than “I learn the grammar by correcting my own mistakes”?

And further:

> Can bad hints then harm the learner's longer-term competence?

The answer from current code inspection is:

- **Yes, this is implementable**
- **But it is not the current semantics of the code**

Right now, hints mostly act as task-completion assistance, not as a strong learner-internal information source.

---

## 1. What the current code already does

### 1.1 Hint changes completion, not learner inference

Current hint application:

- [action_generators.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/action_generators.py:101>)

Current behavior:

1. Hint writes correct colors directly into `state.completion`
2. Hint marks those positions in `state.assist_mask`
3. Learner then continues using the same `target_output`

This means the learner sees:

- “some slots are already filled”

but not:

- “I should reinterpret the query because tutor has provided semantic evidence”

### 1.2 Target prediction is still hint-blind

Target prediction:

- [target_predictor.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/target_predictor.py:14>)
- [cls_wrapper.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/cls_wrapper.py:57>)

`predict_target(words)` depends only on:

- query words
- current grammar state

It does **not** depend on:

- `completion`
- `assist_mask`
- tutor hint payload

So hint does not currently induce a “guess-from-tutor” inference mode.

### 1.3 Policy reacts only through remaining gaps

Selection policy:

- [policy.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/policy.py:31>)

The policy uses:

- `state.needed_colors()`
- `state.color_gaps()`

So after hint, the learner reacts behaviorally only because:

- fewer colors remain to be filled

That is useful for task completion, but still not a real hint-conditioned inference model.

### 1.4 Hint is weakly isolated from grammar learning

In Phase 2 / 3:

- hint is applied before grammar feedback update
- but the grammar feedback still uses the failed confirm submission from before the hint
- and Phase 2 / 3 do not pass `assist_mask` into `apply_feedback(...)`

See:

- [run_phase2.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/experiments/run_phase2.py:209>)
- [run_phase2.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/experiments/run_phase2.py:218>)
- [run_phase3.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/experiments/run_phase3.py:195>)

In Phase 4:

- `assist_mask` is finally passed to grammar update
- but only as a **discount**

See:

- [run_phase4.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/experiments/run_phase4.py:234>)
- [run_phase4.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/experiments/run_phase4.py:245>)

So current code explicitly treats hint as:

- assistance
- not clean learner-owned evidence

---

## 2. Which ideas are realistically implementable

Below is the feasibility ranking based on the current codebase.

### A. Hint-as-query-local bias

**Feasibility: high**

Goal:

- after hint, learner re-scores candidate traces / targets using hint compatibility
- learner enters a “guess the rest from tutor clue” mode

Why feasible:

- target prediction is already beam-based
- `beam_posterior(words)` is available
- `assist_mask` and hinted positions are already available in state

### B. Hint-induced dependence via query-level learning discount

**Feasibility: high**

Goal:

- once a query is assisted, its feedback-derived learning counts less
- assisted success is no longer equivalent to autonomous success

Why feasible:

- `FeedbackUpdater` already has a clean learning-rate parameter path
- Goal B work already introduced the idea of `eta_scale`

### C. Hint-as-persistent prior / trust state

**Feasibility: medium**

Goal:

- hints change not only the current query but also future search bias
- bad hints can accumulate a cross-query dependency effect

Why only medium:

- no existing learner-side persistent hint state
- needs a new state object or extension of learner wrapper / predictor

### D. Hint-as-direct pseudo-supervision

**Feasibility: high**

Goal:

- hints directly update grammar as pseudo-labels

Why not recommended first:

- too strong
- easy to produce obvious poisoning
- less natural than bias/dependence mechanisms

---

## 3. Recommended implementation order

Recommended order:

1. **Hint-aware target / beam bias**
2. **Query-level assisted-learning discount**
3. **Persistent hint reliance state**
4. Direct pseudo-supervision only as a stress test

This order is deliberate:

- first make hint change how the learner thinks right now
- then make hint change how much the learner learns from the episode
- only then consider cross-query persistence

---

## 4. Concrete design: Hint-aware target / beam bias

## 4.1 Intended semantics

After hint, the learner should not merely continue filling gaps.
It should behave as if:

> “Tutor has revealed a partial clue about the target. I now infer the rest under that clue.”

This creates the “guessing” effect you want.

## 4.2 Formalization

Let:

- `x` = query words
- `h = {(i, c_i)}` = hint payload at positions
- `π` = latent trace
- `Y(π)` = rendered output of trace
- `s(π; x, θ)` = original trace score

Define a hint compatibility term:

```text
Compat(π; h) =
  Σ_{(i,c_i) ∈ h} 1[Y_i(π) = c_i]
```

Then define a hint-aware posterior:

```text
q_hint(π | x, h)
∝ exp(s(π; x, θ) + β_hint · Compat(π; h))
```

For a hard version:

```text
Compat_hard(π; h) =
  1, if trace output matches all hinted positions
  0, otherwise
```

Then:

```text
q_hint_hard(π | x, h)
∝ exp(s(π; x, θ)) · Compat_hard(π; h)
```

## 4.3 Minimal implementation

Add to:

- [target_predictor.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/target_predictor.py:14>)

New methods:

```python
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
```

Implementation pattern:

1. call `beam_posterior(words)`
2. compute compatibility for each `(score, trace, Y_k)`
3. re-rank candidates
4. return the top candidate's `Y_k` as new target

## 4.4 Where to apply it

After hint is written into state, immediately recompute target:

- [run_phase2.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/experiments/run_phase2.py:137>)
- [run_phase3.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/experiments/run_phase3.py:136>)
- [run_phase4.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/experiments/run_phase4.py:134>)

Pseudo-flow:

```python
state = apply_hint_to_state(state, hint_action)
new_target = target_pred.predict_target_with_hint(
    state.query_words,
    hint_action.hint_positions,
    beta_hint=cfg.learner.beta_hint,
    mode=cfg.learner.hint_infer_mode,
)
if new_target != list(state.target_output):
    state.target_output = new_target
    state.completion = _merge_completion_with_hint(
        old_completion=state.completion,
        old_assist_mask=state.assist_mask,
        new_target=new_target,
    )
```

Important:

- do **not** blindly reset completion to all empty
- preserve tutor-filled positions where possible

So add a helper:

```python
def _merge_completion_with_hint(...):
    ...
```

## 4.5 Expected effect

Good hint:

- faster convergence
- fewer confirms
- target flips toward GT

Bad hint:

- target flips toward wrong basins
- learner produces more coherent wrong answers
- wrong-confirm updates may now be more systematically biased

This is the first mechanism that can make hint quality matter in a learner-facing way.

---

## 5. Concrete design: Query-level assisted-learning discount

## 5.1 Intended semantics

If a query is solved with tutor help, the learner should not get full “I learned this myself” credit.

This creates dependency pressure naturally:

- tutor can boost short-term success
- but excessive assistance can weaken autonomous learning

## 5.2 Formalization

Let:

- `n_hint` = number of hint events in the query
- `a_hint` = assisted fraction of positions

Define an effective feedback learning rate:

```text
η_fb_eff = η_fb · ρ_query_hint
```

Minimal form:

```text
ρ_query_hint =
  1.0, if no hint occurred
  ρ_assisted_query, otherwise
```

Slightly richer form:

```text
ρ_query_hint = exp(-λ_hint · n_hint)
```

or

```text
ρ_query_hint = 1 / (1 + λ_hint · a_hint)
```

## 5.3 Minimal implementation

Modify:

- [feedback_update.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/feedback_update.py:22>)

Add:

```python
def apply_feedback(..., eta_scale: float = 1.0):
    ...
```

and in `differential_m_step(...)`:

```text
η_eff = eta_fb · eta_scale
```

Then in query loops:

- if `diag['had_hint']` or `state.assist_mask` non-empty, reduce `eta_scale`

Minimal rule:

```python
eta_scale = 1.0 if not had_hint else cfg.learner.hint_eta_scale
```

New learner config fields:

```python
hint_eta_scale: float = 0.5
beta_hint: float = 2.0
hint_infer_mode: str = "hard"
```

## 5.4 Expected effect

Good hint:

- may still improve current query success
- but contributes less to long-term grammar growth

Bad hint:

- can still distort current target inference
- and because autonomous correction is reduced, later recovery can become harder

This is a much more direct route to “hint can affect learning ability”.

---

## 6. Concrete design: Persistent hint reliance / trust prior

## 6.1 Intended semantics

The learner should gradually change how much it trusts tutor-provided information.

This makes bad hints capable of long-term harm even when they are not directly written into grammar.

## 6.2 Recommended state

Do **not** put this into `QueryMemory`.

Current `QueryMemory`:

- [memory.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/memory.py:1>)

is query-local and reset each query.

Instead add a persistent learner-side state, e.g.:

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

## 6.3 How it affects inference

Use:

```text
β_hint_eff = β_hint_base · trust
```

Then:

```text
q_hint(π | x, h)
∝ exp(s(π; x, θ) + β_hint_eff · Compat(π; h))
```

As trust increases:

- learner relies more strongly on hints

As trust decreases:

- learner behaves more independently

## 6.4 How to update trust

Minimal update signal:

- if hinted query later succeeds with fewer confirms, trust slightly up
- if hinted query still fails / times out, trust slightly down

Simple rule:

```text
trust_{t+1} = clip(trust_t + α_trust · reward_hint, 0, 1)
```

where:

```text
reward_hint =
  +1, if hinted query succeeds quickly
  -1, if hinted query still fails or times out
```

This is intentionally coarse for the first version.

## 6.5 Expected effect

Good hints:

- accelerate hint use and query success

Bad hints:

- if trust is slow to correct, they can create a persistent over-reliance regime

This gives you the long-term dependency phenomenon in a more natural way than direct poisoning.

---

## 7. What should NOT be the first implementation

### 7.1 Direct grammar poisoning by hint

Possible:

- use hint as pseudo-supervision and update role/emission/repeat stats directly

Why not first:

- too strong
- too easy to make “bad hint obviously harmful”
- less behaviorally natural

Use only as an upper bound / stress test.

### 7.2 Overloading `assist_mask`

`assist_mask` is currently a positional provenance marker.

Do not overload it into:

- trust
- hint quality
- persistence
- learner attitude

Create new state instead.

---

## 8. Existing modules that are useful immediately

### Behavioral tutor can already produce imperfect hints

- [tutor_behavioral.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/tutor_behavioral.py:216>)

`BehavioralTutor` uses its own grammar prediction, so its hints may already be wrong.

This is ideal for studying bad-hint effects without inventing synthetic noise.

### Counterfactual scaffold already exists

- [counterfactual.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/counterfactual.py:39>)

This module already compares hint vs wait branches.

It currently models hint impact through feedback-update differences.

It can be extended to:

- use hint-aware target prediction
- include query-level learning discount
- estimate autonomy loss more explicitly

### Hint policy already has learning-loss language

- [hint_policy.py](<F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/hint_policy.py:237>)

`compute_learning_loss(...)` already encodes the idea that hinting can short-circuit learning.

That makes it a natural place to connect:

- hint selection
- downstream learner-side cost

---

## 9. Minimal experiment matrix

Run these four first.

### E1. Current baseline

Condition:

- current hint implementation

### E2. Hint-aware inference only

Condition:

- hint-aware target prediction on
- no query-level discount

Question:

- does hint now change learner proposals, not just completion?

### E3. Hint-aware inference + query discount

Condition:

- hint-aware target prediction on
- `hint_eta_scale < 1`

Question:

- does tutor assistance now create a short-term success / long-term autonomy tradeoff?

### E4. Persistent trust

Condition:

- E3 plus persistent trust state

Question:

- do bad hints now create lasting dependency or cross-query bias?

---

## 10. Metrics to log

In addition to current teaching/eval metrics, add:

1. `post_hint_target_flip`
   - whether target prediction changed after hint
2. `post_hint_target_matches_gt`
   - whether hint-conditioned target moved toward or away from GT
3. `hint_conditioned_beam_shift`
   - posterior shift before vs after hint conditioning
4. `assisted_query_eta_scale`
5. `autonomy_gap`
   - performance with tutor on vs tutor removed
6. `hint_reliance`
   - persistent trust value if enabled
7. `assisted_success_rate`
8. `unassisted_success_rate`

These metrics are necessary to show “hint causes guessing / dependence” rather than just “hint helps”.

---

## 11. Final recommendation

The most realistic and code-compatible path is:

1. **Hint-aware target / beam bias**
2. **Query-level learning discount**
3. **Persistent trust / reliance state**

This combination can naturally produce:

- good hints that help
- bad hints that bias search
- assistance that weakens autonomous learning

That is the cleanest route to the phenomenon you want:

> tutor hint changes not only what the learner does, but how the learner thinks and how much the learner truly learns.
