# Current System Mechanism Report

This report summarizes the current `cls_color_selection` system from code.

It covers:

- task and environment setup
- learner architecture
- tutor hierarchy
- grammar update paths
- danger / risk modeling
- hint mechanisms
- current Goal B and hint-modeling findings

This document is intended as a code-grounded system note, not a paper-style claim sheet.

---

## 1. Task and Environment

### 1.1 Core task

The task is a color-sequence production problem with two coupled difficulties:

1. **Grammar difficulty**
   - infer the target output sequence from input words
   - target comes from a latent compositional grammar

2. **Risk difficulty**
   - choose balls from a candidate pool without selecting dangerous balls
   - safe / danger is only partially observable through noisy vectors

The main environment is implemented in:

- [grammar_task_env.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/grammar_task_env.py:1)
- [generator.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/generator.py:1)
- [transition.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/transition.py:1)
- [state.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/state.py:1)

### 1.2 Grammar task files

Each task file contains:

- `SUPPORT`
- `QUERY`
- `GRAMMAR`

Parsing is handled by:

- [parse_task_file()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/generator.py:159)

The parsed grammar has:

- noun rules: `word -> COLOR`
- compositional rules: `pattern -> template`

Rendering uses recursive rewriting:

- [render_with_grammar()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/generator.py:211)

### 1.3 Query generation

Queries can come from:

- original task text
- resampled text
- generated expressions from the parsed grammar
- hybrid mixtures

Generation logic is in:

- [query_generator.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/query_generator.py:1)

Difficulty tiers currently control:

- number of nouns
- number of operator applications
- maximum output length

Important current property:

- although `hard` allows outputs up to length 16, many actual grammar families are still structurally small
- this matters for Goal B because GT-compatible latent trace families are often not very rich

### 1.4 Candidate pool and danger generation

The environment generates a fresh candidate pool after each select / retry.

Pool generation:

- [generate_candidate_pool()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/generator.py:78)

Danger model generation:

- [generate_danger_model()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/generator.py:33)

Each ball has:

- a color from the grammar palette
- a latent danger type
- a clean `danger_vec`
- a noisy `observed_vec`

The learner only sees the noisy vector.

### 1.5 Query timeline

A query unfolds as:

1. learner selects balls from the candidate pool
2. tutor may warn
3. if no warning and danger selected, learner dies unless immortal mode is on
4. if safe, balls are auto-placed into the current target-aligned completion
5. learner may confirm
6. if confirm is wrong, learner receives feedback
7. tutor may hint after confirm failure
8. learner may retry until success / timeout / death

The main environment loop is implemented in:

- [run_query_loop()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/experiments/run_phase4.py:78)

Transition functions are in:

- [select_balls()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/transition.py:13)
- [auto_place()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/transition.py:57)
- [confirm()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/transition.py:79)
- [retry_refresh()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/transition.py:118)

---

## 2. Query State and Episode State

### 2.1 QueryState

Per-query mutable state is defined in:

- [QueryState](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/state.py:11)

Key fields:

- `query_words`
- `target_output`
- `ground_truth`
- `completion`
- `candidate_pool`
- `confirm_count`
- `retry_count`
- `outcome`

Important distinction:

- `ground_truth` is the real answer from the task grammar
- `target_output` is the learner's current inferred answer

The completion is aligned to `target_output`, not directly to ground truth.

This is a crucial design choice:

- grammar updates can change `target_output`
- when `target_output` changes, the query's working completion may reset
- therefore grammar learning changes not only future queries, but also the internal trajectory of the current query

### 2.2 Hint-related query state

Recent hint-modeling experiments added:

- `assist_mask`
- `hinted_this_query`
- `hint_count`
- `hint_payload`
- `hint_target_flipped`

These live in:

- [state.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/state.py:43)

These are extensions, not part of the original minimal Phase 1 design.

### 2.3 QueryMemory

Short-term memory is separate from `QueryState`:

- [QueryMemory](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/memory.py:11)

It stores query-local observations such as:

- warned sets
- safe placements
- death observations
- retry count

It does **not** persist across queries.

---

## 3. Learner Architecture

The learner has two semi-separable subsystems:

1. **grammar learner**
2. **risk learner**

and one policy layer that uses both.

### 3.1 Grammar learner: CLS wrapper

Grammar learning is wrapped by:

- [CLSSequencePredictor](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/cls_wrapper.py:25)

This wraps `BASIC/cls_learner` and exposes:

- `fit_support(examples)`
- `predict_target(words)`
- `beam_posterior(words)`

Important design point:

- the wrapper does not rewrite `CLSAgent` internals
- grammar knowledge lives in the underlying CLS cortex / concept library

### 3.2 Support learning

Support examples are studied once at episode start:

- [fit_support()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/cls_wrapper.py:35)

This calls `CLSAgent.study(...)` on support examples.

Current default learner config:

- `n_sup = 14`
- `n_em = 3`
- `use_hpc = False`
- `cls_mode = 'ast'`

Defined in:

- [LearnerConfig](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/config.py:24)

### 3.3 Target prediction

Target prediction is managed by:

- [TargetPredictor](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/target_predictor.py:16)

Responsibilities:

- cache `target_output`
- cache beam posteriors
- invalidate caches after grammar changes

Normal mode:

- `predict_target(words)` uses the CLS learner directly

Recent extension:

- `predict_target_with_hint(...)`
- `beam_posterior_with_hint(...)`

These let tutor hints bias the learner's **inference**, not just its completion.

### 3.4 Risk learner

Risk learning is handled by:

- [DangerTypeBelief](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/risk_belief.py:12)

This is a Bayesian multi-class classifier over:

- safe type `0`
- danger types `1..K`

Observation model:

- each ball has a noisy observed vector
- the learner computes `P(z=k | x)`

Core methods:

- `single_ball_posterior`
- `batch_posterior`
- `set_danger_probability`
- `update_from_death`
- `update_from_safe_observation`

This subsystem is independent from the grammar learner.

#### Core risk formulas

For each ball observation `x_i`, the learner models a latent type:

```text
z_i ∈ {0, 1, ..., K}
```

where:

- `0` = safe
- `1..K` = danger types

The posterior is:

```text
P(z_i = k | x_i) ∝ P(x_i | z_i = k) P(z_i = k)
```

with a diagonal Gaussian likelihood:

```text
P(x_i | z_i = k) = N(x_i ; μ_k, Σ_k + Σ_obs)
```

This is implemented in:

- [single_ball_posterior()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/risk_belief.py:56)
- [batch_posterior()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/risk_belief.py:88)

For a selected set of balls `S`, the learner estimates danger probability as:

```text
P(∃ danger in S) = 1 - ∏_{i ∈ S} P(z_i = 0 | x_i)
```

implemented in:

- [set_danger_probability()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/risk_belief.py:114)

### 3.5 Learner policy

Action selection is handled by:

- [ColorSelectionPolicy](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/policy.py:14)

The policy has three parts:

1. choose a subset of balls
2. decide whether to confirm
3. optionally trigger courage behavior

Selection utility currently combines:

- fill utility
- danger penalty
- waste penalty

Important config weights:

- `alpha_fill`
- `alpha_risk`
- `alpha_waste`
- `epsilon_policy`
- `confirm_fill_threshold`

Policy is therefore not a learned controller in the RL sense.
It is mostly a rule-based / utility-greedy action layer operating on current beliefs.

#### Core policy formulas

For each candidate ball `i` with color `c_i`, the policy computes a marginal utility:

```text
u_i = α_fill · g_fill(i) - α_risk · p_danger(i) - α_waste · g_waste(i)
```

where:

- `g_fill(i) = 1` if the color still fills a remaining gap, else `0`
- `p_danger(i) = 1 - P(z_i = 0 | x_i)`
- `g_waste(i) = 1` if the color is currently unnecessary, else `0`

This is implemented in:

- [select_set()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/policy.py:22)

Confirm is currently threshold-based:

```text
confirm if fill_ratio ≥ τ_confirm
```

with:

```text
fill_ratio = filled_count / |target_output|
```

implemented in:

- [should_confirm()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/policy.py:118)

---

## 4. Learner Update Mechanisms

The system currently has several distinct update channels.

### 4.1 Support learning

This is the initial grammar learning stage from support examples.

Effect:

- builds the initial long-term grammar

Channel:

- `fit_support()`

### 4.2 Safe observation update

When safe balls are placed, the risk learner updates:

- [update_from_safe_observation()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/risk_belief.py:139)

Effect:

- moves danger prototypes toward safe observations
- improves future danger discrimination

### 4.3 Death update

When the learner dies on a dangerous selection:

- [update_from_death()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/risk_belief.py:124)

Effect:

- danger posterior shifts toward danger types
- future similar vectors look more dangerous

### 4.4 Warning update

Warnings act as set-conditional evidence:

- [warning_set_bayes_update()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/warning_update.py:15)

Semantics:

- "at least one ball in the selected set is dangerous"

This is an important update path for risk learning because it teaches without death.

### 4.5 Wrong confirm feedback update

This is the main online **grammar** learning path in the real learner.

Implemented in:

- [FeedbackUpdater](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/feedback_update.py:18)

Pipeline:

1. get beam posterior over latent traces
2. compute feedback likelihood under wrong-only or wrong-position feedback
3. reweight beam posterior
4. apply differential M-step to concept statistics
5. invalidate target cache
6. re-predict `target_output`

Important current fact:

- this is the primary learner-side online grammar update path

#### Core grammar-feedback formulas

Let:

- `x` be the query words
- `Ŷ` be the learner's submitted output at confirm time
- `π_k` be a beam trace candidate
- `Y_k = Y(π_k)` be the rendered output of `π_k`
- `q_k` be the base beam posterior

The base posterior over beam traces is:

```text
q_k ∝ exp(score(π_k))
```

implemented by softmax over beam scores in:

- [reweight_beam_posterior()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/feedback_update.py:79)

With feedback `F`, the learner computes:

```text
q̃_k ∝ q_k · P(F | Ŷ, Y_k)
```

##### Wrong-only mode

In `wrong_only`, the learner only knows the submission is wrong:

```text
P(F_wrong | Ŷ, Y_k) =
  ε_wrong,         if Y_k = Ŷ
  1 - ε_wrong,     if Y_k ≠ Ŷ
```

implemented in:

- [compute_feedback_likelihood_wrong_only()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/feedback_update.py:26)

##### Wrong-position mode

In `wrong_positions`, the learner gets a correctness mask `m_ℓ` over positions:

```text
s_{k,ℓ} =
  1 - ε_eq,   if Y_{k,ℓ} = Ŷ_ℓ
  ε_eq,       otherwise
```

If position `ℓ` is tutor-assisted, evidence can be discounted by `ρ_assist`:

```text
w_ℓ =
  ρ_assist,   if assisted
  1,          otherwise
```

Then:

```text
log P(F_mask | Ŷ, Y_k, m)
= Σ_{ℓ: m_ℓ = 1} w_ℓ log s_{k,ℓ}
  Σ_{ℓ: m_ℓ = 0} w_ℓ log (1 - s_{k,ℓ})
```

implemented in:

- [compute_feedback_likelihood_wrong_positions()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/feedback_update.py:39)

##### Differential M-step

After reweighting, the learner applies a differential update to concept statistics:

```text
Δθ ∝ η_fb · Σ_k (q̃_k - q_k) · features(π_k)
```

In code, this is applied to:

- role counts
- emit statistics
- color counts
- repeat counts

via:

- [differential_m_step()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/feedback_update.py:124)

This is the main reason failed confirm can change the learner's grammar online.

### 4.6 Successful confirm

In the original learner pipeline, successful confirm does **not** trigger an explicit real-learner grammar consolidation step.

That is why Goal B required adding:

- [correct_update.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/correct_update.py:1)

This is currently an experimental Goal B module, not the default historical learning path.

#### Experimental correct-update formula

The experimental positive update uses GT-constrained inference:

```text
q_correct(π | x, GT) ∝ exp(score(π; x, θ)) · 1[Y(π) = GT]
```

and then applies a weighted M-step:

```text
Δθ_correct ∝ η_correct · Σ_{π ∈ Π_GT} q_correct(π) · features(π)
```

This is implemented in:

- [apply_correct_answer()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/correct_update.py:22)

This module was added specifically to study Goal B and is not the original online learner update path.

### 4.7 Hint-related updates

Hints affect the learner through multiple possible channels:

1. **completion assistance**
   - direct fill into `state.completion`
   - implemented by [apply_hint_to_state()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/action_generators.py:101)

2. **assist-aware feedback discount**
   - tutor-assisted positions can be downweighted in grammar feedback
   - controlled by `rho_assist`
   - implemented inside [compute_feedback_likelihood_wrong_positions()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/feedback_update.py:39)

3. **hint-aware target inference**
   - recent extension
   - hints can bias the beam / target prediction itself
   - implemented in `target_predictor.py`

4. **autonomy shift**
   - recent extension
   - hints can lower confirm thresholds and reduce exploration
   - implemented in `policy.py`

5. **persistent hint reliance**
   - recent extension
   - cross-query trust / reliance state
   - implemented in [hint_reliance.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/hint_reliance.py:1)

These channels have very different semantics and should not be conflated.

---

## 5. Tutor Hierarchy

The codebase contains multiple tutor designs with increasing modeling sophistication.

### 5.1 No tutor / baseline tutors

Defined in:

- [dummy_tutor.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/dummy_tutor.py:1)

Variants include:

- `NoTutor`
- `NoTutorImmortalWarnlike`
- `NoTutorImmortalNoTimeout`
- `OracleWarningTutor`

These are useful as lower and upper bounds.

### 5.2 T0 RuleTutor

Defined in:

- [tutor_rule.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/tutor_rule.py:14)

Characteristics:

- always warns on danger
- hints by threshold rule based on timeout risk
- courage when stuck and safe-needed exists

This tutor is simple and truthful, but not deeply learner-model based.

### 5.3 T1 ProxyTutor

Defined in:

- [tutor_proxy.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/tutor_proxy.py:18)

Characteristics:

- chooses actions by low-dimensional utility proxy
- compares WAIT / WARNING / COURAGE / HINT using belief-based gains and costs
- does not run a full internal learner clone

This is a belief-based utility tutor, not a true inverse learner model.

#### Proxy utility formula

`ProxyTutor` uses an additive utility of the form:

```text
Q(a) = Σ_i λ_i G_i(a) - Σ_j λ_j C_j(a)
```

where gains and costs summarize things like:

- teaching success gain
- eval gain proxy
- death prevention
- timeout prevention
- over-help penalty
- fixed intervention cost

This is the main modeling idea behind:

- [ProxyTutor](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/tutor_proxy.py:18)

and the supporting utility code in `tutor_api/utility.py`.

### 5.4 T2 ShadowTutor

Defined in:

- [tutor_shadow.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/tutor_shadow.py:24)

Characteristics:

- clones the learner into a shadow state
- simulates candidate actions on the shadow
- estimates counterfactual eval gain

This is the most direct "copy-and-simulate" tutor architecture.

### 5.5 T3 BehavioralTutor

Defined in:

- [tutor_behavioral.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/tutor_behavioral.py:1)

Characteristics:

- does not read learner internals
- builds its own grammar from support
- infers learner competence from behavior
- can give **wrong hints**, because hint content comes from the tutor's own grammar prediction

This is the main source of natural bad-hint content in current experiments.

### 5.6 T3 InverseTutor

Defined in:

- [tutor_inverse.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/tutor_inverse.py:1)

Characteristics:

- separates:
  - `TutorTaskModel`: what is actually correct
  - `TutorLearnerModel`: what the tutor thinks the learner believes
- hint content is oracle-correct
- hint timing / gating depends on inferred learner state
- warning is an oracle safety gate in the current main line

This tutor is the main Phase 4 architecture.

Important distinction:

- `InverseTutor` is ideal for studying **bad learner modeling / over-help**
- `BehavioralTutor` is ideal for studying **bad hint content**

#### Inverse hint decision formulas

The current `InverseTutor` separates:

```text
Hint content  ← TutorTaskModel (ground truth)
Hint decision ← TutorLearnerModel + hint policy
```

The coarse hint utility in the current policy is:

```text
Q_hint
= λ_to · (1 - P_succ_wait)
+ λ_err · E_wrong
+ λ_unc · (1 - margin)
+ λ_H · H_beam_norm
- λ_over · σ(competence)
- λ_int
```

implemented in:

- [compute_coarse_hint_utility()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/hint_policy.py:66)

If `Q_hint > 0`, the tutor then scores positions.

Current position scoring uses:

```text
score_i = 1[wrong_i] · (λ_pos · H_i + λ_impact · I_i + λ_trace · T_i)
```

where:

- `H_i` = position uncertainty
- `I_i` = expected impact on success if hinted
- `T_i` = trace salience / structural uncertainty

implemented in:

- [score_positions()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/hint_policy.py:100)

The tutor then greedily adds hint positions while marginal gain stays positive:

```text
ΔQ > 0  ⇒ keep adding positions
ΔQ ≤ 0  ⇒ stop
```

implemented in:

- [select_hint_positions_greedy()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/hint_policy.py:157)

---

## 6. Tutor Knowledge Decomposition

### 6.1 TutorTaskModel

Defined in:

- [task_model.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/task_model.py:1)

Represents:

- correct outputs
- danger truth
- correct hint content

This is teacher knowledge, not learner modeling.

### 6.2 TutorLearnerModel

Used by `InverseTutor`.

Represents:

- tutor's inferred model of learner grammar

It is updated from:

- observation records
- submitted outputs
- wrong-position feedback

This allows the tutor to estimate:

- learner beam entropy
- likely failure positions
- whether hinting is useful

### 6.3 Tutor belief summaries

Earlier tutor lines also use summarized beliefs:

- [TutorBelief](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/tutor_state.py:1)
- [belief_update.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/belief_update.py:1)

These are lighter-weight than the inverse learner model.

---

## 7. Risk and Danger Mechanism

The risk subsystem is separate from the grammar subsystem.

### 7.1 Environment side

Each episode samples a new danger model:

- safe and danger prototype clusters
- noisy per-ball observations

Thus:

- danger structure varies by episode
- grammar and danger are coupled only through the learner's need to survive while collecting colors

### 7.2 Learner side

The learner maintains probabilistic beliefs over danger types.

This affects:

- set selection
- whether risky needed colors are chosen
- whether the learner becomes over-avoidant

The policy uses:

- `p_danger = 1 - P(safe | x)`

in [policy.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/policy.py:35)

### 7.3 Tutor side

Current tutors generally have oracle access to whether a selected set contains danger.

Differences among tutors are less about raw danger detection and more about:

- whether to intervene
- whether to let the learner act autonomously
- whether to risk non-intervention for learning

In the current `InverseTutor` main line, warning is effectively oracle and always truthful.

#### Warning-as-evidence formula

When a warning is issued for a selected set `S`, the learner gets set-level evidence:

```text
warning ≡ ∃ i ∈ S : z_i ≠ 0
```

Using the independence approximation, the learner updates selected-ball posteriors by conditioning on:

```text
P(Z | X, warning) ∝ 1[not all safe] · P(Z | X)
```

implemented in:

- [warning_set_bayes_update()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/warning_update.py:15)

---

## 8. Hint Mechanisms

Hints are one of the most overloaded concepts in this codebase, so they need to be separated carefully.

### 8.1 Original hint semantics

Originally, hints are mainly **task-completion assistance**:

- after confirm failure, the tutor may place one or two correct balls directly
- this helps finish the current query

Core generation logic:

- [generate_post_confirm_actions()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/action_generators.py:49)

Core application logic:

- [apply_hint_to_state()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/action_generators.py:101)

### 8.2 Assist discount

When the learner later receives wrong-position feedback, tutor-assisted positions can be discounted.

Motivation:

- a tutor-filled position is not clean evidence that the learner itself understood the grammar

This is controlled by:

- `rho_assist`

Important empirical finding:

- in current experiments, `rho_assist` has little measurable effect on top-line results

### 8.3 Hint-aware inference

Recent experiments added a stronger path:

- hint can change what the learner thinks the answer is

This is implemented through beam conditioning in:

- [target_predictor.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/target_predictor.py:16)

Current modes:

- `hard`: filter to traces consistent with hint positions
- `soft`: reweight beam by hint compatibility

This mechanism is the first one that clearly makes the learner "guess along with the tutor".

#### Hint-bias formulas

Let the hint payload be:

```text
h = {(i, c_i)}
```

where position `i` is hinted to have color `c_i`.

Current hard mode:

```text
q_hint(π | x, h) ∝ exp(score(π)) · 1[Y_i(π) = c_i for all hinted i]
```

Current soft mode:

```text
Compat(π; h) = Σ_{(i,c_i)∈h} 1[Y_i(π) = c_i]
q_hint(π | x, h) ∝ exp(score(π) + β_hint · Compat(π; h))
```

implemented in:

- [beam_posterior_with_hint()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/target_predictor.py:37)

The predicted target then becomes:

```text
Y*_hint = argmax_π q_hint(π | x, h)
```

implemented in:

- [predict_target_with_hint()](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/target_predictor.py:89)

### 8.4 Hint-induced autonomy shift

Recent experiments also added a behavior-level shift:

- hint can make the learner confirm earlier
- hint can reduce exploration

Implemented in:

- [policy.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/policy.py:14)

Current empirical finding:

- this is weaker than hint-aware inference
- by itself, it has shown little standalone effect so far

#### Autonomy-shift formulas

Current autonomy shift is implemented as a policy modulation after hint:

```text
τ_confirm_eff = max(0, τ_confirm - hint_confirm_bonus)
ε_eff = ε_policy · (1 - hint_exploration_drop)
```

so the learner:

- confirms earlier
- explores less

This is implemented in:

- [policy.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/policy.py:109)

### 8.5 Persistent reliance

Recent extension:

- [hint_reliance.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/hint_reliance.py:1)

This tracks cross-query trust in hints and modulates:

- hint bias strength
- confirm bonus
- exploration drop

Current empirical finding:

- reliance can diagnose helpful vs harmful hint regimes
- but if hint content itself is wrong and direct completion fill is the main damage path, trust dampening alone is not enough

#### Reliance-update formulas

The recent reliance module tracks a scalar trust state:

```text
r_t ∈ [0, 1]
```

Helpful hints push trust upward:

```text
r_{t+1} = r_t + lr_up · (1 - r_t)
```

Harmful hints push trust downward:

```text
r_{t+1} = r_t - lr_down · r_t
```

No-hint queries decay trust toward baseline:

```text
r_{t+1} = r_t + decay_rate · (baseline - r_t)
```

implemented in:

- [HintRelianceState](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/hint_reliance.py:12)

Trust then modulates effective hint strength, for example:

```text
β_hint_eff = β_hint_base · r_t
```

and similarly for confirm bonus and exploration drop.

---

## 9. Goal B Mechanism Status

Goal B asks whether:

```text
wrong × k + final correct answer learning
```

can outperform:

```text
direct correct answer learning
```

### 9.1 What the current learner naturally supports

The real learner naturally supports:

- wrong-confirm grammar update

It did **not** historically support:

- explicit success-side positive consolidation

That is why the experimental module:

- [correct_update.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/correct_update.py:1)

was added.

### 9.2 Current Goal B result

Current evidence strongly suggests:

- `correct × N` beats `wrong × k + correct` under matched budget
- weak wrong writes can reduce anti-synergy, but do not reverse the ranking
- pure history reweighting on fixed GT support does not help enough
- current task families do not expose enough extra latent structure for wrong histories to add much beyond direct GT supervision

### 9.3 Why this happens in current code

Mechanistically:

1. direct correct update is already very strong because it uses GT-constrained inference
2. wrong-side update acts on the same long-term grammar object
3. current trace families are often too sparse / discrete
4. many wrong histories do not add extra structure beyond what GT already tells the learner

This is why Goal B currently looks like a mostly negative result in this task family.

#### Goal B formalization

Let:

- `x` be the query words
- `GT` be the correct output
- `D^- = {(Ŷ_t, m_t)}_{t=1..k}` be the wrong-confirm history
- `π` be a latent trace
- `score(π; x, θ)` be the current grammar score

Direct correct learning is:

```text
q_dir(π | x, GT, θ)
∝ exp(score(π; x, θ)) · 1[Y(π) = GT]
```

Goal B requires something genuinely different:

```text
q_goalB(π | x, GT, D^-, θ)
∝ exp(score(π; x, θ))
  · 1[π ∈ Π_goalB(GT, D^-)]
  · C(π; D^-)
```

where:

- `Π_goalB(GT, D^-)` is the final support set
- `C(π; D^-)` is a history-dependent weighting factor

Current negative results imply:

- merely sharpening a fixed `Π_GT` is not enough
- wrong history must either change the support family or provide extra structural discrimination not already contained in GT supervision

---

## 10. Current Hint-Modeling Result

Current hint experiments support the following summary.

### 10.1 Timing matters

Earlier hints improve teaching success more than later hints.

### 10.2 Mechanism details matter less than timing in the original assistance setup

In the old assistance-only setting:

- leftmost vs random position mattered little
- `rho_assist` mattered little
- larger hint payloads mainly improved task completion efficiency

### 10.3 Hint-aware inference creates real learner reaction

Once hint enters target prediction:

- hints can flip the learner's predicted target
- the learner converges faster after hint
- this is the clearest current evidence that hint is affecting inference, not just doing the task for the learner

### 10.4 Bad hint content is genuinely harmful

With `BehavioralTutor`, hint content can be wrong.

Current result:

- wrong hint content can substantially harm teaching outcomes
- the dominant damage path is often direct wrong completion fill, not only inference bias

### 10.5 Over-help / dependency is not yet strongly visible in eval

So far:

- hint-aware inference clearly changes teach-time behavior
- but long-term eval degradation remains weak or absent

The likely reasons are:

- teach horizons are still short
- hint effects are not yet strongly persistent across queries
- current task family may not amplify dependency strongly enough

---

## 11. Current System-Level Takeaways

### 11.1 What the current system does well

- separates grammar learning from danger learning cleanly
- supports multiple tutor architectures with increasing modeling sophistication
- supports both truthful and imperfect hint regimes
- supports detailed per-query diagnostics through beam, trace, and shadow analysis

### 11.2 What is currently the main online learner-side teaching channel

For grammar:

- wrong confirm feedback

For risk:

- warning / safe observation / death updates

Hints historically acted more as:

- task assistance

and only recently began to act as:

- inference-shaping signals

### 11.3 What the codebase currently suggests

The most important current distinction is:

- **bad content** problems are best studied with `BehavioralTutor`
- **bad learner modeling / over-help** problems are best studied with `InverseTutor`

And for Goal B:

- the main blocker is not missing code plumbing anymore
- it is the combination of strong direct-correct updates and limited latent family richness in the current task setup

---

## 12. Practical Reading Guide

If you want to understand the current system quickly, read in this order:

1. [config.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/config.py:1)
2. [grammar_task_env.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/grammar_task_env.py:1)
3. [state.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/environment/state.py:1)
4. [cls_wrapper.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/cls_wrapper.py:1)
5. [risk_belief.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/risk_belief.py:1)
6. [policy.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/policy.py:1)
7. [feedback_update.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/feedback_update.py:1)
8. [action_generators.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/action_generators.py:1)
9. [tutor_rule.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/tutor_rule.py:1)
10. [tutor_proxy.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/tutor_proxy.py:1)
11. [tutor_shadow.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/tutor_shadow.py:1)
12. [tutor_behavioral.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/tutor_behavioral.py:1)
13. [tutor_inverse.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/tutor_api/tutor_inverse.py:1)
14. [run_phase4.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/experiments/run_phase4.py:1)

If you specifically care about current research extensions, then add:

15. [target_predictor.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/target_predictor.py:1)
16. [hint_reliance.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/hint_reliance.py:1)
17. [correct_update.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/correct_update.py:1)
18. [goal_b_state.py](F:/SCAI/Learning-agent/cls_color_selection/cls_color_selection/learner/goal_b_state.py:1)
