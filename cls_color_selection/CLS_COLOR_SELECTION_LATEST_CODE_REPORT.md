# cls_color_selection: Latest Code-Based Mechanism Report

Generated from direct code inspection on 2026-04-18.

This report is based on the current `cls_color_selection` code under:

- `cls_color_selection/cls_color_selection/config.py`
- `cls_color_selection/cls_color_selection/constants.py`
- `cls_color_selection/cls_color_selection/interfaces.py`
- `cls_color_selection/cls_color_selection/environment/*`
- `cls_color_selection/cls_color_selection/learner/*`
- `cls_color_selection/cls_color_selection/tutor_api/*`
- `cls_color_selection/cls_color_selection/experiments/run_phase4.py`
- `cls_color_selection/cls_color_selection/experiments/registry_phase4.py`

It is intentionally code-grounded. Where a behavior is not wired into the current main runner, this report marks it as experimental or disconnected instead of treating it as a live system feature.

---

## 1. Executive Summary

`cls_color_selection` is a tutoring-and-learning environment built around two coupled but distinct learning problems:

1. grammar learning:
   the learner must infer a symbolic color grammar from support examples and online confirm feedback
2. danger learning:
   the learner must infer which candidate balls are dangerous from noisy continuous observations

The latest orchestrated benchmark in the repo is `experiments/run_phase4.py`. The default condition is `T3_traceInfer`, and `registry_phase4.py` explicitly labels the inverse-inference tutors as the "NEW main line". In this main line:

- the learner is a CLS-based grammar learner plus a Bayesian danger belief plus a greedy selection policy
- the tutor is split into:
  - `TutorTaskModel`: what is actually correct
  - `TutorLearnerModel`: what the tutor infers the learner currently believes
- the observation phase is result-level only for the inverse tutor
- the teach phase is process-level
- the eval phase is frozen and tutor-free

The most important current code-level facts are:

- the learner's main online grammar update channel is still failed confirm feedback via `FeedbackUpdater`
- successful confirm does not perform a standard positive grammar consolidation update in the main runner
- the inverse tutor's hint content is oracle-correct, but its decision about whether to hint and where to hint is learner-model-driven
- risk tutoring is mostly oracle-warning-based in the current main line
- several newer hint-learning and Goal B modules exist, but they are not fully wired into the default `phase4` execution path

---

## 2. What the Current "Latest" System Is

### 2.1 Current main experiment line

The clearest code-level indicator of the latest active line is:

- `experiments/run_phase4.py`
  - file header says "Phase 4 experiment runner: Inverse Inference Tutor"
  - parser default is `--conditions T3_traceInfer`
- `experiments/registry_phase4.py`
  - comments label `T3_roleInfer` / `T3_traceInfer` as the "NEW main line"

So the most defensible answer to "what is latest" is:

> The current latest orchestrated line is Phase 4, centered on `InverseTutor`, with `T3_traceInfer` as the default condition.

### 2.2 Older but still present lines

The codebase still contains older or parallel tutor lines:

- Phase 1 / simple tutor baselines:
  - `dummy_tutor.py`
  - `tutor_rule.py`
- Phase 2 / Phase 3 tutor-belief and behavioral ToM lines:
  - `observation.py`
  - `tutor_proxy.py`
  - `tutor_shadow.py`
  - `tutor_behavioral.py`
- Phase 4 inverse tutor:
  - `observation_v2.py`
  - `task_model.py`
  - `learner_model.py`
  - `tutor_inverse.py`

So the repo is not single-line clean. It is a layered research codebase where earlier tutor families remain available as baselines or comparison points.

---

## 3. Scenario and Environment

### 3.1 Task source

Tasks come from BASIC text files under `BASIC/cls_learner/data`, resolved by `FullConfig.resolve_data_dir()`.

Each task file defines:

- support examples
- query examples
- a latent grammar

Grammar parsing and rendering live in:

- `environment/generator.py`

### 3.2 Query semantics

For an input query `x = [w_1, ..., w_n]`, the environment has:

- ground-truth output:
  - `GT(x)`
- learner-predicted target:
  - `Y*(x; theta_grammar)`

This distinction is central.

The query state stores both:

- `ground_truth`
- `target_output`

The learner fills a completion aligned to `target_output`, not directly to `ground_truth`.

This means the task is not just "pick correct colors"; it is:

1. infer what output the query should have
2. select balls under danger uncertainty
3. place them into the learner's currently believed target template
4. get confirm feedback against the actual ground truth

### 3.3 Candidate pool and danger generation

Each candidate ball has:

- a discrete color
- a latent danger type
- a continuous observed vector

From `interfaces.py`:

- `CandidateBall(index, color, danger_vec, observed_vec, is_danger, danger_type)`

Danger generation is in `environment/generator.py`.

Let:

- `z_i = 0` mean safe type
- `z_i in {1, ..., K}` mean danger type

Then ball generation is:

- choose color from grammar palette
- sample `is_danger ~ Bernoulli(danger_ratio)`
- choose latent prototype type
- sample latent vector around the type prototype
- add observation noise

A compact code-faithful description is:

```text
z_i ~ categorical(prior over safe/danger types)
d_i ~ N(mu_{z_i}, Sigma_{z_i})
x_i = d_i + epsilon_i,  epsilon_i ~ N(0, obs_sigma^2 I)
```

Defaults from `EnvConfig`:

- `n_candidates = 8`
- `n_confirm_max = 5`
- `danger_ratio = 0.3`
- `n_safe_types = 1`
- `n_danger_types = 3`
- `obs_sigma = 0.3`
- `cluster_sigma = 0.5`

### 3.4 Query splits

The experiment protocol uses:

- observation queries
- teach queries
- eval queries

Defaults in `ExpConfig`:

- `n_obs_queries = 4`
- `n_teach_queries = 8`
- `n_eval_queries = 8`

Default source mode is:

- `query_source_mode = "generated"`

Generation logic is in `environment/query_generator.py`.

Difficulty tiers:

- easy: shorter, simpler expressions
- medium
- hard: longer and more compositional expressions

The generator also supports:

- `txt_only`
- `txt_resample`
- `hybrid`
- `generated`

---

## 4. Core Query Timeline

The environment semantics are implemented mainly in:

- `environment/grammar_task_env.py`
- `environment/transition.py`

A single teach/eval query proceeds as follows.

### 4.1 Initialization

For query `x`, the runner computes:

```text
Y* = learner_target_predictor.predict_target(x)
```

Then the environment initializes:

- `target_output = Y*`
- `ground_truth = GT`
- `completion = [None] * len(Y*)`
- new candidate pool

### 4.2 Selection

The learner selects a subset of indices from the candidate pool.

Tutor hook:

- `tutor.on_select(state, selected_balls)`

Then environment transition:

- if tutor warns: selection is voided
- if tutor waits and any selected ball is dangerous: death
- if tutor waits and all selected balls are safe: balls auto-place into the completion according to `target_output`

### 4.3 Auto-placement

Placement is deterministic and target-conditioned:

```text
Place(z_t, Y*, S_t):
for each selected ball color c:
  place into the leftmost unfilled slot in Y* that expects c
```

So even safe and useful balls can be placed "wrong" with respect to the real grammar if the learner's current `Y*` is wrong.

### 4.4 Confirm

When the learner confirms, `transition.confirm()` compares the current completion to `ground_truth`, not to `target_output`.

It returns:

- `correct`
- `mask`
- `submitted`

If all positions match ground truth:

- `Outcome.SUCCESS`

If confirm budget is exhausted:

- `Outcome.TIMEOUT`

Otherwise the learner continues after feedback.

---

## 5. Learner Architecture

The learner has three main components:

1. grammar model
2. danger belief model
3. selection/confirm policy

### 5.1 Grammar learner: CLSSequencePredictor

Grammar learning is wrapped in:

- `learner/cls_wrapper.py`

The underlying engine is BASIC CLS / NS learner code. This repo does not reimplement symbolic induction from scratch; it wraps the external CLS learner.

The wrapper exposes:

- `fit_support(examples)`
- `predict_target(words)`
- `beam_posterior(words)`

Defaults:

- `n_sup = 14`
- `n_em = 3`
- `cls_mode = "ast"`

Interpretation:

- support examples initialize the learner's grammar
- `predict_target(words)` gives the learner's current best output hypothesis
- `beam_posterior(words)` gives a ranked set of latent explanations and outputs

### 5.2 Target predictor layer

`learner/target_predictor.py` provides caching around grammar predictions.

Main standard methods:

- `predict_target(words)`
- `beam_posterior(words)`

There are also newer hint-aware methods:

- `beam_posterior_with_hint(...)`
- `predict_target_with_hint(...)`
- `merge_completion_after_target_flip(...)`

These are real code, but they are not wired into the default `run_phase4.py` query loop. So they exist as implementation branches, not as current mainline behavior.

### 5.3 Danger belief model

Danger learning lives in:

- `learner/risk_belief.py`

The learner models the latent type of each ball as:

```text
z_i in {0, 1, ..., K}
0 = safe
1..K = danger subtypes
```

For a ball observation `x_i`, the learner computes:

```text
P(z_i = k | x_i) proportional to P(x_i | z_i = k) P(z_i = k)
```

The code uses a diagonal Gaussian likelihood around learned prototypes.

For a set of selected balls `S`, the learner's estimated probability that the set contains any danger is:

```text
P(exists danger in S | X)
= 1 - product_i P(z_i = 0 | x_i)
```

This independence approximation is reused in the warning update too.

### 5.4 Selection policy

The selection policy is in:

- `learner/policy.py`

The learner computes ball-level marginal utility using:

```text
u_i
= alpha_fill * g_fill(i)
- alpha_risk * p_danger(i)
- alpha_waste * g_waste(i)
```

where:

- `g_fill(i) = 1` if the ball color is still needed in the current target
- `g_waste(i) = 1` if the color is not needed
- `p_danger(i)` comes from the danger posterior

Default weights:

- `alpha_fill = 1.0`
- `alpha_risk = 2.0`
- `alpha_waste = 0.3`

The policy greedily adds balls until no additional ball has worthwhile utility.

There is also:

- optional risk gate `risk_gate_tau`
- epsilon exploration `epsilon_policy`

### 5.5 Confirm policy

The learner confirms when:

```text
is_complete
or
fill_ratio >= confirm_fill_threshold
```

Default:

- `confirm_fill_threshold = 1.0`

So the default learner is conservative: it usually confirms only when all positions in its current target are filled.

---

## 6. Learner Update Mechanisms

### 6.1 Support study

Initial grammar learning comes from support examples:

```text
theta_grammar <- CLS.study(support)
```

This is done through `CLSSequencePredictor.fit_support(...)`.

### 6.2 Safe observation update

When the learner selects safe balls and survives, the danger belief is updated using:

- `risk_belief.update_from_safe_observation(x_i)`

This acts like a labeled safe prototype update.

### 6.3 Death update

When the learner dies on a dangerous ball:

- `risk_belief.update_from_death(x_i)`

This performs a danger-labeled update and redistributes mass among danger types.

### 6.4 Warning update

When the tutor issues `WARNING`, the semantics are:

> at least one ball in the selected set is dangerous

The learner-side warning update is in:

- `learner/warning_update.py`

The posterior update is:

```text
P(Z | X, warning)
proportional to 1[exists i: z_i != 0] * P(Z | X)
```

The implementation uses an independence approximation to update the selected balls' danger posteriors and then updates the prototype statistics.

### 6.5 Wrong-confirm grammar update

This is the main online grammar learning channel in the standard system.

Code:

- `learner/feedback_update.py`

For a wrong submission `Y_hat`, the learner:

1. gets a beam posterior over candidate traces
2. scores each beam element against confirm feedback
3. reweights the posterior
4. applies a differential M-step to concept statistics

The reweighting is:

```text
q_new(k) proportional to q_old(k) * L_feedback(Y_hat, Y_k, mask)
```

Two likelihood families exist:

- `wrong_only`
- `wrong_positions`

The default is:

- `feedback_mode = "wrong_positions"`

For `wrong_positions`, the per-position likelihood uses:

```text
s_i =
  1 - eps_eq    if candidate agrees with submission at position i
  eps_eq        otherwise
```

combined over correct/wrong positions according to the feedback mask.

Hint-assisted positions can be discounted with:

- `rho_assist`

Then the update to grammar parameters is differential:

```text
Delta theta
= eta_fb * sum_k (q_new(k) - q_old(k)) * sufficient_stats(trace_k)
```

In the code this is applied to:

- `role_counts`
- `emit_stats`
- `repeat_counts`
- `color_counts`

depending on the trace content.

### 6.6 Important asymmetry: successful confirm

In the mainline runner, successful confirm does not trigger a symmetric positive grammar-consolidation update.

That is why the experimental file:

- `learner/correct_update.py`

exists. But that module is not part of the default `run_phase4.py` loop.

So the code-grounded current reality is:

> standard online grammar learning is dominated by failed-confirm feedback, not by explicit success-side positive consolidation.

---

## 7. Tutor Architecture

The tutor family is layered.

### 7.1 T0: RuleTutor

Code:

- `tutor_api/tutor_rule.py`

Behavior:

- warns truthfully on danger
- hints when timeout risk is high and confirms are nearly exhausted
- courage when learner appears stuck and a safe needed ball exists

Timeout risk is estimated from tutor belief by:

```text
P(timeout)
approximately (1 - a_probe) ^ confirms_left
```

where `a_probe` is the tutor's current estimate of learner grammar success.

### 7.2 T1: ProxyTutor

Code:

- `tutor_api/tutor_proxy.py`
- `tutor_api/utility.py`

Proxy tutor scores candidate actions using:

```text
Q(a)
= lambda_eval * G_eval(a)
+ lambda_teach * G_teach(a)
+ lambda_death * G_death(a)
+ lambda_to * G_to(a)
- lambda_over * C_over(a)
- lambda_int * C_int(a)
```

For hints, the current implementation simplifies this to:

```text
Q_hint
= G_teach + G_to + G_eval - C_over - C_int
```

with:

- `G_eval = 0` in the proxy hint utility
- `G_teach` from extra filled positions
- `G_to` from timeout reduction
- `C_over` from a sigmoid of learner competence
- `C_int` fixed intervention cost

So ProxyTutor is a belief-based utility tutor, but not a full counterfactual learner clone.

### 7.3 T2: ShadowTutor

Code:

- `tutor_api/tutor_shadow.py`

This tutor keeps a shadow snapshot of the learner's state and evaluates counterfactual tutor actions by simulating updates on the shadow copy.

For hint actions, it computes:

```text
Q_a = g_eval + g_teach - c_over - c_int
```

where:

- `g_eval = lambda_eval * estimate_shadow_eval_gain(...)`
- `g_teach = lambda_teach * (k / remaining_slots)`
- `c_over` increases with learner competence and hint size
- `c_int` is a fixed intervention cost

This is the strongest internal-oracle baseline among the shipped tutor families.

### 7.4 T3 legacy line: BehavioralTutor

Code:

- `tutor_api/tutor_behavioral.py`
- used in `experiments/run_phase3.py`

BehavioralTutor does not copy the learner model. It:

- studies support independently
- tracks behavioral competence statistics
- gives hints from its own grammar model

Important consequence:

> its hints can be wrong if its own grammar is wrong.

Key competence estimates are:

```text
gram_competence = confirm_success / confirm_total
risk_competence = safe_selected / (safe_selected + danger_selected)
stuck_tendency = min(1, avg_retries / 10)
```

This tutor is still important for bad-hint-content studies, but it is not the current default mainline.

### 7.5 T3 current main line: InverseTutor

Code:

- `tutor_api/tutor_inverse.py`
- `tutor_api/task_model.py`
- `tutor_api/learner_model.py`
- `tutor_api/hint_policy.py`

This is the newest main tutor design.

Its architecture is explicitly split:

- `TutorTaskModel`
  - knows ground truth outputs
  - knows danger oracle
  - generates correct hint content
- `TutorLearnerModel`
  - tries to infer the learner's grammar state from observed behavior

This is the central conceptual separation in the current repo.

---

## 8. Inverse Tutor Mechanism

### 8.1 Observation model

Phase 4 introduces `observation_v2.py`.

Unlike the older `observation.py`, this new observation phase exposes only result-level information:

- query words
- submitted output if any
- end-of-query completion
- outcome
- wrong-position mask if available

The point is to be more realistic:

> obs phase is result-level only, not full process-level introspection.

### 8.2 TutorLearnerModel

The tutor's learner model is in:

- `tutor_api/learner_model.py`

It maintains its own CLS agent and updates it from observed learner outputs.

There are three init modes:

- `support`
  - oracle upper bound: tutor starts from same support as learner
- `cold`
  - vocabulary only
- `teacher_prior`
  - tutor studies task queries, not learner support

There are three update depths:

- `role_only`
- `role_emit`
- `full_trace`

Update from observed output:

```text
theta_tutor_learner <- constrained_beam_M_step(words, Y_submit)
```

This is not a copy of learner internals. It is inverse fitting:

> find latent traces that could have produced the learner's observed output, then update the tutor's model of the learner accordingly.

Update from feedback:

```text
q_new(k) proportional to q_old(k) * L_feedback
theta <- theta + eta_inv * sum_k (q_new - q_old) * stats_k
```

So the tutor tries to absorb two distinct evidence channels:

- what the learner produced
- how that production was judged

### 8.3 Inverse tutor warning

In the current Phase 4 main line, warning is intentionally simple:

- `InverseTutor.on_select()` uses oracle safety gating
- if selected set contains any danger, it warns

So the inverse tutor's distinctive value is not mainly in danger detection. That part is mostly held oracle-clean in the current main line.

### 8.4 Inverse tutor hint utility

The fine-grained hint policy is in:

- `tutor_api/hint_policy.py`

Coarse hint utility:

```text
Q_hint
= lambda_to  * (1 - P_succ_wait)
+ lambda_err * E_wrong
+ lambda_unc * (1 - margin)
+ lambda_H   * H_beam_norm
- lambda_over * sigmoid(competence)
- lambda_int
```

where:

- `P_succ_wait = 1 - (1 - p_exact)^c_left`
- `E_wrong` is expected wrong ratio under the tutor's estimated learner beam
- `margin` is top-1 vs top-2 gap
- `H_beam_norm` is normalized beam entropy

If `Q_hint <= 0`, the tutor waits.

### 8.5 Position selection

For wrong positions, the tutor scores candidate hint locations using:

```text
score_i
= lambda_pos   * H_i
+ lambda_impact * I_i
+ lambda_trace * T_i
```

where:

- `H_i` is per-position color entropy
- `I_i` is estimated success gain from fixing that position
- `T_i` is trace salience from structural uncertainty

Then positions are added greedily until marginal gain is no longer positive.

### 8.6 Trace salience

`tutor_api/trace_analysis.py` estimates structural uncertainty per word:

- role entropy
- repeat entropy
- emit-color entropy

These are projected to output positions through an approximate alignment matrix:

```text
T_i = sum_w A_{iw} * (gamma_r * H_role(w)
                    + gamma_k * H_rep(w)
                    + gamma_e * H_emit(w))
```

This is how the tutor tries to turn beam-level grammar ambiguity into hint-position salience.

### 8.7 Optional counterfactual veto

`tutor_inverse.py` also contains an optional one-step counterfactual hook using:

- `tutor_api/counterfactual.py`

This is off by default in the mainline path and only activated by private Phase 6 overrides.

So it is best described as an experimental augmentation, not as core current behavior.

---

## 9. Tutor Belief State for Non-Inverse Tutors

The older baseline tutors use:

- `tutor_api/tutor_state.py`
- `tutor_api/belief_update.py`

The belief is factorized into:

- `B_sem`
- `B_risk`
- `B_type`

### 9.1 Beta posterior primitives

The core primitive is:

```text
Beta(alpha, beta)
mean = alpha / (alpha + beta)
```

Used for:

- semantic success rate
- danger detection rate
- over-avoidance rate

### 9.2 Type inference

Optional learner type inference uses Gaussian profiles over observed statistics:

- `balanced`
- `risk_averse`
- `slow_uncertain`

The update is:

```text
log P(T=t | H) <- log P(T=t | H) + log P(obs_stats | T=t)
```

This machinery is still in the repo and still used by the older proxy-style tutor family.

---

## 10. Current Runtime Protocol in Phase 4

The default Phase 4 episode is:

### 10.1 Support

Learner studies support with CLS.

### 10.2 Observation

If enabled:

- inverse tutor:
  - uses `run_observation_phase_v2`
  - result-level observations only
  - updates `TutorLearnerModel`
- baselines:
  - use `run_observation_phase`
  - process-level stats summarized into `TutorBelief`

### 10.3 Teach

For each teach query:

- learner predicts `Y*`
- learner selects balls using current grammar + risk belief
- tutor can warn on selection
- learner updates danger belief from warnings / safe observations / death
- learner may confirm
- on failed confirm:
  - tutor may hint
  - learner may apply grammar feedback update

For inverse tutor specifically:

- per-query process-level tutor decisions happen online
- after the query finishes, `update_after_query(...)` updates the tutor's learner model from the final outcome and submission

### 10.4 Eval

Eval is frozen:

- no tutor
- no grammar feedback update

This makes teach-time improvements easier to attribute to training rather than online eval assistance.

---

## 11. What Is Mainline vs Experimental

### 11.1 Mainline right now

These are clearly wired into the current main runner:

- `run_phase4.py`
- `T3_traceInfer`
- grammar update from failed confirm via `FeedbackUpdater`
- risk update from warning, safe observation, and death
- inverse tutor hint gating and position scoring

### 11.2 Present but not mainline-wired

These are real code but not part of the default `phase4` execution path:

- `learner/correct_update.py`
- `learner/goal_b_state.py`
- `learner/hint_reliance.py`
- `TargetPredictor.predict_target_with_hint(...)`
- `TargetPredictor.beam_posterior_with_hint(...)`

These belong to Goal B or hint-dependency / over-help investigation branches.

So the repo currently has a substantial experimental periphery around the mainline.

---

## 12. Current System Status

Based on code rather than claims in docs, the current system status is:

### 12.1 Strongly implemented

- grammar-plus-danger environment
- CLS learner wrapper
- Bayesian danger posterior
- selection/confirm loop
- failed-confirm grammar update
- tutor family hierarchy:
  - rule
  - proxy
  - shadow
  - behavioral
  - inverse
- Phase 4 inverse tutor benchmark

### 12.2 Clearly transitional

- older tutor belief line and newer inverse line coexist
- Phase 3 behavioral tutor still exists and is still runnable
- Phase 4 is the latest mainline, but baselines still reuse earlier machinery

### 12.3 Research branches present in code

- Goal B positive-correct-update experiments
- hint-aware inference and autonomy experiments
- persistent hint trust / reliance
- counterfactual learning-loss checks

These are useful and nontrivial, but not yet fully normalized into the default runner.

---

## 13. Code-Grounded Defects, Mismatches, and Limitations

This section is deliberately limited to issues that are directly visible from the code.

### 13.1 `TutorConfig.tutor_policy_mode` comment is stale

In `config.py`, the comment says:

- `'rule' | 'proxy' | 'short_rollout'`

But the actual current code uses at least:

- `rule`
- `proxy`
- `shadow`
- `inverse`
- `none`

So config comments lag the real system.

### 13.2 `risk_update_lr` exists but is not used in `DangerTypeBelief`

`LearnerConfig` defines:

- `risk_update_lr`

But the current danger belief updates in `risk_belief.py` perform direct prototype moment updates and do not appear to use this parameter.

So this is currently a misleading or vestigial config field.

### 13.3 `enable_hint_bias` exists, but hint-aware target prediction is not wired into `run_phase4.py`

The following methods exist:

- `beam_posterior_with_hint(...)`
- `predict_target_with_hint(...)`

But repo-wide search shows no standard runtime call path from the main Phase 4 loop into those methods.

Therefore:

> hint-aware target prediction is implemented as a code branch, not as current mainline behavior.

### 13.4 `hint_target_flipped` field exists but is not used in the main runner

`QueryState` includes:

- `hint_target_flipped`

But current package search shows it is not actively updated in the standard runtime.

So the state is ahead of the mainline plumbing.

### 13.5 `hint_stop_shift` exists in config but is not implemented in policy

`LearnerConfig` includes:

- `hint_stop_shift`

But `learner/policy.py` only applies:

- confirm threshold shift
- exploration drop

It does not actually use `hint_stop_shift` to alter the selection stop criterion.

### 13.6 `HintRelianceState` exists but is not integrated into the mainline episode loop

`learner/hint_reliance.py` is a concrete implementation, but current package search shows no runtime integration into the standard Phase 4 runner.

So persistent trust/reliance is presently an experimental branch.

### 13.7 `correct_update.py` and Goal B state are experimental, not standard learner logic

`learner/correct_update.py` and `learner/goal_b_state.py` exist and are fairly substantial, but they are not invoked from the default `run_phase4.py` query loop.

So any statement like "the learner now learns from correct answers explicitly" would be false for the current mainline unless a specialized experimental runner is used.

### 13.8 Observation protocol is asymmetric across tutor families

Baselines use:

- `observation.py`
  - process-level summarized observation

Inverse tutor uses:

- `observation_v2.py`
  - result-level only

This asymmetry is intentional, but it is still a benchmark caveat:

> the baseline families and inverse tutor do not start from identical observation interfaces.

### 13.9 Current hint action is still structurally coarse

In `action_generators.py`, a hint is ultimately:

- direct insertion of correct `(position, color)` pairs into `state.completion`

So even though the inverse tutor computes sophisticated hint utilities and structural salience, the delivered action is still coarse.

This is a real mechanism bottleneck visible in the code:

- complex tutor inference
- simple intervention channel

### 13.10 Tutor danger handling is still largely oracle-clean

In the current inverse mainline:

- warning is an oracle safety gate

So inverse tutor quality mostly affects hint decisions, not the risk-warning channel.

This is not a bug, but it is a limitation on what Phase 4 can currently demonstrate about learner-aware risk tutoring.

### 13.11 Mainline learner still learns much more from failure than from success

Code-wise, this is one of the biggest conceptual asymmetries in the system.

Standard mainline:

- failed confirm -> explicit grammar update
- successful confirm -> no symmetric positive grammar update

That asymmetry is visible directly in the runner and helps explain why Goal B needed separate experimental code.

---

## 14. Bottom-Line Interpretation

If someone asks "what system does `cls_color_selection` currently implement?", the most accurate short answer is:

> It is a color-sequence tutoring environment where a CLS grammar learner and a Bayesian danger learner interact inside a query loop with selection, warning, confirm, and feedback; the latest mainline tutor is an inverse-inference tutor that knows the task ground truth but tries to infer the learner's grammar state from result-level behavior, while the learner's main online grammar update still comes primarily from failed confirm feedback.

If someone asks "what is solid vs speculative?", the answer is:

Solid and current:

- Phase 4 inverse tutor benchmark
- wrong-confirm grammar learning
- Bayesian danger belief
- oracle warning tutoring
- generated-query teach/eval protocol

Present but not mainline:

- hint-aware target prediction
- persistent hint trust/reliance
- explicit correct-answer grammar updates
- Goal B constrained-correct protocols

---

## 15. Suggested Reading Order

For future work on this repo, the highest-signal reading path is:

1. `config.py`
2. `environment/grammar_task_env.py`
3. `environment/transition.py`
4. `learner/cls_wrapper.py`
5. `learner/risk_belief.py`
6. `learner/feedback_update.py`
7. `learner/policy.py`
8. `tutor_api/task_model.py`
9. `tutor_api/learner_model.py`
10. `tutor_api/hint_policy.py`
11. `tutor_api/tutor_inverse.py`
12. `experiments/registry_phase4.py`
13. `experiments/run_phase4.py`

That sequence will get a new reader to the actual current mainline faster than starting from older phase files or older reports.
