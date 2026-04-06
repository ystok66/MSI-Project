# Step 7 Tutor Mechanism Report — Complete Teacher Architecture & Formulas

> **Handoff document** — self-contained reference for the pedagogical tutor (robot) side.
> Covers: two-layer architecture, macro intervention policy, micro internalization tutor,
> observer, posterior inference, RSA warning, behavior bridge, bottleneck diagnosis,
> perceptual model, and all decision formulas.

---

## 1. Two-Layer Tutor Architecture

The tutor operates at two distinct planning levels:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MACRO LAYER (Planner-level)                       │
│                                                                      │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────────────────┐ │
│  │ Robot Belief   │  │ Agent         │  │ Intervention Policy      │ │
│  │ (surrogate)    │  │ Predictor     │  │ {WAIT, WARN, UNLOCK,     │ │
│  │  b̃_A           │  │ P̂(a|s,b)     │  │  ITEM_DROP}              │ │
│  └───────┬───────┘  └───────┬───────┘  └──────────┬───────────────┘ │
│          └──────────────────┴──────────────────────┘                 │
│                              │                                       │
│  ┌───────────────────────────┴────────────────────────────────────┐  │
│  │  Bottleneck Diagnosis          Perceptual Model                │  │
│  │  (epistemic/structural/outcome)  ρ_i = P(seen by agent)        │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  Goal-Conditional Curriculum Hook                              │   │
│  │  S_macro(ℓ) = E_{q(g,θ)} [Q_online(ℓ|g,θ)] + β_κ·g(κ̂)       │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    MICRO LAYER (Internalization-level)                │
│                                                                      │
│  ┌────────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ 5D Observer     │  │ Joint        │  │ BCICTv4 Micro Tutor      │ │
│  │ m̂_t=(τ̂,ν̂,γ̂_gen)│ │ Posterior    │  │ {WAIT, SOFT, WARN}       │ │
│  │                 │  │ q(g,θ,z)     │  │ dose ∈ {0, 0.5, 1.0}    │ │
│  └────────┬────────┘  └──────┬───────┘  └──────────┬───────────────┘ │
│           └─────────────────┬┴─────────────────────┘                 │
│                             │                                        │
│  ┌──────────────────────────┴────────────────────────────────────┐   │
│  │  Behavior Bridge: m → ẑ (probe predictions)                   │   │
│  │  RSA Warning Channel: u → S₁(u|r,c) → b⁺(r)                  │   │
│  │  Intervention Semantics: WARN/UNLOCK/ITEM_DROP formal effects  │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Macro Layer: Intervention Policy

**Source**: `src/teachers/intervention_policy.py` (430 lines)

### 2.1 Action Space

```
A_macro = {WAIT, WARN, UNLOCK, ITEM_DROP}
```

| Action | Mechanism | Target |
|:-------|:----------|:-------|
| WAIT | No intervention | Let agent learn on its own |
| WARN | Belief evidence | Updates agent's risk belief, NOT world topology |
| UNLOCK | Affordance reveal | Changes world topology (opens locked doors), NOT risk |
| ITEM_DROP | Traversal mitigation | Drops shield (halves risk cost), NOT belief |

### 2.2 Counterfactual Scoring

Each action is scored via counterfactual surrogate rollout:

```
For each action a ∈ A_macro:
  1. Simulate: pred_a = predict_agent_prefix_after_a(...)
  2. Compute: risk_a, cost_a, failure_modes_a
  3. Score: Q(a) = benefit(a) − cost(a) + bonus(a)
```

**WAIT score**:

```
Q_wait = λ_learn · learning_gain − λ_cat · wait_risk − λ_dead · deadline_miss
```

| Weight | Symbol | Default | Meaning |
|:-------|:------:|:-------:|:--------|
| learning_gain | λ_learn | 1.0 | How much agent would learn from exploring |
| catastrophe | λ_cat | 5.0 | Risk of catastrophe without intervention |
| deadline | λ_dead | 2.0 | Deadline miss probability |

**WARN score**:

```
Q_warn = λ_warn · max(0, wait_risk − warn_risk) − λ_auto
```

| Weight | Symbol | Default | Meaning |
|:-------|:------:|:-------:|:--------|
| warn_effect | λ_warn | 3.0 | Catastrophe reduction from warning |
| autonomy | λ_auto | 1.0 | Penalty for intervening (preserving autonomy) |

**UNLOCK score**:

```
Q_unlock = λ_unlock · (cat_reduction + topology_improvement · 0.1) − λ_auto
```

**ITEM_DROP score**:

```
Q_item = λ_item · max(0, wait_risk − item_risk) − λ_item_cost
```

| Weight | Symbol | Default | Meaning |
|:-------|:------:|:-------:|:--------|
| item_drop | λ_item | 3.0 | Risk reduction from shield |
| item_cost | λ_item_cost | 1.5 | Higher than autonomy: items are expensive |

### 2.3 Phase 10: Bottleneck-Augmented Scoring

After computing raw Q-values, three adjustments are applied:

**1. Bottleneck-Intervention Matching Bonus**:

```
Q'(a) = Q(a) + β_b · M(a, bottleneck)
```

Where M maps each action to its natural bottleneck lever:
- WARN ↔ epistemic
- UNLOCK ↔ structural  
- ITEM_DROP ↔ outcome

**2. Redundancy Penalty (WARN only)**:

```
Q'(WARN) -= β_R · R_warn

R_warn = (1/|D|) Σ_{i∈D} ρ_i · exp(−u_r_i / τ_u)
```

High redundancy → agent already knows about risky cells → warning is wasted.

**3. Warn Damping (outcome-dominant)**:

When the dominant bottleneck is `outcome` (not `epistemic`), WARN is damped:

```
Q'(WARN) -= min(2.0, outcome_dominance · 0.5)
```

**4. Repeat Penalty**:

```
Q'(WARN) -= n_warns_before · 0.5
Q'(UNLOCK) -= 2·λ_auto    [if no locked doors remain]
```

---

## 3. Robot Belief (Surrogate Model)

**Source**: `src/teachers/robot_belief.py` (161 lines)

### 3.1 Role

The robot maintains an **approximate copy** of the agent's internal state — NOT full Bayesian nested inference. This enables counterfactual rollouts without reading hidden agent internals.

### 3.2 State

```python
RobotBelief:
  agent_belief_mean: ndarray  (H, W, d)   # estimated agent belief over features
  agent_belief_var:  ndarray  (H, W, d)   # estimated agent uncertainty
  # + surrogate competence parameters:
  agent_search_budget: int = 30
  agent_risk_weight: float = 3.0
  agent_uncertainty_weight: float = 0.5
  # + read-only snapshot of agent's predictor weights:
  _predictor_cost_w, _predictor_cost_b
  _predictor_risk_w, _predictor_risk_b
```

### 3.3 Copy Modes

| Mode | Mechanism | Fidelity |
|:-----|:----------|:---------|
| `exact` | Full copy every step | Perfect surrogate |
| `noisy` | Full copy + Gaussian noise (σ = 0.05) | Perturbed surrogate |
| `stale` | Sync every N steps only | Delayed surrogate |

### 3.4 Competence Mismatch

The robot can deliberately mis-estimate agent competence:

```
surrogate_budget = true_budget + budget_mismatch
surrogate_risk_weight = true_risk_weight + risk_weight_mismatch
```

This models the robot's uncertainty about how good the agent's planner is.

---

## 4. Agent Predictor

**Source**: `src/teachers/agent_predictor.py` (209 lines)

### 4.1 Purpose

Predicts the agent's likely behavior using the robot's surrogate model. All operations are **read-only** — never mutates real agent or environment state.

### 4.2 Prediction Pipeline

```
predict_agent_prefix(rb, pos, goal, ...) →
  1. Build surrogate predictor from RobotBelief snapshot
  2. Run plan_from_belief() using surrogate weights
  3. Run plan_with_alternatives_v2() for candidate scores
  4. Compute failure modes: (catastrophe, deadline_miss, suboptimality)
  → AgentPrediction(plan, failure_modes, candidate_scores)
```

### 4.3 Counterfactual Variants

| Variant | Modification | What changes |
|:--------|:-------------|:-------------|
| `predict_after_warn` | Add `warn_extra_cost` to risky cells | Agent avoids warned cells |
| `predict_after_unlock` | Set locked cells to passable | Agent can traverse doors |
| `predict_after_item_drop` | Clone inventory + add shield | Agent's traversal risk reduced |

### 4.4 Learning Gain Estimation

```
learning_gain = mean(uncertainty(cell)) for cell in predicted_prefix
```

Higher uncertainty along the predicted path → more the agent would learn by exploring → stronger case for WAIT.

---

## 5. Bottleneck Diagnosis

**Source**: `src/teachers/bottleneck_diagnosis.py` (146 lines)

### 5.1 Three-Way Classification

| Bottleneck | Score Formula | Natural Lever |
|:-----------|:-------------|:--------------|
| **Epistemic** | s_epi = U_D · (1 + max(0, q_warn − q_wait)) | WARN |
| **Structural** | s_str = g_t · (1 + max(0, q_unlock − q_wait)) | UNLOCK |
| **Outcome** | s_out = min_path_risk · (1 + max(0, q_item − q_wait)) | ITEM_DROP |

Where:

**Epistemic severity** (decision-relevant uncertainty):

```
U_D = (1/|D|) Σ_{i∈D} [ω_ρ · (1 − ρ_i) + ω_u · u_r_i]
```

- `ρ_i` = probability agent has seen cell i (from perceptual model)
- `u_r_i` = risk prediction uncertainty at cell i

**Structural urgency** (time pressure):

```
g_t = exp(−slack / τ_s)    where slack = (t_max − t) − shortest_path_len
```

- If `slack ≤ 0` or locked doors exist: `g_t = 1.0` (critical)
- `τ_s = 3.0` (temperature controlling urgency decay)

**Outcome severity**: Direct `min_path_risk` — unavoidable risk on any feasible path.

### 5.2 Dominant Bottleneck

```python
dominant = argmax(epistemic, structural, outcome)
```

The dominant bottleneck determines which intervention family should be preferred via the matching bonus (§2.3).

---

## 6. Tutor Perceptual Model

**Source**: `src/teachers/perceptual_model.py` (143 lines)

### 6.1 Core State Variable

The tutor tracks what the agent has **effectively seen**:

```
ρ_{i,t} = P(agent has effectively seen cell i by time t)
```

### 6.2 Update Rule (monotonically non-decreasing)

```
ρ_{i,t+1} = 1 − (1 − ρ_{i,t}) · (1 − p_see_{i,t+1})
```

Where:

```
p_see(i) = [d(i, agent) ≤ r_patch] · exp(−λ_d · d) · q_obs

q_obs = 1 / (1 + σ²_obs)

σ²_obs = {
  0.01   if d = 0  (visiting = perfect observation)
  0.08·d if d > 0  (distance-decayed)
}
```

| Parameter | Symbol | Default | Meaning |
|:----------|:------:|:-------:|:--------|
| Patch radius | r_patch | 2 | Manhattan distance limit |
| Distance decay | λ_d | 0.8 | Exponential decay |
| Self variance | σ²_self | 0.01 | Noise at distance 0 |
| Neighbor variance | σ²_nbr | 0.08 | Noise per unit distance |

### 6.3 Redundancy Computation

```
R_warn = (1/|D|) Σ_{i∈D} ρ_i · exp(−u_r_i / τ_u)
```

- High ρ_i → agent has already seen cell i → less value in warning about it
- High u_r_i → agent is uncertain → more value in warning despite having seen it
- τ_u = 0.3 controls the uncertainty-discount strength

---

## 7. Intervention Semantics (Formal Effects)

**Source**: `src/teachers/intervention_semantics.py` (222 lines)

### 7.1 WARN = Belief Evidence

```
μ⁺_i = μ_i + α_warn · v̂_warn     where v̂ = warn_direction / ||warn_direction||
Σ⁺_i = (1 − β_warn) · Σ_i

α_warn = 0.3    (belief shift magnitude)
β_warn = 0.2    (uncertainty reduction fraction)
```

**Invariant**: WARN **never** changes world topology. It only updates the agent's belief.

### 7.2 UNLOCK = Affordance Reveal

```
s^world_{t+1} = Unlock(s^world_t)     [door cells become passable]
b^{A,env}_{t+1} = AffordanceReveal(b^{A,env}_t, s^world_{t+1})

σ²_{unlocked} *= (1 − 0.3)            [uncertainty reduced on newly accessible cells]
```

**Invariant**: UNLOCK **never** changes risk values. It only changes passability.

### 7.3 ITEM_DROP = Traversal Mitigation

```
TraversalCost^shield(i) = λ_r · (1 − γ_shield) · φ(r̂_i)

where φ(r) = −ln(1 − r)    [risk-to-cost transform]
```

Default: `γ_shield = 0.5` (shield halves risk cost).

**Invariant**: ITEM_DROP **never** changes agent belief or world topology.

### 7.4 Semantic Invariant Summary

| Intervention | Modifies Belief | Modifies Topology | Modifies Risk |
|:-------------|:---------------:|:-----------------:|:-------------:|
| WARN | ✅ μ, Σ | ❌ | ❌ |
| UNLOCK | ❌ | ✅ passable | ❌ |
| ITEM_DROP | ❌ | ❌ | ✅ traversal cost |

---

## 8. Micro Layer: BCICTv4 Internalization Tutor

**Source**: `src/teachers/internalization_control_tutor_v4.py` (456 lines)

### 8.1 Action Space

```
A_micro = {WAIT, SOFT, WARN}    (dose = {0.0, 0.5, 1.0})
```

### 8.2 Decision Function

```
Q(dose) = Q_online(dose) + λ_teach · V_full(dose) − λ_over · R_over(dose)
```

The tutor evaluates all three doses and picks argmax Q.

### 8.3 Q_online (Scene-Based Value)

**For WARN (dose=1.0)**:

```
Q_online_warn = 1.0·Δs + 2.0·ΔVOI + 1.5·(1 − p_self) + 1.0·tempt − 0.05
```

**For WAIT (dose=0.0)**:

```
Q_online_wait = 2.0·p_self·Δs − 1.5·p_fail + 2.0
```

**For SOFT (dose=0.5)**:

```
Q_online_soft = 0.5·Q_warn + 0.5·Q_wait
```

| Symbol | Meaning |
|:-------|:--------|
| Δs | Information gain: full-view safety difference minus partial-view |
| ΔVOI | Value of information: σ(Δs_full) − σ(Δs_partial) |
| p_self | Self-discovery probability: σ((d_commit − d_reveal) / τ_v) |
| p_fail | Failure-if-wait: 1 − p_self |
| tempt | Temptation strength of current scene |

### 8.4 V_full (Teaching Value)

```
V_full = V + λ_sd · p_sd − λ_dep · p_blind
```

Where:
- **V = L_now − L_next**: Behavior loss reduction from intervention
- **p_sd**: Probability of self-discovery if we wait
  ```
  p_sd = p_self · (0.8 if sd_subtype else 0.4) · (1 − dose)
  ```
- **p_blind**: Probability of blind obedience if we warn
  ```
  p_blind = (0.7 if no_self_evidence else 0.2) · dose
  ```

| Weight | Symbol | Default | Meaning |
|:-------|:------:|:-------:|:--------|
| Teaching | λ_teach | 3.5 | Weight on teaching value |
| Overteach | λ_over | 4.0 | Weight on overteaching penalty |
| Self-discovery | λ_sd | 1.5 | Bonus for self-discovery |
| Dependence | λ_dep | 2.0 | Penalty for blind obedience |

### 8.5 One-Step Rollout (m → m')

For each dose, the tutor simulates the post-intervention internalization state:

```python
m' = _predict_m(m, dose, tempt, risk, subtype, has_self_evidence)
```

**If dose > 0 (warning)**:
1. `update_risk(0.05, 0.15)` — risk prediction adjustment
2. `update_trust(warn_helpful=(risk > 0.25))` — trust from correct warning
3. If no self-evidence: `update_dependence(blind_obey=True)` — dependence risk
4. `update_gamma_gen(sustained_pressure=True)` — suppression from sustained warning
5. If dose < 1.0 (SOFT): linear interpolation between no-warn and full-warn effects

**If dose = 0 (wait)**:
1. `update_risk(risk, 0.15)` — natural risk learning
2. If self-discovery subtype with evidence: `update_dependence(self_discovery=True)`
3. If tempt > 0.5 and risk > 0.3: `update_gamma_spec(tempt_error=True)`
4. If novel or self-evidence: `update_gamma_gen(successful_exploration=True)`

### 8.6 Behavior Bridge (m → probes)

**Source**: `src/agents/behavior_bridge.py` (115 lines)

Maps internalization state to observable behavior via semi-parametric logistic bridge:

```
ẑ_p(m, c) = σ(w_p · φ(m, c))
```

**Feature vector** φ(m, c) — 13 dimensions:

```
φ = [1, κ, τ, ν, γ_spec, γ_gen,     # raw state (6)
     κ·risk, γ_spec·lure, γ_gen·novelty,  # state×context (3)
     τ·(1−self_ev), ν·self_ev,       # trust/dep×evidence (2)
     κ², γ_gen²]                     # quadratic terms (2)
```

### 8.7 Five Behavior Probes

| Probe | Name | Bridge Drives | Healthy Range |
|:------|:-----|:--------------|:--------------|
| RC | Risk Calibration | κ, κ·risk, κ² | Band around calibrated |
| TR | Temptation Resistance | γ_spec, γ_spec·lure | Band around calibrated |
| EP | Exploration Preservation | γ_gen, γ_gen·novelty, γ_gen² | High (agent still explores) |
| VA | Valid-Advice uptake | τ, τ·(1−self_ev), ν | High (follows good advice) |
| IA | Invalid-Advice rejection | ν, ν·self_ev | High (rejects bad advice) |

### 8.8 Behavior Loss (L_beh) and Overteach Penalty (R_over)

**Behavior Loss**:

```
L_beh(m, z) = Σ_p w_p · max(0, z_lo − ẑ_p)² + max(0, ẑ_p − z_hi)²
```

Probe weights: RC=1.0, TR=1.2, EP=2.5, VA=1.5, IA=2.5.

**Overteach Penalty**:

```
R_over = 2.5·max(0, ẑ_IA − z_IA_hi)² + 2.5·max(0, z_EP_lo − ẑ_EP)² + 1.5·max(0, ẑ_TR − z_TR_hi)²
```

Penalizes: high IA (too compliant), low EP (suppressed exploration), high TR (overly inhibited).

### 8.9 Empirical Zone Calibration

Zones are calibrated from rolling data using quantiles:

```
z_lo = quantile(ẑ_history, α)      α = 0.15
z_hi = quantile(ẑ_history, 1−α)
```

Minimum zone width: 0.1 (prevents degenerate thresholds).

---

## 9. Shadow Modes (Research Branches)

BCICTv4 supports several shadow modes that override the canonical decision while logging diagnostics:

### 9.1 EPU Shadow (Phase 7 Refactor 1)

One-step rollout Expected Pedagogical Utility:

```
EU(dose) = V_true(π*(b^dose)) − Cost(dose)
V_true = 3.0·survival + 2.0·learning − 1.5·overteaching

survival = 1 − κ·risk·max(1 − dose·0.5, 0.3)
learning = 0.4·(1−ν) + 0.4·τ + 0.2·(1−γ_gen)
overteaching = max(ν + γ_gen − 1, 0)
```

### 9.2 Belief-Horizon p_self (Phase 7 Refactor 2)

Hybrid geometric + belief-based self-discovery probability:

```
p_self = (1−η)·p_geom + η·p_belief

p_belief = risk_awareness · update_gain · info_window
risk_awareness = min(κ·2, 1)
update_gain = max(1−ν, 0.1)
info_window = max(d_commit − d_reveal, 0) / max(d_commit, 1)
```

η = 0.5 (mixing weight).

### 9.3 EIG Observation (Phase 7 Refactor 3)

Mutual information between action and latent preference:

```
I(A; θ) = Σ_{a,θ} P(a,θ)·log[P(a,θ) / (P(a)·P(θ))]
```

Computed with θ ∈ {safe, shiny}. Higher I → agent's next action is more informative → stronger case for WAIT (observe).

### 9.4 Micro-Bayes Shadow (Steps 1–4)

Research alternative decision engines with increasing sophistication:

| Version | Key Feature | Status |
|:--------|:-----------|:-------|
| v1 `micro_bayes_shadow` | Basic Bayesian scoring | Research |
| v2 `micro_bayes_shadow_v2` | Conservative-gated | Research |
| v2.1 `micro_bayes_shadow_v2_1` | Converged recommended | Research |
| v3 `micro_bayes_shadow_v3` | Causal-dependence model | Research |

### 9.5 p_self Posterior Shadow (Steps 1–4)

Alternative p_self computations via posterior inference:

| Mode | Mechanism |
|:-----|:----------|
| `old_blend` | Legacy geometric+model blend |
| `posterior_A` | Bayesian posterior A (basic) |
| `posterior_B` | Bayesian posterior B (refined) |
| `posterior_C` | Three-outcome (self/fail/undecided) |

---

## 10. Goal-Conditional Curriculum Hook

**Source**: `src/teachers/goal_conditional_curriculum_hook.py` (168 lines)

### 10.1 Macro Score

```
S_macro(option) = E_{q(g,θ)} [success_lift(option | g, θ)]
                 + λ_teach · teaching_value · 𝟙[option ≠ NONE]
                 − λ_infl · inflation_cost
                 + β_κ · g_κ(κ̂) · scaling_factor
```

| Weight | Symbol | Default | Meaning |
|:-------|:------:|:-------:|:--------|
| Teaching value | λ_teach | 0.5 | Entropy-based teaching incentive |
| Inflation penalty | λ_infl | 4.0 | Dependence accumulation cost |
| κ̂ bonus | β_κ | 0.02 | Risk sensitivity macro bonus |
| Min confidence | min_conf | 0.15 | Posterior confidence threshold |

### 10.2 Success Lift Under Posterior

For each (goal, preference) hypothesis:

```
lift(option|g,θ) = P(safe_branch | branches_after_option, g, θ)
                 − P(safe_branch | branches_before, g, θ)
```

Weighted by `q(g)·q(θ)` to get expected lift.

### 10.3 Teaching Value

```
teaching_value = H(q) / H_max    ∈ [0, 1]
```

Higher posterior entropy → more value in intervening (more uncertain about goal).

### 10.4 Inflation Cost

```
inflation_cost = max(0, ν̂) · 0.1    [if option ≠ NONE]
```

Prevents intervention when agent is already dependent.

### 10.5 κ̂ as Additive Macro State

> **RED LINE**: κ̂ enters as an additive term `β_κ · κ̂`, NOT as a posterior latent. It is NOT part of q(g,θ,z). This is a macro curriculum signal, not a belief state.

---

## 11. 5D Observer (Shadow State Estimator)

**Source**: `src/teachers/internalization_observer.py` (720 lines)

### 11.1 Observed Dimensions

The observer estimates 3 of the 5 internalization dimensions:

| Dimension | Observed | Primary Signal |
|:----------|:--------:|:---------------|
| τ̂ (trust) | ✅ | warn_correct, warn_wrong, probe_VA |
| ν̂ (dependence) | ✅ | blind_obey, self_discovery, probe_IA |
| γ̂_gen (suppression) | ✅ | pressure EMA, exploration success, probe_EP |
| κ̂ (risk sensitivity) | ❌ | NOT observed (macro state only) |
| γ̂_spec (temptation) | ❌ | NOT observed (implicit in behavior) |

### 11.2 Observer Parameters (Conservative)

| Param | Observer | Agent (True) | Ratio |
|:------|:--------:|:------------:|:-----:|
| α_τ⁺ | 0.22 | 0.25 | 0.88× |
| α_τ⁻ | 0.10 | 0.12 | 0.83× |
| α_ν⁺ | 0.18 | 0.20 | 0.90× |
| α_ν⁻ | 0.13 | 0.15 | 0.87× |
| α_γ⁺ | 0.07 | 0.08 | 0.88× |
| α_γ⁻ | 0.10 | 0.12 | 0.83× |

**Design**: Observer is systematically **conservative** (~85-90% of true rates) to avoid overconfident state estimation.

### 11.3 Probe Correction

When probes are available, the observer performs a correction step:

```
τ̂ += β_τ_probe · (probe_VA − τ̂)           β = 0.15
ν̂ += β_ν_probe · ((1 − probe_IA) − ν̂)     β = 0.10
γ̂ += β_γ_probe · ((1 − probe_EP) − γ̂)     β = 0.10
```

Probes are inverse: higher IA (invalid-advice rejection) → lower dependence, etc.

### 11.4 Confidence Tracking

Per-dimension confidence via EMA:

```
conf_dim = (1 − ρ)·conf_dim + ρ·eq_dim     ρ = 0.15
```

Where `eq_dim` is the evidence quality for that dimension:
- τ: max(e_trust+, e_trust−, has_probe_VA)
- ν: max(e_blind, e_selfdisc, has_probe_IA) + 0.2·|p_self − 0.5|·2
- γ: max(min(e_pressure, 1), e_explore+, has_probe_EP)

### 11.5 Shadow Decision

The observer can compute what BCICTv4 would decide using m̂ instead of m_true:

```python
a_infer, dose_infer, Q_infer = tutor.decide(..., m_hat_state)
```

Agreement rate between (a_oracle, a_infer) is the primary observer quality metric.

---

## 12. Decision Flow: Complete Per-Step Pipeline

```
┌─────────── Step t: Agent at position (r,c) ──────────────────────┐
│                                                                    │
│  1. Agent observes local features within patch radius              │
│  2. Agent's risk/cost heads predict feature → (ρ̂, ĉ)              │
│  3. Agent's planner computes A* paths with predicted costs         │
│  4. Agent selects branch: a ~ P(a|s,θ,m)                          │
│                                                                    │
│  ═══ Tutor Observes ═══════════════════════════════════            │
│                                                                    │
│  5. Perceptual model: update ρ(cell i) for all visible cells       │
│  6. Robot belief sync: copy/noisy/stale agent belief to surrogate  │
│  7. Posterior update: q(g,θ,z) ∝ q·P(a|g,θ,z)  [Bayesian]        │
│  8. Observer update: m̂ from (dose, follow, correct, p_self, etc)  │
│                                                                    │
│  ═══ Macro Decision ══════════════════════════════════════         │
│                                                                    │
│  9.  Agent predictor: predict_prefix_{wait, warn, unlock, item}   │
│  10. Bottleneck diagnosis: (epistemic, structural, outcome)        │
│  11. Score interventions: Q(WAIT), Q(WARN), Q(UNLOCK), Q(ITEM)   │
│  12. Select best macro action a_macro                              │
│                                                                    │
│  ═══ Micro Decision ══════════════════════════════════════         │
│                                                                    │
│  13. Compute p_self, p_fail, Δs, ΔVOI                             │
│  14. Bridge: predict probes ẑ_p from m̂                            │
│  15. One-step rollout: m → m' for each dose                       │
│  16. Behavior loss: L = L_now − L_next                            │
│  17. Overteach penalty: R_over from bridge predictions             │
│  18. Q(dose) = Q_online + λ·V_full − λ_over·R_over               │
│  19. Select best micro action + dose                               │
│                                                                    │
│  ═══ If Warning ══════════════════════════════════════════         │
│                                                                    │
│  20. RSA Channel: S₁ speaker selects utterance                     │
│  21. Agent belief update: b⁺(r) ∝ S₁(u|r,c)^η_τ · b⁻(r)         │
│  22. Planner adapter: Δρ = E[risk|b⁺] − E[risk|uniform]          │
│                                                                    │
│  ═══ Agent State Update ══════════════════════════════════         │
│                                                                    │
│  23. Internalization m_{t+1}: apply update rules to (κ,τ,ν,γs,γg) │
│  24. Risk head: online MAP update from traversal outcome           │
│  25. Agent moves to new position based on selected branch          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 13. Module Cross-Reference (Teacher Side)

| Mechanism | Module | Lines | Layer |
|:----------|:-------|:-----:|:-----:|
| Macro intervention scoring | `intervention_policy.py` | 430 | Macro |
| Robot belief surrogate | `robot_belief.py` | 161 | Macro |
| Agent prefix prediction | `agent_predictor.py` | 209 | Macro |
| Bottleneck diagnosis | `bottleneck_diagnosis.py` | 146 | Macro |
| Tutor perceptual model | `perceptual_model.py` | 143 | Macro |
| Goal-conditional curriculum | `goal_conditional_curriculum_hook.py` | 168 | Macro |
| Intervention semantics | `intervention_semantics.py` | 222 | Both |
| Micro tutor BCICTv4 | `internalization_control_tutor_v4.py` | 456 | Micro |
| 5D Observer | `internalization_observer.py` | 720 | Micro |
| Joint posterior q(g,θ,z) | `joint_goal_pref_posterior.py` | 329 | Micro |
| Structural prior P₀(g\|c₀) | `compositional_goal_prior.py` | 332 | Micro |
| RSA warning channel | `rsa_warning_channel.py` | 499 | Micro |
| Behavior bridge m→ẑ | `behavior_bridge.py` | 115 | Micro |
| Action predictor P(a\|s,b) | `action_predictor.py` | 123 | Both |
| Self-discovery p_self | `self_discovery.py` (metrics) | 73 | Micro |
| P_self posterior shadow | `p_self_posterior_shadow.py` | — | Shadow |
| Micro Bayes shadows v1–v3 | `micro_bayes_shadow*.py` | — | Shadow |

---

## 14. Key Design Principles

1. **Read-only counterfactuals**: All prediction/simulation is read-only. Never mutate real agent or env state during planning.
2. **Surrogate ≠ ground truth**: Robot belief is an *approximate* copy. Competence mismatch is a feature, not a bug.
3. **Conservative observer**: Observer rates are ~85-90% of true agent rates to avoid overconfident state estimation.
4. **Semantic invariants**: Each intervention type has a strict "what it does/doesn't change" contract (§7.4).
5. **Shadow-first research**: All experimental mechanisms run as shadow modes (logged, not canonically executed) until promotion.
6. **κ̂ is additive, not posterior**: Risk sensitivity is a macro curriculum signal, never enters q(g,θ,z).
7. **Overteaching is penalized**: R_over explicitly penalizes excessive compliance (IA↑), suppressed exploration (EP↓), and over-inhibition (TR↑).
