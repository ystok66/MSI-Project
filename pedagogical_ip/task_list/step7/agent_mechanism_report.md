# Step 7 Agent & Mechanism Report — Formulas, Architecture, and Decision Flow

> **Handoff document** — self-contained reference for the pedagogical tutor's complete mathematical framework.
> Covers: agent model, internalization dynamics, observer, posterior inference, RSA warning, and micro tutor decision.

---

## 1. System Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    TEACHER (Robot)                            │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  Observer    │  │  Posterior    │  │   Micro Tutor      │  │
│  │  m̂_t        │  │  q(g,θ,z)    │  │   BCICTv4          │  │
│  │  (τ̂,ν̂,γ̂)   │  │  Step 4      │  │   {WAIT, WARN}     │  │
│  └──────┬──────┘  └──────┬───────┘  └──────┬─────────────┘  │
│         │                │                  │                │
│         │         ┌──────┴───────┐   ┌──────┴──────┐        │
│         │         │ Goal Prior   │   │ RSA Warning │        │
│         │         │ P₀(g|c₀)    │   │ Channel     │        │
│         │         └──────────────┘   └─────────────┘        │
└─────────┼───────────────────────────────────────────────────┘
          │  observes
          ▼
┌──────────────────────────────────────────────────────────────┐
│                    AGENT (Learner)                            │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ Internal     │  │  Risk Head   │  │  Policy            │  │
│  │ State m_t    │  │  ρ̂=σ(w·z+b)  │  │  P(π|s,θ,m_t)     │  │
│  │ (κ,τ,ν,γs,γg)│ │  Bayesian    │  │  Softmax + lapse   │  │
│  └──────────────┘  └──────────────┘  └────────────────────┘  │
│                                                              │
│  Latent: θ ∈ {safe, shiny}, goal g, temptation z             │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Agent Policy — Bounded Rational Decision Making

**Source**: `src/agents/stochastic_agent_policy.py` (121 lines)

### 2.1 Base Policy (J1)

The agent selects branches via softmax + lapse noise, conditioned on latent preference θ:

$$P(\pi \mid s, \theta) = (1 - \varepsilon) \cdot \text{softmax}(\beta \cdot U(\pi; \theta)) + \varepsilon \cdot \text{Uniform}$$

where:

$$U(\pi \mid \theta) = R_{\text{goal}}(\pi) + \lambda_\theta \cdot R_{\text{pref}}(\pi; \theta) - J_{\text{risk}}(\pi)$$

### 2.2 Parameters

| Parameter | Symbol | Default | Meaning |
|:----------|:------:|:-------:|:--------|
| Temperature | β | 4.0 | Rationality (higher = more optimal) |
| Lapse rate | ε | 0.1 | Random exploration probability |
| Pref weight | λ_θ | 1.0 | How much preferences influence choice |

### 2.3 Preference Reward Weights

Each θ maps to a 4D reward weight vector `[safety, temptation, novelty, shortcut]`:

| θ Type | safety | tempt | novelty | shortcut | Behavioral Signature |
|:-------|:------:|:-----:|:-------:|:--------:|:---------------------|
| `safe` | +2.0 | −1.0 | 0.0 | 0.0 | Avoids temptation, prefers safe paths |
| `risky` | −0.5 | +0.5 | 0.0 | 0.0 | Slight risk-seeking |
| `shiny` | 0.0 | +3.0 | 0.0 | 0.0 | Strongly drawn to temptation |
| `shortcut` | 0.0 | 0.0 | 0.0 | +2.0 | Prefers faster paths |
| `neutral` | +0.3 | 0.0 | 0.0 | 0.0 | Mild safety preference |

**Canonical θ-types** (Θ₂): {safe, shiny}. Research Θ_K adds risky, shortcut, neutral.

### 2.4 Branch Attributes

Each branch is summarized as a `BranchAttributes` dataclass with 5 scalars:

```python
BranchAttributes(
    safety_score   : float  # higher = safer (derived from cell features)
    temptation_score: float # higher = more tempting (feature dim 1)
    texture_novelty: float  # novelty indicator
    shortcut_bonus : float  # speed advantage
    risk_penalty   : float  # cumulative risk along branch
)
```

The reward is computed as:

```
R_pref(π; θ) = PREF_REWARD[θ] · [safety, temptation, novelty, shortcut]ᵀ
```

### 2.5 Likelihood Function (for posterior update)

```python
P(chose branch_i | θ, params) = compute_choice_probs()[i]
```

This is the observation model that drives Bayesian inference in the posterior.

---

## 3. Internalization State — 5D Latent Dynamics

**Source**: `src/agents/internalization_state_v3.py` (217 lines)

### 3.1 State Vector

The agent has a hidden 5D internal state $m_t = (\kappa, \tau, \nu, \gamma_{\text{spec}}, \gamma_{\text{gen}})$:

| Dim | Symbol | Range | Init | Meaning | Healthy Direction |
|:---:|:------:|:-----:|:----:|:--------|:------------------|
| 0 | κ | [0.3, 3.0] | 1.0 | Risk sensitivity | Moderate (regression to κ₀) |
| 1 | τ | [0, 1] | 0.3 | Tutor trust | High ↑ (quality-driven) |
| 2 | ν | [0, 0.8] | 0.1 | Tutor dependence | Low ↓ (blind compliance) |
| 3 | γ_spec | [0, 0.7] | 0.0 | Temptation-specific suppression | Moderate |
| 4 | γ_gen | [0, 0.5] | 0.0 | General exploration suppression | Low ↓ (overteaching marker) |

### 3.2 Update Equations

**κ (risk sensitivity)** — regression-to-baseline with error correction:

```
κ_{t+1} = clip[(1 - β_κ)·κ_t + β_κ·κ₀ + α_κ·(r_real - r_expected), κ_min, κ_max]
```

| Param | Value | Meaning |
|:------|:-----:|:--------|
| β_κ | 0.08 | Regression rate to κ₀ |
| α_κ | 0.40 | Prediction error learning rate |
| κ₀ | 1.0 | Baseline risk sensitivity |

**τ (trust)** — asymmetric EMA driven by warning quality:

```
if warn_helpful:  τ += α_τ⁺ · (1 - τ)      # toward 1.0
if warn_bad:      τ -= α_τ⁻ · τ              # toward 0.0
```

| α_τ⁺ | 0.25 | Trust gain from correct warning |
|:------|:----:|:------|
| α_τ⁻ | 0.12 | Trust loss from incorrect warning |

**ν (dependence)** — blind obedience vs self-discovery:

```
if blind_obey:     ν += α_ν⁺ · (1 - ν)
if self_discovery: ν -= α_ν⁻ · ν
```

| α_ν⁺ | 0.20 | Dependence gain from blind compliance |
|:------|:----:|:------|
| α_ν⁻ | 0.15 | Dependence reduction from self-discovery |

**γ_spec (temptation suppression)** — learned from temptation errors:

```
if tempt_error:       γ_spec += α_gs⁺ · (1 - γ_spec)
if false_suppression: γ_spec -= α_gs⁻ · γ_spec
```

**γ_gen (exploration suppression)** — overteaching indicator:

```
if sustained_pressure:     γ_gen += α_gg⁺ · (1 - γ_gen)
if successful_exploration: γ_gen -= α_gg⁻ · γ_gen
```

### 3.3 Factored Utility (with internalization state)

When the agent uses its internal state for decision-making:

```
U(π|θ,m) = λ_θ·R_pref(π;θ) - κ²·risk_penalty - γ_spec·temptation
           - γ_gen·(0.3 if novel else 0) + τ·warn_bonus - ν·(0.2 if warned else 0)
```

**B2 Extension (epistemic risk shaping)**:

When `use_epistemic_risk=True`, the risk term becomes uncertainty-modulated:

```
α = α_min + (1 - α_min)·exp(-γ_epi · ũ_r)
risk_term = κ² · (ρ + (1-ρ)·α) · risk_penalty
```

where `ũ_r = clip(risk_unc / u_ref, 0, 1)`, `α_min = 0.25`, `ρ = 0.35`, `γ_epi = 3.0`.

This guarantees a **risk floor** of `κ²·α_min·ρ·risk_penalty` while attenuating risk penalty in high-uncertainty regions.

---

## 4. Risk Learning — Bayesian Linear Models

**Source**: `src/agents/risk_model.py` (127 lines) + `src/agents/cost_risk_model.py` (237 lines)

### 4.1 Risk Head

```
ρ̂ = σ(w_r · z + b_r)     where σ = sigmoid
```

- **Prior**: w_r ~ N(0, I/prior_var)
- **Update**: Online MAP via SGD on negative log-posterior
- **Gradient**: ∇_w NLL = -(y - ρ̂)·z + w/prior_var
- **Gradient clipping**: ||∇w|| ≤ 5.0, ||w|| ≤ 10.0

### 4.2 Cost Head

```
ĉ = max(w_c · z + b_c, 0.1)    (Gaussian likelihood)
```

Same MAP update structure as risk head but with Gaussian likelihood.

### 4.3 Predictive Uncertainty (Laplace Approximation)

```
Var(ρ̂|z) ≈ ρ̂(1-ρ̂) · (1 + z^T H^{-1} z)
where H = X^T X / n + I/prior_var    (empirical Hessian)
```

### 4.4 WorldWeights (Environment-Side)

The **true** cost/risk mapping (known to environment, hidden from agent):

```python
w_risk[2] ~ Uniform(2.0, 4.0)   # texture_1: strong positive
w_risk[3] ~ Uniform(1.5, 3.5)   # texture_2: strong positive
w_risk[0] ~ Uniform(-0.5, 0.5)  # lane_id: mild
w_risk[1] ~ Uniform(-0.3, 0.3)  # gate: mild
b_risk    ~ Uniform(-3.0, -1.5) # bias: keeps most cells low-risk
```

**Design intent**: Texture features (dims 2,3) are the primary risk drivers. The agent must learn this from experience.

---

## 5. Observer — Shadow State Estimation

**Source**: `src/teachers/internalization_observer.py` (720 lines)

### 5.1 Role

The observer estimates the agent's latent state $\hat{m}_t = (\hat{\tau}, \hat{\nu}, \hat{\gamma}_{gen})$ using **only tutor-visible signals**. It does NOT read `m_t` directly. This is a shadow module that mirrors agent update mechanics externally.

> **Note**: κ and γ_spec are NOT observed — only (τ, ν, γ_gen).

### 5.2 Soft Events (from ObsEvent)

| Event | Formula | Meaning |
|:------|:--------|:--------|
| e_trust+ | warned ∧ followed ∧ correct | Warning was correct AND followed |
| e_trust− | warned ∧ followed ∧ wrong | Warning was wrong AND followed |
| e_blind | warned ∧ followed · (1 − p_self) | Followed without own evidence |
| e_selfdisc | self_discovery · p_self | Self-discovered without tutor |
| e_pressure | EMA(dose, α=0.3) | Sustained warning pressure |
| e_explore+ | beneficial_novelty ∨ self_discovery | Successful exploration |

### 5.3 Update Equations (with probe correction)

**τ̂ (trust estimate)**:

```
τ̂ₙ = (1-λ_τ)·τ̂ + λ_τ·τ₀       # mean reversion
    + α_τ⁺·e_trust+ · (1-τ̂)      # up from correct warning
    - α_τ⁻·e_trust− · τ̂          # down from bad warning
    + β_τ·(probe_VA − τ̂)         # probe correction
```

| Observer Param | Value | vs Agent |
|:---------------|:-----:|:--------:|
| α_τ⁺ | 0.22 | < 0.25 (conservative) |
| α_τ⁻ | 0.10 | < 0.12 (conservative) |
| β_τ (probe) | 0.15 | Observer-only |
| λ_τ (reversion) | 0.02 | Observer-only |

**ν̂ (dependence estimate)**:

```
ν̂ₙ = (1-λ_ν)·ν̂ + λ_ν·ν₀
    + α_ν⁺·e_blind · (ν_max - ν̂)
    - α_ν⁻·e_selfdisc · ν̂
    + β_ν·((1-probe_IA) − ν̂)    # IA = invalid-advice rejection
```

**γ̂_gen (suppression estimate)**:

```
γ̂ₙ = (1-λ_γ)·γ̂ + λ_γ·γ₀
    + α_γ⁺·e_pressure · (γ_max - γ̂)
    - α_γ⁻·e_explore+ · γ̂
    + β_γ·((1-probe_EP) − γ̂)    # EP = exploration preservation
```

### 5.4 Confidence Tracking

Per-dimension confidence via EMA of evidence quality:

```
conf_τ = (1-ρ)·conf_τ + ρ·eq_τ
eq_τ = max(e_trust+, e_trust−, has_probe_VA)
```

Confidence is boosted when p_self is far from 0.5 (more discriminative event).

---

## 6. Self-Discovery Probability (p_self)

**Source**: `src/metrics/self_discovery.py` (73 lines)

### 6.1 Geometric p_self

```
p_self = σ((d_commit − d_reveal − margin) / τ_v)
```

| Variable | Meaning |
|:---------|:--------|
| d_commit | Steps before agent is committed to branch |
| d_reveal | Depth at which strong cues become visible |
| Δ = d_commit − d_reveal | "Intervention window" |
| τ_v | Temperature (default 1.0) |

- Δ >> 0 → p_self → 1 (agent sees cues before committing)
- Δ << 0 → p_self → 0 (agent commits blind)

### 6.2 Failure-if-Wait

```
p_fail = 1 − σ((d_commit − d_reveal) / τ_f)    where τ_f = 1.5
```

High p_fail → tutor should warn (agent will commit to wrong branch without help).

---

## 7. Joint Posterior — Bayesian Goal-Preference Inference

**Source**: `src/teachers/joint_goal_pref_posterior.py` (329 lines)

### 7.1 Hypothesis Space

The posterior maintains a 3D grid of hypotheses:

```
q(g, θ, z) ∈ ℝ^{n_g × n_θ × n_z}
```

| Dimension | Canonical Size | Values |
|:----------|:-------------:|:-------|
| Goals (g) | 8 | 4 atomic + 4 composite |
| Preferences (θ) | 2 | {safe, shiny} |
| Temptation (z) | 4 | {0.0, 0.3, 0.6, 0.9} |
| **Total cells** | **64** | Exact discrete inference |

### 7.2 Prior Initialization

```
q₀(g, θ, z) = P₀(g|c₀) · P₀(θ) · P₀(z)
```

**P₀(g|c₀) — Structural Prior** (CANONICAL, Step 4):

```
log P₀(g|c₀) = −β_len·complexity(g) − β_red·R(g; c₀) + β_feas·F(g; c₀) − log Z
```

| Feature | Formula | Meaning |
|:--------|:--------|:--------|
| complexity(g) | \|g\| − 1 | Occam penalty (atomic=0, composite=1) |
| redundancy R(g) | cosine(w₁, w₂) | Constituent overlap (composite only) |
| feasibility F(g) | ∈ {0, 1} | Hard mask (invalid goals get P=0) |
| β_len | 1.0 | Complexity penalty weight |
| β_red | 0.5 | Redundancy penalty weight |

**P₀(θ)**: Uniform over {safe, shiny}.

**P₀(z)**: (0.4, 0.3, 0.2, 0.1) — declining prior over temptation levels.

### 7.3 Bayesian Update

**Structural/PCFG mode** (CANONICAL — pure likelihood):

```
q_t(g, θ, z) ∝ q_{t-1}(g, θ, z) · P(a_obs | g, θ, z)
```

**Legacy mode** (DEPRECATED):

```
q_t(g, θ, z) ∝ q_{t-1}(g, θ, z) · P(a_obs | g, θ, z) · exp(β_C · C_t(g))
```

The likelihood P(a_obs | g, θ, z) is computed via goal-conditioned utility through the `GoalHypothesisSpace`, which modifies branch attributes based on the goal hypothesis.

### 7.4 Forgetting / Diffusion

```
q' = (1 − f) · q + f · Uniform       where f = 0.01
```

Prevents posterior collapse by continuously mixing toward uniform.

### 7.5 Key Queries

| Query | Formula | API |
|:------|:--------|:----|
| Goal marginal | P(g) = Σ_{θ,z} q(g,θ,z) | `marginal_goal()` |
| Pref marginal | P(θ) = Σ_{g,z} q(g,θ,z) | `marginal_pref()` |
| Tempt marginal | P(z) = Σ_{g,θ} q(g,θ,z) | `marginal_tempt()` |
| Subgoal marginal | q(u) = Σ_{g∋u} q(g) | `subgoal_marginals()` |
| MAP hypothesis | argmax q(g,θ,z) | `map_hypothesis()` |
| Entropy | −Σ q·log q | `entropy()` |
| Goal-conditional pref | P(θ\|g) = q(g,θ)/q(g) | `goal_conditional_pref()` |

### 7.6 Temptation Modification

For hypothesis z_tempt, the risky branch's temptation score is boosted:

```
tempt'_risky = tempt_risky + z_tempt
```

This creates a latent model of how susceptible the agent is to temptation.

---

## 8. RSA Warning Channel

**Source**: `src/agents/rsa_warning_channel.py` (499 lines)

### 8.1 Hypothesis Space

4-way segment-level risk hypotheses:

| Hypothesis | Meaning |
|:-----------|:--------|
| LEFT_RISKY | Upper/left lane is risky |
| RIGHT_RISKY | Lower/right lane is risky |
| BOTH_SAFE | Both lanes are safe |
| HAZARD_AHEAD | General hazard ahead |

### 8.2 Utterance Inventory

| Utterance | Cost | Specificity |
|:----------|:----:|:----------:|
| WARN_LEFT | 0.0 | Directional |
| WARN_RIGHT | 0.0 | Directional |
| WARN_AHEAD | 0.2 | Semi-specific |
| GENERIC_WARN | 0.5 | Vague |

### 8.3 RSA Tower

**L0 — Literal Listener**:

```
L₀(r|u,c) ∝ exp(λ_sem · match(u, r, c)) · P(r|c)
```

- `match(u, r, c) ∈ [0, 1]` measures semantic alignment between utterance and hypothesis
- `λ_sem = 3.0` controls semantic sharpness

**S1 — Pragmatic Speaker**:

```
S₁(u|r,c) ∝ exp(α · [log L₀(r|u,c) − λ_C · Cost(u)])
```

- `α_RSA = 2.0` controls speaker rationality
- `λ_C = 1.0` weights utterance cost

**S1 Belief Update (CANONICAL)**:

```
b⁺(r) ∝ S₁(u|r,c) · b⁻(r)
```

**S1+Trust Variant**:

```
b⁺(r) ∝ [S₁(u|r,c)]^{η_τ} · b⁻(r)
where η_τ = clip(τ̂, 0.3, 2.0)
```

Higher trust → stronger evidence from warning. Lower trust → warning is discounted.

### 8.4 Planner Adapter

Converts posterior belief to a risk penalty for the planner:

```
Δρ = E_{r~b⁺}[ρ(r)] − E_{r~Uniform}[ρ(r)]
```

Or in log-odds form:

```
logit(ρ̃) = logit(ρ̂) + λ_u · log[P(u|r=1,c) / P(u|r=0,c)]
```

---

## 9. Micro Tutor — BCICTv4 Decision Function

**Source**: `src/teachers/internalization_control_tutor_v4.py` (456 lines)

### 9.1 Action Space

```
A_micro = {WAIT, SOFT, WARN}     (3 dose levels)
dose:       0.0   0.5   1.0
```

- **WAIT** (dose=0): No intervention. Agent acts on its own.
- **SOFT** (dose=0.5): Mild warning. Reduced dependence impact.
- **WARN** (dose=1.0): Full warning. Maximum information but highest dependence risk.

### 9.2 Q-Function Structure

```
Q(a, dose) = Q_online(a) + λ_teach · V_full(dose) − λ_over · R_over(dose)
```

| Component | Meaning | Weight |
|:----------|:--------|:------:|
| Q_online | Scene-based value (ΔVOI, p_self, temptation) | 1.0 |
| V_full | Teaching value (internalization improvement) | λ_teach = 3.5 |
| R_over | Overteaching penalty (dependence, suppression) | λ_over = 4.0 |

### 9.3 Q_online Components

**For WARN**:

```
Q_online_warn = 1.0·Δs + 2.0·ΔVOI + 1.5·(1 − p_self) + 1.0·tempt − 0.05
```

**For WAIT**:

```
Q_online_wait = 2.0·p_self·Δs − 1.5·p_fail + 2.0
```

Where:
- `Δs` = information gain from full observation vs partial
- `ΔVOI` = value of information (sigmoid difference)
- `p_self` = self-discovery probability
- `p_fail` = failure probability if wait

### 9.4 Teaching Value V_full

```
V_full = V + λ_sd · p_sd − λ_dep · p_blind
```

Where:
- `V = L_now − L_next` (behavior loss reduction from intervention)
- `p_sd = p_self · (0.8 if self_discovery_subtype else 0.4) · (1 − dose)`
- `p_blind = (0.7 if no self_evidence else 0.2) · dose`

### 9.5 Bridge Behavior Loss

The tutor predicts how the agent's internal state manifests as observable behavior using the behavior bridge:

```python
L(m, zones) = Σ_probe w_probe · max(0, m_probe − z_healthy)²
```

where `z_healthy` are empirically-calibrated behavior zone thresholds.

### 9.6 One-Step Rollout (m → m')

For each dose, the tutor simulates the post-intervention state:

```python
m' = predict_m(m, dose, tempt, risk, subtype, has_self_evidence)
```

This applies the same update rules (§3.2) prospectively to evaluate each action's impact on the internalization state.

---

## 10. Compositional Goal Structure

**Source**: `src/teachers/compositional_goal_hypotheses.py` (185 lines)

### 10.1 Goal Space (8 goals)

| # | Goal | Type | Components | Weight Vector |
|:-:|:-----|:----:|:-----------|:--------------|
| 1 | collect_red | Atomic | — | [0.0, 2.5, 0.5, 0.0] |
| 2 | avoid_blue | Atomic | — | [2.0, −1.0, 0.0, 0.0] |
| 3 | use_safe | Atomic | — | [3.0, −0.5, 0.0, 0.0] |
| 4 | reach_fast | Atomic | — | [0.0, 0.0, 0.0, 3.0] |
| 5 | collect_red+avoid_blue | Composite | 1∧2 | Sum of 1,2 |
| 6 | collect_red+use_safe | Composite | 1∧3 | Sum of 1,3 |
| 7 | avoid_blue+use_safe | Composite | 2∧3 | Sum of 2,3 |
| 8 | reach_fast+avoid_blue | Composite | 4∧2 | Sum of 4,2 |

### 10.2 Goal-Conditioned Utility

Each goal modifies how branch attributes are evaluated:

```
U(π; g, θ) = R_goal(π; g) + λ_θ · R_pref(π; θ)
R_goal(π; g) = GoalWeight[g] · BranchAttrs(π)
```

For composite goals:

```
GoalWeight[g₁∧g₂] = GoalWeight[g₁] + GoalWeight[g₂]
```

### 10.3 Subgoal Marginals (Step 4 Primary Metric)

```
q(u) = Σ_{g ∋ u} q(g)
```

For each atomic subgoal u, sum posterior mass over all goals containing u. This resolves the observational equivalence between composite goals sharing a constituent.

---

## 11. Information Flow Diagram

```
Episode Start:
  Environment → (GridMap, features, cell_types, risk, cost)
  Posterior ← P₀(g|c₀) · P₀(θ) · P₀(z)

Per Step:
  1. Agent observes local features z_visible
  2. Risk head updates: ρ̂ = σ(w·z + b)
  3. Agent computes U(π|θ,m) for each branch
  4. Agent selects branch: a ~ P(a|s,θ,m)
  
  5. Tutor observes: a, branch_attributes, cell outcomes
  6. Posterior update: q(g,θ,z) ∝ q·P(a|g,θ,z)
  7. Observer update: m̂ from (dose, follow, correct, p_self, probes)
  8. Tutor decides: Q(WAIT), Q(SOFT), Q(WARN) → best action
  
  9. If WARN/SOFT: RSA channel → select utterance → belief update
  10. Agent m_t update: κ, τ, ν, γ_spec, γ_gen

Post-Episode:
  Risk head updated from full trajectory outcomes
  Mastery model updated (Beta-Bernoulli)
  Curriculum controller adjusts next episode parameters
```

---

## 12. Key Design Constraints (Red Lines)

1. **κ̂ is NOT a posterior latent** — it is an additive macro state used by the curriculum hook, not part of the joint posterior q(g,θ,z).
2. **Observer does NOT read m_t** — it must infer from tutor-visible events only.
3. **Prior depends on c₀ only** — P₀(g|c₀) must NOT incorporate observed actions (solved by structural prior).
4. **Micro action space is {WAIT, WARN}** (2-act canonical, 3-act with SOFT dose). DO NOT expand.
5. **Reward is rigid discrete table** (PREF_REWARD). Not learned, not continuous.
6. **Observational equivalence**: DO NOT optimize for exact composite goal label top-1. Use subgoal marginals instead.

---

## 13. Module Cross-Reference

| Mechanism | Primary Module | Lines |
|:----------|:--------------|:-----:|
| Agent policy P(π\|θ) | `stochastic_agent_policy.py` | 121 |
| Internalization m_t dynamics | `internalization_state_v3.py` | 217 |
| Risk prediction ρ̂ | `risk_model.py` | 127 |
| Cost+Risk joint heads | `cost_risk_model.py` | 237 |
| Observer m̂_t | `internalization_observer.py` | 720 |
| Joint posterior q(g,θ,z) | `joint_goal_pref_posterior.py` | 329 |
| Structural prior P₀(g\|c₀) | `compositional_goal_prior.py` | 332 |
| Goal hypothesis space | `compositional_goal_hypotheses.py` | 185 |
| RSA warning channel | `rsa_warning_channel.py` | 499 |
| Self-discovery p_self | `self_discovery.py` | 73 |
| Micro tutor BCICTv4 | `internalization_control_tutor_v4.py` | 456 |
| Behavior bridge probes | `behavior_bridge.py` | 114 |
| Behavior zones | `behavior_probes.py` | 145 |
