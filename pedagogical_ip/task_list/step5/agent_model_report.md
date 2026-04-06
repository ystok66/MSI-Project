# Agent Model Report: Learner Architecture & Decision Mechanisms

> **Handoff document** — self-contained reference covering the simulated learner agent: internalization state, choice model, risk/cost perception, branch concepts, and how the tutor observes and influences the agent.

---

## 1. Design Identity

The agent is a **simulated bounded-rational learner**, not a real human. It exists to:

1. Provide a realistic, mechanistic model of learner behavior for the tutor to reason about
2. Generate behaviorally-grounded responses to tutor interventions
3. Maintain internal state (internalization) that changes over teaching episodes

The agent is **not** an RL policy, not a neural network, and not trainable end-to-end. It is a structured probabilistic model with interpretable parameters.

---

## 2. Internalization State: 5D Factored Model

**Source**: `src/agents/internalization_state_v3.py`

$$m_t = (\kappa_t, \tau_t, \nu_t, \gamma_t^{spec}, \gamma_t^{gen})$$

| Dim | Field | Name | Default | Bounds | Pedagogical Valence |
|:---:|-------|------|:-------:|:------:|:-------------------:|
| κ | `kappa` | Risk sensitivity | 1.0 | [0.3, 3.0] | ↑ = more cautious |
| τ | `tau` | Trust | 0.3 | [0, 1] | ↑ = better |
| ν | `nu` | Dependence | 0.1 | [0, 0.8] | **↑ = worse** |
| γ_spec | `gamma_spec` | Temptation suppression | 0.0 | [0, 0.7] | ↑ = better |
| γ_gen | `gamma_gen` | General suppression | 0.0 | [0, 0.5] | **↑ = worse** |

### 2.1 Update Rules

#### κ (Risk Sensitivity) — Regression to Baseline

$$\kappa_{t+1} = \text{clip}\Big((1-\beta_\kappa)\kappa_t + \beta_\kappa \kappa_0 + \alpha_\kappa \cdot (r_{real} - r_{expected})\Big)$$

| Param | Value | Meaning |
|-------|:-----:|---------|
| κ_0 | 1.0 | Baseline risk sensitivity |
| β_κ | 0.08 | Regression rate |
| α_κ | 0.40 | Error-driven update strength |

**Mechanic**: When real risk exceeds expected, κ increases (learner becomes more cautious). When it's lower, κ decreases.

#### τ (Trust) — Quality-Driven

$$\tau_{t+1} = \tau_t + \alpha_\tau^+ (1-\tau_t) \cdot \mathbf{1}[helpful] - \alpha_\tau^- \tau_t \cdot \mathbf{1}[bad]$$

| Param | Value |
|-------|:-----:|
| α_τ⁺ | 0.25 |
| α_τ⁻ | 0.12 |

**Mechanic**: Trust increases when warnings are helpful (correct and followed), decreases when they are wrong. Asymmetric: trust is easier to build than to destroy.

#### ν (Dependence) — Compliance Without Evidence

$$\nu_{t+1} = \nu_t + \alpha_\nu^+ (1-\nu_t) \cdot \mathbf{1}[blind] - \alpha_\nu^- \nu_t \cdot \mathbf{1}[self\_disc]$$

| Param | Value |
|-------|:-----:|
| α_ν⁺ | 0.20 |
| α_ν⁻ | 0.15 |

**Mechanic**: Increases when learner follows warnings without independent evidence (blind obey). Decreases via self-discovery. **This is the overteaching marker** — high ν means the learner has lost independent decision-making ability.

#### γ_spec (Temptation-Specific Suppression)

$$\gamma^{spec}_{t+1} = \gamma^{spec}_t + \alpha_{gs}^+ (1-\gamma^{spec}_t) \cdot \mathbf{1}[tempt\_error] - \alpha_{gs}^- \gamma^{spec}_t \cdot \mathbf{1}[false\_supp]$$

| Param | Value |
|-------|:-----:|
| α_gs⁺ | 0.22 |
| α_gs⁻ | 0.10 |

**Mechanic**: Increases when learner makes temptation-driven errors (learns to suppress temptation). Decreases when learner over-suppresses (false suppression cost).

#### γ_gen (General Exploration Suppression)

$$\gamma^{gen}_{t+1} = \gamma^{gen}_t + \alpha_{gg}^+ (1-\gamma^{gen}_t) \cdot \mathbf{1}[pressure] - \alpha_{gg}^- \gamma^{gen}_t \cdot \mathbf{1}[explore\_ok]$$

| Param | Value |
|-------|:-----:|
| α_gg⁺ | 0.08 |
| α_gg⁻ | 0.12 |

**Mechanic**: Increases under sustained tutoring pressure. Decreases via successful exploration. **Asymmetric design**: easier to reduce suppression (0.12) than to increase it (0.08), because the system is biased toward preserving exploration.

---

## 3. Choice Model: Bounded-Rational Softmax

**Source**: `src/agents/stochastic_agent_policy.py`

### 3.1 Policy

$$P(\pi \mid s, \theta) = (1-\varepsilon) \cdot \text{softmax}(\beta \cdot U(\pi; \theta)) + \varepsilon \cdot \text{Uniform}$$

| Param | Symbol | Default | Meaning |
|-------|:------:|:-------:|---------|
| Temperature | β | 4.0 | Higher = more rational |
| Lapse rate | ε | 0.1 | Random exploration probability |
| Pref weight | λ_θ | 1.0 | Preference strength |

### 3.2 Preference Types (θ)

Each θ has a reward weight vector over `[safety, temptation, texture_novelty, shortcut]`:

| θ | Safety | Temptation | Novelty | Shortcut | Behavioral Signature |
|---|:------:|:----------:|:-------:|:--------:|---------------------|
| `safe` | **+2.0** | −1.0 | 0.0 | 0.0 | Risk-averse, avoids temptation |
| `shiny` | 0.0 | **+3.0** | 0.0 | 0.0 | Novelty-seeking, temptation-susceptible |
| `risky` | −0.5 | +0.5 | 0.0 | 0.0 | Risk-tolerant (research only) |
| `shortcut` | 0.0 | 0.0 | 0.0 | +2.0 | Path-length optimizer (research only) |
| `neutral` | +0.3 | 0.0 | 0.0 | 0.0 | Mild safety preference |

**Canonical types**: `safe` and `shiny` only. Others are for research extensions.

### 3.3 Branch Attributes

Each branch is described by a `BranchAttributes` vector:

| Field | Meaning | Source |
|-------|---------|--------|
| `safety_score` | How safe the branch appears | Branch summary S_MEAN_RISK (inverted) |
| `temptation_score` | How tempting/attractive | Mean temptation feature along branch |
| `texture_novelty` | Visual novelty | Feature diversity (research only) |
| `shortcut_bonus` | Path efficiency bonus | Not used in canonical |
| `risk_penalty` | True/estimated risk cost | Cost-risk model prediction |

---

## 4. Factored Utility Function

**Source**: `src/agents/internalization_state_v3.py` → `compute_factored_utility()`

$$U(\pi; \theta, m_t) = \lambda_\theta R_{pref}(\pi; \theta) - \kappa^2 \rho_{risk}(\pi) - \gamma^{spec} \cdot tempt(\pi) - \gamma^{gen} \cdot novel(\pi) + \tau \cdot warn(\pi) - \nu \cdot obey(\pi)$$

| Term | Formula | Effect |
|------|---------|--------|
| $R_{pref}$ | `PREF_REWARD[θ] · branch_attrs` | Type-specific preference reward |
| Risk cost | $\kappa^2 \cdot \rho_{risk}$ | Quadratic risk penalty (high κ → very cautious) |
| Tempt cost | $\gamma^{spec} \cdot tempt$ | Temptation suppression |
| Novel cost | $\gamma^{gen} \cdot 0.3 \cdot \mathbf{1}[novel]$ | General exploration suppression |
| Warn bonus | $\tau \cdot warn_{bonus}$ | Trust in tutor's recommendation |
| Obey cost | $\nu \cdot 0.2 \cdot \mathbf{1}[warned]$ | Dependence compliance cost |

### 4.1 Epistemic Risk Extension (B2, Research)

When `use_epistemic_risk=True`:

$$\rho_{risk}^{B2}(\pi) = \kappa^2 \cdot [\rho + (1-\rho) \cdot \alpha(\tilde{u}_r)] \cdot risk\_penalty$$

$$\alpha(\tilde{u}) = \alpha_{min} + (1-\alpha_{min}) \cdot e^{-\gamma \tilde{u}}$$

| Param | Value | Meaning |
|-------|:-----:|---------|
| α_min | 0.25 | Minimum risk attention floor |
| ρ | 0.35 | Base risk attention ratio |
| γ | 3.0 | Uncertainty decay rate |

**Effect**: When risk uncertainty is high, risk penalty attenuated (learner is less certain about danger). When uncertainty is low, full risk penalty applies.

---

## 5. Risk & Cost Perception

**Source**: `src/agents/cost_risk_model.py`

### 5.1 Architecture

```
4D Feature Vector (z)  ────→  BayesianCostHead  ────→  cost_hat = w_c·z + b_c
                         └──→  BayesianRiskHead  ────→  risk_hat = σ(w_r·z + b_r)
```

Two independent Bayesian linear heads that learn from visited cells:

| Head | Likelihood | Prediction | Prior |
|------|-----------|-----------|-------|
| Cost | Gaussian | `w·z + b` | w∼N(0, σ²I), b=1.0 |
| Risk | Bernoulli | `σ(w·z + b)` | w∼N(0, σ²I) |

### 5.2 Update

Online MAP estimation via gradient descent with L2 regularization:

$$w_{t+1} = w_t - \eta \cdot (-e \cdot x \cdot w + w / \sigma^2_{prior})$$

| Param | Cost Head | Risk Head |
|-------|:---------:|:---------:|
| Learning rate | 0.1 | 0.3 |
| Prior variance | 1.0 | 1.0 |
| Gradient clipping | 5.0 | 5.0 |
| Weight norm cap | 10.0 | 10.0 |

### 5.3 Uncertainty Estimation

Approximate posterior via Hessian:

$$\sigma^2_{pred}(x) = x^\top H^{-1} x, \quad H = \frac{1}{n}\sum_i x_i x_i^\top + \frac{I}{\sigma^2_{prior}}$$

Also supports belief-variance-based uncertainty: $\sigma^2 = w^\top \text{diag}(x_{var}) \cdot w$

### 5.4 World Weights

The environment generates true cost/risk from latent features via `WorldWeights`:

```
true_cost(z) = w_cost · z + b_cost
true_risk(z) = σ(w_risk · z + b_risk)
```

**Orthogonal risk weights** (from `semantic_subspace.py`): `w_risk[identity_dim] = 0`, ensuring risk only depends on semantic features.

### 5.5 Risk Supervision Modes

| Mode | Input | Realism |
|------|-------|:-------:|
| `oracle_visited` | True risk value for visited cell | Higher |
| `binary_outcome` | 0/1 hazard event | Lower |

---

## 6. Branch Perception Pipeline

### 6.1 Branch Summary (8D)

**Source**: `src/agents/branch_summary.py`

Per-branch semantic vector computed from cell-level predictions:

| Dim | Name | Source |
|:---:|------|--------|
| 0 | Mean risk | `mean(risk_hat[cells])` |
| 1 | Max risk | `max(risk_hat[cells])` |
| 2 | Mean cost | `mean(cost_hat[cells])` |
| 3 | Risk uncertainty | `mean(risk_unc[cells])` |
| 4 | Cost uncertainty | `mean(cost_unc[cells])` |
| 5 | Cue count | Fraction of cells with texture > 0.3 |
| 6 | Cue variance | Diversity of texture features |
| 7 | Length | Normalized branch length |

### 6.2 Gaussian Branch Concepts (3 concepts)

**Source**: `src/agents/branch_concepts.py`

Three diagonal Gaussian concepts over the 8D summary space:

| Concept | Mean Risk | Mean Max Risk | Represent |
|---------|:---------:|:------------:|-----------|
| `safe_branch` | 0.10 | 0.15 | Low-risk, low-uncertainty branch |
| `risky_branch` | 0.50 | 0.70 | High-risk branch |
| `ambiguous_branch` | 0.30 | 0.40 | Uncertain, medium-risk branch |

**Scoring**: KL-based log-inclusion:

$$\text{score}(\pi, k) = -\frac{1}{\tau} \text{KL}\big(N(x_\pi, \sigma^2_{obs}I) \| N(\mu_k, \Sigma_k)\big)$$

**Update**: Online Welford-Knuth with weighted observations.

### 6.3 Branch Scorer Probe (Diagnostic)

**Source**: `src/agents/branch_scorer_probe.py`

Linear safety scorer over 12D input: `[summary(8) + familiarity(1) + concept_scores(3)]`

$$Q_{branch}(\pi) = v \cdot \text{input}(\pi) + b$$

Trained by SGD with BCE loss to predict oracle safety label. **Diagnostic only** — not connected to the planner.

---

## 7. How the Agent Makes Choices

Complete decision flow for a single fork-branch episode:

```
1. Agent arrives at fork cell
2. Observe visible cells → build belief means (noisy features)
3. For each branch:
   a. Summarize: cells → 8D branch summary
   b. Predict risk/cost via BayesianCostRiskHead
   c. Match to Gaussian concepts (safe/risky/ambiguous)
   d. Build BranchAttributes: (safety, tempt, risk_penalty, ...)
4. Compute factored utility U(π; θ, m_t) for each branch
5. Apply softmax + lapse:
   P(π) = (1−ε) · softmax(β · U) + ε · Uniform
6. Sample branch choice
7. After traversal: update m_t based on outcome
   (risk update, trust/dependence events, γ updates)
```

---

## 8. How the Tutor Interacts with the Agent

| Tutor Action | Agent Effect |
|-------------|-------------|
| **WARN** | +warn_bonus to warned branch → shifts U; if followed blindly → ν↑; if correct → τ↑ |
| **WAIT** | No direct effect; if self-discovery happens → ν↓, γ_gen↓ |
| High dose | γ_gen↑ (sustained pressure), ν↑ (if p_self low) |
| Low dose / zero dose | γ_gen↓ (exploration preserved), self-discovery possible |

The fundamental tension:
- **Warn** improves immediate safety but risks ν↑ and γ_gen↑ (overteaching)
- **Wait** preserves autonomy but risks failure if p_self is low

---

## 9. Source File Index

| File | Lines | Role |
|------|:-----:|------|
| `src/agents/internalization_state_v3.py` | 217 | 5D factored state: κ,τ,ν,γ_spec,γ_gen + utility + choice |
| `src/agents/stochastic_agent_policy.py` | 121 | Preference types, BranchAttributes, softmax+lapse policy |
| `src/agents/cost_risk_model.py` | 237 | Bayesian cost/risk heads, world weights, epistemic extension |
| `src/agents/risk_model.py` | ~100 | BayesianRiskHead (Bernoulli likelihood) |
| `src/agents/branch_summary.py` | 125 | 8D branch semantic summary |
| `src/agents/branch_concepts.py` | 147 | 3 Gaussian concepts: safe/risky/ambiguous |
| `src/agents/branch_scorer_probe.py` | 117 | Linear scorer diagnostic (12D input) |
| `src/agents/behavior_bridge.py` | 115 | m→ẑ mapping for tutor behavior prediction |
| `src/agents/behavior_probes.py` | ~150 | RC/TR/EP/VA/IA probe definitions |
| `src/agents/familiarity.py` | ~80 | Familiarity score for branch concepts |
| `src/agents/observation_model.py` | ~170 | Noisy feature observation |
| `src/agents/feature_belief.py` | ~190 | Bayesian feature belief map |
| `src/agents/belief.py` | ~250 | θ posterior update (Bayesian) |
| `src/agents/belief_planning.py` | ~250 | Planning under belief uncertainty |
| `src/envs/semantic_subspace.py` | 119 | Orthogonal risk weights, identity neutralization |

---

## 10. Key Design Principles

1. **Bounded rationality**: Softmax+lapse, not argmax. Real learners have noise and lapses.
2. **Factored state**: Each dim (κ,τ,ν,γ_spec,γ_gen) has clear pedagogical meaning and independent update rule.
3. **Asymmetric dynamics**: ν and γ_gen are harder to increase than decrease — the system is biased toward preserving learner autonomy.
4. **Quadratic risk**: $\kappa^2$ penalty makes high-κ learners disproportionately cautious.
5. **Feature-as-latent**: The cell's 4D feature vector IS its latent state — no separate hidden variable.
6. **Semantic/identity separation**: Risk weights are zeroed on identity dims to prevent positional shortcuts.
7. **Concepts are learned**: Gaussian branch concepts update incrementally from experience, not hard-coded.
8. **Epistemic awareness** (B2 extension): Risk attention is modulated by uncertainty — more uncertain → less penalty.
