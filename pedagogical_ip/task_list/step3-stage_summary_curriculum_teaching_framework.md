# Stage Summary: Bayesian Pedagogical Curriculum Teaching Framework

> **Handoff document** — written for the next agent, collaborator, or future self.
> Self-contained; does not require conversation history to understand.

---

## 1. Project Definition

We study a **Bayesian pedagogical tutoring framework** in which a tutor maintains beliefs over a learner's latent type, internalization dynamics, and curriculum effects, and decides:

1. **Within an episode** — when and how to intervene (WARN / UNLOCK / ITEM_DROP), respecting the semantic distinction among these three intervention types.
2. **Across episodes** — what kind of lesson to present next, when to probe mastery (EVAL), and when to stop teaching (STOP).

The framework is **non-RL, belief-updating, model-based, bounded-planning**. It does not use reward-maximizing policy gradient or value iteration. Instead, it combines Bayesian posterior updating (over learner state and lesson response) with constrained greedy planning under explicit risk budgets.

The research trajectory has progressed from *"can the tutor detect when to warn?"* through *"can the tutor model internalization dynamics?"* to the current frontier: *"can the curriculum planner rank lessons, stop at the right time, and avoid overteaching — while respecting learner-specific risk constraints?"*

---

## 2. Scenario / Environment

### 2.1 Scenario Families

The environment is not a single gridworld benchmark but a family of **controllable pedagogical scenarios**, each designed to isolate specific teaching mechanisms:

| Family | Purpose |
|--------|---------|
| **PP-MRB** (persistent-profile) | Validates persistent learner modelling and "learned → talk less" selectivity |
| **TIC / TIC-v4** (teaching-internalization corridor) | Tests internalization dynamics, valid/invalid advice, beneficial novelty |
| **CGC / CGC-v2** (compositional goal corridor) | Structured goal composition, held-out generalization |

Key **lesson subtypes** within these families:

| Subtype | Tests |
|---------|-------|
| `ppmrb_standard` / `ppmrb_self_discovery` | Persistent profile, dose calibration |
| `warn_rescue` / `tic_rescue_heavy` | Trust building, advice validity |
| `tic_temptation` | Temptation resistance, specific suppression |
| `beneficial_novelty` | Exploration preservation vs. general suppression |
| `false_suppression_cost` | Cost of over-suppression |
| `self_discovery_needed` | Dependence reduction |
| `sparse_valid_advice` / `sparse_invalid_advice` | Advice reliability discrimination |

### 2.2 Key Information Structure

The core of each scenario is not spatial complexity but **informational structure**:

- **Reveal / commit timing** — when the learner sees danger vs. when they must decide
- **Valid vs. invalid advice** — tutor sometimes gives wrong signals
- **Temptation vs. self-discovery** — risky branches that are genuinely more rewarding
- **Beneficial novelty** — novel options the learner should explore, not suppress
- **Overteaching risk** — continued teaching after the learner has internalized

### 2.3 Why Current Map Complexity Is Sufficient

The current abstraction level is **deliberately maintained**. The primary bottleneck is not environmental realism but **lesson ranking discriminability and risk–gain trade-off control** at the curriculum layer. Increasing map complexity now would obscure, not resolve, these issues.

---

## 3. Tutor–Agent Architecture (Canonical Form)

### 3.1 Learner Latent Belief

The tutor maintains a posterior over learner type:

$$q_t(\theta), \quad \theta \in \{\text{safe}, \text{shiny}\}$$

Early work explored preference-only vs. joint latent belief. The current curriculum line inherits the **factored** form: separate posteriors over learner type, coupled with a rich internalization state.

### 3.2 Internalization State

The **canonical internalization state** (from ICT-v2 onward):

$$m_t = (\kappa_t,\; \tau_t,\; \nu_t,\; \gamma_t^{\text{spec}},\; \gamma_t^{\text{gen}})$$

| Dim | Name | Meaning |
|-----|------|---------|
| $\kappa$ | Risk calibration | Accuracy of internal risk estimates |
| $\tau$ | Trust | Confidence in valid tutor evidence |
| $\nu$ | Dependence | Obedience without own evidence (blind following) |
| $\gamma^{\text{spec}}$ | Specific suppression | Temptation-specific inhibition |
| $\gamma^{\text{gen}}$ | General suppression | Broad exploration inhibition |

**Critical distinctions** (key mechanistic contributions of ICT-v2):
- **Trust ≠ dependence**: $\tau$ and $\nu$ are separately updated and have opposite pedagogical valence.
- **Specific ≠ general suppression**: $\gamma^{\text{spec}}$ is beneficial; $\gamma^{\text{gen}}$ indicates overteaching.

### 3.3 Mastery State

The curriculum controller maintains an explicit mastery state:

$$u_t = (u_t^{RC},\; u_t^{TR},\; u_t^{EP},\; u_t^{VA},\; u_t^{IA})$$

| Dim | Name | Meaning |
|-----|------|---------|
| $RC$ | Risk calibration | Can the learner assess danger? |
| $TR$ | Temptation resistance | Can the learner resist short-term lures? |
| $EP$ | Exploration preservation | Does the learner still explore when appropriate? |
| $VA$ | Valid-advice uptake | Does the learner use correct advice? |
| $IA$ | Invalid-advice resistance | Can the learner reject wrong advice? |

This transition from deficit-only to **progress/mastery-aware curriculum** was key to enabling meaningful lesson sequencing.

### 3.4 Macro State (Curriculum Planner Input)

$$x_t = (q_t,\; m_t,\; u_t,\; h_t,\; B_t)$$

where $h_t$ is curriculum history / recent lesson counts, and $B_t$ is the remaining teaching budget.

---

## 4. Key Mechanism Formulas (Canonical Set)

This section presents the **stable, inheritable** formulas. Historical intermediate versions are omitted.

### 4.1 Mastery Update (Beta-Bernoulli)

$$u_{k,t} = \frac{a_{k,t}}{a_{k,t} + b_{k,t}}$$

$$a_{k,t+1} = \lambda\, a_{k,t} + y_{k,t}, \qquad b_{k,t+1} = \lambda\, b_{k,t} + (1 - y_{k,t})$$

where $y_{k,t} \in \{0,1\}$ is the probe outcome for dimension $k$, and $\lambda$ is a decay factor for non-stationarity.

### 4.2 Hierarchical / Hybrid Lesson Response

$$\mu^{\text{gain}}_{\text{hyb}}(x,\ell) = \mu^{\text{hier}}_{\text{gain}}(\ell, b) + r^{\text{ctx}}_{\text{gain}}(x, \ell)$$

$$\mu^{\text{harm}}_{\text{hyb}}(x,\ell) = \mu^{\text{hier}}_{\text{harm}}(\ell, b) + r^{\text{ctx}}_{\text{harm}}(x, \ell)$$

- $\mu^{\text{hier}}$: hierarchical empirical Bayes backbone — low variance, stable, borrows strength across sparse buckets.
- $r^{\text{ctx}}$: Bayesian linear residual — learns learner-state-conditional deviations.

**Provenance**: v8's strong transfer comes from $\mu^{\text{hier}}$; v10's actionability breakthrough comes from $r^{\text{ctx}}$.

### 4.3 Pairwise / Dueling Lesson Comparison

$$\Delta \mu_C(x, \ell_i, \ell_j) = w_C^\top \big(\phi(x,\ell_i) - \phi(x,\ell_j)\big)$$

$$\Delta \mu_E(x, \ell_i, \ell_j) = w_E^\top \big(\phi(x,\ell_i) - \phi(x,\ell_j)\big)$$

$$\Delta \mu_H(x, \ell_i, \ell_j) = w_H^\top \big(\phi(x,\ell_i) - \phi(x,\ell_j)\big)$$

Purpose: learn **relative** lesson value (who benefits this learner more), not just absolute scores. This directly attacks the PCR = 0% problem observed in v9.

**Status**: Pairwise heads in v11 receive massive updates (PostUp ≈ 1000) but `G_pw` PCR remains 0%. The direction is sound; maturation is incomplete.

### 4.4 Curriculum Objective (Constrained Form)

$$\max_{\ell \in \mathcal{L} \cup \{\text{EVAL}, \text{STOP}\}} \; J_t(\ell)$$

where:

$$J_t(\ell) = w_C\, \mu_C(x_t, \ell) + w_E\, \mu_E(x_t, \ell) + \lambda_{\text{unc}}(B_t)\, U_t(\ell)$$

subject to **explicit risk constraints**:

$$\mu_{\text{OTR}}(x_t,\ell) + \beta_O\, \sigma_{\text{OTR}}(x_t,\ell) \le \eta_O(x_t)$$

$$\mu_{\nu}(x_t,\ell) + \beta_\nu\, \sigma_{\nu}(x_t,\ell) \le \eta_\nu(x_t)$$

$$\mu_{\gamma^g}(x_t,\ell) + \beta_g\, \sigma_{\gamma^g}(x_t,\ell) \le \eta_g(x_t)$$

**Implementation**: Filter+rank — first eliminate lessons violating risk budgets, then rank remaining by gain. This is more natural and more effective than penalty-sum, confirmed across multiple ablations.

### 4.5 Adaptive Risk Budgets

$$\eta_O(x_t) = \eta_{O,0} + a_O(1-u_t^{IA}) + b_O(1-u_t^{EP}) - c_O\,\nu_t - d_O\,\gamma_t^{\text{gen}}$$

$$\eta_\nu(x_t) = \eta_{\nu,0} + a_\nu(1-u_t^{VA}) - c_\nu\,\nu_t$$

$$\eta_g(x_t) = \eta_{g,0} + a_g(1-u_t^{EP}) - c_g\,\gamma_t^{\text{gen}}$$

Risk budgets increase when the learner needs more exploration (low mastery) and tighten when the learner is already fragile (high $\nu$ or $\gamma^{\text{gen}}$).

### 4.6 Exploration Decay

$$\lambda_{\text{unc}}(B_t) = \lambda_0 \cdot \sigma\!\left(\frac{B_t - B_{\text{mid}}}{\tau_B}\right)$$

$$\lambda_{\text{unc}}^{\text{eff}}(t) = \lambda_{\text{unc}}(B_t) \cdot \exp\!\left(-\frac{n_{\text{post}}(x_t,\ell)}{\tau_n}\right)$$

- Budget-conditioned: explore more when budget is ample.
- Maturity-decayed: explore less for well-observed lesson–state pairs.
- Motivation: v10 showed `no_unc` outperforming `full` on shiny — exploration was hurting exploitation once the model had signal.

### 4.7 STOP / EVAL

**STOP** (marginal-value rule):

$$\text{STOP if } \max_{\ell \in \mathcal{L}} J_t(\ell) < \epsilon_{\text{stop}}(x_t)$$

$$\epsilon_{\text{stop}}(x_t) = \epsilon_0 + a_s\,\nu_t + b_s\,\gamma_t^{\text{gen}} - c_s \sum_k w_k\, u_{k,t}$$

**EVAL** (information-value rule):

$$J_t(\text{EVAL}) = \lambda_{\text{info}}\!\left(\text{Var}[u_t] + \max_\ell \text{Var}[V(x_t,\ell)]\right) - \lambda_{\text{cost}}\, c_{\text{eval}}$$

STOP has been **repeatedly confirmed as the single most valuable curriculum component** across v3, v4, v7, v8, and v11.

---

## 5. Experimental Stages

### Stage A: Persistent Learner Model & Selectivity

- Established PP-MRB family and persistent learner modelling.
- Joint latent `(preference, type)` posterior shown necessary for conflict scenarios.
- "Learned → talk less" demonstrated for the first time.

### Stage B: Internalization & Mechanism Disentanglement

- ICT-v1 / ICT-v2 introduced the factored internalization state $(κ, τ, ν, γ^{\text{spec}}, γ^{\text{gen}})$.
- Proved $\tau \neq \nu$ and $\gamma^{\text{spec}} \neq \gamma^{\text{gen}}$ — each pair must be modelled separately.
- Identified the overteaching inverted-U: moderate teaching intensity is optimal.

### Stage C: Behavioral Probes & Mechanism-Consistent Evaluation

- BI-ICT-v3 / BC-ICT-v4 / MC-ICT-v5: progressive refinement of mechanism-consistent tutoring.
- FICA-v1 framework audit: 19/21 invariants passed; failures were assertion granularity, not framework collapse.
- **MCA (Mechanism-Consistent Accuracy)** introduced — caught "accidental correctness" in earlier tutors.
- Bridge / Jacobian audit: confirmed state → behavior mapping has correct sign and sparsity.

### Stage D: Curriculum Loop Closure

| Version | Key Result |
|---------|------------|
| **CCT-v1** | `micro_PCR = 0%` — exposed macro–micro disconnect |
| **CCT-v2** | `ERCR = 100%` — lesson → episode → micro action chain **first truly closed** |
| **CCT-v3** | 5 lessons + 1 eval > 12 fixed lessons; STOP first truly effective |

### Stage E: Budget / Mastery / Cross-Session

| Version | Key Result |
|---------|------------|
| **CCT-v4** | Budget inverted-U first confirmed; `BdgBlk > 0` for the first time |
| **CCT-v5** | Single-session Bayesian: no significant improvement (data scale insufficient) |
| **CCT-v6** | Cross-session posterior sharing improved shiny C by +7pp; `PostUp = 59` |

### Stage F: Risk / Actionability / Pairwise

| Version | Key Metric | Result |
|---------|-----------|--------|
| **CCT-v7** | OTR, ν | Harm posterior active → lowest OTR/ν, but C–E tradeoff too conservative |
| **CCT-v8** | C, E, SE | **Transfer best**: safe C=78%, shiny E=66%; hier shrinkage +13pp E |
| **CCT-v9** | PCR audit | **Actionability audit introduced**: only U has PCR (100%); all gain/harm = 0% |
| **CCT-v10** | G PCR | **Actionability breakthrough**: G PCR = 36–38%, G_res = 32–36% |
| **CCT-v11** | SE, OTR | **Efficiency best**: SE ≈ 0.27 (2× any prior), OTR = 0.07, #T = 2 |

---

## 6. Current Best Results

### 6.1 Transfer Best: CCT-v8

| θ | C | E | SE | OTR |
|---|:-:|:-:|:--:|:---:|
| safe | **78%** | 53% | 0.152 | 0.066 |
| shiny | 62% | **66%** | 0.139 | 0.280 |

Hierarchical EB shrinkage gives the best raw transfer scores.

### 6.2 Efficiency–Safety Best: CCT-v11

| θ | #T | C | E | **SE** | **OTR** |
|---|:--:|:-:|:-:|:------:|:-------:|
| safe | 2 | 53% | 53% | **0.273** | **0.085** |
| shiny | 2 | 62% | 53% | **0.270** | **0.072** |

With only 2 lessons, achieves 2× the stop efficiency and near-lowest OTR.

### 6.3 Actionability Best: CCT-v10

| Term | PCR |
|------|:---:|
| G (total gain) | **36–38%** |
| G_res (contextual residual) | **32–36%** |
| U (uncertainty) | 19–23% |

First version where gain terms actually change the lesson argmax.

> **There is no single dominant model.** Different versions represent optima on different dimensions. Unifying these is the goal of the next phase.

---

## 7. Stable Conclusions (Established Facts)

1. **Curriculum control >> micro-warning tuning** — the marginal value of further micro-tutor refinement is small compared to curriculum-level decisions.
2. **The `lesson → episode → micro action` loop is genuinely closed** — ERCR/micro_PCR went from 0% to high.
3. **STOP is repeatedly the highest-value single component** — across v3, v4, v7, v8, v11.
4. **Budget inverted-U is a cross-level invariant** — confirmed 9 times across versions, budget levels, and learner types.
5. **Cross-session posterior sharing is genuinely valuable** — PostUp > 0 consistently correlates with improved C on shiny.
6. **Fixed harm penalties cause over-conservatism** — adaptive budget or explicit constraint is strictly better.
7. **Hierarchical sharing significantly improves statistical efficiency** — v8's hier shrinkage gave +13pp E on shiny.
8. **The primary bottleneck now is lesson ranking discriminability** — gain/harm terms have the right sign but insufficient cross-lesson variance to change argmax.
9. **This is not a code bug, nor a map-simplicity issue** — patterns are stable, reproducible, and directionally consistent.
10. **Framework identity**: non-RL, belief-updating, model-based, bounded-planning. Do not jump to full POMDP or reward-maximizing RL.

---

## 8. Open Problems (Honest Assessment)

### 8.1 Pairwise Gain Has Not Matured

- `G_pw` PCR = 0% in v11 despite PostUp ≈ 1000.
- Pairwise training is occurring, but the model is not yet producing cross-lesson gain variance.
- Likely cause: most pairwise comparisons come from the same macro-state across sessions, not diverse state × lesson combinations.

### 8.2 Harm Has Conceptual Value but Low Actionability

- Harm / constraint is a **high-value structural component** (removing it drops shiny C by 6–31pp).
- But harm terms themselves (H_pw, H_res) have PCR ≈ 0% — they function as a **gate/filter**, not a ranking signal.

### 8.3 Exploration Still Interferes with Exploitation

- `no_unc` outperforms `full` in multiple shiny ablations (v10: C 66% vs 50%; v11: safe C 69% vs 53%).
- Uncertainty bonus needs budget-conditioned **and** posterior-maturity-conditioned decay.

### 8.4 No Single Dominant Model

- v8: transfer best (safe C=78%, shiny E=66%)
- v10: actionability best (G PCR=36%)
- v11: efficiency–safety best (SE=0.27, OTR=0.07)
- The framework has not yet converged to a single dominant configuration.

---

## 9. Diagnosis: What Kind of Problem Is This?

### Not the Primary Cause

| Factor | Assessment |
|--------|-----------|
| Code bugs | Engineering bugs occurred (filter scale, warm-up) but were diagnosed and fixed. Remaining patterns are stable and reproducible. |
| Map too simple | Current abstraction is deliberately sufficient. Adding map complexity would obscure the ranking problem. |
| Tutor/agent too simple | Current agent already exposes trust, dependence, novelty, advice validity, and overteaching. |

### Primary Causes

| Category | Description |
|----------|-------------|
| **Modelling** | Absolute-value lesson response lacks cross-lesson discriminability. Pairwise/relative modelling is the right direction but not yet mature. |
| **Algorithm** | Gain, harm, and uncertainty terms are not yet combined in a way that gives all three non-zero PCR simultaneously. |
| **Statistical efficiency** | Contextual and pairwise heads require more diverse state × lesson observation pairs. Hierarchical shrinkage compensates but limits learner-conditionality. |

---

## 10. Next Phase: CCT-v12

### Title

**CCT-v12: Pairwise Counterfactual Constrained Bayesian Curriculum Planner**

*(hybrid residual + pairwise ranking + constrained risk-aware planner)*

### Core Objectives

1. **Make pairwise lesson ranking actually work** — `G_pw` PCR > 0 is the primary success criterion.
2. **Upgrade harm from negative reward to explicit constraint** — filter+rank, not penalty-sum.
3. **Achieve non-zero PCR for gain, harm, AND uncertainty simultaneously**.
4. **Pareto improvement on shiny**: C ≥ 62% (v8 level), E ≥ 53% (v11 level), OTR ≤ 0.10.

---

## 11. Formulas for Next Phase

### 11.1 Pairwise Ranking

$$\Delta \mu_C(x, \ell_i, \ell_j) = w_C^\top\big(\phi(x,\ell_i) - \phi(x,\ell_j)\big)$$

$$\Delta \mu_E(x, \ell_i, \ell_j) = w_E^\top\big(\phi(x,\ell_i) - \phi(x,\ell_j)\big)$$

$$\Delta \mu_H(x, \ell_i, \ell_j) = w_H^\top\big(\phi(x,\ell_i) - \phi(x,\ell_j)\big)$$

Bradley–Terry form:

$$P(\ell_i \succ \ell_j \mid x) = \sigma\!\big(s(x,\ell_i) - s(x,\ell_j)\big)$$

### 11.2 Counterfactual Replay Labels

Short-horizon surrogate value:

$$U_{\text{short}}(x_t, \ell) = w_C\,\Delta C + w_E\,\Delta E - \lambda_O\,\Delta\text{OTR} - \lambda_\nu\,\Delta\nu - \lambda_\gamma\,\Delta\gamma^{\text{gen}}$$

Pairwise label:

$$y_{ij} = \mathbf{1}\!\big[U_{\text{short}}(x_t, \ell_i) > U_{\text{short}}(x_t, \ell_j)\big]$$

Key insight: do not rely solely on natural online trajectories. Use **short-horizon replay** to generate richer pairwise training data across diverse state × lesson pairs.

### 11.3 Constrained Planner

$$\max_{\ell} \; J_t(\ell) = w_C\,\mu_C(x_t,\ell) + w_E\,\mu_E(x_t,\ell) + \lambda_{\text{unc}}^{\text{eff}}(t)\,U_t(\ell)$$

subject to the three risk constraints from §4.4.

### 11.4 Adaptive Exploration Decay

$$\lambda_{\text{unc}}^{\text{eff}}(t) = \lambda_0 \cdot \sigma\!\left(\frac{B_t - B_{\text{mid}}}{\tau_B}\right) \cdot \exp\!\left(-\frac{n_{\text{post}}(x_t,\ell)}{\tau_n}\right)$$

---

## 12. Experiment Plan

### Exp A: v12 vs v11 vs v10 vs v8

Compare all Pareto-frontier versions. Metrics: C, E, MCA_C, MCA_E, SE, OTR, ν, γg, PostUp, #Teach, #Eval.

### Exp B: Actionability Audit 4.0

Track AM and PCR per term. **Success criterion**: at least 2 of {G_pw, H terms, U} have PCR > 0.

### Exp C: Counterfactual Replay Ablation

| Condition | Data Source |
|-----------|------------|
| `natural_only` | Online trajectory pairs only |
| `replay_pw` | Counterfactual replay pairwise |
| `replay_pw + hier_prior` | Replay + hierarchical prior initialization |

Goal: confirm that pairwise gain's PCR = 0 is a **data problem**, not a method problem.

### Exp D: Constraint vs Penalty

| Condition |
|-----------|
| Fixed penalty |
| Adaptive penalty |
| Explicit constraint (filter+rank) |

Goal: confirm filter+rank dominates penalty across C, E, OTR.

### Exp E: Held-out Generalization

Test on held-out subtype, novelty intensity, advice reliability, lesson composition.
Metrics: SE, MCA, LF, PostUp. Goal: confirm planner learns mechanisms, not lesson IDs.

---

## 13. Regression / Audit Gates

Every new version must pass these checks before being considered a valid successor:

- [ ] Old-family invariants (PP-MRB, TIC correctness)
- [ ] MCA paradox check (no accidental correctness)
- [ ] Lesson Fidelity (LF) ≥ 0.8
- [ ] Episode Realization Compliance Rate (ERCR) ≥ 80%
- [ ] `micro_PCR` > 0
- [ ] OTR ≤ previous version's OTR (no regression)
- [ ] SE ≥ previous version's SE (no regression)
- [ ] AM / PCR audit per term

> Full FICA re-run is not needed every version; but these point checks are **mandatory infrastructure**.

---

## 14. If You Only Read One Page, Read This

1. **This is a Bayesian pedagogical tutoring project** — non-RL, belief-updating, model-based, bounded-planning. Do NOT jump to full POMDP or policy gradient.

2. **Transfer best = CCT-v8** — safe C = 78%, shiny E = 66%. Uses hierarchical EB shrinkage.

3. **Efficiency–safety best = CCT-v11** — SE ≈ 0.27 (2× any prior), OTR ≈ 0.07, only 2 lessons needed. Uses pairwise + constraint + decaying exploration.

4. **Actionability best = CCT-v10** — gain PCR = 36–38%. First time gain terms actually change lesson ranking. Uses hybrid (hier + contextual residual).

5. **The primary bottleneck is lesson ranking discriminability** — gain/harm posteriors predict similar values across lessons → argmax doesn't change. Pairwise ranking is the theoretically correct direction but not yet mature.

6. **Constraint (filter+rank) is a core high-value component** — removing it drops shiny C by 6–31pp. It functions as a structural inductive bias more valuable than posterior precision.

7. **STOP is repeatedly the most valuable single component** — true since CCT-v3, confirmed 5+ times.

8. **Budget inverted-U is a framework invariant** — confirmed 9 times. Do not ignore budget constraints.

9. **Do NOT** primarily work on: more complex maps, full POMDP, further micro-warning tuning, or adding new scenario families.

10. **DO** primarily work on: making pairwise gain ranking actionable (PCR > 0), counterfactual replay data generation, exploration decay calibration, and unifying v8/v10/v11 into a single dominant model.

---

## Key Source Files

| File | Description |
|------|-------------|
| `src/curriculum/lesson_response_model_v3.py` | Hierarchical EB backbone |
| `src/curriculum/hybrid_response_model.py` | Hier + residual + dueling |
| `src/curriculum/pairwise_response_model.py` | Hier + residual + pairwise counterfactual |
| `src/curriculum/curriculum_controller_v8.py` | Transfer-best planner |
| `src/curriculum/curriculum_controller_v10.py` | Actionability-best planner |
| `src/curriculum/curriculum_controller_v11.py` | Efficiency-best planner |
| `src/agents/internalization_state_v3.py` | Factored internalization state |
| `src/agents/behavior_probes.py` | Mechanism-consistent probe battery |
| `src/curriculum/mastery_model.py` | Beta-Bernoulli mastery tracker |
| `src/curriculum/adaptive_episode_generator_v2.py` | Mastery-conditioned lesson → episode |
| `src/teachers/internalization_control_tutor_v4.py` | BC-ICT-v4 micro-tutor |
| `results/cct_v8_report.md` | Transfer-best report |
| `results/cct_v10_report.md` | Actionability-best report |
| `results/cct_v11_report.md` | Efficiency-best report |
