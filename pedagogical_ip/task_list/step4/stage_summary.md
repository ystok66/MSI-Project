# Stage Summary: Canonical Pedagogical Curriculum Controller

> **Handoff document** — self-contained reference for the next agent or collaborator.
> Covers project identity, architecture, formulas, experimental evidence, and canonical configuration.
> Does not require conversation history to understand.

---

## 1. Project Identity

This is a **non-RL, belief-updating, model-based, bounded-planning** pedagogical tutoring and curriculum control framework.

The system maintains Bayesian beliefs over a learner's latent type, internalization dynamics, and mastery state, and makes two classes of decisions:

1. **Within an episode** (micro): when and how to intervene — `WAIT / WARN / UNLOCK / ITEM_DROP`
2. **Across episodes** (macro): what lesson to teach next, when to evaluate, when to stop — `lesson / EVAL / STOP`

The goal is **not** just current task success. It is to jointly optimize:
- Immediate safety (correct branch choice)
- Long-term learner autonomy (reduced dependence, preserved exploration)
- Teaching efficiency (avoid overteaching)

This is **not** an end-to-end RL agent, not an exact POMDP solver, and not a reward-maximizing policy gradient system. The framework uses constrained greedy planning with Bayesian posterior updating.

---

## 2. Two-Layer Control Architecture

### 2.1 Micro Layer: Within-Episode Tutor

**Action space:** $\{\text{WAIT},\; \text{WARN},\; \text{UNLOCK},\; \text{ITEM\_DROP}\}$

| Action | Meaning |
|--------|---------|
| WAIT | Observe without intervening; let learner act on own evidence |
| WARN | Provide risk/evidence signal; builds trust if accurate |
| UNLOCK | Modify topology/affordance; changes what learner can reach |
| ITEM_DROP | Provide traversal mitigation / protection items |

The micro tutor uses **persistent learner belief** — it does not reset each episode. Trust, dependence, and suppression states carry across episodes. The canonical micro tutor is **BC-ICT-v4** (`internalization_control_tutor_v4.py`).

### 2.2 Macro Layer: Curriculum Controller

**Action space:** $\{\text{lesson}_1, \ldots, \text{lesson}_K,\; \text{EVAL},\; \text{STOP}\}$

| Action | Meaning |
|--------|---------|
| lesson $\ell$ | Select and present a specific teaching episode from family/subtype |
| EVAL | Explicit mastery evaluation — updates mastery belief without teaching |
| STOP | Terminate teaching — no further lessons |

The macro layer is **not** simple schedule dispatch. It is a state-dependent decision controller that uses learner posterior, internalization state, mastery belief, curriculum history, and remaining budget to choose actions.

The canonical macro controller is **CurriculumControllerV13** (`curriculum_controller_v13.py`).

---

## 3. State Representation

### 3.1 Macro State (Controller Input)

$$x_t = (q_t,\; m_t,\; u_t,\; h_t,\; B_t)$$

| Symbol | Name | Description |
|--------|------|-------------|
| $q_t(\theta)$ | Learner posterior | Posterior over learner type $\theta \in \{\text{safe}, \text{shiny}\}$ |
| $m_t$ | Internalization state | 5D factored state tracking trust, dependence, suppression |
| $u_t$ | Mastery state | 5D Beta-Bernoulli estimates of learner competency |
| $h_t$ | Curriculum history | Lessons taught, families used, lesson counts |
| $B_t$ | Remaining budget | Dose/teaching budget left |

### 3.2 Internalization State

$$m_t = (\kappa_t,\; \tau_t,\; \nu_t,\; \gamma_t^{\text{spec}},\; \gamma_t^{\text{gen}})$$

| Dim | Name | Meaning | Pedagogical Valence |
|-----|------|---------|---------------------|
| $\kappa$ | Risk calibration | Accuracy of learner's internal risk estimates | Higher = better |
| $\tau$ | Trust | Confidence in tutor's evidence/warnings | Higher = better |
| $\nu$ | Dependence | Blind following without own evidence | **Higher = worse** |
| $\gamma^{\text{spec}}$ | Specific suppression | Temptation-specific inhibition | Higher = better |
| $\gamma^{\text{gen}}$ | General suppression | Broad exploration inhibition | **Higher = worse** |

**Critical mechanistic distinctions** (established in Stage B):
- **Trust ≠ dependence**: $\tau$ and $\nu$ are separately updated and have opposite pedagogical valence
- **Specific ≠ general suppression**: $\gamma^{\text{spec}}$ is beneficial; $\gamma^{\text{gen}}$ indicates overteaching

### 3.3 Mastery State

$$u_t = (u_t^{RC},\; u_t^{TR},\; u_t^{EP},\; u_t^{VA},\; u_t^{IA})$$

| Dim | Name | What It Measures |
|-----|------|-----------------|
| $RC$ | Risk calibration | Can the learner assess danger? |
| $TR$ | Temptation resistance | Can the learner resist short-term lures? |
| $EP$ | Exploration preservation | Does the learner still explore when appropriate? |
| $VA$ | Valid-advice uptake | Does the learner use correct advice? |
| $IA$ | Invalid-advice resistance | Can the learner reject wrong advice? |

Updated via Beta-Bernoulli model with decay:

$$a_{k,t+1} = \lambda\, a_{k,t} + y_{k,t}, \quad b_{k,t+1} = \lambda\, b_{k,t} + (1 - y_{k,t})$$

$$u_{k,t} = \frac{a_{k,t}}{a_{k,t} + b_{k,t}}$$

---

## 4. Canonical Lesson Ranking & Risk Control

### 4.1 Lesson Objective

$$J_t(\ell) = G_{pw}(x_t,\ell) + \alpha_h\, G_{hier}(x_t,\ell) + \alpha_r\, G_{res}(x_t,\ell) + \lambda^{\text{eff}}_{\text{unc}}(t)\, U_t(\ell) + b^{\text{eff}}_{\text{fam}(\ell)}(q_t, h_t)$$

| Term | Role | Status |
|------|------|--------|
| $G_{pw}$ | Pairwise replay ranking — main sorting signal | **Primary driver** (PCR 79–83%) |
| $G_{hier}$ | Hierarchical empirical Bayes backbone | Low solo PCR but composite contributor |
| $G_{res}$ | Contextual residual — learner-state-dependent correction | Low solo PCR but composite contributor |
| $U_t$ | Uncertainty-driven exploration value | Secondary; budget-decayed |
| $b^{\text{eff}}_{\text{fam}}$ | Family prior with exponential saturation | Learner-conditional curriculum bias |

**Key facts:**
- $G_{pw}$ is what actually changes the lesson argmax
- $G_{hier}/G_{res}$ were confirmed non-dead by ablation (removing them hurts)
- `close-gap` was confirmed dead code and **has been removed** from canonical

### 4.2 Feasible Set (Risk Filter)

$$\mathcal{L}_t^{\text{feas}} = \left\{\ell : \mu_j(x_t,\ell) + \beta_j\,\sigma_j(x_t,\ell) \le \eta_j(x_t, q_t), \;\forall\, j \in \{OTR, \nu, \gamma^g\}\right\}$$

- **Filter, then rank** — not penalty-sum
- First eliminate risk-violating lessons, then rank remaining by gain
- Confirmed superior to penalty-sum across multiple ablations

### 4.3 Theta-Adaptive Risk Budget

$$\eta_j(x_t, q_t) = \sum_\theta q_t(\theta)\,\eta_j^{(\theta)}(x_t)$$

Risk budgets tighten when learner is fragile ($\nu$ high, $\gamma^{\text{gen}}$ high), loosen when learner needs more exploration (low mastery). Per-θ budgets are essential — safe and shiny have different optimal risk tolerances.

### 4.4 Family Prior with Saturation

Raw family prior:

$$b_f(q_t) = \sum_\theta q_t(\theta)\, b_f^{(\theta)}$$

| θ | PP-MRB | TIC | TIC-v4 |
|---|:------:|:---:|:------:|
| safe | 0.00 | 0.00 | +0.10 |
| shiny | +0.20 | −0.30 | +0.25 |

**Exponential saturation** (canonical, decay only):

$$b^{\text{eff}}_f(q_t, h_t) = b_f(q_t) \cdot \exp\!\left(-\frac{n_f(h_t)}{\tau^{(\theta)}_{\text{fam}}}\right)$$

| Parameter | safe | shiny |
|-----------|:----:|:-----:|
| $\tau_{\text{fam}}$ | 3.0 | 2.0 |

**No repetition penalty** in canonical config — verified that decay-only > decay+rep (rep hurts shiny by −4pp).

**Purpose:** Prevent family concentration. Before saturation, shiny had −28pp PP-MRB held-out dependency. After saturation, dependency inverted to +12pp (PP-MRB was over-steering, not necessary).

### 4.5 Exploration Decay

$$\lambda^{\text{eff}}_{\text{unc}}(t) = \lambda_0 \cdot \sigma\!\left(\frac{B_t - B_{\text{mid}}}{\tau_B}\right) \cdot \exp\!\left(-\frac{n_{\text{post}}(x_t,\ell)}{\tau_n}\right)$$

Budget-conditioned (explore more when budget is ample) × maturity-decayed (explore less for well-observed pairs).

---

## 5. Canonical EVAL & STOP

### 5.1 EVAL

EVAL is a **formal curriculum action**, not decoration.

$$J_t(\text{EVAL}) = \lambda_{\text{info}}\,\text{Var}[u_t] - \lambda_{\text{cost}}\, c_{\text{eval}}$$

Its primary value is **not** directly changing the lesson argmax. It works through:

$$u_{t+1} = \text{BayesUpdate}(u_t, y_t^{\text{eval}})$$

which then influences subsequent STOP decisions, feasible sets, and constraint bounds.

**Canonical:** Full EVAL (not probe-only). Probe-only EVAL was experimentally rejected (hurts shiny).

### 5.2 Gated STOP (Canonical)

**Single-threshold STOP was rejected.** It causes premature stopping for shiny learners because their higher $\nu$ and $\gamma^{\text{gen}}$ values inflate the threshold.

**Canonical form:** three-condition gated STOP.

**Margin gate:**

$$M_{\text{base}}(t) = \epsilon_{\text{stop}}(x_t, q_t) - \max\!\Big(J_t(\text{EVAL}),\; \max_{\ell \in \mathcal{L}_t^{\text{feas}}} J_t(\ell)\Big)$$

where the STOP threshold uses per-θ coefficients:

$$\epsilon_{\text{stop}} = \epsilon_0^{(\theta)} + a^{(\theta)}_\nu\,\nu_t + b^{(\theta)}_\gamma\,\gamma_t^{\text{gen}} - c^{(\theta)}_u\,\bar{u}_t - d^{(\theta)}_B\,B_t$$

| Coeff | safe | shiny | Rationale |
|-------|:----:|:-----:|-----------|
| $\epsilon_0$ | 0.00 | −0.10 | Lower intercept for shiny |
| $a_\nu$ | 0.04 | **0.005** | 8× less ν penalty for shiny |
| $b_\gamma$ | 0.05 | **0.005** | 10× less γ penalty for shiny |
| $c_u$ | 0.03 | 0.03 | Same mastery reward |
| $d_B$ | −0.02 | −0.02 | Same budget sensitivity |

**Warm-up gate:**

$$G_{\text{warm}}(t) = \mathbf{1}\!\big[N_{\text{teach}}(t) \ge T_{\min}^{(\theta)}\big]$$

Canonical: $T_{\min}^{(\text{safe})} = 2$, $T_{\min}^{(\text{shiny})} = 3$.

**Plateau gate:**

$$\Delta_u(t) = \sum_k w_k\,(u_t^{(k)} - u_{t-1}^{(k)})$$

$$\bar{\Delta}_u(t) = \frac{1}{w}\sum_{i=t-w+1}^{t} \Delta_u(i)$$

$$G_{\text{plateau}}(t) = \mathbf{1}\!\big[\bar{\Delta}_u(t) \le \tau_u^{(\theta)}\big]$$

Canonical: $w = 2$, $\tau_u^{(\text{safe})} = 0.02$, $\tau_u^{(\text{shiny})} = 0.015$.

**Combined:**

$$\text{STOP} \iff (M_{\text{base}}(t) > 0) \;\land\; G_{\text{warm}}(t) \;\land\; G_{\text{plateau}}(t)$$

> **Status:** Gated STOP is the canonical STOP mechanism. It successfully resolved premature stopping for shiny (shiny #T@STOP went from 2.8 to 4.2). It should not be reverted to single-threshold.

---

## 6. Scenario Families

### 6.1 Family Overview

| Family | Full Name | Primary Mechanism Tested |
|--------|-----------|------------------------|
| **PP-MRB** | Persistent-Profile Mixed-Reveal-Commit | Persistent learner modelling, selective fading, WAIT/WARN selectivity |
| **TIC / TIC-v4** | Teaching-Internalization Corridor | Lesson ranking, EVAL/STOP, curriculum-level behavior, advice validity |
| **CGC-v2** | Compositional Goal Corridor | Coupled posterior, goal-preference conflict, compositional goals |

### 6.2 Key Lesson Subtypes

| Subtype | Family | Tests |
|---------|--------|-------|
| `ppmrb_standard` / `ppmrb_self_discovery` | PP-MRB | Dose calibration, persistent profile |
| `warn_rescue` / `tic_rescue_heavy` | TIC | Trust building, advice validity |
| `tic_temptation` | TIC | Temptation resistance, specific suppression |
| `beneficial_novelty` | TIC | Exploration preservation vs. general suppression |
| `false_suppression_cost` | TIC | Cost of over-suppression |
| `self_discovery_needed` | TIC | Dependence reduction |
| `sparse_valid_advice` / `sparse_invalid_advice` | TIC-v4 | Advice reliability discrimination |

### 6.3 Family Coverage & Dependencies

The system is **not** a single benchmark — it uses multiple scenario families. However, coverage is **not uniform**:

| θ | Most Critical Family | Known Dependencies |
|---|---------------------|-------------------|
| safe | TIC-v4 | 100% of teaching uses TIC-v4; H_fam = 0 |
| shiny | TIC-v4 + PP-MRB | ~69% TIC-v4, ~31% PP-MRB after saturation |

**Key finding:** Family prior saturation resolved the shiny PP-MRB over-dependency.

- Before saturation: shiny PP-MRB held-out = **−28pp** (severe dependency)
- After saturation: shiny PP-MRB held-out = **+12pp** (dependency inverted — PP-MRB was over-steering)

Safe's H_fam = 0 is **not a bug** — safe genuinely only needs TIC-v4 lessons. Saturation is non-disruptive for safe.

---

## 7. Staged Conclusions (Stage 1 → Stage 6.x)

| Stage | Core Question | Key Mechanism | Main Conclusion | Status |
|-------|--------------|---------------|----------------|--------|
| **1** | Can the tutor model learner persistence? | PP-MRB, persistent belief | Tutor **must** model learner latent state; resetting each episode fails | ✅ Stable |
| **2** | Is coupled joint latent needed? | CGC, coupled posterior | Joint latent modelling direction correct; factorized insufficient under conflict | ✅ Stable |
| **3** | Can pairwise ranking actually change argmax? | Pairwise replay, counterfactual labels | PCR went from 0% to 79–83%; active replay is the decisive mechanism | ✅ Stable |
| **4** | Are EVAL/STOP real curriculum actions? | Full EVAL, STOP value | EVAL is genuine action (mastery update pathway); STOP is most valuable single component | ✅ Stable |
| **5** | Is learner-conditional calibration necessary? | θ-adaptive risk budget, OOD | Per-θ calibration is necessary; system shows graceful OOD degradation | ✅ Stable |
| **6** | Can we consolidate into a canonical controller? | Family prior, per-θ STOP, saturation | Family prior + gated STOP + saturation = canonical; PP-MRB dependency resolved | ✅ Canonical |
| **6.5** | Which mechanisms are dead weight? | Ablation suite | close-gap = dead; G_hier/G_res = alive (composite); STOP is θ-conditional | ✅ Done |
| **6.6–6.7** | Per-θ coefficients or per-θ intercept? | v13.1 → v13.2 | Per-θ coefficients needed; family prior + full EVAL confirmed | ✅ Done |
| **6.8+** | Can gated STOP fix premature stopping? | v13.3 gated STOP + saturation | Gated STOP = canonical; saturation resolves family dependency | ✅ **Current** |

---

## 8. Stable vs. Conditional Conclusions

### 8.1 Stable Conclusions (Established Claims)

1. **Persistent learner modelling is necessary** — resetting beliefs each episode destroys selective fading
2. **Curriculum control >> micro-tutor refinement** — marginal value of micro-tutor tuning is small vs. curriculum decisions
3. **STOP is the single most valuable curriculum component** — confirmed across v3, v4, v7, v8, v11, v13
4. **Pairwise replay ranking makes lesson selection actionable** — PCR from 0% to 79–83%
5. **EVAL is a formal curriculum action** — works through mastery belief update, not direct reranking
6. **Risk must use filter+rank, not penalty-sum** — confirmed across multiple ablations
7. **Theta-adaptive calibration is necessary** — safe and shiny need different risk budgets and STOP thresholds
8. **Budget inverted-U is a framework invariant** — confirmed 9+ times across versions and learner types
9. **System degrades gracefully under OOD** — no collapse under sign-flip, noise-heavy perturbations
10. **Framework identity**: non-RL, belief-updating, model-based, bounded-planning — do not change

### 8.2 Conditional Conclusions (With Boundaries)

1. **Gated STOP is canonical but not "final theory"** — it fixes premature stopping; further calibration may refine gate parameters
2. **Safe and shiny respond differently to all mechanisms** — no single parameter set is optimal for both; per-θ treatment is structural, not tuning
3. **Family coverage is improved but not perfectly balanced** — saturation resolved PP-MRB over-dependency, but safe's 100% TIC-v4 usage means family diversity is learner-type-dependent
4. **OTR must be decomposed to interpret** — total OTR ≠ overteaching; eval overhead vs. teaching dose must be separated
5. **G_hier/G_res have low solo PCR but are not dead** — they contribute through composite gain; do not remove based on single-term PCR alone

---

## 9. Canonical Code Paths

### 9.1 Primary (Canonical) Files

| File | Role |
|------|------|
| `src/curriculum/curriculum_controller_v13.py` | **Canonical macro controller** — gated STOP, per-θ coefficients, family-saturated scoring |
| `src/curriculum/family_prior.py` | **Family prior with exp-decay saturation** |
| `src/curriculum/pairwise_response_model.py` | Hierarchical + residual + pairwise response model |
| `src/curriculum/mastery_model.py` | Beta-Bernoulli mastery tracker |
| `src/curriculum/risk_budget_calibration.py` | Theta-adaptive risk budget |
| `src/curriculum/lesson_library_v2.py` | Lesson catalog with families and subtypes |
| `src/curriculum/adaptive_episode_generator_v2.py` | Mastery-conditioned lesson → episode generator |
| `src/curriculum/pedagogical_framework.py` | Unified runtime API (orchestrates macro + micro) |
| `src/agents/internalization_state_v3.py` | Factored internalization state $(κ, τ, ν, γ^s, γ^g)$ |
| `src/agents/behavior_probes.py` | Mechanism-consistent probe battery (RC, TR, EP, VA, IA) |
| `src/agents/trainable_bridge.py` | State → behavior mapping bridge |
| `src/agents/stochastic_agent_policy.py` | Learner choice model with theta-dependent biases |
| `src/teachers/internalization_control_tutor_v4.py` | **BC-ICT-v4 canonical micro tutor** |
| `src/teachers/preference_aware_policy_v2.py` | Stage 1 canonical micro tutor |
| `src/teachers/joint_latent_tutor_v2.py` | Stage 2 canonical joint tutor |

### 9.2 Legacy / Ablation Files

Files like `curriculum_controller_v8.py`, `curriculum_controller_v10.py`, `curriculum_controller_v11.py` are **historical Pareto-frontier versions** retained for regression comparison and ablation baselines. They should **not** be used as the default path.

---

## 10. Key Experimental Evidence Patterns

### 10.1 Persistent Learner Modelling & Selective Fading

The tutor must remember what it has already taught. Without persistent belief, the system repeats interventions for already-internalized skills. PP-MRB family first demonstrated "learned → talk less" selectivity: once a learner type has mastered a risk concept, the tutor selectively reduces warnings for that dimension while maintaining them for unmastered dimensions.

### 10.2 Pairwise Replay: From Concept to Actionability

The transition from PCR ≈ 0% (v9) to PCR = 79–83% (v12+) was the defining actionability breakthrough. The key unlock was **active counterfactual replay** — generating synthetic pairwise training data across diverse `(state, lesson)` combinations rather than relying solely on online trajectories where most comparisons share the same macro-state.

### 10.3 EVAL & STOP: Curriculum Actions, Not Side Effects

EVAL's value is **not** measured by rank-change. Its mechanism is belief update: after EVAL, the mastery posterior $u_{t+1}$ becomes more precise, which changes STOP eligibility, feasible sets, and constraint bounds. Rank-change ≈ 0 is expected and correct behavior.

STOP has been the single most valuable curriculum component since v3. The core insight is: **stopping at the right time > choosing the perfect lesson**. Overtesting is mildly wasteful; over-teaching is actively harmful (increases $\nu$ and $\gamma^{\text{gen}}$).

### 10.4 Gated STOP: Fixing Premature Termination

Single-threshold STOP was too aggressive for shiny learners because their higher $\nu$ and $\gamma^{\text{gen}}$ values inflate the threshold. Gated STOP resolved this:
- shiny #T@STOP increased from 2.8 → 4.2
- shiny C improved from 41% → 47% vs v13 baseline

The warm-up gate ensures minimum exposure before any stopping is allowed. The plateau gate ensures mastery has genuinely plateaued.

### 10.5 Family Saturation: Resolving Over-Dependency

Without saturation, shiny's lesson selection was dominated by PP-MRB (50% of selections), creating a −28pp held-out dependency. After exponential decay saturation:
- PP-MRB usage dropped from 50% to 31%
- Held-out dependency inverted from −28pp to +12pp
- This means PP-MRB was over-steering shiny, not vital for it
- Safe was completely unaffected (all 3 saturation variants = same C)

### 10.6 OOD & Robustness

Under adversarial OOD conditions (sign-flip, noise-heavy), the canonical controller shows **graceful degradation without collapse**:

| θ | ID C | sign_flip C | noise_heavy C |
|---|:----:|:-----------:|:-------------:|
| safe | 47% | 56% (+9) | 75% (+28) |
| shiny | 47% | 34% (−13) | 44% (−3) |

---

## 11. Known Deficiencies

1. **STOP calibration is ongoing** — gated STOP is canonical but gate parameters (T_min, τ_u, per-θ coefficients) may benefit from further tuning
2. **Family usage balance is learner-type-dependent** — safe uses only TIC-v4 (H_fam = 0); shiny uses TIC-v4 + PP-MRB. This is structural, not a bug, but limits family generalization claims for safe
3. **Safe/shiny trade-offs are inherent** — improvements for one may come at small cost to the other (e.g., gated STOP: safe +6pp, shiny =0 in some seeds)
4. **OTR must be decomposed** — total OTR includes EVAL overhead; teaching-only OTR vs eval overhead should be reported separately
5. **G_hier/G_res contribute through composition** — their solo PCR is near 0, but removing them hurts. Do not interpret low solo PCR as "dead code"
6. **Sample variance across seeds** — some metrics show 5–15pp variance across seed sets (e.g., shiny C ranges 41–66% across different experiments). Mean trends are stable but individual runs vary

---

## 12. Recommended Next Steps

The system is at the point where the next step is **not** inventing new modules. It is convergence and documentation:

> **Primary goal:** Lock canonical configuration, complete robustness verification, and prepare the framework for publication/handoff.

### 12.1 Immediate Tasks

1. **Final STOP calibration** — confirm gated STOP parameters are stable across more seeds; consider whether T_min_shiny should be 3 or 4
2. **Post-STOP counterfactual audit** — at each STOP point, compute best-available-lesson value to confirm STOP regret is low
3. **Cross-family gated STOP check** — verify gated STOP on held-out families (already partially done)
4. **Documentation** — write the paper narrative connecting Stages 1–6 into a coherent scientific story

### 12.2 Do NOT Do

- Convert to RL or exact POMDP
- Rewrite the controller head
- Increase map complexity to "solve" curriculum problems
- Delete family prior or EVAL
- Revert to single-threshold STOP
- Add new mechanism modules before stabilizing existing ones

---

## Appendix: Canonical Configuration Summary

```
Controller: CurriculumControllerV13 (v13.3)
  Ranking:      G_pw + α_h·G_hier + α_r·G_res + λ_unc·U + b_eff_fam
  Risk:         Filter+rank (not penalty-sum), θ-adaptive budgets
  close-gap:    REMOVED (confirmed dead code)
  
STOP: Gated (M_base > 0) ∧ G_warm ∧ G_plateau
  T_min:        safe=2, shiny=3
  plateau:      window=2, τ_safe=0.02, τ_shiny=0.015
  coefficients: per-θ (shiny: a_ν=0.005, b_γ=0.005)

EVAL: Full EVAL (not probe-only)
  λ_info=0.8, λ_cost=0.05

Family Prior: Exponential decay saturation (no rep penalty)
  τ_fam_safe=3.0, τ_fam_shiny=2.0
  Priors: safe(TIC-v4=+0.10), shiny(PP-MRB=+0.20, TIC=−0.30, TIC-v4=+0.25)

Micro Tutor: BC-ICT-v4
Agent: FactoredInternalizationState + stochastic_agent_policy
Mastery: Beta-Bernoulli with decay (5 dimensions: RC, TR, EP, VA, IA)
```

### Key Unresolved Issues

1. STOP gate parameters may need minor recalibration
2. Safe family entropy = 0 (structural, not bug)
3. Seed variance in shiny C is 5–15pp

### Recommended First Experiment for Next Session

```
Exp: Same-dose fair comparison under canonical
  v13 (shared STOP, no FP) vs canonical (gated STOP + saturated FP)
  Fixed dose, NS=16+ seeds
  Report: C, E, OTR, #T, H_fam, per-family usage
```
