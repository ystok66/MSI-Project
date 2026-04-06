# Stage 6 Summary: POMDP Intervention Stack + Compositional Goals

> **Handoff document** — self-contained reference for the next agent or collaborator.
> Covers Tasks 3–7: POMDP interface refactoring, intervention semantic expansion, hidden information inference, consequence-grounded planning, and compositional goal integration.
> Does not require conversation history to understand.

---

## 1. Stage Identity & Goal

This stage builds the **decision-theoretic intervention layer** on top of Stage 5's frozen 5D observer architecture. The goal:

> **Enable the robot tutor to (a) maintain structured beliefs over agent hidden states (goals, preferences, temptation), (b) select interventions with grounded consequence modeling, and (c) handle compositional goals — all without modifying the frozen canonical micro tutor or 5D observer.**

The stage operates under strict **shadow / minimal-diff** principles: new modules sit on top of existing architecture, no frozen module is reopened, and all changes are validated via regression.

---

## 2. Task Lineage & Status

| Task | Title | Status | Tests | Key Contribution |
|:----:|-------|:------:|:-----:|------------------|
| **3** | Belief / ToM / POMDP Interface | ✅ Closed | 101 | 4-layer state separation: WorldState / AgentBelief / RobotBeliefOverAgent / ActionPredictor |
| **4** | Intervention Semantic Expansion | ✅ Closed | 52 | UNLOCK / ITEM_DROP / WARN option layer; family-selective intervention |
| **5** | Hidden Information & Consequence Grounding | ✅ Closed | 21 | GoalTemptationPosterior q(g,z); ConsequenceGroundedRollout |
| **7** | Compositional Goals + Multi-Type Posterior | ✅ Conditionally Closed | 52 | q(g,θ,z) joint posterior; CGC-v2 bridge; goal-conditional curriculum |
| | | **Total** | **226** | |

---

## 3. Architecture: POMDP Intervention Stack

```
┌───────────────────────────────────────────────────────────────────┐
│  Layer 4 (NEW): Compositional Goal Hypothesis Space              │
│  G = {4 atomic + 4 composite} goals from CGC-v2                  │
│  compositional_goal_hypotheses.py                                 │
├───────────────────────────────────────────────────────────────────┤
│  Layer 3b (NEW): Joint Posterior q(g, θ, z)                      │
│  (8 goals × 2-5 prefs × 1-4 tempt)                              │
│  + compatibility bonus exp(β_C · C(g))                           │
│  joint_goal_pref_posterior.py + composite_goal_compatibility.py   │
├───────────────────────────────────────────────────────────────────┤
│  Layer 3a (NEW): Goal-Conditional Curriculum                     │
│  S(ℓ) = E_q[lift] + λ·teach - λ·infl + β_κ·κ̂                  │
│  goal_conditional_curriculum_hook.py + compositional_goal_bridge  │
├───────────────────────────────────────────────────────────────────┤
│  Layer 2b (NEW): Consequence-Grounded Option Control (T4+T5)     │
│  WARN → ↑risk_penalty, UNLOCK → ↓risk, ITEM_DROP → ↓risk        │
│  consequence_grounded_option_rollout.py                           │
│  intervention_semantics.py + option_intervention_controller.py    │
├───────────────────────────────────────────────────────────────────┤
│  Layer 2a (NEW): POMDP 4-Layer State (T3)                        │
│  WorldState / AgentBelief / RobotBeliefOverAgent / ActionPredictor│
│  world_state.py / agent_belief_state.py / action_predictor.py    │
├───────────────────────────────────────────────────────────────────┤
│  Layer 1 (FROZEN from Stage 5): 5D Observer                      │
│  m̂_t = (τ̂, ν̂, γ̂_gen, γ̂_spec, κ̂)                              │
│  internalization_observer.py (A1MtObserverFrozen)                 │
├───────────────────────────────────────────────────────────────────┤
│  Layer 0 (FROZEN from Stage 5): Micro Tutor                      │
│  A_micro = {WAIT, WARN}   (2-act canonical)                      │
│  internalization_control_tutor_v4.py (BCICTv4)                    │
└───────────────────────────────────────────────────────────────────┘
```

### Design Principle

**"Sit on top, don't reach in."** Every new module in this stage reads from the frozen 5D observer and micro tutor outputs. None modifies their internals.

$\hat\kappa$ enters macro scoring as an additive term, NOT as a hidden posterior latent:

$$x_t^{macro} = \big(q_t(g,\theta,z),\; \hat\tau_t,\hat\nu_t,\hat\gamma_t^{gen},\hat\gamma_t^{spec},\hat\kappa_t\big)$$

---

## 4. Task 3: POMDP Interface Refactoring

### 4.1 Core Contribution

Separated the monolithic agent state into four distinct layers:

| Layer | Class | Semantic |
|-------|-------|----------|
| **World State** | `WorldState` | Public observable state of the environment |
| **Agent Belief** | `AgentBelief` | Agent's own beliefs about the world |
| **Robot Belief over Agent** | `RobotBeliefOverAgent` | Robot's posterior over agent's hidden state |
| **Action Predictor** | `ActionPredictor` | Bounded-rational policy model P(a|s,θ) |

### 4.2 Validation

- Shadow parity: existing tutor behavior preserved exactly
- OOD parity: robust across perturbation conditions
- 101 tests covering POMDP interface, shadow bridge, followup, and observer

---

## 5. Task 4: Intervention Semantic Expansion

### 5.1 Core Contribution

Extended intervention vocabulary from canonical `{WAIT, WARN}` to a family-selective option layer:

| Intervention | Semantic | Best Family | Mechanism |
|:------------:|----------|:-----------:|-----------|
| **WARN** | Belief evidence | fork_trap | ↑ risk_penalty on risky branches |
| **UNLOCK** | Affordance expansion | deadline_gate | ↓ risk_penalty, unlock paths |
| **ITEM_DROP** | Traversal mitigation | hazard_belt | ↓ risk for traversal |

### 5.2 Key Design Rule

**Canonical micro stays at 2-act.** UNLOCK and ITEM_DROP live in the option layer (`OptionInterventionController`), not in the micro tutor.

### 5.3 Family Selection Gap (SelGap)

| Family | Best Option | SelGap |
|--------|:-----------:|:------:|
| fork_trap | WARN | +0.05 |
| hazard_belt | ITEM_DROP | +0.03 |
| deadline_gate | UNLOCK | +0.04 |

---

## 6. Task 5: Hidden Information & Consequence Grounding

### 6.1 GoalTemptationPosterior

Factorized Bayesian posterior over hidden agent goals and temptation:

$$q_t(g, z) \propto q_{t-1}(g, z) \cdot P(a_t^{obs} | s_t, g, z)$$

- **Goal**: `{true_goal, decoy_goal}` — 2 discrete hypotheses
- **Temptation grid**: $z \in \{0.0, 0.3, 0.6, 0.9\}$ with low-temptation prior

### 6.2 ConsequenceGroundedRollout

Maps intervention effects into `BranchAttributes`, enabling counterfactual action prediction:

| Option | Effect on BranchAttributes |
|:------:|---------------------------|
| WARN | ↑ risk_penalty on risky branches; ↑ safety on safe branches |
| UNLOCK | ↓ risk_penalty (makes paths accessible) |
| ITEM_DROP | ↓ risk (hazard mitigation) |

### 6.3 Key Result

SR landscape is **no longer flat**: WARN (0.523) > NONE (0.451) > ITEM_DROP (0.369) for temptation-seeking agents.

### 6.4 Inflation Decomposition

$$\Delta\hat\nu / n_{int} \approx 0.028$$ constant across all policies → inflation is a per-intervention cost of the frozen observer, not the controller.

---

## 7. Task 7: Compositional Goals + Multi-Type Posterior

### 7.1 Goal Hypothesis Space

| Type | Goals | Source |
|------|:-----:|--------|
| Atomic | collect_red, avoid_blue, use_safe, reach_fast | CGC-v2 |
| Composite | collect_red+avoid_blue, collect_red+use_safe, avoid_blue+use_safe, reach_fast+avoid_blue | Valid conjunctions |
| **Total** | **8** | |

Each goal has a 4D reward weight vector from `GOAL_WEIGHTS`:

```python
"collect_red":  [0.0,  2.5, 0.5, 0.0]   # temptation-seeker
"avoid_blue":   [2.0, -1.0, 0.0, 0.0]   # risk-averse
"use_safe":     [3.0, -0.5, 0.0, 0.0]   # safety priority
"reach_fast":   [0.0,  0.0, 0.0, 3.0]   # speed priority
```

### 7.2 Joint Posterior q(g, θ, z)

Full Bayesian posterior over goal × preference × temptation:

$$q_t(g,\theta,z) \propto q_{t-1}(g,\theta,z) \cdot P(a_t^{obs}|s_t,g,\theta,z) \cdot \exp(\beta_C \cdot C_t(g))$$

| Component | Size | Description |
|-----------|:----:|-------------|
| Goals $\mathcal{G}$ | 8 | 4 atomic + 4 composite |
| Preferences $\Theta_2$ | 2 | safe, shiny (canonical) |
| Preferences $\Theta_K$ | 5 | safe, shiny, risky, shortcut, neutral (research) |
| Temptation $Z$ | 4 | {0.0, 0.3, 0.6, 0.9} |
| **Grid size (canonical)** | **64** | 8 × 2 × 4 |
| **Grid size (K-type)** | **160** | 8 × 5 × 4 |

### 7.3 Composite Goal Compatibility

Structural prior to improve composite-goal discrimination:

$$C_t(g) = \text{progress}(g) - \lambda_{red} \cdot \text{redundancy}(g) - \lambda_{comp} \cdot (|g|-1)$$

| Term | Meaning |
|------|---------|
| progress(g) | Mean per-subgoal log-likelihood from accumulated observations |
| redundancy(g) | KL-based penalty when subgoals predict identical action distributions |
| complexity penalty | $\lambda_{comp} \cdot (|g|-1)$: atomics get 0, composites get penalized |

### 7.4 Goal-Conditional Curriculum Hook

Macro intervention scoring becomes goal-conditional:

$$S_{macro}(\ell) = \mathbb{E}_{q(g,\theta,z)}\big[\text{success\_lift}(\ell|g,\theta,z)\big] + \lambda_{teach} \cdot V_{teach} - \lambda_{infl} \cdot R_{infl} + \beta_\kappa \cdot g_\kappa(\hat\kappa_t)$$

$\hat\kappa_t$ enters as an **additive** macro state term, NOT as a posterior latent.

### 7.5 Key Experimental Results

#### Atomic Goal Recovery

| Goal | θ | Config | Goal Acc |
|:----:|:-:|:------:|:--------:|
| collect_red | safe | baseline | 0.73 |
| collect_red | safe | +compat | **1.00** ✅ |
| collect_red | shiny | baseline | 1.00 |
| use_safe | safe | baseline | 0.60 |

#### Composite Goal: Observational Equivalence Finding

> **Critical diagnosis**: exact composite label recovery is fundamentally limited by **observational equivalence**. `avoid_blue` and `use_safe` both produce identical safe-branch-choosing actions. Their conjunction is indistinguishable from either atomic goal via action-only evidence.

Correct metric: **subgoal marginals** $q(u) = \sum_{g \ni u} q(g)$:

| Goal | Subgoal Marginal Acc |
|:----:|:--------------------:|
| avoid_blue+use_safe | **1.00** |
| collect_red+avoid_blue (shiny) | **0.82** |
| collect_red+use_safe | 0.50 |

#### q(g,θ) vs q(g,θ,z) Ablation

| z_true | q(g,θ) NLL | q(g,θ,z) NLL | Δ |
|:------:|:----------:|:------------:|:-:|
| 0.0 | 0.464 | 0.459 | −0.005 (zero cost) |
| 0.6 | 0.717 | 0.806 | +0.088 (expected) |

#### Θ₂ vs Θ_K under Compositional + Temptation

| Condition | Θ₂ NLL | Θ_K NLL | Improvement |
|-----------|:------:|:-------:|:-----------:|
| collect_red, shiny, z=0.3 | 0.876 | **0.777** | **+11%** |
| composite, shiny, z=0.3 | 0.816 | **0.666** | **+18%** |

> **First evidence Θ_K has value** on difficult cases. Not yet promoted to canonical.

#### Stability

| Check | Result |
|-------|:------:|
| q(g,θ,z) collapse check | ✅ No collapse (entropy 2.722) |
| E[z] tracks z_true | ✅ (0.42–0.55 at z_true=0.6) |
| Overall subgoal marginal acc | **0.747** |

---

## 8. Complete Module Inventory (Stage 6)

### 8.1 POMDP Interface (Task 3)

| File | Role | Lines |
|------|------|:-----:|
| `src/agents/world_state.py` | Public observable state | ~60 |
| `src/agents/agent_belief_state.py` | Agent's own beliefs | ~80 |
| `src/agents/robot_belief_over_agent.py` | Robot's posterior over agent | ~90 |
| `src/teachers/action_predictor.py` | Bounded-rational policy model | ~100 |
| `src/teachers/shadow_bridge.py` | Shadow parity adapter | ~120 |

### 8.2 Intervention Expansion (Task 4)

| File | Role | Lines |
|------|------|:-----:|
| `src/teachers/intervention_semantics.py` | WARN/UNLOCK/ITEM_DROP type definitions | ~100 |
| `src/teachers/option_intervention_controller.py` | Family-selective option controller | ~180 |

### 8.3 Hidden Info & Consequence Grounding (Task 5)

| File | Role | Lines |
|------|------|:-----:|
| `src/teachers/goal_temptation_posterior.py` | q(g,z) Bayesian posterior | ~180 |
| `src/teachers/consequence_grounded_option_rollout.py` | Intervention → BranchAttributes | ~180 |

### 8.4 Compositional Goals (Task 7)

| File | Role | Lines |
|------|------|:-----:|
| `src/teachers/compositional_goal_hypotheses.py` | Goal hypothesis space (8 goals) | ~170 |
| `src/teachers/joint_goal_pref_posterior.py` | q(g,θ,z) + compatibility | ~295 |
| `src/teachers/composite_goal_compatibility.py` | Structural prior for composites | ~170 |
| `src/teachers/compositional_goal_bridge.py` | CGC-v2 → POMDP adapter | ~130 |
| `src/teachers/goal_conditional_curriculum_hook.py` | Macro scoring with κ̂ | ~155 |

### 8.5 Pre-Existing Environments Used

| File | Role |
|------|------|
| `src/envs/cgc_v2_family.py` | CGC-v2 compositional goal corridor |
| `src/envs/compositional_goal_corridor_v2.py` | Factor-vector CGC2 with train/heldout pools |
| `src/agents/joint_posterior_v2.py` | Legacy joint posterior (not used in new stack, kept for reference) |
| `src/agents/goal_posterior_v1.py` | Legacy 5-type goal posterior (reference) |

### 8.6 Test Files

| File | Tests | Coverage |
|------|:-----:|---------|
| `test_pomdp_interface.py` | ~20 | WorldState, AgentBelief, ActionPredictor |
| `test_shadow_bridge.py` | ~15 | Shadow parity |
| `test_t3_followup.py` | ~20 | T3 follow-up validation |
| `test_internalization_observer.py` | 55 | 5D observer (pre-existing from Stage 5) |
| `test_intervention_semantics.py` | ~12 | WARN/UNLOCK/ITEM_DROP |
| `test_option_intervention_controller.py` | ~15 | Family-selective controller |
| `test_intervention_policy.py` | ~10 | Intervention policy |
| `test_interventions_api.py` | ~8 | API surface |
| `test_item_drop.py` | ~7 | ITEM_DROP mechanics |
| `test_goal_temptation_posterior.py` | 11 | q(g,z) posterior |
| `test_consequence_grounded_option_rollout.py` | 10 | Consequence grounding |
| `test_compositional_goal_hypotheses.py` | 13 | Goal hypothesis space |
| `test_joint_goal_pref_posterior.py` | 14 | q(g,θ,z) joint posterior |
| `test_compositional_goal_bridge.py` | 12 | Bridge + curriculum |
| `test_composite_goal_compatibility.py` | 13 | Compatibility prior |
| **Total** | **226** | |

### 8.7 Experiment Scripts & Reports

| Script | Report | Experiment |
|--------|--------|-----------|
| `run_t3_exp3c_saturation_audit.py` | `t3_*.md` | POMDP shadow parity |
| `run_t7_compositional_posterior_audit.py` | `t7_compositional_posterior_audit.md` | T7-1/2/3: atomic vs composite, Θ₂ vs Θ_K |
| `run_t7b_composite_recovery_audit.py` | `t7b_composite_recovery_audit.md` | T7-B1: compatibility prior |
| `run_t7b2_compositional_temptation_audit.py` | `t7b2_compositional_temptation_audit.md` | T7-B2: q(g,θ,z) stability |
| Several T5 scripts | `t5_*.md` | Hidden temptation + consequence grounding |

---

## 9. Frozen vs. Active vs. Deferred

### Frozen (DO NOT MODIFY)

| Module | Reason |
|--------|--------|
| `internalization_observer.py` (A1Frozen 5D) | Stage 5 canonical; all new modules read from it |
| `internalization_control_tutor_v4.py` (2-act) | SOFT proven redundant; micro Q unchanged |
| `internalization_state_v3.py` | True state unchanged |
| `stochastic_agent_policy.py` | Utility computation unchanged |
| Task 4 intervention semantics | Stable, tested |
| Task 5 consequence grounding | Used by T7, not modified |

### Active (Current Main Line)

| Module | Status |
|--------|--------|
| q(g,θ,z) joint posterior | Canonical for compositional settings |
| Compositional goal bridge | CGC-v2 → POMDP adapter |
| Goal-conditional curriculum | Macro scoring with posterior + κ̂ |
| Compatibility prior | Structural composite-goal discrimination |

### Deferred / Pending Decision

| Module | Reason |
|--------|--------|
| Θ_K canonical promotion | First evidence of value (+11-18% NLL) but needs more data |
| Particle posterior fallback | Not needed at current grid size (64–160 cells) |
| κ̂ as posterior latent | Kept in macro state per "estimate first" principle |
| Full CGC-v2 pipeline integration | Current validation uses synthetic branches |
| γ̂_spec macro utility | No clean design yet (same as Stage 5) |

---

## 10. Known Deficiencies & Open Questions

1. **Exact composite label recovery is inherently limited** by observational equivalence. Safety-aligned atomics produce identical action traces to their conjunction. Subgoal marginals are the correct reporting metric.

2. **Θ_K shows promise but is not canonical.** On difficult cases (shiny + temptation), K-type improves NLL by 11–18%. But on simple cases it adds entropy without benefit. Needs systematic held-out evaluation before promotion.

3. **q(g,θ,z) NLL is slightly higher than q(g,θ) at high temptation** (+0.088 at z=0.6). Expected: larger hypothesis space = harder to concentrate. The tradeoff is temptation tracking capability vs. marginal NLL cost.

4. **Inflation remains a structural cost.** $\Delta\hat\nu / n_{int} \approx 0.028$ is constant — an inherent per-intervention cost of the frozen observer, not the controller. Cannot be eliminated without reopening the observer.

5. **Pre-existing over-warn** in 3 families (from Stage 5) is unchanged. Caused by near-tie Q boundaries in the frozen micro tutor, not by Stage 6 additions.

6. **Synthetic branches for validation.** Current compositional goal experiments use hand-crafted `BranchAttributes`, not the full CGC-v2 scenario generation pipeline. Full integration testing is deferred.

---

## 11. Scientific Conclusions

### 11.1 What Stage 6 Proves

1. **POMDP interface separation works.** 4-layer state decomposition preserves all existing behavior (shadow parity) while enabling modular extension.

2. **Family-selective intervention is better than uniform.** Different families have different best interventions: WARN for fork_trap, UNLOCK for deadline_gate, ITEM_DROP for hazard_belt. Positive SelGap confirmed.

3. **Consequence grounding breaks flat SR landscapes.** By routing intervention effects through BranchAttributes → ActionPredictor, the tutor can now counterfactually predict how each option changes agent behavior. SR is no longer flat.

4. **Compositional goal inference is tractable** via exact discrete posterior on moderate grid sizes (64–160 cells). No particle filter needed yet.

5. **Observational equivalence is the true bottleneck** for composite label identification. This is a fundamental property of action-only inverse planning, not a module design failure.

6. **Multi-factor hidden state (g,θ,z) coexists stably.** No posterior collapse. Temptation tracking works. Subgoal marginals remain interpretable.

### 11.2 What Stage 6 Does NOT Prove

1. Θ_K > Θ₂ in general (only on specific hard cases so far).
2. Exact composite label recovery beyond observational equivalence.
3. κ̂ needs to enter the posterior (it doesn't; macro state is sufficient).
4. The full CGC-v2 pipeline works end-to-end with the new posterior.

---

## 12. Recommended Next Steps

### 12.1 Immediate

1. **Θ_K promotion experiment**: systematic held-out evaluation across all families and conditions to decide canonical vs shadow.
2. **Full CGC-v2 integration test**: connect `generate_cgc_episode_scenario()` → posterior → curriculum hook end-to-end.
3. **Subgoal marginal as canonical metric**: formalize $q(u) = \sum_{g \ni u} q(g)$ as the primary compositional goal reporting standard.

### 12.2 Next Research Phase

1. **Persistent profile**: multi-session learner tracking with compositional goals.
2. **Deep RSA integration**: warning subtypes into recursive ToM.
3. **Joint planner**: full joint optimization over option + utterance + goal_posterior.
4. **κ̂ directional split**: $g_\kappa^+$ vs $g_\kappa^-$ for different risk lesson subtypes.

### 12.3 Do NOT Do

- Do not reopen frozen micro tutor or observer
- Do not put κ̂ into the posterior without specific evidence it helps
- Do not optimize for exact composite label top-1 (observational equivalence makes this the wrong metric)
- Do not explode the latent grid beyond tractable exact inference without strong evidence for particles
- Do not modify WARN/UNLOCK/ITEM_DROP semantics from Task 4

---

## Appendix: Canonical Configuration Summary

```
Stage 5 (FROZEN):
  Observer: A1MtObserverFrozen (5D)
    Layer 1: (τ̂, ν̂, γ̂_gen, γ̂_spec, κ̂)
    Layer 2: (τ̂, ν̂, γ̂_gen)              ← micro Q input
    Layer 3: full 5D                       ← macro/diagnostic
  
  Micro Tutor: BCICTv4 (2-act)
    Action space: {WAIT, WARN}

Stage 6 (NEW):
  POMDP Interface:
    WorldState / AgentBelief / RobotBeliefOverAgent / ActionPredictor
  
  Intervention Options:
    {WARN, UNLOCK, ITEM_DROP} via OptionInterventionController
    Canonical micro still {WAIT, WARN}
  
  Hidden State Posterior:
    q(g, θ, z) ∝ q_{t-1} · P(a|g,θ,z) · exp(β_C · C(g))
    Goal space: 8 (4 atomic + 4 composite from CGC-v2)
    Preference: Θ₂ = {safe, shiny} (canonical), Θ_K = {safe, shiny, risky, shortcut, neutral} (research)
    Temptation: z ∈ {0.0, 0.3, 0.6, 0.9}, prior = (0.4, 0.3, 0.2, 0.1)
  
  Compatibility Prior:
    β_compat = 0.5, λ_comp = 0.3, λ_redund = 0.2
  
  Goal-Conditional Curriculum:
    S(ℓ) = E_q[lift] + λ_teach·V_teach - λ_infl·R_infl + β_κ·g_κ(κ̂)
    κ̂ enters as additive macro state, NOT posterior latent
  
  Tests: 226/226 passing
```
