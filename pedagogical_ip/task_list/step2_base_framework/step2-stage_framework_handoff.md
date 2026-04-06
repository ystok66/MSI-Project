# Pedagogical Decision Framework — Stage Handoff

> **Document type**: Technical handoff / source-of-truth for next-session agent.
> **Date**: 2026-03-24
> **Status**: Framework-level verification complete; multi-latent inference operational; next phase = persistent profile + compositional goals.

---

## 1. Project Purpose and Current Thesis

This project studies **pedagogical robot assistance** in a partially observable grid/lattice environment. The central research question:

> When a robot tutor assists a learning agent, should it optimize for **current task completion** or for the agent's **long-term ability to independently complete future tasks**? Under finite time and safety constraints, how should the tutor balance these two objectives?

### Environment and setup

- **World**: Partially observable grid/lattice with latent cost and risk functions over cell features.
- **Agent**: Learns latent cost/risk mappings via experience. Observes a local patch around its position. Must navigate from start to goal through branching corridors.
- **Robot tutor**: Observes more than the agent. Intervenes via `WAIT / WARN / UNLOCK / ITEM_DROP`.
- **Evaluation**: Both online success (Safe Branch Choice Rate, SBCR) and post-training no-tutor transfer (`LG(k)`, `PE(k)`).

### Architectural identity

> **This system is NOT reinforcement learning.** It is a structured **belief-updating, model-based, bounded-planning** pedagogical tutoring framework. The tutor does not learn a policy via reward maximization. Instead, it reasons explicitly about what the agent knows, what information exists, and whether intervention would help or hinder long-term learning.

This architectural identity is a core stable conclusion and must not be changed.

---

## 2. System Architecture (Current Stable Stack)

### 2.1 Runtime data flow

The system is organized as a 6-layer chain of dataclasses defined in `src/core/state_types.py`:

```
Layer 1: WorldState              (ground truth — agent MUST NOT access)
    ↓
Layer 2: AgentObservation        (what agent actually sees at timestep t)
    ↓
Layer 3: AgentBelief             (agent's internal model: risk/cost heads, branch summaries, concepts)
    ↓
Layer 5: BranchPosterior         (posterior over branch safety — shared by planner + tutor)
    ↓
Layer 4: RobotBeliefOnAgent      (tutor's estimate of agent's mental state)
    ↓
Layer 6: TutorDecisionTrace      (complete serializable record of why tutor chose WAIT or WARN)
```

Composed as:

```
WorldState → AgentObservation → AgentBelief
WorldState + ObsHistory → RobotBeliefOnAgent
AgentBelief → BranchPosterior → PlannerChoice
RobotBeliefOnAgent + BranchPosterior → TutorDecisionTrace
```

The adapter layer (`src/core/adapters.py`) bridges legacy runner/planner code to this runtime API.

### 2.2 Component inventory

| Component | Purpose | Key module(s) |
|-----------|---------|---------------|
| Latent world semantics | Ground-truth cost/risk from d-dim features | `cost_risk_model.py`, `semantic_subspace.py` |
| Patch observation | Agent sees local obs_radius around position | `observation_mask.py` |
| Feature belief map | Agent's running estimate of cell features | Maintained in trial loop |
| Linear cost/risk heads | Agent's learned c(z), r(z) | `cost_risk_model.py` (`LatentCostRiskHead`) |
| Branch summary | Aggregate branch-level statistics from cell beliefs | `branch_summary.py` |
| Branch concepts | Prototype library of branch patterns | `branch_concepts.py` |
| Branch scorer | Probe that predicts branch safety from summary+concepts | `branch_scorer_probe.py` |
| Branch-aware planner | Cell-level J(π) augmented with λ_b · S_branch | `branch_reranker.py` |
| RSA warning | Referential warning grounded in branch semantics | `rsa_warning.py` |
| Tutor v4 | Smooth selectivity law (p_self + DVOI) | `learning_aware_policy_v4.py` |
| Tutor pref_v2 | v4 + ΔPrefInfo + observation value | `preference_aware_policy_v2.py` |
| Tutor goal_v1 | v4 + ΔGoalInfo (factorized) | `goal_preference_aware_policy_v1.py` |
| Tutor joint_v1 | Nonmyopic tutor with coupled q(g,θ) | `joint_latent_tutor_v1.py` |
| Stochastic agent | Softmax + lapse bounded-rational policy | `stochastic_agent_policy.py` |
| Preference posterior | Bayesian q(θ) from behavior | `preference_posterior_v2.py` |
| Goal posterior | Bayesian q(g) from behavior | `goal_posterior_v1.py` |
| Joint factorized | q(g)·q(θ) scaffold | `joint_latent_belief.py` |
| Joint coupled | Full q(g,θ) table with compatibility prior | `joint_posterior_v2.py` |
| Unified pipeline | Config-driven experiment runner | `src/evals/pipeline.py` |

### 2.3 Future hooks (already in place)

`WorldState` includes:
- `latent_goal_vector: Optional[np.ndarray]`
- `latent_preference_vector: Optional[np.ndarray]`
- `hidden_temptation_cells: list`

These are placeholders for compositional goal and temptation integration.

---

## 3. Core Mathematical Objects and Formulas

### 3.1 Cell-level planner (canonical formula)

```
J(π) = Σ_{i∈π} [
    λ_c · ĉ_i
  + φ(r̂_i) · (α + (1-α)(1-n(π)))
  + λ_{uc} · (1-n(π)) · u_i^(c)
  + λ_{ur} · (1-n(π)) · u_i^(r)
  - λ_m · m_i
]
```

Where:
- `φ(r) = -ln(1-r)` — risk penalty (log-barrier)
- `α = min(1, n_updates/10)` — prior-to-posterior blend
- `n(π)` — route necessity (0 = can avoid, 1 = must traverse)

This is the canonical planner foundation. It has been stable since the earliest phases and should not be modified.

### 3.2 Intervention semantics (strict 3-way split)

| Intervention | Semantic layer | Effect |
|-------------|----------------|--------|
| **WARN** | Belief / evidence | Updates agent's feature estimates via semantic communication; changes what agent *believes* about a branch |
| **UNLOCK** | Topology / affordance | Removes gates or reveals passability; changes what agent *can reach* |
| **ITEM_DROP** | Traversal / outcome mitigation | Reduces effective traversal cost; changes the *consequences* of a path |

> **Critical constraint**: These three must remain strictly separated. Do NOT collapse them into "all modify risk." Each operates on a different layer of the agent's decision problem.

### 3.3 Branch-aware planner

```
J_hybrid(π) = J_cell(π) - λ_b · S_branch(π)
```

Where `S_branch(π)` is the branch-level safety score from the scorer probe.

**Why this matters**: This was the key breakthrough that resolved the prediction→planning interface gap. Before this, the agent could learn correct branch representations but the planner never queried them. Once `λ_b · S_branch` was added, transfer on topology-neutral families (ELCB) immediately recovered.

### 3.4 Stochastic bounded-rational agent

The agent selects branches via softmax + lapse mixture:

```
P_A(π | s, θ) = (1-ε) · softmax(β · U(π; θ)) + ε · 1/|Π|
```

Where:

```
U(π | θ) = R_goal(π) + λ_θ · R_pref(π; θ) - J_risk(π)
```

Parameters from `AgentPolicyParams`:
- `β = 4.0` — softmax temperature (higher = more rational)
- `ε = 0.1` — lapse rate (random exploration)
- `λ_θ = 1.0` — preference weight

Preference types `θ ∈ {safe, risky, shiny, shortcut, neutral}` with reward weight vectors:

| θ | safety_bonus | tempt_bonus | texture_novelty | shortcut_bonus |
|---|-------------|-------------|-----------------|----------------|
| safe | 2.0 | -1.0 | 0.0 | 0.0 |
| risky | -0.5 | 0.5 | 0.0 | 0.0 |
| shiny | 0.0 | 3.0 | 0.0 | 0.0 |
| shortcut | 0.0 | 0.0 | 0.0 | 2.0 |
| neutral | 0.3 | 0.0 | 0.0 | 0.0 |

**Why this matters**: Previous deterministic safe-first agents provided zero behavioral variance, making preference inference impossible (PrefAcc = 0%). The stochastic agent is the prerequisite for all latent posterior work.

### 3.5 Preference posterior

```
q_t(θ) ∝ q_{t-1}(θ) · P_A(a_t | s_t, θ)
```

With optional forgetting/diffusion (ρ = 0.02):

```
q_t'(θ) = (1-ρ) · q_t(θ) + ρ · 1/|Θ|
```

**Verified result**: PrefAcc = 60% (3× chance of 20%). `shiny` is 100% identifiable; `risky` at 84%.

### 3.6 Goal posterior

5 discrete goal types: `{goal_safe_short, goal_safe_long, goal_collect, goal_explore, goal_direct}`.

```
q_t(g) ∝ q_{t-1}(g) · P_A(a_t | s_t, g)
```

Each goal has a distinct reward weight vector over branch attributes.

**Verified result**: GoalAcc = 74% (3.7× chance of 20%). `goal_direct` is 100% identifiable; `goal_collect` at 84%.

### 3.7 Coupled joint posterior

```
q_t(g, θ) ∝ q_{t-1}(g, θ) · P_A(a_t | s_t, g, θ)
```

With compatibility prior:

```
q_0(g, θ) ∝ exp(C_{g,θ})    where C_{g,θ} = 0.1 · w_g^T · w_θ
```

Joint utility for the agent:

```
U(π | g, θ) = R_goal(π; g) + λ_θ · R_pref(π; θ) - J_risk(π)
```

Marginals recovered by summation:

```
q_t(g) = Σ_θ q_t(g, θ)
q_t(θ) = Σ_g q_t(g, θ)
```

**Why this matters**: Factorized `q(g)·q(θ)` collapses under latent conflict — when pref and goal push in opposite directions, the independent posteriors drop to 0% on the conflicting factor. The coupled posterior maintains 56–68% joint accuracy under the same conflict. This is the key upgrade from K2→L1.

### 3.8 Self-discovery probability

```
p_self = σ((d_commit - d_reveal - m) / τ_v)
```

Where:
- `d_commit` = steps before agent is committed to a branch
- `d_reveal` = depth at which strong cues begin
- `m` = margin offset (default 0)
- `τ_v` = temperature controlling transition sharpness

**Why smooth matters**: v3 used binary `V_self = 1[d_commit ≥ d_reveal]`, which created sharp transitions at the boundary. v4's smooth sigmoid fixes boundary-regime instability and enables cross-family selectivity law.

Failure-if-wait:

```
P(fail | wait) = 1 - p_self
```

### 3.9 Decision-aware information metrics

Three metrics from `src/metrics/decision_info.py`:

**Decision Bayes Risk**:
```
BR = 1 - max(p_safe_A, p_safe_B)
```

**Margin Gain**:
```
M_t = J(π_risky) - J(π_safe)
ΔM = M_post - M_pre
```

**DVOI** (Decision Value of Information):
```
DVOI = σ(τ_m · M_post) - σ(τ_m · M_pre)
```

**Directional Correctness Gain**:
```
DCG = 1[post picks safe] - 1[pre picks safe]
```

### 3.10 Tutor objective (v4 — current canonical)

The v4 tutor compares Q_warn vs Q_wait:

```
Q_warn = λ_S · ΔS + λ_I · DVOI + λ_M · (1-p_self) - λ_C · cost - λ_R · redundancy
Q_wait = λ_V · p_self · Ĝ_self - λ_F · P(fail|wait)
```

Default weights from `TutorV4Config`:

| Weight | Value | Component |
|--------|-------|-----------|
| λ_S | 1.0 | Success gain (margin improvement) |
| λ_I | 2.0 | DVOI (decision value of information) |
| λ_M | 1.5 | Missed-window ~ (1-p_self) |
| λ_C | 0.05 | Intervention cost |
| λ_R | 0.3 | Redundancy penalty |
| λ_V | 2.0 | Self-discovery value |
| λ_F | 1.5 | Failure-if-wait penalty |
| τ_v | 1.0 | p_self temperature |
| τ_m | 1.0 | DVOI margin sensitivity |

The extended joint tutor (`JointLatentTutorV1`) adds:

```
Q_warn += λ_J · ΔJointInfo + λ_T · tempt_risk
Q_wait += λ_O · V_obs^joint + λ_A · total_uncertainty
```

Where `V_obs^joint = Var_{(g,θ)~q}[P(π|s,g,θ)]` — how much different latent combinations would cause different agent behavior.

---

## 4. Scenario Families and What Each Family Proves

### A. Canonical 3 families (lever verification)

| Family | Primary lever | What it proves |
|--------|--------------|----------------|
| `fork_trap` | WARN | Robot-belief timing determines whether agent avoids trap |
| `hazard_belt` | ITEM_DROP | Traversal cost reduction enables passage through hazard zone |
| `deadline_gate` | UNLOCK | Topology/affordance unlock reveals safe path before deadline |

These are stable conclusions. The next conversation must not re-question them.

### B. Topology-neutral planner benchmark

| Family | Purpose |
|--------|---------|
| `elcb` | Proves that once topology confounds are removed, semantic branch score alone can flip branch choice. Also proves that the planner interface (`λ_b · S_branch`) is the key bottleneck. |

### C. Tutor-sensitive timing benchmark

| Family | Purpose |
|--------|---------|
| `elcb_po` | Proves that partial observability must exist in the agent's *usable decision representation*, not just in raw feature textures. Warning timing, strong cue reveal depth, and commit depth interact to determine tutor value. |

### D. Cross-family selectivity families

| Family | Purpose |
|--------|---------|
| `delayed_corridor` | Proves that WAIT vs WARN depends on Δ = d_commit - d_reveal. When Δ < 0, must warn. When Δ >> 0, should wait. |
| `distractor_cue` | Proves that tutor uses diagnostic cues (dims 2, 3) rather than being distracted by irrelevant features (dim 1). |

### E. Hidden preference / joint latent families

| Family | Purpose |
|--------|---------|
| `temptation_corridor` | First family with hidden preference latent θ. Agent is stochastic; tutor must reason about temptation risk + preference posterior. |
| `joint_conflict_corridor` | Goal cue (dim 2) points to branch A; temptation cue (dim 1) points to branch B. Staggered reveal depths create asynchronous evidence. Forces robot to disentangle "is agent heading to B because of goal or temptation?" |

---

## 5. Key Milestones Across This Session

The session follows a strict causal chain—each milestone builds on the previous:

### 5.1 POMDP interface cleanup

Created 6 core dataclasses (`state_types.py`) and adapter layer (`adapters.py`) to formalize the data flow. This was not about "improving scores" but about giving the framework a clean interface for future latent extensions.

### 5.2 Smooth selectivity law (v3 → v4)

Binary `V_self` in v3 caused sharp transitions at Δ=0. Replaced with sigmoid `p_self` and DVOI. Validated across 5 families: v4 matches oracle on all, with monotonically decreasing WarnRate as Δ increases.

### 5.3 Stochastic bounded-rational agent

Deterministic safe-first agents had zero behavioral variance → PrefAcc = 0%. Softmax + lapse agent creates preference-dependent behavior distributions. Agent now only picks safe 62–64% of the time (was 100%).

### 5.4 Preference posterior v2

First time q(θ) learns from behavior: **PrefAcc = 60% (3× chance)**. `shiny` = 100% identifiable; `risky` = 84%. Behavior-driven Bayesian update with forgetting/diffusion.

### 5.5 Goal posterior v1

**GoalAcc = 74% (3.7× chance)**. `goal_direct` = 100%; `goal_collect` = 84%. Same Bayesian architecture as preference posterior.

### 5.6 Joint factorized → coupled posterior

K2 revealed the limit of factorized q(g)·q(θ):
- Aligned latents: both posteriors work (88%)
- Conflicting latents: factorized drops to 0% on the conflicting factor

L1 introduced coupled joint posterior q(g,θ) with compatibility prior:
- **Conflict: coupled maintains 56–68% joint accuracy where factorized collapses**

> This is the session's most important theoretical conclusion about multi-latent inference.

### 5.7 Cross-family verification

All tutors (v4, pref_v2, goal_v1, joint_v1) match oracle SBCR across all tested families (ELCB-PO, delayed corridor, temptation corridor, joint conflict corridor). The framework is now beyond single-family debugging.

---

## 6. Current Scientific Conclusions

### 6.1 Planner interface is the bottleneck — solved for branch-level semantics

On topology-neutral families, branch-level representation is sufficient. The bottleneck was the planner never querying it. Once `J_hybrid = J_cell - λ_b · S_branch` was introduced, transfer immediately recovered. This is a verified causal finding.

### 6.2 Partial observability must exist in the agent's usable decision representation

It is not sufficient for weak/strong cue differences to exist only in raw features. The latent heads, branch summaries, and planner interface must all preserve this information structure for tutor value to be measurable.

### 6.3 Optimal tutor action follows a cross-family selectivity law

The key structural variable is:

```
Δ = d_commit - d_reveal
```

- Δ < 0  → must warn (agent commits before seeing evidence)
- Δ >> 0 → should wait (agent will self-discover)
- Boundary → smooth probability + decision-aware info determine action

This law holds across ELCB-PO, delayed corridor, distractor cue, and temptation corridor. It is not a family-specific heuristic.

### 6.4 Preference and goal latents can be learned from behavior

- PrefAcc = 60% >> 20% chance (5 types)
- GoalAcc = 74% >> 20% chance (5 types)
- Both use the same Bayesian update mechanism; both require stochastic bounded-rational agent

### 6.5 Coupled posterior is necessary under latent conflict

Under conflict (pref→tempt, goal→safe or vice versa):
- Factorized q(g)·q(θ): **collapses to 0%** on the conflicting factor
- Coupled q(g,θ): **maintains 56–68% joint accuracy**

This is a verified finding from L1 (4 conditions × 50 trials each).

### 6.6 The framework is beyond "current problem fixing"

The system now handles:
- 7 scenario families across 5 functional categories
- 5 tutor variants (always_wait, always_warn, v4, pref_v2, goal_v1, joint_v1)
- 3 latent dimensions (semantics, preference, goal)
- Both independent and coupled inference
- Config-driven unified pipeline

It can legitimately be called a general pedagogical decision framework rather than a family-specific solution.

---

## 7. Stable vs Provisional Components

### Stable / Canonical

| Component | Status |
|-----------|--------|
| V2 runner/env path | Stable |
| Latent world semantics (d=4 features, orthogonal weights) | Stable |
| Patch observation (obs_radius) | Stable |
| Cell-level bounded planning J(π) | Stable |
| 3-way intervention semantics (WARN/UNLOCK/ITEM_DROP) | Stable — must not merge |
| Branch-aware planner J_hybrid | Stable |
| Semantic subspace + observation mask + identity neutralization | Stable |
| Smooth selectivity law (v4, sigmoid p_self, DVOI) | Stable — verified cross-family |
| Stochastic bounded-rational agent (softmax + lapse) | Stable |
| Preference posterior v2 (Bayesian q(θ), PrefAcc=60%) | Stable |
| Goal posterior v1 (Bayesian q(g), GoalAcc=74%) | Stable |
| Coupled joint posterior v2 (q(g,θ) with compatibility prior) | Stable |
| 6-layer POMDP dataclass chain (state_types.py) | Stable |
| Unified pipeline skeleton (pipeline.py) | Stable |

### Provisional / Still Under Tuning

| Component | Status | What's needed |
|-----------|--------|---------------|
| In-context joint latent accuracy in conflict corridors | Low (6–18% in-context) | More interaction steps per episode / richer observation window |
| Persistent-profile intervention reduction | Not yet demonstrated | Current families saturate "always warn"; need family with real WAIT space |
| Joint tutor outperforming simpler tutors in hard families | Not yet shown | All tutors currently match oracle; need families that differentiate |
| Goal-family design richness | Only 2 latent-aware families | Need compositional goal corridor |
| Full OOD/stress robustness suite | Not yet run | Mirror swap, noise sweep, parameter shift, calibration |
| Pipeline completeness (YAML configs, per-step JSONL traces) | Basic skeleton | Worth consolidating further |

---

## 8. Current Failure Modes / Remaining Gaps

### 8.1 Factorized posterior fails under conflict

This is an observed, verified failure. When preference and goal push in opposite directions, the independent posteriors q(g) and q(θ) each collapse to 0% on the conflicting factor. The coupled posterior resolves this but needs further validation on richer families.

### 8.2 Current conflict families have limited in-context evidence

With only ~50 train seeds and 1 branch choice per seed, the observation window for in-context joint posterior updates is small. JointAcc in the actual conflict corridor is 4–12%, far below the isolated test's 56–68%. More interaction steps per episode would help.

### 8.3 Persistent profile effect not yet fully demonstrated

The persistent-profile infrastructure works: θ is carried across episodes, PrefAcc reaches 100% for `shiny`. But because all current corridors are structurally "always warn" territory, the key signal — "robot warns less over time as it learns this learner" — cannot be observed. Requires a family where some episodes genuinely favor WAIT.

### 8.4 Current family set is still easier than future real tasks

Although 7 families exist across 5 functional categories, they are all single-fork, single-latent-dimension corridors. True compositional goals, multi-step temptation trajectories, and social goal negotiation remain future work.

### 8.5 No explicit handling of latent drift

Current model assumes θ is fixed within an episode (or across episodes for persistent profile). If the agent's preferences shift mid-episode (e.g., fatigue → risk-seeking), the posterior would lag. This is a known limitation, not a bug.

---

## 9. Recommended Next Phase

### Priority 1: Persistent-profile family with real WAIT space

The persistent-profile infrastructure is ready, but the testbed needs a family where:
- Some episodes naturally have Δ >> 0 (WAIT-favorable)
- Other episodes have Δ < 0 (WARN-necessary)
- θ stays fixed across episodes; g varies

This would allow demonstrating: "robot learns this learner → warns less in WAIT-favorable episodes → same or better SBCR."

### Priority 2: Joint-latent conflict tutor v2 / nonmyopic observation value

The coupled posterior is proven better than factorized. The next question: can the tutor exploit "observing one more agent action" to better disentangle goal vs preference? This requires tutor to explicitly compare multi-step observation value against immediate warning.

### Priority 3: Compositional goal corridor

Use the `latent_goal_vector` hooks to create a family with true multi-factor goals (e.g., "collect red AND avoid blue"). This moves beyond single-goal-type inference to structured goal representation.

### Priority 4: Unified pipeline completion + robustness suite

- Consolidate all experiments into YAML-based configs
- Run systematic sweeps: mirror invariance, side swap, noise sweep, family parameter shift
- Report ECE, reliability curves, posterior entropy decay for q(θ), q(g), q(g,θ)
- Generate family × latent transfer matrix

---

## 10. Exact Modules / Files Added This Session

### Core / API layer

| File | Purpose |
|------|---------|
| `src/core/state_types.py` | 6-layer POMDP dataclasses: WorldState → AgentObservation → AgentBelief → RobotBeliefOnAgent → BranchPosterior → TutorDecisionTrace |
| `src/core/adapters.py` | Bridges legacy runner/planner code to new runtime API |

### Metrics layer

| File | Purpose |
|------|---------|
| `src/metrics/self_discovery.py` | Smooth p_self via sigmoid; P(fail\|wait); observation-model-based p_self |
| `src/metrics/decision_info.py` | Decision Bayes Risk, Margin Gain, DCG, DVOI |
| `src/metrics/calibration.py` | ECE for p_self, WarnRate(Δ) curve analysis, empirical self-discovery |

### Agent layer

| File | Purpose |
|------|---------|
| `src/agents/stochastic_agent_policy.py` | Softmax + lapse agent; 5 θ types with PREF_REWARD vectors; BranchAttributes dataclass |
| `src/agents/preference_posterior_v2.py` | Bayesian q(θ) with forgetting; posterior_predictive_variance for observation value |
| `src/agents/goal_posterior_v1.py` | Bayesian q(g) over 5 goal types with GOAL_REWARD vectors |
| `src/agents/joint_latent_belief.py` | Factorized q(g)·q(θ) wrapper (scaffold only — use coupled for conflict) |
| `src/agents/joint_posterior_v2.py` | Full q(g,θ) table with compatibility prior C_{g,θ}; joint utility; joint likelihood |

### Tutor layer

| File | Purpose |
|------|---------|
| `src/teachers/learning_aware_policy_v4.py` | Smooth selectivity: Q_warn vs Q_wait with p_self + DVOI |
| `src/teachers/preference_aware_policy_v2.py` | v4 + ΔPrefInfo + observation value + temptation risk |
| `src/teachers/goal_preference_aware_policy_v1.py` | v4 + ΔGoalInfo (uses factorized joint) |
| `src/teachers/joint_latent_tutor_v1.py` | Nonmyopic tutor with coupled q(g,θ) and V_obs^joint |

### Environment / Families

| File | Added families |
|------|---------------|
| `src/envs/scenario_families.py` | `delayed_corridor`, `distractor_cue`, `temptation_corridor`, `joint_conflict_corridor` |

### Pipeline / Evaluation

| File | Purpose |
|------|---------|
| `src/evals/__init__.py` | Module init |
| `src/evals/pipeline.py` | Config-driven ExperimentConfig + run_experiment + write_report |

### Experiment scripts

| File | What it runs |
|------|-------------|
| `scripts/run_stochastic_agent.py` | J1 behavior validation + J2 posterior convergence + J3 temptation experiment |
| `scripts/run_multi_latent.py` | K1 goal convergence + K2 joint convergence + K3+K6 cross-family |
| `scripts/run_joint_conflict.py` | L1 coupled vs factorized + L2+L3 conflict corridor + L4 persistent profile |

### Results

| File | Content |
|------|---------|
| `results/stochastic_agent_report.md` | PrefAcc=60%, agent behavioral diversity |
| `results/multi_latent_report.md` | GoalAcc=74%, joint convergence, cross-family robustness |
| `results/joint_conflict_report.md` | Coupled vs factorized (56–68% vs 0%), persistent profile |

---

## 11. Do Not Break / Common Misinterpretations

### 11.1 Do NOT describe this project as RL

This is a **belief-updating, model-based, bounded-planning** pedagogical tutoring framework. The tutor does not learn a policy via reward maximization. It performs explicit reasoning over what the agent knows, what information is available, and whether intervention helps or hinders long-term learning. This architectural identity is a core stable conclusion.

### 11.2 Do NOT merge the three intervention types

WARN is belief/evidence. UNLOCK is topology/affordance. ITEM_DROP is traversal mitigation. They operate on different layers of the agent's decision problem. Do not write them as "all modify risk parameters."

### 11.3 Do NOT treat factorized joint as the final solution

`q(g)·q(θ)` was a useful scaffold but is now superseded by coupled `q(g,θ)` for any scenario involving latent conflict. The factorized version collapses to 0% on conflicting factors. Always use `JointPosteriorV2` for conflict-capable inference.

### 11.4 Do NOT interpret oracle-matching as "problem solved"

All current tutors match oracle SBCR on existing families. This means the families are now **framework verification benchmarks**, not challenging test beds. The framework needs harder families (persistent-profile, compositional goal, multi-step temptation) to differentiate tutor quality.

### 11.5 Do NOT rewrite as full POMDP solver or recursive pragmatics

The current system's value lies in being:
- Minimal diff per iteration
- Every component readable and explainable
- Research-useful rather than engineering-maximal

A full POMDP solver or RSA recursion stack would lose these properties. Stick to the bounded-planning, belief-update architecture.

### 11.6 Do NOT neglect the Δ = d_commit - d_reveal structure

This timing parameter is the fundamental structural variable governing WAIT vs WARN. Any new family must specify both `commit_depth` and `reveal_depth` so that the selectivity law can be evaluated.

---

## 12. Minimal Reproduction / How to Continue

### 12.1 Understanding the framework

Read in this order:
1. **This file** (`stage_framework_handoff.md`) — complete technical state
2. `task_list/step1_new_scene/project_handoff_summary.md` — earlier-phase context
3. `AGENT_BRIEF.md` — project-level overview and three identified gaps
4. `proposal.md` — research proposal and evaluation criteria

### 12.2 Reproducing experiments

```bash
# Activate environment
conda activate pedip310
cd pedagogical_ip

# Run stochastic agent validation (PrefAcc=60%)
python scripts/run_stochastic_agent.py

# Run multi-latent framework (GoalAcc=74%, cross-family)
python scripts/run_multi_latent.py

# Run joint conflict (coupled vs factorized)
python scripts/run_joint_conflict.py
```

All results go to `results/*.md`.

### 12.3 Continuing research

**Priority order**:

1. **Persistent-profile family** — Design a family where some episodes favor WAIT, so persistent θ knowledge reduces intervention over time
2. **Richer joint-conflict family** — More interaction steps per episode; multi-step temptation trajectories
3. **Compositional goal corridor** — Multi-factor goals using `latent_goal_vector` hooks
4. **Pipeline consolidation** — YAML configs, per-step JSONL traces, systematic robustness sweeps

### 12.4 Key invariants to maintain

When modifying any module, ensure:
- `state_types.py` data flow chain remains intact
- WARN / UNLOCK / ITEM_DROP semantics remain separated
- p_self remains smooth (sigmoid, not binary)
- Branch-aware planner interface (`λ_b · S_branch`) is preserved
- Coupled posterior is used for any conflict scenario
- Agent stochasticity (softmax + lapse) is maintained — do not revert to deterministic
