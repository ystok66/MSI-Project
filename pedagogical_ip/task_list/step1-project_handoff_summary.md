# Project Handoff Summary — Phase 0 through Current

> **Purpose**: Onboarding document for future agents. Covers project architecture, current canonical system, scenario families, formulas, results, and next steps. This is a source-of-truth reference, not a marketing document.

---

## 1. Project Purpose and Research Question

This project studies **pedagogical robot assistance** in partially observable grid worlds.

**Core research question:**

> Is the robot helping the agent *complete the task*, or helping it *learn to complete the task on its own* — and what is the tradeoff?

The agent navigates a lattice with risky traps, hidden costs, and time deadlines, learning from noisy partial observations. A robot tutor observes the agent and intervenes (**WAIT / WARN / UNLOCK / ITEM_DROP**) to help. The system measures both **online task performance** and **post-training transfer** (no-tutor) to separate "help completion" from "help learning."

**Key modeling insight (Phase 10):** intervention semantics must enter the agent model in different architectural layers:

| Intervention | Affects | NOT |
|---|---|---|
| WARN | Semantic belief / risk prior | Topology or traversal |
| UNLOCK | Affordance / reachability / uncertainty prior | Risk mean directly |
| ITEM_DROP | Traversal outcome dynamics (effective risk penalty) | Belief mean or variance |

The system is **not RL-based**. It is a **belief-updating, model-based, bounded-planning** system.

---

## 2. Evolution Across Phases

| Phase | What it did | Canonical artifact |
|-------|------------|-------------------|
| 0 | Froze legacy baselines (9/80/68/99/100%) | `baseline_freeze/v2_l2c1_baseline.txt` |
| 1 | Planner deduplication + warning abstraction | `warning_update.py`, `pragmatic_warning.py` |
| 2 | Runner platformization | `lattice_v2_runner.py` |
| 3 | Environment facade | `lattice_v2_env.py` (`Observation`, `StepResult`) |
| 4 | **Latent world semantics** | `cost_risk_model.py` (`LatentCostRiskHead`), `feature_belief.py` |
| 5 | Patch observation + prefix prediction | `prefix_prediction.py`, `observe_features_patch()` |
| 6 | Belief-conditioned bounded planning | `belief_planning.py` (`BeliefPlan`, `FailureModeEstimate`) |
| 7 | Approximate robot belief | `robot_belief.py`, `agent_predictor.py`, `intervention_policy.py` |
| 8 | Unified intervention family + shield | `interventions.py` (4-way scoring), `InventoryState` |
| 9 | Evaluation system + experiment matrix | `phase9_metrics.py`, `transfer_eval.py` |
| **10** | **Intervention-conditioned belief model** | `route_necessity.py`, `CellMemoryMeta`, necessity-aware planner |

### Phase 10 (Current)

Phase 10 is the most recent and most conceptually important. It added:

- **`CellMemoryMeta`** provenance per cell (ever_seen, ever_traversed, intervention_tags, reachable_since_t)
- **Directional uncertainty** `u_c = w_c^T Σ w_c`, `u_r = w_r^T Σ w_r`
- **Route-level structural necessity** `n(π) ∈ [0,1]` via BFS
- **Learning-factor risk discount** — untrained prior risk discounted by necessity
- **Intervention-conditioned belief updates**: `apply_unlock_update()`, `apply_warn_update()`
- **Numerical stability**: gradient clipping, weight norm clamping, NaN safety

---

## 3. Current Canonical Architecture

### Canonical execution path

```
LatticeV2Runner.reset(seed, latent_mode=True, scenario_family=..., ...)
  → generate_scenario() → GridMap + cell_features + LatticeV2Meta
  → init FeatureBeliefMap + LatentCostRiskHead + RobotBelief + InventoryState

while not s.done:
    runner.step(s)
        1. observe(s)         — patch observation → Kalman belief update
        2. apply_tutor(s)     — robot-belief scoring → execute WAIT/WARN/UNLOCK/ITEM_DROP
        3. plan_and_move(s)   — compute necessity → A* plan → move → outcome → online learning
    
runner.get_metrics(s)
```

### Key files (read these first)

| File | Role |
|------|------|
| `src/envs/lattice_v2_runner.py` | **Main runner** — episode loop, 3-substep step(), all wiring |
| `src/envs/scenario_families.py` | **Scenario generation** — fork_trap, hazard_belt, deadline_gate |
| `src/agents/feature_belief.py` | **Belief system** — FeatureBeliefMap, CellMemoryMeta |
| `src/agents/planner_astar.py` | **Planner** — cell_cost_v2_latent, plan_next_action_v2 |
| `src/agents/cost_risk_model.py` | **Cost/risk heads** — LatentCostRiskHead |
| `src/agents/route_necessity.py` | **Necessity** — BFS route-level scalar |
| `src/teachers/intervention_policy.py` | **Tutor scoring** — score_interventions() |
| `src/teachers/robot_belief.py` | **Robot belief** — surrogate of agent belief |

### What is NOT canonical (legacy)

| File | Status |
|------|--------|
| `bounded_agent.py` | V0 agent, not used by runner |
| `observation_model.py` | V0 observation, replaced by inline `observe_features()` |
| `belief.py` (`BeliefMap`) | V0 belief, replaced by `FeatureBeliefMap` |
| `eval_v1.py` | V0 metrics, replaced by `phase9_metrics.py` |
| `map_generator.py` | V0 8×8 grid, replaced by `lattice_v2.py` |
| `pedagogical_grid.py` | V0 env, replaced by `lattice_v2_env.py` |

---

## 4. Scenario Families

### 4.1 baseline_v2

**Role**: Regression anchor. Default V2 lattice with 3 segments.  
**Main lever**: None specific — tests general capability.
**Gate mode**: `block_risky` (tutor closes risky lanes).

### 4.2 fork_trap

**Role**: Test **WARN** as belief evidence.  
**Structure**: Ambiguous lane fork — agent can't distinguish safe from dangerous path by early observation alone.  
**Failure mode**: Wrong-branch commitment → death.  
**Main lever**: **WARN / robot_belief** — tutor provides semantic evidence about which branch is dangerous.  
**Current result**: `robot_belief=65%` vs `no_tutor=5%` (20 seeds, medium).

### 4.3 hazard_belt

**Role**: Test **ITEM_DROP** as survival mechanism.  
**Structure**: Unavoidable risky cells across the path — cannot avoid, must survive.  
**Failure mode**: Death from guaranteed risk exposure.  
**Main lever**: **ITEM_DROP (shield)** — reduces effective risk on traversal.  
**Current result**: `item_only=60%` vs `no_tutor=30%` (20 seeds, medium).

### 4.4 deadline_gate

**Role**: Test **UNLOCK** as topology assistance.  
**Structure**: Tight deadline + gated shortcut. Safe path exists but may timeout. Shortcut (row 1) is behind a locked gate.  
**Failure mode**: Timeout (can't reach goal via safe path) or **bounce** (enters shortcut but bouncing between rows due to uncertainty about unknown cells).  
**Main lever**: **UNLOCK** — tutor opens gate + necessity mechanism allows traversal of unknown route.  
**Current result**: `unlock_only=100%` vs `no_tutor=70%` (20 seeds, medium). Agent traverses shortcut in 20 steps vs 24 via safe path.

---

## 5. Agent Design

### 5.1 Belief Representation

**`FeatureBeliefMap`** (`feature_belief.py`): Gaussian posterior over 4D feature vectors per cell.

- `mean[H,W,d]`: posterior mean `μ_i`
- `var[H,W,d]`: posterior variance `Σ_i` (diagonal)
- `memory[H,W]`: `CellMemoryMeta` provenance grid

Prior: `mean = [0.5, 0, 0.5, 0.5]`, `var = [0.25, 0.25, 0.25, 0.25]`

Updated via **Kalman filter** from noisy observations:
```
K = var_prior / (var_prior + obs_var)
mean_post = mean_prior + K * (obs - mean_prior)
var_post = (1 - K) * var_prior
```

### 5.2 Cost/Risk Prediction

**`LatentCostRiskHead`** (`cost_risk_model.py`): Composes two linear heads.

- `BayesianCostHead`: `ĉ = w_c^T μ + b_c` (Gaussian likelihood, online MAP)
- `BayesianRiskHead`: `r̂ = sigmoid(w_r^T μ + b_r)` (Bernoulli likelihood)

Both start with `w=0, b=0` → `ĉ=0, r̂=0.5` (uninformative prior).

**Online learning**: after each traversal, heads update from `(feature, label)` pair via SGD on log-posterior. Gradient clipping (`max_norm=5.0`) and weight norm clamping (`max_norm=10.0`) prevent NaN.

### 5.3 Memory Provenance

**`CellMemoryMeta`** per cell:

| Field | Meaning |
|-------|---------|
| `ever_seen` | Cell observed at least once |
| `seen_count` | Number of observation events |
| `ever_traversed` | Agent has stepped on this cell |
| `traversed_count` | Number of traversals |
| `last_seen_t`, `last_traversed_t` | Most recent timestamps |
| `best_view_quality` | Minimum observation noise |
| `reachable_since_t` | When cell became passable (UNLOCK) |
| `intervention_tags` | Set of tags: `"unlocked"`, etc. |

### 5.4 Planning

**`cell_cost_v2_latent()`** computes per-cell planning cost. See Section 7 for the full formula.

**`plan_next_action_v2()`** / **`plan_with_alternatives_v2()`** run A* with this cost function and return next action + path. A* budget is 30 expansions.

### 5.5 Route Necessity

**`compute_route_necessity()`** (`route_necessity.py`): BFS-based route-level scalar `n ∈ [0,1]`.

Uses **all unvisited cells** as "route cells to evaluate." If avoiding unvisited territory makes the goal unreachable within deadline → `n = 1.0` (maximum necessity). If safe alternative exists → `n` drops.

---

## 6. Tutor / Robot-Belief Design

### 6.1 Tutor Variants

| Tutor | Logic | Used by |
|-------|-------|---------|
| None (`tutor_mode="none"`) | No teaching | Baselines + all modern experiments |
| `TimeAwareDoorTutor` | Closes risky gate at fixed times | Legacy |
| `OracleTeacher` | Full visibility, utility-maximizing | Calibration benchmarks |
| **Robot-belief** | Approximate agent belief → counterfactual scoring | **Canonical for experiments** |

### 6.2 Robot Belief

**`RobotBelief`** (`robot_belief.py`): The robot's approximate surrogate of the agent's belief.

- Belief copy modes: `exact`, `noisy` (+ Gaussian noise), `stale` (sync every N steps)
- Competence params: `agent_search_budget`, `agent_risk_weight`
- `build_surrogate_predictor()` creates a frozen `LatentCostRiskHead`

### 6.3 Intervention Scoring

**`score_interventions()`** (`intervention_policy.py`): Evaluates 4 actions via **counterfactual prefix prediction**.

| Action | Counterfactual |
|--------|---------------|
| WAIT | Baseline prefix (no change) |
| WARN | Re-predict after belief bias |
| UNLOCK | Re-predict after gate opened |
| ITEM_DROP | Re-predict with shield added |

Returns `InterventionDecision` with action, scores, reason, decision_margin, predicted_prefix.

### 6.4 Intervention Semantics (Phase 10)

**WARN** → `apply_warn_update()`:
- Shifts posterior mean μ along `warn_direction` (aligned with risk head weights, NOT hardcoded to feature indices)
- `warn_strength=0.15`, `warn_confidence=2.0`
- Also applies legacy heuristic lane bias (`warned_cell_extra`)

**UNLOCK** → `apply_unlock_update()`:
- Reduces posterior variance: `Σ_i^+ = (1 - β) Σ_i` where `β=0.5`
- Does NOT change mean μ
- Sets `reachable_since_t`, adds "unlocked" to intervention_tags

**ITEM_DROP** → adds shield to inventory:
- Changes effective risk penalty in planner: `φ(r̂) × (1 - γ_shield)` where `γ_shield=0.5`
- Does NOT change μ or Σ

### 6.5 Current Limitations

- Heuristic warning (`warning_mode="lane"`) adds zero marginal value — only robot_belief timing works
- Robot_belief on hazard_belt selects suboptimally when multiple interventions available (40% < item_only 60%)
- Tutor perceptual model not yet implemented (no `patch_radius` / `obs_var` in RobotBelief)

---

## 7. Core Formulas

### Cell posterior

$$q_i(z_i) = \mathcal{N}(\mu_i, \Sigma_i)$$

where `μ_i, Σ_i` are maintained per cell in `FeatureBeliefMap`.

### Predictions

$$\hat{c}_i = w_c^\top \mu_i + b_c$$

$$\hat{r}_i = \sigma(w_r^\top \mu_i + b_r)$$

### Directional uncertainty

$$u_i^{(c)} = w_c^\top \Sigma_i \, w_c$$

$$u_i^{(r)} = w_r^\top \Sigma_i \, w_r$$

### Planner score

$$J(\pi) = \sum_{i \in \pi} \Big[ \lambda_c \hat{c}_i + \text{risk\_penalty}_i + \lambda_{uc} (1-n) \, u_i^{(c)} + \lambda_{ur} (1-n) \, u_i^{(r)} \Big]$$

Where risk penalty includes a **learning-factor-aware necessity discount**:

$$\text{risk\_penalty}_i = \varphi(\hat{r}_i) \times \big[\alpha + (1-\alpha)(1-n(\pi))\big]$$

$$\varphi(r) = -\ln(1 - r)$$

### Learning factor

$$\alpha = \min\!\big(1,\; \frac{n_{\text{updates}}}{10}\big)$$

When `n_updates=0` (untrained prior): `α=0` → risk penalty fully discounted by necessity.  
When `n_updates≥10` (data-driven): `α=1` → risk penalty always applies.

### Route necessity

$$n(\pi) \in [0, 1]$$

Currently computed as a route-level scalar via BFS. If avoiding all unvisited cells makes the goal unreachable within deadline → `n=1`. If a safe visited-only path exists → `n=0`.

### Intervention-conditioned updates

**WARN**: `μ_i^+ = μ_i + α_{warn} \cdot v_{warn}` where `v_warn` is aligned with risk head weight direction.

**UNLOCK**: `Σ_i^+ = (1 - \beta_{unlock}) \Sigma_i` — uncertainty reduction only, no mean change.

**ITEM_DROP**: `\varphi(\hat{r}_i) \to (1 - \gamma_{shield}) \varphi(\hat{r}_i)` — traversal term only, no belief change.

### Default parameters

| Symbol | Value | Meaning |
|--------|-------|---------|
| `λ_c` | 1.0 | Cost weight |
| `λ_r` | 5.0 | Risk penalty weight |
| `λ_uc` | 0.1 | Cost uncertainty weight |
| `λ_ur` | 0.1 | Risk uncertainty weight |
| `β_unlock` | 0.5 | UNLOCK variance reduction |
| `α_warn` | 0.15 | WARN mean shift strength |
| `γ_shield` | 0.5 | Shield risk reduction |
| `K` | 10 | Learning factor denominator |

---

## 8. Current Experimental Results

### 3-Family Matrix (20 seeds, medium difficulty)

| Family | no_tutor | warning | item_only | unlock_only | robot_belief |
|--------|----------|---------|-----------|-------------|-------------|
| **fork_trap** | 5% | 5% | — | — | **65%** |
| **hazard_belt** | 30% | 30% | **60%** | — | 40% |
| **deadline_gate** | 70% | — | — | **100%** | **100%** |

### Key conclusions

1. **Each intervention family has ≥1 scenario where it is the strongest lever**:
   - UNLOCK → deadline_gate (+30pp over no_tutor)
   - ITEM_DROP → hazard_belt (+30pp)
   - WARN/robot_belief → fork_trap (+60pp)

2. **Heuristic warning alone = 0 marginal effect** on fork_trap and hazard_belt. Only robot_belief-timed WARN works.

3. **Structural necessity solves the "unknown = dangerous" problem**: Without necessity, the agent bounced on shortcut indefinitely (see Phase 10 debug chain). With necessity + learning-factor risk discount, the agent cleanly traverses unknown territory when structurally required.

4. **Robot_belief tutor on hazard_belt (40%) is below item_only (60%)**: The multi-intervention tutor doesn't always select the optimal action when several are available.

5. **All 406 unit tests pass.** Legacy baselines preserved.

---

## 9. Stable vs Provisional Components

### Stable / Canonical

| Component | Status |
|-----------|--------|
| V2 runner/env path | ✅ Stable since Phase 2 |
| Latent world semantics (`LatentCostRiskHead`) | ✅ Stable since Phase 4 |
| Patch observation | ✅ Stable since Phase 5 |
| Belief-conditioned bounded planning (`BeliefPlan`) | ✅ Stable since Phase 6 |
| Robot-belief surrogate | ✅ Stable since Phase 7 |
| Unified intervention family (4-way scoring) | ✅ Stable since Phase 8 |
| Phase 9 metrics/reporting | ✅ Stable |
| Legacy baselines (9/80/68/99/100%) | ✅ Frozen |
| Scenario families (fork_trap, hazard_belt, deadline_gate) | ✅ Stable structure, params may tune |
| CellMemoryMeta provenance | ✅ Stable interface |
| Route necessity mechanism | ✅ Working, scalar approach stable |
| Learning-factor risk discount | ✅ Validated across families |

### Provisional / Still Under Tuning

| Component | Status |
|-----------|--------|
| Tutor perceptual model | ❌ Not yet implemented |
| Full cross-difficulty matrix | ❌ Only medium tested with 20 seeds |
| Transfer experiments | ❌ Not yet run |
| Warning semantics (pure evidence factor vs heuristic) | ⚠️ Only works via robot_belief timing |
| `apply_warn_update` strength/confidence params | ⚠️ May need per-family tuning |
| Exact necessity computation method | ⚠️ BFS-based, could be refined |
| `delayed_corridor` and `distractor_cue` families | ⚠️ Not yet tested with Phase 10 |

---

## 10. Open Issues / Remaining Weaknesses

1. **Heuristic warning has zero effect**: `warning_mode="lane"` adds nothing beyond no_tutor. The timing and mechanism (fixed proximity trigger + lane bias) are too crude.

2. **Robot_belief multi-action selection**: When ITEM_DROP, WARN, and UNLOCK are all available, the tutor doesn't always pick the best one. On hazard_belt, `robot_belief=40%` < `item_only=60%`.

3. **Tutor perceptual model missing**: `RobotBelief` doesn't model `patch_radius`, `self_obs_var`, or `neighbor_obs_var`. This causes redundant warnings and suboptimal timing.

4. **Transfer experiments not yet run**: The transfer pipeline (`transfer_eval.py`) exists but hasn't been exercised with the Phase 10 system. This is critical for the paper — the key question is "did the agent learn, or was it just helped?"

5. **Cross-difficulty sweep pending**: Only medium difficulty tested with 20 seeds. Easy and hard need systematic coverage.

6. **Config files not programmatically loaded**: Runner uses explicit kwargs. Config files are reference docs only.

7. **Naming inconsistencies**: `closures` vs `unlock_count`, `warnings_sent` vs `warn_count` across modules.

8. **`delayed_corridor` and `distractor_cue` families**: Defined but not yet tested with Phase 10 mechanisms.

---

## 11. Recommended Next Steps

### Rank 1: Cross-difficulty sweep (3 families × 3 difficulties × 20 seeds)

Immediate value: validates that results hold across easy/medium/hard. Quick to run.

### Rank 2: Stage 4 — Tutor perceptual model

Add `patch_radius`, `self_obs_var`, `neighbor_obs_var` to `RobotBelief`. Expected improvement: better WARN timing, reduced redundant interventions, improved robot_belief on hazard_belt.

### Rank 3: Transfer evaluation

Run `transfer_eval.py` with Phase 10 system: train with tutor assistance, then evaluate without tutor. Critical for the paper's central question.

### Rank 4: Paper-facing result tables and plots

Generate publication-quality tables and figures from the 3-family matrix results.

### Rank 5 (optional): Principled WARN evidence refinements

More precise `warn_direction` alignment, configurable per-family `warn_strength`, and RSA-style evidence factors.

---

## 12. Practical Quickstart for the Next Agent

### Files to read first (in order)

1. **This document** — for context
2. `src/envs/lattice_v2_runner.py` — the main loop; understand `step()` substeps
3. `src/agents/planner_astar.py` — `cell_cost_v2_latent()` is the core formula
4. `src/agents/feature_belief.py` — belief system + `CellMemoryMeta`
5. `src/envs/scenario_families.py` — how families generate different worlds

### How to identify the canonical path

- Look for `latent_mode=True` — this enables the current system
- Look for `robot_belief_mode=True` — this enables the smart tutor
- `belief_planning_mode=True` is optional and not required for basic experiments
- `tutor_mode="none"` + `robot_belief_mode=True` is the canonical tutor configuration

### Most informative outputs

- Run `scripts/stage3_experiment.py` for the full 3-family comparison
- Run `python -m pytest tests/ -q` to verify 406/406 tests pass
- Check `src/agents/route_necessity.py` for the necessity computation

### What NOT to touch casually

- Do NOT change `risk_model.py` `BayesianRiskHead` weight initialization (`w=0, b=0`) — the learning-factor mechanism depends on this being the uninformative prior
- Do NOT remove the NaN safety in `cell_cost_v2_latent` — online updates can produce weight explosions
- Do NOT modify legacy baselines or V0 code (`bounded_agent.py`, `belief.py`, `map_generator.py`)
- Do NOT remove `CellMemoryMeta` backwards-compatibility aliases (`observed`, `visit_count`, `last_observed_t`)
- Config files are reference docs, NOT loaded by the runner
