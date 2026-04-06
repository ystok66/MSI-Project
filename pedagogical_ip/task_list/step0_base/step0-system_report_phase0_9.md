# System Report: Phase 0–9

## A. Project Purpose and Research Question

This project studies **pedagogical robot assistance** in partially observable grid worlds. The core research question:

> **Is the robot helping the agent complete the task, or helping it learn to complete the task on its own — and what is the tradeoff?**

The agent navigates a lattice with risky traps, hidden costs, and time deadlines, learning from noisy partial observations. A robot tutor observes the agent and intervenes (WAIT / WARN / UNLOCK / ITEM_DROP) to help. The experimental platform measures both online task performance and post-training transfer (no-tutor) performance to separate "help completion" from "help learning."

---

## B. Evolution Across Phase 0–9

| Phase | What it did | Core artifacts | Tests |
|-------|------------|----------------|-------|
| 0 | Froze legacy baselines | `baseline_freeze/v2_l2c1_baseline.txt` | 77 |
| 1 | Planner deduplication + warning abstraction + belief protocol cleanup | `warning_update.py`, `pragmatic_warning.py`, `rsa_warning.py`, `belief_protocol.py` | 91 |
| 2 | Runner platformization | `lattice_v2_runner.py` (extracted from inline logic) | 97 |
| 3 | Environment facade | `lattice_v2_env.py` (`Observation`, `TeacherInfo`, `StepResult`) | 105 |
| 4 | Latent world semantics | `cost_risk_model.py` (`LatentCostRiskHead`), `feature_belief.py` | 124 |
| 5 | Patch observation + prefix prediction | `prefix_prediction.py`, `observe_features_patch()` | 150 |
| 6 | Belief-conditioned bounded planning | `belief_planning.py` (`BeliefPlan`, `FailureModeEstimate`) | 177 |
| 7 | Approximate robot belief | `robot_belief.py`, `agent_predictor.py`, `intervention_policy.py` | 209 |
| 8 | Unified intervention family + shield | `interventions.py` (ItemType, InventoryState), 4-way scoring | 245 |
| 9 | Evaluation system + experiment matrix | `phase9_metrics.py`, `transfer_eval.py`, config-driven matrix | 279 |

---

## C. Environment and Task/Scenario Design

### C.1 V2 Lattice Structure

Source: [lattice_v2.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/lattice_v2.py)

The world is a **7-row × W-col** grid with 3 segments. Each segment has:

| Row | Purpose |
|-----|---------|
| 0 | Wall |
| 1 | **Risky lane** — straight path (Ls cells) |
| 2 | Wall (separator) |
| 3 | **Safe lane** — zigzag path (Ls + 2×detour cells) |
| 4 | Wall except detour columns |
| 5 | Safe lane detour |
| 6 | Wall |

Key design: risky lane is shorter/cheaper but has traps; safe lane is longer but safe.

### C.2 Cell Features

`FEATURE_DIM = 4`: `[lane_id, gate_flag, texture_1, texture_2]`. These are the **latent state** — hidden from agent, revealed only through noisy observation. The agent learns to predict cost and risk from these features.

### C.3 Segments and Traps

Each [SegmentMeta](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/lattice_v2.py#L46-L61) tracks:
- `risky_cells` / `safe_cells` — cells on each lane
- `risky_entry_gate` / `safe_entry_gate` — entry points the tutor can close
- `trap_cell` — the single lethal cell in the risky lane
- `weak_cue_cells` — cells that hint at danger

### C.4 Difficulty Scaling

`generate_lattice_v2(difficulty=...)` controls risk values:
- **easy**: trap_risk ∈ [0.1, 0.3]
- **medium**: trap_risk ∈ [0.3, 0.5]
- **hard**: trap_risk ∈ [0.5, 0.8]

### C.5 Deadline

`t_max = int(shortest_safe × time_ratio)` where `time_ratio=1.3`. The agent must reach the goal within `t_max` steps or the episode times out.

### C.6 Observation Model

The V2 runner uses **inline feature observation** functions (NOT the V0 `observation_model.py`):

- `observe_features()` in runner — observes 4D feature vectors from `cell_features`, not cost/risk scalars
- `observe_features_patch()` — extended patch version (Phase 5)

Noise levels:
- **Self cell**: σ²=0.01 (near-exact)
- **1-hop neighbors**: σ²=0.08 (blurry but informative)
- **Patch mode** (Phase 5): extends to `patch_radius` hops

> [!NOTE]
> The V0 `observation_model.py` observes cost/risk scalars from `true_cost/true_risk`. It is used by `BoundedRationalAgent` (V0) only, not by the V2 runner.

### C.7 Tutor Actions in the Environment

| Action | Effect |
|--------|--------|
| WAIT | No action |
| WARN | Biases agent's cost map for risky-lane cells (`warned_cell_extra`) |
| UNLOCK (close gate) | Sets `passable[gate]=False`, forcing safe lane |
| ITEM_DROP (shield) | Adds shield to inventory; first risky traversal reduces death probability by 50% |

---

## D. Agent Architecture

### D.1 Belief System

**V0 agent**: [bounded_agent.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/bounded_agent.py) uses `BeliefMap` (cost/risk mean+var per cell from `belief.py`).

**V2 agent** (Phase 4+): uses [FeatureBeliefMap](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/feature_belief.py) — Gaussian belief over 4D feature vectors per cell, with Kalman updates. Tracks `visit_count` and `last_observed_t`.

### D.2 Cost-Risk Prediction

Source: [cost_risk_model.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/cost_risk_model.py)

`LatentCostRiskHead` composes two heads:
- `BayesianCostHead`: `cost_hat = w_c · z + b_c` (Gaussian likelihood, online MAP)
- `BayesianRiskHead`: `risk_hat = sigmoid(w_r · z + b_r)` (Bernoulli likelihood)

Both learn from supervision during episodes (oracle_visited or binary_outcome).

### D.3 Planning

Source: [planner_astar.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/planner_astar.py)

Two coexisting planner paths:

| Version | Function | Cost formula |
|---------|----------|-------------|
| V0 | `bounded_astar()` | `belief_cost + λ_r × (-log(1-ρ)) + λ_u × σ` |
| V2 (latent) | `cell_cost_v2_latent()` via `plan_with_alternatives_v2()` | see below |

V2 latent cost formula (from `cell_cost_v2_latent`, 4 separate weights):

```
score = λ_c · cost_hat
      + λ_r · risk_penalty(risk_hat)   [× (1 - shield_reduction) if shield]
      + λ_uc · cost_uncertainty
      + λ_ur · risk_uncertainty
```

where `risk_penalty = λ_r · (-log(1 - risk_hat))`. Defaults: `λ_c=1.0, λ_r=5.0, λ_uc=0.1, λ_ur=0.1`.

Budget is sampled from `sample_search_budget()` (negative-binomial approximation). Phase 8 added `inventory_state` parameter for shield-aware cost reduction.

### D.4 Belief-Conditioned Planning (Phase 6)

Source: [belief_planning.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/belief_planning.py)

`plan_from_belief()` returns `BeliefPlan` (12 fields):
- `action`, `next_pos`, `planned_prefix`, `full_path`
- `expected_cost`, `expected_risk`, `uncertainty`
- `runner_up_gap`, `action_confidence`
- `dominant_reason` (from `DOMINANT_REASONS`: lower_risk / lower_cost / lower_uncertainty / deadline_pressure / mixed)
- `score_breakdown` (`ScoreBreakdown`: cost_term, risk_term, uncertainty_term)
- `prefix_prediction` (embedded `PrefixPrediction`, optional)

`estimate_failure_modes()` returns `FailureModeEstimate` (5 fields):
- `high_cumulative_risk`, `high_uncertainty`, `deadline_miss`, `no_safe_route`, `warning_insufficient`

### D.5 Prefix Prediction (Phase 5)

Source: [prefix_prediction.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/prefix_prediction.py)

Read-only diagnostic over the first H cells of the planned path:
- Per-cell cost/risk/uncertainty predictions
- Cumulative risk (independence approximation)
- Risky prefix cells flagged

---

## E. Tutor / Robot-Belief / Intervention Architecture

### E.1 Heuristic Tutors

Multiple coexisting tutor classes:

| File | Tutor | Logic |
|------|-------|-------|
| [time_aware_door_tutor.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/time_aware_door_tutor.py) | `TimeAwareDoorTutor` | Closes risky gate when agent approaches, respects closure budget |
| [oracle_teacher.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/oracle_teacher.py) | `OracleTeacher` | Sees true state, optimizes utility |
| [oracle_cause_teacher.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/oracle_cause_teacher.py) | `OracleCauseTeacher` | Cause-based explanatory interventions |
| [particle_teacher.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/particle_teacher.py) | `ParticleTeacher` | Particle filter inference over agent goals |

### E.2 Robot Belief (Phase 7)

Source: [robot_belief.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/robot_belief.py)

`RobotBelief` is the robot's **approximate surrogate** of the agent's belief + competence. Key features:
- **Belief copy modes**: `exact` (full copy), `noisy` (+ Gaussian noise), `stale` (sync every N steps)
- **Competence parameters**: `agent_search_budget`, `agent_heuristic_noise_std`, `agent_risk_weight`, `agent_uncertainty_weight`
- **Mismatch knobs**: `budget_mismatch`, `risk_weight_mismatch`
- `build_surrogate_predictor()` creates a frozen `LatentCostRiskHead` from snapshot

### E.3 Agent Predictor (Phase 7)

Source: [agent_predictor.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/agent_predictor.py)

`predict_agent_prefix()` runs a **counterfactual surrogate rollout** using robot's belief to predict what the agent would do. Phase 8 added `predict_agent_prefix_after_item_drop()` for shield counterfactual.

### E.4 Intervention Policy (Phase 7–8)

Source: [intervention_policy.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/intervention_policy.py)

`score_interventions()` evaluates 4 actions via counterfactual prefix prediction:

| Action | Counterfactual |
|--------|---------------|
| WAIT | Baseline prefix (no change) |
| WARN | Re-predict after belief bias |
| UNLOCK | Re-predict after gate closure |
| ITEM_DROP | Re-predict with shield added |

Returns `InterventionDecision` (8 fields):
- `action` (WAIT/WARN/UNLOCK/ITEM_DROP)
- `scores` (action → score dict)
- `reason` (dominant reason string)
- `decision_margin` (gap between best and 2nd best)
- `predicted_prefix` (list of predicted cells)
- `predicted_failure_modes` (`FailureModeEstimate`)
- `counterfactual_scores` (action → (predicted_risk, predicted_cost))
- `expected_item_effect` (optional `ItemEffect`)

### E.5 Unified Intervention Schema (Phase 8)

Source: [interventions.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/interventions.py)

- `InterventionType` enum: WAIT/WARN/UNLOCK/DROP_SHIELD/BLOCK_PATH
- `MAIN_INTERVENTION_FAMILY`: {WAIT, WARN, UNLOCK, DROP_SHIELD} — BLOCK_PATH excluded
- `ItemType` enum: SHIELD only
- `InventoryState`: binary shield {0,1}, `add_shield()`, `consume_shield()`, `clone()`
- `SHIELD_DEFAULT_RISK_REDUCTION = 0.5`

---

## F. Data Flow / Call Graph for a Typical Episode

```
runner.reset(seed, ...)
│
├── generate_lattice_v2() → GridMap, cell_features, LatticeV2Meta
├── init FeatureBeliefMap, BayesianRiskHead, LatentCostRiskHead
├── init TimeAwareDoorTutor (if tutor_mode != "none")
├── init RobotBelief (if robot_belief_mode=True)
├── init InventoryState (if intervention_family_mode=True)
└── return V2EpisodeState

runner.step(s)
│
├── 1. observe(s)
│   ├── observe_features() or observe_features_patch()
│   ├── FeatureBeliefMap.update() per observed cell
│   └── sync_robot_belief() if robot_belief_mode
│
├── 2. apply_tutor(s)
│   ├── _apply_tutor_dispatch(s)
│   │   ├── [robot_belief path]
│   │   │   ├── score_interventions() → InterventionDecision
│   │   │   ├── pick best action
│   │   │   └── execute: WARN→bias belief / UNLOCK→close gate / ITEM_DROP→add shield
│   │   ├── [heuristic path]
│   │   │   └── TimeAwareDoorTutor / warn_first / fixed_warning logic
│   │   └── [none] → no action
│
├── 3. plan_and_move(s)
│   ├── [belief_planning_mode]
│   │   ├── plan_from_belief() → BeliefPlan
│   │   └── estimate_failure_modes()
│   ├── [legacy/latent path]
│   │   ├── plan_next_action_v2() → next_pos, path
│   │   └── compute_prefix_predictions() if prefix_horizon > 0
│   │
│   ├── agent moves to next_pos
│   ├── outcome resolution:
│   │   ├── if RISKY cell → roll against true_risk (shield reduces)
│   │   ├── if trap → agent dies
│   │   └── if goal → success
│   └── risk_head / latent_predictor online learning update
│
└── check terminal: death / goal / timeout → s.done
```

---

## G. Config Switches and Execution Modes

### G.1 Runner Reset Parameters → Mode Switches

| Parameter | Values | Phase | Effect |
|-----------|--------|-------|--------|
| `tutor_mode` | `none` / `time_aware` / `warn_first` | P0 | Heuristic tutor selection |
| `warning_mode` | `none` / `fixed` | P1 | Enable warnings |
| `latent_mode` | `False` / `True` | P4 | Feature-as-latent semantics |
| `patch_radius` | 1 / 2+ | P5 | Extended observation |
| `prefix_horizon` | 0 / 5+ | P5 | Prefix diagnostics |
| `belief_planning_mode` | `False` / `True` | P6 | Belief-conditioned planner |
| `robot_belief_mode` | `False` / `True` | P7 | Robot-belief tutor |
| `intervention_family_mode` | `False` / `True` | P8 | Unified 4-action scoring |
| `item_drop_enabled` | `False` / `True` | P8 | Shield interventions |

### G.2 Legacy-Compatible Path

When all Phase 4+ switches are `False`:
- Uses V0 `BayesianRiskHead` only (no `LatentCostRiskHead`)
- `plan_next_action_v2()` uses `belief_cost` from risk_head
- No patch observation, no prefix prediction, no belief planning
- Heuristic tutor or none
- **Baselines preserved**: 9/80/68/99/100%

### G.3 Config Files

| File | Contents |
|------|----------|
| [agent.yaml](file:///f:/SCAI/Learning-agent/pedagogical_ip/configs/agent.yaml) | Belief priors, observation params, planner params |
| [teacher.yaml](file:///f:/SCAI/Learning-agent/pedagogical_ip/configs/teacher.yaml) | Particle/oracle weights, intervention costs, rollout depth |
| [env.yaml](file:///f:/SCAI/Learning-agent/pedagogical_ip/configs/env.yaml) | Environment params |
| [experiment.yaml](file:///f:/SCAI/Learning-agent/pedagogical_ip/configs/experiment.yaml) | 7-line minimal config (episodes, seed, log_dir) |
| [phase9_eval.yaml](file:///f:/SCAI/Learning-agent/pedagogical_ip/configs/phase9_eval.yaml) | Agent×teacher×env matrix with `allowed_interventions` |

> [!WARNING]
> Config files are NOT programmatically loaded by the runner. The runner uses explicit keyword args. Configs are reference files, not a driving config system.

---

## H. Metrics and Experiment System

### H.1 Existing Metrics Modules

| File | Metrics | Status |
|------|---------|--------|
| [eval_v1.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/metrics/eval_v1.py) | ECE, CFA, ToM-MSE | Uses V0 `BeliefMap`, not V2 `FeatureBeliefMap`. **Stale, not wired into runner.** |
| [online_metrics.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/metrics/online_metrics.py) | `epistemic_gain`, `frustration_score` | Raw functions, no schema. |
| [phase9_metrics.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/metrics/phase9_metrics.py) | Unified Episode/Transfer/Aggregate | **Current canonical metrics layer.** |

### H.2 Phase 9 Metrics Schema

**Episode-level** (`EpisodeSummary`): `success: bool`, `death: bool`, `timeout: bool`, `cumulative_cost`, `cumulative_risk`, `intervention_count`, `cost_prediction_error`, `risk_calibration_gap`, `uncertainty_reduction_visited`, `uncertainty_reduction_nearby`, `information_gain`, `boredom_proxy`, `frustration_proxy`, `intervention_timing_quality`.

**Transfer-level** (`TransferSummary`): same task metrics but separate type. No pedagogical metrics (no tutor present).

**Aggregate-level** (`AggregateMetrics`): `success_rate`, `success_rate_sem`, `death_rate`, `timeout_rate`, `cost_mean/std`, `risk_mean/std`, etc. All with `mean/std/n/sem`.

### H.3 Experiment Matrix

`phase9_eval.yaml`: 3 agent levels × 6 teacher conditions × 3 env conditions = **54 cells**.
Each teacher condition explicitly declares `allowed_interventions`.
Support for `--smoke` (3 jobs) and `--filter` subsets.

### H.4 Transfer Evaluation

[transfer_eval.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/metrics/transfer_eval.py):
- **Copy**: learned predictor weights (cost w/b, risk w/b)
- **Reset**: episodic belief state, position, inventory
- **Disable**: tutor (tutor_mode="none")

---

## I. Stable Interfaces

### I.1 Public / Caller-Facing Stable APIs

Entry points that external callers (scripts, experiments, tests) use directly:

| Interface | File | Stability |
|-----------|------|-----------|
| Grid generation | `generate_lattice_v2()` → `(GridMap, cell_features, LatticeV2Meta)` | ✅ Stable |
| Runner API | `LatticeV2Runner.reset()/step()/get_metrics()/get_extended_metrics()` | ✅ Stable |
| Env facade | `LatticeV2Env.reset()/step_full()/get_metrics()` | ✅ Stable |
| `FeatureBeliefMap` API | `.update()/.get_belief()/.copy()/.reset()` | ✅ Stable |
| `LatentCostRiskHead` API | `.predict_cost()/.predict_risk()/.update()` | ✅ Stable |
| Phase 9 metrics | `compute_episode_summary()`, `aggregate_summaries()` | ✅ Stable |
| Transfer eval | `run_transfer_episodes()` | ✅ Stable |
| Legacy baselines | 9/80/68/99/100% | ✅ Frozen |

### I.2 Internal Stable Schemas

Dataclass outputs that are internally stable but are NOT caller entry points:

| Schema | File | Fields |
|--------|------|--------|
| `V2EpisodeState` | `lattice_v2_runner.py` | 30+ fields (additive only) |
| `BeliefPlan` | `belief_planning.py` | 12 fields: action, next_pos, planned_prefix, full_path, expected_cost, expected_risk, uncertainty, runner_up_gap, action_confidence, dominant_reason, score_breakdown, prefix_prediction |
| `ScoreBreakdown` | `belief_planning.py` | 3 fields: cost_term, risk_term, uncertainty_term |
| `FailureModeEstimate` | `belief_planning.py` | 5 fields: high_cumulative_risk, high_uncertainty, deadline_miss, no_safe_route, warning_insufficient |
| `InterventionDecision` | `intervention_policy.py` | 8 fields: action, scores, reason, decision_margin, predicted_prefix, predicted_failure_modes, counterfactual_scores, expected_item_effect |
| `PrefixPrediction` | `prefix_prediction.py` | 8 fields: prefix_cells, cost_predictions, risk_predictions, cost_uncertainties, risk_uncertainties, cumulative_cost, cumulative_risk, risky_prefix_cells |
| `EpisodeSummary` | `phase9_metrics.py` | 20 fields (see interface_inventory.md) |
| `TransferSummary` | `phase9_metrics.py` | 12 fields (see interface_inventory.md) |
| `AggregateMetrics` | `phase9_metrics.py` | 22 fields (see interface_inventory.md) |
| `InterventionType` / `InventoryState` | `interventions.py` | Enum + state with `add_shield()/consume_shield()/clone()` |

---

## J. Unstable / Messy / Under-documented Areas

### J.1 Parallel Planner Paths

`plan_and_move()` in [lattice_v2_runner.py:284–335](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/lattice_v2_runner.py#L284-L335) has two parallel branches:
- `belief_planning_mode` → `plan_from_belief()`
- else → `plan_next_action_v2()`

Both call into the same underlying A*, but with different wrappers. The non-belief path still imports and uses `BayesianRiskHead` for cost computation even in latent mode, creating conceptual overlap.

### J.2 BoundedRationalAgent vs Runner

`BoundedRationalAgent` in [bounded_agent.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/bounded_agent.py) is the V0 agent class. The V2 runner **does not use it** — it inlines equivalent logic directly. The class still exists but is orphaned from the main execution path.

### J.3 eval_v1.py is Legacy-Dead

[eval_v1.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/metrics/eval_v1.py) defines ECE/CFA/ToM-MSE using V0 `BeliefMap`. It is not imported by Phase 9 metrics, the runner, or any script. The `compute_episode_metrics()` convenience function is never called in production.

### J.4 online_metrics.py Overlaps Phase 9

[online_metrics.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/metrics/online_metrics.py) defines `epistemic_gain()` and `frustration_score()`. Phase 9's `phase9_metrics.py` reimplements these as `_information_gain()` and `_frustration_proxy()` with different signatures. Two implementations, no shared code.

### J.5 Config Files Not Programmatically Loaded

`agent.yaml`, `teacher.yaml`, `env.yaml`, `experiment.yaml` are **reference documents**, not loaded by the runner. The runner uses explicit keyword arguments. `phase9_eval.yaml` IS loaded by `run_phase9_matrix.py` but does not feed directly into `runner.reset()` — manual parameter mapping happens in the script.

### J.6 Naming Inconsistencies

| Concept | Name in runner | Name in intervention_policy | Name in phase9 | Name in legacy |
|---------|---------------|----------------------------|-----------------|----------------|
| Risky cells entered | `risky_entered` | — | `cumulative_risk` | `risky` |
| Door closures | `closures` | `UNLOCK` | `intervention_count` (merged) | `closures` |
| Warnings | `warnings_sent` | `WARN` | included in `intervention_count` | `warnings` |
| Shield | `inventory.shield` | `ITEM_DROP` | `shield_remaining` | n/a |

### J.7 Teacher Conditions Not Enforced at Runner Level

`phase9_eval.yaml` declares `allowed_interventions` per teacher condition, but the runner does not enforce this — it is only checked by tests. A `robot_belief` tutor could still score and select disallowed actions.

### J.8 Multiple Observation Models

- `observation_model.py` → V0 cost/risk observations from `true_cost/true_risk`
- `observe_features()` in runner → V2 feature observations from `cell_features`
- `observe_features_patch()` → extended patch version

The V0 `observation_model.py` is still imported by `bounded_agent.py` but not used by the V2 runner path.

---

## K. Prioritized Gap-Analysis Checklist

See [gap_checklist.md](file:///f:/SCAI/Learning-agent/pedagogical_ip/task_list/gap_checklist.md) for the full ranked list.

---

## Diagnostic Questions

### 1. What is the current canonical execution path for V2?

```
LatticeV2Runner.reset(seed, latent_mode=True, ...)
  → generate_lattice_v2() + init FeatureBeliefMap + LatentCostRiskHead
while not done:
  LatticeV2Runner.step(s)
    → observe() → apply_tutor() → plan_and_move()
LatticeV2Runner.get_metrics(s)
```

The canonical path uses `latent_mode=True`. `belief_planning_mode` and `robot_belief_mode` are optional additions. The `LatticeV2Env` facade wraps this but is less commonly used directly.

### 2. Which parts are legacy-compatible but no longer conceptually central?

- `BoundedRationalAgent` — V0 agent, not used by runner
- `observation_model.py` — V0 observation model, not in V2 path
- `belief.py` (`BeliefMap`) — V0 belief, replaced by `FeatureBeliefMap`
- `eval_v1.py` — V0 metrics, replaced by `phase9_metrics.py`
- `map_generator.py` — V0 8×8 grid, replaced by `lattice_v2.py`
- `pedagogical_grid.py` — V0 env, replaced by `lattice_v2_env.py`

### 3. Which interfaces are stable enough to treat as APIs?

See Section I above. Key stable APIs: runner reset/step/get_metrics, env facade, FeatureBeliefMap, LatentCostRiskHead, InterventionType/Decision, Phase 9 output schema.

### 4. Where are the remaining sources of duplication?

- `online_metrics.py` vs `phase9_metrics.py` (frustration, info gain)
- `eval_v1.py` vs `phase9_metrics.py` (ECE concept)
- `observation_model.py` vs runner-inline `observe_features()`
- `BoundedRationalAgent` vs runner-inline agent logic
- Config files vs runner keyword args

### 5. Smallest next cleanup pass for maintainability?

1. Mark `bounded_agent.py`, `observation_model.py`, `eval_v1.py`, `pedagogical_grid.py` as `# DEPRECATED — legacy V0` with docstring warnings
2. Consolidate `online_metrics.py` into `phase9_metrics.py`
3. Add `allowed_interventions` enforcement in `score_interventions()`
4. Unify naming: `closures` → `unlock_count`, `warnings_sent` → `warn_count` consistently
