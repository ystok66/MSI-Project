# Interface Inventory

## 1. Environment API

### `generate_lattice_v2()`
- **File**: `src/envs/lattice_v2.py`
- **Input**: `seed, difficulty, n_segments, latent_mode`
- **Output**: `(GridMap, cell_features: ndarray[H,W,4], LatticeV2Meta)`

### `LatticeV2Runner`
- **File**: `src/envs/lattice_v2_runner.py`
- `reset(seed, **config) → V2EpisodeState`
- `step(s) → V2EpisodeState` (mutates and returns state)
- `observe(s) → None`
- `apply_tutor(s) → None`
- `plan_and_move(s) → None`
- `get_metrics(s) → dict` (9 fields)
- `get_extended_metrics(s) → dict` (Phase 9, 17 fields)

### `LatticeV2Env`
- **File**: `src/envs/lattice_v2_env.py`
- `reset(seed, **config) → Observation`
- `step_full() → StepResult`
- `step_teacher() → TeacherInfo`
- `step_agent() → StepResult`
- `get_metrics() → dict`
- `get_state() → StateSnapshot`

---

## 2. Agent APIs

### `FeatureBeliefMap`
- **File**: `src/agents/feature_belief.py`
- `.update(row, col, obs_mean, obs_var, t)`
- `.get_belief(row, col) → (mean, var)`
- `.copy() → FeatureBeliefMap`
- Fields: `.mean[H,W,d]`, `.var[H,W,d]`, `.observed[H,W]`, `.visit_count[H,W]`, `.last_observed_t[H,W]`

### `LatentCostRiskHead`
- **File**: `src/agents/cost_risk_model.py`
- `.predict_cost(x) → float`
- `.predict_risk(x) → float`
- `.predict_cost_uncertainty(x) → float`
- `.predict_risk_uncertainty(x) → float`
- `.update(x, true_cost, true_risk) → None`
- Composes: `BayesianCostHead` (w, b scalar, MAP) + `BayesianRiskHead` (w, b scalar, Bernoulli)

### `plan_with_alternatives_v2()`
- **File**: `src/agents/planner_astar.py`
- **Input**: pos, goal, belief_cost, feature_mean, risk_head, budget, passable, warned, latent_predictor, λ_c, λ_uc, λ_ur, inventory_state
- **Output**: `(action: str, next_pos: tuple, path: list, candidate_scores: dict[str, float])`

### `plan_from_belief()` → `BeliefPlan` (12 fields)
- **File**: `src/agents/belief_planning.py`
- **Output fields**:
  - `action: str`
  - `next_pos: tuple[int, int]`
  - `planned_prefix: list[tuple[int, int]]`
  - `full_path: list[tuple[int, int]]`
  - `expected_cost: float`
  - `expected_risk: float` (cumulative independence approx)
  - `uncertainty: float` (mean over prefix)
  - `runner_up_gap: float` (best vs 2nd path)
  - `action_confidence: float` (normalized: gap / (gap + temperature))
  - `dominant_reason: str` (from `DOMINANT_REASONS` frozenset)
  - `score_breakdown: ScoreBreakdown` (cost_term, risk_term, uncertainty_term)
  - `prefix_prediction: Optional[PrefixPrediction]`

### `estimate_failure_modes()` → `FailureModeEstimate` (5 fields)
- **File**: `src/agents/belief_planning.py`
- **Output fields**:
  - `high_cumulative_risk: float`
  - `high_uncertainty: float`
  - `deadline_miss: float` (time pressure vs path length)
  - `no_safe_route: float` (fraction of candidates with high risk)
  - `warning_insufficient: float` (risky cells not influenced by warning)

### `PrefixPrediction` (8 fields)
- **File**: `src/agents/prefix_prediction.py`
- **Output fields**:
  - `prefix_cells: list[tuple[int, int]]`
  - `cost_predictions: list[float]`
  - `risk_predictions: list[float]`
  - `cost_uncertainties: list[float]`
  - `risk_uncertainties: list[float]`
  - `cumulative_cost: float`
  - `cumulative_risk: float` (independence approximation)
  - `risky_prefix_cells: list[tuple[int, int]]`

---

## 3. Teacher APIs

### `RobotBelief` (17 fields)
- **File**: `src/teachers/robot_belief.py`
- `init_robot_belief(mean, var, predictor, copy_mode, ...) → RobotBelief`
- `sync_robot_belief(rb, mean, var, predictor, t)`
- `build_surrogate_predictor(rb) → LatentCostRiskHead`
- **Key fields**: `agent_belief_mean/var`, `agent_search_budget`, `agent_heuristic_noise_std`, `agent_risk_weight`, `agent_uncertainty_weight`, `agent_lambda_c/uc/ur`, `copy_mode`, `belief_noise_std`, `stale_interval`, `budget_mismatch`, `risk_weight_mismatch`, `_predictor_cost_w/b`, `_predictor_risk_w/b`

### `predict_agent_prefix()`
- **File**: `src/teachers/agent_predictor.py`
- **Input**: robot_belief, agent_pos, goal, belief_cost, predictor, meta
- **Output**: `AgentPrediction` (predicted trajectory from surrogate rollout)
- **Variants**: `predict_agent_prefix_after_warn()`, `predict_agent_prefix_after_unlock()`, `predict_agent_prefix_after_item_drop()`

### `score_interventions()` → `InterventionDecision` (8 fields)
- **File**: `src/teachers/intervention_policy.py`
- **Output fields**:
  - `action: str` — **note: still `str`, not `InterventionType` enum.** Code also defines `VALID_ACTIONS = frozenset({"WAIT", "WARN", "UNLOCK", "ITEM_DROP"})` as string-level guard.
  - `scores: dict` (action → scalar score)
  - `reason: str` (dominant reason for choice)
  - `decision_margin: float` (gap between best and 2nd best)
  - `predicted_prefix: list[tuple[int, int]]` (baseline prediction)
  - `predicted_failure_modes: FailureModeEstimate` (from baseline)
  - `counterfactual_scores: dict` (action → (predicted_risk, predicted_cost))
  - `expected_item_effect: Optional[ItemEffect]` (Phase 8, shield effect)

### Intervention Schema
- **File**: `src/teachers/interventions.py`
- `InterventionType`: WAIT, WARN, UNLOCK, DROP_SHIELD, BLOCK_PATH
- `MAIN_INTERVENTION_FAMILY`: {WAIT, WARN, UNLOCK, DROP_SHIELD}
- `InventoryState`: `.has_shield()`, `.add_shield()`, `.consume_shield()`, `.clone()`
- `ItemEffect`: `item_type, target_cell, risk_reduction`
- `SHIELD_DEFAULT_RISK_REDUCTION = 0.5`

---

## 4. Metrics APIs

### Phase 9 Metrics
- **File**: `src/metrics/phase9_metrics.py`
- `compute_episode_summary(state, ...) → EpisodeSummary`
- `compute_transfer_summary(state, ...) → TransferSummary`
- `aggregate_summaries(list[EpisodeSummary]) → AggregateMetrics`
- `aggregate_transfer_summaries(list[TransferSummary]) → AggregateMetrics`

### Transfer Evaluation
- **File**: `src/metrics/transfer_eval.py`
- `snapshot_learned_params(state) → dict`
- `apply_learned_params(state, snapshot)`
- `run_transfer_episodes(runner, trained_state, n, seeds) → list[TransferSummary]`

---

## 5. Output Schemas (complete field lists)

### `get_metrics()` — 9 fields
```
survived, reached_goal, steps, t_max, traps, risky, closures, warnings, cue_seen
```

### `get_extended_metrics()` — 17 core + 2 conditional fields
```
(all 9 above) + success, death, timeout, cumulative_cost, cumulative_risk,
intervention_count, has_inventory, shield_remaining

Conditional (only when last_intervention is not None):
  last_intervention_action, last_intervention_scores
```

### `EpisodeSummary` — 20 fields
```
Identity (4):  seed, agent_level, teacher_condition, env_condition
Task (8):      success, death, timeout, steps, cumulative_cost, cumulative_risk,
               intervention_count, intervention_types_used
Learning (4):  cost_prediction_error, risk_calibration_gap,
               uncertainty_reduction_visited, uncertainty_reduction_nearby
Pedagogical (4): information_gain, boredom_proxy, frustration_proxy,
                  intervention_timing_quality
```

### `TransferSummary` — 12 fields
```
Identity (4):  seed, agent_level, teacher_condition, env_condition
Task (6):      success, death, timeout, steps, cumulative_cost, cumulative_risk
Learning (2):  cost_prediction_error, risk_calibration_gap
```

### `AggregateMetrics` — 22 fields
```
Identity (4):  agent_level, teacher_condition, env_condition, n
Task (9):      success_rate, success_rate_sem, death_rate, timeout_rate,
               cost_mean, cost_std, risk_mean, risk_std, intervention_count_mean
Learning (5):  cost_error_mean, cost_error_std, calibration_gap_mean,
               calibration_gap_std, uncertainty_reduction_mean
Pedagogical (4): info_gain_mean, boredom_mean, frustration_mean, timing_quality_mean
```
