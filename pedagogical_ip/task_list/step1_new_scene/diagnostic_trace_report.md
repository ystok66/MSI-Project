# Diagnostic Report: Episode Traces + Learner Architecture + Directory Trees

---

## Part 1: Full Episode Traces (seed=0, medium)

### 1.1 deadline_gate — UNLOCK condition

```
Grid: 7×27  t_max=33  goal=(2, 25)
Door at: (1,2)   |   Risky cells: (3,3) (3,5) (3,8) (3,11) (3,18) (3,20) (3,24)
Shield: NO       |   allowed_interventions: {UNLOCK}

Step  Pos      σ²_obs    Tutor   Move               r̂      n_upd  Event
─────────────────────────────────────────────────────────────────────────
  1   (2,1)    0.0096    UNLOCK  (2,1) → (2,2)      0.500    0
  2   (2,2)    0.0086    UNLOCK  (2,2) → (1,2)      0.492    1     ← enters shortcut
  3   (1,2)    0.0069    UNLOCK  (1,2) → (1,3)      0.487    2
  4   (1,3)    0.0078    UNLOCK  (1,3) → (1,4)      0.488    3
  …   (row 1 straight line, one cell per step)
 25   (1,24)   0.0086    UNLOCK  (1,24)→ (2,24)     0.417   24
 26   (2,24)   0.0078    UNLOCK  (2,24)→ (2,25)     0.409   25     GOAL

>>> GOAL after 26 steps | risky_entered=0 | traps_hit=0 | unlocks=1 | warns=0
```

**Analysis:**
- Tutor issues UNLOCK at step 1 → gate at (1,2) opens.
- Agent enters row 1 at step 2 and goes straight to goal. Clean, no bouncing.
- `r̂` starts at 0.5 (uninformative prior) and gradually decreases as the agent traverses safe row-1 cells and online-learns "these features → low risk".
- `n_updates` increments by 1 per step (each traversal produces a label).
- **Key**: Robot_belief keeps issuing UNLOCK every step (because the scoring still finds it the best action). Only the first one actually opens the gate; the rest are no-ops. This is harmless but reveals the tutor doesn't track "I already did this."

---

### 1.2 hazard_belt — robot_belief (WARN + ITEM_DROP available)

```
Grid: 7×20  t_max=36  goal=(2, 18)
Doors at: (1,2) (1,8) (1,13)
Risky cells: (1,3)(1,4)(1,5) (1,9)(1,10) (1,14)(1,15)(1,16) (3,9)(3,10)
Shield: NO  |  allowed_interventions: all

Step  Pos      σ²_obs    Tutor   Move               r̂      n_upd  Event
─────────────────────────────────────────────────────────────────────────
  1   (2,1)    0.0096    WARN    (2,1) → (2,2)      0.597    3
  2   (2,2)    0.0086    WARN    (2,2) → (1,2)      0.648    8
  3   (1,2)    0.0078    WARN    (1,2) → (1,3)      0.658   12     RISKY
  4   (1,3)    0.0076    WARN    (1,3) → (1,4)      0.579   13     RISKY
  5   (1,4)    0.0084    WARN    (1,4) → (1,5)      0.513   14     RISKY
  6   (1,5)    0.0084    WARN    (1,5) → (1,6)      0.460   15
  7   (1,6)    0.0086    WARN    (1,6) → (2,6)      0.452   16     ← drops to row 2
  8   (2,6)    0.0078    WARN    (2,6) → (2,7)      0.446   17
  9   (2,7)    0.0078    WARN    (2,7) → (2,8)      0.444   18
 10   (2,8)    0.0086    WARN    (2,8) → (3,8)      0.440   19     ← drops to row 3
 11   (3,8)    0.0078    WARN    (3,8) → (3,9)      0.436   20     RISKY
 12   (3,9)    0.0076    WARN    (3,9) → (3,10)     0.412   21     RISKY
 13-25 (safe lane zigzag through rows 3-4)                          ...
 25   (2,17)   0.0086    WARN    (2,17)→(2,18)      0.382   34     GOAL

>>> GOAL after 25 steps | risky_entered=5 | traps_hit=0 | unlocks=0 | warns=3
```

**Analysis:**
- Tutor chose **WARN every single step**. Never chose ITEM_DROP.
- Agent survived 5 risky cells (luck — risk rolls succeeded).
- `n_updates` jumps from 3→8→12 in first 3 steps (patch observation labels multiple cells per step).
- Agent navigated a mixed route: row 1 → row 2 → row 3 safe zigzag.
- **Problem**: Tutor never drops a shield despite the belt being unavoidable. WARN is scored higher than ITEM_DROP in the counterfactual. This is why `robot_belief=40% < item_only=60%` — the tutor systematically picks the wrong intervention for this family.
- `r̂` stays around 0.4–0.65 throughout — the risk head learns "moderate risk" from surviving risky cells, but never sees a death to calibrate properly to high risk.

---

### 1.3 fork_trap — robot_belief (WARN + ITEM_DROP available)

```
Grid: 7×10  t_max=27  goal=(2, 8)
Door at: (3,2)
Risky cells: (3,3) (3,4) (3,5) (3,6)
Shield: NO  |  allowed_interventions: all

Step  Pos      σ²_obs    Tutor   Move               r̂      n_upd  Event
─────────────────────────────────────────────────────────────────────────
  1   (2,1)    0.0096    WARN    (2,1) → (2,2)      0.614    4
  2   (2,2)    0.0086    WARN    (2,2) → (3,2)      0.595    5     ← enters row 3
  3   (3,2)    0.0078    WARN    (3,2) → (3,3)      0.579    6     DEATH

>>> DEATH after 3 steps | risky_entered=1 | traps_hit=0 | unlocks=0 | warns=1
```

**Analysis:**
- Agent dies in 3 steps. Tutor fires WARN at step 1 but agent continues into risky lane anyway.
- **Key problem**: WARN fires too late (step 1) and applies to forward segments, but agent at (2,2) still plans (3,2)→(3,3) because:
  - The WARN lane bias affects `warned_cell_extra` for risky cells, but the agent's planner may not weigh it enough
  - Or the WARN is applied to the wrong segment (segment ahead of agent, not the one at (3,2))
- `r̂ = 0.579` at death cell → the risk penalty is `5 × -ln(1-0.579) = 4.33`, but with `n_updates=6` → `α=0.6`, and necessity could be high → risk penalty further discounted. The planner may not see the risky lane as sufficiently dangerous.
- **This is a classic "warning doesn't change the plan fast enough" problem**. The robot_belief achieves 65% over 20 seeds because in many seeds the fork geometry is favorable enough for warnings to redirect the agent, but in seed=0 it fails.

---

## Part 2: Agent Learning Architecture

### Core classes and their roles

```
src/agents/
├── feature_belief.py
│   └── FeatureBeliefMap          ← Gaussian posterior q(z) = N(μ,Σ) per cell
│       .mean[H,W,4]              ← posterior mean μ_i (float64)
│       .var[H,W,4]               ← posterior variance Σ_i (diagonal, float64)
│       .memory[H,W]              ← CellMemoryMeta provenance grid
│       .update(r, c, obs, var)   ← Kalman filter update
│       .apply_unlock_update()    ← UNLOCK: Σ *= (1-β)
│       .apply_warn_update()      ← WARN: μ += α·v_warn
│       .mark_traversed()         ← provenance update
│
├── cost_risk_model.py
│   └── LatentCostRiskHead        ← composes cost + risk linear heads
│       .cost_head                  → BayesianCostHead
│       .risk_head                  → BayesianRiskHead
│       .predict_cost(x)           → float (w_c·x + b_c, clipped ≥ 0.1)
│       .predict_risk(x)           → float (sigmoid(w_r·x + b_r))
│       .predict_cost_uncertainty_from_var(v) → float (w_c^T diag(v) w_c)
│       .predict_risk_uncertainty_from_var(v) → float (w_r^T diag(v) w_r)
│       .update_from_outcome(x, cost_label, risk_label)  ← online MAP update
│       .n_updates                 → int (from risk_head)
│
├── risk_model.py
│   └── BayesianRiskHead          ← the actual risk learning head
│       w: ndarray[4]             ← weight vector, init=zeros
│       b: float                  ← bias, init=0.0
│       lr: float                 ← 0.3 (SGD step size)
│       prior_var: float          ← 1.0 (Gaussian weight prior)
│       n_updates: int            ← counter
│       .predict_risk(x) → sigmoid(w·x + b)
│       .update_from_label(x, y, weight)  ← SGD on negative log-posterior
│         grad_w = -(y-p)·x + w/prior_var
│         w -= lr * clip(grad_w, max_norm=5.0)
│         w = clamp(w, max_norm=10.0)
│
├── planner_astar.py
│   └── cell_cost_v2_latent(r, c, belief_mean, predictor, passable,
│                           feature_belief_var, route_necessity)
│       → per-cell planning score (the core formula)
│   └── plan_next_action_v2()     ← A* planning entry point
│   └── plan_with_alternatives_v2() ← top-k alternative plans
│   └── _astar_core()             ← raw A* with budget=30
│
├── route_necessity.py
│   └── compute_route_necessity(pos, goal, passable, t, t_max, unvisited)
│       → float in [0,1] via BFS comparison
│
├── belief_planning.py
│   └── plan_from_belief()        ← belief-conditioned planning (optional path)
│   └── estimate_failure_modes()  ← FailureModeEstimate
│
├── prefix_prediction.py
│   └── compute_prefix_predictions()  ← diagnostic over planned path prefix
│
├── belief.py                     ← LEGACY (V0 BeliefMap, not used by V2 runner)
├── bounded_agent.py              ← LEGACY (V0 agent class, not used)
├── observation_model.py          ← LEGACY (V0 obs model, not used)
├── warning_update.py             ← warning heuristic application
├── pragmatic_warning.py          ← RSA-inspired warning selection
└── belief_protocol.py            ← belief interface protocol
```

### What kind of learner is this?

**It is: 2 × Online Bayesian linear heads + Kalman filter belief tracking.**

Specifically:

| Component | Method | Details |
|-----------|--------|---------|
| **Belief over features** | Diagonal Gaussian + Kalman | `FeatureBeliefMap.update()`: standard Kalman with per-cell `(mean, var)`. Updated from noisy observations. **Not learned** — pure Bayesian. |
| **Cost predictor** | Linear + SGD on Gaussian posterior | `BayesianCostHead`: `ĉ = w^T x + b`. Updated via SGD on `-(y-pred)·x + w/σ²_prior`. Learning rate 0.3. Gradient clipped. |
| **Risk predictor** | Logistic linear + SGD on Bernoulli posterior | `BayesianRiskHead`: `r̂ = sigmoid(w^T x + b)`. Updated via SGD on `-(y-p)·x + w/σ²_prior`. Labels: 0.0=safe, 1.0=fatal, 0.15-0.25=mild. |

**No neural networks. No hidden layers. No replay buffers. No RL.** The entire learning system is:
1. A 4D → 1 linear cost head
2. A 4D → 1 logistic risk head
3. A per-cell diagonal Gaussian belief
4. MAP estimation with L2 prior (equivalent to ridge regression)

**Online learning sequence per step:**
```python
# In lattice_v2_runner.py plan_and_move(), after the agent moves:
x = s.feature_belief.mean[new_pos]  # current belief about cell features
cost_label = true_cost(cell)        # ground truth traversal cost
risk_label = oracle_risk or binary_outcome  # depends on supervision mode
s.latent_predictor.update_from_outcome(x, cost_label, risk_label)
```

### Transfer mechanism

`transfer_eval.py` copies **only** `(w_cost, b_cost, w_risk, b_risk)` — the 10 learned scalars (4+1 per head). Resets everything else (position, belief, inventory). Tests whether the agent learned generalizable cost/risk prediction.

---

## Part 3: Directory Trees

### src/envs/ (9 files)

```
src/envs/
├── __init__.py                     (118B)   imports
├── lattice_v2.py                   (14KB)   grid generation, SegmentMeta, LatticeV2Meta
├── lattice_v2_env.py               (9.5KB)  environment facade (Observation, StepResult)
├── lattice_v2_runner.py            (28KB)   ★ MAIN RUNNER — step loop, all wiring
├── map_families.py                 (28KB)   low-level family grid builders
├── map_generator.py                (6KB)    CellType enum, GridMap dataclass
├── scenario_families.py            (49KB)   ★ scenario family generation (fork_trap etc.)
├── benchmark_generator.py          (1.3KB)  benchmark utils
└── pedagogical_grid.py             (15KB)   LEGACY V0 environment
```

### src/agents/ (14 files)

```
src/agents/
├── __init__.py                     (78B)    imports
├── feature_belief.py               (7.7KB)  ★ FeatureBeliefMap + CellMemoryMeta
├── cost_risk_model.py              (8.4KB)  ★ LatentCostRiskHead (composes two heads)
├── risk_model.py                   (4.2KB)  ★ BayesianRiskHead (logistic linear + SGD)
├── planner_astar.py                (18KB)   ★ cell_cost_v2_latent, A* planning
├── route_necessity.py              (4.1KB)  ★ BFS route necessity (Phase 10)
├── belief_planning.py              (10KB)   belief-conditioned planning (optional path)
├── prefix_prediction.py            (3.3KB)  prefix diagnostics
├── warning_update.py               (7.7KB)  warning application logic
├── pragmatic_warning.py            (1.8KB)  RSA-style warning selection
├── belief_protocol.py              (1.2KB)  belief interface protocol
├── belief.py                       (10KB)   LEGACY V0 BeliefMap
├── bounded_agent.py                (11KB)   LEGACY V0 agent
└── observation_model.py            (6.9KB)  LEGACY V0 observation model
```

### src/teachers/ (13 files)

```
src/teachers/
├── __init__.py                     (220B)   imports
├── intervention_policy.py          (9.3KB)  ★ score_interventions() — 4-way scoring
├── interventions.py                (4.9KB)  ★ InterventionType, InventoryState
├── robot_belief.py                 (5.5KB)  ★ RobotBelief surrogate
├── agent_predictor.py              (6.6KB)  ★ counterfactual prefix prediction
├── cause_scoring.py                (13KB)   cause-based scoring (Phase 8)
├── block_scoring.py                (6.5KB)  block-path scoring
├── particle_teacher.py             (24KB)   ParticleTeacher (particle filter inference)
├── oracle_teacher.py               (8.7KB)  OracleTeacher (full visibility)
├── oracle_cause_teacher.py         (11KB)   cause-based oracle
├── time_aware_door_tutor.py        (5.7KB)  heuristic door tutor
├── rsa_warning.py                  (9.5KB)  RSA warning generation
└── utilities.py                    (2.6KB)  teacher utilities
```

★ = most important files (read these first)

---

## Part 4: Trace-Based Diagnostic Summary

### What the traces reveal about each layer

| Layer | deadline_gate | hazard_belt | fork_trap |
|-------|--------------|-------------|-----------|
| **Planner** | ✅ Correct. Straight line through row 1 | ✅ Navigates mixed route | ⚠️ Enters risky immediately |
| **Belief** | ✅ `r̂` correctly decreases with traversals | ✅ `r̂` calibrates toward moderate | ⚠️ No time to learn before death |
| **Tutor timing** | ⚠️ Re-issues UNLOCK every step | ❌ WARN every step, never ITEM_DROP | ❌ WARN at step 1 but too late |
| **Tutor selection** | ✅ UNLOCK is correct choice | ❌ Should choose ITEM_DROP | ⚠️ WARN is correct but ineffective |
| **Outcome** | GOAL (26 steps) | GOAL (lucky: survived 5 risky) | DEATH (3 steps) |

### Root cause per family

**deadline_gate**: Working correctly. The only issue is cosmetic — tutor keeps issuing UNLOCK after the gate is already open.

**hazard_belt**: **Tutor scoring bug**. `score_interventions()` ranks WARN above ITEM_DROP because the counterfactual for WARN (biased belief → altered plan) shows larger change than ITEM_DROP (shield → reduced risk penalty). But WARN can't protect against unavoidable risky cells — only shield can. The scoring doesn't adequately model "this risk is unavoidable, mitigation > information."

**fork_trap**: **WARN too late + too weak**. Agent at (2,2) receives WARN but the warning applies to segment cells ahead — the risky cells at (3,3)-(3,6). The agent's planner still prefers row 3 because the warned_cell_extra bias doesn't outweigh the shorter-path attraction. Also: with `n_updates=5` the risk penalty is partially discounted by learning_factor → necessity discount combination, making the risky path look more attractive than it should.
