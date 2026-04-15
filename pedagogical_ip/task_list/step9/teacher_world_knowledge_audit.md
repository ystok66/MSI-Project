# Report: Teacher World Knowledge + Trajectory-Based Learner Inference Audit

## 1. Executive Summary

### What Teacher Actually Knows Today

The teacher's canonical decision path (`score_interventions()`) receives:

| Category | Data | Is True World Value? |
|----------|------|:---:|
| **Agent belief snapshot** | `RobotBelief` (mean/var, exact/noisy/stale copy) | ❌ |
| **Predictor snapshot** | `deepcopy(LatentCostRiskHead)` | ❌ (agent's learned model) |
| **Topology** | `passable[H,W]`, `belief_cost[H,W]` | ✅ (shared mutable state) |
| **Segment structure** | `meta.segments` (risky_cells, trap_cell, gates) | ✅ (topology ground truth) |
| **Agent position / goal** | `agent_pos`, `goal` | ✅ |
| **Time** | `t`, `t_max` | ✅ |
| **Inventory** | `InventoryState` | ✅ |
| **Perceptual access** | `PerceptualAccessState` (ρ map) | ❌ (teacher's own estimate) |

**NOT passed to `score_interventions()`:**
- `gridmap.true_cost[H,W]`
- `gridmap.true_risk[H,W]`
- `meta.world_weights` (WorldWeights)
- `meta.cell_features[H,W,d]` (true latent z vectors)

### What Learner-State Modeling Is Today

Point-copy with 3 modes:
- **exact**: `rb.agent_belief_mean = agent_belief_mean.copy()`
- **noisy**: `exact + N(0, σ²)` noise
- **stale**: skip sync unless `(t - last_sync_t) >= stale_interval`

Predictor: `deepcopy(latent_predictor)` — full snapshot of agent's learned Bayesian heads.

**Zero trajectory-based inference.** No action likelihood, no inverse planning, no posterior update from observed behavior.

### Must They Be Changed Together?

**No: they CAN be changed independently, but there's a strong design coupling in the teacher scoring interface.**

**Recommended: Option A** — World-aware can be added first as a read-only hazard-timing signal, without touching learner state modeling. Trajectory inference can be added later as a separate upgrade to `sync_robot_belief()`.

### Recommended First Step

Add a `WorldHazardView` (read-only accessor for `true_risk` thresholded into hazard zones) to `score_interventions()`. This does NOT let the teacher plan for the agent — it only improves **when** to intervene, not **what path** the agent should take.

---

## 2. Current Canonical Dataflow

```
┌──────────────────── ENVIRONMENT ────────────────────┐
│ GridMap:                                             │
│   .true_cost[H,W]    ← world fact                   │
│   .true_risk[H,W]    ← world fact                   │
│   .cell_types[H,W]   ← world fact                   │
│                                                      │
│ LatticeV2Meta:                                       │
│   .segments[]         ← structure metadata           │
│   .cell_features[H,W,d] ← true latent z             │
│   .world_weights      ← WorldWeights generative fn   │
│   .all_door_positions ← structure metadata           │
└─────────────────────┬───────────────────────────────┘
                      │
    ┌─────── reset ───┘
    │
    ▼
┌──────────────────── AGENT STATE ────────────────────┐
│ FeatureBeliefMap:                                    │
│   .mean[H,W,d]        ← agent's Kalman estimate     │
│   .var[H,W,d]         ← agent's uncertainty         │
│                                                      │
│ LatentCostRiskHead:                                  │
│   .cost_head (BayesianCostHead)                      │
│   .risk_head (BayesianRiskHead)                      │
│   updates from outcome: true_cost, risk_label        │
│                                                      │
│ V2EpisodeState:                                      │
│   .passable[H,W]      ← shared mutable (tutor can   │
│   .belief_cost[H,W]      modify via WARN/UNLOCK)     │
└─────────────────────┬───────────────────────────────┘
                      │
    ┌── sync_robot_belief (exact/noisy/stale copy) ──┘
    │     copies: mean, var, deepcopy(predictor)
    ▼
┌──────────────────── TEACHER STATE ──────────────────┐
│ RobotBelief:                                         │
│   .agent_belief_mean   ← copied from agent           │
│   .agent_belief_var    ← copied from agent           │
│   ._predictor_snapshot ← deepcopy of predictor       │
│   .agent_planner_weights ← PlannerWeights            │
│   .agent_search_budget ← configured (with mismatch)  │
│                                                      │
│ PerceptualAccessState:                               │
│   .seen_prob[H,W]     ← teacher's OWN estimate      │
│   .effective_obs_var   ← teacher's OWN estimate      │
└─────────────────────┬───────────────────────────────┘
                      │
    ┌── score_interventions(rb, agent_pos, goal,
    │     belief_cost, passable, meta, ...) ──────────┘
    │
    ▼
┌──────────────────── COUNTERFACTUAL ROLLOUT ──────────┐
│ predict_agent_prefix(rb, ...) →                      │
│   build_surrogate_predictor(rb) → deepcopy snapshot  │
│   plan_from_belief(surrogate belief, surrogate head) │
│   → BeliefPlan (prefix, risk, cost, uncertainty)     │
│                                                      │
│ predict_agent_prefix_after_warn(...)                 │
│ predict_agent_prefix_after_unlock(...)               │
│ predict_agent_prefix_after_item_drop(...)            │
│   → counterfactual BeliefPlan for each action        │
│                                                      │
│ diagnose_bottleneck(...) → epistemic/structural/     │
│   outcome scores                                     │
│                                                      │
│ FINAL: argmax Q_action → InterventionDecision        │
└──────────────────────────────────────────────────────┘
```

### Critical annotations:
- **world fact** labels: `gridmap.true_cost/risk`, `meta.cell_features`, `meta.world_weights`
- **copied data** labels: `rb.agent_belief_mean/var`, `rb._predictor_snapshot`
- **oracle data that IS passed**: `meta.segments` (topology truth), `passable`, `belief_cost`
- **oracle data NOT passed**: `gridmap.true_cost`, `gridmap.true_risk`, `meta.cell_features`
- **family bypass**: DTMB oracle (`dtmb_helpers.py`) reads `meta.commitment_points`, `meta.belt_cells`, `meta.door_positions` directly — **not going through `score_interventions`** at all

---

## 3. Audit A — Teacher World Knowledge Boundary

### 3.1 Current information accessible to teacher

| Information Item | Object / Variable | Who Passes It | Which Functions Read It | Is True World Value? | Only Structure? |
|---|---|---|---|:---:|:---:|
| Agent belief mean | `rb.agent_belief_mean` | `sync_robot_belief()` | `predict_agent_prefix()` | ❌ | — |
| Agent belief var | `rb.agent_belief_var` | `sync_robot_belief()` | `estimate_learning_gain()`, bottleneck | ❌ | — |
| Predictor snapshot | `rb._predictor_snapshot` | `sync_robot_belief()` (deepcopy) | `build_surrogate_predictor()` | ❌ | — |
| `passable[H,W]` | `V2EpisodeState.passable` | runner → `score_interventions()` | all counterfactual rollouts | ✅ | structure |
| `belief_cost[H,W]` | `V2EpisodeState.belief_cost` | runner → `score_interventions()` | all counterfactual rollouts | ⚠️ mixed | modified by tutor |
| `meta.segments` | `LatticeV2Meta.segments` | runner → `score_interventions()` | `_build_warn_extra()`, `_find_unlockable_cells()` | ✅ | structure |
| `meta.segments[i].risky_cells` | `SegmentMeta` | via meta | `compute_redundancy()`, `_build_warn_extra()` | ✅ topology | structure |
| `meta.segments[i].trap_cell` | `SegmentMeta` | via meta | **NOT read by teacher decision code** | ✅ | structure |
| `meta.all_door_positions` | `LatticeV2Meta` | via meta | `_find_unlockable_cells()` | ✅ | structure |
| `agent_pos`, `goal` | `V2EpisodeState` | runner | all scoring | ✅ | — |
| `t`, `t_max` | `V2EpisodeState` | runner | deadline scoring | ✅ | — |
| `meta.cell_features[H,W,d]` | `LatticeV2Meta` | **NOT passed** | only runner `observe()` reads it | ✅ world | NOT accessed |
| `gridmap.true_cost[H,W]` | `GridMap` | **NOT passed** | only runner `plan_and_move()` outcome | ✅ world | NOT accessed |
| `gridmap.true_risk[H,W]` | `GridMap` | **NOT passed** | only runner outcome resolution | ✅ world | NOT accessed |
| `meta.world_weights` | `WorldWeights` | **NOT passed** | only generators use it | ✅ world | NOT accessed |
| `PerceptualAccessState` | teacher-owned | runner init | `compute_redundancy()`, `diagnose_bottleneck()` | ❌ | teacher estimate |

**Key finding**: `meta.segments[i].trap_cell` — the exact location of the trap — IS in the SegmentMeta struct that gets passed to `score_interventions()` via `meta`, but **the scoring code never reads `.trap_cell`**. It only reads `.risky_cells` (list of all risky-lane cells), `.risky_entry_gate`, and `.col_start/.col_end`. So teacher knows **which cells are on the risky lane** but doesn't explicitly use the trap location.

### 3.2 `meta.segments` vs `true_cost/true_risk` — Per-Family Analysis

#### **fork_trap**

- **What teacher knows via meta.segments**: segment topology — one segment with `risky_row=1, safe_row=3`, risky_cells = rows 1 positions, safe_cells = rows 3/4/5. `risky_entry_gate = (1, seg_start)`.
- **What teacher does NOT know**: numeric `risk[r,c]` values (e.g. trap_risk=0.45 vs pre-trap=0.05). The predictor snapshot has learned weights that **predict** risk from belief features, but that prediction has both model error and belief noise.
- **With true_cost/true_risk**: Teacher would know that cell (1,4) has risk=0.45 and cell (1,2) has risk=0.03. This means teacher could:
  - **Precisely time the warning** to when agent is about to commit to the trap cell, not just when agent is near the risky lane
  - **Distinguish harmless risky-lane cells from the actual trap**
- **Oracle controller risk**: Medium. With true risk, teacher could compute exact Q(WARN) - Q(WAIT) differences and WARN exactly when marginal benefit is highest. But it can't force the agent to take a specific path — WARN only adds belief_cost bias.
- **Assessment**: For fork_trap, **teacher arguably should know true risk** — in a real pedagogical scenario, a knowledgeable tutor knows which areas are truly dangerous, not just structurally risky.

#### **hazard_belt**

- **What teacher knows via meta.segments**: belt is segment index=1, both-lane risky. knows risky_cells positions.
- **What teacher does NOT know**: specific risk value (belt_risk=0.30 ± noise). Teacher's surrogate predictor estimates this from learned Bayesian weights on noisy feature beliefs.
- **With true_cost/true_risk**: Teacher would know exact belt_risk_val, allowing **precise ITEM_DROP timing** — currently teacher relies on surrogate risk prediction which may under/overestimate belt danger.
- **Oracle controller risk**: Low. ITEM_DROP gives a shield — it's a mitigation action, not a steering action. Knowing exact risk only affects timing, not direction.
- **Assessment**: Knowing true risk **strongly helps** ITEM_DROP timing but doesn't create oracle-planner problems.

#### **deadline_gate**

- **What teacher knows via meta.segments**: shortcut gate position (locked door), long path segments, safe lane hazards.
- **What teacher does NOT know**: exact `safe_risk_val` on long path (0.15–0.25 per cell depending on difficulty), exact shortcut cost.
- **With true_cost/true_risk**: Teacher would know the exact cumulative risk along the long path, enabling **precise UNLOCK timing** to minimize total expected harm.
- **Oracle controller risk**: Very low. UNLOCK opens a door — it's purely a structural intervention. Agent still must decide whether to take the shortcut.
- **Assessment**: This is **the clearest case for world-aware teacher**. Whether to UNLOCK depends on the risk-budget tradeoff of long vs short path — without true risk, teacher relies on surrogate prediction which may misjudge the long-path danger.

#### **DTMB**

- **Special case**: DTMB oracle (`dtmb_helpers.py`) already has a **fully oracle teacher path** (`tutor_mode="dtmb_oracle"`). The oracle reads `meta.commitment_points_by_stage`, `meta.reveal_events_by_stage`, `meta.belt_cells_by_stage`, `meta.all_door_positions` — all topology truth.
- **Does NOT read** `gridmap.true_cost/true_risk` even in oracle mode. Uses distance heuristics + door presence for scoring.
- **Assessment**: DTMB already demonstrates that a partially oracle teacher (topology metadata) works without needing cell-level true_cost/risk.

### 3.3 If Teacher Becomes World-Aware

**Functions that would need signature changes:**

| Function | Current Inputs | New Input Needed | Change Type |
|---|---|---|---|
| `score_interventions()` | `rb, agent_pos, goal, belief_cost, passable, meta` | + `true_risk: np.ndarray` or `hazard_view: WorldHazardView` | Optional param |
| `diagnose_bottleneck()` | `passable, risk_uncertainty_map, min_path_risk` | + `true_min_path_risk: float` | Optional param |
| `_build_warn_extra()` | `meta, warned_segs, agent_pos` | + `true_risk` for risk-weighted warning bias | Low priority |

**Logic that would change:**

1. **Warning timing**: Currently `Q(WARN) = catastrophe_reduction × warn_effect_weight - autonomy_penalty`. With true risk, `catastrophe_reduction` computation could use **actual risk** instead of surrogate-predicted risk, reducing false positives (warn when safe) and false negatives (don't warn when dangerous).

2. **Learning gain estimation**: `estimate_learning_gain()` currently uses `mean(agent_belief_var[prefix])` — purely epistemic. World-aware version could weight this by `true_risk[prefix]` to prioritize learning gain on **dangerous** cells.

3. **Boredom penalty**: `B_wait = avg_prefix_cost / (ε + LG)`. With true cost, the teacher could use `true_cost` instead of predictor-estimated cost, making the boredom metric more stable.

4. **Bottleneck `min_path_risk`**: Currently proxied by `wait_risk` (surrogate prediction). With true risk, this becomes exact.

**Important non-change**: Counterfactual rollout would **still use surrogate predictor** to simulate what the **agent** would do. Adding `true_risk` to the teacher doesn't mean the teacher plans for the agent — it means the teacher evaluates **its own** decision quality using truth.

### 3.4 Recommended World-Knowledge Design Options

#### Option W0: Maintain Status Quo

- **What**: Teacher uses only surrogate belief + predictor snapshot + topology metadata
- **Scientific meaning**: Pure nested-belief pedagogical tutor, must infer danger from same noisy signal as learner (with lag/noise)
- **Implementation cost**: Zero
- **Proposal alignment**: Closest to "robot uses Theory of Mind to track learner belief"
- **Risk**: Warning timing instability, especially early in episode when predictor has few updates
- **Recommendation**: ⚠️ Baseline for comparison only

#### Option W1: Structure-Only Oracle Enhancement

- **What**: Teacher gets access to `meta.segments[i].trap_cell` position (already in struct but unused), plus a `is_belt_segment(i)` flag
- **Scientific meaning**: Teacher knows WHAT is dangerous (topology) but not HOW dangerous (numeric risk)
- **Implementation cost**: Very low — read existing fields, no new data
- **Proposal alignment**: Compatible — teacher has expert structural knowledge
- **Risk**: Still cannot distinguish "risk = 0.15" from "risk = 0.50"
- **Recommendation**: ✅ Good minimum viable step

#### Option W2: Hazard-Aware Timing (RECOMMENDED)

- **What**: Teacher gets a `WorldHazardView`:
  - `hazard_zone[H,W] = true_risk > threshold` (boolean)
  - `max_hazard_in_segment(seg_i): float` (aggregate only)
  - Does NOT expose cell-level `true_risk` to surrogate planner
- **Scientific meaning**: Teacher is a domain expert who knows "areas of elevated risk" but not the exact probability values. Uses this for **intervention timing** only, not for planning.
- **Implementation cost**: Medium — new dataclass, one optional param to `score_interventions()`
- **Proposal alignment**: Compatible with "safety-constrained helper" role. Teacher knows danger exists, helps learner learn to identify it.
- **Contraint**: `predict_agent_prefix()` MUST still use surrogate belief only — the world-aware signal feeds only into the Q-value comparison, not into the counterfactual plan itself
- **Risk**: Moderate. Must enforce clean separation between "teacher knows danger zones for timing" and "teacher replans for agent"
- **Recommendation**: ✅ **Recommended primary approach**

#### Option W3: Full Oracle Teacher

- **What**: Teacher has access to `gridmap.true_cost[H,W]` and `gridmap.true_risk[H,W]`
- **Scientific meaning**: Omniscient teacher. Knows everything about the world.
- **Implementation cost**: Low (just pass arrays)
- **Proposal alignment**: ⛔ **Violates** proposal's pedagogical premise — teacher should model learner, not be a god
- **Risk**: High oracle-controller risk. If teacher can compute exact optimal path using true values, it degenerates into a planner, not a teacher.
- **Recommendation**: ❌ Only useful as experiment condition "oracle baseline", NOT as default design

---

## 4. Audit B — Learner-State Modeling Upgrade

### 4.1 Current exact/noisy/stale Implementation

**Init path** (runner):
```
runner.reset() →
  init_robot_belief(
    fb.mean, fb.var,          # agent's FeatureBeliefMap arrays
    latent_predictor=lp,      # agent's LatentCostRiskHead
    copy_mode=belief_copy_mode,
    planner_weights=state.planner_weights,
    ...
  )
```
[lattice_v2_runner.py L364-371](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/lattice_v2_runner.py#L364-L371)

**Sync path** (every step, in `observe()`):
```
sync_robot_belief(
    s.robot_belief,
    s.feature_belief.mean,    # agent's current belief mean
    s.feature_belief.var,     # agent's current belief var
    latent_predictor=s.latent_predictor,  # agent's current predictor
    t=s.t, rng=s.rng,
)
```
[lattice_v2_runner.py L447-451](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/lattice_v2_runner.py#L447-L451)

**What gets copied:**

| Field | How Copied | Timing |
|---|---|---|
| `agent_belief_mean` | `.copy()` (+ noise in noisy mode) | every sync |
| `agent_belief_var` | `.copy()` | every sync |
| `_predictor_snapshot` | `deepcopy(latent_predictor)` | every sync |
| `agent_search_budget` | set once at init (with mismatch) | init only |
| `agent_planner_weights` | set once at init | init only |
| `risk_weight_mismatch` | stored but NOT applied to canonical weight | init only |

### 4.2 What Is Copied vs Inferred vs Teacher-Owned

| RobotBelief Field | Category | Source |
|---|---|---|
| `agent_belief_mean` | **agent belief snapshot** | copied from `FeatureBeliefMap.mean` |
| `agent_belief_var` | **agent belief snapshot** | copied from `FeatureBeliefMap.var` |
| `_predictor_snapshot` | **agent predictor snapshot** | deepcopy of `LatentCostRiskHead` |
| `agent_planner_weights` | **agent competence estimate** | configured at init (PlannerWeights) |
| `agent_search_budget` | **agent competence estimate** | configured at init (with mismatch) |
| `copy_mode` | **teacher config** | set by experiment condition |
| `belief_noise_std` | **teacher config** | set by experiment condition |
| `risk_weight_mismatch` | **competence mismatch** | stored, not currently used |
| `PerceptualAccessState` | **teacher-owned estimate** | maintained independently by teacher |

**Key observation**: There is NO inferred data. Everything is either direct-copied or configured.

### 4.3 Gap to Belief-Over-Belief / Sequential Inverse Planning

**What exists (`robot_belief_over_agent.py`):**
- `RobotBeliefOverAgent` class — shell/prototype, **not connected to mainline**
- Has `update_from_action()` that does Bayesian theta_posterior update via ActionPredictor likelihood
- Has `predict_after_observation()` — stub (returns current mean)
- `RobotBeliefState` tracks: `mean_m` (τ, ν, γ_gen, γ_spec, κ), `theta_posterior`, `entropy`

**What exists (`action_predictor.py`):**
- `ActionPredictor` class — wraps `compute_choice_probs()` into P(a|s,b) distribution
- Has `score()` returning log P(a_obs | s_world, b_A) — **the exact inverse-planning likelihood function**
- But **never called from mainline** — only from `RobotBeliefOverAgent.update_from_action()`

**Specifically missing for full trajectory-based inference:**

| Missing Piece | Description | Difficulty |
|---|---|---|
| **Action observation** | Runner doesn't record agent's chosen action/path as a teacher-observable signal | Easy (add to step state) |
| **State transition model** | P(b'|b, a, o) — how agent belief changes after action+observation | Medium (wrap FeatureBeliefMap update logic) |
| **Trajectory likelihood** | Π_t P(a_t | s_t, b_t^A) over episode | Medium (compose ActionPredictor calls) |
| **Posterior over planner weights** | P(λ | trajectory) — infer which weights the agent is using from its behavior | Hard (new model) |
| **Warning response model** | How agent adjusts behavior after warning — trust parameter | Hard (new model) |
| **Multi-episode profile** | Persistent estimate of agent competence/trust across episodes | Medium (need persistence layer) |
| **Integration into `score_interventions()`** | Replace `sync_robot_belief` with posterior-based belief update | Medium but high coupling risk |

### 4.4 Trajectory-Based Inference Design Options

#### Option T0: Maintain Point Copy

- **What**: Keep current exact/noisy/stale modes
- **New modules**: None
- **Pro**: Stable, tested, simple
- **Con**: Teacher's belief about agent may diverge from reality (especially after warnings)
- **Recommendation**: Current default, fine for now

#### Option T1: Point Copy + Local Trajectory Correction

- **What**: After point-copying belief, apply a correction based on agent's last N actions:
  - If agent avoided warned cells → increase teacher's estimate of agent trust
  - If agent entered warned cells → decrease trust estimate
- **New modules**: `trajectory_correction.py` (~100 lines), add 1 field to RobotBelief
- **State needed**: `last_N_actions: list[(pos, t)]` — already partially available from `feature_belief.visit_count`
- **Tests needed**: Unit test that correction adjusts trust → changes Q(WARN) magnitude
- **Compatibility risk**: Very low — additive correction, doesn't change data flow
- **Recommendation**: ✅ **Good incremental step**

#### Option T2: Within-Episode Posterior over Learner State

- **What**: Maintain `P(θ, λ | trajectory_0:t)` updated each step via `ActionPredictor.score()`
- **New modules**: Extend `RobotBeliefOverAgent` to connect to mainline. New `TeacherBeliefUpdater` class.
- **State needed**:
  - Agent's chosen action per step (not currently recorded as teacher-observable)
  - `BranchAttributes` for each step (need to extract from planner output)
  - Integration with `plan_from_belief()` output
- **Tests needed**: Posterior convergence test, likelihood calibration test
- **Compatibility risk**: Medium — needs planner to export candidate branches, needs runner to record actions
- **Recommendation**: ⚠️ Full architecture change, defer to after W2+T1

#### Option T3: Cross-Episode Persistent Profile + Within-Episode Tracking

- **What**: `profile_manager.py` + `profile_state.py` (already exist!) + T2 within-episode
- **New modules**: Bridge between existing profile system and `score_interventions()`
- **State needed**: Everything in T2 + cross-episode persistence
- **Tests needed**: Transfer compatibility, profile bootstrap, profile reset semantics
- **Compatibility risk**: High — touches transfer/snapshot/session API boundaries
- **Recommendation**: ❌ Too much coupling risk for now

---

## 5. Joint Design Audit — Must These Two Changes Be Coupled?

### 5.1 Interface Coupling Map

| Module | World-Aware Impact | Trajectory-Inference Impact | Cross-Impact |
|---|---|---|---|
| `score_interventions()` | +1 optional param (`hazard_view`) | None (reads from `rb`) | Independent |
| `diagnose_bottleneck()` | +1 optional param (`true_min_path_risk`) | None | Independent |
| `RobotBelief` | No change | +trust/trajectory fields | Independent |
| `sync_robot_belief()` | No change | Major refactor or replacement | Independent |
| `agent_predictor.py` | Surrogate rollout unchanged | Potential action-distribution upgrade | Weakly coupled |
| `intervention_policy.py` | +hazard timing signal | +trust-weighted Q adjustments | Weakly coupled |
| `V2EpisodeState` | +`hazard_view` field | +`last_action` field | Independent |
| `runner.step()` | pass `hazard_view` to `apply_tutor()` | record action for teacher | Independent |
| `step_logger` | log hazard timing | log trajectory features | Independent |
| `transfer_eval.py` | No impact | Profile persistence coupling | T-only |
| `SlowFast / StructuredBasis` | No impact | None | None |

### 5.2 Recommended Sequencing

```
Step 0: Read-only validation (THIS REPORT)
    → No code changes. Confirm design with user.

Step 1 (W1): Structure-only oracle enhancement
    → Read meta.segments[i].trap_cell in score_interventions
    → Add is_belt_segment helper
    → ~30 lines changed, zero new files
    → Tests: existing regression suite

Step 2 (W2): WorldHazardView
    → New dataclass: WorldHazardView (hazard zones + segment max risk)
    → Add optional param to score_interventions()
    → Add hazard_zone_risk to boredom/bottleneck scoring
    → ~150 lines new, 1 new file
    → Tests: unit test for hazard view, behavior-equivalence smoke

Step 3 (T1): Local trajectory correction
    → Add trust_estimate to RobotBelief
    → Adjust Q(WARN) by observed warning-compliance signal
    → ~100 lines new
    → Tests: trust adjustment unit test, Q(WARN) shift test

Step 4 (T2): Optional — full within-episode posterior
    → Connect RobotBeliefOverAgent to mainline
    → Requires action recording in runner
    → ~300 lines changed
    → Tests: posterior convergence, likelihood calibration
```

### 5.3 Hard Blockers / Likely Breakages

| Risk | Severity | Affected By |
|---|---|---|
| **predictor shape**: StructuredBasis/SlowFast have different `.predict_*()` signatures | Low | Neither W nor T |
| **transfer snapshot API**: `predictor_protocol.snapshot_predictor()` | Low | T3 only |
| **step logger theta extraction**: expects specific observer fields | Medium | T2/T3 |
| **DTMB oracle bypass**: `tutor_mode="dtmb_oracle"` skips `score_interventions()` entirely | None | Already independent |
| **GTET factor adapter**: modifies `InterventionConfig` weights | Low | W2 must be compatible |
| **`belief_cost` mutation by WARN**: `s.belief_cost[gate] = 100.0` | Low | W2 independent |
| **`passable` mutation by UNLOCK**: `s.passable[door] = True` | Low | W2 independent |
| **regression in `test_robot_belief.py`**: expects specific field names | Medium | T1 (if adding trust field) |

---

## 6. Minimal-Diff Refactor Sketch

### Step 1 (W1): Structure-Only Oracle
```
MODIFY: src/teachers/intervention_policy.py
  - In _build_warn_extra(): check seg.trap_cell proximity for timing bonus
  - No new params. meta already has all needed data.

MODIFY: tests/test_batch_d_cleanup.py (or new test file)
  - Test that trap_cell proximity triggers higher warn score
```

### Step 2 (W2): WorldHazardView

```
NEW: src/teachers/world_hazard_view.py (~80 lines)
  @dataclass
  class WorldHazardView:
      hazard_zone: np.ndarray       # bool[H,W]: true_risk > threshold
      segment_max_risk: dict[int, float]  # seg_index → max true_risk
      threshold: float = 0.15

  def build_hazard_view(gridmap, meta, threshold=0.15) → WorldHazardView

MODIFY: src/teachers/intervention_policy.py
  - score_interventions() gains optional hazard_view param
  - Use segment_max_risk to weight Q(WARN) timing
  - Use hazard_zone in boredom penalty (weight by true danger)

MODIFY: src/envs/lattice_v2_runner.py
  - In reset(): build hazard_view from gridmap
  - In _apply_tutor_dispatch(): pass hazard_view to score_interventions()

NO CHANGE: agent_predictor.py, robot_belief.py, belief_planning.py
  → Surrogate rollout stays purely belief-based
```

### Step 3 (T1): Trajectory Correction

```
MODIFY: src/teachers/robot_belief.py
  - Add trust_estimate: float = 0.5 to RobotBelief
  - Add update_trust_from_action(rb, last_action, warned_cells) function

MODIFY: src/teachers/intervention_policy.py
  - Multiply Q(WARN) by trust factor from rb.trust_estimate

MODIFY: src/envs/lattice_v2_runner.py
  - In plan_and_move(): record last_action for teacher
  - In observe(): call update_trust_from_action()
```

### NOT recommended to change:
- `agent_predictor.py` — counterfactual rollout should stay surrogate-based
- `belief_planning.py` — agent-side, not teacher
- `planner_astar.py` — agent-side, not teacher
- `cost_risk_model.py` — agent-side
- `scenario_families.py` — generation, not teacher
- `bounded_agent.py` — archival, frozen

---

## 7. Test & Validation Plan

### P0: Unit Tests

- `test_world_hazard_view.py`: build from known gridmap, verify hazard zones
- `test_trust_correction.py`: verify trust goes up/down with compliance/defiance

### P1: Behavior-Equivalence Smoke

- Run `score_interventions()` **with and without** `hazard_view` on same episode state
- Verify: actions change only in timing, not in fundamental logic
- Assert: agent_predictor counterfactual plans are IDENTICAL (not world-aware)

### P2: World-Aware vs Current Baseline Comparison

- For each family (fork_trap, hazard_belt, deadline_gate, DTMB):
  - Run 100 episodes with W0 (current) vs W2 (hazard-aware)
  - Compare: warning timing distribution, intervention count, survival rate, learning gain
  - **Explicit test**: verify teacher does NOT degenerate into oracle planner by checking that agent's path is still generated from surrogate, not from true values

### P3: Trajectory-Inference Sanity Tests

- T1 only: verify trust estimate converges (agent complies → trust ↑, agent ignores → trust ↓)
- Verify Q(WARN) modulated by trust: low trust → WARN suppressed

### P4: Combined Regression

- Run full test suite: `test_batch_a_hotfix`, `test_batch_d_cleanup`, `test_robot_belief`, `test_v2_latent_path`
- Verify zero regressions when hazard_view is None (backward compat)

**Anti-oracle verification**:
- Compute correlation between teacher's chosen action and action that an omniscient planner would choose
- W2 should have HIGHER correlation for timing but NOT for path selection
- If correlation with path selection is too high, teacher is leaking oracle info

---

## 8. Final Recommendation

### World-Knowledge Design: **Option W2 (WorldHazardView)**

Teacher knows danger zones (thresholded boolean + per-segment max risk) for timing decisions only. Surrogate rollout stays belief-based. This is the right pedagogical balance: a competent teacher knows WHERE danger is, but doesn't know what the student thinks or how the student will react.

### Trajectory-Inference Design: **Option T1 (Local Trajectory Correction)**

Point copy + trust estimate from observed warning compliance. Minimal new machinery, high marginal value (stops teacher from spamming warnings at agent who ignores them).

### Coupling: **No — implement independently**

W2 and T1 don't share interface surfaces. W2 touches `score_interventions()` input params. T1 touches `RobotBelief` fields and Q-score modulation. They can be tested and verified independently.

### Suggested First Implementation Step

**W1 (Structure-Only Oracle Enhancement)** — read `trap_cell` proximity in `_build_warn_extra()`. ~30 lines, zero new files, zero risk. This gives us data on whether structural-only oracle info already improves warning timing before investing in the full WorldHazardView.
