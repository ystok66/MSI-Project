# Batch D Investigation Report: P2 Maintenance Audit

> Audit date: 2026-04-08
> Scope: `src/`, `tests/`, `archive/`, `scripts/`
> Method: Static call-chain analysis via grep + file inspection
> Status: **Investigation only — no code changes**

---

## Part A. Maintenance Hot-Zone Map

| Hot Zone | Complexity | Main-line Dependency | Files |
|----------|-----------|---------------------|-------|
| **Planner weight naming** | 🔴 High | ✅ Core path | `planner_astar.py`, `belief_planning.py`, `robot_belief.py`, `agent_predictor.py` |
| **scenario_families** | 🟠 Medium | ✅ Core | `scenario_families.py` (2700 lines, 12 generators) |
| **V0/V1 planner paths** | 🟡 Low | ⚠️ Runner legacy fallback | `planner_astar.py`, `bounded_agent.py` |
| **Observation interfaces** | 🟢 Low | ✅ Dispatch-only | `observation_model.py` |
| **Warning legacy state** | 🟡 Medium | ⚠️ Legacy WARN path | `warning_update.py`, runner state |
| **Archival modules** | 🟢 None | ❌ No callers | `pedagogical_grid.py`, `bounded_agent.py`, archive/ |

---

## Part B. D1–D2 Detailed Findings

---

### D1.1: Parameter Naming Census

#### `lambda_risk` / `lambda_r` / `agent_risk_weight`

All three names refer to the **same concept**: risk penalty weight in cell cost objective.

| Name | Default | File | Path | Main-line? |
|------|---------|------|------|-----------|
| `lambda_risk` | **3.0** | `belief_planning.py:91` | `plan_from_belief()` | ✅ Canonical |
| `lambda_risk` | **5.0** | `planner_astar.py:269` | `cell_cost_v2()` | ⚠️ Legacy V2 fallback |
| `lambda_risk` | **5.0** | `planner_astar.py:397` | `plan_next_action_v2()` | ⚠️ Legacy fallback |
| `lambda_risk` | **5.0** | `planner_astar.py:459` | `plan_with_alternatives_v2()` | ✅ Called via `plan_from_belief` |
| `lambda_risk` | **3.0** | `planner_astar.py:199` | `bounded_astar()` V0 | ❌ Archive only |
| `lambda_r` | **5.0** | `planner_astar.py:306` | `cell_cost_v2_latent()` | ✅ Core cell cost |
| `agent_risk_weight` | **3.0** | `robot_belief.py:37` | `RobotBelief` dataclass | ✅ Tutor surrogate |
| `lambda_risk` | **3.0** | `cause_scoring.py:90,203,236` | Counterfactual scoring | ✅ Tutor |
| `lambda_risk` | 1.5/0.8 | micro_bayes_shadow*.py | Shadow tutors | ✅ ICTv4 branches (archival) |

**Key finding:** The runner's canonical path (`plan_from_belief` at L483) does NOT explicitly pass `lambda_risk`. It falls through to `plan_from_belief`'s default **3.0**, which then passes to `plan_with_alternatives_v2`'s internal call, overriding its own default of **5.0**. Meanwhile `RobotBelief` also defaults to **3.0**. So the **de facto canonical value is 3.0** — the V2 function defaults of 5.0 are dead defaults that never activate on main-line.

#### `lambda_uncertainty` / `lambda_uc` / `lambda_ur` / `agent_uncertainty_weight`

| Name | Default | File | Path | Main-line? |
|------|---------|------|------|-----------|
| `lambda_uncertainty` | **0.5** | `belief_planning.py:92` | `plan_from_belief()` | ✅ Canonical (falls through) |
| `lambda_uncertainty` | **0.1** | `planner_astar.py:270,398,460` | V2 functions | ⚠️ Dead default |
| `lambda_uc` | **0.1** | `belief_planning.py:94` | `plan_from_belief()` | ✅ Canonical |
| `lambda_ur` | **0.1** | `belief_planning.py:95` | `plan_from_belief()` | ✅ Canonical |
| `agent_uncertainty_weight` | **0.5** | `robot_belief.py:38` | `RobotBelief` | ✅ Tutor surrogate |
| `agent_lambda_uc` | **0.1** | `robot_belief.py:40` | `RobotBelief` | ✅ Tutor surrogate |

**Key finding:** Runner does NOT pass `lambda_uncertainty`/`lambda_uc`/`lambda_ur` to `plan_from_belief`. De facto canonical: `lambda_risk=3.0`, `lambda_uncertainty=0.5`, `lambda_uc=0.1`, `lambda_ur=0.1`. These match `RobotBelief` defaults, so tutor and agent are actually consistent *by accident*. But the runner has zero explicit control.

---

### D1.2: Planner Weight Source Tracing

| Path | Enters Main Experiment? | Weight Source | Defaults Active? | Suggest Unify? |
|------|------------------------|---------------|-----------------|----------------|
| runner → `plan_from_belief()` | ✅ Yes | Function defaults | ✅ `λ_r=3.0, λ_unc=0.5, λ_uc=0.1` | ✅ |
| runner → `plan_next_action_v2()` (legacy) | ⚠️ Only `belief_planning_mode=False` | Function defaults | ✅ `λ_r=5.0, λ_unc=0.1` | ✅ |
| tutor → `predict_agent_prefix` → `plan_from_belief()` | ✅ Yes | `RobotBelief` fields | ✅ `λ_r=3.0, λ_unc=0.5` | ✅ |
| `cause_scoring` → `bounded_astar` | ✅ Yes | Hardcoded `3.0` | ✅ | 🟡 Low priority |
| transfer_eval → snapshot/restore | ✅ Yes | Snapshot has predictor only | N/A (no weights stored) | ⚠️ |
| `utilities.py` → `bounded_astar` | ⚠️ | `lambda_risk=3.0` | ✅ | 🟡 |

**Conclusion:** The canonical weight entry should be a single PlannerConfig stored on V2EpisodeState. Currently weights are correct by coincidence (defaults happen to align), but this is fragile.

---

### D1.3: Latent Path Redundant Parameters

| Parameter | Used in Latent Path? | Used in Legacy Path? | Status |
|-----------|---------------------|---------------------|--------|
| `belief_cost` (runner state) | ❌ `cell_cost_v2_latent` ignores it; uses `latent_predictor.predict_cost(x)` | ✅ `cell_cost_v2` reads it | **compatibility-only** on latent |
| `risk_model` (in planners) | ❌ `cell_cost_v2_latent` uses `latent_predictor.predict_risk(x)` | ✅ `cell_cost_v2` uses it | **compatibility-only** on latent |
| `lambda_uncertainty` single | ❌ Latent uses `lambda_uc`/`lambda_ur` split | ✅ Legacy uses it | **compatibility-only** |
| `belief_cost` 100.0 for walls | ❌ Latent uses `passable[r,c]=False` check | ✅ Legacy fallback | **dead value** on latent |

**However**, `belief_cost` is **actively mutated** by the runner for warning effects:
- L628: `s.belief_cost[gate_cell] = 100.0` (close gate)
- L725: `s.belief_cost[r,c] = 1.0` (unlock gate)
- L1249: `s.belief_cost[r,c] += 5.0` (DTMB warning)

These mutations pass through to `bounded_astar` calls in `cause_scoring`, so removing `belief_cost` entirely would break those. **Verdict: keep but document as "legacy-path only; latent path ignores per-cell values".**

---

### D2.1: `scenario_families.py` Structure Analysis

**Current state:** 2700 lines, 12 generator functions, 1 registry dict.

| Function | Lines | Stability | Shares Helpers? | Split Candidate? |
|----------|-------|-----------|----------------|-----------------|
| `generate_scenario` (dispatch) | 19 | ✅ Stable | N/A | Keep as registry |
| `generate_baseline_v2` | 16 | ✅ Stable | No | ✅ Core |
| `generate_fork_trap` | 247 | ✅ Stable (Batch B fixed) | No | ✅ Core |
| `generate_hazard_belt` | 238 | ✅ Stable | No | ✅ Core |
| `generate_deadline_gate` | 260 | ✅ Stable | No | ✅ Core |
| `generate_funnel_trap` | 386 | ✅ Stable | No | ✅ Advanced |
| `generate_elcb` | 214 | ✅ Stable | Shares w/ elcb_po | ✅ Advanced |
| `generate_elcb_po` | 210 | ✅ Stable | Shares w/ elcb | ✅ Advanced |
| `generate_delayed_corridor` | 185 | ✅ Stable (Batch B cleaned) | No | ✅ Advanced |
| `generate_distractor_cue` | 185 | ✅ Stable (Batch B cleaned) | No | ✅ Advanced |
| `generate_temptation_corridor` | 188 | 🟡 Newer | No | ✅ Phase 2 |
| `generate_joint_conflict_corridor` | 207 | 🟡 Newer | No | ✅ Phase 2 |

**Recommended split (NOT for now, just design):**
- `scenario_families.py` → keep as registry (`generate_scenario` + `SCENARIO_REGISTRY` + shared imports)
- `scenario_core.py` → baseline, fork_trap, hazard_belt, deadline_gate
- `scenario_advanced.py` → funnel_trap, elcb, elcb_po, delayed_corridor, distractor_cue
- `scenario_phase2.py` → temptation_corridor, joint_conflict
- DTMB and GTET already have separate modules (`dtmb_helpers.py`, `gtet_*.py`)

**Minimum viable split:** The registry API (`SCENARIO_REGISTRY` dict + `generate_scenario()` dispatcher) must stay in a central location. Individual generators are self-contained and have no cross-dependencies.

---

### D2.2: V0/V1 Planner Path Audit

| Function | Definition | Main-Line Callers | Test Callers | Archive Callers | Recommendation |
|----------|-----------|-------------------|-------------|-----------------|----------------|
| `plan_next_action()` V0 | `planner_astar.py:233` | `bounded_agent.py:250` | `test_planner.py:67,85` | `particle_teacher.py` | **deprecated** (bounded_agent is archival) |
| `bounded_astar()` V0 | `planner_astar.py:192` | `utilities.py:57`, `cause_scoring.py:200,232,284`, `map_families.py:573,578` | `test_planner.py`, `test_planner_v2.py:133` | 12+ archive refs | **keep** — `cause_scoring` is active main-line |
| `plan_next_action_v2()` V2-legacy | `planner_astar.py:390` | `lattice_v2_runner.py:516` (fallback) | `test_planner_v2.py`, `test_prefix_prediction.py` | 8 archive scripts | **deprecated** (runner fallback only) |
| `cell_cost_v2()` | `planner_astar.py:263` | inside `plan_next_action_v2` fallback | None directly | None | **deprecated** (only activates when `latent_predictor is None`) |
| `cell_cost_v2_latent()` | `planner_astar.py:299` | `plan_with_alternatives_v2`, `branch_reranker.py` | via integration | None | ✅ **active** — core cell cost |
| `plan_with_alternatives_v2()` | `planner_astar.py:459` | `belief_planning.py:114`, `agent_predictor.py:83` | via integration | None | ✅ **active** — canonical |
| `BoundedRationalAgent` | `bounded_agent.py` | `pedagogical_grid.py` only | `test_teacher.py` (broken) | `oracle_teacher.py`, `oracle_cause_teacher.py` | ✅ **archive** — zero main-line |
| `pedagogical_grid.py` | `envs/` | Only `__init__.py` re-export | None working | None | ✅ **archive** candidate |

**Key conclusions:**
1. `bounded_astar()` CANNOT be removed — actively used by `cause_scoring.py` (main-line tutor)
2. `plan_next_action()` V0 + `BoundedRationalAgent` — archival, zero main-line callers
3. `plan_next_action_v2()` — runner legacy fallback; should be deprecated but not removed yet
4. `pedagogical_grid.py` — V0 Gym env wrapper; purely archival

---

### D2.3: Observation Model Interfaces

| Function | File | Canonical? | Callers |
|----------|------|-----------|---------|
| `generate_observations()` V0 | `observation_model.py:28` | ❌ **DEPRECATED** in docstring | Zero callers |
| `observe_features()` | `observation_model.py:100` | ✅ Active | `runner:429`, `observe_features_patch:180` (legacy delegate) |
| `observe_features_patch()` | `observation_model.py:156` | ✅ Active | `runner:425` (when `patch_radius > 1`) |

**Structure:** `observe_features_patch` with `patch_radius=1` delegates to `observe_features()` for RNG compatibility. This is clean; no change needed.

**Recommendation:**
- `generate_observations()` V0 (L28-87): **removable** — zero callers, already deprecated
- `Observation` dataclass (L18-25): **removable** — only used by `generate_observations`
- `observe_features()` + `observe_features_patch()`: **keep** — active canonical

---

### D2.4: Warning Legacy State Census

| Field | Defined | Written By | Read By | RSA Path Uses? | Legacy Path Uses? | Status |
|-------|---------|-----------|---------|---------------|-------------------|--------|
| `warned_segments: set` | Runner L127 | `_apply_warn` L928 | Intervention dispatch L653,712,815 | ✅ Yes | ✅ Yes | **active** — tracks which segments warned |
| `warned_lane_bias: dict` | Runner L128 | `update_warning_belief_integrate` | `_build_warned_cell_extra` L202-213 | ❌ **Never populated** (RSA uses delta.planner_cell_penalties → warned_cell_extra directly) | ✅ Yes | **compatibility-only** — RSA path bypasses |
| `warned_cell_extra: dict` | Runner L129 | Both paths write to it | Planner reads as extra cost | ✅ Yes (via `apply_planner_adapter`) | ✅ Yes (via `_build_warned_cell_extra`) | **active** — convergence point |
| `lambda_lane_warn: float` | Runner L130 | Never modified | `_build_warned_cell_extra` L208 | ❌ | ✅ Only legacy | **compatibility-only** |
| `rsa_channel` | Runner L167 | `reset()` L321 | `_apply_rsa_warn_s1` L829 | ✅ | N/A | **active** |
| `rsa_belief_state` | Runner L168 | RSA update | RSA diagnostic | ✅ | N/A | **active** |
| `rsa_warn_diagnostics` | Runner L169 | RSA update | Logging only | ✅ Logging | N/A | **logging-only** |

**Conclusion:**
- **dead/legacy:** `warned_lane_bias`, `lambda_lane_warn` — not populated on RSA path (the canonical warning path)
- **active:** `warned_segments`, `warned_cell_extra`, `rsa_channel`, `rsa_belief_state`
- **logging-only:** `rsa_warn_diagnostics`

However, `warned_lane_bias` IS still read by legacy WARN dispatch in `_apply_legacy_warn` (L888-905). Since some families/configurations may still use the legacy WARN path (non-RSA), removing it would require verifying ALL families use RSA warnings exclusively. **Recommend: mark deprecated, do NOT remove yet.**

---

## Part C. Pre-Cleanup Test Checklist

### API Signature Regression
- [ ] `plan_from_belief()` signature unchanged (10 kwargs)
- [ ] `plan_with_alternatives_v2()` signature unchanged
- [ ] `compute_episode_summary()` backward compatible (new `boredom_trace` is Optional)
- [ ] `init_robot_belief()` backward compatible

### Canonical Path Behavior Regression
- [ ] Runner with `belief_planning_mode=True` produces identical trajectories before/after
- [ ] `cause_scoring` via `bounded_astar` produces identical scores
- [ ] Transfer eval snapshot/restore still works

### No-Caller Verification
- [ ] `BoundedRationalAgent` — only imported by `pedagogical_grid.py` and `__init__.py`
- [ ] `generate_observations()` V0 — zero callers confirmed
- [ ] `plan_next_action()` V0 — only `bounded_agent.py` (archival)

### File Move / Import Stability
- [ ] If scenario families split: `SCENARIO_REGISTRY` import resolves from all tests
- [ ] If observation V0 removed: no test import breaks

### Scenario Registry Integrity
- [ ] All 14 families still generate valid maps (existing `test_scenario_families.py`)
- [ ] Contract validation passes (existing tests)

---

## Part D. Archive/Delete/Deprecate Recommendations

### Immediately Deletable (zero callers, zero risk)

| Object | File | Reason |
|--------|------|--------|
| `generate_observations()` V0 + `Observation` dataclass | `observation_model.py:18-87` | Deprecated in docstring, zero callers |

### Mark Deprecated (still has callers, but shouldn't grow)

| Object | File | Remaining Callers |
|--------|------|-------------------|
| `plan_next_action()` V0 | `planner_astar.py:233` | `bounded_agent.py` (archival) |
| `plan_next_action_v2()` | `planner_astar.py:390` | Runner legacy fallback L516, tests |
| `cell_cost_v2()` (non-latent) | `planner_astar.py:263` | Inside `plan_next_action_v2` |
| `warned_lane_bias` + `lambda_lane_warn` | Runner state L128,130 | Legacy WARN path |
| `BoundedRationalAgent` | `bounded_agent.py` | `pedagogical_grid.py` (archival env) |
| `pedagogical_grid.py` | `envs/` | `__init__.py` re-export only |

### Handle in PlannerConfig / File Split (future)

| Object | Action | Prerequisite |
|--------|--------|-------------|
| `lambda_risk` / `lambda_r` / `agent_risk_weight` unification | Introduce `PlannerWeights` dataclass | All tests green |
| `scenario_families.py` split into 3-4 files | Registry stays central, generators move | Import stability test |
| Runner planner weight storage | Add `PlannerWeights` to `V2EpisodeState` | PlannerWeights defined |
| `belief_cost` documentation as legacy-only in latent path | Docstring update only | None |

---

## Hard Questions — Direct Answers

### 1. What should the canonical planner weight entry be?

**A single `PlannerWeights` dataclass stored on `V2EpisodeState`, passed to both `plan_from_belief()` and `predict_agent_prefix()`.**  Currently both rely on function defaults (3.0/0.5/0.1/0.1), which happen to match. But this is fragile.

### 2. Which names can unify vs need compat layer?

- `lambda_risk` → keep as canonical name (matches most callsites)
- `lambda_r` → rename to `lambda_risk` in `cell_cost_v2_latent()` only (1 file, internal)
- `agent_risk_weight` → keep in `RobotBelief` (represents tutor's *estimate* of agent's weight — semantically different)
- `lambda_uncertainty` → deprecate; always use `(lambda_uc, lambda_ur)` pair

### 3. Which latent path parameters are completely redundant?

- `belief_cost` per-cell values: **ignored** by `cell_cost_v2_latent()`, but still mutated by warning system for `cause_scoring` (which uses V0 `bounded_astar`). **Cannot remove array, but latent path doesn't read per-cell values.**
- `risk_model` parameter in V2 planners: **unused** when `latent_predictor is not None`
- `lambda_uncertainty` single: **unused** in latent path (replaced by `lambda_uc`/`lambda_ur`)

### 4. Is `scenario_families.py` worth splitting now?

**Yes, but the minimum safe split is well-defined:** generators are self-contained (no cross-dependencies). Registry + dispatch stays central. Cost: ~30 min work + import stability test. **Recommend: defer to after PlannerConfig, since splitting doesn't fix any bugs and the file is stable.**

### 5. Which V0/V1 planner paths still have real main-line callers?

- `bounded_astar()`: **YES** — `cause_scoring.py` (main-line tutor), `utilities.py`, `map_families.py`
- `plan_next_action()` V0: **NO** — only `bounded_agent.py` (archival)
- `plan_next_action_v2()`: **YES** — runner legacy fallback at L516 (when `belief_planning_mode=False`)

### 6. How much real use does `observation_model.py` V0 have?

**Zero.** `generate_observations()` (V0) has zero callers. The V2 functions (`observe_features`, `observe_features_patch`) are active and should stay.

### 7. Which warning legacy state can be removed?

- `warned_lane_bias`: **mark deprecated** — not populated on RSA path, but still read by legacy WARN dispatch
- `lambda_lane_warn`: **mark deprecated** — same
- `rsa_warn_diagnostics`: **keep** — logging only but harmless
- `warned_segments`: **keep** — actively used by all WARN paths
- `warned_cell_extra`: **keep** — convergence point for both paths

### 8. Summary classification

| Category | Objects |
|----------|---------|
| **Should unify** | `lambda_risk`/`lambda_r`/`agent_risk_weight` naming + storage |
| **Should deprecate** | `plan_next_action()` V0, `plan_next_action_v2()`, `cell_cost_v2()`, `warned_lane_bias`, `BoundedRationalAgent`, `pedagogical_grid.py` |
| **Can archive/delete** | `generate_observations()` V0 + `Observation` dataclass |
| **Keep + document** | `bounded_astar()`, `belief_cost` array, `observe_features*()`, `warned_segments`, `warned_cell_extra` |
