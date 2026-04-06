# Gap Checklist — Prioritized Cleanup Tasks

## Priority 1: Experiment Correctness (do first)

### 1.1 Enforce `allowed_interventions` at policy level
- [ ] `score_interventions()` should accept `allowed_actions: set` parameter
- [ ] Filter scored actions to only `allowed_actions` before selecting best
- [ ] Runner should pipe `allowed_interventions` from config to policy
- [ ] Add test: robot_belief tutor under `warning_only` condition never selects UNLOCK

**Why highest priority**: matrix experiment correctness depends on this. Without enforcement, teacher conditions are advisory, not binding. This directly affects paper results.

### 1.2 Mark V0 files as DEPRECATED
- [ ] `src/agents/bounded_agent.py` — add `# DEPRECATED: V0 agent, not used by V2 runner`
- [ ] `src/agents/observation_model.py` — V0 obs model, not in V2 path
- [ ] `src/metrics/eval_v1.py` — V0 metrics, replaced by `phase9_metrics.py`
- [ ] `src/envs/pedagogical_grid.py` — V0 env, replaced by `lattice_v2_env.py`

**Why**: prevents confusion, costs nothing.

### 1.3 Consolidate online_metrics.py into phase9_metrics.py
- [ ] `epistemic_gain()` → already reimplemented as `_information_gain()`
- [ ] `frustration_score()` → already reimplemented as `_frustration_proxy()`
- [ ] Mark `online_metrics.py` as deprecated or delete

**Why**: two implementations of same concept, no shared code.

---

## Priority 2: Naming and Consistency

### 2.1 Unify naming across layers

| Current | Suggested | Where |
|---------|-----------|-------|
| `closures` | `unlock_count` | runner, get_metrics |
| `warnings_sent` | `warn_count` | runner, get_metrics |
| `risky_entered` → `risky` | `risky_entered` everywhere | get_metrics output |
| `cue_seen` → `cue_cells_seen` | `cue_cells_seen` | get_metrics output |

### 2.2 Standardize intervention action naming
- Runner uses `tutor_mode="time_aware"` for UNLOCK
- Policy uses `InterventionType.UNLOCK`
- Config uses `unlock_only`
- These should map to each other explicitly

---

## Priority 3: Documentation Gaps

### 3.1 Missing docstrings / architecture docs
- [ ] `lattice_v2_runner.py` — 528 lines, no architecture overview
- [ ] `V2EpisodeState` — 30+ fields, no field-level documentation
- [ ] `planner_astar.py` — two parallel cost functions, unclear which is canonical

### 3.2 Missing README sections
- [ ] How to run experiments (currently undocumented)
- [ ] How the config files relate to runner kwargs
- [ ] How transfer evaluation works

---

## Priority 4: Testing Gaps

### 4.1 Modules with no dedicated test file
- [ ] `feature_belief.py` — tested indirectly, no `test_feature_belief.py`
- [ ] `observation_model.py` (V0) — orphaned, but should be tested if kept
- [ ] `visualize.py` — no tests (understandable for plotting)

### 4.2 Missing test scenarios
- [ ] `allowed_interventions` enforcement (config → policy → runner)
- [ ] Full matrix end-to-end with all 6 teacher conditions
- [ ] Shield consumption under belief-planning mode
- [ ] Transfer eval with belief_planning_mode agent

---

## Priority 5: Structural Cleanup

### 5.1 Config not programmatically loaded
- [ ] `agent.yaml` / `teacher.yaml` / `env.yaml` are reference-only
- [ ] Consider either: (a) wire them into runner, or (b) document they are reference
- [ ] `phase9_eval.yaml` IS loaded but uses manual parameter mapping

### 5.2 Parallel planner paths in runner
- [ ] `plan_and_move()` has if/else for belief_planning vs legacy
- [ ] Both paths call same A* but through different wrappers
- [ ] Consider making `plan_from_belief()` the single path with a `mode` flag

### 5.3 Runner is growing too large
- [ ] 528 lines, 30+ fields in state
- [ ] Consider extracting: (a) observation sub-module, (b) outcome resolution sub-module
- [ ] Low priority — works fine as-is

---

## Priority 6: Future-Facing

### 6.1 Full experiment matrix not yet run
- [ ] Only smoke subset (3 jobs) validated
- [ ] Need full 54-cell matrix with N=100 seeds
- [ ] Need transfer eval with meaningful training length

### 6.2 Config-driven teacher conditions
- [ ] `phase9_eval.yaml` defines conditions but runner doesn't consume YAML directly
- [ ] Matrix script does manual mapping — works but fragile

### 6.3 Pedagogical metrics need validation
- [ ] `boredom_proxy` and `frustration_proxy` are untested heuristics
- [ ] `intervention_timing_quality` uses simple first-risky-step proxy
- [ ] These should be calibrated against actual experiment data
