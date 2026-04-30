# Implementation Plan Status

Generated on `2026-04-21`.

This report is the pre-full-experiment checkpoint for the scaffold-memory
implementation plan. It separates:

- implemented and validated
- implemented but not yet run in representative 4-seed form
- not started

## Completed and validated

### 1. Safety shield + scaffold split

Status: `done`

Implemented:

- safety-first tutor split is already wired
- waypoint candidate families are already separated into:
  - frontier
  - landmark
  - bottleneck
  - oracle

Key files:

- `risky_maze/tutor/inverse_planner.py`
- `risky_maze/tutor/candidates.py`
- `risky_maze/tutor/factory.py`
- `risky_maze/runner/fixed_episode_runner.py`

Validation:

- `tests/test_safety_scaffold_split.py`

### 2. Success-gated consolidation

Status: `done`

Implemented:

- teach-phase consolidation modes:
  - `none`
  - `always_commit`
  - `success_gated`
  - `success_gated_assist_discounted`
- teach failure no longer commits long-term transfer structure in success-gated modes

Key files:

- `risky_maze/learner/objective_agent.py`
- `risky_maze/config.py`
- `risky_maze/runner/fixed_block_runner.py`
- `risky_maze/experiments/run_fixed_maze.py`

Validation:

- `tests/test_transfer_consolidation.py::test_success_gated_commit_requires_teach_success`

### 3. Assist-discounted long-term memory / autonomy credit

Status: `done`

Implemented:

- `autonomy_credit = exp(-assist_discount * assist_leakage)`
- autonomy credit is stored in long-term transfer memory
- long-term route and landmark confidence increments are scaled by autonomy credit

Key files:

- `risky_maze/learner/objective_agent.py`
- `risky_maze/runner/fixed_metrics.py`

Validation:

- `tests/test_transfer_consolidation.py::test_assist_discount_reduces_autonomy_credit_and_commit_strength`

### 4. Route graph / landmark graph transfer structure

Status: `done`

Implemented:

- `TransferGraphMemory`
- route node confidence
- route edge confidence
- landmark confidence
- landmark graph confidence
- route/landmark bonuses in planner cost
- eval ablation can clear long-term transfer memory

Key files:

- `risky_maze/learner/objective_agent.py`
- `risky_maze/runner/fixed_metrics.py`
- `risky_maze/runner/fixed_block_runner.py`
- `risky_maze/experiments/run_fixed_maze.py`

Validation:

- `tests/test_transfer_consolidation.py::test_clone_for_eval_can_clear_long_term_memory`
- `tests/test_eval_ablation_clone.py`

### 5. Gem / door / key / exit as explicit learning events

Status: `done`

Implemented:

- learning events are now emitted on landmark contact, not only on objective completion
- event stream is independent from objective progression
- learner-side consolidation now consumes per-episode learning events

Key files:

- `risky_maze/env/objectives.py`
- `risky_maze/env/pomdp_episode.py`
- `risky_maze/runner/fixed_episode_runner.py`
- `risky_maze/learner/objective_agent.py`

Validation:

- `tests/test_transfer_consolidation.py::test_step_on_door_emits_learning_event_without_objective_advance`

### 6. TutorDidacticMazeSuite_v1 as the current diagnostic suite

Status: `done`

Implemented:

- dedicated three-map didactic suite retained as the main compact scaffold testbed:
  - `TutorSafetyScaffoldGate_v1`
  - `TutorAutonomyLoop_v1`
  - `TutorPrincipleDoorTransfer_v1`
- scenario docs and suite index are present

Key files:

- `risky_maze/scenarios/tutor_didactic_maze_suite_v1.py`
- `docs/specs/TUTOR_DIDACTIC_MAZE_SUITE_V1.md`
- `docs/specs/SCENARIO_SUITE_INDEX.md`

Validation:

- `tests/test_fixed_runtime_smoke.py::test_didactic_suite_specs_run_no_tutor_smoke`

### 7. Didactic runner now enables transfer-learning config by default

Status: `done`

Implemented:

- didactic map config now injects:
  - `learner_consolidation_mode`
  - `learner_long_term_memory_weight`
  - `learner_autonomy_assist_discount`
  - `learner_enable_objective_learning_events`
  - `learner_use_long_term_route_graph`
  - `learner_use_landmark_graph`

Key files:

- `risky_maze/experiments/run_didactic_tutor_suite.py`

Validation:

- `tests/test_transfer_consolidation.py::test_didactic_maps_enable_transfer_learning_defaults`
- `tests/test_didactic_suite_runner.py`

### 8. Scaffold ablation runner around the new architecture

Status: `done`

Implemented:

- new didactic scaffold ablation entrypoint
- conditions cover:
  - success-gated assist-discounted minimal scaffold
  - success-gated only
  - always-commit
  - clear-eval-long-term-memory
  - no-route-graph
  - no-landmark-graph
  - no-objective-learning-events
  - oracle scaffold comparisons

Key files:

- `risky_maze/experiments/run_didactic_scaffold_ablation.py`

Validation:

- smoke run completed:
  - `runs/smoke_didactic_scaffold_ablation/`
  - contains `DIDACTIC_SCAFFOLD_ABLATION_REPORT.md`

## Implemented and validated by broader smoke/regression

Passed:

- `tests/test_transfer_consolidation.py`
- `tests/test_eval_ablation_clone.py`
- `tests/test_didactic_suite_runner.py`
- `tests/test_fixed_runtime_smoke.py`
- `tests/test_safety_scaffold_split.py`
- `tests/test_formal_matrix_runner.py`

## Still pending before the work can be called fully complete

### 9. Representative 4-seed rerun under the new transfer-memory stack

Status: `pending execution`

Reason:

- the old 4-seed didactic report was generated before the new success-gated /
  assist-discounted transfer stack was actually activated in the didactic runner
- a fresh 4-seed run is still needed to verify whether:
  - `TutorAutonomyLoop_v1` now penalizes over-help more clearly
  - `TutorPrincipleDoorTransfer_v1` now shows stronger teach-success to eval-transfer coupling

Planned runner:

- `python -m risky_maze.experiments.run_didactic_tutor_suite ...`
- optionally followed by:
  - `python -m risky_maze.experiments.run_didactic_scaffold_ablation ...`

### 10. Full experiment matrix under the new scaffold architecture

Status: `not started`

Reason:

- per request, full experiments should not start before this implementation
  status report is available for review

## Summary

At this checkpoint:

- all missing learner-side mechanics from the implementation plan are now coded
- the new mechanics are wired through fixed runner, eval ablation, didactic suite,
  and scaffold ablation entrypoints
- unit tests and smoke tests pass
- the remaining work is experimental execution and reporting, not missing code
