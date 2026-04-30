# TutorDidacticMazeSuite_v1

This document records the new three-map compact suite for the current tutor research direction.

Code entry points:

- [suite Python spec](../../risky_maze/scenarios/tutor_didactic_maze_suite_v1.py)
- [TutorSafetyScaffoldGate JSON](../../risky_maze/scenarios/tutor_safety_scaffold_gate_v1.json)
- [TutorAutonomyLoop JSON](../../risky_maze/scenarios/tutor_autonomy_loop_v1.json)
- [TutorPrincipleDoorTransfer JSON](../../risky_maze/scenarios/tutor_principle_door_transfer_v1.json)
- [combined suite JSON](../../risky_maze/scenarios/tutor_didactic_maze_suite_v1.json)
- [validation report](../../risky_maze/scenarios/tutor_didactic_maze_suite_v1.validation.json)

## Positioning

This suite is not the same as `MiniRiskyMazeSuite_v0`.

- `MiniRiskyMazeSuite_v0`
  is the legacy micro diagnostic suite for fast warning / wait / waypoint debugging.
- `TutorDidacticMazeSuite_v1`
  is the new main compact suite for:
  - truthful safety warning as a shield
  - minimal waypoint / hint scaffolding
  - over-help and assist leakage diagnosis
  - success-gated transfer from teach to eval

All three maps are `31 x 19`, intentionally small enough for rapid inverse-rollout debugging.

## Shared Design

- The learner does not see `r/m/q` directly.
- The learner observes noisy risk vectors sampled from latent safe / danger prototypes.
- Eval remains tutor-off.
- The intended mainline interpretation is:
  - `WARNING` = safety shield
  - `WAYPOINT / HINT` = pedagogical scaffold
  - successful teach should matter for later eval transfer

## TutorSafetyScaffoldGate_v1

Purpose:
`safety-hard gate + minimal scaffold`

This map is designed so the short route is dangerous and the safe route is longer.
It should answer:

- does the safety shield warning trigger in time?
- can a minimal detour waypoint help without revealing the full route?
- does no-tutor fail when HP is tight?

Primary metrics:

- `teach_safe_success_rate`
- `teach_death_rate`
- `preventable_death_rate`
- `warning_actionability`
- `eval_regret_to_oracle_safe_path`
- `assist_leakage`

## TutorAutonomyLoop_v1

Purpose:
`autonomy / useful exploration / over-help`

This map is mostly about branching structure and route reuse, not raw catastrophe.
It should answer:

- when is `WAIT` pedagogically useful?
- when does waypointing reduce no-progress without replacing exploration?
- when does over-waypointing hurt eval reuse?

Primary metrics:

- `useful_exploration_rate`
- `useful_wait_rate`
- `bad_wait_rate`
- `map_reuse_eval`
- `eval_regret_to_oracle_safe_path`
- `assist_leakage`
- `waypoint_progress_gift`

## TutorPrincipleDoorTransfer_v1

Purpose:
`success-gated transfer`

This map is designed so successful teach should commit a reusable
key-door / bottleneck route principle, while failed teach should not.
It should answer:

- does teach success predict eval success?
- does over-help reduce autonomy credit and transfer quality?
- does success-gated consolidation actually matter?

Primary metrics:

- `P(eval_success | teach_success)`
- `P(eval_success | teach_fail)`
- `eval_regret | teach_success`
- `eval_regret | teach_fail`
- `autonomy_credit`
- `route_graph_confidence`

## Practical Recommendation

Use this suite for future compact tutor experiments first.

Suggested order:

1. `TutorSafetyScaffoldGate_v1`
   to debug safety shield timing and warning actionability.
2. `TutorAutonomyLoop_v1`
   to compare minimal scaffolding against over-help.
3. `TutorPrincipleDoorTransfer_v1`
   to test success-gated and assist-discounted transfer.

