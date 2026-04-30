# Tutor Inverse Planning overlay

This overlay adds a lightweight finite-profile inverse-planning tutor for the existing `risky_maze/` prototype.
It is designed to be copied into the current repository with minimal disruption.

## Files to copy

Copy these files into `risky_maze/tutor/`:

```text
compat.py
world_model.py
context.py
profiles.py
shadow.py
path_predictor.py
candidates.py
rollout.py
diagnostics.py
inverse_planner.py
baselines.py
factory.py
```

Then update `risky_maze/tutor/__init__.py` if you want the new classes exported. The overlay's `__init__.py` shows the exports.

## Minimal runner integration

The new tutor prefers a single context object:

```python
from risky_maze.tutor.context import TutorDecisionContext

context = TutorDecisionContext(
    true_env_state=env.state,
    true_layout=env.layout,
    learner_observation=obs,
    learner_memory_snapshot=learner.memory,
    learner_risk_belief_snapshot=learner.risk_belief,
    learner_policy_snapshot=getattr(learner, "last_policy_snapshot", None),
    history=history,
    phase=phase,
    remaining_time=env.state.time_limit - env.state.step_count,
)
tutor_action = tutor.act(context)
```

The implementation also tolerates many legacy call shapes such as `tutor.act(state, layout, obs, learner, phase=...)`, but the context object is cleaner.

## Factory integration

In the existing `risky_maze/tutor/warning_policies.py` `build_tutor()` function, add something like:

```python
if name in {
    "risk_threshold_warning",
    "always_waypoint",
    "warning_only_inverse",
    "full_inverse",
    "inverse_planning",
}:
    from risky_maze.tutor.factory import build_inverse_tutor
    return build_inverse_tutor(name, config)
```

Keep the existing `NoTutor`, `AlwaysWarnTutor`, and heuristic `InverseWarnTutor` names unchanged for backward compatibility.

## Optional TutorAction extension

If current `core/types.py::TutorAction` lacks `waypoint`, `reason`, or `diagnostics`, add them:

```python
@dataclass
class TutorAction:
    kind: Literal["WAIT", "WARNING", "WAYPOINT"]
    cells: tuple[Coord, ...] = ()
    waypoint: Coord | None = None
    reason: str = ""
    diagnostics: dict[str, float] = field(default_factory=dict)
```

The overlay has a compatibility constructor that can attach these fields dynamically, but adding them to the core dataclass is better for logging and type checking.

## Runner behavior for WAYPOINT

For `WAYPOINT(g)`, the runner or learner should set a temporary subgoal/prior rather than receiving a full path. A minimal first pass:

```python
if tutor_action.kind == "WAYPOINT":
    learner.temporary_waypoint = tutor_action.waypoint
```

Then in the learner planner, use `temporary_waypoint` as the immediate target until reached or stale. Do not convert it into a full oracle route.

## Diagnostics

Every returned action includes `action.diagnostics`, including:

- `candidate_count`
- `q_wait`
- `q_best_warning`
- `q_best_waypoint`
- `selected_q`
- rollout risk/damage/map-gain/IG/leakage terms

The tutor also stores append-only decision logs at:

```python
tutor.diagnostics.decisions
```

After the actual env step, the runner can call:

```python
tutor.diagnostics.update_last_actual(
    actual_next_damage=outcome.damage,
    actual_next_new_cells=new_cells,
    actual_warning_ig=warning_ig,
    actual_progress=progress,
)
```

## Policy names implemented

- `risk_threshold_warning`
- `always_waypoint`
- `warning_only_inverse`
- `full_inverse` / `inverse_planning`

## Current scope

This overlay targets the current random-maze runtime described in the implementation report. It does not implement the fixed-map objective machine (`pickup/pass/collect_gem/exit`); that should remain a separate integration step.
