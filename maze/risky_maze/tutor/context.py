from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Literal

from .compat import get_any
from .world_model import remaining_time_from_state


@dataclass
class TutorDecisionContext:
    true_env_state: Any
    true_layout: Any
    learner_observation: Any = None
    learner_memory_snapshot: Any = None
    learner_risk_belief_snapshot: Any = None
    learner_policy_snapshot: Any = None
    history: Any = None
    phase: Literal["teach", "eval"] | str = "teach"
    remaining_time: int = 0


@dataclass
class EpisodeHistory:
    """Minimal append-only history used by the new tutor.

    Existing runners can continue to use their own history objects; all tutor
    code below reads history through duck-typed accessors.
    """

    steps: list[Any] = field(default_factory=list)

    def append(self, **kwargs: Any) -> None:
        self.steps.append(SimpleNamespace(**kwargs))


def coerce_context(*args: Any, **kwargs: Any) -> TutorDecisionContext:
    """Accept either the new context object or common legacy act signatures."""
    if args and isinstance(args[0], TutorDecisionContext):
        return args[0]
    if args and all(hasattr(args[0], name) for name in ("true_env_state", "true_layout")):
        ctx = args[0]
        return TutorDecisionContext(
            true_env_state=ctx.true_env_state,
            true_layout=ctx.true_layout,
            learner_observation=get_any(ctx, ["learner_observation", "observation", "obs"], None),
            learner_memory_snapshot=get_any(ctx, ["learner_memory_snapshot", "memory"], None),
            learner_risk_belief_snapshot=get_any(ctx, ["learner_risk_belief_snapshot", "risk_belief"], None),
            learner_policy_snapshot=get_any(ctx, ["learner_policy_snapshot", "policy_snapshot"], None),
            history=get_any(ctx, ["history"], None),
            phase=get_any(ctx, ["phase"], "teach"),
            remaining_time=int(get_any(ctx, ["remaining_time"], 0) or 0),
        )

    # Common old prototype shape: act(state, layout, observation, learner, ...)
    state = kwargs.get("state") or kwargs.get("true_env_state") or (args[0] if len(args) > 0 else None)
    layout = kwargs.get("layout") or kwargs.get("true_layout") or (args[1] if len(args) > 1 else None)
    obs = kwargs.get("observation") or kwargs.get("learner_observation") or (args[2] if len(args) > 2 else None)
    learner = kwargs.get("learner") or (args[3] if len(args) > 3 else None)
    memory = kwargs.get("learner_memory_snapshot") or get_any(learner, ["memory", "map_memory"], None)
    belief = kwargs.get("learner_risk_belief_snapshot") or get_any(learner, ["risk_belief", "belief"], None)
    policy = kwargs.get("learner_policy_snapshot") or get_any(learner, ["policy_snapshot", "last_policy_snapshot"], None)
    remaining = kwargs.get("remaining_time")
    if remaining is None and state is not None:
        remaining = remaining_time_from_state(state, 0)
    return TutorDecisionContext(
        true_env_state=state,
        true_layout=layout,
        learner_observation=obs,
        learner_memory_snapshot=memory,
        learner_risk_belief_snapshot=belief,
        learner_policy_snapshot=policy,
        history=kwargs.get("history"),
        phase=kwargs.get("phase", "teach"),
        remaining_time=int(remaining or 0),
    )
