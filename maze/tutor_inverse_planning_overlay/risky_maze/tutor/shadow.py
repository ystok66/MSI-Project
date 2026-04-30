from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any

from .profiles import LearnerProfile
from .world_model import objective_coord


@dataclass
class ObjectiveState:
    target: tuple[int, int] | None = None
    has_gem: bool = False


@dataclass
class ShadowLearnerState:
    memory_hat: Any
    risk_belief_hat: Any
    objective_state_hat: ObjectiveState
    profile: LearnerProfile

    def clone(self) -> "ShadowLearnerState":
        return ShadowLearnerState(
            memory_hat=copy.deepcopy(self.memory_hat),
            risk_belief_hat=copy.deepcopy(self.risk_belief_hat),
            objective_state_hat=copy.deepcopy(self.objective_state_hat),
            profile=self.profile,
        )


def clone_from_snapshots(
    memory_snapshot: Any,
    risk_belief_snapshot: Any,
    profile: LearnerProfile,
    env_state: Any = None,
    layout: Any = None,
) -> ShadowLearnerState:
    target = objective_coord(layout, env_state) if env_state is not None and layout is not None else None
    has = bool(getattr(env_state, "has_gem", False)) if env_state is not None else False
    return ShadowLearnerState(
        memory_hat=copy.deepcopy(memory_snapshot),
        risk_belief_hat=copy.deepcopy(risk_belief_snapshot),
        objective_state_hat=ObjectiveState(target=target, has_gem=has),
        profile=profile,
    )


def clone_from_learner(learner: Any, profile: LearnerProfile, env_state: Any = None, layout: Any = None) -> ShadowLearnerState:
    memory = getattr(learner, "memory", getattr(learner, "map_memory", None))
    belief = getattr(learner, "risk_belief", getattr(learner, "belief", None))
    return clone_from_snapshots(memory, belief, profile, env_state=env_state, layout=layout)
