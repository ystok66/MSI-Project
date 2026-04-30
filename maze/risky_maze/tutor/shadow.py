from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import Any

from .compat import as_coord, get_any
from .profiles import LearnerProfile
from .world_model import objective_coord


@dataclass
class ObjectiveState:
    target: tuple[int, int] | None = None
    sequence: list[Any] = field(default_factory=list)
    index: int = 0
    has_gem: bool = False
    has_key: bool = False
    collected_gems: set[tuple[int, int]] = field(default_factory=set)

    def current(self) -> Any | None:
        if 0 <= self.index < len(self.sequence):
            return self.sequence[self.index]
        return None

    def advance_if_reached(self, pos: tuple[int, int]) -> bool:
        current = self.current()
        if current is None:
            return False
        coord = _objective_coord_from_item(current)
        if coord is None or coord != pos:
            return False
        kind = str(get_any(current, ["kind", "type", "name"], "") or "").lower()
        if kind == "pickup":
            self.has_key = True
        elif kind == "collect_gem":
            self.has_gem = True
            self.collected_gems.add(pos)
        self.index += 1
        self.target = _objective_coord_from_item(self.current())
        return self.index >= len(self.sequence)


@dataclass
class ShadowLearnerState:
    memory_hat: Any
    risk_belief_hat: Any
    objective_state_hat: ObjectiveState
    profile: LearnerProfile

    def clone(self) -> "ShadowLearnerState":
        return ShadowLearnerState(
            memory_hat=_clone_snapshot(self.memory_hat),
            risk_belief_hat=_clone_snapshot(self.risk_belief_hat),
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
    sequence = list(getattr(env_state, "objective_sequence", []) or []) if env_state is not None else []
    index = int(getattr(env_state, "objective_index", 0) or 0) if env_state is not None else 0
    has_key = bool(getattr(env_state, "has_key", False)) if env_state is not None else False
    collected_gems = set(getattr(env_state, "collected_gems", set()) or []) if env_state is not None else set()
    return ShadowLearnerState(
        memory_hat=_clone_snapshot(memory_snapshot),
        risk_belief_hat=_clone_snapshot(risk_belief_snapshot),
        objective_state_hat=ObjectiveState(
            target=target,
            sequence=copy.deepcopy(sequence),
            index=index,
            has_gem=has,
            has_key=has_key,
            collected_gems=collected_gems,
        ),
        profile=profile,
    )


def clone_from_learner(learner: Any, profile: LearnerProfile, env_state: Any = None, layout: Any = None) -> ShadowLearnerState:
    memory = getattr(learner, "memory", getattr(learner, "map_memory", None))
    belief = getattr(learner, "risk_belief", getattr(learner, "belief", None))
    return clone_from_snapshots(memory, belief, profile, env_state=env_state, layout=layout)


def _objective_coord_from_item(item: Any) -> tuple[int, int] | None:
    if item is None:
        return None
    return as_coord(get_any(item, ["coord", "at", "pos", "position"], item))


def _clone_snapshot(snapshot: Any) -> Any:
    if snapshot is None:
        return None
    for name in ("clone_for_shadow", "clone"):
        method = getattr(snapshot, name, None)
        if callable(method):
            try:
                return method()
            except Exception:
                pass
    return copy.deepcopy(snapshot)
