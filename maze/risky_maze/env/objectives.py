"""Objective-machine primitives for fixed risky-maze POMDP tasks.

This module is intentionally small and dependency-light.  It can be added next
alongside the existing random-maze prototype without changing that code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

Coord = tuple[int, int]
ObjectiveKind = Literal["pickup", "pass", "collect_gem", "exit"]

_VALID_KINDS = {"pickup", "pass", "collect_gem", "exit"}
_KIND_TO_REQUIRED_CHAR = {
    "pickup": "K",
    "pass": "D",
    "collect_gem": "g",
    "exit": "E",
}


@dataclass(frozen=True, slots=True)
class Objective:
    """One step in a task-level objective sequence.

    Coordinates use the project convention ``(row, col)``.
    """

    kind: ObjectiveKind
    coord: Coord

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"Unsupported objective kind: {self.kind!r}")
        if len(self.coord) != 2:
            raise ValueError(f"Objective coord must be a pair, got {self.coord!r}")


@dataclass(slots=True)
class Inventory:
    """Mutable inventory/state touched by the objective machine."""

    has_key: bool = False
    collected_gems: set[Coord] = field(default_factory=set)


@dataclass(slots=True)
class ObjectiveEvent:
    """Event emitted by ``ObjectiveState.update`` after a movement step."""

    advanced: bool = False
    completed_objective: Optional[Objective] = None
    completed_all: bool = False
    picked_key: bool = False
    passed_door: bool = False
    collected_gem: Optional[Coord] = None
    reached_exit: bool = False
    message: str = ""


@dataclass(frozen=True, slots=True)
class LandmarkLearningEvent:
    """Explicit landmark-learning event emitted on cell contact.

    Unlike ``ObjectiveEvent``, this is not tied to the current objective sequence.
    It fires whenever the agent steps onto a landmark cell such as ``K/D/g/E`` so
    the learner can consolidate door/gem/key/exit structure even when the cell is
    not the active objective.
    """

    kind: ObjectiveKind
    coord: Coord
    objective_advanced: bool = False
    source: str = "landmark_contact"


@dataclass(slots=True)
class ObjectiveState:
    """State machine for ordered objectives.

    Runtime semantics are deliberately simple for v0:

    * ``pickup`` completes when the agent is on the key coordinate.
    * ``pass`` completes when the agent reaches the pass/bottleneck coordinate.
    * ``collect_gem`` completes when the agent is on the gem coordinate.
    * ``exit`` completes when the agent is on the exit coordinate; if this is
      the final objective, the task succeeds.

    Door locking is not enforced here; the door can be treated as a bottleneck
    objective first and upgraded to locked-door semantics later.
    """

    objectives: list[Objective]
    index: int = 0

    def current(self) -> Optional[Objective]:
        if 0 <= self.index < len(self.objectives):
            return self.objectives[self.index]
        return None

    @property
    def done(self) -> bool:
        return self.index >= len(self.objectives)

    def update(self, pos: Coord, layout: Any, inventory: Inventory) -> ObjectiveEvent:
        obj = self.current()
        if obj is None:
            return ObjectiveEvent(completed_all=True, message="objective sequence already complete")
        if pos != obj.coord:
            return ObjectiveEvent()

        cell_char = _layout_char(layout, pos)
        required = _KIND_TO_REQUIRED_CHAR[obj.kind]

        # The fixed spec validator already checks coordinates.  This tolerant
        # runtime guard prevents a malformed/custom test map from silently
        # advancing on the wrong object type.
        if cell_char is not None and cell_char != required:
            return ObjectiveEvent(
                advanced=False,
                message=f"at objective coordinate but expected {required!r}, saw {cell_char!r}",
            )

        picked_key = False
        passed_door = False
        collected_gem: Optional[Coord] = None
        reached_exit = False
        if obj.kind == "pickup":
            inventory.has_key = True
            picked_key = True
        elif obj.kind == "pass":
            passed_door = True
        elif obj.kind == "collect_gem":
            inventory.collected_gems.add(obj.coord)
            collected_gem = obj.coord
        elif obj.kind == "exit":
            reached_exit = True

        self.index += 1
        return ObjectiveEvent(
            advanced=True,
            completed_objective=obj,
            completed_all=self.done,
            picked_key=picked_key,
            passed_door=passed_door,
            collected_gem=collected_gem,
            reached_exit=reached_exit,
            message=f"completed {obj.kind} at {obj.coord}",
        )


def parse_objective(obj: Any) -> Objective:
    """Parse an objective from dict/dataclass/list-like forms.

    Supported examples:

    * ``{"kind": "pickup", "coord": [3, 4]}``
    * ``{"type": "exit", "at": [9, 10]}``
    * ``("collect_gem", (5, 7))``
    * dataclasses/objects with ``kind`` and ``coord`` attributes
    """

    if isinstance(obj, Objective):
        return obj

    if isinstance(obj, dict):
        kind = obj.get("kind", obj.get("type", obj.get("name")))
        coord = obj.get("coord", obj.get("at", obj.get("pos", obj.get("position"))))
        return Objective(kind=str(kind), coord=parse_runtime_coord(coord))  # type: ignore[arg-type]

    if isinstance(obj, (tuple, list)) and len(obj) == 2 and isinstance(obj[0], str):
        return Objective(kind=obj[0], coord=parse_runtime_coord(obj[1]))  # type: ignore[arg-type]

    kind = getattr(obj, "kind", getattr(obj, "type", None))
    coord = getattr(obj, "coord", getattr(obj, "at", getattr(obj, "pos", None)))
    if kind is None or coord is None:
        raise TypeError(f"Cannot parse objective from {obj!r}")
    return Objective(kind=str(kind), coord=parse_runtime_coord(coord))  # type: ignore[arg-type]


def parse_objective_sequence(values: Sequence[Any]) -> list[Objective]:
    return [parse_objective(v) for v in values]


def parse_runtime_coord(value: Any) -> Coord:
    if value is None:
        raise ValueError("Missing coordinate")
    if isinstance(value, dict):
        if "row" in value and "col" in value:
            return int(value["row"]), int(value["col"])
        if "r" in value and "c" in value:
            return int(value["r"]), int(value["c"])
        if "y" in value and "x" in value:
            return int(value["y"]), int(value["x"])
    if isinstance(value, (tuple, list)) and len(value) == 2:
        # Fixed-map scenario specs are authored in external (x, y) coordinates,
        # while the runtime uses internal (row, col) = (y, x).
        return int(value[1]), int(value[0])
    raise ValueError(f"Invalid coordinate: {value!r}")


_parse_coord = parse_runtime_coord


def _layout_char(layout: Any, coord: Coord) -> Optional[str]:
    """Best-effort cell char lookup across old/new layout representations."""

    if hasattr(layout, "char_at"):
        return layout.char_at(coord)
    if hasattr(layout, "cell_char"):
        return layout.cell_char(coord)
    if hasattr(layout, "get_char"):
        return layout.get_char(coord)
    row, col = coord
    grid = getattr(layout, "grid", None)
    if grid is None:
        grid = getattr(layout, "rows", None)
    if grid is not None:
        try:
            return str(grid[row][col])
        except Exception:
            return None
    try:
        return str(layout[coord])
    except Exception:
        return None
