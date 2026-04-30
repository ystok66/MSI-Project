from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

Pos = tuple[int, int]


class CellKind(str, Enum):
    WALL = "wall"
    FLOOR = "floor"
    GEM = "gem"
    EXIT = "exit"


class Action(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    STAY = "stay"


ACTION_DELTAS: dict[Action, Pos] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.STAY: (0, 0),
}


@dataclass(frozen=True)
class VisibleCell:
    pos: Pos
    kind: CellKind
    walkable: bool
    observed_vec: np.ndarray | None


@dataclass(frozen=True)
class Observation:
    agent_pos: Pos
    gem_pos: Pos
    exit_pos: Pos
    has_gem: bool
    hp: int
    time_remaining: int
    visible_cells: tuple[VisibleCell, ...]


@dataclass(frozen=True)
class StepOutcome:
    observation: Observation
    moved_to: Pos
    damage: int
    trap_type: int
    success: bool
    died: bool
    timeout: bool


@dataclass(frozen=True)
class TutorAction:
    kind: str
    cells: tuple[Pos, ...] = ()
    waypoint: Pos | None = None
    reason: str = ""
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicySnapshot:
    target: Pos
    planned_path: tuple[Pos, ...]
