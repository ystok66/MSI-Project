from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from ..core import CellKind, Observation, Pos


@dataclass
class MapMemory:
    known_walls: set[Pos] = field(default_factory=set)
    known_walkable: set[Pos] = field(default_factory=set)
    seen_kind: dict[Pos, CellKind] = field(default_factory=dict)
    observed_vectors: dict[Pos, np.ndarray] = field(default_factory=dict)
    confirmed_traps: dict[Pos, int] = field(default_factory=dict)
    visited_count: dict[Pos, int] = field(default_factory=dict)
    warning_suspicion: dict[Pos, float] = field(default_factory=dict)

    def copy(self) -> "MapMemory":
        return MapMemory(
            known_walls=set(self.known_walls),
            known_walkable=set(self.known_walkable),
            seen_kind=dict(self.seen_kind),
            observed_vectors={
                pos: vec.copy() for pos, vec in self.observed_vectors.items()
            },
            confirmed_traps=dict(self.confirmed_traps),
            visited_count=dict(self.visited_count),
            warning_suspicion=dict(self.warning_suspicion),
        )

    def observe(self, obs: Observation) -> None:
        for cell in obs.visible_cells:
            self.seen_kind[cell.pos] = cell.kind
            if cell.walkable:
                self.known_walkable.add(cell.pos)
                if cell.observed_vec is not None:
                    self.observed_vectors[cell.pos] = cell.observed_vec.copy()
            else:
                self.known_walls.add(cell.pos)

    def mark_visit(self, pos: Pos) -> None:
        self.visited_count[pos] = self.visited_count.get(pos, 0) + 1

    def confirm_trap(self, pos: Pos, trap_type: int) -> None:
        self.confirmed_traps[pos] = trap_type

    def is_known_wall(self, pos: Pos) -> bool:
        return pos in self.known_walls

    def is_known_walkable(self, pos: Pos) -> bool:
        return pos in self.known_walkable

    def visited(self, pos: Pos) -> int:
        return self.visited_count.get(pos, 0)

    def observed_feature(self, pos: Pos) -> np.ndarray | None:
        feat = self.observed_vectors.get(pos)
        return None if feat is None else feat.copy()

    def add_warning_suspicion(self, cells: Iterable[Pos], amount: float = 0.8) -> None:
        for pos in cells:
            self.warning_suspicion[pos] = min(
                0.95,
                self.warning_suspicion.get(pos, 0.0) + amount,
            )
