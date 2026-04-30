from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..config import MazeScenarioConfig
from ..core import ACTION_DELTAS, CellKind, Pos
from .prototypes import PrototypeBank


@dataclass
class MazeLayout:
    cfg: MazeScenarioConfig
    bank: PrototypeBank
    kind_grid: np.ndarray
    trap_grid: np.ndarray
    feature_grid: np.ndarray
    start: Pos
    gem: Pos
    exit: Pos

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.kind_grid.shape[0]), int(self.kind_grid.shape[1])

    def in_bounds(self, pos: Pos) -> bool:
        r, c = pos
        h, w = self.shape
        return 0 <= r < h and 0 <= c < w

    def kind_at(self, pos: Pos) -> CellKind:
        return CellKind(self.kind_grid[pos])

    def is_wall(self, pos: Pos) -> bool:
        return self.kind_at(pos) == CellKind.WALL

    def is_walkable(self, pos: Pos) -> bool:
        return not self.is_wall(pos)

    def trap_type_at(self, pos: Pos) -> int:
        return int(self.trap_grid[pos])

    def feature_at(self, pos: Pos) -> np.ndarray:
        return self.feature_grid[pos].copy()

    def neighbors(self, pos: Pos) -> list[Pos]:
        out: list[Pos] = []
        for dr, dc in ACTION_DELTAS.values():
            if (dr, dc) == (0, 0):
                continue
            nxt = (pos[0] + dr, pos[1] + dc)
            if self.in_bounds(nxt) and self.is_walkable(nxt):
                out.append(nxt)
        return out

    def traversable_positions(self) -> list[Pos]:
        h, w = self.shape
        out: list[Pos] = []
        for r in range(h):
            for c in range(w):
                pos = (r, c)
                if self.is_walkable(pos):
                    out.append(pos)
        return out

    def shortest_path_length(self, start: Pos, goal: Pos) -> int:
        if start == goal:
            return 0
        queue: deque[tuple[Pos, int]] = deque([(start, 0)])
        seen = {start}
        while queue:
            pos, dist = queue.popleft()
            for nxt in self.neighbors(pos):
                if nxt in seen:
                    continue
                if nxt == goal:
                    return dist + 1
                seen.add(nxt)
                queue.append((nxt, dist + 1))
        return max(self.shape) * max(self.shape)
