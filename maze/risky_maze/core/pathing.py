from __future__ import annotations

import heapq
from typing import Callable

from .types import Pos


def astar_path(
    start: Pos,
    goal: Pos,
    neighbors_fn: Callable[[Pos], list[Pos]],
    cost_fn: Callable[[Pos], float],
    heuristic_fn: Callable[[Pos, Pos], float],
) -> list[Pos]:
    frontier: list[tuple[float, float, Pos]] = []
    heapq.heappush(frontier, (heuristic_fn(start, goal), 0.0, start))
    came_from: dict[Pos, Pos | None] = {start: None}
    g_score: dict[Pos, float] = {start: 0.0}

    while frontier:
        _, cur_g, current = heapq.heappop(frontier)
        if current == goal:
            path = [current]
            while came_from[current] is not None:
                current = came_from[current]  # type: ignore[assignment]
                path.append(current)
            return list(reversed(path))
        if cur_g > g_score.get(current, float("inf")) + 1e-9:
            continue
        for nxt in neighbors_fn(current):
            cand_g = g_score[current] + cost_fn(nxt)
            if cand_g + 1e-9 < g_score.get(nxt, float("inf")):
                g_score[nxt] = cand_g
                came_from[nxt] = current
                f_score = cand_g + heuristic_fn(nxt, goal)
                heapq.heappush(frontier, (f_score, cand_g, nxt))
    return [start]


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
