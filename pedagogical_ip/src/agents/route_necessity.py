"""
Route Necessity — Phase 10.

Computes route-level structural necessity: how much worse is the best
alternative path if we avoid the current candidate route?

    n_route = exp(-Δ / τ)
    Δ = J(π_avoid) - J(π_best)

When Δ is large (no good alternative), n_route → 1: uncertainty
penalty should be discounted because the agent HAS to use this route.
When Δ is small (good alternatives exist), n_route → 0: full
uncertainty penalty applies.

This addresses the "unknown ≠ dangerous" principle: unvisited cells
on the ONLY viable path should not receive the same uncertainty
penalty as unvisited cells with safe alternatives.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


def compute_route_necessity(
    agent_pos: tuple[int, int],
    goal: tuple[int, int],
    passable: np.ndarray,
    t: int,
    t_max: int,
    route_cells: set[tuple[int, int]] | None = None,
    tau: float = 3.0,
) -> float:
    """Compute scalar necessity for a candidate route.

    Uses BFS path lengths to compare:
      - best path (using all passable cells)
      - best path avoiding route_cells

    Returns n ∈ [0, 1]:
      - 1.0 if avoiding route makes goal unreachable or exceeds deadline
      - 0.0 if equally good alternatives exist

    Args:
        agent_pos: current agent position
        goal: target position
        passable: (H, W) bool mask
        t: current timestep
        t_max: episode deadline
        route_cells: set of cells on the candidate route to evaluate.
            If None, returns 0.0 (no specific route to evaluate).
        tau: temperature controlling sharpness. Lower = more binary.
    """
    if route_cells is None or len(route_cells) == 0:
        return 0.0

    H, W = passable.shape
    remaining = t_max - t

    # BFS: shortest path using all passable cells
    best_len = _bfs_shortest(agent_pos, goal, passable, H, W)

    # BFS: shortest path avoiding route_cells
    avoid_mask = passable.copy()
    for r, c in route_cells:
        avoid_mask[r, c] = False
    avoid_len = _bfs_shortest(agent_pos, goal, avoid_mask, H, W)

    # If best path is unreachable, necessity is meaningless
    if best_len >= 999:
        return 0.0

    # If avoid path is unreachable or exceeds deadline → necessity = 1
    if avoid_len >= 999 or avoid_len > remaining:
        return 1.0

    # If best path itself exceeds deadline, high necessity
    if best_len > remaining:
        return 0.8  # route is needed but even best path is tight

    delta = avoid_len - best_len
    if delta <= 0:
        return 0.0  # avoiding route is equally good or better

    return float(np.exp(-delta / tau))


def compute_necessity_for_path(
    path: list[tuple[int, int]],
    agent_pos: tuple[int, int],
    goal: tuple[int, int],
    passable: np.ndarray,
    t: int,
    t_max: int,
    tau: float = 3.0,
) -> float:
    """Compute necessity for the cells on a planned path.

    Treats all path cells as the "candidate route" and computes
    how much worse alternatives are if we avoid them.
    """
    if not path or len(path) < 2:
        return 0.0
    route_cells = set(path[1:])  # exclude current position
    return compute_route_necessity(
        agent_pos, goal, passable, t, t_max,
        route_cells=route_cells, tau=tau,
    )


def _bfs_shortest(
    start: tuple[int, int],
    goal: tuple[int, int],
    passable: np.ndarray,
    H: int,
    W: int,
) -> int:
    """BFS shortest path length. Returns 999 if unreachable."""
    if start == goal:
        return 0
    visited = set()
    queue = deque([(start, 0)])
    visited.add(start)
    while queue:
        pos, dist = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                if passable[nr, nc]:
                    if (nr, nc) == goal:
                        return dist + 1
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))
    return 999
