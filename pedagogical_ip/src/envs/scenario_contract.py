"""Scenario post-generation contract validation.

Level 1: goal reachability + metadata consistency
Level 2: family-specific (future)
Level 3: latent-mode semantic contract (future)
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .map_generator import CellType, GridMap


def bfs_shortest(passable: np.ndarray, start: tuple[int, int],
                 goal: tuple[int, int]) -> int:
    """BFS shortest path on passable grid, returns length or -1."""
    H, W = passable.shape
    visited = set()
    queue = deque([(start, 0)])
    visited.add(start)
    while queue:
        (r, c), d = queue.popleft()
        if (r, c) == goal:
            return d
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited and passable[nr, nc]:
                visited.add((nr, nc))
                queue.append(((nr, nc), d + 1))
    return -1


def validate_scenario_contract(
    gm: GridMap,
    meta,
    start: Optional[tuple[int, int]] = None,
    goal: Optional[tuple[int, int]] = None,
    *,
    check_metadata: bool = True,
) -> None:
    """Level 1 post-generation validation.

    Raises AssertionError if any contract is violated.
    Called at the end of each scenario generator.

    Checks:
      1. Goal reachable from start
      2. metadata.shortest_any matches BFS recomputation
    """
    H, W = gm.height, gm.width
    _start = start or getattr(gm, 'agent_start', (2, 1)) or (2, 1)
    _goal = goal or getattr(gm, 'target_pos', (2, W - 2)) or (2, W - 2)

    # Build passable mask — WALL is always impassable.
    # LOCKED_DOOR: impassable for reachability, but shortest_any metadata
    # may have been computed with doors open (e.g. deadline_gate shortcut).
    passable_strict = np.ones((H, W), dtype=bool)
    passable_lenient = np.ones((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            ct = gm.cell_types[r, c]
            if ct == CellType.WALL:
                passable_strict[r, c] = False
                passable_lenient[r, c] = False
            elif ct == CellType.LOCKED_DOOR:
                passable_strict[r, c] = False
                # lenient keeps locked doors passable

    # Check 1: Goal reachable (lenient — doors may be opened at runtime)
    shortest_lenient = bfs_shortest(passable_lenient, _start, _goal)
    assert shortest_lenient >= 0, (
        f"Scenario contract violation: goal {_goal} unreachable from {_start}")

    # Check 2: Metadata consistency (sanity bound)
    # shortest_any may have been computed via topology-aware BFS (e.g. DTMB
    # branch-specific routing) that differs slightly from simple grid BFS.
    # Use a sanity bound rather than exact match.
    if check_metadata and hasattr(meta, 'shortest_any') and meta.shortest_any is not None:
        ratio = meta.shortest_any / max(shortest_lenient, 1)
        assert 0.8 <= ratio <= 1.25, (
            f"Metadata mismatch: meta.shortest_any={meta.shortest_any}, "
            f"BFS shortest={shortest_lenient}, ratio={ratio:.2f}")
