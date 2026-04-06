"""
Map generator for PedagogicalGridEnv.

Generates fixed or procedural 8×8 grid maps with cell types,
true cost maps, and true risk maps.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class CellType(enum.IntEnum):
    """Cell types in the grid world."""
    NORMAL = 0
    WALL = 1
    HIGH_COST = 2
    RISKY = 3
    LOCKED_DOOR = 4
    TARGET = 5
    OBJECT_SPAWN = 6


@dataclass
class GridMap:
    """Immutable representation of a grid world layout."""
    height: int
    width: int
    cell_types: np.ndarray          # (H, W) int – CellType values
    true_cost: np.ndarray           # (H, W) float
    true_risk: np.ndarray           # (H, W) float  (probability of risk event)
    object_spawn: tuple[int, int]   # (row, col)
    target_pos: tuple[int, int]     # (row, col)
    agent_start: tuple[int, int]    # (row, col)
    door_positions: list[tuple[int, int]] = field(default_factory=list)


def generate_default_map() -> GridMap:
    """
    Hand-designed 8×8 map.

    Layout sketch (0-indexed, row-major):
        A = agent start (0,0)
        O = object spawn (0,6)
        T = target / delivery (7,7)
        W = wall
        D = locked door
        H = high cost
        R = risky
        . = normal

        . . . . H H O .
        . W W . H H . .
        . W W . . . . R
        . . . D . . R R
        . . . . . . R .
        H H . . . . . .
        H H . W W . . .
        . . . W W . . T
    """
    H, W = 8, 8
    cell_types = np.full((H, W), CellType.NORMAL, dtype=np.int32)

    # Walls
    for r, c in [(1, 1), (1, 2), (2, 1), (2, 2),
                 (6, 3), (6, 4), (7, 3), (7, 4)]:
        cell_types[r, c] = CellType.WALL

    # High-cost terrain
    for r, c in [(0, 4), (0, 5), (1, 4), (1, 5),
                 (5, 0), (5, 1), (6, 0), (6, 1)]:
        cell_types[r, c] = CellType.HIGH_COST

    # Risky terrain
    for r, c in [(2, 7), (3, 6), (3, 7), (4, 6)]:
        cell_types[r, c] = CellType.RISKY

    # Locked door
    cell_types[3, 3] = CellType.LOCKED_DOOR

    # Object spawn
    cell_types[0, 6] = CellType.OBJECT_SPAWN

    # Target
    cell_types[7, 7] = CellType.TARGET

    # --- Build cost and risk maps ---
    true_cost = np.ones((H, W), dtype=np.float64)
    true_risk = np.zeros((H, W), dtype=np.float64)

    for r in range(H):
        for c in range(W):
            ct = CellType(cell_types[r, c])
            if ct == CellType.WALL:
                true_cost[r, c] = np.inf
                true_risk[r, c] = 0.0
            elif ct == CellType.HIGH_COST:
                true_cost[r, c] = 5.0
            elif ct == CellType.RISKY:
                true_cost[r, c] = 1.0
                true_risk[r, c] = 0.3
            elif ct == CellType.LOCKED_DOOR:
                true_cost[r, c] = np.inf   # impassable until unlocked
                true_risk[r, c] = 0.0

    door_positions = [(3, 3)]

    return GridMap(
        height=H,
        width=W,
        cell_types=cell_types,
        true_cost=true_cost,
        true_risk=true_risk,
        object_spawn=(0, 6),
        target_pos=(7, 7),
        agent_start=(0, 0),
        door_positions=door_positions,
    )


def generate_random_map(
    height: int = 8,
    width: int = 8,
    wall_frac: float = 0.10,
    high_cost_frac: float = 0.10,
    risky_frac: float = 0.08,
    num_doors: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> GridMap:
    """
    Procedurally generate a random map.

    Guarantees walkability is NOT checked (v0 — simple).
    """
    if rng is None:
        rng = np.random.default_rng()

    cell_types = np.full((height, width), CellType.NORMAL, dtype=np.int32)
    total = height * width

    # Reserve corners for agent, object, target
    agent_start = (0, 0)
    object_spawn = (0, width - 1)
    target_pos = (height - 1, width - 1)
    reserved = {agent_start, object_spawn, target_pos}

    free_cells = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in reserved
    ]
    rng.shuffle(free_cells)

    idx = 0
    n_walls = int(total * wall_frac)
    n_high = int(total * high_cost_frac)
    n_risky = int(total * risky_frac)

    for _ in range(n_walls):
        if idx >= len(free_cells):
            break
        cell_types[free_cells[idx]] = CellType.WALL
        idx += 1

    for _ in range(n_high):
        if idx >= len(free_cells):
            break
        cell_types[free_cells[idx]] = CellType.HIGH_COST
        idx += 1

    for _ in range(n_risky):
        if idx >= len(free_cells):
            break
        cell_types[free_cells[idx]] = CellType.RISKY
        idx += 1

    door_positions: list[tuple[int, int]] = []
    for _ in range(num_doors):
        if idx >= len(free_cells):
            break
        pos = free_cells[idx]
        cell_types[pos] = CellType.LOCKED_DOOR
        door_positions.append(pos)
        idx += 1

    cell_types[object_spawn] = CellType.OBJECT_SPAWN
    cell_types[target_pos] = CellType.TARGET

    # Build cost / risk
    true_cost = np.ones((height, width), dtype=np.float64)
    true_risk = np.zeros((height, width), dtype=np.float64)
    for r in range(height):
        for c in range(width):
            ct = CellType(cell_types[r, c])
            if ct == CellType.WALL:
                true_cost[r, c] = np.inf
            elif ct == CellType.HIGH_COST:
                true_cost[r, c] = 3.0 + rng.uniform(0, 4)
            elif ct == CellType.RISKY:
                true_cost[r, c] = 1.0
                true_risk[r, c] = 0.15 + rng.uniform(0, 0.35)
            elif ct == CellType.LOCKED_DOOR:
                true_cost[r, c] = np.inf

    return GridMap(
        height=height,
        width=width,
        cell_types=cell_types,
        true_cost=true_cost,
        true_risk=true_risk,
        object_spawn=object_spawn,
        target_pos=target_pos,
        agent_start=agent_start,
        door_positions=door_positions,
    )
