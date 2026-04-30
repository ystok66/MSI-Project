from __future__ import annotations

from collections.abc import Iterable
from collections import deque

import numpy as np

from ..config import MazeScenarioConfig
from ..core import CellKind, Pos, astar_path, manhattan
from .layout import MazeLayout
from .prototypes import PrototypeBank


def _farthest_reachable(layout: MazeLayout, start: Pos) -> Pos:
    queue: deque[Pos] = deque([start])
    seen = {start}
    last = start
    while queue:
        pos = queue.popleft()
        last = pos
        for nxt in layout.neighbors(pos):
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return last


def _carve_maze(
    height: int,
    width: int,
    rng: np.random.Generator,
    extra_loop_prob: float,
) -> np.ndarray:
    grid = np.full((height, width), CellKind.WALL.value, dtype=object)
    start = (1, 1)
    grid[start] = CellKind.FLOOR.value
    stack = [start]
    dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    while stack:
        r, c = stack[-1]
        candidates: list[tuple[int, int, int, int]] = []
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nr = r + dr
            nc = c + dc
            if 1 <= nr < height - 1 and 1 <= nc < width - 1:
                if grid[nr, nc] == CellKind.WALL.value:
                    candidates.append((nr, nc, r + dr // 2, c + dc // 2))
        if not candidates:
            stack.pop()
            continue
        nr, nc, mr, mc = candidates[0]
        grid[mr, mc] = CellKind.FLOOR.value
        grid[nr, nc] = CellKind.FLOOR.value
        stack.append((nr, nc))

    for r in range(1, height - 1, 2):
        for c in range(1, width - 1, 2):
            if rng.random() < extra_loop_prob:
                dr, dc = dirs[int(rng.integers(0, len(dirs)))]
                mr = r + dr // 2
                mc = c + dc // 2
                nr = r + dr
                nc = c + dc
                if 1 <= nr < height - 1 and 1 <= nc < width - 1:
                    grid[mr, mc] = CellKind.FLOOR.value
                    grid[nr, nc] = CellKind.FLOOR.value
    return grid


def generate_layout(
    cfg: MazeScenarioConfig,
    rng: np.random.Generator,
    bank: PrototypeBank | None = None,
) -> MazeLayout:
    bank = bank or PrototypeBank.random(cfg, rng)
    kind_grid = _carve_maze(cfg.height, cfg.width, rng, cfg.extra_loop_prob)
    probe = MazeLayout(
        cfg=cfg,
        bank=bank,
        kind_grid=kind_grid.copy(),
        trap_grid=np.zeros((cfg.height, cfg.width), dtype=int),
        feature_grid=np.zeros((cfg.height, cfg.width, cfg.risk_dim), dtype=float),
        start=(1, 1),
        gem=(1, 1),
        exit=(1, 1),
    )
    start = (1, 1)
    gem = _farthest_reachable(probe, start)
    exit_pos = _farthest_reachable(probe, gem)
    kind_grid[gem] = CellKind.GEM.value
    kind_grid[exit_pos] = CellKind.EXIT.value

    safe_path = astar_path(start, gem, probe.neighbors, lambda _: 1.0, manhattan)
    safe_path.extend(astar_path(gem, exit_pos, probe.neighbors, lambda _: 1.0, manhattan)[1:])
    safe_path_set = set(safe_path)

    shortcut_candidates: list[Pos] = []
    for r in range(1, cfg.height - 1):
        for c in range(1, cfg.width - 1):
            pos = (r, c)
            if kind_grid[pos] != CellKind.WALL.value:
                continue
            walkable_ns = (
                kind_grid[r - 1, c] != CellKind.WALL.value
                and kind_grid[r + 1, c] != CellKind.WALL.value
            )
            walkable_ew = (
                kind_grid[r, c - 1] != CellKind.WALL.value
                and kind_grid[r, c + 1] != CellKind.WALL.value
            )
            if not (walkable_ns or walkable_ew):
                continue
            neighbor_positions = {
                (r - 1, c),
                (r + 1, c),
                (r, c - 1),
                (r, c + 1),
            }
            if neighbor_positions & safe_path_set:
                shortcut_candidates.append(pos)

    rng.shuffle(shortcut_candidates)
    n_shortcuts = min(len(shortcut_candidates), max(1, len(safe_path) // 8))
    forced_traps = set(shortcut_candidates[:n_shortcuts])
    for pos in forced_traps:
        kind_grid[pos] = CellKind.FLOOR.value

    trap_grid = np.zeros((cfg.height, cfg.width), dtype=int)
    feature_grid = np.zeros((cfg.height, cfg.width, cfg.risk_dim), dtype=float)

    protected = {start, gem, exit_pos, *safe_path}
    for r in range(cfg.height):
        for c in range(cfg.width):
            pos = (r, c)
            if kind_grid[pos] == CellKind.WALL.value:
                continue
            trap_type = 0
            if pos in forced_traps:
                trap_type = int(rng.integers(1, cfg.n_trap_types + 1))
            elif pos not in protected and rng.random() < cfg.trap_density:
                trap_type = int(rng.integers(1, cfg.n_trap_types + 1))
            trap_grid[pos] = trap_type
            feature_grid[pos] = bank.sample_feature(trap_type, rng)

    return MazeLayout(
        cfg=cfg,
        bank=bank,
        kind_grid=kind_grid,
        trap_grid=trap_grid,
        feature_grid=feature_grid,
        start=start,
        gem=gem,
        exit=exit_pos,
    )


def sample_starts(
    layout: MazeLayout,
    n: int,
    rng: np.random.Generator,
    forbid: Iterable[Pos] = (),
) -> list[Pos]:
    forbidden = set(forbid)
    candidates = [
        pos
        for pos in layout.traversable_positions()
        if pos not in forbidden
    ]
    if not candidates:
        return [layout.start] * n
    rng.shuffle(candidates)
    out = candidates[:n]
    while len(out) < n:
        out.append(candidates[len(out) % len(candidates)])
    return out
