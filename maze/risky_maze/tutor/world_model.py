from __future__ import annotations

"""Small duck-typed maze accessors used by the inverse-planning tutor."""

from collections import deque
from functools import lru_cache
import heapq as _heapq
from typing import Any, Callable, Iterable

import numpy as np

from .compat import Coord, as_coord, get_any, neighbors4, safe_float


def layout_size(layout: Any) -> tuple[int, int]:
    height = get_any(layout, ["height", "h", "n_rows", "rows"], None)
    width = get_any(layout, ["width", "w", "n_cols", "cols"], None)
    grid = get_any(layout, ["grid", "tiles", "map"], None)
    if height is None and grid is not None:
        try:
            height = len(grid)
        except Exception:
            height = 0
    if width is None and grid is not None:
        try:
            width = len(grid[0])
        except Exception:
            width = 0
    return int(height or 0), int(width or 0)


def in_bounds(layout: Any, coord: Coord) -> bool:
    method = getattr(layout, "in_bounds", None)
    if callable(method):
        try:
            return bool(method(coord))
        except TypeError:
            return bool(method(coord[0], coord[1]))
    method = get_any(layout, ["in_bounds"], None)
    if callable(method):
        try:
            return bool(method(coord))
        except TypeError:
            return bool(method(coord[0], coord[1]))
    h, w = layout_size(layout)
    return 0 <= coord[0] < h and 0 <= coord[1] < w


def grid_value(layout: Any, coord: Coord) -> Any:
    grid = get_any(layout, ["grid", "tiles", "map"], None)
    if grid is None:
        return None
    row, col = coord
    try:
        return grid[row][col]
    except Exception:
        return None


def is_wall(layout: Any, coord: Coord) -> bool:
    method = getattr(layout, "is_wall", None)
    if callable(method):
        try:
            return bool(method(coord))
        except TypeError:
            return bool(method(coord[0], coord[1]))
    method = get_any(layout, ["is_wall"], None)
    if callable(method):
        try:
            return bool(method(coord))
        except TypeError:
            return bool(method(coord[0], coord[1]))
    walls = get_any(layout, ["walls", "wall_cells", "blocked"], None)
    if walls is not None:
        try:
            if coord in walls:
                return True
        except Exception:
            pass
    v = grid_value(layout, coord)
    if isinstance(v, str):
        return v == "#"
    kind = get_any(v, ["kind", "cell_kind", "name", "value"], v)
    if isinstance(kind, str):
        return kind.upper() in {"WALL", "#"}
    return False


def is_walkable(layout: Any, coord: Coord) -> bool:
    method = getattr(layout, "is_walkable", None)
    if callable(method):
        try:
            return bool(method(coord))
        except TypeError:
            return bool(method(coord[0], coord[1]))
    method = get_any(layout, ["is_walkable", "walkable"], None)
    if callable(method):
        try:
            return bool(method(coord))
        except TypeError:
            return bool(method(coord[0], coord[1]))
    if not in_bounds(layout, coord):
        return False
    return not is_wall(layout, coord)


def true_damage(layout: Any, coord: Coord) -> float:
    method = getattr(layout, "trap_damage", None)
    if callable(method):
        try:
            return safe_float(method(coord), 0.0)
        except TypeError:
            return safe_float(method(coord[0], coord[1]), 0.0)
    method = get_any(layout, ["trap_damage", "damage_at", "get_damage"], None)
    if callable(method):
        try:
            return safe_float(method(coord), 0.0)
        except TypeError:
            return safe_float(method(coord[0], coord[1]), 0.0)
    traps = get_any(layout, ["traps", "trap_cells", "trap_types", "damage"], None)
    if isinstance(traps, dict):
        if coord in traps:
            value = traps[coord]
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            return max(0.0, safe_float(value, 1.0))
    elif traps is not None:
        try:
            return 1.0 if coord in traps else 0.0
        except Exception:
            pass
    v = grid_value(layout, coord)
    if isinstance(v, str) and v.upper() in {"T", "X", "TRAP"}:
        return 1.0
    return 0.0


def is_true_danger(layout: Any, coord: Coord) -> bool:
    return true_damage(layout, coord) > 0.0


def feature_at(layout: Any, coord: Coord) -> Any:
    method = get_any(layout, ["feature_at", "risk_feature", "get_feature"], None)
    if callable(method):
        try:
            return method(coord)
        except TypeError:
            return method(coord[0], coord[1])
    for name in ("cell_features", "features", "risk_vectors", "feature_map", "risk_latent_features"):
        mapping = get_any(layout, [name], None)
        if isinstance(mapping, dict) and coord in mapping:
            return mapping[coord]
    return None


def _normalize_feature_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return value
    if arr.size == 0:
        return None
    if arr.ndim == 0:
        return np.asarray([float(arr)], dtype=float)
    if arr.ndim == 1:
        return arr
    return np.asarray(arr.mean(axis=0), dtype=float)


def observed_vector(memory: Any, layout: Any, coord: Coord, allow_oracle: bool = False) -> Any:
    if memory is not None:
        mean_method = get_any(memory, ["mean_vector"], None)
        if callable(mean_method):
            try:
                feat = mean_method(coord)
            except Exception:
                feat = None
            if feat is not None:
                return _normalize_feature_value(feat)
        mapping = get_any(memory, ["observed_vectors", "features", "cell_features"], None)
        if isinstance(mapping, dict) and coord in mapping:
            return _normalize_feature_value(mapping[coord])
    return _normalize_feature_value(feature_at(layout, coord)) if allow_oracle else None


def current_pos(state: Any) -> Coord:
    value = get_any(state, ["pos", "position", "agent_pos", "coord", "location"], None)
    coord = as_coord(value)
    if coord is not None:
        return coord
    raise ValueError("Cannot infer learner position from MazeState-like object")


def has_gem(state: Any) -> bool:
    return bool(get_any(state, ["has_gem", "carrying_gem", "gem_collected"], False))


def hp_left(state: Any, default: float = 1.0) -> float:
    return safe_float(get_any(state, ["hp", "health", "life"], default), default)


def step_count(state: Any) -> int:
    return int(get_any(state, ["step_count", "steps", "t"], 0) or 0)


def time_limit(state: Any, default: int = 10**9) -> int:
    return int(get_any(state, ["time_limit", "max_steps", "horizon"], default) or default)


def remaining_time_from_state(state: Any, fallback: int = 10**9) -> int:
    return max(0, time_limit(state, fallback) - step_count(state))


def objective_coord(layout: Any, state: Any) -> Coord | None:
    current = get_any(state, ["current_objective"], None)
    coord = as_coord(current)
    if coord is not None:
        return coord
    objective_state = get_any(state, ["objective_state"], None)
    current_method = get_any(objective_state, ["current"], None)
    if callable(current_method):
        try:
            coord = as_coord(current_method())
        except Exception:
            coord = None
        if coord is not None:
            return coord
    # Current random prototype is start -> gem -> exit.  For a later objective
    # machine, this accessor can be replaced without changing tutor logic.
    if has_gem(state):
        value = get_any(layout, ["exit", "exit_coord", "goal", "goal_coord"], None)
    else:
        value = get_any(layout, ["gem", "gem_coord", "target", "target_coord"], None)
    coord = as_coord(value)
    if coord is not None:
        return coord
    objectives = get_any(state, ["objectives", "objective_queue"], None)
    if objectives:
        try:
            return as_coord(objectives[0])
        except Exception:
            return None
    return None


def known_walls(memory: Any) -> set[Coord]:
    data = get_any(memory, ["known_walls", "walls"], set()) or set()
    fast = _fast_coord_set(data)
    if fast is not None:
        return fast
    try:
        return {as_coord(c) for c in data if as_coord(c) is not None}  # type: ignore[misc]
    except Exception:
        return set()


def known_walkable(memory: Any) -> set[Coord]:
    data = get_any(memory, ["known_walkable", "walkable", "seen_walkable"], set()) or set()
    fast = _fast_coord_set(data)
    if fast is not None:
        return fast
    try:
        return {as_coord(c) for c in data if as_coord(c) is not None}  # type: ignore[misc]
    except Exception:
        return set()


def visited_count(memory: Any, coord: Coord) -> int:
    mapping = get_any(memory, ["visited_count", "visits"], None)
    if isinstance(mapping, dict):
        return int(mapping.get(coord, 0) or 0)
    method = get_any(memory, ["visit_count", "n_visits"], None)
    if callable(method):
        try:
            return int(method(coord))
        except Exception:
            return 0
    return 0


def is_known_to_learner(memory: Any, obs: Any, coord: Coord) -> bool:
    if coord in known_walkable(memory) or coord in known_walls(memory):
        return True
    visible = visible_cells(obs)
    return coord in visible


def visible_cells(obs: Any) -> set[Coord]:
    out: set[Coord] = set()
    if obs is None:
        return out
    for name in ("visible_cells", "cells", "observed_cells"):
        data = get_any(obs, [name], None)
        if data is None:
            continue
        if isinstance(data, dict):
            iterable = data.keys()
        else:
            iterable = data
        try:
            for c in iterable:
                cc = as_coord(c)
                if cc is not None:
                    out.add(cc)
        except Exception:
            pass
    for name in ("cell_features", "observed_vectors", "features"):
        data = get_any(obs, [name], None)
        if isinstance(data, dict):
            for c in data:
                cc = as_coord(c)
                if cc is not None:
                    out.add(cc)
    return out


def mark_memory_observed(memory: Any, layout: Any, coord: Coord, allow_oracle_feature: bool = True) -> None:
    if memory is None:
        return
    try:
        if is_walkable(layout, coord):
            data = get_any(memory, ["known_walkable"], None)
            if data is not None and hasattr(data, "add"):
                data.add(coord)
        else:
            data = get_any(memory, ["known_walls"], None)
            if data is not None and hasattr(data, "add"):
                data.add(coord)
        vecs = get_any(memory, ["observed_vectors"], None)
        if isinstance(vecs, dict):
            feat = _normalize_feature_value(feature_at(layout, coord) if allow_oracle_feature else None)
            if feat is not None:
                cache = get_any(memory, ["_mean_vector_cache"], None)
                if isinstance(cache, dict):
                    cache.pop(coord, None)
                if coord in vecs:
                    existing = vecs[coord]
                    if isinstance(existing, list):
                        existing.append(np.array(feat, copy=True))
                    else:
                        vecs[coord] = np.array(feat, copy=True)
                else:
                    if hasattr(memory, "mean_vector"):
                        vecs[coord] = [np.array(feat, copy=True)]
                    else:
                        vecs[coord] = np.array(feat, copy=True)
    except Exception:
        return


def shadow_traversable(layout: Any, memory: Any, coord: Coord, known_wall_cells: set[Coord] | None = None) -> bool:
    """Check if *coord* is traversable from the shadow learner's perspective.

    A cell is traversable if:
      1. The layout says it is walkable (not a wall), AND
      2. The learner's memory has NOT recorded it as a known wall.

    Performance Design
    ------------------
    This function is called ~17 M times per inverse-tutor episode via the A*
    inner loop.  Two code-paths exist:

    **Fast path** (``FixedRuntimeLayout`` with ``_walkable_coords``):
      Single ``coord in frozenset`` check + ``coord not in walls`` set check.
      Skips the entire ``in_bounds → is_wall → char_at → len()`` chain.
      Detected automatically via ``getattr(layout, '_walkable_coords', None)``.

    **Generic fallback** (any layout without ``_walkable_coords``):
      Original duck-typed call chain preserved for compatibility with
      non-fixed layouts or standalone tests.
    """
    # Fast path: FixedRuntimeLayout pre-computes a walkable frozenset at init.
    walkable_set = getattr(layout, '_walkable_coords', None)
    if walkable_set is not None:
        if coord not in walkable_set:
            return False
        walls = known_wall_cells if known_wall_cells is not None else known_walls(memory)
        return coord not in walls
    # Generic fallback for non-fixed layouts
    if not in_bounds(layout, coord):
        return False
    walls = known_wall_cells if known_wall_cells is not None else known_walls(memory)
    if coord in walls:
        return False
    if is_wall(layout, coord):
        return False
    return True


def true_neighbors(layout: Any, coord: Coord) -> list[Coord]:
    return [n for n in neighbors4(coord) if is_walkable(layout, n)]


def degree(layout: Any, coord: Coord, traversable: Callable[[Coord], bool] | None = None) -> int:
    f = traversable or (lambda c: is_walkable(layout, c))
    return sum(1 for n in neighbors4(coord) if f(n))


def shortest_path(
    layout: Any,
    start: Coord,
    goal: Coord,
    traversable: Callable[[Coord], bool] | None = None,
    extra_cost: Callable[[Coord], float] | None = None,
    max_expansions: int = 5000,
    extra_cost_map: dict[Coord, float] | None = None,
    extra_cost_offset: float = 0.0,
) -> list[Coord]:
    """A* search returning ``[start, ..., goal]``; empty list if unreachable.

    This is the **primary A* interface** used by the entire tutor subsystem.
    All path-planning throughout the codebase routes through this function.

    Parameters
    ----------
    layout : Any
        Grid layout providing ``is_walkable`` / ``_walkable_coords``.
    start, goal : Coord
        Source and destination (row, col) tuples.
    traversable : callable, optional
        ``f(coord) -> bool``.  Overrides the default ``is_walkable`` check.
        When provided from ``predict_topk``, this is typically a pre-computed
        ``coord in frozenset`` lookup for performance.
    extra_cost : callable, optional
        ``f(coord) -> float`` giving additional traversal cost for *coord*.
        The effective edge cost is ``1.0 + max(0, extra_cost(coord) - offset)``.
    extra_cost_map : dict, optional
        Dict-based alternative to *extra_cost* for faster lookup when the
        cost cache is already materialised.  Takes priority over *extra_cost*.
    extra_cost_offset : float
        Subtracted from the extra cost value before applying.  This allows
        callers to pass ``cell_cost_cached`` directly (which returns values
        starting at 1.0) without wrapping in a ``lambda c: max(0, f(c) - 1)``
        closure — eliminating ~13 M lambda calls per episode.
    max_expansions : int
        Safety limit to prevent unbounded search in large grids.

    Performance Design
    ------------------
    The inner loop is micro-optimised for CPython:

    * ``neighbors4`` is inlined as a tuple literal (avoids function call +
      list allocation per expansion node).
    * ``heapq`` functions are bound to locals (``_heappush``, ``_heappop``)
      to skip module-level ``getattr`` on each call.
    * ``g.get`` is bound to a local ``g_get`` to avoid method resolution.
    * ``float('inf')`` is pre-computed once as ``_inf``.

    These micro-optimisations collectively reduce the A* self-time from
    48.7 s to 7.7 s (84 %% reduction) across a 65-step inverse-tutor episode.
    """
    if start == goal:
        return [start]
    trav = traversable or (lambda c: is_walkable(layout, c))
    if not trav(start) or not trav(goal):
        return []
    _heappush = _heapq.heappush
    _heappop = _heapq.heappop
    frontier: list[tuple[float, float, Coord]] = []
    _heappush(frontier, (abs(start[0] - goal[0]) + abs(start[1] - goal[1]), 0.0, start))
    came: dict[Coord, Coord | None] = {start: None}
    g: dict[Coord, float] = {start: 0.0}
    g_get = g.get
    _inf = float("inf")
    use_map = extra_cost_map is not None
    cost_fn = extra_cost
    expansions = 0
    while frontier and expansions < max_expansions:
        _, gcost, cur = _heappop(frontier)
        expansions += 1
        if cur == goal:
            break
        if gcost > g_get(cur, _inf) + 1e-9:
            continue
        cr, cc = cur
        # Inline neighbors4 to avoid function-call overhead on every expansion.
        for nxt in ((cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)):
            if not trav(nxt):
                continue
            ng = gcost + 1.0
            if use_map:
                ec = extra_cost_map.get(nxt, 0.0) - extra_cost_offset
                if ec > 0.0:
                    ng += ec
            elif cost_fn is not None:
                ec = cost_fn(nxt) - extra_cost_offset
                if ec > 0.0:
                    ng += ec
            if ng + 1e-9 < g_get(nxt, _inf):
                g[nxt] = ng
                came[nxt] = cur
                _heappush(frontier, (ng + abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]), ng, nxt))
    if goal not in came:
        return []
    path: list[Coord] = []
    cur: Coord | None = goal
    while cur is not None:
        path.append(cur)
        cur = came[cur]
    return list(reversed(path))


def reachable_known_frontiers(layout: Any, memory: Any) -> list[Coord]:
    kw = known_walkable(memory)
    walls = known_walls(memory)
    out: list[Coord] = []
    for c in kw:
        if any(in_bounds(layout, n) and n not in kw and n not in walls for n in neighbors4(c)):
            out.append(c)
    return out


def cells_in_radius(layout: Any, center: Coord, radius: int) -> set[Coord]:
    out: set[Coord] = set()
    row0, col0 = center
    for dr, dc in _square_radius_offsets(int(radius)):
        c = (row0 + dr, col0 + dc)
        if in_bounds(layout, c):
            out.add(c)
    return out


def _fast_coord_set(data: Any) -> set[Coord] | None:
    if not isinstance(data, set):
        return None
    if not data:
        return data
    try:
        sample = next(iter(data))
    except StopIteration:
        return data
    if (
        isinstance(sample, tuple)
        and len(sample) == 2
        and isinstance(sample[0], int)
        and isinstance(sample[1], int)
    ):
        return data
    return None


@lru_cache(maxsize=16)
def _square_radius_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (dr, dc)
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
    )
