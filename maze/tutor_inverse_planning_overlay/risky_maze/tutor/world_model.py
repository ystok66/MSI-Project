from __future__ import annotations

"""Small duck-typed maze accessors used by the inverse-planning tutor."""

from collections import deque
from typing import Any, Callable, Iterable

from .compat import Coord, as_coord, get_any, neighbors4, safe_float


def layout_size(layout: Any) -> tuple[int, int]:
    width = get_any(layout, ["width", "w", "n_cols", "cols"], None)
    height = get_any(layout, ["height", "h", "n_rows", "rows"], None)
    grid = get_any(layout, ["grid", "tiles", "map"], None)
    if width is None and grid is not None:
        try:
            width = len(grid[0])
        except Exception:
            width = 0
    if height is None and grid is not None:
        try:
            height = len(grid)
        except Exception:
            height = 0
    return int(width or 0), int(height or 0)


def in_bounds(layout: Any, coord: Coord) -> bool:
    method = get_any(layout, ["in_bounds"], None)
    if callable(method):
        try:
            return bool(method(coord))
        except TypeError:
            return bool(method(coord[0], coord[1]))
    w, h = layout_size(layout)
    return 0 <= coord[0] < w and 0 <= coord[1] < h


def grid_value(layout: Any, coord: Coord) -> Any:
    grid = get_any(layout, ["grid", "tiles", "map"], None)
    if grid is None:
        return None
    x, y = coord
    try:
        return grid[y][x]
    except Exception:
        return None


def is_wall(layout: Any, coord: Coord) -> bool:
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
    for name in ("cell_features", "features", "risk_vectors", "feature_map"):
        mapping = get_any(layout, [name], None)
        if isinstance(mapping, dict) and coord in mapping:
            return mapping[coord]
    return None


def observed_vector(memory: Any, layout: Any, coord: Coord, allow_oracle: bool = False) -> Any:
    if memory is not None:
        mapping = get_any(memory, ["observed_vectors", "features", "cell_features"], None)
        if isinstance(mapping, dict) and coord in mapping:
            return mapping[coord]
    return feature_at(layout, coord) if allow_oracle else None


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
    try:
        return {as_coord(c) for c in data if as_coord(c) is not None}  # type: ignore[misc]
    except Exception:
        return set()


def known_walkable(memory: Any) -> set[Coord]:
    data = get_any(memory, ["known_walkable", "walkable", "seen_walkable"], set()) or set()
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
        if isinstance(vecs, dict) and coord not in vecs:
            feat = feature_at(layout, coord) if allow_oracle_feature else None
            if feat is not None:
                vecs[coord] = feat
    except Exception:
        return


def shadow_traversable(layout: Any, memory: Any, coord: Coord) -> bool:
    if not in_bounds(layout, coord):
        return False
    if coord in known_walls(memory):
        return False
    # Do not allow planner to knowingly pass through true walls when tutor has
    # oracle layout; this is a stable rollout approximation, not learner cheating.
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
) -> list[Coord]:
    """A small A* returning [start, ..., goal]; empty if unreachable."""
    import heapq

    if start == goal:
        return [start]
    trav = traversable or (lambda c: is_walkable(layout, c))
    cost_fn = extra_cost or (lambda c: 0.0)
    if not trav(start) or not trav(goal):
        return []
    frontier: list[tuple[float, float, Coord]] = []
    heapq.heappush(frontier, (abs(start[0] - goal[0]) + abs(start[1] - goal[1]), 0.0, start))
    came: dict[Coord, Coord | None] = {start: None}
    g: dict[Coord, float] = {start: 0.0}
    expansions = 0
    while frontier and expansions < max_expansions:
        _, gcost, cur = heapq.heappop(frontier)
        expansions += 1
        if cur == goal:
            break
        if gcost > g.get(cur, float("inf")) + 1e-9:
            continue
        for nxt in neighbors4(cur):
            if not trav(nxt):
                continue
            ng = gcost + 1.0 + max(0.0, safe_float(cost_fn(nxt), 0.0))
            if ng + 1e-9 < g.get(nxt, float("inf")):
                g[nxt] = ng
                came[nxt] = cur
                h = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                heapq.heappush(frontier, (ng + h, ng, nxt))
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
    out: list[Coord] = []
    for c in kw:
        if any(in_bounds(layout, n) and n not in kw and n not in known_walls(memory) for n in neighbors4(c)):
            out.append(c)
    return out


def cells_in_radius(layout: Any, center: Coord, radius: int) -> set[Coord]:
    out: set[Coord] = set()
    cx, cy = center
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            c = (x, y)
            if abs(x - cx) + abs(y - cy) <= radius and in_bounds(layout, c):
                out.add(c)
    return out
