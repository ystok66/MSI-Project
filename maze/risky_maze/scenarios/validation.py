from __future__ import annotations

from collections import Counter, deque


OBJECTIVE_SYMBOL = {
    "pickup": "K",
    "pass": "D",
    "collect_gem": "g",
    "exit": "E",
}

INTERACTIVE_START_SYMBOLS = {"K", "D", "g", "E"}


def _symbol_at(map_lines: list[str], pos: tuple[int, int]) -> str:
    x, y = pos
    return map_lines[y][x]


def _in_bounds(width: int, height: int, pos: tuple[int, int]) -> bool:
    x, y = pos
    return 0 <= x < width and 0 <= y < height


def _is_walkable(map_lines: list[str], pos: tuple[int, int]) -> bool:
    return _symbol_at(map_lines, pos) != "#"


def _shortest_path_length(
    map_lines: list[str],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> int | None:
    if start == goal:
        return 0
    width = len(map_lines[0])
    height = len(map_lines)
    queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
    seen = {start}
    while queue:
        (x, y), dist = queue.popleft()
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nxt in seen or not _in_bounds(width, height, nxt):
                continue
            if not _is_walkable(map_lines, nxt):
                continue
            if nxt == goal:
                return dist + 1
            seen.add(nxt)
            queue.append((nxt, dist + 1))
    return None


def summarize_fixed_map_spec(spec: dict) -> dict:
    map_lines = spec["map"]
    counts = Counter("".join(map_lines))
    passable = sum(count for symbol, count in counts.items() if symbol != "#")
    return {
        "name": spec["name"],
        "width": len(map_lines[0]),
        "height": len(map_lines),
        "passable_cells": passable,
        "symbol_counts": dict(counts),
        "n_teach_tasks": len(spec["tasks"].get("teach", [])),
        "n_eval_same_map_tasks": len(spec["tasks"].get("eval_same_map_no_tutor", [])),
    }


def validate_fixed_map_spec(spec: dict) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    map_lines = spec["map"]

    if not map_lines:
        return {"errors": ["map is empty"], "warnings": [], "summary": {}}

    width = len(map_lines[0])
    height = len(map_lines)
    valid_symbols = set(spec["legend"].keys())

    for y, row in enumerate(map_lines):
        if len(row) != width:
            errors.append(f"row {y} has width {len(row)} but expected {width}")
        for x, symbol in enumerate(row):
            if symbol not in valid_symbols:
                errors.append(f"unknown symbol '{symbol}' at {(x, y)}")

    if spec.get("width") != width:
        errors.append(f"spec width={spec.get('width')} but actual width={width}")
    if spec.get("height") != height:
        errors.append(f"spec height={spec.get('height')} but actual height={height}")

    for split_name, tasks in spec["tasks"].items():
        for task in tasks:
            task_id = task["id"]
            start = tuple(task["start"])
            if not _in_bounds(width, height, start):
                errors.append(f"{task_id}: start {start} is out of bounds")
                continue
            if not _is_walkable(map_lines, start):
                errors.append(f"{task_id}: start {start} is a wall")
                continue

            start_symbol = _symbol_at(map_lines, start)
            if start_symbol in INTERACTIVE_START_SYMBOLS:
                warnings.append(
                    f"{task_id}: start {start} is on interactive tile '{start_symbol}'"
                )

            route = [start]
            for objective_name, objective_pos in task["objectives"]:
                pos = tuple(objective_pos)
                if objective_name not in OBJECTIVE_SYMBOL:
                    errors.append(f"{task_id}: unknown objective type '{objective_name}'")
                    continue
                if not _in_bounds(width, height, pos):
                    errors.append(f"{task_id}: objective {objective_name} at {pos} is out of bounds")
                    continue
                symbol = _symbol_at(map_lines, pos)
                expected = OBJECTIVE_SYMBOL[objective_name]
                if symbol != expected:
                    errors.append(
                        f"{task_id}: objective {objective_name} at {pos} has symbol '{symbol}', "
                        f"expected '{expected}'"
                    )
                route.append(pos)

            shortest_total = 0
            route_ok = True
            for src, dst in zip(route, route[1:]):
                distance = _shortest_path_length(map_lines, src, dst)
                if distance is None:
                    errors.append(f"{task_id}: no walkable path from {src} to {dst}")
                    route_ok = False
                    break
                shortest_total += distance
            if route_ok and shortest_total > task["time_limit"]:
                errors.append(
                    f"{task_id}: time_limit={task['time_limit']} is below walkable shortest-path "
                    f"lower bound {shortest_total}"
                )

    return {
        "errors": errors,
        "warnings": warnings,
        "summary": summarize_fixed_map_spec(spec),
    }
