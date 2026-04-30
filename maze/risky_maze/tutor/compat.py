from __future__ import annotations

"""Compatibility helpers for the lightweight inverse-planning tutor.

The current prototype already has its own ``core.types.TutorAction`` and several
maze/learner classes.  These helpers deliberately use duck typing so the new
modules can be copied into the existing repository without a large refactor.
"""

from dataclasses import dataclass, field, fields, is_dataclass
import inspect
from typing import Any, Iterable, Mapping

Coord = tuple[int, int]

try:  # Prefer the project's existing action dataclass when available.
    from risky_maze.core.types import TutorAction as _ProjectTutorAction  # type: ignore
except Exception:  # pragma: no cover - used by standalone tests/overlay preview.
    _ProjectTutorAction = None


@dataclass
class FallbackTutorAction:
    kind: str
    cells: tuple[Coord, ...] = ()
    waypoint: Coord | None = None
    reason: str = ""
    diagnostics: dict[str, float] = field(default_factory=dict)


TutorAction = _ProjectTutorAction or FallbackTutorAction


ACTION_DELTAS: dict[str, Coord] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
    "WAIT": (0, 0),
}


def as_coord(value: Any) -> Coord | None:
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, list) and len(value) == 2:
        return int(value[0]), int(value[1])
    if hasattr(value, "coord"):
        return as_coord(value.coord)
    if hasattr(value, "x") and hasattr(value, "y"):
        return int(value.y), int(value.x)
    if hasattr(value, "row") and hasattr(value, "col"):
        return int(value.row), int(value.col)
    return None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_action_name(action: Any) -> str:
    if action is None:
        return "WAIT"
    if isinstance(action, str):
        return action.upper()
    name = getattr(action, "name", None)
    if name is not None:
        return str(name).upper()
    value = getattr(action, "value", None)
    if isinstance(value, str):
        return value.upper()
    return str(action).upper()


def action_between(a: Coord, b: Coord) -> str:
    dx, dy = b[0] - a[0], b[1] - a[1]
    for name, delta in ACTION_DELTAS.items():
        if delta == (dx, dy):
            return name
    return "WAIT"


def neighbors4(coord: Coord) -> list[Coord]:
    row, col = coord
    return [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]


def get_any(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def set_if_missing(obj: Any, name: str, value: Any) -> None:
    try:
        if not hasattr(obj, name):
            setattr(obj, name, value)
    except Exception:
        pass


def _constructor_kwargs(cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    if is_dataclass(cls):
        allowed = {f.name for f in fields(cls)}
        return {k: v for k, v in kwargs.items() if k in allowed}
    try:
        sig = inspect.signature(cls)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(kwargs)
        return {k: v for k, v in kwargs.items() if k in params}
    except Exception:
        return dict(kwargs)


def make_tutor_action(
    kind: str,
    cells: Iterable[Coord] | None = None,
    waypoint: Coord | None = None,
    reason: str = "",
    diagnostics: Mapping[str, float] | None = None,
) -> Any:
    """Create a TutorAction while tolerating older dataclass signatures.

    Older prototypes may only support ``kind`` and ``cells``.  In that case we
    still attach ``waypoint``/``diagnostics`` dynamically when possible.  This
    avoids forcing a broad core/types.py migration just to run the new tutor.
    """
    payload = {
        "kind": kind,
        "cells": tuple(cells or ()),
        "waypoint": waypoint,
        "reason": reason,
        "diagnostics": dict(diagnostics or {}),
    }
    cls = TutorAction
    try:
        obj = cls(**_constructor_kwargs(cls, payload))
    except Exception:
        obj = FallbackTutorAction(**payload)
    for k, v in payload.items():
        set_if_missing(obj, k, v)
    # If an older dataclass has reason/diagnostics fields with defaults, update them too.
    for k in ("reason", "diagnostics", "waypoint"):
        try:
            setattr(obj, k, payload[k])
        except Exception:
            pass
    return obj


def action_key(action: Any) -> tuple[str, tuple[Coord, ...], Coord | None]:
    return (
        str(get_any(action, ["kind"], "WAIT")),
        tuple(get_any(action, ["cells"], ()) or ()),
        as_coord(get_any(action, ["waypoint"], None)),
    )

