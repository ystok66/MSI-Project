"""Fixed-map loader for RiskyGemMaze POMDP runtime.

The loader is deliberately tolerant about the exact spec representation because
existing scenarios may expose either a dataclass object, module constants, or a
JSON file.  The public outputs are simple dataclasses used by the new runtime.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from .objectives import Coord, Objective, parse_objective_sequence

SplitName = Literal["teach", "eval_same_map", "eval_same_map_no_tutor", "eval", "eval_new_map_same_risk"]

SAFE_CHARS = {".", ",", ":", "K", "D", "g", "E", "S"}
TRAP_CHARS = {"r", "m", "q"}
WALL_CHARS = {"#", "W", "X", " "}
WALKABLE_CHARS = SAFE_CHARS | TRAP_CHARS
LEGAL_CHARS = WALKABLE_CHARS | WALL_CHARS

SAFE_CLASS = {".": 0, ",": 1, ":": 2, "K": 0, "D": 0, "g": 0, "E": 0, "S": 0}
TRAP_CLASS = {"r": 3, "m": 4, "q": 5}
TRAP_TYPE = {"r": 1, "m": 2, "q": 3}
TRAP_DAMAGE = {"r": 1, "m": 2, "q": 2}


@dataclass(frozen=True, slots=True)
class FixedRuntimeLayout:
    """Runtime layout backed by fixed text map lines.

    It intentionally hides trap labels from observations; the labels are only
    available to environment dynamics/tutors/oracle comparators.
    """

    rows: tuple[str, ...]
    name: str = "fixed"

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("FixedRuntimeLayout requires at least one row")
        width = len(self.rows[0])
        if width == 0:
            raise ValueError("FixedRuntimeLayout rows cannot be empty")
        for i, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(f"Non-rectangular fixed map at row {i}: {len(row)} != {width}")
            bad = set(row) - LEGAL_CHARS
            if bad:
                raise ValueError(f"Illegal map chars at row {i}: {sorted(bad)!r}")

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        return len(self.rows[0])

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def in_bounds(self, coord: Coord) -> bool:
        r, c = coord
        return 0 <= r < self.height and 0 <= c < self.width

    def char_at(self, coord: Coord) -> str:
        if not self.in_bounds(coord):
            return "#"
        r, c = coord
        return self.rows[r][c]

    def is_wall(self, coord: Coord) -> bool:
        return self.char_at(coord) in WALL_CHARS

    def is_walkable(self, coord: Coord) -> bool:
        return self.in_bounds(coord) and self.char_at(coord) not in WALL_CHARS

    def is_trap(self, coord: Coord) -> bool:
        return self.char_at(coord) in TRAP_CHARS

    def trap_type(self, coord: Coord) -> int:
        return TRAP_TYPE.get(self.char_at(coord), 0)

    def trap_damage(self, coord: Coord) -> int:
        return TRAP_DAMAGE.get(self.char_at(coord), 0)

    def latent_class_id(self, coord: Coord) -> int:
        ch = self.char_at(coord)
        if ch in TRAP_CLASS:
            return TRAP_CLASS[ch]
        return SAFE_CLASS.get(ch, 0)

    def visible_kind(self, coord: Coord) -> str:
        """Learner-facing symbol class; never exposes r/m/q."""
        ch = self.char_at(coord)
        if ch in WALL_CHARS:
            return "wall"
        if ch == "K":
            return "key"
        if ch == "D":
            return "door"
        if ch == "g":
            return "gem"
        if ch == "E":
            return "exit"
        return "walkable"

    def walkable_cells(self) -> list[Coord]:
        return [(r, c) for r in range(self.height) for c in range(self.width) if self.is_walkable((r, c))]

    def trap_cells(self) -> list[Coord]:
        return [coord for coord in self.walkable_cells() if self.is_trap(coord)]

    def neighbors4(self, coord: Coord, *, allow_traps: bool = True) -> Iterable[Coord]:
        r, c = coord
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if self.is_walkable(nb) and (allow_traps or not self.is_trap(nb)):
                yield nb


@dataclass(slots=True)
class MazeTask:
    task_id: str
    split: str
    start: Coord
    objectives: list[Objective]
    time_limit: int
    raw: Any = None


@dataclass(slots=True)
class FixedMazeSpec:
    name: str
    map_lines: list[str]
    teach_tasks: dict[str, Any] = field(default_factory=dict)
    eval_tasks: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    def layout(self) -> FixedRuntimeLayout:
        return layout_from_fixed_spec(self)

    def task_ids(self, split: str) -> list[str]:
        tasks = _select_tasks(self, split)
        return list(tasks.keys())


class FixedSpecLoadError(RuntimeError):
    pass


def layout_from_fixed_spec(spec: FixedMazeSpec | Any) -> FixedRuntimeLayout:
    fixed = spec if isinstance(spec, FixedMazeSpec) else _normalise_spec(spec, name=_spec_name(spec))
    return FixedRuntimeLayout(rows=tuple(fixed.map_lines), name=fixed.name)


def load_fixed_spec(name: str = "HugeRiskyGemMaze_v0") -> FixedMazeSpec:
    """Load a fixed-maze spec from the scenario module or JSON file."""

    module_candidates = _module_candidates(name)
    errors: list[str] = []
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - best effort diagnostics
            errors.append(f"{module_name}: {exc}")
            continue

        raw = _extract_spec_object(module)
        if raw is not None:
            return _normalise_spec(raw, name=name)

        json_path = Path(getattr(module, "__file__", "")).with_suffix(".json")
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as f:
                return _normalise_spec(json.load(f), name=name)

    # Fallback: direct JSON next to this file's parent package/scenarios.
    root = Path(__file__).resolve().parents[1]
    for stem in {_snake_case(name), name, name.lower()}:
        path = root / "scenarios" / f"{stem}.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return _normalise_spec(json.load(f), name=name)

    raise FixedSpecLoadError("Could not load fixed spec. Tried: " + "; ".join(errors + module_candidates))


def build_layout_from_spec(spec: FixedMazeSpec, config: Any | None = None) -> FixedRuntimeLayout:
    del config  # reserved for future fixed-layout transforms
    return layout_from_fixed_spec(spec)


def build_task_from_spec(spec: FixedMazeSpec, split: str, task_id: str) -> MazeTask:
    tasks = _select_tasks(spec, split)
    if task_id not in tasks:
        available = ", ".join(tasks.keys())
        raise KeyError(f"Unknown task_id={task_id!r} for split={split!r}; available: {available}")
    return parse_task(tasks[task_id], task_id=task_id, split=split)


def list_task_ids(spec: FixedMazeSpec, split: str) -> list[str]:
    return list(_select_tasks(spec, split).keys())


def parse_task(task: Any, *, task_id: str | None = None, split: str = "teach") -> MazeTask:
    d = _as_dict(task)
    tid = str(task_id or d.get("task_id") or d.get("id") or d.get("name") or "task")
    start = _parse_coord(_first_present(d, "start", "start_coord", "initial_pos", "pos"))
    objectives_raw = _first_present(d, "objective_sequence", "objectives", "goals", "route")
    if objectives_raw is None:
        raise ValueError(f"Task {tid!r} has no objectives/objective_sequence")
    objectives = parse_objective_sequence(objectives_raw)
    time_limit = int(_first_present(d, "time_limit", "horizon", "max_steps") or 0)
    if time_limit <= 0:
        # Conservative fallback when old specs omitted a per-task value.
        time_limit = 4 * max(1, len(objectives)) * 100
    return MazeTask(task_id=tid, split=split, start=start, objectives=objectives, time_limit=time_limit, raw=task)


def make_fixed_episode(
    spec_name: str,
    task_id: str,
    split: Literal["teach", "eval_same_map_no_tutor", "eval_same_map", "eval"] = "teach",
    config: Any | None = None,
    seed: int = 0,
    *,
    baseline_mode: str = "mortal",
    phase: str | None = None,
):
    """Create a fixed-map POMDP runtime episode.

    Import is intentionally local to avoid a circular dependency between loader
    and environment runtime.
    """

    from .pomdp_episode import RiskyMazePOMDPEnv

    spec = load_fixed_spec(spec_name)
    layout = build_layout_from_spec(spec, config)
    task = build_task_from_spec(spec, split, task_id)
    return RiskyMazePOMDPEnv(
        layout=layout,
        task=task,
        config=config,
        seed=seed,
        prototype_seed=seed,
        baseline_mode=baseline_mode,
        phase=phase or ("eval" if split.startswith("eval") else "teach"),
    )


def _select_tasks(spec: FixedMazeSpec, split: str) -> dict[str, Any]:
    if split in {"teach", "train"}:
        return spec.teach_tasks
    if split in {"eval", "eval_same_map", "eval_same_map_no_tutor", "eval_new_map_same_risk", "test"}:
        return spec.eval_tasks
    raise ValueError(f"Unsupported split: {split!r}")


def _normalise_spec(raw: Any, *, name: str) -> FixedMazeSpec:
    d = _as_dict(raw)
    spec_name = str(d.get("name") or d.get("spec_name") or name)
    lines = _extract_map_lines(d)

    teach_tasks = _extract_task_dict(d, preferred_keys=("teach_tasks", "train_tasks", "teach", "train"))
    eval_tasks = _extract_task_dict(
        d,
        preferred_keys=("eval_tasks", "test_tasks", "eval_same_map_tasks", "eval", "test"),
    )

    # Some specs keep all tasks in one list with an explicit split field.
    all_tasks = _first_present(d, "tasks", "task_suite", "episodes")
    if all_tasks is not None and (not teach_tasks or not eval_tasks):
        teach_from_all, eval_from_all = _partition_tasks(all_tasks)
        teach_tasks = teach_tasks or teach_from_all
        eval_tasks = eval_tasks or eval_from_all

    return FixedMazeSpec(
        name=spec_name,
        map_lines=list(lines),
        teach_tasks=teach_tasks,
        eval_tasks=eval_tasks,
        raw=raw,
    )


def _extract_map_lines(d: dict[str, Any]) -> list[str]:
    value = _first_present(d, "map_lines", "lines", "layout", "grid", "maze", "map")
    if value is None:
        raise FixedSpecLoadError("Spec has no map_lines/layout/grid field")
    if isinstance(value, str):
        lines = [line.rstrip("\n") for line in value.splitlines() if line.rstrip("\n")]
    else:
        lines = ["".join(row) if isinstance(row, (tuple, list)) else str(row) for row in value]
    if not lines:
        raise FixedSpecLoadError("Spec map is empty")
    return lines


def _extract_task_dict(d: dict[str, Any], *, preferred_keys: tuple[str, ...]) -> dict[str, Any]:
    for key in preferred_keys:
        if key in d and d[key] is not None:
            return _task_collection_to_dict(d[key])
    return {}


def _partition_tasks(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    teach: dict[str, Any] = {}
    eval_: dict[str, Any] = {}
    for tid, task in _task_collection_to_dict(value).items():
        td = _as_dict(task)
        split = str(td.get("split") or td.get("phase") or "teach").lower()
        if split.startswith("eval") or split in {"test", "validation"}:
            eval_[tid] = task
        else:
            teach[tid] = task
    return teach, eval_


def _task_collection_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        # Either already {task_id: task}, or a wrapper with a nested list/dict.
        if all(isinstance(v, (dict, object)) for v in value.values()) and any(
            k in _as_dict(next(iter(value.values())) if value else {}).keys()
            for k in ("start", "objective_sequence", "objectives")
        ):
            return {str(k): v for k, v in value.items()}
        for nested_key in ("tasks", "items", "episodes"):
            if nested_key in value:
                return _task_collection_to_dict(value[nested_key])
        return {str(k): v for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        out: dict[str, Any] = {}
        for i, item in enumerate(value):
            d = _as_dict(item)
            tid = str(d.get("task_id") or d.get("id") or d.get("name") or f"task_{i:03d}")
            out[tid] = item
        return out
    raise TypeError(f"Unsupported task collection: {type(value).__name__}")


def _extract_spec_object(module: Any) -> Any | None:
    names = (
        "HUGE_RISKY_GEM_MAZE_V0",
        "HugeRiskyGemMaze_v0",
        "HugeRiskyGemMazeV0",
        "SPEC",
        "spec",
        "FIXED_SPEC",
    )
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    for factory in ("get_spec", "load_spec", "build_spec"):
        if hasattr(module, factory):
            fn = getattr(module, factory)
            if callable(fn):
                return fn()
    return None


def _module_candidates(name: str) -> list[str]:
    snake = _snake_case(name)
    return [
        f"risky_maze.scenarios.{snake}",
        f"risky_maze.scenarios.{name}",
        f"risky_maze.scenarios.{name.lower()}",
    ]


def _snake_case(name: str) -> str:
    out: list[str] = []
    prev_lower = False
    for ch in name:
        if ch.isupper() and prev_lower:
            out.append("_")
        if ch in {"-", " ", "."}:
            out.append("_")
            prev_lower = False
        else:
            out.append(ch.lower())
            prev_lower = ch.islower() or ch.isdigit()
    return "".join(out).replace("__", "_")


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    raise TypeError(f"Cannot convert {type(obj).__name__} to dict")


def _first_present(d: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in d:
            return d[key]
    return None


def _parse_coord(value: Any) -> Coord:
    if value is None:
        raise ValueError("Missing coordinate")
    if isinstance(value, dict):
        for a, b in (("row", "col"), ("r", "c"), ("y", "x")):
            if a in value and b in value:
                return int(value[a]), int(value[b])
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Invalid coordinate: {value!r}")


def _spec_name(raw: Any) -> str:
    d = _as_dict(raw)
    return str(d.get("name") or d.get("spec_name") or "fixed")
