from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .candidate_dataset import build_candidate_dataset_rows, write_candidate_dataset
from .config import OneHintConfig
from .experiment_presets import config_from_overrides, parse_seed_spec


def _load_grid_spec(path: str) -> dict:
    spec_path = Path(path)
    text = spec_path.read_text(encoding="utf-8")
    suffix = spec_path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to load YAML grid specs; use JSON or install PyYAML."
            ) from exc
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported grid spec extension: {suffix}")


def _experiment_cfg(base_cfg: OneHintConfig, experiment_spec: Dict[str, Any]) -> OneHintConfig:
    overrides = {
        key: value
        for key, value in experiment_spec.items()
        if key not in {"name", "description", "tasks", "seeds"}
    }
    return config_from_overrides(base_cfg, overrides)


def _find_experiment(grid_spec: dict, name: str) -> Dict[str, Any]:
    for experiment in list(grid_spec.get("experiments", []) or []):
        if str(experiment.get("name", "")) == str(name):
            return dict(experiment)
    raise ValueError(f"Experiment '{name}' not found in grid")


def _dataset_job(task_id: str, seed: int, cfg: OneHintConfig, families: Sequence[str]) -> List[dict]:
    return build_candidate_dataset_rows(
        task_id=str(task_id),
        cfg=cfg,
        seed=int(seed),
        families=tuple(str(f) for f in families),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export candidate dataset rows for a grid experiment.")
    parser.add_argument("--grid", required=True, help="Path to a JSON/YAML grid spec.")
    parser.add_argument("--experiment", required=True, help="Experiment name within the grid.")
    parser.add_argument("--out", required=True, help="Output JSONL file.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument(
        "--executor",
        choices=("thread", "process"),
        default="thread",
        help="Parallel executor kind.",
    )
    args = parser.parse_args()

    grid_spec = _load_grid_spec(str(args.grid))
    experiment = _find_experiment(grid_spec, str(args.experiment))
    base_cfg = config_from_overrides(OneHintConfig(), dict(grid_spec.get("base", {}) or {}))
    cfg = _experiment_cfg(base_cfg, experiment)
    families = tuple(str(f) for f in getattr(cfg, "hint_families", ()) or ())
    tasks = [str(task) for task in experiment.get("tasks", grid_spec.get("tasks", []))]
    seeds = parse_seed_spec(experiment.get("seeds", grid_spec.get("seeds", [0])))
    jobs: List[Tuple[int, str, int]] = []
    for task_id in tasks:
        for seed in seeds:
            jobs.append((len(jobs), str(task_id), int(seed)))

    indexed_rows: List[Tuple[int, List[dict]]] = []
    if max(1, int(args.workers)) <= 1 or len(jobs) <= 1:
        for idx, task_id, seed in jobs:
            indexed_rows.append((idx, _dataset_job(task_id, seed, cfg, families)))
    else:
        executor_cls = ThreadPoolExecutor if str(args.executor).lower() != "process" else ProcessPoolExecutor
        with executor_cls(max_workers=max(1, int(args.workers))) as executor:
            future_map = {
                executor.submit(_dataset_job, task_id, seed, cfg, families): idx
                for idx, task_id, seed in jobs
            }
            for future in as_completed(future_map):
                indexed_rows.append((future_map[future], future.result()))

    indexed_rows.sort(key=lambda item: item[0])
    rows: List[dict] = []
    for _, batch in indexed_rows:
        rows.extend(batch)

    write_candidate_dataset(rows, str(args.out))
    print(
        json.dumps(
            {
                "status": "ok",
                "grid": str(args.grid),
                "experiment": str(args.experiment),
                "tasks": tasks,
                "seeds": seeds,
                "families": list(families),
                "rows": len(rows),
                "out": str(Path(args.out)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
