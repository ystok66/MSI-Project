from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from risky_maze.experiments.run_fixed_maze import _finalize_experiment_outputs, _progress_bar, _run_seed_job, _write_csv, _write_json


DEFAULT_SEEDS = list(range(8))
DEFAULT_TEACH_TASKS = ["T01_NW_key_gem_NE_exit"]
DEFAULT_EVAL_TASKS = ["E01_WestGarden_to_NE_exit"]


def default_conditions() -> list[dict[str, Any]]:
    base_config = {
        "risk_dim": 8,
        "obs_noise": 0.25,
        "cluster_std": 0.35,
        "view_radius": 2,
        "hp": 3,
        "n_safe_types": 3,
        "n_trap_types": 3,
        "tutor_rollout_horizon": 6,
        "tutor_top_k_paths": 2,
        "tutor_max_candidates": 8,
        "tutor_waypoint_cooldown_steps": 6,
        "tutor_max_waypoints_per_episode": 3,
        "tutor_profile_count": 3,
        "warning_update_mode": "effective_sample",
    }
    return [
        {
            "condition_name": "no_tutor_mortal",
            "tutor_name": "no_tutor",
            "baseline_mode": "mortal",
            "seeds": DEFAULT_SEEDS,
            "config_overrides": dict(base_config),
        },
        {
            "condition_name": "no_tutor_immortal_warnlike",
            "tutor_name": "no_tutor",
            "baseline_mode": "immortal_warnlike",
            "seeds": DEFAULT_SEEDS,
            "config_overrides": dict(base_config),
        },
        {
            "condition_name": "no_tutor_immortal_no_timeout",
            "tutor_name": "no_tutor",
            "baseline_mode": "immortal_no_timeout",
            "seeds": DEFAULT_SEEDS,
            "config_overrides": dict(base_config),
        },
        {
            "condition_name": "always_warn_mortal",
            "tutor_name": "always_warn",
            "baseline_mode": "mortal",
            "seeds": DEFAULT_SEEDS,
            "config_overrides": dict(base_config),
        },
        {
            "condition_name": "risk_threshold_warn_mortal",
            "tutor_name": "risk_threshold_warn",
            "baseline_mode": "mortal",
            "seeds": DEFAULT_SEEDS,
            "config_overrides": dict(base_config),
        },
        {
            "condition_name": "inverse_warn_mortal",
            "tutor_name": "inverse_warn",
            "baseline_mode": "mortal",
            "seeds": DEFAULT_SEEDS,
            "config_overrides": dict(base_config),
        },
        {
            "condition_name": "inverse_plan_warn_only",
            "tutor_name": "inverse_plan_warn_only",
            "baseline_mode": "mortal",
            "seeds": DEFAULT_SEEDS,
            "config_overrides": {
                **base_config,
                "tutor_rollout_horizon": 5,
                "tutor_top_k_paths": 3,
                "tutor_max_candidates": 8,
                "tutor_profile_count": 5,
                "tutor_safety_shield_enabled": True,
            },
        },
        {
            "condition_name": "inverse_plan_full",
            "tutor_name": "inverse_plan_full",
            "baseline_mode": "mortal",
            "seeds": DEFAULT_SEEDS,
            "config_overrides": {
                **base_config,
                "tutor_rollout_horizon": 5,
                "tutor_top_k_paths": 3,
                "tutor_max_candidates": 10,
                "tutor_profile_count": 5,
                "tutor_safety_shield_enabled": True,
            },
        },
    ]


def run_phase12_batch(
    *,
    spec_name: str = "HugeRiskyGemMaze_v0",
    out_dir: str | Path = "runs/phase12_batch",
    workers: int = 16,
    teach_task_ids: list[str] | None = None,
    eval_task_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    teach_task_ids = list(teach_task_ids or DEFAULT_TEACH_TASKS)
    eval_task_ids = list(eval_task_ids or DEFAULT_EVAL_TASKS)
    conditions = default_conditions()

    batch_manifest = {
        "spec_name": spec_name,
        "workers": workers,
        "teach_task_ids": teach_task_ids,
        "eval_task_ids": eval_task_ids,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "conditions": conditions,
    }
    (out_path / "batch_config.json").write_text(json.dumps(batch_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    jobs: list[dict[str, Any]] = []
    results_by_condition: dict[str, list[dict[str, Any]]] = {str(c["condition_name"]): [] for c in conditions}
    for condition in conditions:
        condition_name = str(condition["condition_name"])
        run_dir = out_path / condition_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "trajectories").mkdir(parents=True, exist_ok=True)
        _write_json(
            run_dir / "config.json",
            {
                "spec_name": spec_name,
                "tutor_name": str(condition["tutor_name"]),
                "baseline_mode": str(condition["baseline_mode"]),
                "seeds": list(condition["seeds"]),
                "workers": int(workers),
                "teach_task_ids": teach_task_ids,
                "eval_task_ids": eval_task_ids,
                "condition_name": condition_name,
                "config_overrides": dict(condition["config_overrides"]),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        for seed in condition["seeds"]:
            jobs.append(
                {
                    "spec_name": spec_name,
                    "tutor_name": str(condition["tutor_name"]),
                    "baseline_mode": str(condition["baseline_mode"]),
                    "seed": int(seed),
                    "teach_task_ids": teach_task_ids,
                    "eval_task_ids": eval_task_ids,
                    "config_overrides": dict(condition["config_overrides"]),
                    "condition_name": condition_name,
                }
            )

    total_jobs = len(jobs)
    print(
        f"[run_phase12_batch] start conditions={len(conditions)} blocks={total_jobs} workers={workers}",
        flush=True,
    )

    if workers <= 1 or total_jobs <= 1:
        for idx, job in enumerate(jobs, start=1):
            result = _run_seed_job(job)
            condition_name = str(job["condition_name"])
            results_by_condition[condition_name].append(result)
            print(
                f"[run_phase12_batch] block {idx}/{total_jobs} "
                f"{_progress_bar(idx, total_jobs)} "
                f"condition={condition_name} seed={job['seed']} "
                f"elapsed={result.get('elapsed_sec', 0.0):.2f}s",
                flush=True,
            )
    else:
        future_map = {}
        try:
            with ProcessPoolExecutor(max_workers=min(workers, total_jobs)) as pool:
                for job in jobs:
                    future = pool.submit(_run_seed_job, job)
                    future_map[future] = job
                for idx, future in enumerate(as_completed(future_map), start=1):
                    job = future_map[future]
                    result = future.result()
                    condition_name = str(job["condition_name"])
                    results_by_condition[condition_name].append(result)
                    print(
                        f"[run_phase12_batch] block {idx}/{total_jobs} "
                        f"{_progress_bar(idx, total_jobs)} "
                        f"condition={condition_name} seed={job['seed']} "
                        f"elapsed={result.get('elapsed_sec', 0.0):.2f}s",
                        flush=True,
                    )
        except PermissionError:
            with ThreadPoolExecutor(max_workers=min(workers, total_jobs)) as pool:
                for job in jobs:
                    future = pool.submit(_run_seed_job, job)
                    future_map[future] = job
                for idx, future in enumerate(as_completed(future_map), start=1):
                    job = future_map[future]
                    result = future.result()
                    condition_name = str(job["condition_name"])
                    results_by_condition[condition_name].append(result)
                    print(
                        f"[run_phase12_batch] block {idx}/{total_jobs} "
                        f"{_progress_bar(idx, total_jobs)} "
                        f"condition={condition_name} seed={job['seed']} "
                        f"elapsed={result.get('elapsed_sec', 0.0):.2f}s",
                        flush=True,
                    )

    all_summaries: list[dict[str, Any]] = []
    total_conditions = len(conditions)
    for idx, condition in enumerate(conditions, start=1):
        condition_name = str(condition["condition_name"])
        run_dir = out_path / condition_name
        condition_results = sorted(results_by_condition[condition_name], key=lambda row: row["seed"])
        summary, _episodes = _finalize_experiment_outputs(
            out_path=run_dir,
            results=condition_results,
            spec_name=spec_name,
            tutor_name=str(condition["tutor_name"]),
            baseline_mode=str(condition["baseline_mode"]),
            condition_name=condition_name,
            run_config={
                "spec_name": spec_name,
                "tutor_name": str(condition["tutor_name"]),
                "baseline_mode": str(condition["baseline_mode"]),
                "seeds": list(condition["seeds"]),
                "workers": int(workers),
                "teach_task_ids": teach_task_ids,
                "eval_task_ids": eval_task_ids,
                "condition_name": condition_name,
                "config_overrides": dict(condition["config_overrides"]),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        all_summaries.append(dict(summary))
        print(
            f"[run_phase12_batch] condition {idx}/{total_conditions} done "
            f"{_progress_bar(idx, total_conditions, width=18)} "
            f"{condition_name} teach_success={summary.get('teach_success_rate')} "
            f"eval_success={summary.get('eval_success_rate')}",
            flush=True,
        )

    _write_csv(out_path / "phase12_summary.csv", all_summaries)
    _write_readme(out_path / "README.md", spec_name, workers, teach_task_ids, eval_task_ids, all_summaries)
    return all_summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Phase 1/2 fixed-maze comparison batch.")
    parser.add_argument("--spec", default="HugeRiskyGemMaze_v0")
    parser.add_argument("--out", default="runs/phase12_batch")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--teach-task-ids", nargs="*", default=None)
    parser.add_argument("--eval-task-ids", nargs="*", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_phase12_batch(
        spec_name=args.spec,
        out_dir=args.out,
        workers=args.workers,
        teach_task_ids=args.teach_task_ids,
        eval_task_ids=args.eval_task_ids,
    )
    return 0


def _write_readme(
    path: Path,
    spec_name: str,
    workers: int,
    teach_task_ids: list[str],
    eval_task_ids: list[str],
    summaries: list[dict[str, Any]],
) -> None:
    lines = [
        "# Phase 1/2 Batch",
        "",
        f"- spec: `{spec_name}`",
        f"- workers: `{workers}`",
        f"- teach_task_ids: `{teach_task_ids}`",
        f"- eval_task_ids: `{eval_task_ids}`",
        "",
        "## Conditions",
        "",
    ]
    for row in summaries:
        lines.append(
            f"- `{row['condition']}`: teach_success_rate=`{row.get('teach_success_rate')}`, "
            f"eval_success_rate=`{row.get('eval_success_rate')}`, warnings=`{row.get('warnings')}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
