from __future__ import annotations

import argparse
import multiprocessing
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from risky_maze.env.fixed_loader import load_fixed_spec
from risky_maze.experiments.run_fixed_maze import (
    _finalize_experiment_outputs,
    _progress_bar,
    _run_seed_job,
    _summarize_results,
    _write_csv,
    _write_json,
)
from risky_maze.experiments.run_formal_tutor_matrix import (
    DEFAULT_SEEDS,
    _configure_stdio_for_progress,
    _emit_progress,
    _persist_incremental_group_outputs,
    _print_block_progress,
    _summarize_tutor_behavior,
)
from risky_maze.scenarios.tutor_didactic_maze_suite_v1 import TUTOR_DIDACTIC_MAZE_SUITE_V1


DIDACTIC_VARIANTS = ("base", "autonomy_hard", "transfer_hard", "didactic_hard")


def _variant_overrides(spec_name: str, variant: str) -> tuple[dict[str, Any], str]:
    key = str(variant or "base").lower()
    if key == "base":
        return {}, ""
    if key in {"autonomy_hard", "didactic_hard"} and spec_name == "TutorAutonomyLoop_v1":
        return (
            {
                "eval_time_limit_scale": 0.80,
                "learner_unknown_penalty": 0.35,
                "learner_long_term_memory_weight": 0.55,
                "tutor_max_waypoints_per_episode": 1,
            },
            "__autonomy_hard",
        )
    if key in {"transfer_hard", "didactic_hard"} and spec_name == "TutorPrincipleDoorTransfer_v1":
        return (
            {
                "teach_time_limit_scale": 0.85,
                "eval_time_limit_scale": 0.75,
                "learner_unknown_penalty": 0.40,
                "learner_long_term_memory_weight": 0.60,
            },
            "__transfer_hard",
        )
    return {}, ""


def didactic_maps(*, variant: str = "base") -> list[dict[str, Any]]:
    maps: list[dict[str, Any]] = []
    for raw in list(TUTOR_DIDACTIC_MAZE_SUITE_V1.get("maps", ())):
        spec_name = str(raw["name"])
        spec = load_fixed_spec(spec_name)
        tasks = dict(raw.get("tasks", {}) or {})
        recommended = dict(raw.get("recommended_config", {}) or {})
        noise = dict((raw.get("risk_feature_spec", {}) or {}).get("suggested_noise", {}) or {})
        variant_cfg, suffix = _variant_overrides(spec_name, variant)
        map_name = f"{spec_name}{suffix}" if suffix else spec_name
        description = str(raw.get("purpose", ""))
        if suffix:
            description = f"{description} [{suffix.removeprefix('__')}]"
        maps.append(
            {
                "map_name": map_name,
                "spec_name": spec_name,
                "description": description,
                "teach_task_ids": list(spec.teach_tasks.keys()) if spec.teach_tasks else list(tasks.get("teach", [])),
                "eval_task_ids": list(spec.eval_tasks.keys()) if spec.eval_tasks else list(tasks.get("eval_same_map_no_tutor", [])),
                "recommended_config": {
                    "risk_dim": int((raw.get("risk_feature_spec", {}) or {}).get("feature_dim", 12)),
                    "obs_noise": float(noise.get("obs_sigma", 0.40)),
                    "cluster_std": float(noise.get("cluster_sigma", 0.45)),
                    "view_radius": 2,
                    "hp": int(recommended.get("hp", 3)),
                    "n_safe_types": 3,
                    "n_trap_types": 3,
                    "tutor_rollout_horizon": 5,
                    "tutor_top_k_paths": 2,
                    "tutor_max_candidates": 8,
                    "tutor_waypoint_cooldown_steps": 6,
                    "tutor_max_waypoints_per_episode": 2,
                    "tutor_profile_count": 3,
                    "tutor_safety_shield_enabled": True,
                    "warning_update_mode": "effective_sample",
                    "learner_consolidation_mode": str(recommended.get("consolidation", "success_gated_assist_discounted")),
                    "learner_long_term_memory_weight": float(recommended.get("long_term_memory_weight", 0.35)),
                    "learner_autonomy_assist_discount": float(recommended.get("assist_discount", 0.35)),
                    "learner_enable_objective_learning_events": bool(
                        recommended.get("enable_objective_learning_events", True)
                    ),
                    "learner_use_long_term_route_graph": bool(recommended.get("use_long_term_route_graph", True)),
                    "learner_use_landmark_graph": bool(recommended.get("use_landmark_graph", True)),
                    "record_step_details": True,
                    **variant_cfg,
                },
            }
        )
    return maps


def didactic_conditions() -> list[dict[str, Any]]:
    return [
        {
            "condition_name": "no_tutor_mortal",
            "tutor_name": "no_tutor",
            "baseline_mode": "mortal",
            "config_overrides": {"record_step_details": False},
        },
        {
            "condition_name": "no_tutor_immortal_warnlike",
            "tutor_name": "no_tutor",
            "baseline_mode": "immortal_warnlike",
            "config_overrides": {"record_step_details": False},
        },
        {
            "condition_name": "always_warn_mortal",
            "tutor_name": "always_warn",
            "baseline_mode": "mortal",
            "config_overrides": {},
        },
        {
            "condition_name": "safety_shield_only",
            "tutor_name": "safety_shield_only",
            "baseline_mode": "mortal",
            "config_overrides": {},
        },
        {
            "condition_name": "shield_plus_minimal_waypoint",
            "tutor_name": "shield_plus_minimal_waypoint",
            "baseline_mode": "mortal",
            "config_overrides": {},
        },
        {
            "condition_name": "shield_plus_frontier_waypoint",
            "tutor_name": "shield_plus_frontier_waypoint",
            "baseline_mode": "mortal",
            "config_overrides": {},
        },
        {
            "condition_name": "always_waypoint_mortal",
            "tutor_name": "always_waypoint",
            "baseline_mode": "mortal",
            "config_overrides": {},
        },
        {
            "condition_name": "shield_plus_oracle_when_needed",
            "tutor_name": "shield_plus_oracle_when_needed",
            "baseline_mode": "mortal",
            "config_overrides": {},
        },
    ]


def run_didactic_tutor_suite(
    *,
    out_dir: str | Path = "runs/didactic_tutor_suite",
    workers: int = 16,
    seeds: list[int] | None = None,
    maps: list[dict[str, Any]] | None = None,
    conditions: list[dict[str, Any]] | None = None,
    variant: str = "base",
    show_progress: bool = True,
) -> dict[str, Any]:
    _configure_stdio_for_progress()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    progress_log_path = out_path / "progress.log"
    progress_log_path.write_text("", encoding="utf-8")
    seeds = list(seeds or DEFAULT_SEEDS[:4])
    maps = list(maps or didactic_maps(variant=variant))
    conditions = list(conditions or didactic_conditions())

    manifest = {
        "suite_name": "TutorDidacticMazeSuite_v1",
        "variant": variant,
        "workers": int(workers),
        "seeds": seeds,
        "maps": maps,
        "conditions": conditions,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(out_path / "suite_config.json", manifest)

    jobs: list[dict[str, Any]] = []
    results_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    run_dir_by_key: dict[tuple[str, str], Path] = {}
    completed_rows: list[dict[str, Any]] = []

    for map_spec in maps:
        map_name = str(map_spec["map_name"])
        base_cfg = dict(map_spec.get("recommended_config", {}) or {})
        for condition in conditions:
            condition_name = str(condition["condition_name"])
            key = (map_name, condition_name)
            results_by_key[key] = []
            run_dir = out_path / map_name / condition_name
            run_dir_by_key[key] = run_dir
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "trajectories").mkdir(parents=True, exist_ok=True)
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            run_config = {
                "suite_name": "TutorDidacticMazeSuite_v1",
                "map_name": map_name,
                "spec_name": str(map_spec["spec_name"]),
                "map_description": str(map_spec["description"]),
                "teach_task_ids": list(map_spec["teach_task_ids"]),
                "eval_task_ids": list(map_spec["eval_task_ids"]),
                "tutor_name": str(condition["tutor_name"]),
                "baseline_mode": str(condition["baseline_mode"]),
                "condition_name": condition_name,
                "config_overrides": {**base_cfg, **dict(condition.get("config_overrides", {}) or {})},
                "seeds": seeds,
                "workers": int(workers),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _write_json(run_dir / "config.json", run_config)
            for seed in seeds:
                jobs.append(
                    {
                        "spec_name": str(map_spec["spec_name"]),
                        "tutor_name": str(condition["tutor_name"]),
                        "baseline_mode": str(condition["baseline_mode"]),
                        "seed": int(seed),
                        "teach_task_ids": list(map_spec["teach_task_ids"]),
                        "eval_task_ids": list(map_spec["eval_task_ids"]),
                        "config_overrides": {**base_cfg, **dict(condition.get("config_overrides", {}) or {})},
                        "map_name": map_name,
                        "condition_name": condition_name,
                    }
                )

    total_jobs = len(jobs)
    started = time.perf_counter()
    if show_progress:
        _emit_progress(
            f"[run_didactic_tutor_suite] start maps={len(maps)} conditions={len(conditions)} blocks={total_jobs} workers={workers}",
            log_path=progress_log_path,
        )

    if workers <= 1 or total_jobs <= 1:
        for idx, job in enumerate(jobs, start=1):
            result = _run_seed_job(job)
            key = (str(job["map_name"]), str(job["condition_name"]))
            results_by_key[key].append(result)
            _persist_incremental_group_outputs(
                out_path=out_path,
                run_dir=run_dir_by_key[key],
                spec_name=str(job["spec_name"]),
                tutor_name=str(job["tutor_name"]),
                baseline_mode=str(job["baseline_mode"]),
                condition_name=str(job["condition_name"]),
                slice_name=str(job["map_name"]),
                result=result,
                results=results_by_key[key],
                completed_rows=completed_rows,
            )
            if show_progress:
                _emit_progress(
                    f"[run_didactic_tutor_suite] {_progress_bar(idx, total_jobs)} "
                    f"map={job['map_name']} condition={job['condition_name']} seed={job['seed']} "
                    f"elapsed={float(result.get('elapsed_sec', 0.0)):.2f}s "
                    f"teach_success={result.get('aggregate', {}).get('teach_success_rate')} "
                    f"eval_success={result.get('aggregate', {}).get('eval_success_rate')}",
                    log_path=progress_log_path,
                )
    else:
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait

        future_map: dict[Any, dict[str, Any]] = {}
        try:
            with ProcessPoolExecutor(max_workers=min(workers, total_jobs)) as pool:
                for submit_idx, job in enumerate(jobs, start=1):
                    future = pool.submit(_run_seed_job, job)
                    future_map[future] = job
                    if show_progress and (submit_idx == 1 or submit_idx % 24 == 0 or submit_idx == total_jobs):
                        _emit_progress(
                            f"[run_didactic_tutor_suite] submitted {_progress_bar(submit_idx, total_jobs)} "
                            f"submitted={submit_idx}/{total_jobs}",
                            log_path=progress_log_path,
                        )
                _consume_didactic_futures(
                    future_map=future_map,
                    results_by_key=results_by_key,
                    run_dir_by_key=run_dir_by_key,
                    completed_rows=completed_rows,
                    out_path=out_path,
                    total_jobs=total_jobs,
                    started=started,
                    show_progress=show_progress,
                    log_path=progress_log_path,
                    wait_fn=wait,
                    first_completed=FIRST_COMPLETED,
                )
        except PermissionError:
            with ThreadPoolExecutor(max_workers=min(workers, total_jobs)) as pool:
                for submit_idx, job in enumerate(jobs, start=1):
                    future = pool.submit(_run_seed_job, job)
                    future_map[future] = job
                    if show_progress and (submit_idx == 1 or submit_idx % 24 == 0 or submit_idx == total_jobs):
                        _emit_progress(
                            f"[run_didactic_tutor_suite] submitted {_progress_bar(submit_idx, total_jobs)} "
                            f"submitted={submit_idx}/{total_jobs}",
                            log_path=progress_log_path,
                        )
                _consume_didactic_futures(
                    future_map=future_map,
                    results_by_key=results_by_key,
                    run_dir_by_key=run_dir_by_key,
                    completed_rows=completed_rows,
                    out_path=out_path,
                    total_jobs=total_jobs,
                    started=started,
                    show_progress=show_progress,
                    log_path=progress_log_path,
                    wait_fn=wait,
                    first_completed=FIRST_COMPLETED,
                )

    matrix_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    total_groups = len(maps) * len(conditions)
    done_groups = 0
    for map_spec in maps:
        map_name = str(map_spec["map_name"])
        for condition in conditions:
            done_groups += 1
            condition_name = str(condition["condition_name"])
            key = (map_name, condition_name)
            results = sorted(results_by_key[key], key=lambda row: row["seed"])
            config_overrides = {
                **dict(map_spec.get("recommended_config", {}) or {}),
                **dict(condition.get("config_overrides", {}) or {}),
            }
            summary, episodes = _finalize_experiment_outputs(
                out_path=run_dir_by_key[key],
                results=results,
                spec_name=str(map_spec["spec_name"]),
                tutor_name=str(condition["tutor_name"]),
                baseline_mode=str(condition["baseline_mode"]),
                condition_name=f"{map_name}__{condition_name}",
                run_config={
                    "suite_name": "TutorDidacticMazeSuite_v1",
                    "map_name": map_name,
                    "spec_name": str(map_spec["spec_name"]),
                    "map_description": str(map_spec["description"]),
                    "teach_task_ids": list(map_spec["teach_task_ids"]),
                    "eval_task_ids": list(map_spec["eval_task_ids"]),
                    "tutor_name": str(condition["tutor_name"]),
                    "baseline_mode": str(condition["baseline_mode"]),
                    "condition_name": condition_name,
                    "config_overrides": config_overrides,
                    "seeds": seeds,
                    "workers": int(workers),
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            row = dict(summary)
            row["map_name"] = map_name
            row["map_description"] = str(map_spec["description"])
            row["condition_name"] = condition_name
            matrix_rows.append(row)

            behavior = _summarize_tutor_behavior(results, episodes)
            behavior["map_name"] = map_name
            behavior["map_description"] = str(map_spec["description"])
            behavior["condition_name"] = condition_name
            behavior["tutor_name"] = str(condition["tutor_name"])
            behavior["baseline_mode"] = str(condition["baseline_mode"])
            behavior_rows.append(behavior)
            if show_progress:
                _emit_progress(
                    f"[run_didactic_tutor_suite] finalize {_progress_bar(done_groups, total_groups, width=18)} "
                    f"map={map_name} condition={condition_name} "
                    f"teach_success={row.get('teach_success_rate')} eval_success={row.get('eval_success_rate')}",
                    log_path=progress_log_path,
                )

    _write_csv(out_path / "matrix_summary.csv", matrix_rows)
    _write_csv(out_path / "matrix_tutor_behavior.csv", behavior_rows)
    _write_didactic_report(
        path=out_path / "DIDACTIC_TUTOR_SUITE_REPORT.md",
        workers=workers,
        seeds=seeds,
        maps=maps,
        conditions=conditions,
        rows=matrix_rows,
    )
    return {"summary": matrix_rows, "behavior": behavior_rows}


def _consume_didactic_futures(
    *,
    future_map: dict[Any, dict[str, Any]],
    results_by_key: dict[tuple[str, str], list[dict[str, Any]]],
    run_dir_by_key: dict[tuple[str, str], Path],
    completed_rows: list[dict[str, Any]],
    out_path: Path,
    total_jobs: int,
    started: float,
    show_progress: bool,
    log_path: Path | None,
    wait_fn: Any,
    first_completed: Any,
    heartbeat_sec: float = 15.0,
) -> None:
    pending = set(future_map)
    done_count = 0
    while pending:
        finished, pending = wait_fn(pending, timeout=heartbeat_sec, return_when=first_completed)
        if not finished:
            if show_progress:
                _emit_progress(
                    f"[run_didactic_tutor_suite] heartbeat {_progress_bar(done_count, total_jobs)} "
                    f"done={done_count}/{total_jobs} pending={len(pending)} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    log_path=log_path,
                )
            continue
        for future in finished:
            job = future_map[future]
            result = future.result()
            done_count += 1
            key = (str(job["map_name"]), str(job["condition_name"]))
            results_by_key[key].append(result)
            _persist_incremental_group_outputs(
                out_path=out_path,
                run_dir=run_dir_by_key[key],
                spec_name=str(job["spec_name"]),
                tutor_name=str(job["tutor_name"]),
                baseline_mode=str(job["baseline_mode"]),
                condition_name=str(job["condition_name"]),
                slice_name=str(job["map_name"]),
                result=result,
                results=results_by_key[key],
                completed_rows=completed_rows,
            )
            if show_progress:
                _emit_progress(
                    f"[run_didactic_tutor_suite] {_progress_bar(done_count, total_jobs)} "
                    f"map={job['map_name']} condition={job['condition_name']} seed={job['seed']} "
                    f"elapsed={float(result.get('elapsed_sec', 0.0)):.2f}s "
                    f"teach_success={result.get('aggregate', {}).get('teach_success_rate')} "
                    f"eval_success={result.get('aggregate', {}).get('eval_success_rate')}",
                    log_path=log_path,
                )


def _write_didactic_report(
    *,
    path: Path,
    workers: int,
    seeds: list[int],
    maps: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# TutorDidacticMazeSuite_v1 Report",
        "",
        f"- generated_at_utc: `{datetime.now(timezone.utc).isoformat()}`",
        f"- workers: `{workers}`",
        f"- seeds: `{seeds}`",
        "",
        "## Maps",
        "",
    ]
    for map_spec in maps:
        lines.append(
            f"- `{map_spec['map_name']}`: teach=`{map_spec['teach_task_ids']}` "
            f"eval=`{map_spec['eval_task_ids']}` ({map_spec['description']})"
        )
    lines.extend(["", "## Conditions", ""])
    for condition in conditions:
        lines.append(
            f"- `{condition['condition_name']}`: tutor=`{condition['tutor_name']}`, baseline=`{condition['baseline_mode']}`"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| map | condition | teach_success | teach_safe_success | teach_cost | teach_sec | teach_sec_step | eval_success | eval_cost | eval_sec | eval_sec_step | assist_leakage | autonomy_credit | route_graph_conf | landmark_graph_conf | teach_commits | objective_events | map_reuse_eval | useful_exploration | warning_actionability | useful_wait | bad_wait | waypoints | warnings |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + f"{row['map_name']} | "
            + f"{row['condition_name']} | "
            + f"{_fmt(row.get('teach_success_rate'))} | "
            + f"{_fmt(row.get('teach_safe_success_rate'))} | "
            + f"{_fmt(row.get('teach_cost'))} | "
            + f"{_fmt(row.get('teach_mean_elapsed_seconds'))} | "
            + f"{_fmt(row.get('teach_mean_seconds_per_step'))} | "
            + f"{_fmt(row.get('eval_success_rate'))} | "
            + f"{_fmt(row.get('eval_cost'))} | "
            + f"{_fmt(row.get('eval_mean_elapsed_seconds'))} | "
            + f"{_fmt(row.get('eval_mean_seconds_per_step'))} | "
            + f"{_fmt(row.get('assist_leakage'))} | "
            + f"{_fmt(row.get('autonomy_credit'))} | "
            + f"{_fmt(row.get('route_graph_confidence'))} | "
            + f"{_fmt(row.get('landmark_graph_confidence'))} | "
            + f"{int(row.get('successful_teach_commits') or 0)}/{int(row.get('total_teach_commits') or 0)} | "
            + f"{_fmt(row.get('objective_learning_event_count'))} | "
            + f"{_fmt(row.get('map_reuse_eval'))} | "
            + f"{_fmt(row.get('useful_exploration_rate'))} | "
            + f"{_fmt(row.get('warning_actionability'))} | "
            + f"{_fmt(row.get('useful_wait_rate'))} | "
            + f"{_fmt(row.get('bad_wait_rate'))} | "
            + f"{_fmt(row.get('teach_mean_waypoints'))} | "
            + f"{_fmt(row.get('teach_mean_warnings'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TutorDidacticMazeSuite_v1 compact tutor comparisons.")
    parser.add_argument("--out", default="runs/didactic_tutor_suite")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--variant", default="base", choices=list(DIDACTIC_VARIANTS))
    parser.add_argument(
        "--maps",
        nargs="*",
        default=None,
        choices=sorted({m["map_name"] for v in DIDACTIC_VARIANTS for m in didactic_maps(variant=v)}),
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        choices=[c["condition_name"] for c in didactic_conditions()],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    maps = didactic_maps(variant=str(args.variant))
    if args.maps:
        selected = set(args.maps)
        maps = [m for m in maps if m["map_name"] in selected]
    conditions = didactic_conditions()
    if args.conditions:
        selected_conditions = set(args.conditions)
        conditions = [c for c in conditions if c["condition_name"] in selected_conditions]
    run_didactic_tutor_suite(
        out_dir=args.out,
        workers=args.workers,
        seeds=list(args.seeds) if args.seeds else None,
        maps=maps,
        conditions=conditions,
        variant=str(args.variant),
        show_progress=True,
    )
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
