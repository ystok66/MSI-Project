from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from risky_maze.experiments.run_fixed_maze import (
    _finalize_experiment_outputs,
    _progress_bar,
    _run_seed_job,
    _summarize_results,
    _write_csv,
    _write_json,
)


DEFAULT_SEEDS = list(range(16))


def _emit_progress(message: str, *, log_path: Path | None = None) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except Exception:
        pass
    if log_path is not None:
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
        except Exception:
            pass


def default_slices() -> list[dict[str, Any]]:
    return [
        {
            "slice_name": "t04_e05",
            "teach_task_ids": ["T04_CentralCore_to_SE_exit"],
            "eval_task_ids": ["E05_Central_to_SWgem_SE_exit"],
            "description": "central diagnostic / over-waypoint slice",
        },
        {
            "slice_name": "t08_e07",
            "teach_task_ids": ["T08_LongDiagonal_multi_gem"],
            "eval_task_ids": ["E07_East_to_WestGem_NE_exit"],
            "description": "long-horizon positive slice",
        },
        {
            "slice_name": "full_suite",
            "teach_task_ids": [
                "T01_NW_key_gem_NE_exit",
                "T02_NorthLab_to_NE_gem",
                "T03_WestGarden_to_SE_exit",
                "T04_CentralCore_to_SE_exit",
                "T05_EastFoundry_to_NE_exit",
                "T06_SWReservoir_to_SE_exit",
                "T07_SouthVault_key_door_exit",
                "T08_LongDiagonal_multi_gem",
            ],
            "eval_task_ids": [
                "E01_WestGarden_to_NE_exit",
                "E02_SE_to_NWgem_NE_exit",
                "E03_Foundry_to_NorthLab_exit",
                "E04_SW_to_EastGem_SE_exit",
                "E05_Central_to_SWgem_SE_exit",
                "E06_NW_to_SouthVaultGem_SE_exit",
                "E07_East_to_WestGem_NE_exit",
                "E08_SouthVault_to_NE_exit",
            ],
            "description": "full teach/eval suite",
        },
    ]


def representative_slices() -> list[dict[str, Any]]:
    wanted = {"t04_e05", "t08_e07"}
    return [row for row in default_slices() if str(row.get("slice_name")) in wanted]


def formal_matrix_conditions() -> list[dict[str, Any]]:
    base_config = {
        "risk_dim": 8,
        "obs_noise": 0.25,
        "cluster_std": 0.35,
        "view_radius": 2,
        "hp": 3,
        "n_safe_types": 3,
        "n_trap_types": 3,
        "tutor_rollout_horizon": 5,
        "tutor_top_k_paths": 3,
        "tutor_max_candidates": 10,
        "tutor_waypoint_cooldown_steps": 6,
        "tutor_max_waypoints_per_episode": 2,
        "tutor_profile_count": 5,
        "tutor_safety_shield_enabled": True,
        "warning_update_mode": "effective_sample",
        "record_step_details": True,
    }
    return [
        {
            "condition_name": "no_tutor_mortal",
            "tutor_name": "no_tutor",
            "baseline_mode": "mortal",
            "config_overrides": {
                **base_config,
                "record_step_details": False,
            },
        },
        {
            "condition_name": "no_tutor_immortal_warnlike",
            "tutor_name": "no_tutor",
            "baseline_mode": "immortal_warnlike",
            "config_overrides": {
                **base_config,
                "record_step_details": False,
            },
        },
        {
            "condition_name": "no_tutor_immortal_no_timeout",
            "tutor_name": "no_tutor",
            "baseline_mode": "immortal_no_timeout",
            "config_overrides": {
                **base_config,
                "record_step_details": False,
            },
        },
        {
            "condition_name": "always_warn_mortal",
            "tutor_name": "always_warn",
            "baseline_mode": "mortal",
            "config_overrides": dict(base_config),
        },
        {
            "condition_name": "inverse_plan_warn_only",
            "tutor_name": "inverse_plan_warn_only",
            "baseline_mode": "mortal",
            "config_overrides": dict(base_config),
        },
        {
            "condition_name": "always_waypoint_mortal",
            "tutor_name": "always_waypoint",
            "baseline_mode": "mortal",
            "config_overrides": dict(base_config),
        },
        {
            "condition_name": "inverse_plan_full",
            "tutor_name": "inverse_plan_full",
            "baseline_mode": "mortal",
            "config_overrides": dict(base_config),
        },
        {
            "condition_name": "inverse_plan_full_frontier_only",
            "tutor_name": "inverse_plan_full_frontier_only",
            "baseline_mode": "mortal",
            "config_overrides": {
                **base_config,
                "tutor_frontier_only_waypoint": True,
            },
        },
    ]


def representative_conditions() -> list[dict[str, Any]]:
    wanted = {
        "no_tutor_mortal",
        "no_tutor_immortal_warnlike",
        "always_warn_mortal",
        "inverse_plan_warn_only",
        "always_waypoint_mortal",
        "inverse_plan_full",
        "inverse_plan_full_frontier_only",
    }
    return [row for row in formal_matrix_conditions() if str(row.get("condition_name")) in wanted]


def run_formal_tutor_matrix(
    *,
    spec_name: str = "HugeRiskyGemMaze_v0",
    out_dir: str | Path = "runs/formal_tutor_matrix",
    workers: int = 16,
    seeds: list[int] | None = None,
    slices: list[dict[str, Any]] | None = None,
    conditions: list[dict[str, Any]] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    _configure_stdio_for_progress()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    progress_log_path = out_path / "progress.log"
    progress_log_path.write_text("", encoding="utf-8")
    seeds = list(seeds or DEFAULT_SEEDS)
    slices = list(slices or default_slices())
    conditions = list(conditions or formal_matrix_conditions())

    manifest = {
        "spec_name": spec_name,
        "workers": int(workers),
        "seeds": seeds,
        "slices": slices,
        "conditions": conditions,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(out_path / "matrix_config.json", manifest)

    jobs: list[dict[str, Any]] = []
    results_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    run_dir_by_key: dict[tuple[str, str], Path] = {}
    run_config_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    completed_rows: list[dict[str, Any]] = []
    for slice_spec in slices:
        slice_name = str(slice_spec["slice_name"])
        for condition in conditions:
            condition_name = str(condition["condition_name"])
            key = (slice_name, condition_name)
            results_by_key[key] = []
            run_dir = out_path / slice_name / condition_name
            run_dir_by_key[key] = run_dir
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "trajectories").mkdir(parents=True, exist_ok=True)
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            job_config_overrides = dict(condition["config_overrides"])
            if slice_name == "full_suite":
                job_config_overrides["record_step_details"] = False
            run_config = {
                "spec_name": spec_name,
                "slice_name": slice_name,
                "slice_description": str(slice_spec["description"]),
                "teach_task_ids": list(slice_spec["teach_task_ids"]),
                "eval_task_ids": list(slice_spec["eval_task_ids"]),
                "tutor_name": str(condition["tutor_name"]),
                "baseline_mode": str(condition["baseline_mode"]),
                "condition_name": condition_name,
                "config_overrides": job_config_overrides,
                "seeds": seeds,
                "workers": int(workers),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            run_config_by_key[key] = run_config
            _write_json(run_dir / "config.json", run_config)
            for seed in seeds:
                jobs.append(
                    {
                        "spec_name": spec_name,
                        "tutor_name": str(condition["tutor_name"]),
                        "baseline_mode": str(condition["baseline_mode"]),
                        "seed": int(seed),
                        "teach_task_ids": list(slice_spec["teach_task_ids"]),
                        "eval_task_ids": list(slice_spec["eval_task_ids"]),
                        "config_overrides": job_config_overrides,
                        "slice_name": slice_name,
                        "condition_name": condition_name,
                    }
                )

    total_jobs = len(jobs)
    started = time.perf_counter()
    if show_progress:
        _emit_progress(
            f"[run_formal_tutor_matrix] start slices={len(slices)} conditions={len(conditions)} blocks={total_jobs} workers={workers}",
            log_path=progress_log_path,
        )

    if workers <= 1 or total_jobs <= 1:
        for idx, job in enumerate(jobs, start=1):
            result = _run_seed_job(job)
            key = (str(job["slice_name"]), str(job["condition_name"]))
            results_by_key[key].append(result)
            _persist_incremental_group_outputs(
                out_path=out_path,
                run_dir=run_dir_by_key[key],
                spec_name=spec_name,
                tutor_name=str(job["tutor_name"]),
                baseline_mode=str(job["baseline_mode"]),
                condition_name=str(job["condition_name"]),
                slice_name=str(job["slice_name"]),
                result=result,
                results=results_by_key[key],
                completed_rows=completed_rows,
            )
            if show_progress:
                _print_block_progress(idx, total_jobs, job, result, log_path=progress_log_path)
    else:
        future_map: dict[Any, dict[str, Any]] = {}
        try:
            with ProcessPoolExecutor(max_workers=min(workers, total_jobs)) as pool:
                for submit_idx, job in enumerate(jobs, start=1):
                    future = pool.submit(_run_seed_job, job)
                    future_map[future] = job
                    if show_progress and (submit_idx == 1 or submit_idx % 32 == 0 or submit_idx == total_jobs):
                        _emit_progress(
                            f"[run_formal_tutor_matrix] submitted {_progress_bar(submit_idx, total_jobs)} "
                            f"submitted={submit_idx}/{total_jobs}",
                            log_path=progress_log_path,
                        )
                _consume_futures(
                    future_map=future_map,
                    results_by_key=results_by_key,
                    run_dir_by_key=run_dir_by_key,
                    run_config_by_key=run_config_by_key,
                    completed_rows=completed_rows,
                    out_path=out_path,
                    spec_name=spec_name,
                    total_jobs=total_jobs,
                    started=started,
                    show_progress=show_progress,
                    log_path=progress_log_path,
                )
        except PermissionError:
            with ThreadPoolExecutor(max_workers=min(workers, total_jobs)) as pool:
                for submit_idx, job in enumerate(jobs, start=1):
                    future = pool.submit(_run_seed_job, job)
                    future_map[future] = job
                    if show_progress and (submit_idx == 1 or submit_idx % 32 == 0 or submit_idx == total_jobs):
                        _emit_progress(
                            f"[run_formal_tutor_matrix] submitted {_progress_bar(submit_idx, total_jobs)} "
                            f"submitted={submit_idx}/{total_jobs}",
                            log_path=progress_log_path,
                        )
                _consume_futures(
                    future_map=future_map,
                    results_by_key=results_by_key,
                    run_dir_by_key=run_dir_by_key,
                    run_config_by_key=run_config_by_key,
                    completed_rows=completed_rows,
                    out_path=out_path,
                    spec_name=spec_name,
                    total_jobs=total_jobs,
                    started=started,
                    show_progress=show_progress,
                    log_path=progress_log_path,
                )

    matrix_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    total_groups = len(slices) * len(conditions)
    done_groups = 0
    for slice_spec in slices:
        slice_name = str(slice_spec["slice_name"])
        for condition in conditions:
            done_groups += 1
            condition_name = str(condition["condition_name"])
            key = (slice_name, condition_name)
            run_dir = out_path / slice_name / condition_name
            results = sorted(results_by_key[key], key=lambda row: row["seed"])
            finalize_config_overrides = dict(condition["config_overrides"])
            if slice_name == "full_suite":
                finalize_config_overrides["record_step_details"] = False
            summary, episodes = _finalize_experiment_outputs(
                out_path=run_dir,
                results=results,
                spec_name=spec_name,
                tutor_name=str(condition["tutor_name"]),
                baseline_mode=str(condition["baseline_mode"]),
                condition_name=f"{slice_name}__{condition_name}",
                run_config={
                    "spec_name": spec_name,
                    "slice_name": slice_name,
                    "slice_description": str(slice_spec["description"]),
                    "teach_task_ids": list(slice_spec["teach_task_ids"]),
                    "eval_task_ids": list(slice_spec["eval_task_ids"]),
                    "tutor_name": str(condition["tutor_name"]),
                    "baseline_mode": str(condition["baseline_mode"]),
                    "condition_name": condition_name,
                    "config_overrides": finalize_config_overrides,
                    "seeds": seeds,
                    "workers": int(workers),
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            row = dict(summary)
            row["slice_name"] = slice_name
            row["slice_description"] = str(slice_spec["description"])
            row["condition_name"] = condition_name
            matrix_rows.append(row)
            behavior = _summarize_tutor_behavior(results, episodes)
            behavior["slice_name"] = slice_name
            behavior["slice_description"] = str(slice_spec["description"])
            behavior["condition_name"] = condition_name
            behavior["tutor_name"] = str(condition["tutor_name"])
            behavior["baseline_mode"] = str(condition["baseline_mode"])
            behavior_rows.append(behavior)
            if show_progress:
                _emit_progress(
                    f"[run_formal_tutor_matrix] finalize {_progress_bar(done_groups, total_groups, width=18)} "
                    f"slice={slice_name} condition={condition_name} "
                    f"teach_success={row.get('teach_success_rate')} eval_success={row.get('eval_success_rate')}",
                    log_path=progress_log_path,
                )

    _write_csv(out_path / "matrix_summary.csv", matrix_rows)
    _write_csv(out_path / "matrix_tutor_behavior.csv", behavior_rows)
    _write_report(out_path / "FORMAL_TUTOR_MATRIX_REPORT.md", spec_name, workers, seeds, slices, conditions, matrix_rows)
    return {"summary": matrix_rows, "behavior": behavior_rows}


def _summarize_tutor_behavior(results: list[dict[str, Any]], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    teach_episodes = [ep for ep in episodes if str(ep.get("phase", "")) == "teach"]
    decisions = [row for ep in teach_episodes for row in ep.get("tutor_decisions", [])]
    total_decisions = len(decisions)
    warning_selected = sum(1 for row in decisions if str(row.get("selected_action", "")).upper() == "WARNING")
    wait_selected = sum(1 for row in decisions if str(row.get("selected_action", "")).upper() == "WAIT")
    waypoint_selected = sum(1 for row in decisions if str(row.get("selected_action", "")).upper() == "WAYPOINT")
    return {
        "teach_episode_count": len(teach_episodes),
        "teach_decision_count": total_decisions,
        "warning_selected_count": warning_selected,
        "wait_selected_count": wait_selected,
        "waypoint_selected_count": waypoint_selected,
        "warning_selected_rate": _safe_div(warning_selected, total_decisions),
        "wait_selected_rate": _safe_div(wait_selected, total_decisions),
        "waypoint_selected_rate": _safe_div(waypoint_selected, total_decisions),
        "warnings_selected_per_teach_episode": _safe_div(warning_selected, len(teach_episodes)),
        "mean_q_wait": _mean_numeric(decisions, "q_wait"),
        "mean_q_best_warning": _mean_numeric(decisions, "q_best_warning"),
        "mean_q_best_waypoint": _mean_numeric(decisions, "q_best_waypoint"),
        "mean_predicted_p_death_wait": _mean_numeric(decisions, "predicted_p_death_wait"),
        "mean_predicted_p_timeout_wait": _mean_numeric(decisions, "predicted_p_timeout_wait"),
        "mean_predicted_map_gain_wait": _mean_numeric(decisions, "predicted_map_gain_wait"),
        "mean_predicted_risk_ig_warning": _mean_numeric(decisions, "predicted_risk_ig_warning"),
        "mean_assist_leakage_per_decision": _mean_numeric(decisions, "assist_leakage"),
    }


def _print_block_progress(
    done: int,
    total: int,
    job: dict[str, Any],
    result: dict[str, Any],
    *,
    log_path: Path | None = None,
) -> None:
    aggregate = result.get("aggregate", {}) or {}
    _emit_progress(
        f"[run_formal_tutor_matrix] {_progress_bar(done, total)} "
        f"slice={job['slice_name']} condition={job['condition_name']} seed={job['seed']} "
        f"elapsed={float(result.get('elapsed_sec', 0.0)):.2f}s "
        f"teach_success={aggregate.get('teach_success_rate')} eval_success={aggregate.get('eval_success_rate')}",
        log_path=log_path,
    )


def _consume_futures(
    *,
    future_map: dict[Any, dict[str, Any]],
    results_by_key: dict[tuple[str, str], list[dict[str, Any]]],
    run_dir_by_key: dict[tuple[str, str], Path],
    run_config_by_key: dict[tuple[str, str], dict[str, Any]],
    completed_rows: list[dict[str, Any]],
    out_path: Path,
    spec_name: str,
    total_jobs: int,
    started: float,
    show_progress: bool,
    log_path: Path | None = None,
    heartbeat_sec: float = 15.0,
) -> None:
    pending = set(future_map)
    done_count = 0
    while pending:
        finished, pending = wait(pending, timeout=heartbeat_sec, return_when=FIRST_COMPLETED)
        if not finished:
            if show_progress:
                _emit_progress(
                    f"[run_formal_tutor_matrix] heartbeat {_progress_bar(done_count, total_jobs)} "
                    f"done={done_count}/{total_jobs} pending={len(pending)} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    log_path=log_path,
                )
            continue
        for future in finished:
            job = future_map[future]
            result = future.result()
            done_count += 1
            key = (str(job["slice_name"]), str(job["condition_name"]))
            results_by_key[key].append(result)
            _persist_incremental_group_outputs(
                out_path=out_path,
                run_dir=run_dir_by_key[key],
                spec_name=spec_name,
                tutor_name=str(job["tutor_name"]),
                baseline_mode=str(job["baseline_mode"]),
                condition_name=str(job["condition_name"]),
                slice_name=str(job["slice_name"]),
                result=result,
                results=results_by_key[key],
                completed_rows=completed_rows,
            )
            if show_progress:
                _print_block_progress(done_count, total_jobs, job, result, log_path=log_path)


def _configure_stdio_for_progress() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True, write_through=True)
            except Exception:
                pass


def _persist_incremental_group_outputs(
    *,
    out_path: Path,
    run_dir: Path,
    spec_name: str,
    tutor_name: str,
    baseline_mode: str,
    condition_name: str,
    slice_name: str,
    result: dict[str, Any],
    results: list[dict[str, Any]],
    completed_rows: list[dict[str, Any]],
) -> None:
    seed = int(result["seed"])
    checkpoint_payload = {
        "slice_name": slice_name,
        "condition_name": condition_name,
        "spec_name": spec_name,
        "tutor_name": tutor_name,
        "baseline_mode": baseline_mode,
        **result,
    }
    _write_json(run_dir / "checkpoints" / f"seed_{seed:03d}.json", checkpoint_payload)

    aggregate = dict(result.get("aggregate", {}) or {})
    completed_rows[:] = [
        row
        for row in completed_rows
        if not (
            row.get("slice_name") == slice_name
            and row.get("condition_name") == condition_name
            and int(row.get("seed", -1)) == seed
        )
    ]
    completed_rows.append(
        {
            "slice_name": slice_name,
            "condition_name": condition_name,
            "spec_name": spec_name,
            "tutor_name": tutor_name,
            "baseline_mode": baseline_mode,
            "seed": seed,
            "elapsed_sec": result.get("elapsed_sec"),
            **aggregate,
        }
    )
    completed_rows.sort(key=lambda row: (str(row.get("slice_name")), str(row.get("condition_name")), int(row.get("seed", 0))))
    _write_csv(out_path / "completed_blocks.csv", completed_rows)

    seed_rows = [row["aggregate"] | {"seed": row["seed"]} for row in sorted(results, key=lambda item: item["seed"])]
    _write_csv(run_dir / "partial_seed_summary.csv", seed_rows)
    partial_summary = _summarize_results(
        results=results,
        spec_name=spec_name,
        tutor_name=tutor_name,
        baseline_mode=baseline_mode,
        condition_name=f"{slice_name}__{condition_name}",
    )
    _write_csv(run_dir / "partial_summary.csv", [partial_summary])
    _write_json(
        run_dir / "partial_status.json",
        {
            "slice_name": slice_name,
            "condition_name": condition_name,
            "completed_seeds": [int(row["seed"]) for row in sorted(results, key=lambda item: item["seed"])],
            "completed_count": len(results),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    partial_matrix_rows: list[dict[str, Any]] = []
    for partial_path in sorted(out_path.rglob("partial_summary.csv")):
        try:
            import csv

            with partial_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    partial_matrix_rows.append(row)
        except Exception:
            continue
    if partial_matrix_rows:
        _write_csv(out_path / "partial_matrix_summary.csv", partial_matrix_rows)


def _write_report(
    path: Path,
    spec_name: str,
    workers: int,
    seeds: list[int],
    slices: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Formal Tutor Matrix",
        "",
        f"- generated_at_utc: `{datetime.now(timezone.utc).isoformat()}`",
        f"- spec: `{spec_name}`",
        f"- workers: `{workers}`",
        f"- seeds: `{seeds}`",
        "",
        "## Slices",
        "",
    ]
    for slice_spec in slices:
        lines.append(
            f"- `{slice_spec['slice_name']}`: teach=`{slice_spec['teach_task_ids']}` eval=`{slice_spec['eval_task_ids']}` ({slice_spec['description']})"
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
            "| slice | condition | teach_success | eval_success | eval_regret | warnings | waypoints | warning_ig | assist_leakage |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + f"{row['slice_name']} | "
            + f"{row['condition_name']} | "
            + f"{_fmt(row.get('teach_success_rate'))} | "
            + f"{_fmt(row.get('eval_success_rate'))} | "
            + f"{_fmt(row.get('eval_regret_to_oracle_safe_path'))} | "
            + f"{_fmt(row.get('teach_mean_warnings'))} | "
            + f"{_fmt(row.get('teach_mean_waypoints'))} | "
            + f"{_fmt(row.get('warning_information_gain'))} | "
            + f"{_fmt(row.get('assist_leakage'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_div(a: float, b: float) -> float:
    return float(a) / max(1.0, float(b))


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the formal T04/T08/full-suite tutor comparison matrix.")
    parser.add_argument("--spec", default="HugeRiskyGemMaze_v0")
    parser.add_argument("--out", default="runs/formal_tutor_matrix")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--slices", nargs="*", default=None, choices=[s["slice_name"] for s in default_slices()])
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        choices=[c["condition_name"] for c in formal_matrix_conditions()],
    )
    parser.add_argument(
        "--representative",
        action="store_true",
        help="Run only the representative T04/T08 slices with a smaller representative condition set.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    slices = representative_slices() if args.representative else default_slices()
    if args.slices:
        selected = set(args.slices)
        slices = [s for s in slices if s["slice_name"] in selected]
    conditions = representative_conditions() if args.representative else formal_matrix_conditions()
    if args.conditions:
        selected_conditions = set(args.conditions)
        conditions = [c for c in conditions if c["condition_name"] in selected_conditions]
    run_formal_tutor_matrix(
        spec_name=args.spec,
        out_dir=args.out,
        workers=args.workers,
        seeds=list(args.seeds) if args.seeds else None,
        slices=slices,
        conditions=conditions,
    )
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
