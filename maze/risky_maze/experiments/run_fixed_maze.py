from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any, Iterable

from risky_maze.runner.fixed_block_runner import run_fixed_block_detailed


def run_fixed_experiment(
    *,
    spec_name: str,
    tutor_name: str,
    baseline_mode: str,
    seeds: list[int],
    out_dir: str | Path,
    workers: int = 1,
    teach_task_ids: list[str] | None = None,
    eval_task_ids: list[str] | None = None,
    config_overrides: dict[str, Any] | None = None,
    condition_name: str | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "trajectories").mkdir(parents=True, exist_ok=True)

    config_overrides = dict(config_overrides or {})
    condition_name = condition_name or f"{tutor_name}_{baseline_mode}"
    run_config = {
        "spec_name": spec_name,
        "tutor_name": tutor_name,
        "baseline_mode": baseline_mode,
        "seeds": list(seeds),
        "workers": int(workers),
        "teach_task_ids": list(teach_task_ids or []),
        "eval_task_ids": list(eval_task_ids or []),
        "condition_name": condition_name,
        "config_overrides": config_overrides,
        "show_progress": bool(show_progress),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    _write_json(out_path / "config.json", run_config)
    results = _run_seeds(
        spec_name=spec_name,
        tutor_name=tutor_name,
        baseline_mode=baseline_mode,
        seeds=seeds,
        workers=workers,
        teach_task_ids=teach_task_ids,
        eval_task_ids=eval_task_ids,
        config_overrides=config_overrides,
        show_progress=show_progress,
        progress_label=condition_name,
    )
    summary_row, episodes = _finalize_experiment_outputs(
        out_path=out_path,
        results=results,
        spec_name=spec_name,
        tutor_name=tutor_name,
        baseline_mode=baseline_mode,
        condition_name=condition_name,
        run_config=run_config,
    )

    return {
        "config": run_config,
        "summary": summary_row,
        "seed_results": results,
        "episodes": episodes,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed-map risky_maze experiments.")
    parser.add_argument("--spec", default="HugeRiskyGemMaze_v0")
    parser.add_argument("--tutor", default="no_tutor")
    parser.add_argument("--baseline", default="mortal")
    parser.add_argument("--out", required=True)
    parser.add_argument("--condition-name", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--teach-task-ids", nargs="*", default=None)
    parser.add_argument("--eval-task-ids", nargs="*", default=None)
    parser.add_argument("--risk-dim", type=int, default=8)
    parser.add_argument("--obs-noise", type=float, default=0.25)
    parser.add_argument("--cluster-std", type=float, default=0.35)
    parser.add_argument("--view-radius", type=int, default=2)
    parser.add_argument("--hp", type=int, default=3)
    parser.add_argument("--n-safe-types", type=int, default=3)
    parser.add_argument("--n-trap-types", type=int, default=3)
    parser.add_argument("--tutor-rollout-horizon", type=int, default=6)
    parser.add_argument("--tutor-top-k-paths", type=int, default=2)
    parser.add_argument("--tutor-max-candidates", type=int, default=8)
    parser.add_argument("--tutor-waypoint-cooldown-steps", type=int, default=6)
    parser.add_argument("--tutor-max-waypoints-per-episode", type=int, default=3)
    parser.add_argument("--tutor-profile-count", type=int, default=3)
    parser.add_argument("--tutor-safety-shield-enabled", action="store_true")
    parser.add_argument("--tutor-catastrophe-threshold", type=float, default=0.35)
    parser.add_argument("--tutor-catastrophe-damage-threshold", type=float, default=2.0)
    parser.add_argument("--tutor-frontier-only-waypoint", action="store_true")
    parser.add_argument("--learner-risk-weight", type=float, default=4.0)
    parser.add_argument("--learner-revisit-penalty", type=float, default=0.15)
    parser.add_argument("--learner-unknown-penalty", type=float, default=0.20)
    parser.add_argument("--learner-warning-suspicion-weight", type=float, default=2.0)
    parser.add_argument(
        "--learner-warning-suspicion-mode",
        choices=["persistent", "episode_decay", "replan_only", "query_only", "none"],
        default="persistent",
    )
    parser.add_argument("--learner-warning-suspicion-decay", type=float, default=1.0)
    parser.add_argument(
        "--learner-consolidation-mode",
        choices=["none", "always_commit", "success_gated", "success_gated_assist_discounted"],
        default="none",
    )
    parser.add_argument("--learner-long-term-memory-weight", type=float, default=0.35)
    parser.add_argument("--learner-autonomy-assist-discount", type=float, default=0.05)
    parser.add_argument("--learner-disable-objective-learning-events", action="store_true")
    parser.add_argument("--learner-disable-long-term-route-graph", action="store_true")
    parser.add_argument("--learner-disable-landmark-graph", action="store_true")
    parser.add_argument("--warning-update-mode", choices=["literal", "effective_sample"], default="effective_sample")
    parser.add_argument("--warning-eta0", type=float, default=0.35)
    parser.add_argument("--warning-kl-epsilon", type=float, default=1e-6)
    parser.add_argument("--tutor-warning-actionability-threshold", type=float, default=0.0)
    parser.add_argument("--tutor-waypoint-damage-veto-margin", type=float, default=float("inf"))
    parser.add_argument("--ablate-warning-update", action="store_true")
    parser.add_argument("--ablate-trap-risk-update", action="store_true")
    parser.add_argument("--ablate-safe-risk-update", action="store_true")
    parser.add_argument("--ablate-eval-clear-map-memory", action="store_true")
    parser.add_argument("--ablate-eval-clear-risk-belief", action="store_true")
    parser.add_argument("--ablate-eval-clear-warning-suspicion", action="store_true")
    parser.add_argument("--ablate-eval-clear-long-term-memory", action="store_true")
    parser.add_argument("--no-step-detail", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config_overrides = {
        "risk_dim": args.risk_dim,
        "obs_noise": args.obs_noise,
        "cluster_std": args.cluster_std,
        "view_radius": args.view_radius,
        "hp": args.hp,
        "n_safe_types": args.n_safe_types,
        "n_trap_types": args.n_trap_types,
        "tutor_rollout_horizon": args.tutor_rollout_horizon,
        "tutor_top_k_paths": args.tutor_top_k_paths,
        "tutor_max_candidates": args.tutor_max_candidates,
        "tutor_waypoint_cooldown_steps": args.tutor_waypoint_cooldown_steps,
        "tutor_max_waypoints_per_episode": args.tutor_max_waypoints_per_episode,
        "tutor_profile_count": args.tutor_profile_count,
        "tutor_safety_shield_enabled": args.tutor_safety_shield_enabled,
        "tutor_catastrophe_threshold": args.tutor_catastrophe_threshold,
        "tutor_catastrophe_damage_threshold": args.tutor_catastrophe_damage_threshold,
        "tutor_frontier_only_waypoint": args.tutor_frontier_only_waypoint,
        "learner_risk_weight": args.learner_risk_weight,
        "learner_revisit_penalty": args.learner_revisit_penalty,
        "learner_unknown_penalty": args.learner_unknown_penalty,
        "learner_warning_suspicion_weight": args.learner_warning_suspicion_weight,
        "learner_warning_suspicion_mode": args.learner_warning_suspicion_mode,
        "learner_warning_suspicion_decay": args.learner_warning_suspicion_decay,
        "learner_consolidation_mode": args.learner_consolidation_mode,
        "learner_long_term_memory_weight": args.learner_long_term_memory_weight,
        "learner_autonomy_assist_discount": args.learner_autonomy_assist_discount,
        "learner_enable_objective_learning_events": not args.learner_disable_objective_learning_events,
        "learner_use_long_term_route_graph": not args.learner_disable_long_term_route_graph,
        "learner_use_landmark_graph": not args.learner_disable_landmark_graph,
        "warning_update_mode": args.warning_update_mode,
        "warning_eta0": args.warning_eta0,
        "warning_kl_epsilon": args.warning_kl_epsilon,
        "tutor_warning_actionability_threshold": args.tutor_warning_actionability_threshold,
        "tutor_waypoint_damage_veto_margin": args.tutor_waypoint_damage_veto_margin,
        "ablate_warning_update": args.ablate_warning_update,
        "ablate_trap_risk_update": args.ablate_trap_risk_update,
        "ablate_safe_risk_update": args.ablate_safe_risk_update,
        "ablate_eval_clear_map_memory": args.ablate_eval_clear_map_memory,
        "ablate_eval_clear_risk_belief": args.ablate_eval_clear_risk_belief,
        "ablate_eval_clear_warning_suspicion": args.ablate_eval_clear_warning_suspicion,
        "ablate_eval_clear_long_term_memory": args.ablate_eval_clear_long_term_memory,
        "record_step_details": not args.no_step_detail,
    }
    run_fixed_experiment(
        spec_name=args.spec,
        tutor_name=args.tutor,
        baseline_mode=args.baseline,
        seeds=list(args.seeds),
        out_dir=args.out,
        workers=args.workers,
        teach_task_ids=args.teach_task_ids,
        eval_task_ids=args.eval_task_ids,
        config_overrides=config_overrides,
        condition_name=args.condition_name,
        show_progress=not args.quiet,
    )
    return 0


def _run_seeds(
    *,
    spec_name: str,
    tutor_name: str,
    baseline_mode: str,
    seeds: list[int],
    workers: int,
    teach_task_ids: list[str] | None,
    eval_task_ids: list[str] | None,
    config_overrides: dict[str, Any],
    show_progress: bool,
    progress_label: str,
) -> list[dict[str, Any]]:
    jobs = [
        {
            "spec_name": spec_name,
            "tutor_name": tutor_name,
            "baseline_mode": baseline_mode,
            "seed": seed,
            "teach_task_ids": teach_task_ids,
            "eval_task_ids": eval_task_ids,
            "config_overrides": config_overrides,
        }
        for seed in seeds
    ]
    total = len(jobs)
    started = time.perf_counter()
    if show_progress:
        print(
            f"[run_fixed_maze] start condition={progress_label} seeds={total} workers={max(1, workers)}",
            flush=True,
        )
    if workers <= 1 or len(jobs) <= 1:
        out = []
        for idx, job in enumerate(jobs, start=1):
            result = _run_seed_job(job)
            out.append(result)
            if show_progress:
                _print_seed_progress(progress_label, idx, total, result, started)
        return out

    out: list[dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            futures = [pool.submit(_run_seed_job, job) for job in jobs]
            for idx, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                out.append(result)
                if show_progress:
                    _print_seed_progress(progress_label, idx, total, result, started)
    except PermissionError:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            futures = [pool.submit(_run_seed_job, job) for job in jobs]
            for idx, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                out.append(result)
                if show_progress:
                    _print_seed_progress(progress_label, idx, total, result, started)
    out.sort(key=lambda row: row["seed"])
    return out


def _run_seed_job(job: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    config = SimpleNamespace(**dict(job.get("config_overrides", {}) or {}))
    result = run_fixed_block_detailed(
        config=config,
        spec_name=str(job["spec_name"]),
        teach_task_ids=list(job.get("teach_task_ids") or []) or None,
        eval_task_ids=list(job.get("eval_task_ids") or []) or None,
        tutor_name=str(job["tutor_name"]),
        baseline_mode=str(job["baseline_mode"]),
        seed=int(job["seed"]),
    )
    return {
        "seed": int(job["seed"]),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "aggregate": dict(result.aggregate),
        "episodes": [
            _serialize_episode(ep, seed=int(job["seed"]), spec_name=str(job["spec_name"]), tutor_name=str(job["tutor_name"]), baseline_mode=str(job["baseline_mode"]))
            for ep in [*result.teach, *result.eval_same_map]
        ],
    }


def _serialize_episode(ep: Any, *, seed: int, spec_name: str, tutor_name: str, baseline_mode: str) -> dict[str, Any]:
    return {
        "seed": seed,
        "spec_name": spec_name,
        "tutor_name": tutor_name,
        "baseline_mode": baseline_mode,
        "task_id": ep.task_id,
        "phase": ep.phase,
        "summary": ep.as_dict(),
        "path": list(ep.path),
        "step_records": [
            {
                "seed": seed,
                "spec_name": spec_name,
                "tutor_name": tutor_name,
                "baseline_mode": baseline_mode,
                "task_id": ep.task_id,
                **row,
            }
            for row in ep.step_records
        ],
        "tutor_decisions": [
            {
                "seed": seed,
                "spec_name": spec_name,
                "tutor_name": tutor_name,
                "baseline_mode": baseline_mode,
                "task_id": ep.task_id,
                **row,
            }
            for row in ep.tutor_decisions
        ],
    }


def _finalize_experiment_outputs(
    *,
    out_path: Path,
    results: list[dict[str, Any]],
    spec_name: str,
    tutor_name: str,
    baseline_mode: str,
    condition_name: str,
    run_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_rows = [r["aggregate"] | {"seed": r["seed"]} for r in results]
    episodes = [ep for result in results for ep in result["episodes"]]
    steps = [row for ep in episodes for row in ep["step_records"]]
    decisions = [row for ep in episodes for row in ep["tutor_decisions"]]

    summary_row = _summarize_results(
        results=results,
        spec_name=spec_name,
        tutor_name=tutor_name,
        baseline_mode=baseline_mode,
        condition_name=condition_name,
    )

    _write_csv(out_path / "summary.csv", [summary_row])
    _write_csv(out_path / "seed_summary.csv", seed_rows)
    _write_csv(out_path / "episodes.csv", [_episode_csv_row(ep) for ep in episodes])
    _write_csv(out_path / "steps.csv", steps)
    _write_csv(out_path / "tutor_decisions.csv", decisions)
    _write_csv(out_path / "risk_eval.csv", [_risk_eval_row(r) for r in results])
    _write_csv(out_path / "map_reuse.csv", [_map_reuse_row(r) for r in results])
    _write_csv(out_path / "objective_progress.csv", [_objective_progress_row(ep) for ep in episodes])
    _write_trajectories(out_path / "trajectories", episodes)
    _write_readme(out_path / "README.md", summary_row, run_config)
    return summary_row, episodes


def _summarize_results(
    *,
    results: list[dict[str, Any]],
    spec_name: str,
    tutor_name: str,
    baseline_mode: str,
    condition_name: str,
) -> dict[str, Any]:
    aggregate_rows = [row["aggregate"] for row in results]
    numeric_keys = sorted({key for row in aggregate_rows for key, value in row.items() if isinstance(value, (int, float))})
    summary: dict[str, Any] = {
        "condition": condition_name,
        "spec_name": spec_name,
        "tutor_name": tutor_name,
        "baseline_mode": baseline_mode,
        "seed_count": len(results),
    }
    for key in numeric_keys:
        values = [float(row[key]) for row in aggregate_rows if row.get(key) is not None]
        summary[key] = mean(values) if values else None
    return summary


def _episode_csv_row(ep: dict[str, Any]) -> dict[str, Any]:
    row = dict(ep["summary"])
    row.update(
        {
            "seed": ep["seed"],
            "spec_name": ep["spec_name"],
            "tutor_name": ep["tutor_name"],
            "baseline_mode": ep["baseline_mode"],
            "path_length": len(ep["path"]),
        }
    )
    return row


def _risk_eval_row(result: dict[str, Any]) -> dict[str, Any]:
    agg = result["aggregate"]
    return {
        "seed": result["seed"],
        "risk_auc": agg.get("risk_auc"),
        "risk_auc_seen": agg.get("risk_auc_seen"),
        "risk_auc_unseen_same_map": agg.get("risk_auc_unseen_same_map"),
        "risk_nll": agg.get("risk_nll"),
        "risk_calibration_ece": agg.get("risk_calibration_ece"),
        "warning_information_gain": agg.get("warning_information_gain"),
        "posterior_shift_after_warning": agg.get("posterior_shift_after_warning"),
    }


def _map_reuse_row(result: dict[str, Any]) -> dict[str, Any]:
    agg = result["aggregate"]
    return {
        "seed": result["seed"],
        "map_coverage_teach": agg.get("map_coverage_teach"),
        "map_reuse_eval": agg.get("map_reuse_eval"),
        "useful_exploration_rate": agg.get("useful_exploration_rate"),
        "loop_rate": agg.get("loop_rate"),
        "no_info_step_rate": agg.get("no_info_step_rate"),
        "frontier_progress_rate": agg.get("frontier_progress_rate"),
    }


def _objective_progress_row(ep: dict[str, Any]) -> dict[str, Any]:
    summary = ep["summary"]
    return {
        "seed": ep["seed"],
        "task_id": ep["task_id"],
        "phase": ep["phase"],
        "success": summary.get("success"),
        "timeout": summary.get("timeout"),
        "died": summary.get("died"),
        "objective_completed_count": summary.get("objective_completed_count"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_trajectories(path: Path, episodes: Iterable[dict[str, Any]]) -> None:
    for ep in episodes:
        summary = ep["summary"]
        filename = f"seed{ep['seed']}_{ep['phase']}_{ep['task_id']}.txt"
        lines = [
            f"seed: {ep['seed']}",
            f"task_id: {ep['task_id']}",
            f"phase: {ep['phase']}",
            f"success: {summary.get('success')}",
            f"died: {summary.get('died')}",
            f"timeout: {summary.get('timeout')}",
            f"steps: {summary.get('steps')}",
            f"path_length: {len(ep['path'])}",
            "path:",
        ]
        lines.extend(str(pos) for pos in ep["path"])
        (path / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_readme(path: Path, summary: dict[str, Any], run_config: dict[str, Any]) -> None:
    text = "\n".join(
        [
            "# Fixed Maze Run",
            "",
            f"- condition: `{summary['condition']}`",
            f"- spec: `{summary['spec_name']}`",
            f"- tutor: `{summary['tutor_name']}`",
            f"- baseline: `{summary['baseline_mode']}`",
            f"- seed_count: `{summary['seed_count']}`",
            "",
            "## Key Metrics",
            "",
            f"- teach_success_rate: `{summary.get('teach_success_rate')}`",
            f"- eval_success_rate: `{summary.get('eval_success_rate')}`",
            f"- map_reuse_eval: `{summary.get('map_reuse_eval')}`",
            f"- useful_exploration_rate: `{summary.get('useful_exploration_rate')}`",
            f"- risk_auc: `{summary.get('risk_auc')}`",
            "",
            "## Config",
            "",
            "```json",
            json.dumps(run_config, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _print_seed_progress(
    progress_label: str,
    done: int,
    total: int,
    result: dict[str, Any],
    started: float,
) -> None:
    aggregate = result.get("aggregate", {}) or {}
    bar = _progress_bar(done, total)
    print(
        "[run_fixed_maze] "
        f"{progress_label} {bar} {done}/{total} "
        f"seed={result.get('seed')} "
        f"seed_elapsed={float(result.get('elapsed_sec', 0.0)):.2f}s "
        f"teach_success={aggregate.get('teach_success_rate')} "
        f"eval_success={aggregate.get('eval_success_rate')} "
        f"elapsed_total={time.perf_counter() - started:.2f}s",
        flush=True,
    )


def _progress_bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[------------------------] 0%"
    ratio = max(0.0, min(1.0, float(done) / float(total)))
    filled = int(round(ratio * width))
    filled = max(0, min(width, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {ratio * 100:5.1f}%"


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
