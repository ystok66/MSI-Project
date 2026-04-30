from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from risky_maze.experiments.run_fixed_maze import (
    _finalize_experiment_outputs,
    _progress_bar,
    _run_seed_job,
    _write_csv,
    _write_json,
)


DEFAULT_SEEDS = list(range(8))
DEFAULT_TEACH_TASKS = ["T01_NW_key_gem_NE_exit"]
DEFAULT_EVAL_TASKS = ["E01_WestGarden_to_NE_exit"]


def formal_phase12_conditions(seeds: list[int]) -> list[dict[str, Any]]:
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
        "tutor_profile_count": 3,
        "record_step_details": True,
    }
    return [
        {
            "condition_name": "no_tutor_mortal",
            "tutor_name": "no_tutor",
            "baseline_mode": "mortal",
            "config_overrides": dict(base_config),
        },
        {
            "condition_name": "no_tutor_immortal_warnlike",
            "tutor_name": "no_tutor",
            "baseline_mode": "immortal_warnlike",
            "config_overrides": dict(base_config),
        },
        {
            "condition_name": "no_tutor_immortal_no_timeout",
            "tutor_name": "no_tutor",
            "baseline_mode": "immortal_no_timeout",
            "config_overrides": dict(base_config),
        },
        {
            "condition_name": "always_warn_mortal",
            "tutor_name": "always_warn",
            "baseline_mode": "mortal",
            "config_overrides": dict(base_config),
        },
        {
            "condition_name": "risk_threshold_warn_mortal",
            "tutor_name": "risk_threshold_warn",
            "baseline_mode": "mortal",
            "config_overrides": dict(base_config),
        },
        {
            "condition_name": "inverse_warn_mortal",
            "tutor_name": "inverse_warn",
            "baseline_mode": "mortal",
            "config_overrides": dict(base_config),
        },
        {
            "condition_name": "inverse_plan_warn_only_mortal",
            "tutor_name": "inverse_plan_warn_only",
            "baseline_mode": "mortal",
            "config_overrides": {
                **base_config,
                "tutor_rollout_horizon": 3,
                "tutor_top_k_paths": 1,
                "tutor_max_candidates": 4,
                "tutor_profile_count": 2,
            },
        },
    ]


def d4_variants() -> list[dict[str, Any]]:
    return [
        {
            "variant": "before_d4",
            "description": "Ablation that recreates the pre-fix bug: fixed layouts do not expose latent cell features to tutor rollout.",
            "config_overrides": {"attach_fixed_layout_cell_features": False},
        },
        {
            "variant": "after_d4",
            "description": "Current fixed codepath: fixed layouts expose latent cell features to tutor rollout.",
            "config_overrides": {"attach_fixed_layout_cell_features": True},
        },
    ]


def run_d4_fix_comparison(
    *,
    spec_name: str = "HugeRiskyGemMaze_v0",
    out_dir: str | Path = "runs/d4_fix_comparison",
    workers: int = 16,
    seeds: list[int] | None = None,
    teach_task_ids: list[str] | None = None,
    eval_task_ids: list[str] | None = None,
    conditions: list[dict[str, Any]] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    teach_task_ids = list(teach_task_ids or DEFAULT_TEACH_TASKS)
    eval_task_ids = list(eval_task_ids or DEFAULT_EVAL_TASKS)
    seeds = list(seeds or DEFAULT_SEEDS)
    conditions = list(conditions or formal_phase12_conditions(seeds))
    variants = d4_variants()

    manifest = {
        "spec_name": spec_name,
        "workers": int(workers),
        "seeds": seeds,
        "teach_task_ids": teach_task_ids,
        "eval_task_ids": eval_task_ids,
        "conditions": conditions,
        "variants": variants,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(out_path / "comparison_config.json", manifest)

    jobs: list[dict[str, Any]] = []
    results_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for variant in variants:
        variant_name = str(variant["variant"])
        for condition in conditions:
            condition_name = str(condition["condition_name"])
            key = (variant_name, condition_name)
            results_by_key[key] = []
            run_dir = out_path / variant_name / condition_name
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "trajectories").mkdir(parents=True, exist_ok=True)
            run_config = {
                "spec_name": spec_name,
                "variant": variant_name,
                "variant_description": str(variant["description"]),
                "tutor_name": str(condition["tutor_name"]),
                "baseline_mode": str(condition["baseline_mode"]),
                "seeds": seeds,
                "workers": int(workers),
                "teach_task_ids": teach_task_ids,
                "eval_task_ids": eval_task_ids,
                "condition_name": condition_name,
                "config_overrides": {
                    **dict(condition["config_overrides"]),
                    **dict(variant["config_overrides"]),
                },
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _write_json(run_dir / "config.json", run_config)
            for seed in seeds:
                jobs.append(
                    {
                        "spec_name": spec_name,
                        "tutor_name": str(condition["tutor_name"]),
                        "baseline_mode": str(condition["baseline_mode"]),
                        "seed": int(seed),
                        "teach_task_ids": teach_task_ids,
                        "eval_task_ids": eval_task_ids,
                        "config_overrides": run_config["config_overrides"],
                        "condition_name": condition_name,
                        "variant": variant_name,
                    }
                )

    total_jobs = len(jobs)
    if show_progress:
        print(
            f"[run_d4_fix_comparison] start variants={len(variants)} conditions={len(conditions)} blocks={total_jobs} workers={workers}",
            flush=True,
        )

    if workers <= 1 or total_jobs <= 1:
        for idx, job in enumerate(jobs, start=1):
            result = _run_seed_job(job)
            key = (str(job["variant"]), str(job["condition_name"]))
            results_by_key[key].append(result)
            if show_progress:
                _print_block_progress(idx, total_jobs, job, result)
    else:
        future_map: dict[Any, dict[str, Any]] = {}
        try:
            with ProcessPoolExecutor(max_workers=min(workers, total_jobs)) as pool:
                for job in jobs:
                    future = pool.submit(_run_seed_job, job)
                    future_map[future] = job
                for idx, future in enumerate(as_completed(future_map), start=1):
                    job = future_map[future]
                    result = future.result()
                    key = (str(job["variant"]), str(job["condition_name"]))
                    results_by_key[key].append(result)
                    if show_progress:
                        _print_block_progress(idx, total_jobs, job, result)
        except PermissionError:
            with ThreadPoolExecutor(max_workers=min(workers, total_jobs)) as pool:
                for job in jobs:
                    future = pool.submit(_run_seed_job, job)
                    future_map[future] = job
                for idx, future in enumerate(as_completed(future_map), start=1):
                    job = future_map[future]
                    result = future.result()
                    key = (str(job["variant"]), str(job["condition_name"]))
                    results_by_key[key].append(result)
                    if show_progress:
                        _print_block_progress(idx, total_jobs, job, result)

    variant_condition_summaries: list[dict[str, Any]] = []
    variant_condition_behavior: list[dict[str, Any]] = []
    summary_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    behavior_lookup: dict[tuple[str, str], dict[str, Any]] = {}

    total_groups = len(variants) * len(conditions)
    for idx, variant in enumerate(variants, start=1):
        variant_name = str(variant["variant"])
        for jdx, condition in enumerate(conditions, start=1):
            condition_name = str(condition["condition_name"])
            key = (variant_name, condition_name)
            run_dir = out_path / variant_name / condition_name
            results = sorted(results_by_key[key], key=lambda row: row["seed"])
            summary, _episodes = _finalize_experiment_outputs(
                out_path=run_dir,
                results=results,
                spec_name=spec_name,
                tutor_name=str(condition["tutor_name"]),
                baseline_mode=str(condition["baseline_mode"]),
                condition_name=f"{variant_name}__{condition_name}",
                run_config={
                    "spec_name": spec_name,
                    "variant": variant_name,
                    "variant_description": str(variant["description"]),
                    "tutor_name": str(condition["tutor_name"]),
                    "baseline_mode": str(condition["baseline_mode"]),
                    "seeds": seeds,
                    "workers": int(workers),
                    "teach_task_ids": teach_task_ids,
                    "eval_task_ids": eval_task_ids,
                    "condition_name": condition_name,
                    "config_overrides": {
                        **dict(condition["config_overrides"]),
                        **dict(variant["config_overrides"]),
                    },
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            summary_row = dict(summary)
            summary_row["variant"] = variant_name
            summary_row["condition_name"] = condition_name
            variant_condition_summaries.append(summary_row)
            summary_lookup[key] = summary_row

            behavior_row = _summarize_tutor_behavior(results)
            behavior_row.update(
                {
                    "variant": variant_name,
                    "condition_name": condition_name,
                    "tutor_name": str(condition["tutor_name"]),
                    "baseline_mode": str(condition["baseline_mode"]),
                }
            )
            variant_condition_behavior.append(behavior_row)
            behavior_lookup[key] = behavior_row
            if show_progress:
                done = (idx - 1) * len(conditions) + jdx
                print(
                    f"[run_d4_fix_comparison] finalize {_progress_bar(done, total_groups)} "
                    f"variant={variant_name} condition={condition_name} "
                    f"teach_success={summary_row.get('teach_success_rate')} "
                    f"eval_success={summary_row.get('eval_success_rate')}",
                    flush=True,
                )

    comparison_rows = _build_comparison_rows(conditions, summary_lookup)
    behavior_comparison_rows = _build_behavior_comparison_rows(conditions, behavior_lookup)
    _write_csv(out_path / "variant_condition_summaries.csv", variant_condition_summaries)
    _write_csv(out_path / "variant_condition_tutor_behavior.csv", variant_condition_behavior)
    _write_csv(out_path / "comparison_summary.csv", comparison_rows)
    _write_csv(out_path / "comparison_tutor_behavior.csv", behavior_comparison_rows)
    _write_json(
        out_path / "comparison_index.json",
        {
            "spec_name": spec_name,
            "workers": workers,
            "seeds": seeds,
            "teach_task_ids": teach_task_ids,
            "eval_task_ids": eval_task_ids,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "conditions": [str(c["condition_name"]) for c in conditions],
            "variants": [str(v["variant"]) for v in variants],
        },
    )
    _write_report(
        out_path / "D4_FIX_COMPARISON_REPORT.md",
        spec_name=spec_name,
        workers=workers,
        seeds=seeds,
        teach_task_ids=teach_task_ids,
        eval_task_ids=eval_task_ids,
        variants=variants,
        comparison_rows=comparison_rows,
        behavior_comparison_rows=behavior_comparison_rows,
    )
    return {
        "summaries": variant_condition_summaries,
        "behavior": variant_condition_behavior,
        "comparison": comparison_rows,
        "behavior_comparison": behavior_comparison_rows,
    }


def _summarize_tutor_behavior(results: list[dict[str, Any]]) -> dict[str, Any]:
    teach_episodes = [
        ep
        for result in results
        for ep in result.get("episodes", [])
        if str(ep.get("phase", "")) == "teach"
    ]
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
        "mean_predicted_p_death_wait": _mean_numeric(decisions, "predicted_p_death_wait"),
        "mean_predicted_p_timeout_wait": _mean_numeric(decisions, "predicted_p_timeout_wait"),
        "mean_predicted_map_gain_wait": _mean_numeric(decisions, "predicted_map_gain_wait"),
        "mean_predicted_risk_ig_warning": _mean_numeric(decisions, "predicted_risk_ig_warning"),
        "mean_assist_leakage_per_decision": _mean_numeric(decisions, "assist_leakage"),
    }


def _build_comparison_rows(
    conditions: list[dict[str, Any]],
    summary_lookup: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    metric_keys = [
        "teach_success_rate",
        "teach_death_rate",
        "teach_timeout_rate",
        "teach_mean_damage",
        "teach_mean_steps",
        "teach_mean_warnings",
        "warning_information_gain",
        "posterior_shift_after_warning",
        "eval_success_rate",
        "eval_death_rate",
        "eval_timeout_rate",
        "eval_mean_damage",
        "eval_mean_steps",
        "eval_regret_to_oracle_safe_path",
        "map_coverage_teach",
        "map_reuse_eval",
        "useful_exploration_rate",
        "risk_auc",
        "risk_nll",
        "risk_calibration_ece",
        "frontier_progress_rate",
        "loop_rate",
        "no_info_step_rate",
        "assist_leakage",
    ]
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        condition_name = str(condition["condition_name"])
        before = summary_lookup[("before_d4", condition_name)]
        after = summary_lookup[("after_d4", condition_name)]
        row: dict[str, Any] = {
            "condition_name": condition_name,
            "tutor_name": str(condition["tutor_name"]),
            "baseline_mode": str(condition["baseline_mode"]),
        }
        for key in metric_keys:
            before_val = before.get(key)
            after_val = after.get(key)
            row[f"before_{key}"] = before_val
            row[f"after_{key}"] = after_val
            row[f"delta_{key}"] = _delta(after_val, before_val)
        rows.append(row)
    return rows


def _build_behavior_comparison_rows(
    conditions: list[dict[str, Any]],
    behavior_lookup: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    metric_keys = [
        "teach_decision_count",
        "warning_selected_count",
        "wait_selected_count",
        "waypoint_selected_count",
        "warning_selected_rate",
        "wait_selected_rate",
        "warnings_selected_per_teach_episode",
        "mean_q_wait",
        "mean_q_best_warning",
        "mean_predicted_p_death_wait",
        "mean_predicted_p_timeout_wait",
        "mean_predicted_map_gain_wait",
        "mean_predicted_risk_ig_warning",
        "mean_assist_leakage_per_decision",
    ]
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        condition_name = str(condition["condition_name"])
        before = behavior_lookup[("before_d4", condition_name)]
        after = behavior_lookup[("after_d4", condition_name)]
        row: dict[str, Any] = {
            "condition_name": condition_name,
            "tutor_name": str(condition["tutor_name"]),
            "baseline_mode": str(condition["baseline_mode"]),
        }
        for key in metric_keys:
            before_val = before.get(key)
            after_val = after.get(key)
            row[f"before_{key}"] = before_val
            row[f"after_{key}"] = after_val
            row[f"delta_{key}"] = _delta(after_val, before_val)
        rows.append(row)
    return rows


def _print_block_progress(done: int, total: int, job: dict[str, Any], result: dict[str, Any]) -> None:
    aggregate = result.get("aggregate", {}) or {}
    print(
        f"[run_d4_fix_comparison] {_progress_bar(done, total)} "
        f"variant={job['variant']} condition={job['condition_name']} seed={job['seed']} "
        f"elapsed={float(result.get('elapsed_sec', 0.0)):.2f}s "
        f"teach_success={aggregate.get('teach_success_rate')} "
        f"eval_success={aggregate.get('eval_success_rate')}",
        flush=True,
    )


def _write_report(
    path: Path,
    *,
    spec_name: str,
    workers: int,
    seeds: list[int],
    teach_task_ids: list[str],
    eval_task_ids: list[str],
    variants: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    behavior_comparison_rows: list[dict[str, Any]],
) -> None:
    focus = next((row for row in comparison_rows if row["condition_name"] == "inverse_plan_warn_only_mortal"), None)
    focus_behavior = next((row for row in behavior_comparison_rows if row["condition_name"] == "inverse_plan_warn_only_mortal"), None)
    lines = [
        "# D4 Fix Comparison Report",
        "",
        f"- generated_at_utc: `{datetime.now(timezone.utc).isoformat()}`",
        f"- spec: `{spec_name}`",
        f"- workers: `{workers}`",
        f"- seeds: `{seeds}`",
        f"- teach_task_ids: `{teach_task_ids}`",
        f"- eval_task_ids: `{eval_task_ids}`",
        "",
        "## Variants",
        "",
    ]
    for variant in variants:
        lines.append(f"- `{variant['variant']}`: {variant['description']}")
    lines.extend(
        [
            "",
            "## Condition-Level Summary",
            "",
            "| condition | delta_teach_success | delta_teach_warnings | delta_warning_ig | delta_eval_success | delta_eval_regret | delta_risk_auc |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison_rows:
        lines.append(
            "| "
            + f"{row['condition_name']} | "
            + f"{_fmt(row.get('delta_teach_success_rate'))} | "
            + f"{_fmt(row.get('delta_teach_mean_warnings'))} | "
            + f"{_fmt(row.get('delta_warning_information_gain'))} | "
            + f"{_fmt(row.get('delta_eval_success_rate'))} | "
            + f"{_fmt(row.get('delta_eval_regret_to_oracle_safe_path'))} | "
            + f"{_fmt(row.get('delta_risk_auc'))} |"
        )
    lines.extend(["", "## Inverse Tutor Focus", ""])
    if focus is not None:
        lines.extend(
            [
                f"- `inverse_plan_warn_only_mortal` teach_success: `{_fmt(focus.get('before_teach_success_rate'))}` -> `{_fmt(focus.get('after_teach_success_rate'))}`",
                f"- `inverse_plan_warn_only_mortal` teach_mean_warnings: `{_fmt(focus.get('before_teach_mean_warnings'))}` -> `{_fmt(focus.get('after_teach_mean_warnings'))}`",
                f"- `inverse_plan_warn_only_mortal` warning_information_gain: `{_fmt(focus.get('before_warning_information_gain'))}` -> `{_fmt(focus.get('after_warning_information_gain'))}`",
                f"- `inverse_plan_warn_only_mortal` posterior_shift_after_warning: `{_fmt(focus.get('before_posterior_shift_after_warning'))}` -> `{_fmt(focus.get('after_posterior_shift_after_warning'))}`",
                f"- `inverse_plan_warn_only_mortal` eval_success: `{_fmt(focus.get('before_eval_success_rate'))}` -> `{_fmt(focus.get('after_eval_success_rate'))}`",
                f"- `inverse_plan_warn_only_mortal` eval_regret_to_oracle_safe_path: `{_fmt(focus.get('before_eval_regret_to_oracle_safe_path'))}` -> `{_fmt(focus.get('after_eval_regret_to_oracle_safe_path'))}`",
                f"- `inverse_plan_warn_only_mortal` risk_auc: `{_fmt(focus.get('before_risk_auc'))}` -> `{_fmt(focus.get('after_risk_auc'))}`",
            ]
        )
    if focus_behavior is not None:
        lines.extend(
            [
                "",
                "### Tutor Decision Stats",
                "",
                f"- warning_selected_rate: `{_fmt(focus_behavior.get('before_warning_selected_rate'))}` -> `{_fmt(focus_behavior.get('after_warning_selected_rate'))}`",
                f"- warnings_selected_per_teach_episode: `{_fmt(focus_behavior.get('before_warnings_selected_per_teach_episode'))}` -> `{_fmt(focus_behavior.get('after_warnings_selected_per_teach_episode'))}`",
                f"- mean_q_wait: `{_fmt(focus_behavior.get('before_mean_q_wait'))}` -> `{_fmt(focus_behavior.get('after_mean_q_wait'))}`",
                f"- mean_q_best_warning: `{_fmt(focus_behavior.get('before_mean_q_best_warning'))}` -> `{_fmt(focus_behavior.get('after_mean_q_best_warning'))}`",
                f"- mean_predicted_p_death_wait: `{_fmt(focus_behavior.get('before_mean_predicted_p_death_wait'))}` -> `{_fmt(focus_behavior.get('after_mean_predicted_p_death_wait'))}`",
                f"- mean_predicted_risk_ig_warning: `{_fmt(focus_behavior.get('before_mean_predicted_risk_ig_warning'))}` -> `{_fmt(focus_behavior.get('after_mean_predicted_risk_ig_warning'))}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `variant_condition_summaries.csv`: per variant per condition aggregate metrics",
            "- `variant_condition_tutor_behavior.csv`: per variant per condition tutor decision behavior metrics",
            "- `comparison_summary.csv`: before/after/delta metrics for Phase 1/2 conditions",
            "- `comparison_tutor_behavior.csv`: before/after/delta tutor behavior metrics",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except Exception:
            continue
    if not values:
        return None
    return sum(values) / len(values)


def _delta(after: Any, before: Any) -> float | None:
    if after is None or before is None:
        return None
    try:
        return float(after) - float(before)
    except Exception:
        return None


def _safe_div(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return float(num) / float(den)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run before/after D4 fixed-layout tutor comparison.")
    parser.add_argument("--spec", default="HugeRiskyGemMaze_v0")
    parser.add_argument("--out", default="runs/d4_fix_comparison")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--teach-task-ids", nargs="*", default=None)
    parser.add_argument("--eval-task-ids", nargs="*", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_d4_fix_comparison(
        spec_name=args.spec,
        out_dir=args.out,
        workers=args.workers,
        seeds=args.seeds,
        teach_task_ids=args.teach_task_ids,
        eval_task_ids=args.eval_task_ids,
        show_progress=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
