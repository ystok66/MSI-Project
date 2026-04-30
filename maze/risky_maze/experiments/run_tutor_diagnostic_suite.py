from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path
from typing import Any

from risky_maze.experiments.run_formal_tutor_matrix import (
    DEFAULT_SEEDS,
    _fmt,
    default_slices,
    run_formal_tutor_matrix,
)


def diagnostic_slices() -> list[dict[str, Any]]:
    keep = {"t04_e05", "t08_e07"}
    return [row for row in default_slices() if str(row.get("slice_name")) in keep]


def diagnostic_conditions() -> list[dict[str, Any]]:
    base = {
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
            "config_overrides": {**base, "record_step_details": False},
        },
        {
            "condition_name": "always_warn_mortal",
            "tutor_name": "always_warn",
            "baseline_mode": "mortal",
            "config_overrides": {**base, "record_step_details": False},
        },
        {
            "condition_name": "safety_shield_only",
            "tutor_name": "safety_shield_only",
            "baseline_mode": "mortal",
            "config_overrides": dict(base),
        },
        {
            "condition_name": "shield_plus_minimal_waypoint",
            "tutor_name": "shield_plus_minimal_waypoint",
            "baseline_mode": "mortal",
            "config_overrides": dict(base),
        },
        {
            "condition_name": "shield_plus_frontier_waypoint",
            "tutor_name": "shield_plus_frontier_waypoint",
            "baseline_mode": "mortal",
            "config_overrides": dict(base),
        },
        {
            "condition_name": "shield_plus_oracle_waypoint",
            "tutor_name": "shield_plus_oracle_waypoint",
            "baseline_mode": "mortal",
            "config_overrides": dict(base),
        },
        {
            "condition_name": "inverse_plan_warn_only",
            "tutor_name": "inverse_plan_warn_only",
            "baseline_mode": "mortal",
            "config_overrides": dict(base),
        },
        {
            "condition_name": "inverse_plan_full",
            "tutor_name": "inverse_plan_full",
            "baseline_mode": "mortal",
            "config_overrides": dict(base),
        },
        {
            "condition_name": "inverse_plan_full_clear_suspicion_eval",
            "tutor_name": "inverse_plan_full",
            "baseline_mode": "mortal",
            "config_overrides": {
                **base,
                "ablate_eval_clear_warning_suspicion": True,
            },
        },
        {
            "condition_name": "inverse_plan_full_replan_only_suspicion",
            "tutor_name": "inverse_plan_full",
            "baseline_mode": "mortal",
            "config_overrides": {
                **base,
                "learner_warning_suspicion_mode": "replan_only",
            },
        },
        {
            "condition_name": "inverse_plan_full_actionability_gated",
            "tutor_name": "inverse_plan_full",
            "baseline_mode": "mortal",
            "config_overrides": {
                **base,
                "tutor_warning_actionability_threshold": 0.05,
                "tutor_waypoint_damage_veto_margin": 0.0,
            },
        },
        {
            "condition_name": "inverse_plan_full_combined_fix",
            "tutor_name": "inverse_plan_full",
            "baseline_mode": "mortal",
            "config_overrides": {
                **base,
                "ablate_eval_clear_warning_suspicion": True,
                "learner_warning_suspicion_mode": "replan_only",
                "tutor_warning_actionability_threshold": 0.05,
                "tutor_waypoint_damage_veto_margin": 0.0,
            },
        },
    ]


def run_tutor_diagnostic_suite(
    *,
    spec_name: str = "HugeRiskyGemMaze_v0",
    out_dir: str | Path = "runs/tutor_diagnostic_suite",
    workers: int = 16,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds or DEFAULT_SEEDS[:4])
    result = run_formal_tutor_matrix(
        spec_name=spec_name,
        out_dir=out_path,
        workers=workers,
        seeds=seeds,
        slices=diagnostic_slices(),
        conditions=diagnostic_conditions(),
        show_progress=True,
    )
    _write_diagnostic_report(
        out_path / "TUTOR_DIAGNOSTIC_SUITE_REPORT.md",
        spec_name=spec_name,
        seeds=seeds,
        rows=result["summary"],
    )
    return result


def _write_diagnostic_report(path: Path, *, spec_name: str, seeds: list[int], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Tutor Diagnostic Suite",
        "",
        f"- spec: `{spec_name}`",
        f"- seeds: `{seeds}`",
        "- slices: `t04_e05`, `t08_e07`",
        "",
        "## Summary",
        "",
        "| slice | condition | teach_success | teach_safe_success | teach_damage | teach_damage_per_100 | eval_success | eval_regret | eval_damage | eval_damage_per_100 | warnings | waypoints | warning_actionability | suspicion_end_teach | suspicion_on_eval_path |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + f"{row['slice_name']} | "
            + f"{row['condition_name']} | "
            + f"{_fmt(row.get('teach_success_rate'))} | "
            + f"{_fmt(row.get('teach_safe_success_rate'))} | "
            + f"{_fmt(row.get('teach_mean_damage'))} | "
            + f"{_fmt(row.get('teach_mean_damage_per_100_steps'))} | "
            + f"{_fmt(row.get('eval_success_rate'))} | "
            + f"{_fmt(row.get('eval_regret_to_oracle_safe_path'))} | "
            + f"{_fmt(row.get('eval_mean_damage'))} | "
            + f"{_fmt(row.get('eval_mean_damage_per_100_steps'))} | "
            + f"{_fmt(row.get('teach_mean_warnings'))} | "
            + f"{_fmt(row.get('teach_mean_waypoints'))} | "
            + f"{_fmt(row.get('warning_actionability'))} | "
            + f"{_fmt(row.get('warning_suspicion_mass_end_teach'))} | "
            + f"{_fmt(row.get('warning_suspicion_mass_on_eval_path'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run targeted tutor diagnostics on T04/T08 with suspicion/actionability ablations.")
    parser.add_argument("--spec", default="HugeRiskyGemMaze_v0")
    parser.add_argument("--out", default="runs/tutor_diagnostic_suite")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_tutor_diagnostic_suite(
        spec_name=args.spec,
        out_dir=args.out,
        workers=args.workers,
        seeds=list(args.seeds) if args.seeds else None,
    )
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
