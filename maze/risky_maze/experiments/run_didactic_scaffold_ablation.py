from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path
from typing import Any

from risky_maze.experiments.run_didactic_tutor_suite import (
    DIDACTIC_VARIANTS,
    _fmt,
    didactic_maps,
    run_didactic_tutor_suite,
)
from risky_maze.experiments.run_formal_tutor_matrix import DEFAULT_SEEDS


def scaffold_ablation_conditions() -> list[dict[str, Any]]:
    fast = {"record_step_details": False}
    return [
        {
            "condition_name": "no_tutor_mortal",
            "tutor_name": "no_tutor",
            "baseline_mode": "mortal",
            "config_overrides": dict(fast),
        },
        {
            "condition_name": "no_tutor_immortal_warnlike",
            "tutor_name": "no_tutor",
            "baseline_mode": "immortal_warnlike",
            "config_overrides": dict(fast),
        },
        {
            "condition_name": "always_warn_mortal",
            "tutor_name": "always_warn",
            "baseline_mode": "mortal",
            "config_overrides": dict(fast),
        },
        {
            "condition_name": "safety_shield_only",
            "tutor_name": "safety_shield_only",
            "baseline_mode": "mortal",
            "config_overrides": dict(fast),
        },
        {
            "condition_name": "minimal_scaffold_success_gated_assist_discounted",
            "tutor_name": "shield_plus_minimal_waypoint",
            "baseline_mode": "mortal",
            "config_overrides": dict(fast),
        },
        {
            "condition_name": "random_frontier_scaffold_success_gated_assist_discounted",
            "tutor_name": "shield_plus_random_frontier_waypoint",
            "baseline_mode": "mortal",
            "config_overrides": dict(fast),
        },
        {
            "condition_name": "frontier_scaffold_success_gated_assist_discounted",
            "tutor_name": "shield_plus_frontier_waypoint",
            "baseline_mode": "mortal",
            "config_overrides": dict(fast),
        },
        {
            "condition_name": "minimal_scaffold_success_gated",
            "tutor_name": "shield_plus_minimal_waypoint",
            "baseline_mode": "mortal",
            "config_overrides": {**fast, "learner_consolidation_mode": "success_gated"},
        },
        {
            "condition_name": "minimal_scaffold_always_commit",
            "tutor_name": "shield_plus_minimal_waypoint",
            "baseline_mode": "mortal",
            "config_overrides": {**fast, "learner_consolidation_mode": "always_commit"},
        },
        {
            "condition_name": "minimal_scaffold_clear_eval_long_term_memory",
            "tutor_name": "shield_plus_minimal_waypoint",
            "baseline_mode": "mortal",
            "config_overrides": {**fast, "ablate_eval_clear_long_term_memory": True},
        },
        {
            "condition_name": "minimal_scaffold_no_route_graph",
            "tutor_name": "shield_plus_minimal_waypoint",
            "baseline_mode": "mortal",
            "config_overrides": {**fast, "learner_use_long_term_route_graph": False},
        },
        {
            "condition_name": "minimal_scaffold_no_landmark_graph",
            "tutor_name": "shield_plus_minimal_waypoint",
            "baseline_mode": "mortal",
            "config_overrides": {**fast, "learner_use_landmark_graph": False},
        },
        {
            "condition_name": "minimal_scaffold_no_objective_learning_events",
            "tutor_name": "shield_plus_minimal_waypoint",
            "baseline_mode": "mortal",
            "config_overrides": {**fast, "learner_enable_objective_learning_events": False},
        },
        {
            "condition_name": "oracle_scaffold_success_gated_assist_discounted",
            "tutor_name": "shield_plus_oracle_when_needed",
            "baseline_mode": "mortal",
            "config_overrides": dict(fast),
        },
        {
            "condition_name": "oracle_scaffold_always_commit",
            "tutor_name": "shield_plus_oracle_when_needed",
            "baseline_mode": "mortal",
            "config_overrides": {**fast, "learner_consolidation_mode": "always_commit"},
        },
        {
            "condition_name": "always_waypoint_mortal",
            "tutor_name": "always_waypoint",
            "baseline_mode": "mortal",
            "config_overrides": dict(fast),
        },
        {
            "condition_name": "always_oracle_waypoint_mortal",
            "tutor_name": "always_oracle_waypoint",
            "baseline_mode": "mortal",
            "config_overrides": dict(fast),
        },
    ]


def run_didactic_scaffold_ablation(
    *,
    out_dir: str | Path = "runs/didactic_scaffold_ablation",
    workers: int = 16,
    seeds: list[int] | None = None,
    maps: list[dict[str, Any]] | None = None,
    variant: str = "base",
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds or DEFAULT_SEEDS[:4])
    maps = list(maps or didactic_maps(variant=variant))
    result = run_didactic_tutor_suite(
        out_dir=out_path,
        workers=workers,
        seeds=seeds,
        maps=maps,
        conditions=scaffold_ablation_conditions(),
        variant=variant,
        show_progress=True,
    )
    _write_ablation_report(
        path=out_path / "DIDACTIC_SCAFFOLD_ABLATION_REPORT.md",
        seeds=seeds,
        rows=result["summary"],
    )
    return result


def _write_ablation_report(*, path: Path, seeds: list[int], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Didactic Scaffold Ablation",
        "",
        f"- seeds: `{seeds}`",
        "",
        "| map | condition | teach_success | teach_cost | teach_sec | eval_success | eval_cost | eval_sec | autonomy_credit | route_graph_conf | landmark_graph_conf | map_reuse_eval | useful_exploration | objective_events |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + f"{row['map_name']} | "
            + f"{row['condition_name']} | "
            + f"{_fmt(row.get('teach_success_rate'))} | "
            + f"{_fmt(row.get('teach_cost'))} | "
            + f"{_fmt(row.get('teach_mean_elapsed_seconds'))} | "
            + f"{_fmt(row.get('eval_success_rate'))} | "
            + f"{_fmt(row.get('eval_cost'))} | "
            + f"{_fmt(row.get('eval_mean_elapsed_seconds'))} | "
            + f"{_fmt(row.get('autonomy_credit'))} | "
            + f"{_fmt(row.get('route_graph_confidence'))} | "
            + f"{_fmt(row.get('landmark_graph_confidence'))} | "
            + f"{_fmt(row.get('map_reuse_eval'))} | "
            + f"{_fmt(row.get('useful_exploration_rate'))} | "
            + f"{_fmt(row.get('objective_learning_event_count'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run scaffold-memory ablations on TutorDidacticMazeSuite_v1.")
    parser.add_argument("--out", default="runs/didactic_scaffold_ablation")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--variant", default="base", choices=list(DIDACTIC_VARIANTS))
    parser.add_argument("--maps", nargs="*", default=None, choices=sorted({m["map_name"] for v in DIDACTIC_VARIANTS for m in didactic_maps(variant=v)}))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    maps = didactic_maps(variant=str(args.variant))
    if args.maps:
        selected = set(args.maps)
        maps = [m for m in maps if m["map_name"] in selected]
    run_didactic_scaffold_ablation(
        out_dir=args.out,
        workers=args.workers,
        seeds=list(args.seeds) if args.seeds else None,
        maps=maps,
        variant=str(args.variant),
    )
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
