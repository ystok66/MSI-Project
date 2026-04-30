from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path
from typing import Any

from risky_maze.experiments.run_formal_tutor_matrix import DEFAULT_SEEDS, _fmt, default_slices, run_formal_tutor_matrix


def default_ablation_slices() -> list[dict[str, Any]]:
    keep = {"t08_e07", "full_suite"}
    return [s for s in default_slices() if s["slice_name"] in keep]


def memory_risk_ablation_conditions() -> list[dict[str, Any]]:
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
        "record_step_details": False,
    }
    return [
        {
            "condition_name": "full_inverse_full_learner",
            "tutor_name": "inverse_plan_full",
            "baseline_mode": "mortal",
            "config_overrides": dict(base),
        },
        {
            "condition_name": "full_inverse_no_map_memory_eval",
            "tutor_name": "inverse_plan_full",
            "baseline_mode": "mortal",
            "config_overrides": {**base, "ablate_eval_clear_map_memory": True},
        },
        {
            "condition_name": "full_inverse_map_memory_only_eval",
            "tutor_name": "inverse_plan_full",
            "baseline_mode": "mortal",
            "config_overrides": {**base, "ablate_eval_clear_risk_belief": True},
        },
        {
            "condition_name": "full_inverse_no_warning_update",
            "tutor_name": "inverse_plan_full",
            "baseline_mode": "mortal",
            "config_overrides": {**base, "ablate_warning_update": True},
        },
        {
            "condition_name": "full_inverse_no_trap_risk_update",
            "tutor_name": "inverse_plan_full",
            "baseline_mode": "mortal",
            "config_overrides": {**base, "ablate_trap_risk_update": True},
        },
        {
            "condition_name": "full_inverse_no_safe_risk_update",
            "tutor_name": "inverse_plan_full",
            "baseline_mode": "mortal",
            "config_overrides": {**base, "ablate_safe_risk_update": True},
        },
        {
            "condition_name": "full_inverse_no_risk_updates",
            "tutor_name": "inverse_plan_full",
            "baseline_mode": "mortal",
            "config_overrides": {
                **base,
                "ablate_warning_update": True,
                "ablate_trap_risk_update": True,
                "ablate_safe_risk_update": True,
            },
        },
    ]


def run_memory_risk_ablation(
    *,
    spec_name: str = "HugeRiskyGemMaze_v0",
    out_dir: str | Path = "runs/memory_risk_ablation",
    workers: int = 16,
    seeds: list[int] | None = None,
    slices: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds or DEFAULT_SEEDS)
    slices = list(slices or default_ablation_slices())
    result = run_formal_tutor_matrix(
        spec_name=spec_name,
        out_dir=out_path,
        workers=workers,
        seeds=seeds,
        slices=slices,
        conditions=memory_risk_ablation_conditions(),
        show_progress=True,
    )
    _write_ablation_report(out_path / "MEMORY_RISK_ABLATION_REPORT.md", spec_name, seeds, result["summary"])
    return result


def _write_ablation_report(path: Path, spec_name: str, seeds: list[int], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Memory / Risk Ablation",
        "",
        f"- spec: `{spec_name}`",
        f"- seeds: `{seeds}`",
        "",
        "| slice | condition | eval_success | eval_regret | map_reuse_eval | useful_exploration | risk_auc | risk_auc_unseen |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + f"{row['slice_name']} | "
            + f"{row['condition_name']} | "
            + f"{_fmt(row.get('eval_success_rate'))} | "
            + f"{_fmt(row.get('eval_regret_to_oracle_safe_path'))} | "
            + f"{_fmt(row.get('map_reuse_eval'))} | "
            + f"{_fmt(row.get('useful_exploration_rate'))} | "
            + f"{_fmt(row.get('risk_auc'))} | "
            + f"{_fmt(row.get('risk_auc_unseen_same_map'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run memory/risk attribution ablations on T08/full-suite.")
    parser.add_argument("--spec", default="HugeRiskyGemMaze_v0")
    parser.add_argument("--out", default="runs/memory_risk_ablation")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_memory_risk_ablation(
        spec_name=args.spec,
        out_dir=args.out,
        workers=args.workers,
        seeds=list(args.seeds) if args.seeds else None,
    )
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
