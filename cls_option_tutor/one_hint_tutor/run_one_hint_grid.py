from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

from .config import OneHintConfig
from .experiment_matrix import run_experiment_scenario, run_regime_discovery_row
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


def _mean(values: List[float]) -> float | None:
    return None if not values else float(mean(values))


def _aggregate_rows(rows: List[dict]) -> dict:
    no_tutor_t = [float(row["metrics"].get("no_tutor_T_success", 0.0)) for row in rows]
    no_tutor_tplush = [float(row["metrics"].get("no_tutor_TplusH_success", 0.0)) for row in rows]
    random_hard = [float(row["metrics"].get("random_hard_success", 0.0)) for row in rows]
    random_same_pool = [float(row["metrics"].get("random_same_pool_success", 0.0)) for row in rows]
    success = [float(row["metrics"].get("tutor_T6_success", 0.0)) for row in rows]
    band = [float(row["metrics"].get("tutor_T6_band_success", 0.0)) for row in rows]
    early = [float(row["metrics"].get("tutor_T6_early_success", 0.0)) for row in rows]
    soft_tau = [float(row["metrics"].get("tutor_T6_soft_tau_score", 0.0)) for row in rows]
    early_no_transfer = [float(row["metrics"].get("tutor_T6_early_no_transfer", 0.0)) for row in rows]
    eval_exact = [float(row["metrics"].get("tutor_T6_eval_exact", 0.0)) for row in rows]
    eval_cell = [float(row["metrics"].get("tutor_T6_eval_cell", 0.0)) for row in rows]
    delta_s_no_tutor = [float(row["metrics"].get("paired_delta_success_vs_no_tutor_TplusH", 0.0)) for row in rows]
    delta_e_no_tutor = [float(row["metrics"].get("paired_delta_eval_cell_vs_no_tutor_TplusH", 0.0)) for row in rows]
    delta_s_random = [float(row["metrics"].get("paired_delta_success_vs_random_same_pool", 0.0)) for row in rows]
    delta_e_random = [float(row["metrics"].get("paired_delta_eval_cell_vs_random_same_pool", 0.0)) for row in rows]
    oracle_scores = [float(row["metrics"].get("oracle_numeric_score", 0.0)) for row in rows if "oracle_numeric_score" in row["metrics"]]
    oracle_gaps = [float(row["oracle_gap"]) for row in rows if row.get("oracle_gap") is not None]
    failure_counter = Counter()
    family_counter = Counter()
    for row in rows:
        failure = row["metrics"].get("tutor_T6_failure_type")
        if failure:
            failure_counter[str(failure)] += 1
        for family, count in (row.get("candidate_family_counts") or {}).items():
            family_counter[str(family)] += int(count)
    return {
        "n_rows": len(rows),
        "no_tutor_T_success_mean": _mean(no_tutor_t),
        "no_tutor_TplusH_success_mean": _mean(no_tutor_tplush),
        "random_hard_success_mean": _mean(random_hard),
        "random_same_pool_success_mean": _mean(random_same_pool),
        "tutor_T6_success_rate": _mean(success),
        "tutor_T6_band_success_rate": _mean(band),
        "tutor_T6_early_success_rate": _mean(early),
        "tutor_T6_soft_tau_score_mean": _mean(soft_tau),
        "tutor_T6_early_no_transfer_mean": _mean(early_no_transfer),
        "tutor_T6_eval_exact_mean": _mean(eval_exact),
        "tutor_T6_eval_cell_mean": _mean(eval_cell),
        "paired_delta_success_vs_no_tutor_TplusH_mean": _mean(delta_s_no_tutor),
        "paired_delta_eval_cell_vs_no_tutor_TplusH_mean": _mean(delta_e_no_tutor),
        "paired_delta_success_vs_random_same_pool_mean": _mean(delta_s_random),
        "paired_delta_eval_cell_vs_random_same_pool_mean": _mean(delta_e_random),
        "oracle_numeric_score_mean": _mean(oracle_scores),
        "oracle_gap_mean": _mean(oracle_gaps),
        "failure_counts": dict(sorted(failure_counter.items())),
        "candidate_family_mass": dict(sorted(family_counter.items())),
    }


def _aggregate_regime_rows(rows: List[dict], cfg: OneHintConfig) -> dict:
    def _metric(name: str) -> List[float]:
        return [float(row.get("metrics", {}).get(name, 0.0)) for row in rows]

    selected_family_counter = Counter()
    abstain_counter = Counter()
    for row in rows:
        hint = row.get("selected_hint")
        family = None
        if isinstance(hint, dict):
            family = hint.get("metadata", {}).get("family", hint.get("kind"))
        if family:
            selected_family_counter[str(family)] += 1
        pred = row.get("planner_prediction") or {}
        if pred.get("abstained"):
            abstain_counter[str(pred.get("abstain_reason") or "unknown")] += 1

    no_tutor_tplush_success_mean = _mean(_metric("no_tutor_TplusH_success"))
    oracle_success_headroom_mean = _mean(_metric("oracle_success_headroom_vs_no_tutor_TplusH"))
    oracle_band_headroom_mean = _mean(_metric("oracle_band_headroom_vs_no_tutor_TplusH"))
    oracle_eval_cell_headroom_mean = _mean(_metric("oracle_eval_cell_headroom_vs_no_tutor_TplusH"))
    search_discriminative_rate = _mean(_metric("search_discriminative"))
    transfer_discriminative_rate = _mean(_metric("transfer_discriminative"))
    regime_discriminative = (
        no_tutor_tplush_success_mean is not None
        and oracle_success_headroom_mean is not None
        and oracle_eval_cell_headroom_mean is not None
        and float(getattr(cfg, "regime_no_tutor_success_min", 0.35))
        <= float(no_tutor_tplush_success_mean)
        <= float(getattr(cfg, "regime_no_tutor_success_max", 0.75))
        and float(oracle_success_headroom_mean) >= float(getattr(cfg, "regime_oracle_success_headroom_min", 0.15))
        and float(oracle_eval_cell_headroom_mean) >= float(getattr(cfg, "regime_oracle_eval_cell_headroom_min", 0.03))
    )

    return {
        "n_rows": len(rows),
        "no_tutor_T_success_mean": _mean(_metric("no_tutor_T_success")),
        "no_tutor_TplusH_success_mean": no_tutor_tplush_success_mean,
        "random_hard_success_mean": _mean(_metric("random_hard_success")),
        "random_hard_success_std_mean": _mean(_metric("random_hard_success_std")),
        "random_hard_eval_cell_mean": _mean(_metric("random_hard_eval_cell")),
        "random_hard_eval_cell_std_mean": _mean(_metric("random_hard_eval_cell_std")),
        "random_same_pool_success_mean": _mean(_metric("random_same_pool_success")),
        "random_same_pool_success_std_mean": _mean(_metric("random_same_pool_success_std")),
        "random_same_pool_eval_cell_mean": _mean(_metric("random_same_pool_eval_cell")),
        "random_same_pool_eval_cell_std_mean": _mean(_metric("random_same_pool_eval_cell_std")),
        "random_same_pool_tau_mean": _mean(_metric("random_same_pool_tau_mean")),
        "tutor_success_mean": _mean(_metric("tutor_success")),
        "oracle_success_mean": _mean(_metric("oracle_success")),
        "oracle_band_success_mean": _mean(_metric("oracle_band_success")),
        "oracle_eval_cell_mean": _mean(_metric("oracle_eval_cell")),
        "tutor_band_success_mean": _mean(_metric("tutor_band_success")),
        "tutor_early_success_mean": _mean(_metric("tutor_early_success")),
        "tutor_soft_tau_score_mean": _mean(_metric("tutor_soft_tau_score")),
        "tutor_early_no_transfer_mean": _mean(_metric("tutor_early_no_transfer")),
        "tutor_eval_exact_mean": _mean(_metric("tutor_eval_exact")),
        "tutor_eval_cell_mean": _mean(_metric("tutor_eval_cell")),
        "paired_delta_success_vs_no_tutor_TplusH_mean": _mean(_metric("paired_delta_success_vs_no_tutor_TplusH")),
        "paired_delta_eval_cell_vs_no_tutor_TplusH_mean": _mean(_metric("paired_delta_eval_cell_vs_no_tutor_TplusH")),
        "paired_delta_success_vs_random_same_pool_mean": _mean(_metric("paired_delta_success_vs_random_same_pool")),
        "paired_delta_eval_cell_vs_random_same_pool_mean": _mean(_metric("paired_delta_eval_cell_vs_random_same_pool")),
        "oracle_success_headroom_vs_no_tutor_TplusH_mean": oracle_success_headroom_mean,
        "oracle_band_headroom_vs_no_tutor_TplusH_mean": oracle_band_headroom_mean,
        "oracle_eval_cell_headroom_vs_no_tutor_TplusH_mean": oracle_eval_cell_headroom_mean,
        "bonus_attempts_limit_mean": _mean(_metric("bonus_attempts_limit")),
        "bonus_attempts_effective_rate": _mean(_metric("bonus_attempts_effective")),
        "regime_discriminative": bool(regime_discriminative),
        "search_discriminative_rate": search_discriminative_rate,
        "transfer_discriminative_rate": transfer_discriminative_rate,
        "selected_hint_family_counts": dict(sorted(selected_family_counter.items())),
        "abstain_reason_counts": dict(sorted(abstain_counter.items())),
    }


def _markdown_summary(payload: dict) -> str:
    lines: List[str] = []
    grid_mode = str(payload.get("grid_mode", "scenario"))
    lines.append("# One-Hint Tutor Grid Summary")
    lines.append("")
    for experiment in payload.get("experiments", []):
        lines.append(f"## {experiment['name']}")
        aggregate = experiment.get("aggregate", {})
        lines.append("")
        lines.append(f"- Rows: {aggregate.get('n_rows')}")
        if grid_mode == "regime_discovery":
            lines.append(f"- `no_tutor_T` success: {aggregate.get('no_tutor_T_success_mean')}")
            lines.append(f"- `no_tutor_T+H` success: {aggregate.get('no_tutor_TplusH_success_mean')}")
            lines.append(f"- `random_hard_hint_T` success: {aggregate.get('random_hard_success_mean')}")
            lines.append(f"- `random_hard_hint_T` success std: {aggregate.get('random_hard_success_std_mean')}")
            lines.append(f"- `random_hard_hint_T` eval cell: {aggregate.get('random_hard_eval_cell_mean')}")
            lines.append(f"- `random_hard_hint_T` eval cell std: {aggregate.get('random_hard_eval_cell_std_mean')}")
            lines.append(f"- `random_same_pool` success: {aggregate.get('random_same_pool_success_mean')}")
            lines.append(f"- `random_same_pool` success std: {aggregate.get('random_same_pool_success_std_mean')}")
            lines.append(f"- `random_same_pool` eval cell: {aggregate.get('random_same_pool_eval_cell_mean')}")
            lines.append(f"- `random_same_pool` eval cell std: {aggregate.get('random_same_pool_eval_cell_std_mean')}")
            lines.append(f"- `random_same_pool` tau mean: {aggregate.get('random_same_pool_tau_mean')}")
            lines.append(f"- `tutor` success: {aggregate.get('tutor_success_mean')}")
            lines.append(f"- `oracle` success: {aggregate.get('oracle_success_mean')}")
            lines.append(f"- `oracle` band success: {aggregate.get('oracle_band_success_mean')}")
            lines.append(f"- `oracle` eval cell: {aggregate.get('oracle_eval_cell_mean')}")
            lines.append(f"- `tutor` band success: {aggregate.get('tutor_band_success_mean')}")
            lines.append(f"- `tutor` early success: {aggregate.get('tutor_early_success_mean')}")
            lines.append(f"- `tutor` soft tau score: {aggregate.get('tutor_soft_tau_score_mean')}")
            lines.append(f"- `tutor` early-no-transfer: {aggregate.get('tutor_early_no_transfer_mean')}")
            lines.append(f"- `tutor` eval exact: {aggregate.get('tutor_eval_exact_mean')}")
            lines.append(f"- `tutor` eval cell: {aggregate.get('tutor_eval_cell_mean')}")
            lines.append(f"- Delta success vs `no_tutor_T+H`: {aggregate.get('paired_delta_success_vs_no_tutor_TplusH_mean')}")
            lines.append(f"- Delta eval cell vs `no_tutor_T+H`: {aggregate.get('paired_delta_eval_cell_vs_no_tutor_TplusH_mean')}")
            lines.append(f"- Delta success vs `random_same_pool`: {aggregate.get('paired_delta_success_vs_random_same_pool_mean')}")
            lines.append(f"- Delta eval cell vs `random_same_pool`: {aggregate.get('paired_delta_eval_cell_vs_random_same_pool_mean')}")
            lines.append(f"- Oracle success headroom: {aggregate.get('oracle_success_headroom_vs_no_tutor_TplusH_mean')}")
            lines.append(f"- Oracle band headroom: {aggregate.get('oracle_band_headroom_vs_no_tutor_TplusH_mean')}")
            lines.append(f"- Oracle eval-cell headroom: {aggregate.get('oracle_eval_cell_headroom_vs_no_tutor_TplusH_mean')}")
            lines.append(f"- Bonus attempts effective: {aggregate.get('bonus_attempts_effective_rate')}")
            lines.append(f"- Search-discriminative rate: {aggregate.get('search_discriminative_rate')}")
            lines.append(f"- Transfer-discriminative rate: {aggregate.get('transfer_discriminative_rate')}")
            lines.append(f"- Legacy strict discriminative: {aggregate.get('regime_discriminative')}")
            if aggregate.get("selected_hint_family_counts"):
                lines.append(f"- Selected hint families: `{json.dumps(aggregate['selected_hint_family_counts'], ensure_ascii=False)}`")
            if aggregate.get("abstain_reason_counts"):
                lines.append(f"- Abstain reasons: `{json.dumps(aggregate['abstain_reason_counts'], ensure_ascii=False)}`")
        else:
            lines.append(f"- `no_tutor_T` success: {aggregate.get('no_tutor_T_success_mean')}")
            lines.append(f"- `no_tutor_T+H` success: {aggregate.get('no_tutor_TplusH_success_mean')}")
            lines.append(f"- `random_hard_hint_T` success: {aggregate.get('random_hard_success_mean')}")
            lines.append(f"- `random_same_pool` success: {aggregate.get('random_same_pool_success_mean')}")
            lines.append(f"- `tutor_T` success: {aggregate.get('tutor_T6_success_rate')}")
            lines.append(f"- Target-band success: {aggregate.get('tutor_T6_band_success_rate')}")
            lines.append(f"- Early success: {aggregate.get('tutor_T6_early_success_rate')}")
            lines.append(f"- Soft tau score: {aggregate.get('tutor_T6_soft_tau_score_mean')}")
            lines.append(f"- Early-no-transfer: {aggregate.get('tutor_T6_early_no_transfer_mean')}")
            lines.append(f"- Eval exact: {aggregate.get('tutor_T6_eval_exact_mean')}")
            lines.append(f"- Eval cell: {aggregate.get('tutor_T6_eval_cell_mean')}")
            lines.append(f"- Delta success vs `no_tutor_T+H`: {aggregate.get('paired_delta_success_vs_no_tutor_TplusH_mean')}")
            lines.append(f"- Delta eval cell vs `no_tutor_T+H`: {aggregate.get('paired_delta_eval_cell_vs_no_tutor_TplusH_mean')}")
            lines.append(f"- Delta success vs `random_same_pool`: {aggregate.get('paired_delta_success_vs_random_same_pool_mean')}")
            lines.append(f"- Delta eval cell vs `random_same_pool`: {aggregate.get('paired_delta_eval_cell_vs_random_same_pool_mean')}")
            if aggregate.get("oracle_gap_mean") is not None:
                lines.append(f"- Oracle gap: {aggregate.get('oracle_gap_mean')}")
            if aggregate.get("failure_counts"):
                lines.append(f"- Failures: `{json.dumps(aggregate['failure_counts'], ensure_ascii=False)}`")
            if aggregate.get("candidate_family_mass"):
                lines.append(f"- Candidate families: `{json.dumps(aggregate['candidate_family_mass'], ensure_ascii=False)}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _run_grid_job(
    grid_mode: str,
    task_id: str,
    cfg: OneHintConfig,
    seed: int,
) -> dict:
    if grid_mode == "regime_discovery":
        return run_regime_discovery_row(task_id=task_id, cfg=cfg, seed=int(seed))
    return run_experiment_scenario(task_id=task_id, cfg=cfg, seed=int(seed))


def run_grid(
    grid_spec: dict,
    workers: int = 1,
    executor_kind: str = "thread",
) -> dict:
    grid_mode = str(grid_spec.get("grid_mode", "scenario"))
    base_cfg = config_from_overrides(OneHintConfig(), dict(grid_spec.get("base", {}) or {}))
    default_tasks = [str(task) for task in grid_spec.get("tasks", [])]
    default_seeds = parse_seed_spec(grid_spec.get("seeds", [0]))

    payload = {
        "grid_mode": grid_mode,
        "tasks": default_tasks,
        "seeds": default_seeds,
        "experiments": [],
    }

    for experiment_spec in list(grid_spec.get("experiments", []) or []):
        name = str(experiment_spec.get("name", "unnamed"))
        tasks = [str(task) for task in experiment_spec.get("tasks", default_tasks)]
        seeds = parse_seed_spec(experiment_spec.get("seeds", default_seeds))
        cfg = _experiment_cfg(base_cfg, experiment_spec)
        jobs: List[Tuple[int, str, int]] = []
        for idx, task_id in enumerate(tasks):
            for seed in seeds:
                jobs.append((len(jobs), task_id, int(seed)))

        rows: List[dict]
        if max(1, int(workers)) <= 1 or len(jobs) <= 1:
            rows = [
                _run_grid_job(grid_mode=grid_mode, task_id=task_id, cfg=cfg, seed=seed)
                for _, task_id, seed in jobs
            ]
        else:
            indexed_rows: List[Tuple[int, dict]] = []
            executor_cls = ThreadPoolExecutor if str(executor_kind).lower() != "process" else ProcessPoolExecutor
            with executor_cls(max_workers=max(1, int(workers))) as executor:
                future_map = {
                    executor.submit(_run_grid_job, grid_mode, task_id, cfg, seed): job_idx
                    for job_idx, task_id, seed in jobs
                }
                for future in as_completed(future_map):
                    job_idx = future_map[future]
                    indexed_rows.append((job_idx, future.result()))
            indexed_rows.sort(key=lambda item: item[0])
            rows = [row for _, row in indexed_rows]
        payload["experiments"].append(
            {
                "name": name,
                "description": experiment_spec.get("description"),
                "config": {key: value for key, value in experiment_spec.items() if key not in {"name", "description"}},
                "aggregate": _aggregate_regime_rows(rows, cfg) if grid_mode == "regime_discovery" else _aggregate_rows(rows),
                "rows": rows,
            }
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one_hint_tutor experiment grid.")
    parser.add_argument("--grid", required=True, help="Path to a JSON/YAML grid spec.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--summary-md", default=None, help="Optional markdown summary path.")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker threads.")
    parser.add_argument(
        "--executor",
        choices=("thread", "process"),
        default="thread",
        help="Parallel executor kind.",
    )
    args = parser.parse_args()

    payload = run_grid(
        _load_grid_spec(args.grid),
        workers=max(1, int(args.workers)),
        executor_kind=str(args.executor),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.summary_md:
        md_path = Path(args.summary_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_markdown_summary(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "grid_mode": payload.get("grid_mode"),
                "n_experiments": len(payload.get("experiments", [])),
                "out": str(out_path),
                "summary_md": None if not args.summary_md else str(Path(args.summary_md)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
