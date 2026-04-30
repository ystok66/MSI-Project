from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import time
from typing import Iterable, List, Optional

from .config import OneHintConfig
from .protocol import run_one_hint_experiment
from .run_one_hint_experiment import _summary_payload


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None]
    return None if not xs else sum(xs) / len(xs)


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None]
    return None if not xs else float(median(xs))


def _safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or abs(float(den)) < 1e-12:
        return None
    return float(num) / float(den)


def run_calibration(
    task_id: str,
    seeds: List[int],
    refine_update_mode: Optional[str] = None,
) -> dict:
    rows = []
    start_all = time.perf_counter()
    for seed in seeds:
        cfg = OneHintConfig(seed=seed)
        if refine_update_mode is not None:
            cfg.refine_update_mode = str(refine_update_mode)
        start = time.perf_counter()
        result = run_one_hint_experiment(task_id=task_id, cfg=cfg, seed=seed)
        elapsed = time.perf_counter() - start
        summary = _summary_payload(result)

        plan = summary["plan"]
        pred = plan.get("planner_prediction") or {}
        tutor = summary["conditions"].get("tutor_T6") or {}
        tutor_trace = tutor.get("teach_trace_summary") or {}
        actual_attempt_trace = tutor_trace.get("attempt_policy_trace") or []
        actual_attempt_probs = [item.get("correct_prob") for item in actual_attempt_trace]
        actual_attempt_ranks = [item.get("correct_rank") for item in actual_attempt_trace]
        pred_attempt_probs = pred.get("pred_attempt_correct_prob_mean") or []
        pred_attempt_ranks = pred.get("pred_attempt_correct_rank_mean") or []

        row = {
            "seed": seed,
            "wall_time_sec": elapsed,
            "select_hint_wall_time": None if plan.get("planner_counters") is None else plan["planner_counters"]["select_hint_wall_time"],
            "selected_hint": plan.get("selected_hint"),
            "pred_p_success_T6": pred.get("pred_p_success_T6"),
            "pred_tau_mean": pred.get("pred_tau_mean"),
            "pred_correct_prob_after_hint_mean": pred.get("pred_correct_prob_after_hint_mean"),
            "pred_correct_rank_after_hint_mean": pred.get("pred_correct_rank_after_hint_mean"),
            "pred_attempt_correct_prob_mean": pred_attempt_probs,
            "pred_attempt_correct_rank_mean": pred_attempt_ranks,
            "pred_first_wrong_delta": None
            if len(pred_attempt_probs) < 2 or pred_attempt_probs[0] is None or pred_attempt_probs[1] is None
            else float(pred_attempt_probs[1]) - float(pred_attempt_probs[0]),
            "pred_first_wrong_ratio": _safe_ratio(
                None if len(pred_attempt_probs) < 2 else pred_attempt_probs[1],
                None if len(pred_attempt_probs) < 1 else pred_attempt_probs[0],
            ),
            "actual_tutor_success": tutor.get("success_within_limit"),
            "actual_tutor_tau": tutor.get("first_correct_attempt"),
            "actual_initial_correct_prob": tutor_trace.get("actual_initial_correct_prob"),
            "actual_initial_correct_rank": tutor_trace.get("actual_initial_correct_rank"),
            "actual_attempt_correct_prob": actual_attempt_probs,
            "actual_attempt_correct_rank": actual_attempt_ranks,
            "actual_first_wrong_delta": None
            if len(actual_attempt_probs) < 2 or actual_attempt_probs[0] is None or actual_attempt_probs[1] is None
            else float(actual_attempt_probs[1]) - float(actual_attempt_probs[0]),
            "actual_first_wrong_ratio": _safe_ratio(
                None if len(actual_attempt_probs) < 2 else actual_attempt_probs[1],
                None if len(actual_attempt_probs) < 1 else actual_attempt_probs[0],
            ),
        }
        rows.append(row)

    brier = []
    initial_prob_mae = []
    first_wrong_delta_gap = []
    first_wrong_ratio_gap = []
    tau_err = []
    for row in rows:
        pred_succ = row.get("pred_p_success_T6")
        actual_succ = row.get("actual_tutor_success")
        if pred_succ is not None and actual_succ is not None:
            y = 1.0 if actual_succ else 0.0
            brier.append((float(pred_succ) - y) ** 2)
        pred_init = row.get("pred_correct_prob_after_hint_mean")
        actual_init = row.get("actual_initial_correct_prob")
        if pred_init is not None and actual_init is not None:
            initial_prob_mae.append(abs(float(pred_init) - float(actual_init)))
        pred_delta = row.get("pred_first_wrong_delta")
        actual_delta = row.get("actual_first_wrong_delta")
        if pred_delta is not None and actual_delta is not None:
            first_wrong_delta_gap.append(float(pred_delta) - float(actual_delta))
        pred_ratio = row.get("pred_first_wrong_ratio")
        actual_ratio = row.get("actual_first_wrong_ratio")
        if pred_ratio is not None and actual_ratio is not None:
            first_wrong_ratio_gap.append(float(pred_ratio) - float(actual_ratio))
        if row.get("pred_tau_mean") is not None and row.get("actual_tutor_tau") is not None:
            tau_err.append(abs(float(row["pred_tau_mean"]) - float(row["actual_tutor_tau"])))

    return {
        "task_id": task_id,
        "seeds": list(seeds),
        "refine_update_mode": refine_update_mode,
        "total_wall_time_sec": time.perf_counter() - start_all,
        "aggregate": {
            "mean_wall_time_sec": _mean(row["wall_time_sec"] for row in rows),
            "mean_select_hint_wall_time": _mean(row["select_hint_wall_time"] for row in rows),
            "tutor_T6_success_rate": _mean(1.0 if row["actual_tutor_success"] else 0.0 for row in rows),
            "pred_success_brier": _mean(brier),
            "initial_prob_mae": _mean(initial_prob_mae),
            "pred_first_wrong_delta_mean": _mean(row["pred_first_wrong_delta"] for row in rows),
            "actual_first_wrong_delta_mean": _mean(row["actual_first_wrong_delta"] for row in rows),
            "pred_first_wrong_ratio_median": _median(row["pred_first_wrong_ratio"] for row in rows),
            "actual_first_wrong_ratio_median": _median(row["actual_first_wrong_ratio"] for row in rows),
            "first_wrong_delta_gap_mean": _mean(first_wrong_delta_gap),
            "first_wrong_ratio_gap_mean": _mean(first_wrong_ratio_gap),
            "tau_abs_err_on_successes": _mean(tau_err),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reveal-dynamics calibration for one_hint_tutor.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--refine-update-mode", default=None, choices=["proxy", "lazy_cls", "full_cls"])
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    payload = run_calibration(
        task_id=args.task,
        seeds=list(args.seeds),
        refine_update_mode=args.refine_update_mode,
    )
    text = json.dumps(payload, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
