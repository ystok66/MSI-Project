from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import copy
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import OneHintConfig
from .experiment_presets import apply_named_presets, parse_seed_spec
from .hint_planner import select_hint
from .interfaces import ConditionResult, EvalMetrics
from .learner_runner import run_teach_condition
from .menu_builder import build_exposure_sensitive_eval_items
from .protocol import finalize_prepared_experiment, prepare_one_hint_experiment
from .rollout import candidate_signature


def _base_cfg() -> OneHintConfig:
    cfg = OneHintConfig()
    cfg.planner_mode = "cascade"
    cfg.utility_mode = "advantage_mix"
    cfg.utility_mix_eta = 0.5
    cfg.common_randomness = True
    cfg.eval_aware = True
    cfg.prelearn_profile = "4"
    cfg.n_obs = 4
    cfg.teach_difficulty = "hard"
    cfg.menu_difficulty_mode = "rank_stratified"
    cfg.teach_probe_mode = "initial_rank"
    cfg.target_initial_rank_min = 5
    cfg.target_initial_rank_max = 12
    cfg.teach_menu_size = 20
    cfg.max_attempts_main = 5
    cfg.hint_count_budget = 1
    cfg.no_tutor_bonus_attempts = 1
    cfg.hint_mode = "combined"
    cfg.hint_families = (
        "free",
        "operator_probe",
        "answer_neighbor_nonanswer",
        "target_neighborhood_robust_filtered",
    )
    cfg.free_hint_pool_easy = 2
    cfg.free_hint_pool_medium = 3
    cfg.free_hint_pool_hard = 3
    cfg.operator_probe_limit = 8
    cfg.target_neighborhood_limit = 16
    cfg.answer_neighbor_limit = 8
    cfg.prefilter_enabled = True
    cfg.prefilter_top_k = 18
    cfg.objective_bucketed_prefilter = True
    cfg.prefilter_keep_fast = 6
    cfg.prefilter_keep_transfer = 6
    cfg.prefilter_keep_balanced = 6
    cfg.proxy_rollout_mode = "mc"
    cfg.proxy_n_rollouts = 24
    cfg.refine_enabled = True
    cfg.refine_top_k = 9
    cfg.objective_bucketed_refine = True
    cfg.refine_keep_fast = 3
    cfg.refine_keep_transfer = 3
    cfg.refine_keep_balanced = 3
    cfg.refine_update_mode = "first_reveal_cached_cls"
    cfg.refine_n_rollouts = 12
    cfg.first_reveal_top_b = 5
    cfg.transfer_eval_proxy_mode = "off"
    cfg.transfer_gate_mode = "default"
    cfg.transfer_delta_eval_min = 0.005
    cfg.transfer_success_floor = 0.15
    cfg.transfer_success_slack = 0.05
    cfg.transfer_eval_proxy_n_per_diff = 3
    cfg.transfer_eval_proxy_max_items = 9
    cfg.transfer_eval_proxy_beam_top_b = 3
    cfg.transfer_eval_proxy_beam_keep_l = 24
    cfg.lambda_fast_success_T = 8.0
    cfg.lambda_fast_tau2 = 10.0
    cfg.lambda_fast_wrong = 4.0
    cfg.lambda_fast_margin = 2.0
    cfg.lambda_fast_collapse = 0.0
    cfg.lambda_transfer_success = 2.0
    cfg.lambda_transfer_eval_cell = 6.0
    cfg.lambda_transfer_band = 4.0
    cfg.lambda_transfer_exposure = 2.0
    cfg.lambda_transfer_early = 4.0
    cfg.lambda_transfer_collapse = 3.0
    cfg.exposure_sensitive_eval_enabled = True
    cfg.exposure_sensitive_eval_n_per_diff = 6
    cfg.random_hard_n = 5
    cfg.random_same_pool_n = 5
    return cfg


def _experiment_cfg(name: str) -> OneHintConfig:
    cfg = copy.deepcopy(_base_cfg())
    if name == "fast_eta1":
        cfg.utility_mix_eta = 1.0
    elif name in {"norm_eta1_static_evalproxy_gate", "norm_eta100_static_evalproxy_gate"}:
        cfg.utility_mix_eta = 1.0
        cfg.utility_mix_normalize_components = True
        cfg.transfer_eval_proxy_mode = "static_subset"
        cfg.transfer_gate_mode = "eval_delta"
    elif name in {"norm_eta09_static_evalproxy_gate", "norm_eta90_static_evalproxy_gate"}:
        cfg.utility_mix_eta = 0.9
        cfg.utility_mix_normalize_components = True
        cfg.transfer_eval_proxy_mode = "static_subset"
        cfg.transfer_gate_mode = "eval_delta"
    elif name in {"norm_eta08_static_evalproxy_gate", "norm_eta80_static_evalproxy_gate"}:
        cfg.utility_mix_eta = 0.8
        cfg.utility_mix_normalize_components = True
        cfg.transfer_eval_proxy_mode = "static_subset"
        cfg.transfer_gate_mode = "eval_delta"
    elif name in {"norm_eta07_static_evalproxy_gate", "norm_eta70_static_evalproxy_gate"}:
        cfg.utility_mix_eta = 0.7
        cfg.utility_mix_normalize_components = True
        cfg.transfer_eval_proxy_mode = "static_subset"
        cfg.transfer_gate_mode = "eval_delta"
    elif name in {"norm_eta07_beam_evalproxy_gate", "norm_eta70_beam_evalproxy_gate"}:
        cfg.utility_mix_eta = 0.7
        cfg.utility_mix_normalize_components = True
        cfg.transfer_eval_proxy_mode = "beam_leaf_subset"
        cfg.transfer_gate_mode = "eval_delta"
        cfg.transfer_eval_proxy_refine_top_k = 5
    elif name in {"norm_eta07_beam_light_evalproxy_gate", "norm_eta70_beam_light_evalproxy_gate"}:
        cfg.utility_mix_eta = 0.7
        cfg.utility_mix_normalize_components = True
        cfg.transfer_eval_proxy_mode = "beam_leaf_subset"
        cfg.transfer_gate_mode = "eval_delta"
        cfg.refine_top_k = 5
        cfg.refine_keep_fast = 1
        cfg.refine_keep_transfer = 2
        cfg.refine_keep_balanced = 1
        cfg.transfer_eval_proxy_beam_top_b = 2
        cfg.transfer_eval_proxy_beam_keep_l = 8
        cfg.transfer_eval_proxy_n_per_diff = 2
        cfg.transfer_eval_proxy_max_items = 6
        cfg.transfer_eval_proxy_refine_top_k = 5
    elif name in {"norm_eta06_static_evalproxy_gate", "norm_eta60_static_evalproxy_gate"}:
        cfg.utility_mix_eta = 0.6
        cfg.utility_mix_normalize_components = True
        cfg.transfer_eval_proxy_mode = "static_subset"
        cfg.transfer_gate_mode = "eval_delta"
    elif name in {"eta09_static_evalproxy", "mix_eta09_static_evalproxy"}:
        cfg.utility_mix_eta = 0.9
        cfg.transfer_eval_proxy_mode = "static_subset"
    elif name in {"eta08_static_evalproxy", "mix_eta08_static_evalproxy"}:
        cfg.utility_mix_eta = 0.8
        cfg.transfer_eval_proxy_mode = "static_subset"
    elif name in {"eta07_static_evalproxy", "mix_eta07_static_evalproxy"}:
        cfg.utility_mix_eta = 0.7
        cfg.transfer_eval_proxy_mode = "static_subset"
    elif name in {"eta06_static_evalproxy", "mix_eta06_static_evalproxy"}:
        cfg.utility_mix_eta = 0.6
        cfg.transfer_eval_proxy_mode = "static_subset"
    elif name == "balanced_eta05":
        cfg.utility_mix_eta = 0.5
    elif name in {"norm_eta075_static_evalproxy_gate", "norm_eta75_static_evalproxy_gate"}:
        cfg.utility_mix_eta = 0.75
        cfg.utility_mix_normalize_components = True
        cfg.transfer_eval_proxy_mode = "static_subset"
        cfg.transfer_gate_mode = "eval_delta"
    elif name in {"norm_eta05_static_evalproxy_gate", "norm_eta50_static_evalproxy_gate"}:
        cfg.utility_mix_eta = 0.5
        cfg.utility_mix_normalize_components = True
        cfg.transfer_eval_proxy_mode = "static_subset"
        cfg.transfer_gate_mode = "eval_delta"
    elif name in {"norm_eta025_static_evalproxy_gate", "norm_eta25_static_evalproxy_gate"}:
        cfg.utility_mix_eta = 0.25
        cfg.utility_mix_normalize_components = True
        cfg.transfer_eval_proxy_mode = "static_subset"
        cfg.transfer_gate_mode = "eval_delta"
    elif name in {"norm_eta0_static_evalproxy_gate", "norm_eta00_static_evalproxy_gate"}:
        cfg.utility_mix_eta = 0.0
        cfg.utility_mix_normalize_components = True
        cfg.transfer_eval_proxy_mode = "static_subset"
        cfg.transfer_gate_mode = "eval_delta"
    elif name in {"norm_eta0_beam_evalproxy_gate", "norm_eta00_beam_evalproxy_gate"}:
        cfg.utility_mix_eta = 0.0
        cfg.utility_mix_normalize_components = True
        cfg.transfer_eval_proxy_mode = "beam_leaf_subset"
        cfg.transfer_gate_mode = "eval_delta"
        cfg.transfer_eval_proxy_refine_top_k = 5
    elif name in {"norm_eta0_beam_light_evalproxy_gate", "norm_eta00_beam_light_evalproxy_gate"}:
        cfg.utility_mix_eta = 0.0
        cfg.utility_mix_normalize_components = True
        cfg.transfer_eval_proxy_mode = "beam_leaf_subset"
        cfg.transfer_gate_mode = "eval_delta"
        cfg.refine_top_k = 5
        cfg.refine_keep_fast = 1
        cfg.refine_keep_transfer = 2
        cfg.refine_keep_balanced = 1
        cfg.transfer_eval_proxy_beam_top_b = 2
        cfg.transfer_eval_proxy_beam_keep_l = 8
        cfg.transfer_eval_proxy_n_per_diff = 2
        cfg.transfer_eval_proxy_max_items = 6
        cfg.transfer_eval_proxy_refine_top_k = 5
    elif name == "transfer_eta0":
        cfg.utility_mix_eta = 0.0
    elif name == "transfer_eta0_no_abstain":
        cfg.utility_mix_eta = 0.0
        cfg.allow_abstain = False
    elif name == "transfer_eta0_static_evalproxy_gate":
        cfg.utility_mix_eta = 0.0
        cfg.transfer_eval_proxy_mode = "static_subset"
        cfg.transfer_gate_mode = "eval_delta"
    elif name == "transfer_eta0_static_evalproxy_no_abstain":
        cfg.utility_mix_eta = 0.0
        cfg.transfer_eval_proxy_mode = "static_subset"
        cfg.transfer_gate_mode = "eval_delta"
        cfg.allow_abstain = False
    elif name == "transfer_eta0_beam_evalproxy_gate":
        cfg.utility_mix_eta = 0.0
        cfg.transfer_eval_proxy_mode = "beam_leaf_subset"
        cfg.transfer_gate_mode = "eval_delta"
        cfg.transfer_eval_proxy_refine_top_k = 5
    else:
        raise ValueError(f"Unknown frontier experiment: {name}")
    apply_named_presets(cfg)
    return cfg


def _hint_payload(hint) -> Optional[dict]:
    if hint is None:
        return None
    return {
        "kind": hint.kind,
        "difficulty": hint.difficulty,
        "words": list(hint.example.words),
        "metadata": dict(hint.metadata),
        "source_index": hint.source_index,
    }


def _wrong_pick_words(condition: Optional[ConditionResult], teach_case) -> List[List[str]]:
    if condition is None or condition.teach_trace_summary is None:
        return []
    option_by_index = {int(opt.index): opt for opt in teach_case.menu}
    summary = condition.teach_trace_summary
    words: List[List[str]] = []
    for pick_index, chosen_correct in zip(summary.actual_picks, summary.pick_correct_flags):
        if chosen_correct:
            continue
        option = option_by_index.get(int(pick_index))
        if option is None:
            continue
        words.append(list(option.text))
    return words


def _group_cell(metrics: Optional[EvalMetrics], group: str) -> Optional[float]:
    if metrics is None:
        return None
    if group not in metrics.cell_by_group:
        return None
    return float(metrics.cell_by_group[group])


def _group_exact(metrics: Optional[EvalMetrics], group: str) -> Optional[float]:
    if metrics is None:
        return None
    if group not in metrics.exact_by_group:
        return None
    return float(metrics.exact_by_group[group])


def _general_group_cell(metrics: Optional[EvalMetrics]) -> Optional[float]:
    if metrics is None:
        return None
    correct = 0
    total = 0
    for group, cell_total in metrics.cell_total_by_group.items():
        if group == "exposure_sensitive":
            continue
        correct += int(metrics.cell_correct_by_group.get(group, 0))
        total += int(cell_total)
    if total <= 0:
        return None
    return float(correct / total)


def _general_group_exact(metrics: Optional[EvalMetrics]) -> Optional[float]:
    if metrics is None:
        return None
    exact_correct = 0
    total = 0
    for group, count in metrics.n_items_by_group.items():
        if group == "exposure_sensitive":
            continue
        total += int(count)
        exact_correct += int(round(float(metrics.exact_by_group.get(group, 0.0)) * float(count)))
    if total <= 0:
        return None
    return float(exact_correct / total)


def _condition_metrics(condition: Optional[ConditionResult]) -> dict:
    metrics = None if condition is None else condition.eval_metrics
    summary = None if condition is None else condition.teach_trace_summary
    return {
        "success": 0.0 if condition is None else (1.0 if condition.success_within_limit else 0.0),
        "first_correct_attempt": None if condition is None else condition.first_correct_attempt,
        # Keep the realized wrong-pick count even on failed teach episodes: for the
        # frontier diagnostic we want exposure before termination, not success-only.
        "wrong_before_correct": None if condition is None else int(condition.n_wrong_before_correct),
        "teach_updates": None if summary is None else int(getattr(summary, "semantic_updates_applied", 0)),
        "eval_cell_all": None if metrics is None else float(metrics.cell_acc),
        "eval_exact_all": None if metrics is None else float(metrics.exact_acc),
        "eval_cell_exposure_sensitive": _group_cell(metrics, "exposure_sensitive"),
        "eval_exact_exposure_sensitive": _group_exact(metrics, "exposure_sensitive"),
        "eval_cell_general": _general_group_cell(metrics),
        "eval_exact_general": _general_group_exact(metrics),
    }


def _record_payload(record: Optional[dict]) -> Optional[dict]:
    if not isinstance(record, dict):
        return None
    return {
        "hint": _hint_payload(record.get("hint")),
        "selection_score": float(record.get("selection_score", record.get("utility", 0.0))),
        "utility": float(record.get("utility", 0.0)),
        "family": str((record.get("metadata", {}) or {}).get("family", record.get("kind"))),
        "pred_success_prob": float(record.get("success_prob", 0.0)),
        "pred_tau_le2_exact": float(record.get("pred_tau_le2_exact", 0.0)),
        "pred_wrong_before_correct": float(record.get("wrong_before_correct_mean", record.get("safe_wrong_mean", 0.0))),
        "pred_eval_cell": float(record.get("eval_cell_acc", 0.0)),
        "pred_eval_exact": float(record.get("eval_exact_acc", 0.0)),
        "pred_initial_margin": float(record.get("initial_correct_margin_mean", 0.0)),
        "fast_component_delta": None if record.get("fast_component_delta") is None else float(record.get("fast_component_delta", 0.0)),
        "transfer_component_delta": None if record.get("transfer_component_delta") is None else float(record.get("transfer_component_delta", 0.0)),
        "fast_component_z": None if record.get("fast_component_z") is None else float(record.get("fast_component_z", 0.0)),
        "transfer_component_z": None if record.get("transfer_component_z") is None else float(record.get("transfer_component_z", 0.0)),
    }


def _matching_conditions(conditions: Dict[str, ConditionResult], prefix: str) -> List[ConditionResult]:
    matched: List[Tuple[str, ConditionResult]] = []
    for name, condition in conditions.items():
        if name == prefix or name.startswith(f"{prefix}_rep_"):
            matched.append((name, condition))
    matched.sort(key=lambda item: item[0])
    return [condition for _, condition in matched]


def _mean_condition_metrics(conditions: Sequence[ConditionResult]) -> dict:
    rows = [_condition_metrics(condition) for condition in conditions]
    keys = [
        "success",
        "first_correct_attempt",
        "wrong_before_correct",
        "teach_updates",
        "eval_cell_all",
        "eval_exact_all",
        "eval_cell_exposure_sensitive",
        "eval_exact_exposure_sensitive",
        "eval_cell_general",
        "eval_exact_general",
    ]
    return {
        key: _mean([row.get(key) for row in rows])
        for key in keys
    }


def _single_row(task_id: str, seed: int, experiment_name: str) -> dict:
    cfg = _experiment_cfg(experiment_name)
    cfg.seed = int(seed)
    prepared = prepare_one_hint_experiment(task_id=task_id, cfg=cfg, seed=int(seed))

    reference_condition = run_teach_condition(
        base_learner=prepared.base_learner,
        context=prepared.context,
        teach_case=prepared.teach_case,
        max_attempts=int(getattr(prepared.cfg, "max_attempts_main", 5)) + int(getattr(prepared.cfg, "no_tutor_bonus_attempts", 1)),
        hint=None,
        eval_items=None,
        condition_name="reference_no_tutor_TplusH",
    )
    exposure_words = _wrong_pick_words(reference_condition, prepared.teach_case)
    exposure_eval_items = []
    if bool(getattr(prepared.cfg, "exposure_sensitive_eval_enabled", False)):
        exposure_eval_items = build_exposure_sensitive_eval_items(
            prepared.context,
            exposure_words,
            prepared.cfg,
            np.random.default_rng(int(prepared.seed_bundle.get("eval", prepared.seed)) + 913),
        )

    prepared.eval_items = list(prepared.eval_items) + list(exposure_eval_items)
    plan = select_hint(
        posterior=prepared.posterior,
        context=prepared.context,
        teach_case=prepared.teach_case,
        eval_items=prepared.eval_items,
        cfg=prepared.cfg,
        rng=np.random.default_rng(int(prepared.seed_bundle.get("plan", prepared.seed))),
    )
    result = finalize_prepared_experiment(
        prepared,
        plan,
        rng=np.random.default_rng(int(prepared.seed_bundle.get("baseline", prepared.seed))),
    )

    tutor_t = result.conditions.get("tutor_T")
    no_tutor_tplush = result.conditions.get("no_tutor_TplusH")
    random_same_pool_draws = _matching_conditions(result.conditions, "random_same_pool_hint_T")
    selected_sig = candidate_signature(result.plan.selected_hint)
    selected_record = next(
        (item for item in result.plan.candidate_scores if candidate_signature(item.get("hint")) == selected_sig),
        None,
    ) if result.plan.selected_hint is not None else None
    best_pred_eval_record = None
    if result.plan.candidate_scores:
        best_pred_eval_record = max(
            result.plan.candidate_scores,
            key=lambda item: (
                float(item.get("eval_cell_acc", 0.0)),
                float(item.get("selection_score", item.get("utility", 0.0))),
            ),
        )
    best_pred_eval_actual = None
    if best_pred_eval_record is not None and best_pred_eval_record.get("hint") is not None:
        best_sig = candidate_signature(best_pred_eval_record.get("hint"))
        if result.plan.selected_hint is not None and best_sig == selected_sig:
            best_pred_eval_actual = tutor_t
        else:
            best_pred_eval_actual = run_teach_condition(
                base_learner=prepared.base_learner,
                context=prepared.context,
                teach_case=prepared.teach_case,
                max_attempts=int(getattr(prepared.cfg, "max_attempts_main", 5)),
                hint=best_pred_eval_record.get("hint"),
                eval_items=prepared.eval_items,
                condition_name="best_pred_eval_T",
            )
    planner_prediction = result.plan.planner_prediction
    hint_used = result.plan.selected_hint is not None

    return {
        "task_id": task_id,
        "seed": int(seed),
        "experiment": experiment_name,
        "selected_hint": _hint_payload(result.plan.selected_hint),
        "hint_used": bool(hint_used),
        "abstain_reason": None if planner_prediction is None else planner_prediction.abstain_reason,
        "selected_utility": float(result.plan.selected_utility),
        "delta_vs_no_hint": float(result.plan.delta_vs_no_hint),
        "exposure_reference_wrong_words": exposure_words,
        "exposure_eval_count": len(exposure_eval_items),
        "selected_record": _record_payload(selected_record),
        "best_pred_eval_record": _record_payload(best_pred_eval_record),
        "best_pred_eval_actual": _condition_metrics(best_pred_eval_actual),
        "conditions": {
            "tutor_T": _condition_metrics(tutor_t),
            "no_tutor_TplusH": _condition_metrics(no_tutor_tplush),
            "random_same_pool_hint_T": _mean_condition_metrics(random_same_pool_draws),
        },
        "top_candidates": [
            _record_payload(item)
            for item in list(result.plan.candidate_scores[:8])
        ],
    }


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return None if not vals else float(mean(vals))


def _aggregate_rows(rows: Sequence[dict]) -> dict:
    def _metric(condition_name: str, key: str) -> List[Optional[float]]:
        return [row.get("conditions", {}).get(condition_name, {}).get(key) for row in rows]

    def _row_metric(key: str) -> List[Optional[float]]:
        return [row.get(key) for row in rows]

    def _nested_row_metric(parent: str, key: str) -> List[Optional[float]]:
        vals: List[Optional[float]] = []
        for row in rows:
            container = row.get(parent)
            vals.append(None if not isinstance(container, dict) else container.get(key))
        return vals

    selected_family_counts: Dict[str, int] = {}
    abstain_reason_counts: Dict[str, int] = {}
    for row in rows:
        hint = row.get("selected_hint")
        if not isinstance(hint, dict):
            reason = row.get("abstain_reason")
            if reason is not None:
                abstain_reason_counts[str(reason)] = abstain_reason_counts.get(str(reason), 0) + 1
            continue
        family = str(hint.get("metadata", {}).get("family", hint.get("kind")))
        selected_family_counts[family] = selected_family_counts.get(family, 0) + 1

    hint_used_rate = _mean([1.0 if bool(row.get("hint_used")) else 0.0 for row in rows])

    return {
        "rows": len(rows),
        "hint_used_rate": hint_used_rate,
        "tutor_success": _mean(_metric("tutor_T", "success")),
        "no_tutor_TplusH_success": _mean(_metric("no_tutor_TplusH", "success")),
        "random_same_pool_success": _mean(_metric("random_same_pool_hint_T", "success")),
        "tutor_tau_mean": _mean(_metric("tutor_T", "first_correct_attempt")),
        "tutor_wrong_before_correct_mean": _mean(_metric("tutor_T", "wrong_before_correct")),
        "tutor_teach_updates_mean": _mean(_metric("tutor_T", "teach_updates")),
        "tutor_eval_cell_all": _mean(_metric("tutor_T", "eval_cell_all")),
        "tutor_eval_cell_exposure_sensitive": _mean(_metric("tutor_T", "eval_cell_exposure_sensitive")),
        "tutor_eval_cell_general": _mean(_metric("tutor_T", "eval_cell_general")),
        "selected_utility_mean": _mean(_row_metric("selected_utility")),
        "selected_pred_eval_cell_mean": _mean(_nested_row_metric("selected_record", "pred_eval_cell")),
        "selected_pred_success_mean": _mean(_nested_row_metric("selected_record", "pred_success_prob")),
        "selected_pred_tau2_mean": _mean(_nested_row_metric("selected_record", "pred_tau_le2_exact")),
        "selected_fast_component_delta_mean": _mean(_nested_row_metric("selected_record", "fast_component_delta")),
        "selected_transfer_component_delta_mean": _mean(_nested_row_metric("selected_record", "transfer_component_delta")),
        "selected_fast_component_z_mean": _mean(_nested_row_metric("selected_record", "fast_component_z")),
        "selected_transfer_component_z_mean": _mean(_nested_row_metric("selected_record", "transfer_component_z")),
        "best_pred_eval_cell_mean": _mean(_nested_row_metric("best_pred_eval_record", "pred_eval_cell")),
        "best_pred_eval_actual_cell_mean": _mean(_nested_row_metric("best_pred_eval_actual", "eval_cell_all")),
        "best_pred_eval_actual_success_mean": _mean(_nested_row_metric("best_pred_eval_actual", "success")),
        "delta_success_vs_no_tutor_TplusH": (
            (_mean(_metric("tutor_T", "success")) or 0.0)
            - (_mean(_metric("no_tutor_TplusH", "success")) or 0.0)
        ),
        "delta_eval_cell_vs_no_tutor_TplusH": (
            (_mean(_metric("tutor_T", "eval_cell_all")) or 0.0)
            - (_mean(_metric("no_tutor_TplusH", "eval_cell_all")) or 0.0)
        ),
        "delta_success_vs_random_same_pool": (
            (_mean(_metric("tutor_T", "success")) or 0.0)
            - (_mean(_metric("random_same_pool_hint_T", "success")) or 0.0)
        ),
        "delta_eval_cell_vs_random_same_pool": (
            (_mean(_metric("tutor_T", "eval_cell_all")) or 0.0)
            - (_mean(_metric("random_same_pool_hint_T", "eval_cell_all")) or 0.0)
        ),
        "selected_family_counts": dict(sorted(selected_family_counts.items())),
        "abstain_reason_counts": dict(sorted(abstain_reason_counts.items())),
        "exposure_eval_count_mean": _mean([row.get("exposure_eval_count") for row in rows]),
        "rows_with_exposure_eval": sum(1 for row in rows if int(row.get("exposure_eval_count", 0)) > 0),
    }


def _markdown_summary(payload: dict) -> str:
    lines: List[str] = []
    lines.append("# Fast / Transfer Frontier Summary")
    lines.append("")
    lines.append(f"- Task: `{payload.get('task_id')}`")
    lines.append(f"- Seeds: `{payload.get('seeds')}`")
    lines.append("")
    for experiment in payload.get("experiments", []):
        agg = experiment.get("aggregate", {})
        lines.append(f"## {experiment['name']}")
        lines.append("")
        lines.append(f"- Rows: {agg.get('rows')}")
        lines.append(f"- Hint used rate: {agg.get('hint_used_rate')}")
        lines.append(f"- Selected utility mean: {agg.get('selected_utility_mean')}")
        lines.append(f"- Tutor success: {agg.get('tutor_success')}")
        lines.append(f"- No-tutor T+H success: {agg.get('no_tutor_TplusH_success')}")
        lines.append(f"- Random same-pool success: {agg.get('random_same_pool_success')}")
        lines.append(f"- Delta success vs no-tutor T+H: {agg.get('delta_success_vs_no_tutor_TplusH')}")
        lines.append(f"- Delta success vs random same-pool: {agg.get('delta_success_vs_random_same_pool')}")
        lines.append(f"- Tutor tau mean: {agg.get('tutor_tau_mean')}")
        lines.append(f"- Tutor wrong-before-correct mean: {agg.get('tutor_wrong_before_correct_mean')}")
        lines.append(f"- Tutor teach-updates mean: {agg.get('tutor_teach_updates_mean')}")
        lines.append(f"- Tutor eval cell all: {agg.get('tutor_eval_cell_all')}")
        lines.append(f"- Tutor eval cell exposure-sensitive: {agg.get('tutor_eval_cell_exposure_sensitive')}")
        lines.append(f"- Tutor eval cell general: {agg.get('tutor_eval_cell_general')}")
        lines.append(f"- Delta eval cell vs no-tutor T+H: {agg.get('delta_eval_cell_vs_no_tutor_TplusH')}")
        lines.append(f"- Delta eval cell vs random same-pool: {agg.get('delta_eval_cell_vs_random_same_pool')}")
        lines.append(f"- Selected predicted eval cell mean: {agg.get('selected_pred_eval_cell_mean')}")
        lines.append(f"- Selected predicted success mean: {agg.get('selected_pred_success_mean')}")
        lines.append(f"- Selected predicted P(tau<=2) mean: {agg.get('selected_pred_tau2_mean')}")
        lines.append(f"- Selected fast component delta mean: {agg.get('selected_fast_component_delta_mean')}")
        lines.append(f"- Selected transfer component delta mean: {agg.get('selected_transfer_component_delta_mean')}")
        lines.append(f"- Selected fast component z mean: {agg.get('selected_fast_component_z_mean')}")
        lines.append(f"- Selected transfer component z mean: {agg.get('selected_transfer_component_z_mean')}")
        lines.append(f"- Best predicted-eval cell mean: {agg.get('best_pred_eval_cell_mean')}")
        lines.append(f"- Best predicted-eval actual cell mean: {agg.get('best_pred_eval_actual_cell_mean')}")
        lines.append(f"- Best predicted-eval actual success mean: {agg.get('best_pred_eval_actual_success_mean')}")
        lines.append(f"- Skipped updates vs transfer: {agg.get('skipped_updates_vs_transfer')}")
        lines.append(f"- Skipped wrong-before-correct vs transfer: {agg.get('skipped_wrong_before_correct_vs_transfer')}")
        lines.append(f"- Exposure eval count mean: {agg.get('exposure_eval_count_mean')}")
        lines.append(f"- Rows with exposure eval: {agg.get('rows_with_exposure_eval')}")
        lines.append(f"- Selected families: `{json.dumps(agg.get('selected_family_counts', {}), ensure_ascii=False)}`")
        lines.append(f"- Abstain reasons: `{json.dumps(agg.get('abstain_reason_counts', {}), ensure_ascii=False)}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _worker(job: Tuple[str, int, str]) -> dict:
    task_id, seed, experiment_name = job
    return _single_row(task_id=task_id, seed=seed, experiment_name=experiment_name)


def run_frontier(
    task_id: str,
    seeds: Sequence[int],
    workers: int,
    executor_kind: str,
    experiment_names: Optional[Sequence[str]] = None,
) -> dict:
    if experiment_names is None:
        experiment_names = [
            "fast_eta1",
            "balanced_eta05",
            "transfer_eta0",
            "transfer_eta0_no_abstain",
            "transfer_eta0_static_evalproxy_gate",
            "transfer_eta0_static_evalproxy_no_abstain",
            "transfer_eta0_beam_evalproxy_gate",
            "norm_eta1_static_evalproxy_gate",
            "norm_eta075_static_evalproxy_gate",
            "norm_eta05_static_evalproxy_gate",
            "norm_eta025_static_evalproxy_gate",
            "norm_eta0_static_evalproxy_gate",
        ]
    jobs = [(task_id, int(seed), name) for name in experiment_names for seed in seeds]

    rows: List[dict] = []
    if max(1, int(workers)) <= 1:
        rows = [_worker(job) for job in jobs]
    else:
        executor_cls = ThreadPoolExecutor if str(executor_kind).lower() != "process" else ProcessPoolExecutor
        with executor_cls(max_workers=max(1, int(workers))) as executor:
            future_map = {executor.submit(_worker, job): job for job in jobs}
            for future in as_completed(future_map):
                rows.append(future.result())

    payload = {
        "task_id": task_id,
        "seeds": [int(seed) for seed in seeds],
        "experiments": [],
    }
    for name in experiment_names:
        exp_rows = [row for row in rows if row["experiment"] == name]
        exp_rows.sort(key=lambda row: int(row["seed"]))
        payload["experiments"].append(
            {
                "name": name,
                "rows": exp_rows,
                "aggregate": _aggregate_rows(exp_rows),
            }
        )
    transfer_agg = None
    for experiment in payload["experiments"]:
        if experiment["name"] in {"norm_eta0_static_evalproxy_gate", "transfer_eta0_static_evalproxy_gate"}:
            transfer_agg = experiment.get("aggregate", {})
            break
    if transfer_agg is not None:
        transfer_updates = transfer_agg.get("tutor_teach_updates_mean")
        transfer_wrong = transfer_agg.get("tutor_wrong_before_correct_mean")
        for experiment in payload["experiments"]:
            agg = experiment.get("aggregate", {})
            updates = agg.get("tutor_teach_updates_mean")
            wrong = agg.get("tutor_wrong_before_correct_mean")
            agg["skipped_updates_vs_transfer"] = (
                None if transfer_updates is None or updates is None else float(transfer_updates) - float(updates)
            )
            agg["skipped_wrong_before_correct_vs_transfer"] = (
                None if transfer_wrong is None or wrong is None else float(transfer_wrong) - float(wrong)
            )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fast/transfer frontier diagnostics.")
    parser.add_argument("--task", default="000001")
    parser.add_argument("--seeds", default="4:5")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--executor", choices=["thread", "process"], default="process")
    parser.add_argument(
        "--experiments",
        default="",
        help="Comma-separated experiment names. Defaults to the full fast/transfer frontier set.",
    )
    parser.add_argument(
        "--out",
        default="cls_option_tutor/one_hint_tutor/grids/final_presentation/fast_transfer_frontier_seed4_results.json",
    )
    parser.add_argument(
        "--summary-md",
        default="cls_option_tutor/one_hint_tutor/grids/final_presentation/fast_transfer_frontier_seed4_summary.md",
    )
    args = parser.parse_args()

    seeds = parse_seed_spec(args.seeds)
    experiment_names = [name.strip() for name in str(args.experiments).split(",") if name.strip()] or None
    payload = run_frontier(
        task_id=str(args.task),
        seeds=seeds,
        workers=int(args.workers),
        executor_kind=str(args.executor),
        experiment_names=experiment_names,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = Path(args.summary_md)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_markdown_summary(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "task": str(args.task),
                "out": str(out_path),
                "summary_md": str(summary_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
