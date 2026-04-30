from __future__ import annotations

import copy
from collections import Counter
import math
from itertools import combinations, product
from statistics import stdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .baselines import attach_failure_taxonomy
from .experiment_presets import apply_named_presets
from .experiment_presets import resolved_no_tutor_tplush_limit
from .hint_planner import filter_candidate_records_for_stage, select_hint
from .hint_space import build_hint_candidates, build_menu_hint_candidates
from .interfaces import ConditionResult, HintCandidate, HintPlanResult
from .learner_runner import run_teach_condition
from .protocol import finalize_prepared_experiment, prepare_one_hint_experiment
from .rollout import build_score_tables_for_candidates, candidate_signature, prefilter_score_table_under_profiles


def _hint_payload(hint: Optional[HintCandidate]) -> Optional[dict]:
    if hint is None:
        return None
    return {
        "kind": hint.kind,
        "difficulty": hint.difficulty,
        "words": list(hint.example.words),
        "metadata": dict(hint.metadata),
        "source_index": hint.source_index,
    }


def _condition_payload(condition: Optional[ConditionResult]) -> Optional[dict]:
    if condition is None:
        return None
    return {
        "condition": condition.condition,
        "first_correct_attempt": condition.first_correct_attempt,
        "success_within_limit": condition.success_within_limit,
        "n_wrong_before_correct": condition.n_wrong_before_correct,
        "eval_exact_acc": None if condition.eval_metrics is None else float(condition.eval_metrics.exact_acc),
        "eval_cell_acc": None if condition.eval_metrics is None else float(condition.eval_metrics.cell_acc),
        "failure_type": condition.failure_type,
        "failure_details": dict(condition.failure_details),
    }


def _family_name(hint: Optional[HintCandidate]) -> str:
    if hint is None:
        return "none"
    return str(hint.metadata.get("family", hint.kind))


def _family_counts(candidates: Sequence[HintCandidate]) -> Dict[str, int]:
    counter = Counter(_family_name(hint) for hint in candidates)
    return dict(sorted(counter.items()))


def _bounded_band_success(condition: Optional[ConditionResult], cfg) -> float:
    if condition is None or condition.first_correct_attempt is None:
        return 0.0
    tau = int(condition.first_correct_attempt)
    return 1.0 if int(cfg.target_tau_min) <= tau <= int(cfg.target_tau_max) else 0.0


def _bounded_early_success(condition: Optional[ConditionResult], cfg) -> float:
    if condition is None or condition.first_correct_attempt is None:
        return 0.0
    return 1.0 if int(condition.first_correct_attempt) < int(cfg.target_tau_min) else 0.0


def _first_available_condition(
    conditions: Dict[str, ConditionResult],
    *names: str,
) -> Optional[ConditionResult]:
    for name in names:
        condition = conditions.get(name)
        if condition is not None:
            return condition
    return None


def _matching_conditions(
    conditions: Dict[str, ConditionResult],
    prefix: str,
) -> List[ConditionResult]:
    matched: List[Tuple[str, ConditionResult]] = []
    for name, condition in conditions.items():
        if name == prefix or name.startswith(f"{prefix}_rep_"):
            matched.append((name, condition))
    matched.sort(key=lambda item: item[0])
    return [condition for _, condition in matched]


def _condition_success(condition: Optional[ConditionResult]) -> float:
    return 0.0 if condition is None else (1.0 if condition.success_within_limit else 0.0)


def _condition_eval_exact(condition: Optional[ConditionResult]) -> float:
    return 0.0 if condition is None or condition.eval_metrics is None else float(condition.eval_metrics.exact_acc)


def _condition_eval_cell(condition: Optional[ConditionResult]) -> float:
    return 0.0 if condition is None or condition.eval_metrics is None else float(condition.eval_metrics.cell_acc)


def _condition_first_correct(condition: Optional[ConditionResult]) -> Optional[float]:
    return None if condition is None or condition.first_correct_attempt is None else float(condition.first_correct_attempt)


def _series_mean(
    conditions: Sequence[ConditionResult],
    getter,
) -> float:
    vals = []
    for condition in conditions:
        value = getter(condition)
        if value is None:
            continue
        vals.append(float(value))
    return 0.0 if not vals else float(sum(vals) / len(vals))


def _series_std(
    conditions: Sequence[ConditionResult],
    getter,
) -> float:
    vals = []
    for condition in conditions:
        value = getter(condition)
        if value is None:
            continue
        vals.append(float(value))
    return 0.0 if len(vals) <= 1 else float(stdev(vals))


def _actual_soft_tau_score(condition: Optional[ConditionResult], cfg) -> float:
    if condition is None or condition.first_correct_attempt is None:
        return 0.0
    center = float(getattr(cfg, "soft_tau_center", getattr(cfg, "target_attempt", 6)))
    sigma = max(float(getattr(cfg, "soft_tau_sigma", getattr(cfg, "sigma_tau", 1.0))), 1e-6)
    tau = float(condition.first_correct_attempt)
    return float(math.exp(-((tau - center) ** 2) / (2.0 * sigma * sigma)))


def _actual_early_no_transfer(
    condition: Optional[ConditionResult],
    baseline_condition: Optional[ConditionResult],
    cfg,
) -> float:
    if condition is None or condition.first_correct_attempt is None:
        return 0.0
    if int(condition.first_correct_attempt) >= int(getattr(cfg, "target_tau_min", 3)):
        return 0.0
    tutor_eval_cell = _condition_eval_cell(condition)
    baseline_eval_cell = _condition_eval_cell(baseline_condition)
    margin = float(getattr(cfg, "early_success_eval_margin", 0.01))
    return 1.0 if tutor_eval_cell <= baseline_eval_cell + margin else 0.0


def _oracle_metric_names() -> Tuple[str, ...]:
    return ("success", "band", "eval_cell", "fast_success", "composite")


def _oracle_rank_tuple_for_metric(
    metric: str,
    tutor_t6: Optional[ConditionResult],
    tutor_unlimited: Optional[ConditionResult],
    cfg,
) -> Tuple[float, ...]:
    band = _bounded_band_success(tutor_t6, cfg)
    success = 0.0 if tutor_t6 is None else (1.0 if tutor_t6.success_within_limit else 0.0)
    collapse = 1.0 if tutor_t6 is not None and tutor_t6.failure_type == "post_reveal_collapse" else 0.0
    early = _bounded_early_success(tutor_t6, cfg)
    eval_cell = 0.0 if tutor_t6 is None or tutor_t6.eval_metrics is None else float(tutor_t6.eval_metrics.cell_acc)
    eval_exact = 0.0 if tutor_t6 is None or tutor_t6.eval_metrics is None else float(tutor_t6.eval_metrics.exact_acc)
    tau_t6 = float(tutor_t6.first_correct_attempt) if tutor_t6 is not None and tutor_t6.first_correct_attempt is not None else 999.0
    tau_unlimited = (
        float(tutor_unlimited.first_correct_attempt)
        if tutor_unlimited is not None and tutor_unlimited.first_correct_attempt is not None
        else 999.0
    )
    if str(metric) == "success":
        return (
            success,
            band,
            -collapse,
            -early,
            eval_cell,
            eval_exact,
            -tau_t6,
            -tau_unlimited,
        )
    if str(metric) == "band":
        return (
            band,
            success,
            -collapse,
            -early,
            eval_cell,
            eval_exact,
            -tau_t6,
            -tau_unlimited,
        )
    if str(metric) == "eval_cell":
        return (
            eval_cell,
            success,
            band,
            -collapse,
            -early,
            eval_exact,
            -tau_t6,
            -tau_unlimited,
        )
    if str(metric) == "fast_success":
        return (
            success,
            -collapse,
            early,
            -tau_t6,
            band,
            eval_cell,
            eval_exact,
            -tau_unlimited,
        )
    return (
        band,
        success,
        -collapse,
        -early,
        eval_cell,
        eval_exact,
        -tau_t6,
        -tau_unlimited,
    )


def _oracle_numeric_score_for_metric(
    metric: str,
    tutor_t6: Optional[ConditionResult],
    tutor_unlimited: Optional[ConditionResult],
    cfg,
) -> float:
    band = _bounded_band_success(tutor_t6, cfg)
    success = 0.0 if tutor_t6 is None else (1.0 if tutor_t6.success_within_limit else 0.0)
    collapse = 1.0 if tutor_t6 is not None and tutor_t6.failure_type == "post_reveal_collapse" else 0.0
    early = _bounded_early_success(tutor_t6, cfg)
    eval_cell = 0.0 if tutor_t6 is None or tutor_t6.eval_metrics is None else float(tutor_t6.eval_metrics.cell_acc)
    eval_exact = 0.0 if tutor_t6 is None or tutor_t6.eval_metrics is None else float(tutor_t6.eval_metrics.exact_acc)
    tau_t6 = float(tutor_t6.first_correct_attempt) if tutor_t6 is not None and tutor_t6.first_correct_attempt is not None else 999.0
    tau_unlimited = (
        float(tutor_unlimited.first_correct_attempt)
        if tutor_unlimited is not None and tutor_unlimited.first_correct_attempt is not None
        else 999.0
    )
    if str(metric) == "success":
        return (
            100.0 * success
            + 40.0 * band
            - 10.0 * collapse
            - 8.0 * early
            + 5.0 * eval_cell
            + 2.0 * eval_exact
            - 0.05 * tau_t6
            - 0.01 * tau_unlimited
        )
    if str(metric) == "band":
        return (
            100.0 * band
            + 20.0 * success
            - 10.0 * collapse
            - 8.0 * early
            + 5.0 * eval_cell
            + 2.0 * eval_exact
            - 0.05 * tau_t6
            - 0.01 * tau_unlimited
        )
    if str(metric) == "eval_cell":
        return (
            100.0 * eval_cell
            + 20.0 * success
            + 10.0 * band
            - 10.0 * collapse
            - 8.0 * early
            + 2.0 * eval_exact
            - 0.05 * tau_t6
            - 0.01 * tau_unlimited
        )
    if str(metric) == "fast_success":
        return (
            100.0 * success
            - 10.0 * collapse
            + 20.0 * early
            - 0.10 * tau_t6
            + 5.0 * band
            + 1.0 * eval_cell
            + 0.5 * eval_exact
            - 0.01 * tau_unlimited
        )
    return (
        100.0 * band
        + 20.0 * success
        - 10.0 * collapse
        - 8.0 * early
        + 5.0 * eval_cell
        + 2.0 * eval_exact
        - 0.05 * tau_t6
        - 0.01 * tau_unlimited
    )


def _oracle_metric_payload(record: dict) -> dict:
    return {
        "oracle_metric": str(record.get("oracle_metric", "composite")),
        "selected_hints": [_hint_payload(hint) for hint in list(record.get("selected_hints", []))],
        "post_first_wrong_hint": _hint_payload(record.get("post_first_wrong_hint")),
        "candidate_pool_size": int(record.get("candidate_pool_size", 0)),
        "candidate_family_counts": dict(record.get("candidate_family_counts", {})),
        "conditions": {
            name: _condition_payload(cond)
            for name, cond in dict(record.get("conditions", {})).items()
        },
        "score": float(record.get("score", 0.0)),
    }


def _select_oracle_record_for_metric(
    evaluated_records: Sequence[dict],
    candidate_pool_size: int,
    candidate_family_counts: Dict[str, int],
    cfg,
    metric: str,
) -> dict:
    if not evaluated_records:
        raise RuntimeError("oracle selection called without evaluated records")
    best = None
    for record in evaluated_records:
        conditions = record["conditions"]
        rank_key = _oracle_rank_tuple_for_metric(
            metric,
            conditions.get("tutor_T6"),
            conditions.get("tutor_unlimited"),
            cfg,
        )
        numeric = _oracle_numeric_score_for_metric(
            metric,
            conditions.get("tutor_T6"),
            conditions.get("tutor_unlimited"),
            cfg,
        )
        enriched = dict(record)
        enriched["rank_key"] = rank_key
        enriched["score"] = float(numeric)
        enriched["oracle_metric"] = str(metric)
        if best is None or enriched["rank_key"] > best["rank_key"]:
            best = enriched
    assert best is not None
    best["candidate_pool_size"] = int(candidate_pool_size)
    best["candidate_family_counts"] = dict(candidate_family_counts)
    return best


def _evaluate_actual_hint_bundle(
    prepared,
    pre_hints: Sequence[HintCandidate],
    post_first_wrong_hint: Optional[HintCandidate] = None,
) -> Dict[str, ConditionResult]:
    tutor_unlimited = run_teach_condition(
        base_learner=prepared.base_learner,
        context=prepared.context,
        teach_case=prepared.teach_case,
        max_attempts=int(prepared.cfg.teach_menu_size),
        hint=None,
        hints=list(pre_hints),
        post_first_wrong_hint=post_first_wrong_hint,
        eval_items=None,
        condition_name="tutor_unlimited",
    )
    tutor_t6 = run_teach_condition(
        base_learner=prepared.base_learner,
        context=prepared.context,
        teach_case=prepared.teach_case,
        max_attempts=int(prepared.cfg.max_attempts_main),
        hint=None,
        hints=list(pre_hints),
        post_first_wrong_hint=post_first_wrong_hint,
        eval_items=prepared.eval_items,
        condition_name="tutor_T6",
    )
    conditions = {
        "tutor_unlimited": tutor_unlimited,
        "tutor_T6": tutor_t6,
    }
    attach_failure_taxonomy(conditions, prepared.cfg)
    return conditions


def _candidate_pool_cfg(prepared, families: Optional[Sequence[str]] = None):
    cfg = copy.deepcopy(prepared.cfg)
    if families:
        families = tuple(str(part).strip() for part in families if str(part).strip())
        cfg.hint_families = families
        if len(families) == 1:
            cfg.hint_mode = families[0]
        else:
            cfg.hint_mode = "combined"
    apply_named_presets(cfg)
    return cfg


def _rank_candidates_for_oracle(
    prepared,
    families: Optional[Sequence[str]] = None,
    allow_correct_hint: bool = False,
    candidate_limit: Optional[int] = None,
    seed_offset: int = 19391,
) -> List[HintCandidate]:
    cfg = _candidate_pool_cfg(prepared, families=families)
    rng = np.random.default_rng(int(prepared.seed) + int(seed_offset))
    if allow_correct_hint and str(cfg.hint_mode) != "menu_correct_ceiling":
        candidates = build_menu_hint_candidates(
            context=prepared.context,
            teach_case=prepared.teach_case,
            allow_correct_hint=True,
            wrong_limit=0,
            correct_limit=max(1, int(getattr(cfg, "menu_correct_ceiling_limit", 1))),
        )
    else:
        candidates = build_hint_candidates(prepared.context, prepared.teach_case, cfg, rng)
    if not candidates:
        return []

    limit = int(candidate_limit or getattr(cfg, "oracle_candidate_limit", len(candidates)))
    all_candidates: List[HintCandidate] = list(candidates)
    cache = build_score_tables_for_candidates(
        posterior=prepared.posterior,
        candidates=all_candidates,
        teach_case=prepared.teach_case,
        cfg=cfg,
        counters=None,
    )
    profile_weights = prepared.posterior.profiles_for_stage("prefilter", cfg)
    records: List[dict] = []
    for hint in candidates:
        stats = prefilter_score_table_under_profiles(cache[candidate_signature(hint)], profile_weights, cfg)
        records.append(
            {
                "hint": hint,
                "kind": hint.kind,
                "difficulty": hint.difficulty,
                "source_index": hint.source_index,
                "metadata": dict(hint.metadata),
                **stats,
                "utility": float(stats.get("prefilter_score", 0.0)),
            }
        )
    records = filter_candidate_records_for_stage(records, cfg, stage="prefilter")
    records.sort(key=lambda item: float(item.get("utility", 0.0)), reverse=True)
    return [record["hint"] for record in records[: max(1, limit)]]


def _correct_menu_hint(prepared) -> Optional[HintCandidate]:
    candidates = build_menu_hint_candidates(
        context=prepared.context,
        teach_case=prepared.teach_case,
        allow_correct_hint=True,
        wrong_limit=0,
        correct_limit=1,
    )
    return None if not candidates else candidates[0]


def _evaluate_planner_reference(prepared) -> dict:
    rng = np.random.default_rng(int(prepared.seed_bundle.get("plan", prepared.seed)) + 99017)
    plan = select_hint(
        posterior=prepared.posterior,
        context=prepared.context,
        teach_case=prepared.teach_case,
        eval_items=prepared.eval_items,
        cfg=prepared.cfg,
        rng=rng,
    )
    pre_hints: List[HintCandidate] = []
    if plan.selected_hint is not None:
        pre_hints.append(plan.selected_hint)
    conditions = _evaluate_actual_hint_bundle(prepared, pre_hints=pre_hints)
    metric = str(getattr(prepared.cfg, "oracle_metric", "composite"))
    score = _oracle_numeric_score_for_metric(metric, conditions.get("tutor_T6"), conditions.get("tutor_unlimited"), prepared.cfg)
    return {
        "selected_hint": _hint_payload(plan.selected_hint),
        "oracle_metric": metric,
        "score": float(score),
        "conditions": {name: _condition_payload(cond) for name, cond in conditions.items()},
        "abstained": plan.selected_hint is None,
        "abstain_reason": None if plan.planner_prediction is None else plan.planner_prediction.abstain_reason,
    }


def _regime_discovery_metrics(result, oracle_records: Optional[dict], cfg) -> dict:
    conditions = result.conditions
    no_tutor_unlimited = _first_available_condition(conditions, "no_tutor_unlimited")
    no_tutor_t = _first_available_condition(conditions, "no_tutor_T")
    no_tutor_tplush = _first_available_condition(conditions, "no_tutor_TplusH", "no_tutor_T7")
    random_hard_draws = _matching_conditions(conditions, "random_hard_hint_T")
    random_same_pool_draws = _matching_conditions(conditions, "random_same_pool_hint_T")
    random_hard = random_hard_draws[0] if random_hard_draws else _first_available_condition(conditions, "random_hard_hint_T", "random_hard_hint_T6")
    random_same_pool = random_same_pool_draws[0] if random_same_pool_draws else _first_available_condition(conditions, "random_same_pool_hint_T")
    tutor_t = _first_available_condition(conditions, "tutor_T", "tutor_T6")
    tutor_unlimited = _first_available_condition(conditions, "tutor_unlimited")

    no_tutor_tplush_success = _condition_success(no_tutor_tplush)
    no_tutor_tplush_eval_cell = _condition_eval_cell(no_tutor_tplush)
    oracle_success_record = None if oracle_records is None else oracle_records.get("success")
    oracle_band_record = None if oracle_records is None else oracle_records.get("band")
    oracle_eval_record = None if oracle_records is None else oracle_records.get("eval_cell")
    oracle_composite_record = None if oracle_records is None else oracle_records.get("composite")

    oracle_success_conditions = {} if oracle_success_record is None else dict(oracle_success_record.get("conditions", {}))
    oracle_band_conditions = {} if oracle_band_record is None else dict(oracle_band_record.get("conditions", {}))
    oracle_eval_conditions = {} if oracle_eval_record is None else dict(oracle_eval_record.get("conditions", {}))
    oracle_composite_conditions = {} if oracle_composite_record is None else dict(oracle_composite_record.get("conditions", {}))

    oracle_success_t = _first_available_condition(oracle_success_conditions, "tutor_T", "tutor_T6")
    oracle_band_t = _first_available_condition(oracle_band_conditions, "tutor_T", "tutor_T6")
    oracle_eval_t = _first_available_condition(oracle_eval_conditions, "tutor_T", "tutor_T6")
    oracle_composite_t = _first_available_condition(oracle_composite_conditions, "tutor_T", "tutor_T6")
    oracle_composite_unlimited = _first_available_condition(oracle_composite_conditions, "tutor_unlimited")

    random_hard_success = _series_mean(random_hard_draws or ([random_hard] if random_hard is not None else []), _condition_success)
    random_hard_eval_cell = _series_mean(random_hard_draws or ([random_hard] if random_hard is not None else []), _condition_eval_cell)
    random_same_pool_success = _series_mean(random_same_pool_draws or ([random_same_pool] if random_same_pool is not None else []), _condition_success)
    random_same_pool_eval_cell = _series_mean(random_same_pool_draws or ([random_same_pool] if random_same_pool is not None else []), _condition_eval_cell)
    random_hard_success_std = _series_std(random_hard_draws or ([random_hard] if random_hard is not None else []), _condition_success)
    random_hard_eval_cell_std = _series_std(random_hard_draws or ([random_hard] if random_hard is not None else []), _condition_eval_cell)
    random_same_pool_success_std = _series_std(random_same_pool_draws or ([random_same_pool] if random_same_pool is not None else []), _condition_success)
    random_same_pool_eval_cell_std = _series_std(random_same_pool_draws or ([random_same_pool] if random_same_pool is not None else []), _condition_eval_cell)
    random_same_pool_tau_mean = _series_mean(
        [cond for cond in (random_same_pool_draws or ([random_same_pool] if random_same_pool is not None else [])) if cond.first_correct_attempt is not None],
        _condition_first_correct,
    )

    tutor_success = _condition_success(tutor_t)
    tutor_eval_cell = _condition_eval_cell(tutor_t)
    tutor_early_no_transfer = _actual_early_no_transfer(tutor_t, no_tutor_tplush, cfg)
    search_discriminative = (
        (tutor_success - no_tutor_tplush_success) >= float(getattr(cfg, "search_success_delta_vs_no_tutor_min", 0.20))
        and (tutor_success - random_same_pool_success) >= float(getattr(cfg, "search_success_delta_vs_random_min", 0.10))
        and tutor_early_no_transfer <= float(getattr(cfg, "search_early_no_transfer_max", 0.10))
    )
    transfer_discriminative = (
        (tutor_eval_cell - no_tutor_tplush_eval_cell) >= float(getattr(cfg, "transfer_eval_cell_delta_vs_no_tutor_min", 0.03))
        and (tutor_eval_cell - random_same_pool_eval_cell) >= float(getattr(cfg, "transfer_eval_cell_delta_vs_random_min", 0.03))
        and (tutor_success - no_tutor_tplush_success) >= float(getattr(cfg, "transfer_success_delta_vs_no_tutor_min", 0.0))
    )

    return {
        "no_tutor_unlimited_tau": None if no_tutor_unlimited is None else no_tutor_unlimited.first_correct_attempt,
        "no_tutor_T_success": _condition_success(no_tutor_t),
        "no_tutor_TplusH_success": no_tutor_tplush_success,
        "random_hard_success": random_hard_success,
        "random_hard_success_std": random_hard_success_std,
        "random_same_pool_success": random_same_pool_success,
        "random_same_pool_success_std": random_same_pool_success_std,
        "tutor_success": tutor_success,
        "oracle_success": _condition_success(oracle_success_t),
        "oracle_band_success": _bounded_band_success(oracle_band_t, cfg),
        "oracle_eval_cell": _condition_eval_cell(oracle_eval_t),
        "oracle_composite_success": _condition_success(oracle_composite_t),
        "tutor_band_success": _bounded_band_success(tutor_t, cfg),
        "tutor_early_success": _bounded_early_success(tutor_t, cfg),
        "tutor_soft_tau_score": _actual_soft_tau_score(tutor_t, cfg),
        "tutor_early_no_transfer": tutor_early_no_transfer,
        "tutor_eval_exact": _condition_eval_exact(tutor_t),
        "tutor_eval_cell": tutor_eval_cell,
        "no_tutor_TplusH_eval_exact": _condition_eval_exact(no_tutor_tplush),
        "no_tutor_TplusH_eval_cell": no_tutor_tplush_eval_cell,
        "random_same_pool_eval_cell": random_same_pool_eval_cell,
        "random_same_pool_eval_cell_std": random_same_pool_eval_cell_std,
        "random_same_pool_tau_mean": random_same_pool_tau_mean,
        "random_hard_eval_cell": random_hard_eval_cell,
        "random_hard_eval_cell_std": random_hard_eval_cell_std,
        "paired_delta_success_vs_no_tutor_TplusH": tutor_success - no_tutor_tplush_success,
        "paired_delta_eval_cell_vs_no_tutor_TplusH": tutor_eval_cell - no_tutor_tplush_eval_cell,
        "paired_delta_success_vs_random_same_pool": tutor_success - random_same_pool_success,
        "paired_delta_eval_cell_vs_random_same_pool": tutor_eval_cell - random_same_pool_eval_cell,
        "oracle_success_headroom_vs_no_tutor_TplusH": _condition_success(oracle_success_t) - no_tutor_tplush_success,
        "oracle_band_headroom_vs_no_tutor_TplusH": _bounded_band_success(oracle_band_t, cfg) - _bounded_band_success(no_tutor_tplush, cfg),
        "oracle_eval_cell_headroom_vs_no_tutor_TplusH": _condition_eval_cell(oracle_eval_t) - no_tutor_tplush_eval_cell,
        "tutor_unlimited_tau": None if tutor_unlimited is None else tutor_unlimited.first_correct_attempt,
        "oracle_unlimited_tau": None if oracle_composite_unlimited is None else oracle_composite_unlimited.first_correct_attempt,
        "bonus_attempts_limit": int(resolved_no_tutor_tplush_limit(cfg)),
        "bonus_attempts_effective": bool(int(resolved_no_tutor_tplush_limit(cfg)) > int(getattr(cfg, "max_attempts_main", 6))),
        "search_discriminative": bool(search_discriminative),
        "transfer_discriminative": bool(transfer_discriminative),
    }


def _evaluate_single_hint_oracle_pool(prepared, families: Optional[Sequence[str]] = None) -> dict:
    candidates = _rank_candidates_for_oracle(prepared, families=families)
    if not candidates:
        conditions = _evaluate_actual_hint_bundle(prepared, pre_hints=[])
        records = [
            {
                "selected_hints": [],
                "conditions": conditions,
                "post_first_wrong_hint": None,
            }
        ]
        family_counts: Dict[str, int] = {}
        candidate_pool_size = 0
    else:
        records = []
        family_counts = _family_counts(candidates)
        candidate_pool_size = len(candidates)
    for hint in candidates:
        conditions = _evaluate_actual_hint_bundle(prepared, pre_hints=[hint])
        records.append(
            {
                "selected_hints": [hint],
                "conditions": conditions,
                "post_first_wrong_hint": None,
            }
        )
    by_metric = {
        metric: _select_oracle_record_for_metric(
            evaluated_records=records,
            candidate_pool_size=candidate_pool_size,
            candidate_family_counts=family_counts,
            cfg=prepared.cfg,
            metric=metric,
        )
        for metric in _oracle_metric_names()
    }
    return {
        "candidate_pool": list(candidates),
        "candidate_family_counts": family_counts,
        "by_metric": by_metric,
    }


def _best_single_hint(
    prepared,
    families: Optional[Sequence[str]] = None,
    oracle_metric: Optional[str] = None,
) -> dict:
    oracle_pool = _evaluate_single_hint_oracle_pool(prepared, families=families)
    metric = str(oracle_metric or getattr(prepared.cfg, "oracle_metric", "composite"))
    return dict(oracle_pool["by_metric"][metric])


def _best_two_hint_pre(
    prepared,
    families: Optional[Sequence[str]] = None,
    oracle_metric: Optional[str] = None,
) -> dict:
    metric = str(oracle_metric or getattr(prepared.cfg, "oracle_metric", "composite"))
    limit = max(2, int(getattr(prepared.cfg, "two_hint_pair_limit", 8)))
    candidates = _rank_candidates_for_oracle(
        prepared,
        families=families,
        candidate_limit=limit,
        seed_offset=19391,
    )
    if len(candidates) < 2:
        return _best_single_hint(prepared, families=families, oracle_metric=metric)

    best = None
    for first, second in combinations(candidates, 2):
        conditions = _evaluate_actual_hint_bundle(prepared, pre_hints=[first, second])
        rank_key = _oracle_rank_tuple_for_metric(metric, conditions.get("tutor_T6"), conditions.get("tutor_unlimited"), prepared.cfg)
        numeric = _oracle_numeric_score_for_metric(metric, conditions.get("tutor_T6"), conditions.get("tutor_unlimited"), prepared.cfg)
        record = {
            "selected_hints": [first, second],
            "conditions": conditions,
            "rank_key": rank_key,
            "score": numeric,
            "oracle_metric": metric,
        }
        if best is None or record["rank_key"] > best["rank_key"]:
            best = record
    assert best is not None
    best["candidate_pool_size"] = len(candidates)
    best["candidate_family_counts"] = _family_counts(candidates)
    return best


def _best_after_first_wrong(
    prepared,
    families: Optional[Sequence[str]] = None,
    oracle_metric: Optional[str] = None,
) -> dict:
    metric = str(oracle_metric or getattr(prepared.cfg, "oracle_metric", "composite"))
    pre_limit = max(1, int(getattr(prepared.cfg, "after_first_wrong_pre_limit", 6)))
    post_limit = max(1, int(getattr(prepared.cfg, "after_first_wrong_post_limit", 6)))
    pre_candidates = _rank_candidates_for_oracle(
        prepared,
        families=families,
        candidate_limit=pre_limit,
        seed_offset=19391,
    )
    post_candidates = _rank_candidates_for_oracle(
        prepared,
        families=families,
        candidate_limit=post_limit,
        seed_offset=29999,
    )
    if not pre_candidates or not post_candidates:
        return _best_single_hint(prepared, families=families, oracle_metric=metric)

    best = None
    for pre_hint, post_hint in product(pre_candidates, post_candidates):
        if candidate_signature(pre_hint) == candidate_signature(post_hint):
            continue
        conditions = _evaluate_actual_hint_bundle(
            prepared,
            pre_hints=[pre_hint],
            post_first_wrong_hint=post_hint,
        )
        rank_key = _oracle_rank_tuple_for_metric(metric, conditions.get("tutor_T6"), conditions.get("tutor_unlimited"), prepared.cfg)
        numeric = _oracle_numeric_score_for_metric(metric, conditions.get("tutor_T6"), conditions.get("tutor_unlimited"), prepared.cfg)
        record = {
            "selected_hints": [pre_hint],
            "post_first_wrong_hint": post_hint,
            "conditions": conditions,
            "rank_key": rank_key,
            "score": numeric,
            "oracle_metric": metric,
        }
        if best is None or record["rank_key"] > best["rank_key"]:
            best = record
    if best is None:
        return _best_single_hint(prepared, families=families, oracle_metric=metric)
    assert best is not None
    combined_pool: List[HintCandidate] = []
    seen = set()
    for hint in list(pre_candidates) + list(post_candidates):
        sig = candidate_signature(hint)
        if sig in seen:
            continue
        seen.add(sig)
        combined_pool.append(hint)
    best["candidate_pool_size"] = len(combined_pool)
    best["candidate_family_counts"] = _family_counts(combined_pool)
    return best


def _custom_plan_result(hints: Sequence[HintCandidate]) -> HintPlanResult:
    selected_hint = None if not hints else hints[0]
    return HintPlanResult(
        selected_hint=selected_hint,
        selected_utility=0.0,
        no_hint_utility=0.0,
        delta_vs_no_hint=0.0,
        candidate_scores=[],
    )


def _scenario_metrics(conditions: Dict[str, ConditionResult], cfg) -> dict:
    no_tutor_t = _first_available_condition(conditions, "no_tutor_T")
    no_tutor_tplush = _first_available_condition(conditions, "no_tutor_TplusH", "no_tutor_T7")
    random_hard_draws = _matching_conditions(conditions, "random_hard_hint_T")
    random_same_pool_draws = _matching_conditions(conditions, "random_same_pool_hint_T")
    random_hard = random_hard_draws[0] if random_hard_draws else _first_available_condition(conditions, "random_hard_hint_T", "random_hard_hint_T6")
    random_same_pool = random_same_pool_draws[0] if random_same_pool_draws else _first_available_condition(conditions, "random_same_pool_hint_T")
    tutor_t6 = _first_available_condition(conditions, "tutor_T", "tutor_T6")
    tutor_unlimited = _first_available_condition(conditions, "tutor_unlimited")
    random_hard_success = _series_mean(random_hard_draws or ([random_hard] if random_hard is not None else []), _condition_success)
    random_same_pool_success = _series_mean(random_same_pool_draws or ([random_same_pool] if random_same_pool is not None else []), _condition_success)
    random_hard_eval_cell = _series_mean(random_hard_draws or ([random_hard] if random_hard is not None else []), _condition_eval_cell)
    random_same_pool_eval_cell = _series_mean(random_same_pool_draws or ([random_same_pool] if random_same_pool is not None else []), _condition_eval_cell)
    tutor_success = 0.0 if tutor_t6 is None else (1.0 if tutor_t6.success_within_limit else 0.0)
    tutor_eval_cell = 0.0 if tutor_t6 is None or tutor_t6.eval_metrics is None else float(tutor_t6.eval_metrics.cell_acc)
    no_tutor_tplush_success = _condition_success(no_tutor_tplush)
    no_tutor_tplush_eval_cell = _condition_eval_cell(no_tutor_tplush)
    return {
        "no_tutor_T_success": _condition_success(no_tutor_t),
        "no_tutor_TplusH_success": no_tutor_tplush_success,
        "random_hard_success": random_hard_success,
        "random_same_pool_success": random_same_pool_success,
        "random_hard_eval_cell": random_hard_eval_cell,
        "random_same_pool_eval_cell": random_same_pool_eval_cell,
        "paired_delta_success_vs_no_tutor_TplusH": tutor_success - no_tutor_tplush_success,
        "paired_delta_eval_cell_vs_no_tutor_TplusH": tutor_eval_cell - no_tutor_tplush_eval_cell,
        "paired_delta_success_vs_random_same_pool": tutor_success - random_same_pool_success,
        "paired_delta_eval_cell_vs_random_same_pool": tutor_eval_cell - random_same_pool_eval_cell,
        "tutor_T6_success": tutor_success,
        "tutor_T6_band_success": _bounded_band_success(tutor_t6, cfg),
        "tutor_T6_early_success": _bounded_early_success(tutor_t6, cfg),
        "tutor_T6_eval_exact": 0.0 if tutor_t6 is None or tutor_t6.eval_metrics is None else float(tutor_t6.eval_metrics.exact_acc),
        "tutor_T6_eval_cell": tutor_eval_cell,
        "tutor_T6_soft_tau_score": _actual_soft_tau_score(tutor_t6, cfg),
        "tutor_T6_early_no_transfer": _actual_early_no_transfer(tutor_t6, no_tutor_tplush, cfg),
        "tutor_unlimited_first_correct": None if tutor_unlimited is None else tutor_unlimited.first_correct_attempt,
        "tutor_T6_first_correct": None if tutor_t6 is None else tutor_t6.first_correct_attempt,
        "tutor_T6_failure_type": None if tutor_t6 is None else tutor_t6.failure_type,
    }


def run_regime_discovery_row(
    task_id: str,
    cfg,
    seed: int,
) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.seed = int(seed)
    apply_named_presets(cfg)

    prepared = prepare_one_hint_experiment(task_id=task_id, cfg=cfg, seed=seed)
    plan_rng = np.random.default_rng(int(prepared.seed_bundle.get("plan", prepared.seed)))
    baseline_rng = np.random.default_rng(int(prepared.seed_bundle.get("baseline", prepared.seed)))
    plan = select_hint(
        posterior=prepared.posterior,
        context=prepared.context,
        teach_case=prepared.teach_case,
        eval_items=prepared.eval_items,
        cfg=prepared.cfg,
        rng=plan_rng,
    )
    result = finalize_prepared_experiment(prepared, plan, rng=baseline_rng)
    families = list(getattr(prepared.cfg, "hint_families", ()))
    oracle_pool = _evaluate_single_hint_oracle_pool(prepared, families=families or None)
    oracle_records = dict(oracle_pool.get("by_metric", {}))
    composite_oracle = dict(oracle_records.get("composite", {}))

    return {
        "task_id": task_id,
        "seed": int(seed),
        "grid_mode": "regime_discovery",
        "selected_hint": _hint_payload(result.plan.selected_hint),
        "selected_hint_family": _family_name(result.plan.selected_hint),
        "selected_utility": float(result.plan.selected_utility),
        "delta_vs_no_hint": float(result.plan.delta_vs_no_hint),
        "planner_prediction": None if result.plan.planner_prediction is None else {
            "pred_p_success_T": float(result.plan.planner_prediction.pred_p_success_T6),
            "pred_p_tau_band": float(result.plan.planner_prediction.pred_p_tau_band),
            "pred_p_tau_early": float(result.plan.planner_prediction.pred_p_tau_early),
            "abstained": bool(result.plan.planner_prediction.abstained),
            "abstain_reason": result.plan.planner_prediction.abstain_reason,
            "hint_quality_tags": dict(result.plan.planner_prediction.hint_quality_tags),
        },
        "conditions": {name: _condition_payload(cond) for name, cond in result.conditions.items()},
        "teach_case_metadata": dict(getattr(prepared.teach_case, "metadata", {}) or {}),
        "oracle_best_current_pool": _oracle_metric_payload(composite_oracle),
        "oracle_by_metric": {
            metric: _oracle_metric_payload(record)
            for metric, record in oracle_records.items()
        },
        "top_candidates": [
            {
                "hint": _hint_payload(item.get("hint")),
                "utility": float(item.get("utility", 0.0)),
                "selection_score": float(item.get("selection_score", item.get("utility", 0.0))),
                "reranker_score": float(item.get("reranker_score", 0.0)),
                "stage": item.get("stage"),
                "band_success_prob": float(item.get("band_success_prob", 0.0)),
                "early_success_prob": float(item.get("early_success_prob", 0.0)),
                "eval_cell_acc": float(item.get("eval_cell_acc", 0.0)),
            }
            for item in list(result.plan.candidate_scores[:5])
        ],
        "metrics": _regime_discovery_metrics(result, oracle_records, cfg),
    }


def run_experiment_scenario(
    task_id: str,
    cfg,
    seed: int,
) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.seed = int(seed)
    apply_named_presets(cfg)
    mode = str(getattr(cfg, "ceiling_mode", "none"))

    if mode == "none":
        prepared = prepare_one_hint_experiment(task_id=task_id, cfg=cfg, seed=seed)
        rng = np.random.default_rng(int(prepared.seed_bundle.get("plan", prepared.seed)))
        plan = select_hint(
            posterior=prepared.posterior,
            context=prepared.context,
            teach_case=prepared.teach_case,
            eval_items=prepared.eval_items,
            cfg=prepared.cfg,
            rng=rng,
        )
        result = finalize_prepared_experiment(
            prepared,
            plan,
            rng=np.random.default_rng(int(prepared.seed_bundle.get("baseline", prepared.seed))),
        )
        selected_hints = []
        if result.plan.selected_hint is not None:
            selected_hints.append(_hint_payload(result.plan.selected_hint))
        candidate_families = Counter()
        for item in result.plan.candidate_scores:
            metadata = dict(item.get("metadata", {}) or {})
            candidate_families[str(metadata.get("family", item.get("kind", "unknown")))] += 1
        return {
            "task_id": task_id,
            "seed": int(seed),
            "ceiling_mode": mode,
            "selected_hints": selected_hints,
            "post_first_wrong_hint": None,
            "planner_reference": None,
            "oracle_gap": None,
            "candidate_pool_size": int(sum(candidate_families.values())),
            "candidate_family_counts": dict(sorted(candidate_families.items())),
            "teach_case_metadata": dict(getattr(prepared.teach_case, "metadata", {}) or {}),
            "conditions": {name: _condition_payload(cond) for name, cond in result.conditions.items()},
            "metrics": _scenario_metrics(result.conditions, cfg),
        }

    prepared = prepare_one_hint_experiment(task_id=task_id, cfg=cfg, seed=seed)
    families = list(getattr(prepared.cfg, "hint_families", ()))
    planner_reference = _evaluate_planner_reference(prepared) if getattr(cfg, "oracle_include_planner_reference", True) else None

    if mode == "menu_correct":
        hint = _correct_menu_hint(prepared)
        selected_hints = [] if hint is None else [hint]
        outcome = None
        candidate_pool = [] if hint is None else [hint]
        numeric = 0.0
        post_first_wrong_hint = None
    elif mode == "oracle_best_candidate":
        record = _best_single_hint(
            prepared,
            families=families or None,
            oracle_metric=str(getattr(prepared.cfg, "oracle_metric", "composite")),
        )
        selected_hints = list(record.get("selected_hints", []))
        outcome = None
        candidate_pool = _rank_candidates_for_oracle(prepared, families=families or None)
        numeric = float(record["score"])
        post_first_wrong_hint = None
    elif mode == "oracle_best_family":
        record = _best_single_hint(
            prepared,
            families=families or None,
            oracle_metric=str(getattr(prepared.cfg, "oracle_metric", "composite")),
        )
        selected_hints = list(record.get("selected_hints", []))
        outcome = None
        candidate_pool = _rank_candidates_for_oracle(prepared, families=families or None)
        numeric = float(record["score"])
        post_first_wrong_hint = None
    elif mode == "two_hint_pre":
        record = _best_two_hint_pre(
            prepared,
            families=families or None,
            oracle_metric=str(getattr(prepared.cfg, "oracle_metric", "composite")),
        )
        selected_hints = list(record.get("selected_hints", []))
        outcome = dict(record["conditions"])
        candidate_pool = _rank_candidates_for_oracle(
            prepared,
            families=families or None,
            candidate_limit=int(getattr(prepared.cfg, "two_hint_pair_limit", 8)),
        )
        numeric = float(record["score"])
        post_first_wrong_hint = None
    elif mode == "after_first_wrong_hint":
        record = _best_after_first_wrong(
            prepared,
            families=families or None,
            oracle_metric=str(getattr(prepared.cfg, "oracle_metric", "composite")),
        )
        selected_hints = list(record.get("selected_hints", []))
        outcome = dict(record["conditions"])
        candidate_pool = _rank_candidates_for_oracle(
            prepared,
            families=families or None,
            candidate_limit=max(
                int(getattr(prepared.cfg, "after_first_wrong_pre_limit", 6)),
                int(getattr(prepared.cfg, "after_first_wrong_post_limit", 6)),
            ),
        )
        numeric = float(record["score"])
        post_first_wrong_hint = record.get("post_first_wrong_hint")
    else:
        raise ValueError(f"Unknown ceiling_mode: {mode}")

    if mode in {"menu_correct", "oracle_best_candidate", "oracle_best_family"} and post_first_wrong_hint is None and len(selected_hints) <= 1:
        finalized = finalize_prepared_experiment(
            prepared,
            _custom_plan_result(selected_hints),
            rng=np.random.default_rng(int(prepared.seed_bundle.get("baseline", prepared.seed))),
        )
        outcome = dict(finalized.conditions)
        tutor_t = _first_available_condition(outcome, "tutor_T", "tutor_T6")
        tutor_unlimited = _first_available_condition(outcome, "tutor_unlimited")
        numeric = float(
            _oracle_numeric_score_for_metric(
                str(getattr(prepared.cfg, "oracle_metric", "composite")),
                tutor_t,
                tutor_unlimited,
                prepared.cfg,
            )
        )

    assert outcome is not None
    oracle_gap = None
    if planner_reference is not None and planner_reference.get("score") is not None:
        oracle_gap = float(numeric) - float(planner_reference["score"])

    return {
        "task_id": task_id,
        "seed": int(seed),
        "ceiling_mode": mode,
        "selected_hints": [_hint_payload(hint) for hint in selected_hints],
        "post_first_wrong_hint": _hint_payload(post_first_wrong_hint),
        "planner_reference": planner_reference,
        "oracle_gap": oracle_gap,
        "candidate_pool_size": len(candidate_pool),
        "candidate_family_counts": _family_counts(candidate_pool),
        "teach_case_metadata": dict(getattr(prepared.teach_case, "metadata", {}) or {}),
        "conditions": {name: _condition_payload(cond) for name, cond in outcome.items()},
        "metrics": {
            **_scenario_metrics(outcome, cfg),
            "oracle_numeric_score": float(numeric),
        },
    }
