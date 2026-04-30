from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import OneHintConfig
from .experiment_presets import apply_named_presets, parse_seed_spec
from .hint_planner import (
    _bonus_cfg,
    _choose_best_candidate,
    _family_aware_prefilter,
    _record_for_hint,
    filter_candidate_records_for_stage,
)
from .hint_space import build_hint_candidates, build_target_neighborhood_candidates
from .learner_runner import run_teach_condition
from .metrics import conservative_reveal_penalty, soft_tau_score
from .protocol import prepare_one_hint_experiment
from .rollout import (
    _correct_rank_from_probs,
    _full_active_mask,
    _score_table_probs,
    build_first_reveal_tables_for_candidates,
    build_score_tables_for_candidates,
    candidate_signature,
    evaluate_score_table_under_profiles,
    prefilter_score_table_under_profiles,
)


FAST_PRED_THRESHOLD = 0.25
FAST_SUCCESS_WEIGHT = 10.0
FAST_TAU2_WEIGHT = 10.0
FAST_WRONG_WEIGHT = 4.0
FAST_MARGIN_WEIGHT = 2.0
ANSWER_NEIGHBOR_LIMIT = 8


def _base_cfg() -> OneHintConfig:
    cfg = OneHintConfig()
    cfg.planner_mode = "cascade"
    cfg.utility_mode = "advantage_delta"
    cfg.oracle_metric = "composite"
    cfg.common_randomness = True
    cfg.hint_mode = "combined"
    cfg.hint_families = ("free", "operator_probe", "target_neighborhood_robust_filtered")
    cfg.teach_difficulty = "hard"
    cfg.menu_difficulty_mode = "rank_stratified"
    cfg.teach_probe_mode = "initial_rank"
    cfg.target_initial_rank_min = 5
    cfg.target_initial_rank_max = 12
    cfg.prelearn_profile = "4"
    cfg.n_obs = 4
    cfg.teach_menu_size = 20
    cfg.max_attempts_main = 5
    cfg.hint_count_budget = 1
    cfg.no_tutor_bonus_attempts = 1
    cfg.eval_aware = True
    cfg.random_hard_n = 5
    cfg.random_same_pool_n = 5
    cfg.prefilter_top_k = 4
    cfg.proxy_rollout_top_k = 4
    cfg.proxy_n_rollouts = 24
    cfg.refine_n_rollouts = 12
    cfg.free_hint_pool_easy = 2
    cfg.free_hint_pool_medium = 3
    cfg.free_hint_pool_hard = 3
    cfg.operator_probe_limit = 6
    cfg.target_neighborhood_limit = 16
    cfg.lambda_success = 11.0
    cfg.lambda_fail = 0.0
    cfg.lambda_eval_cell = 2.0
    cfg.lambda_soft_tau = 2.0
    cfg.lambda_exposure = 0.5
    cfg.lambda_collapse = 3.0
    cfg.soft_tau_center = 4.5
    cfg.soft_tau_sigma = 1.5
    cfg.refine_enabled = True
    cfg.refine_top_k = 2
    cfg.refine_update_mode = "first_reveal_cached_cls"
    return cfg


def _experiment_cfg(name: str) -> OneHintConfig:
    cfg = copy.deepcopy(_base_cfg())
    if name == "balanced_full":
        pass
    elif name == "fast_soft_tau_center2":
        cfg.soft_tau_center = 2.0
    elif name in {"fast_explicit_dynamic", "fast_explicit_dynamic_answer_neighbor"}:
        pass
    else:
        raise ValueError(f"Unknown diagnostic experiment: {name}")
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


def _dedupe_hint_candidates(candidates: Iterable) -> List:
    deduped: List = []
    seen: set[Tuple[object, ...]] = set()
    for candidate in candidates:
        sig = candidate_signature(candidate)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(candidate)
    return deduped


def _retag_local_family(candidate, family: str):
    tagged = copy.deepcopy(candidate)
    tagged.metadata = {**dict(candidate.metadata), "family": family}
    return tagged


def _build_answer_neighbor_candidates(context, teach_case, cfg, rng: np.random.Generator) -> List:
    base = build_target_neighborhood_candidates(context, teach_case, cfg, rng)
    teach_words = tuple(teach_case.example.words)
    filtered: List = []
    for candidate in base:
        words = tuple(candidate.example.words)
        if words == teach_words:
            continue
        metadata = dict(candidate.metadata)
        source = str(metadata.get("source", ""))
        operator_overlap = int(metadata.get("operator_overlap", 0))
        atom_overlap = int(metadata.get("atom_overlap", 0))
        if source not in {"single_atom_replacement", "operator_swap", "contiguous_subexpr", "pool_overlap"}:
            continue
        if operator_overlap <= 0 and atom_overlap <= 0:
            continue
        if abs(len(words) - len(teach_words)) > 2:
            continue
        filtered.append(_retag_local_family(candidate, "answer_neighbor_nonanswer"))

    filtered.sort(
        key=lambda cand: (
            float(cand.metadata.get("quality_score", 0.0)),
            float(cand.metadata.get("operator_overlap", 0.0)),
            float(cand.metadata.get("atom_overlap", 0.0)),
            -float(len(cand.example.words)),
        ),
        reverse=True,
    )
    return _dedupe_hint_candidates(filtered[:ANSWER_NEIGHBOR_LIMIT])


def _inv_rank(rank: Optional[float]) -> float:
    if rank is None:
        return 0.0
    if float(rank) <= 0.0:
        return 0.0
    return 1.0 / float(rank)


def _initial_prob_rank_margin(score_table, profile_weights, cfg) -> dict:
    active_mask = _full_active_mask(len(score_table.sem))
    weighted_prob = 0.0
    weighted_rank = 0.0
    weighted_rank_mass = 0.0
    weighted_margin = 0.0
    correct_pos = score_table.menu_meta.correct_pos
    for profile, weight in profile_weights:
        if float(weight) <= 0.0:
            continue
        probs = _score_table_probs(
            score_table=score_table,
            profile=profile,
            active_mask=active_mask,
            revealed_text_ids=frozenset(),
            revealed_output_ids=frozenset(),
            cfg=cfg,
        )
        if correct_pos is None:
            continue
        correct_prob = float(probs[correct_pos])
        correct_rank = _correct_rank_from_probs(np.asarray(probs, dtype=float), correct_pos)
        top_wrong = 0.0
        if len(probs) > 1:
            masked = np.asarray(probs, dtype=float).copy()
            masked[correct_pos] = -1.0
            top_wrong = float(masked.max())
        weighted_prob += float(weight) * correct_prob
        weighted_margin += float(weight) * (correct_prob - top_wrong)
        if correct_rank is not None:
            weighted_rank += float(weight) * float(correct_rank)
            weighted_rank_mass += float(weight)
    return {
        "initial_correct_prob_mean": float(weighted_prob),
        "initial_correct_rank_mean": (weighted_rank / weighted_rank_mass) if weighted_rank_mass > 0.0 else None,
        "initial_correct_margin_mean": float(weighted_margin),
    }


def _exactish_tau_le2(score_table, first_reveal_tables, profile_weights, cfg) -> float:
    correct_pos = score_table.menu_meta.correct_pos
    if correct_pos is None:
        return 0.0
    K = len(score_table.sem)
    active_mask = _full_active_mask(K)
    total = 0.0
    for profile, weight in profile_weights:
        if float(weight) <= 0.0:
            continue
        probs0 = _score_table_probs(
            score_table=score_table,
            profile=profile,
            active_mask=active_mask,
            revealed_text_ids=frozenset(),
            revealed_output_ids=frozenset(),
            cfg=cfg,
        )
        p_tau1 = float(probs0[correct_pos])
        p_tau2 = 0.0
        for pos in range(K):
            if pos == correct_pos:
                continue
            p0 = float(probs0[pos])
            if p0 <= 0.0:
                continue
            option_index = int(score_table.menu_meta.option_indices[pos])
            branch_table = None if first_reveal_tables is None else first_reveal_tables.get(option_index)
            next_table = branch_table or score_table
            next_active_mask = active_mask & ~(1 << pos)
            probs1 = _score_table_probs(
                score_table=next_table,
                profile=profile,
                active_mask=next_active_mask,
                revealed_text_ids=frozenset({score_table.menu_meta.text_ids[pos]}),
                revealed_output_ids=frozenset({score_table.menu_meta.output_ids[pos]}),
                cfg=cfg,
            )
            p_tau2 += p0 * float(probs1[correct_pos])
        total += float(weight) * (p_tau1 + p_tau2)
    return float(total)


def _balanced_delta_contribs(stats: dict, baseline: dict, cfg) -> dict:
    success_delta = float(stats.get("success_prob", 0.0)) - float(baseline.get("success_prob", 0.0))
    eval_delta = float(stats.get("eval_cell_acc", 0.0)) - float(baseline.get("eval_cell_acc", 0.0))
    soft_tau_delta = float(soft_tau_score(stats, cfg)) - float(soft_tau_score(baseline, cfg))
    exposure_delta = float(stats.get("safe_wrong_mean", 0.0)) - float(baseline.get("safe_wrong_mean", 0.0))
    collapse_delta = float(conservative_reveal_penalty(stats, cfg)) - float(conservative_reveal_penalty(baseline, cfg))
    contribs = {
        "success_contrib": float(getattr(cfg, "lambda_success", 0.0)) * success_delta,
        "eval_contrib": float(getattr(cfg, "lambda_eval_cell", 0.0)) * eval_delta,
        "soft_tau_contrib": float(getattr(cfg, "lambda_soft_tau", 0.0)) * soft_tau_delta,
        "exposure_contrib": float(getattr(cfg, "lambda_exposure", 0.0)) * exposure_delta,
        "collapse_contrib": -float(getattr(cfg, "lambda_collapse", 0.0)) * collapse_delta,
        "early_gate_penalty": 0.0,
    }
    contribs["final_utility"] = float(sum(contribs.values()))
    return contribs


def _fast_explicit_dynamic_contribs(stats: dict, baseline: dict) -> dict:
    success_delta = float(stats.get("success_prob", 0.0)) - float(baseline.get("success_prob", 0.0))
    tau2_delta = float(stats.get("pred_tau_le2_exact", 0.0)) - float(baseline.get("pred_tau_le2_exact", 0.0))
    wrong_delta = float(stats.get("safe_wrong_mean", 0.0)) - float(baseline.get("safe_wrong_mean", 0.0))
    margin_delta = float(stats.get("initial_correct_margin_mean", 0.0)) - float(
        baseline.get("initial_correct_margin_mean", 0.0)
    )
    contribs = {
        "success_contrib": FAST_SUCCESS_WEIGHT * success_delta,
        "tau2_contrib": FAST_TAU2_WEIGHT * tau2_delta,
        "wrong_contrib": -FAST_WRONG_WEIGHT * wrong_delta,
        "margin_contrib": FAST_MARGIN_WEIGHT * margin_delta,
        "early_gate_penalty": 0.0,
    }
    contribs["final_utility"] = float(sum(contribs.values()))
    return contribs


def _utility_contribs(experiment_name: str, stats: dict, baseline: dict, cfg) -> dict:
    if experiment_name in {"fast_explicit_dynamic", "fast_explicit_dynamic_answer_neighbor"}:
        return _fast_explicit_dynamic_contribs(stats, baseline)
    return _balanced_delta_contribs(stats, baseline, cfg)


def _actual_candidate_result(prepared, hint) -> dict:
    condition = run_teach_condition(
        base_learner=prepared.base_learner,
        context=prepared.context,
        teach_case=prepared.teach_case,
        max_attempts=int(prepared.cfg.max_attempts_main),
        hint=hint,
        hints=None,
        post_first_wrong_hint=None,
        eval_items=prepared.eval_items,
        condition_name="diagnostic",
    )
    tau = condition.first_correct_attempt
    return {
        "success": bool(condition.success_within_limit),
        "tau": tau,
        "tau_le2": bool(tau is not None and int(tau) <= 2),
        "tau_le5": bool(tau is not None and int(tau) <= int(prepared.cfg.max_attempts_main)),
        "wrong_before_correct": int(condition.n_wrong_before_correct),
        "eval_cell": 0.0 if condition.eval_metrics is None else float(condition.eval_metrics.cell_acc),
        "eval_exact": 0.0 if condition.eval_metrics is None else float(condition.eval_metrics.exact_acc),
    }


def _stage_flags(sig, prefilter_sigs, proxy_sigs, refine_sigs, final_sigs, selected_sig) -> dict:
    return {
        "entered_prefilter": sig in prefilter_sigs,
        "entered_proxy": sig in proxy_sigs,
        "entered_refine": sig in refine_sigs,
        "entered_final": sig in final_sigs,
        "selected": sig == selected_sig,
    }


def _candidate_diag_payload(
    hint,
    prefilter_stats: Optional[dict],
    proxy_stats: Optional[dict],
    final_stats: Optional[dict],
    actual: Optional[dict],
    contribs: Optional[dict],
    stage_flags: dict,
) -> dict:
    return {
        "hint": _hint_payload(hint),
        "initial_correct_prob_mean": None if prefilter_stats is None else prefilter_stats.get("initial_correct_prob_mean"),
        "initial_correct_rank_mean": None if prefilter_stats is None else prefilter_stats.get("initial_correct_rank_mean"),
        "initial_correct_margin_mean": None if prefilter_stats is None else prefilter_stats.get("initial_correct_margin_mean"),
        "proxy_pred_success_prob": None if proxy_stats is None else float(proxy_stats.get("success_prob", 0.0)),
        "proxy_pred_tau_le2_exact": None if proxy_stats is None else float(proxy_stats.get("pred_tau_le2_exact", 0.0)),
        "proxy_pred_tau_le5_prob": None if proxy_stats is None else float(proxy_stats.get("success_prob", 0.0)),
        "final_pred_success_prob": None if final_stats is None else float(final_stats.get("success_prob", 0.0)),
        "final_pred_tau_le2_exact": None if final_stats is None else float(final_stats.get("pred_tau_le2_exact", 0.0)),
        "final_pred_tau_le5_prob": None if final_stats is None else float(final_stats.get("success_prob", 0.0)),
        "actual_success": None if actual is None else bool(actual["success"]),
        "actual_tau": None if actual is None else actual["tau"],
        "actual_tau_le2": None if actual is None else bool(actual["tau_le2"]),
        "actual_tau_le5": None if actual is None else bool(actual["tau_le5"]),
        "actual_wrong_before_correct": None if actual is None else int(actual["wrong_before_correct"]),
        "actual_eval_cell": None if actual is None else float(actual["eval_cell"]),
        "actual_eval_exact": None if actual is None else float(actual["eval_exact"]),
        "utility_terms": {} if contribs is None else dict(contribs),
        "is_fast_true": False if actual is None else bool(actual["tau_le2"]),
        "is_fast_pred_proxy": False if proxy_stats is None else float(proxy_stats.get("pred_tau_le2_exact", 0.0)) >= FAST_PRED_THRESHOLD,
        "is_fast_pred_final": False if final_stats is None else float(final_stats.get("pred_tau_le2_exact", 0.0)) >= FAST_PRED_THRESHOLD,
        **stage_flags,
    }


def _availability_summary(
    candidates: Sequence,
    prefilter_records: Dict[Tuple[object, ...], dict],
    proxy_records: Dict[Tuple[object, ...], dict],
    final_records: Dict[Tuple[object, ...], dict],
    actual_truth: Dict[Tuple[object, ...], dict],
    prefilter_sigs: set,
    proxy_sigs: set,
    refine_sigs: set,
    final_sigs: set,
    selected_sig,
) -> dict:
    count_initial_rank_le2 = 0
    count_initial_rank_le5 = 0
    count_pred_tau_le2_proxy = 0
    count_pred_tau_le2_final = 0
    count_true_tau_le2 = 0
    count_true_tau_le5 = 0
    max_proxy_pred_tau_le2 = 0.0
    max_final_pred_tau_le2 = 0.0
    max_proxy_pred_tau_le5 = 0.0
    max_final_pred_tau_le5 = 0.0

    pred_fast_proxy_stage = {"all": 0, "prefilter": 0, "proxy": 0, "refine": 0, "final": 0, "selected": 0}
    pred_fast_final_stage = {"all": 0, "prefilter": 0, "proxy": 0, "refine": 0, "final": 0, "selected": 0}
    true_fast_stage = {"all": 0, "prefilter": 0, "proxy": 0, "refine": 0, "final": 0, "selected": 0}

    for hint in candidates:
        sig = candidate_signature(hint)
        pref = prefilter_records.get(sig)
        prox = proxy_records.get(sig)
        final = final_records.get(sig)
        actual = actual_truth.get(sig)

        if pref is not None and pref.get("initial_correct_rank_mean") is not None:
            rank = float(pref["initial_correct_rank_mean"])
            if rank <= 2.0:
                count_initial_rank_le2 += 1
            if rank <= 5.0:
                count_initial_rank_le5 += 1

        proxy_tau2 = None if prox is None else float(prox["stats"].get("pred_tau_le2_exact", 0.0))
        final_tau2 = None if final is None else float(final["stats"].get("pred_tau_le2_exact", 0.0))
        proxy_tau5 = None if prox is None else float(prox["stats"].get("success_prob", 0.0))
        final_tau5 = None if final is None else float(final["stats"].get("success_prob", 0.0))

        if proxy_tau2 is not None:
            if proxy_tau2 > 0.0:
                count_pred_tau_le2_proxy += 1
            max_proxy_pred_tau_le2 = max(max_proxy_pred_tau_le2, proxy_tau2)
            if proxy_tau2 >= FAST_PRED_THRESHOLD:
                pred_fast_proxy_stage["all"] += 1
                if sig in prefilter_sigs:
                    pred_fast_proxy_stage["prefilter"] += 1
                if sig in proxy_sigs:
                    pred_fast_proxy_stage["proxy"] += 1
                if sig in refine_sigs:
                    pred_fast_proxy_stage["refine"] += 1
                if sig in final_sigs:
                    pred_fast_proxy_stage["final"] += 1
                if sig == selected_sig:
                    pred_fast_proxy_stage["selected"] += 1

        if final_tau2 is not None:
            if final_tau2 > 0.0:
                count_pred_tau_le2_final += 1
            max_final_pred_tau_le2 = max(max_final_pred_tau_le2, final_tau2)
            if final_tau2 >= FAST_PRED_THRESHOLD:
                pred_fast_final_stage["all"] += 1
                if sig in prefilter_sigs:
                    pred_fast_final_stage["prefilter"] += 1
                if sig in proxy_sigs:
                    pred_fast_final_stage["proxy"] += 1
                if sig in refine_sigs:
                    pred_fast_final_stage["refine"] += 1
                if sig in final_sigs:
                    pred_fast_final_stage["final"] += 1
                if sig == selected_sig:
                    pred_fast_final_stage["selected"] += 1

        if proxy_tau5 is not None:
            max_proxy_pred_tau_le5 = max(max_proxy_pred_tau_le5, proxy_tau5)
        if final_tau5 is not None:
            max_final_pred_tau_le5 = max(max_final_pred_tau_le5, final_tau5)

        if actual is not None and bool(actual["tau_le2"]):
            count_true_tau_le2 += 1
            true_fast_stage["all"] += 1
            if sig in prefilter_sigs:
                true_fast_stage["prefilter"] += 1
            if sig in proxy_sigs:
                true_fast_stage["proxy"] += 1
            if sig in refine_sigs:
                true_fast_stage["refine"] += 1
            if sig in final_sigs:
                true_fast_stage["final"] += 1
            if sig == selected_sig:
                true_fast_stage["selected"] += 1
        if actual is not None and bool(actual["tau_le5"]):
            count_true_tau_le5 += 1

    return {
        "candidate_count": len(candidates),
        "count_initial_rank_le2": int(count_initial_rank_le2),
        "count_initial_rank_le5": int(count_initial_rank_le5),
        "count_pred_tau_le2_proxy": int(count_pred_tau_le2_proxy),
        "count_pred_tau_le2_final": int(count_pred_tau_le2_final),
        "count_true_tau_le2": int(count_true_tau_le2),
        "count_true_tau_le5": int(count_true_tau_le5),
        "max_proxy_pred_tau_le2": float(max_proxy_pred_tau_le2),
        "max_final_pred_tau_le2": float(max_final_pred_tau_le2),
        "max_proxy_pred_tau_le5": float(max_proxy_pred_tau_le5),
        "max_final_pred_tau_le5": float(max_final_pred_tau_le5),
        "pred_fast_proxy_stage_counts": pred_fast_proxy_stage,
        "pred_fast_final_stage_counts": pred_fast_final_stage,
        "true_fast_stage_counts": true_fast_stage,
    }


def _top_candidate_payloads(
    ordered_final_records: Sequence[dict],
    candidate_rows_by_sig: Dict[Tuple[object, ...], dict],
    limit: int = 10,
) -> List[dict]:
    payloads: List[dict] = []
    for record in list(ordered_final_records)[: max(1, int(limit))]:
        sig = candidate_signature(record["hint"])
        payloads.append(copy.deepcopy(candidate_rows_by_sig[sig]))
    return payloads


def _attach_dynamic_metrics_to_record(
    record: dict,
    score_table,
    first_reveal_tables,
    profile_weights,
    cfg,
) -> dict:
    record = copy.deepcopy(record)
    metrics = _initial_prob_rank_margin(score_table, profile_weights, cfg)
    record["stats"].update(metrics)
    record["stats"]["pred_tau_le2_exact"] = _exactish_tau_le2(score_table, first_reveal_tables, profile_weights, cfg)
    record["initial_correct_prob_mean"] = record["stats"].get("initial_correct_prob_mean")
    record["initial_correct_rank_mean"] = record["stats"].get("initial_correct_rank_mean")
    record["initial_correct_margin_mean"] = record["stats"].get("initial_correct_margin_mean")
    return record


def _diagnose_row(task_id: str, seed: int, experiment_name: str) -> dict:
    cfg = _experiment_cfg(experiment_name)
    cfg.seed = int(seed)
    prepared = prepare_one_hint_experiment(task_id=task_id, cfg=cfg, seed=int(seed))
    rng = np.random.default_rng(int(prepared.seed_bundle.get("plan", prepared.seed)))

    candidates = build_hint_candidates(prepared.context, prepared.teach_case, cfg, rng)
    if experiment_name == "fast_explicit_dynamic_answer_neighbor":
        answer_neighbors = _build_answer_neighbor_candidates(prepared.context, prepared.teach_case, cfg, rng)
        candidates = _dedupe_hint_candidates([*candidates, *answer_neighbors])

    all_candidates = [None, *candidates]
    no_hint_sig = candidate_signature(None)

    candidate_cache = build_score_tables_for_candidates(
        posterior=prepared.posterior,
        candidates=all_candidates,
        teach_case=prepared.teach_case,
        cfg=cfg,
        counters=None,
    )
    prefilter_profiles = prepared.posterior.profiles_for_stage("prefilter", cfg)
    refine_profiles = prepared.posterior.profiles_for_stage("refine", cfg)

    tau2_cfg = copy.deepcopy(cfg)
    tau2_cfg.first_reveal_top_b = max(1, int(prepared.cfg.teach_menu_size) - 1)
    proxy_tau2_tables = build_first_reveal_tables_for_candidates(
        posterior=prepared.posterior,
        candidates=all_candidates,
        candidate_cache=candidate_cache,
        teach_case=prepared.teach_case,
        profile_weights=prefilter_profiles,
        cfg=tau2_cfg,
        counters=None,
    )

    prefilter_records: Dict[Tuple[object, ...], dict] = {}
    for hint in all_candidates:
        sig = candidate_signature(hint)
        stats = prefilter_score_table_under_profiles(candidate_cache[sig], prefilter_profiles, cfg)
        record = _record_for_hint(
            hint=hint,
            stats=stats,
            utility=float(stats.get("prefilter_score", 0.0)),
            stage="prefilter",
            cfg=cfg,
        )
        prefilter_records[sig] = _attach_dynamic_metrics_to_record(
            record=record,
            score_table=candidate_cache[sig],
            first_reveal_tables=proxy_tau2_tables.get(sig),
            profile_weights=prefilter_profiles,
            cfg=cfg,
        )

    ranked_prefilter = sorted(
        [record for sig, record in prefilter_records.items() if sig != no_hint_sig],
        key=lambda item: float(item["utility"]),
        reverse=True,
    )
    ranked_prefilter = filter_candidate_records_for_stage(ranked_prefilter, cfg, stage="prefilter")
    ranked_prefilter = _family_aware_prefilter(ranked_prefilter, cfg)
    prefilter_sigs = {candidate_signature(record["hint"]) for record in ranked_prefilter}
    proxy_sigs = set(prefilter_sigs)

    proxy_first_reveal_tables = build_first_reveal_tables_for_candidates(
        posterior=prepared.posterior,
        candidates=all_candidates,
        candidate_cache=candidate_cache,
        teach_case=prepared.teach_case,
        profile_weights=prefilter_profiles,
        cfg=cfg,
        counters=None,
    )

    bonus_cfg = _bonus_cfg(cfg)
    proxy_bonus_baseline_stats = evaluate_score_table_under_profiles(
        score_table=candidate_cache[no_hint_sig],
        profile_weights=prefilter_profiles,
        cfg=bonus_cfg,
        seed=int(rng.integers(0, 2**31 - 1)),
        stage="proxy",
        first_reveal_tables=proxy_first_reveal_tables.get(no_hint_sig),
        counters=None,
    )
    no_hint_proxy_record = _record_for_hint(
        hint=None,
        stats=proxy_bonus_baseline_stats,
        utility=0.0,
        stage="proxy",
        cfg=cfg,
    )
    no_hint_proxy_record = _attach_dynamic_metrics_to_record(
        record=no_hint_proxy_record,
        score_table=candidate_cache[no_hint_sig],
        first_reveal_tables=proxy_tau2_tables.get(no_hint_sig),
        profile_weights=prefilter_profiles,
        cfg=cfg,
    )

    proxy_records: Dict[Tuple[object, ...], dict] = {no_hint_sig: no_hint_proxy_record}
    proxy_contribs: Dict[Tuple[object, ...], dict] = {no_hint_sig: {"final_utility": 0.0}}
    for idx, hint in enumerate(candidates):
        sig = candidate_signature(hint)
        stats = evaluate_score_table_under_profiles(
            score_table=candidate_cache[sig],
            profile_weights=prefilter_profiles,
            cfg=cfg,
            seed=int(rng.integers(0, 2**31 - 1)) + idx,
            stage="proxy",
            first_reveal_tables=proxy_first_reveal_tables.get(sig),
            counters=None,
        )
        record = _record_for_hint(hint=hint, stats=stats, utility=0.0, stage="proxy", cfg=cfg)
        record = _attach_dynamic_metrics_to_record(
            record=record,
            score_table=candidate_cache[sig],
            first_reveal_tables=proxy_tau2_tables.get(sig),
            profile_weights=prefilter_profiles,
            cfg=cfg,
        )
        contribs = _utility_contribs(experiment_name, record["stats"], no_hint_proxy_record["stats"], cfg)
        record["utility"] = float(contribs["final_utility"])
        record["selection_score"] = float(record["utility"])
        proxy_records[sig] = record
        proxy_contribs[sig] = dict(contribs)

    ranked_proxy = sorted(
        [proxy_records[sig] for sig in proxy_sigs],
        key=lambda item: float(item["utility"]),
        reverse=True,
    )
    ranked_proxy = filter_candidate_records_for_stage(ranked_proxy, cfg, stage="proxy")
    refine_pool = ranked_proxy[: max(1, int(getattr(cfg, "refine_top_k", len(ranked_proxy))))]
    refine_sigs = {candidate_signature(record["hint"]) for record in refine_pool}

    refine_candidates = [None, *[record["hint"] for record in refine_pool]]
    refine_first_reveal_tables = build_first_reveal_tables_for_candidates(
        posterior=prepared.posterior,
        candidates=refine_candidates,
        candidate_cache=candidate_cache,
        teach_case=prepared.teach_case,
        profile_weights=refine_profiles,
        cfg=cfg,
        counters=None,
    )
    final_tau2_tables = build_first_reveal_tables_for_candidates(
        posterior=prepared.posterior,
        candidates=all_candidates,
        candidate_cache=candidate_cache,
        teach_case=prepared.teach_case,
        profile_weights=refine_profiles,
        cfg=tau2_cfg,
        counters=None,
    )

    refine_bonus_baseline_stats = evaluate_score_table_under_profiles(
        score_table=candidate_cache[no_hint_sig],
        profile_weights=refine_profiles,
        cfg=bonus_cfg,
        seed=int(rng.integers(0, 2**31 - 1)),
        stage="refine",
        first_reveal_tables=refine_first_reveal_tables.get(no_hint_sig),
        counters=None,
    )
    no_hint_final_record = _record_for_hint(
        hint=None,
        stats=refine_bonus_baseline_stats,
        utility=0.0,
        stage="refine",
        cfg=cfg,
    )
    no_hint_final_record = _attach_dynamic_metrics_to_record(
        record=no_hint_final_record,
        score_table=candidate_cache[no_hint_sig],
        first_reveal_tables=final_tau2_tables.get(no_hint_sig),
        profile_weights=refine_profiles,
        cfg=cfg,
    )

    final_records: Dict[Tuple[object, ...], dict] = dict(proxy_records)
    final_records[no_hint_sig] = no_hint_final_record
    final_contribs: Dict[Tuple[object, ...], dict] = dict(proxy_contribs)
    for idx, hint in enumerate([record["hint"] for record in refine_pool]):
        sig = candidate_signature(hint)
        stats = evaluate_score_table_under_profiles(
            score_table=candidate_cache[sig],
            profile_weights=refine_profiles,
            cfg=cfg,
            seed=int(rng.integers(0, 2**31 - 1)) + idx,
            stage="refine",
            first_reveal_tables=refine_first_reveal_tables.get(sig),
            counters=None,
        )
        record = _record_for_hint(hint=hint, stats=stats, utility=0.0, stage="refine", cfg=cfg)
        record = _attach_dynamic_metrics_to_record(
            record=record,
            score_table=candidate_cache[sig],
            first_reveal_tables=final_tau2_tables.get(sig),
            profile_weights=refine_profiles,
            cfg=cfg,
        )
        contribs = _utility_contribs(experiment_name, record["stats"], no_hint_final_record["stats"], cfg)
        record["utility"] = float(contribs["final_utility"])
        record["selection_score"] = float(record["utility"])
        final_records[sig] = record
        final_contribs[sig] = dict(contribs)

    final_candidate_records = filter_candidate_records_for_stage(
        [record for sig, record in final_records.items() if sig != no_hint_sig],
        cfg,
        stage="final",
    )
    final_candidate_records.sort(key=lambda item: float(item["utility"]), reverse=True)
    final_sigs = {candidate_signature(record["hint"]) for record in final_candidate_records}

    no_hint_record = final_records[no_hint_sig]
    selected_record = _choose_best_candidate(
        no_hint_record=no_hint_record,
        candidate_records=final_candidate_records,
        cfg=cfg,
        reference_stats=no_hint_record["stats"],
    )
    selected_hint = selected_record["hint"]
    selected_sig = candidate_signature(selected_hint)

    actual_truth: Dict[Tuple[object, ...], dict] = {}
    for hint in candidates:
        sig = candidate_signature(hint)
        actual_truth[sig] = _actual_candidate_result(prepared, hint)

    candidate_rows_by_sig: Dict[Tuple[object, ...], dict] = {}
    for hint in candidates:
        sig = candidate_signature(hint)
        candidate_rows_by_sig[sig] = _candidate_diag_payload(
            hint=hint,
            prefilter_stats=prefilter_records.get(sig, {}).get("stats"),
            proxy_stats=proxy_records.get(sig, {}).get("stats"),
            final_stats=final_records.get(sig, {}).get("stats"),
            actual=actual_truth.get(sig),
            contribs=final_contribs.get(sig),
            stage_flags=_stage_flags(sig, prefilter_sigs, proxy_sigs, refine_sigs, final_sigs, selected_sig),
        )

    selected_actual = None if selected_hint is None else actual_truth.get(selected_sig)
    availability = _availability_summary(
        candidates=candidates,
        prefilter_records=prefilter_records,
        proxy_records=proxy_records,
        final_records=final_records,
        actual_truth=actual_truth,
        prefilter_sigs=prefilter_sigs,
        proxy_sigs=proxy_sigs,
        refine_sigs=refine_sigs,
        final_sigs=final_sigs,
        selected_sig=selected_sig,
    )

    top_candidates = _top_candidate_payloads(
        ordered_final_records=final_candidate_records,
        candidate_rows_by_sig=candidate_rows_by_sig,
        limit=10,
    )

    return {
        "task_id": task_id,
        "seed": int(seed),
        "experiment": experiment_name,
        "selected_hint": _hint_payload(selected_hint),
        "selected_stage": None if selected_hint is None else selected_record.get("stage"),
        "selected_utility": float(selected_record.get("utility", 0.0)),
        "selected_proxy_pred_tau_le2": None if selected_hint is None else float(
            proxy_records[selected_sig]["stats"].get("pred_tau_le2_exact", 0.0)
        ),
        "selected_final_pred_tau_le2": None if selected_hint is None else float(
            final_records[selected_sig]["stats"].get("pred_tau_le2_exact", 0.0)
        ),
        "selected_proxy_pred_success_prob": None if selected_hint is None else float(
            proxy_records[selected_sig]["stats"].get("success_prob", 0.0)
        ),
        "selected_final_pred_success_prob": None if selected_hint is None else float(
            final_records[selected_sig]["stats"].get("success_prob", 0.0)
        ),
        "selected_actual": selected_actual,
        "availability": availability,
        "candidate_rows": [candidate_rows_by_sig[candidate_signature(hint)] for hint in candidates],
        "top_candidates": top_candidates,
    }


def _mean(values: List[float]) -> Optional[float]:
    return None if not values else float(mean(values))


def _aggregate_rows(rows: Sequence[dict]) -> dict:
    def _sel(path: str) -> List[float]:
        vals: List[float] = []
        for row in rows:
            current = row
            for part in path.split("."):
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(part)
            if current is not None:
                vals.append(float(current))
        return vals

    selected_family_counts: Dict[str, int] = {}
    for row in rows:
        hint = row.get("selected_hint")
        if not isinstance(hint, dict):
            continue
        family = hint.get("metadata", {}).get("family", hint.get("kind"))
        selected_family_counts[str(family)] = selected_family_counts.get(str(family), 0) + 1

    def _stage_mean(section: str, key: str) -> Optional[float]:
        vals = []
        for row in rows:
            payload = row.get("availability", {}).get(section, {})
            if key in payload:
                vals.append(float(payload[key]))
        return _mean(vals)

    return {
        "n_rows": len(rows),
        "selected_success_mean": _mean(_sel("selected_actual.success")),
        "selected_tau_mean": _mean(_sel("selected_actual.tau")),
        "selected_wrong_before_correct_mean": _mean(_sel("selected_actual.wrong_before_correct")),
        "selected_eval_cell_mean": _mean(_sel("selected_actual.eval_cell")),
        "selected_eval_exact_mean": _mean(_sel("selected_actual.eval_exact")),
        "selected_proxy_pred_success_prob_mean": _mean(_sel("selected_proxy_pred_success_prob")),
        "selected_final_pred_success_prob_mean": _mean(_sel("selected_final_pred_success_prob")),
        "selected_proxy_pred_tau_le2_mean": _mean(_sel("selected_proxy_pred_tau_le2")),
        "selected_final_pred_tau_le2_mean": _mean(_sel("selected_final_pred_tau_le2")),
        "candidate_count_mean": _mean(_sel("availability.candidate_count")),
        "count_initial_rank_le2_mean": _mean(_sel("availability.count_initial_rank_le2")),
        "count_initial_rank_le5_mean": _mean(_sel("availability.count_initial_rank_le5")),
        "count_pred_tau_le2_proxy_mean": _mean(_sel("availability.count_pred_tau_le2_proxy")),
        "count_pred_tau_le2_final_mean": _mean(_sel("availability.count_pred_tau_le2_final")),
        "count_true_tau_le2_mean": _mean(_sel("availability.count_true_tau_le2")),
        "count_true_tau_le5_mean": _mean(_sel("availability.count_true_tau_le5")),
        "max_proxy_pred_tau_le2_mean": _mean(_sel("availability.max_proxy_pred_tau_le2")),
        "max_final_pred_tau_le2_mean": _mean(_sel("availability.max_final_pred_tau_le2")),
        "max_proxy_pred_tau_le5_mean": _mean(_sel("availability.max_proxy_pred_tau_le5")),
        "max_final_pred_tau_le5_mean": _mean(_sel("availability.max_final_pred_tau_le5")),
        "pred_fast_proxy_all_mean": _stage_mean("pred_fast_proxy_stage_counts", "all"),
        "pred_fast_proxy_prefilter_mean": _stage_mean("pred_fast_proxy_stage_counts", "prefilter"),
        "pred_fast_proxy_proxy_mean": _stage_mean("pred_fast_proxy_stage_counts", "proxy"),
        "pred_fast_proxy_refine_mean": _stage_mean("pred_fast_proxy_stage_counts", "refine"),
        "pred_fast_proxy_final_mean": _stage_mean("pred_fast_proxy_stage_counts", "final"),
        "pred_fast_proxy_selected_rate": _stage_mean("pred_fast_proxy_stage_counts", "selected"),
        "pred_fast_final_all_mean": _stage_mean("pred_fast_final_stage_counts", "all"),
        "pred_fast_final_prefilter_mean": _stage_mean("pred_fast_final_stage_counts", "prefilter"),
        "pred_fast_final_proxy_mean": _stage_mean("pred_fast_final_stage_counts", "proxy"),
        "pred_fast_final_refine_mean": _stage_mean("pred_fast_final_stage_counts", "refine"),
        "pred_fast_final_final_mean": _stage_mean("pred_fast_final_stage_counts", "final"),
        "pred_fast_final_selected_rate": _stage_mean("pred_fast_final_stage_counts", "selected"),
        "true_fast_all_mean": _stage_mean("true_fast_stage_counts", "all"),
        "true_fast_prefilter_mean": _stage_mean("true_fast_stage_counts", "prefilter"),
        "true_fast_proxy_mean": _stage_mean("true_fast_stage_counts", "proxy"),
        "true_fast_refine_mean": _stage_mean("true_fast_stage_counts", "refine"),
        "true_fast_final_mean": _stage_mean("true_fast_stage_counts", "final"),
        "true_fast_selected_rate": _stage_mean("true_fast_stage_counts", "selected"),
        "selected_family_counts": dict(sorted(selected_family_counts.items())),
    }


def _markdown_summary(payload: dict) -> str:
    lines: List[str] = []
    lines.append("# Fast Utility Diagnostic Summary")
    lines.append("")
    lines.append(f"- Task: `{payload.get('task_id')}`")
    lines.append(f"- Seeds: `{payload.get('seeds')}`")
    lines.append(f"- Fast prediction threshold: `{payload.get('fast_pred_threshold')}`")
    lines.append("")
    for exp in payload.get("experiments", []):
        agg = exp.get("aggregate", {})
        lines.append(f"## {exp['name']}")
        lines.append("")
        lines.append(f"- Rows: {agg.get('n_rows')}")
        lines.append(f"- Selected success: {agg.get('selected_success_mean')}")
        lines.append(f"- Selected tau mean: {agg.get('selected_tau_mean')}")
        lines.append(f"- Selected wrong-before-correct mean: {agg.get('selected_wrong_before_correct_mean')}")
        lines.append(f"- Selected eval cell: {agg.get('selected_eval_cell_mean')}")
        lines.append(f"- Selected eval exact: {agg.get('selected_eval_exact_mean')}")
        lines.append(f"- Selected proxy predicted success: {agg.get('selected_proxy_pred_success_prob_mean')}")
        lines.append(f"- Selected final predicted success: {agg.get('selected_final_pred_success_prob_mean')}")
        lines.append(f"- Selected proxy predicted P(tau<=2): {agg.get('selected_proxy_pred_tau_le2_mean')}")
        lines.append(f"- Selected final predicted P(tau<=2): {agg.get('selected_final_pred_tau_le2_mean')}")
        lines.append(f"- Candidate count: {agg.get('candidate_count_mean')}")
        lines.append(f"- Count initial rank<=2: {agg.get('count_initial_rank_le2_mean')}")
        lines.append(f"- Count initial rank<=5: {agg.get('count_initial_rank_le5_mean')}")
        lines.append(f"- Count proxy predicted tau<=2: {agg.get('count_pred_tau_le2_proxy_mean')}")
        lines.append(f"- Count final predicted tau<=2: {agg.get('count_pred_tau_le2_final_mean')}")
        lines.append(f"- Count true tau<=2: {agg.get('count_true_tau_le2_mean')}")
        lines.append(f"- Count true tau<=5: {agg.get('count_true_tau_le5_mean')}")
        lines.append(f"- Max proxy predicted P(tau<=2): {agg.get('max_proxy_pred_tau_le2_mean')}")
        lines.append(f"- Max final predicted P(tau<=2): {agg.get('max_final_pred_tau_le2_mean')}")
        lines.append(f"- Max proxy predicted P(tau<=5): {agg.get('max_proxy_pred_tau_le5_mean')}")
        lines.append(f"- Max final predicted P(tau<=5): {agg.get('max_final_pred_tau_le5_mean')}")
        lines.append(
            f"- Pred-fast proxy stage means: all={agg.get('pred_fast_proxy_all_mean')}, "
            f"prefilter={agg.get('pred_fast_proxy_prefilter_mean')}, "
            f"proxy={agg.get('pred_fast_proxy_proxy_mean')}, "
            f"refine={agg.get('pred_fast_proxy_refine_mean')}, "
            f"final={agg.get('pred_fast_proxy_final_mean')}, "
            f"selected={agg.get('pred_fast_proxy_selected_rate')}"
        )
        lines.append(
            f"- Pred-fast final stage means: all={agg.get('pred_fast_final_all_mean')}, "
            f"prefilter={agg.get('pred_fast_final_prefilter_mean')}, "
            f"proxy={agg.get('pred_fast_final_proxy_mean')}, "
            f"refine={agg.get('pred_fast_final_refine_mean')}, "
            f"final={agg.get('pred_fast_final_final_mean')}, "
            f"selected={agg.get('pred_fast_final_selected_rate')}"
        )
        lines.append(
            f"- True-fast stage means: all={agg.get('true_fast_all_mean')}, "
            f"prefilter={agg.get('true_fast_prefilter_mean')}, "
            f"proxy={agg.get('true_fast_proxy_mean')}, "
            f"refine={agg.get('true_fast_refine_mean')}, "
            f"final={agg.get('true_fast_final_mean')}, "
            f"selected={agg.get('true_fast_selected_rate')}"
        )
        if agg.get("selected_family_counts"):
            lines.append(f"- Selected families: `{json.dumps(agg['selected_family_counts'], ensure_ascii=False)}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _worker(job: tuple[str, int, str]) -> dict:
    task_id, seed, experiment = job
    return _diagnose_row(task_id=task_id, seed=seed, experiment_name=experiment)


def run_diagnostics(
    task_id: str,
    seeds: Sequence[int],
    workers: int = 1,
    executor_kind: str = "process",
) -> dict:
    experiment_names = [
        "balanced_full",
        "fast_soft_tau_center2",
        "fast_explicit_dynamic",
        "fast_explicit_dynamic_answer_neighbor",
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
        "fast_pred_threshold": FAST_PRED_THRESHOLD,
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
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fast-utility diagnostic experiment.")
    parser.add_argument("--task", default="000001")
    parser.add_argument("--seeds", default="4:5")
    parser.add_argument(
        "--out",
        default="cls_option_tutor/one_hint_tutor/grids/final_presentation/fast_utility_diagnostic_seed4_results.json",
    )
    parser.add_argument(
        "--summary-md",
        default="cls_option_tutor/one_hint_tutor/grids/final_presentation/fast_utility_diagnostic_seed4_summary.md",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--executor", choices=["thread", "process"], default="process")
    args = parser.parse_args()

    seeds = parse_seed_spec(args.seeds)
    payload = run_diagnostics(
        task_id=str(args.task),
        seeds=seeds,
        workers=int(args.workers),
        executor_kind=str(args.executor),
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
                "experiments": [exp["name"] for exp in payload["experiments"]],
                "out": str(out_path),
                "summary_md": str(summary_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
