from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .experiment_matrix import (
    _actual_early_no_transfer,
    _actual_soft_tau_score,
    _bounded_band_success,
    _condition_eval_cell,
    _condition_eval_exact,
    _condition_success,
    _evaluate_actual_hint_bundle,
)
from .experiment_presets import apply_named_presets, resolved_no_tutor_tplush_limit
from .hint_planner import filter_candidate_records_for_stage
from .hint_space import build_hint_candidates
from .learner_runner import run_teach_condition
from .metrics import band_success_prob, compute_hint_utility, early_success_prob
from .protocol import prepare_one_hint_experiment
from .reranker import candidate_feature_map
from .rollout import build_score_tables_for_candidates, candidate_signature, evaluate_score_table_under_profiles, prefilter_score_table_under_profiles


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


def _record_for_export(hint, stats: dict, utility: float, stage: str, cfg) -> dict:
    return {
        "hint": hint,
        "kind": "none" if hint is None else hint.kind,
        "difficulty": "none" if hint is None else hint.difficulty,
        "source_index": None if hint is None else hint.source_index,
        "metadata": {} if hint is None else dict(hint.metadata),
        "utility": float(utility),
        "band_success_prob": float(band_success_prob(stats, cfg)),
        "early_success_prob": float(early_success_prob(stats, cfg)),
        "stage": stage,
        "stats": stats,
        **stats,
    }


def build_candidate_dataset_rows(
    task_id: str,
    cfg,
    seed: int,
    families: Optional[Sequence[str]] = None,
) -> List[dict]:
    cfg = copy.deepcopy(cfg)
    cfg.seed = int(seed)
    if families:
        cfg.hint_mode = "combined"
        cfg.hint_families = tuple(str(f).strip() for f in families if str(f).strip())
    apply_named_presets(cfg)

    prepared = prepare_one_hint_experiment(task_id=task_id, cfg=cfg, seed=seed)
    pool_rng = np.random.default_rng(int(prepared.seed_bundle.get("oracle", prepared.seed)) + 17011)
    candidates = build_hint_candidates(prepared.context, prepared.teach_case, prepared.cfg, pool_rng)
    all_candidates = [None, *candidates]
    cache = build_score_tables_for_candidates(
        posterior=prepared.posterior,
        candidates=all_candidates,
        teach_case=prepared.teach_case,
        cfg=prepared.cfg,
        counters=None,
    )
    profile_weights = prepared.posterior.profiles_for_stage("prefilter", prepared.cfg)
    proxy_seed_rng = np.random.default_rng(int(prepared.seed_bundle.get("plan", prepared.seed)) + 19001)

    no_hint_stats = evaluate_score_table_under_profiles(
        score_table=cache[candidate_signature(None)],
        profile_weights=profile_weights,
        cfg=prepared.cfg,
        seed=int(proxy_seed_rng.integers(0, 2**31 - 1)),
        stage="proxy",
        counters=None,
    )
    no_hint_record = _record_for_export(None, no_hint_stats, 0.0, "proxy", prepared.cfg)

    no_tutor_t = run_teach_condition(
        base_learner=prepared.base_learner,
        context=prepared.context,
        teach_case=prepared.teach_case,
        max_attempts=int(prepared.cfg.max_attempts_main),
        hint=None,
        eval_items=prepared.eval_items,
        condition_name="no_tutor_T",
    )
    no_tutor_tplush = run_teach_condition(
        base_learner=prepared.base_learner,
        context=prepared.context,
        teach_case=prepared.teach_case,
        max_attempts=int(resolved_no_tutor_tplush_limit(prepared.cfg)),
        hint=None,
        eval_items=prepared.eval_items,
        condition_name="no_tutor_TplusH",
    )

    rows: List[dict] = []
    for idx, hint in enumerate(candidates):
        sig = candidate_signature(hint)
        prefilter_stats = prefilter_score_table_under_profiles(cache[sig], profile_weights, prepared.cfg)
        proxy_stats = evaluate_score_table_under_profiles(
            score_table=cache[sig],
            profile_weights=profile_weights,
            cfg=prepared.cfg,
            seed=int(proxy_seed_rng.integers(0, 2**31 - 1)) + idx,
            stage="proxy",
            counters=None,
        )
        proxy_record = _record_for_export(
            hint=hint,
            stats=proxy_stats,
            utility=compute_hint_utility(proxy_stats, prepared.cfg, no_hint_stats=no_hint_stats),
            stage="proxy",
            cfg=prepared.cfg,
        )
        # Keep the oracle dataset broad enough to include candidates that survive
        # the proxy screen, even if later final-stage filters would reject them.
        if not filter_candidate_records_for_stage([proxy_record], prepared.cfg, stage="proxy"):
            continue

        actual_conditions = _evaluate_actual_hint_bundle(prepared, pre_hints=[hint])
        tutor_t = actual_conditions.get("tutor_T6")
        tutor_unlimited = actual_conditions.get("tutor_unlimited")
        actual_success = _condition_success(tutor_t)
        actual_band = _bounded_band_success(tutor_t, prepared.cfg)
        actual_eval_cell = _condition_eval_cell(tutor_t)
        actual_eval_exact = _condition_eval_exact(tutor_t)
        actual_early = 1.0 if tutor_t is not None and tutor_t.first_correct_attempt is not None and int(tutor_t.first_correct_attempt) < int(getattr(prepared.cfg, "target_tau_min", 3)) else 0.0
        actual_early_no_transfer = _actual_early_no_transfer(tutor_t, no_tutor_tplush, prepared.cfg)
        actual_collapse = 1.0 if tutor_t is not None and tutor_t.failure_type == "post_reveal_collapse" else 0.0
        actual_soft_tau = _actual_soft_tau_score(tutor_t, prepared.cfg)
        delta_success_vs_no_tutor = actual_success - _condition_success(no_tutor_tplush)
        delta_eval_cell_vs_no_tutor = actual_eval_cell - _condition_eval_cell(no_tutor_tplush)
        search_target = (
            2.0 * actual_success
            + 1.5 * actual_band
            + 1.0 * actual_soft_tau
            + 1.0 * max(0.0, delta_success_vs_no_tutor)
            - 1.5 * actual_early_no_transfer
            - 1.0 * actual_collapse
        )
        transfer_target = (
            2.0 * max(0.0, delta_eval_cell_vs_no_tutor)
            + 0.5 * actual_success
            + 0.5 * actual_band
            - 1.0 * actual_early_no_transfer
            - 1.0 * actual_collapse
        )
        feature_record = dict(proxy_record)
        feature_record["prefilter_score"] = float(prefilter_stats.get("prefilter_score", 0.0))
        features = candidate_feature_map(feature_record, prepared.context, prepared.teach_case, prepared.cfg)
        rows.append(
            {
                "task_id": task_id,
                "seed": int(seed),
                "teach_case_metadata": dict(getattr(prepared.teach_case, "metadata", {}) or {}),
                "hint": _hint_payload(hint),
                "prefilter": {
                    "prefilter_score": float(prefilter_stats.get("prefilter_score", 0.0)),
                    "initial_correct_prob_mean": prefilter_stats.get("initial_correct_prob_mean"),
                    "initial_correct_rank_mean": prefilter_stats.get("initial_correct_rank_mean"),
                },
                "proxy": {
                    "utility": float(proxy_record.get("utility", 0.0)),
                    "success_prob": float(proxy_record.get("success_prob", 0.0)),
                    "band_success_prob": float(proxy_record.get("band_success_prob", 0.0)),
                    "early_success_prob": float(proxy_record.get("early_success_prob", 0.0)),
                    "eval_cell_acc": float(proxy_record.get("eval_cell_acc", 0.0)),
                    "eval_exact_acc": float(proxy_record.get("eval_exact_acc", 0.0)),
                    "initial_correct_prob_mean": proxy_record.get("initial_correct_prob_mean"),
                    "initial_correct_rank_mean": proxy_record.get("initial_correct_rank_mean"),
                    "mean_first_correct_attempt": proxy_record.get("mean_first_correct_attempt"),
                    "safe_wrong_mean": float(proxy_record.get("safe_wrong_mean", 0.0)),
                },
                "baselines": {
                    "no_tutor_T_success": _condition_success(no_tutor_t),
                    "no_tutor_TplusH_success": _condition_success(no_tutor_tplush),
                    "no_tutor_T_eval_cell": _condition_eval_cell(no_tutor_t),
                    "no_tutor_TplusH_eval_cell": _condition_eval_cell(no_tutor_tplush),
                },
                "actual": {
                    "success": actual_success,
                    "band_success": actual_band,
                    "early_success": actual_early,
                    "early_no_transfer": actual_early_no_transfer,
                    "collapse": actual_collapse,
                    "first_correct_attempt": None if tutor_t is None else tutor_t.first_correct_attempt,
                    "unlimited_first_correct_attempt": None if tutor_unlimited is None else tutor_unlimited.first_correct_attempt,
                    "eval_exact_acc": actual_eval_exact,
                    "eval_cell_acc": actual_eval_cell,
                    "failure_type": None if tutor_t is None else tutor_t.failure_type,
                    "delta_success_vs_no_tutor_TplusH": delta_success_vs_no_tutor,
                    "delta_eval_cell_vs_no_tutor_TplusH": delta_eval_cell_vs_no_tutor,
                    "soft_tau_score": actual_soft_tau,
                },
                "labels": {
                    "search_target": float(search_target),
                    "transfer_target": float(transfer_target),
                    "good_search_label": bool(actual_success >= 1.0 and actual_early_no_transfer <= 0.0),
                    "good_transfer_label": bool(delta_eval_cell_vs_no_tutor >= 0.03 and actual_success >= 1.0),
                },
                "reranker_features": features,
                "planner_reference": {
                    "no_hint_utility": float(no_hint_record.get("utility", 0.0)),
                },
            }
        )
    return rows


def write_candidate_dataset(
    rows: Iterable[dict],
    out_path: str,
) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
