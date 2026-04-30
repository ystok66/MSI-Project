from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .experiment_presets import resolved_no_tutor_tplush_limit
from .hint_space import build_hint_candidates
from .interfaces import (
    HintCandidate,
    HintPlanResult,
    PlannerCounters,
    PlannerPrediction,
    PlannerStageScore,
    TaskContext,
    TeachCase,
)
from .metrics import (
    band_success_prob,
    compute_hint_utility,
    conservative_reveal_penalty,
    early_success_prob,
    passes_success_gate,
)
from .rollout import (
    build_first_reveal_tables_for_candidates,
    build_score_tables_for_candidates,
    candidate_signature,
    eval_proxy_subset_items,
    evaluate_hint_under_posterior,
    evaluate_score_table_under_profiles,
    evaluate_static_hint_eval_proxies,
    prefilter_score_table_under_profiles,
)
from .reranker import apply_reranker


def _hint_identity(hint: Optional[HintCandidate]) -> Tuple[str, str, Optional[int]]:
    if hint is None:
        return ("none", "none", None)
    return (hint.kind, hint.difficulty, hint.source_index)


def _profiles_payload(profile_weights: Sequence[Tuple[object, float]]) -> List[Dict[str, object]]:
    payload: List[Dict[str, object]] = []
    for profile, weight in profile_weights:
        payload.append(
            {
                "name": getattr(profile, "name", str(profile)),
                "weight": float(weight),
            }
        )
    return payload


def _tau_mode(stats: dict) -> Optional[int]:
    tau_vec = [float(x) for x in stats.get("pred_p_tau_1_to_6", [])]
    if not tau_vec:
        return None
    best = max(tau_vec)
    if best <= 0.0:
        return None
    return int(np.argmax(np.asarray(tau_vec, dtype=float))) + 1


def _record_for_hint(
    hint: Optional[HintCandidate],
    stats: dict,
    utility: float,
    stage: str,
    cfg=None,
) -> Dict[str, object]:
    kind, difficulty, source_index = _hint_identity(hint)
    return {
        "hint": hint,
        "kind": kind,
        "difficulty": difficulty,
        "source_index": source_index,
        "metadata": {} if hint is None else dict(hint.metadata),
        "utility": float(utility),
        "band_success_prob": 0.0 if cfg is None else float(band_success_prob(stats, cfg)),
        "early_success_prob": 0.0 if cfg is None else float(early_success_prob(stats, cfg)),
        "conservative_reveal_penalty": 0.0 if cfg is None else float(conservative_reveal_penalty(stats, cfg)),
        "stage": stage,
        "selection_score": float(utility),
        "reranker_score": 0.0,
        "stats": stats,
        **stats,
    }


_DELTA_UTILITY_MODES = {
    "delta_vs_no_tutor_bonus",
    "advantage_delta",
    "advantage_fast_success",
    "advantage_transfer",
    "advantage_mix",
}


def _transfer_eval_proxy_enabled(cfg) -> bool:
    return str(getattr(cfg, "transfer_eval_proxy_mode", "off")) not in {"", "off", "none"}


def _update_bonus_eval_fields(baseline_stats: Optional[dict], no_hint_stats: dict) -> Optional[dict]:
    if baseline_stats is None:
        return None
    updated = dict(baseline_stats)
    for key in (
        "eval_exact_acc",
        "eval_cell_acc",
        "eval_proxy_exact_acc",
        "eval_proxy_cell_acc",
        "eval_proxy_n_items",
        "eval_proxy_mode",
    ):
        if key in no_hint_stats:
            updated[key] = no_hint_stats[key]
    return updated


def _recompute_record_utilities(
    records: Dict[Tuple[object, ...], Dict[str, object]],
    cfg,
    *,
    no_hint_sig: Tuple[object, ...],
    bonus_baseline_stats: Optional[dict],
) -> None:
    utility_mode = str(getattr(cfg, "utility_mode", "legacy"))
    no_hint_stats = records.get(no_hint_sig, {}).get("stats", {})
    for sig, record in list(records.items()):
        stats = dict(record.get("stats", {}) or {})
        if sig == no_hint_sig and utility_mode in _DELTA_UTILITY_MODES:
            utility = 0.0
        else:
            utility = compute_hint_utility(
                stats,
                cfg,
                no_hint_stats=no_hint_stats,
                bonus_baseline_stats=bonus_baseline_stats,
            )
        records[sig] = _record_for_hint(
            hint=record.get("hint"),
            stats=stats,
            utility=utility,
            stage=str(record.get("stage", "proxy")),
            cfg=cfg,
        )


def _attach_transfer_eval_proxy(
    posterior,
    records: Dict[Tuple[object, ...], Dict[str, object]],
    teach_case: TeachCase,
    eval_items,
    cfg,
    rng: np.random.Generator,
    *,
    stage: str,
    counters: Optional[PlannerCounters],
) -> None:
    if not _transfer_eval_proxy_enabled(cfg) or not records:
        return
    mode = str(getattr(cfg, "transfer_eval_proxy_mode", "off"))
    record_items = list(records.items())
    if stage == "refine" and mode in {"beam_leaf_subset", "smc_leaf_subset"}:
        proxy_top_k = int(getattr(cfg, "transfer_eval_proxy_refine_top_k", 0))
        if proxy_top_k > 0:
            none_items = [
                item for item in record_items if candidate_signature(item[1].get("hint")) == candidate_signature(None)
            ]
            candidate_items = [
                item for item in record_items if candidate_signature(item[1].get("hint")) != candidate_signature(None)
            ]
            candidate_items = sorted(
                candidate_items,
                key=lambda item: float(item[1].get("selection_score", item[1].get("utility", 0.0))),
                reverse=True,
            )[:proxy_top_k]
            record_items = none_items + candidate_items
    hints = [record.get("hint") for _, record in record_items]
    if stage != "refine" or mode == "static_subset":
        proxy_by_sig = evaluate_static_hint_eval_proxies(
            posterior=posterior,
            hints=hints,
            eval_items=eval_items,
            cfg=cfg,
            counters=counters,
        )
    elif mode in {"beam_leaf_subset", "smc_leaf_subset"}:
        proxy_by_sig: Dict[Tuple[object, ...], dict] = {}
        if str(getattr(cfg, "refine_update_mode", "proxy")) in {"full_cls", "lazy_cls"}:
            # Full/lazy refine already evaluates post-teach leaf states. Reuse
            # those eval-aware stats instead of repeating the beam expansion.
            for sig, record in record_items:
                stats = dict(record.get("stats", {}) or {})
                if "eval_cell_acc" not in stats:
                    continue
                proxy_by_sig[sig] = {
                    "eval_exact_acc": float(stats.get("eval_exact_acc", 0.0)),
                    "eval_cell_acc": float(stats.get("eval_cell_acc", 0.0)),
                    "eval_proxy_exact_acc": float(stats.get("eval_exact_acc", 0.0)),
                    "eval_proxy_cell_acc": float(stats.get("eval_cell_acc", 0.0)),
                    "eval_proxy_n_items": len(list(eval_items or [])),
                    "eval_proxy_mode": "refine_leaf_reuse",
                }
        if proxy_by_sig:
            pass
        else:
            subset = eval_proxy_subset_items(eval_items, cfg)
            proxy_cfg = copy.deepcopy(cfg)
            proxy_cfg.eval_aware = True
            proxy_cfg.rollout_mode = "beam"
            proxy_cfg.beam_top_b = int(getattr(cfg, "transfer_eval_proxy_beam_top_b", cfg.beam_top_b))
            proxy_cfg.beam_keep_l = int(getattr(cfg, "transfer_eval_proxy_beam_keep_l", cfg.beam_keep_l))
            # Use the same semantic reveal semantics as refine. The current
            # "first_reveal_cached_cls" value enters the full semantic update branch
            # inside rollout, which is intentional for a leaf eval proxy.
            proxy_cfg.planning_update_mode = str(getattr(cfg, "refine_update_mode", cfg.planning_update_mode))
            proxy_cfg.profile_top_mass = float(getattr(cfg, "refine_profile_top_mass", cfg.profile_top_mass))
            proxy_cfg.profile_min_keep = int(getattr(cfg, "refine_profile_min_keep", cfg.profile_min_keep))
            for idx, hint in enumerate(hints):
                stats = evaluate_hint_under_posterior(
                    posterior=posterior,
                    hint=hint,
                    teach_case=teach_case,
                    eval_items=subset,
                    cfg=proxy_cfg,
                    seed=int(rng.integers(0, 2**31 - 1)) + idx,
                    counters=counters,
                )
                proxy_by_sig[candidate_signature(hint)] = {
                    "eval_exact_acc": float(stats.get("eval_exact_acc", 0.0)),
                    "eval_cell_acc": float(stats.get("eval_cell_acc", 0.0)),
                    "eval_proxy_exact_acc": float(stats.get("eval_exact_acc", 0.0)),
                    "eval_proxy_cell_acc": float(stats.get("eval_cell_acc", 0.0)),
                    "eval_proxy_n_items": int(len(subset)),
                    "eval_proxy_mode": "beam_leaf_subset",
                }
    else:
        return

    for sig, proxy_stats in proxy_by_sig.items():
        record = records.get(sig)
        if record is None:
            continue
        stats = dict(record.get("stats", {}) or {})
        stats.update(proxy_stats)
        record["stats"] = stats
        record.update(proxy_stats)


def _zscore(values: Sequence[float], eps: float) -> List[float]:
    vals = [float(value) for value in values]
    if not vals:
        return []
    mean = float(sum(vals) / len(vals))
    var = float(sum((value - mean) ** 2 for value in vals) / len(vals))
    std = max(float(var ** 0.5), float(eps))
    return [(value - mean) / std for value in vals]


def _apply_normalized_mix_selection_scores(
    candidate_records: Sequence[Dict[str, object]],
    no_hint_record: Dict[str, object],
    cfg,
    reference_stats: Optional[dict],
) -> List[Dict[str, object]]:
    if str(getattr(cfg, "utility_mode", "legacy")) != "advantage_mix":
        return list(candidate_records)
    if not bool(getattr(cfg, "utility_mix_normalize_components", False)):
        return list(candidate_records)
    records = [dict(record) for record in candidate_records]
    if not records:
        return records

    eta = max(0.0, min(1.0, float(getattr(cfg, "utility_mix_eta", 0.5))))
    eps = max(1e-12, float(getattr(cfg, "utility_mix_normalize_eps", 1e-6)))
    no_hint_stats = dict(no_hint_record.get("stats", {}) or {})
    baseline_stats = reference_stats

    fast_cfg = copy.copy(cfg)
    fast_cfg.utility_mode = "advantage_fast_success"
    transfer_cfg = copy.copy(cfg)
    transfer_cfg.utility_mode = "advantage_transfer"

    fast_values: List[float] = []
    transfer_values: List[float] = []
    for record in records:
        stats = dict(record.get("stats", {}) or {})
        fast_values.append(
            compute_hint_utility(
                stats,
                fast_cfg,
                no_hint_stats=no_hint_stats,
                bonus_baseline_stats=baseline_stats,
            )
        )
        transfer_values.append(
            compute_hint_utility(
                stats,
                transfer_cfg,
                no_hint_stats=no_hint_stats,
                bonus_baseline_stats=baseline_stats,
            )
        )

    fast_z = _zscore(fast_values, eps)
    transfer_z = _zscore(transfer_values, eps)
    out: List[Dict[str, object]] = []
    for record, fast_raw, transfer_raw, fast_norm, transfer_norm in zip(
        records,
        fast_values,
        transfer_values,
        fast_z,
        transfer_z,
    ):
        score = eta * float(fast_norm) + (1.0 - eta) * float(transfer_norm)
        updated = dict(record)
        updated["fast_component_delta"] = float(fast_raw)
        updated["transfer_component_delta"] = float(transfer_raw)
        updated["fast_component_z"] = float(fast_norm)
        updated["transfer_component_z"] = float(transfer_norm)
        updated["selection_score"] = float(score)
        updated["utility"] = float(score)
        out.append(updated)
    return out


def _stage_score(stage: str, hint: Optional[HintCandidate], score: float) -> PlannerStageScore:
    kind, difficulty, source_index = _hint_identity(hint)
    return PlannerStageScore(
        stage=stage,
        hint_kind=kind,
        hint_difficulty=difficulty,
        source_index=source_index,
        score=float(score),
    )


def _record_family(record: Dict[str, object]) -> str:
    metadata = dict(record.get("metadata", {}) or {})
    family = metadata.get("family")
    if family is not None:
        return str(family)
    kind = record.get("kind")
    return "none" if kind is None else str(kind)


def _target_family_filter_passes(
    record: Dict[str, object],
    cfg,
    stage: str,
) -> bool:
    family = _record_family(record)
    if family not in {
        "target_neighborhood_loose",
        "target_neighborhood_rank_filtered",
        "target_neighborhood_robust_filtered",
    }:
        return True

    hint = record.get("hint")
    if hint is None:
        return True
    metadata = dict(record.get("metadata", {}) or {})
    words = list(getattr(hint.example, "words", []) or [])
    min_words = max(1, int(getattr(cfg, "target_neighborhood_min_words", 2)))
    operator_overlap = int(metadata.get("operator_overlap", 0))
    if len(words) < min_words or operator_overlap <= 0:
        return False

    if family == "target_neighborhood_loose":
        return True

    initial_rank = record.get("initial_correct_rank_mean")
    if initial_rank is None:
        return False
    rank_min = int(getattr(cfg, "target_neighborhood_rank_min", 3))
    rank_max = int(getattr(cfg, "target_neighborhood_rank_max", 10))
    if not (rank_min <= float(initial_rank) <= rank_max):
        return False

    # Prefilter stats only include initial rank/prob proxies, not rollout-derived
    # post-reveal dynamics, so robust filtering has to wait until later stages.
    if family == "target_neighborhood_rank_filtered" or stage == "prefilter":
        return True

    attempt_probs = list(record.get("pred_attempt_correct_prob_mean", []) or [])
    attempt_ranks = list(record.get("pred_attempt_correct_rank_mean", []) or [])
    if len(attempt_ranks) < 2:
        return False
    second_rank = attempt_ranks[1]
    if second_rank is None:
        return False
    robust_rank_max = int(getattr(cfg, "target_neighborhood_robust_rank_max", 10))
    if float(second_rank) > robust_rank_max:
        return False

    if len(attempt_probs) >= 2 and attempt_probs[0] is not None and attempt_probs[1] is not None:
        base_prob = float(attempt_probs[0])
        next_prob = float(attempt_probs[1])
        if base_prob > 1e-12:
            ratio = next_prob / base_prob
            robust_ratio = float(getattr(cfg, "target_neighborhood_robust_collapse_ratio", 0.863))
            if ratio < robust_ratio:
                return False
    return True


def filter_candidate_records_for_stage(
    records: Sequence[Dict[str, object]],
    cfg,
    stage: str,
) -> List[Dict[str, object]]:
    return [record for record in records if _target_family_filter_passes(record, cfg, stage)]


def _family_aware_prefilter(
    ranked_records: Sequence[Dict[str, object]],
    cfg,
) -> List[Dict[str, object]]:
    records = list(ranked_records)
    if not records:
        return []

    top_k = max(1, int(getattr(cfg, "prefilter_top_k", len(records))))
    if not getattr(cfg, "prefilter_family_aware", False):
        return records[:top_k]

    min_per_family = max(1, int(getattr(cfg, "prefilter_min_per_family", 1)))
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(_record_family(record), []).append(record)

    selected: List[Dict[str, object]] = []
    seen = set()
    for family in sorted(grouped):
        for record in grouped[family][:min_per_family]:
            sig = candidate_signature(record.get("hint"))
            if sig in seen:
                continue
            selected.append(record)
            seen.add(sig)

    target = max(top_k, len(selected))
    for record in records:
        sig = candidate_signature(record.get("hint"))
        if sig in seen:
            continue
        selected.append(record)
        seen.add(sig)
        if len(selected) >= target:
            break
    return selected


def _record_fast_prefilter_key(record: Dict[str, object]) -> Tuple[float, float, float]:
    stats = dict(record.get("stats", {}) or {})
    return (
        float(stats.get("initial_correct_margin_mean", 0.0)),
        float(stats.get("initial_correct_prob_mean", 0.0)),
        float(record.get("selection_score", record.get("utility", 0.0))),
    )


def _record_fast_dynamic_key(record: Dict[str, object]) -> Tuple[float, float, float, float, float]:
    stats = dict(record.get("stats", {}) or {})
    return (
        float(stats.get("pred_tau_le2_exact", 0.0)),
        float(stats.get("initial_correct_margin_mean", 0.0)),
        float(stats.get("initial_correct_prob_mean", 0.0)),
        -float(stats.get("wrong_before_correct_mean", stats.get("safe_wrong_mean", 0.0))),
        float(record.get("selection_score", record.get("utility", 0.0))),
    )


def _record_transfer_key(record: Dict[str, object], cfg) -> Tuple[float, float, float, float]:
    stats = dict(record.get("stats", {}) or {})
    metadata = dict(record.get("metadata", {}) or {})
    target_bonus = 1.0 if str(metadata.get("family", "")) in {
        "target_neighborhood_rank_filtered",
        "target_neighborhood_robust_filtered",
        "answer_neighbor_nonanswer",
    } else 0.0
    return (
        float(stats.get("eval_cell_acc", 0.0)),
        float(stats.get("safe_wrong_mean", 0.0)),
        float(band_success_prob(stats, cfg)),
        target_bonus + float(record.get("selection_score", record.get("utility", 0.0))),
    )


def _objective_bucketed_records(
    ranked_records: Sequence[Dict[str, object]],
    cfg,
    *,
    stage: str,
) -> List[Dict[str, object]]:
    records = list(ranked_records)
    if not records:
        return []

    enabled = bool(
        getattr(
            cfg,
            "objective_bucketed_prefilter" if stage == "prefilter" else "objective_bucketed_refine",
            False,
        )
    )
    if not enabled:
        return records

    keep_fast = max(0, int(getattr(cfg, "prefilter_keep_fast" if stage == "prefilter" else "refine_keep_fast", 0)))
    keep_transfer = max(
        0,
        int(getattr(cfg, "prefilter_keep_transfer" if stage == "prefilter" else "refine_keep_transfer", 0)),
    )
    keep_balanced = max(
        0,
        int(getattr(cfg, "prefilter_keep_balanced" if stage == "prefilter" else "refine_keep_balanced", 0)),
    )
    if keep_fast <= 0 and keep_transfer <= 0 and keep_balanced <= 0:
        return records

    selected: List[Dict[str, object]] = []
    seen = set()
    target = int(getattr(cfg, "prefilter_top_k" if stage == "prefilter" else "refine_top_k", len(records)))

    def _extend(top_records: Sequence[Dict[str, object]]) -> None:
        for record in top_records:
            if len(selected) >= target:
                break
            sig = candidate_signature(record.get("hint"))
            if sig in seen:
                continue
            selected.append(record)
            seen.add(sig)

    if keep_fast > 0:
        fast_key = _record_fast_prefilter_key if stage == "prefilter" else _record_fast_dynamic_key
        fast_ranked = sorted(records, key=fast_key, reverse=True)
        _extend(fast_ranked[:keep_fast])
    if keep_transfer > 0:
        transfer_ranked = sorted(records, key=lambda item: _record_transfer_key(item, cfg), reverse=True)
        _extend(transfer_ranked[:keep_transfer])
    if keep_balanced > 0:
        balanced_ranked = sorted(records, key=lambda item: float(item.get("selection_score", item.get("utility", 0.0))), reverse=True)
        _extend(balanced_ranked[:keep_balanced])

    for record in records:
        if len(selected) >= target:
            break
        sig = candidate_signature(record.get("hint"))
        if sig in seen:
            continue
        selected.append(record)
        seen.add(sig)
    return selected


def _choose_best_candidate(
    no_hint_record: Dict[str, object],
    candidate_records: Sequence[Dict[str, object]],
    cfg,
    reference_stats: Optional[dict] = None,
) -> Dict[str, object]:
    if not candidate_records:
        return no_hint_record
    utility_mode = str(getattr(cfg, "utility_mode", "legacy"))
    delta_threshold = float(
        getattr(
            cfg,
            "advantage_delta_min" if utility_mode in {"advantage_delta", "advantage_fast_success", "advantage_transfer", "advantage_mix"} else "delta_min_use_hint",
            0.0,
        )
    )

    gated = [
        record
        for record in candidate_records
        if passes_success_gate(record["stats"], no_hint_record["stats"], cfg, reference_stats=reference_stats)
    ]
    if getattr(cfg, "allow_abstain", False) and not gated:
        return no_hint_record
    pool = gated if gated else list(candidate_records)
    best = max(pool, key=lambda item: float(item.get("selection_score", item["utility"])))
    transfer_like = utility_mode == "advantage_transfer"
    if utility_mode == "advantage_mix":
        eta = max(0.0, min(1.0, float(getattr(cfg, "utility_mix_eta", 0.5))))
        transfer_like = eta <= 0.25
    if (
        getattr(cfg, "allow_abstain", False)
        and transfer_like
        and str(getattr(cfg, "transfer_gate_mode", "default")) == "eval_delta"
        and gated
    ):
        # In transfer mode, the gate itself encodes "worth using": positive
        # eval delta subject to a success floor. Do not re-abstain just because
        # the mixed utility is negative relative to the search-oriented baseline.
        return best
    if getattr(cfg, "allow_abstain", False) and (
        float(best.get("selection_score", best["utility"]))
        <= float(no_hint_record.get("selection_score", no_hint_record["utility"])) + delta_threshold
    ):
        return no_hint_record
    return best


def _infer_abstain_reason(
    no_hint_record: Dict[str, object],
    candidate_records: Sequence[Dict[str, object]],
    cfg,
    reference_stats: Optional[dict] = None,
) -> Optional[str]:
    if not getattr(cfg, "allow_abstain", False):
        return None
    utility_mode = str(getattr(cfg, "utility_mode", "legacy"))
    delta_threshold = float(
        getattr(
            cfg,
            "advantage_delta_min" if utility_mode in {"advantage_delta", "advantage_fast_success", "advantage_transfer", "advantage_mix"} else "delta_min_use_hint",
            0.0,
        )
    )
    if not candidate_records:
        return "no_candidates"

    ref = reference_stats or no_hint_record.get("stats", {})
    transfer_like = utility_mode == "advantage_transfer"
    if utility_mode == "advantage_mix":
        eta = max(0.0, min(1.0, float(getattr(cfg, "utility_mix_eta", 0.5))))
        transfer_like = eta <= 0.25
    if transfer_like and str(getattr(cfg, "transfer_gate_mode", "default")) == "eval_delta":
        eval_delta_min = float(getattr(cfg, "transfer_delta_eval_min", 0.005))
        eval_candidates = [
            record
            for record in candidate_records
            if float(record.get("eval_cell_acc", 0.0)) - float(ref.get("eval_cell_acc", 0.0)) >= eval_delta_min
        ]
        if not eval_candidates:
            return "no_eval_delta_candidate"
        success_floor = max(
            float(getattr(cfg, "transfer_success_floor", 0.15)),
            float(ref.get("success_prob", 0.0)) - float(getattr(cfg, "transfer_success_slack", 0.05)),
        )
        if not any(float(record.get("success_prob", 0.0)) >= success_floor for record in eval_candidates):
            return "below_transfer_success_floor"

    no_hint_band = float(band_success_prob(ref, cfg))
    no_hint_early = float(early_success_prob(ref, cfg))
    band_threshold = no_hint_band + float(getattr(cfg, "delta_band_min", 0.0))
    band_candidates = [
        record for record in candidate_records if float(record.get("band_success_prob", 0.0)) >= band_threshold
    ]
    if not band_candidates:
        return "no_band_candidate"

    early_threshold = max(no_hint_early, 0.05)
    if all(float(record.get("early_success_prob", 0.0)) > early_threshold for record in band_candidates):
        return "early_only_candidates"

    collapse_threshold = max(
        float(getattr(cfg, "conservative_reveal_monotone_margin", 0.0)),
        1e-6,
    )
    if all(
        float(record.get("conservative_reveal_penalty", 0.0)) > collapse_threshold
        for record in band_candidates
    ):
        return "collapse_risk"

    gated = [
        record
        for record in candidate_records
        if passes_success_gate(record["stats"], no_hint_record["stats"], cfg, reference_stats=ref)
    ]
    if not gated:
        if utility_mode in {"advantage_delta", "advantage_transfer", "advantage_mix"}:
            return "early_no_transfer_only"
        return "no_success_gate_candidate"

    best_gated = max(gated, key=lambda item: float(item.get("selection_score", item["utility"])))
    if float(best_gated.get("selection_score", best_gated["utility"])) <= float(
        no_hint_record.get("selection_score", no_hint_record["utility"])
    ) + delta_threshold:
        if utility_mode == "delta_vs_no_tutor_bonus":
            return "no_positive_delta_vs_bonus"
        if utility_mode in {"advantage_delta", "advantage_fast_success", "advantage_transfer", "advantage_mix"}:
            return "no_positive_advantage_delta"
        return "no_positive_delta"
    return None


def _bonus_cfg(cfg):
    bonus_limit = int(resolved_no_tutor_tplush_limit(cfg))
    if bonus_limit == int(getattr(cfg, "max_attempts_main", bonus_limit)):
        return cfg
    bonus_cfg = copy.deepcopy(cfg)
    # The bonus baseline is the fair no-hint comparator with T+H attempts.
    # Downstream band metrics intentionally follow that longer horizon.
    bonus_cfg.max_attempts_main = bonus_limit
    return bonus_cfg


def _selected_hint_quality_tags(
    context: TaskContext,
    teach_case: TeachCase,
    hint: Optional[HintCandidate],
    stats: dict,
    cfg,
) -> Dict[str, object]:
    if hint is None:
        return {}

    operator_names = {spec.name for spec in context.operator_specs}
    teach_operators = {tok for tok in teach_case.example.words if tok in operator_names}
    hint_words = list(hint.example.words)
    hint_operators = {tok for tok in hint_words if tok in operator_names}

    initial_rank = stats.get("initial_correct_rank_mean")
    second_prob = None
    second_rank = None
    attempt_probs = list(stats.get("pred_attempt_correct_prob_mean", []))
    attempt_ranks = list(stats.get("pred_attempt_correct_rank_mean", []))
    if len(attempt_probs) > 1:
        second_prob = attempt_probs[1]
    if len(attempt_ranks) > 1:
        second_rank = attempt_ranks[1]

    first_prob = attempt_probs[0] if attempt_probs else None
    robust = None
    if first_prob is not None and second_prob is not None:
        robust = float(second_prob) >= float(first_prob) * float(getattr(cfg, "collapse_ratio_median", 1.0))
    elif second_rank is not None:
        robust = float(second_rank) <= float(int(cfg.max_attempts_main) + 2)

    initial_rank_in_band = None if initial_rank is None else (
        int(getattr(cfg, "target_tau_min", 3))
        <= float(initial_rank)
        <= int(getattr(cfg, "target_tau_max", int(cfg.max_attempts_main)))
    )
    initial_rank_reachable = None if initial_rank is None else float(initial_rank) <= float(int(cfg.max_attempts_main) + 2)

    return {
        "operator_relevant": bool(hint_operators & teach_operators) or hint.kind == "operator_probe",
        "target_neighborhood": hint.kind == "target_neighborhood" or str(hint.metadata.get("family", "")) == "answer_neighbor_nonanswer",
        "menu_based": hint.kind.startswith("menu_"),
        "ceiling": hint.kind in {"menu_correct_ceiling", "direct_answer"},
        "initial_rank_in_band": initial_rank_in_band,
        "initial_rank_reachable": initial_rank_reachable,
        "post_first_wrong_robust": robust,
        "early_success_risk": float(early_success_prob(stats, cfg)) > max(0.05, float(stats.get("success_prob", 0.0)) * 0.5),
        "transfer_proxy": float(stats.get("eval_cell_acc", 0.0)) if getattr(cfg, "eval_aware", False) else float(stats.get("safe_wrong_mean", 0.0)),
        "teach_operator_overlap": sorted(teach_operators & hint_operators),
    }


def _single_stage_select_hint(
    posterior,
    context: TaskContext,
    teach_case: TeachCase,
    eval_items,
    cfg,
    rng: np.random.Generator,
) -> HintPlanResult:
    counters = PlannerCounters() if getattr(cfg, "collect_cost_counters", False) else None
    t0 = time.perf_counter()
    eval_items = list(eval_items)

    no_hint_stats = evaluate_hint_under_posterior(
        posterior,
        None,
        teach_case,
        eval_items,
        cfg,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    bonus_baseline_stats = None
    utility_mode = str(getattr(cfg, "utility_mode", "legacy"))
    if utility_mode in {"delta_vs_no_tutor_bonus", "advantage_delta", "advantage_fast_success", "advantage_transfer", "advantage_mix"}:
        bonus_cfg = _bonus_cfg(cfg)
        bonus_baseline_stats = evaluate_hint_under_posterior(
            posterior,
            None,
            teach_case,
            eval_items,
            bonus_cfg,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        no_hint_utility = 0.0
    else:
        no_hint_utility = compute_hint_utility(no_hint_stats, cfg)

    candidates = build_hint_candidates(context, teach_case, cfg, rng)
    scores: List[Dict[str, object]] = []
    no_hint_record = _record_for_hint(None, no_hint_stats, no_hint_utility, stage="single_stage", cfg=cfg)
    stage_scores: List[PlannerStageScore] = [_stage_score("single_stage", None, no_hint_utility)]

    for idx, candidate in enumerate(candidates):
        stats = evaluate_hint_under_posterior(
            posterior,
            candidate,
            teach_case,
            eval_items,
            cfg,
            seed=int(rng.integers(0, 2**31 - 1)) + idx,
        )
        utility = compute_hint_utility(
            stats,
            cfg,
            no_hint_stats=no_hint_stats,
            bonus_baseline_stats=bonus_baseline_stats,
        )
        record = _record_for_hint(candidate, stats, utility, stage="single_stage", cfg=cfg)
        scores.append(record)
        stage_scores.append(_stage_score("single_stage", candidate, utility))

    scores = apply_reranker(scores, context, teach_case, cfg)
    best_record = _choose_best_candidate(
        no_hint_record=no_hint_record,
        candidate_records=scores,
        cfg=cfg,
        reference_stats=bonus_baseline_stats if bonus_baseline_stats is not None else no_hint_stats,
    )

    scores.sort(key=lambda item: float(item.get("selection_score", item["utility"])), reverse=True)
    if getattr(cfg, "plan_candidate_limit", None):
        scores = scores[: max(1, int(cfg.plan_candidate_limit))]

    selected_stats = best_record["stats"]
    selected_hint = best_record["hint"]
    selected_utility = float(best_record["utility"])
    delta = max(0.0, selected_utility - float(no_hint_utility)) if selected_hint is not None else 0.0
    abstain_reason = _infer_abstain_reason(
        no_hint_record,
        scores,
        cfg,
        reference_stats=bonus_baseline_stats if bonus_baseline_stats is not None else no_hint_stats,
    ) if selected_hint is None else None
    hint_quality_tags = _selected_hint_quality_tags(context, teach_case, selected_hint, selected_stats, cfg)
    if counters is not None:
        counters.select_hint_wall_time = float(time.perf_counter() - t0)

    kept_profiles = posterior.profiles_for_stage("prefilter", cfg)
    planner_prediction = PlannerPrediction(
        pred_p_success_T6=float(selected_stats.get("success_prob", 0.0)),
        pred_tau_mean=selected_stats.get("mean_first_correct_attempt"),
        pred_tau_mode=_tau_mode(selected_stats),
        pred_p_tau_1_to_6=[float(x) for x in selected_stats.get("pred_p_tau_1_to_6", [])],
        pred_p_tau_band=float(band_success_prob(selected_stats, cfg)),
        pred_p_tau_early=float(early_success_prob(selected_stats, cfg)),
        pred_attempt_correct_prob_mean=list(selected_stats.get("pred_attempt_correct_prob_mean", [])),
        pred_attempt_correct_rank_mean=list(selected_stats.get("pred_attempt_correct_rank_mean", [])),
        pred_correct_prob_no_hint_mean=no_hint_stats.get("initial_correct_prob_mean"),
        pred_correct_prob_after_hint_mean=selected_stats.get("initial_correct_prob_mean"),
        pred_correct_rank_no_hint_mean=no_hint_stats.get("initial_correct_rank_mean"),
        pred_correct_rank_after_hint_mean=selected_stats.get("initial_correct_rank_mean"),
        abstained=selected_hint is None,
        abstain_reason=abstain_reason,
        hint_quality_tags=hint_quality_tags,
        kept_profiles=_profiles_payload(kept_profiles),
    )

    return HintPlanResult(
        selected_hint=selected_hint,
        selected_utility=selected_utility if selected_hint is not None else float(no_hint_utility),
        no_hint_utility=float(no_hint_utility),
        delta_vs_no_hint=float(delta),
        candidate_scores=scores,
        planner_prediction=planner_prediction,
        planner_counters=counters,
        stage_scores=stage_scores,
    )


def select_hint(
    posterior,
    context: TaskContext,
    teach_case: TeachCase,
    eval_items,
    cfg,
    rng: np.random.Generator,
) -> HintPlanResult:
    if str(getattr(cfg, "planner_mode", "cascade")) != "cascade":
        return _single_stage_select_hint(posterior, context, teach_case, eval_items, cfg, rng)

    counters = PlannerCounters() if getattr(cfg, "collect_cost_counters", False) else None
    t_select = time.perf_counter()
    stage_scores: List[PlannerStageScore] = []

    candidates = build_hint_candidates(context, teach_case, cfg, rng)
    all_candidates: List[Optional[HintCandidate]] = [None, *candidates]
    candidate_cache = build_score_tables_for_candidates(
        posterior=posterior,
        candidates=all_candidates,
        teach_case=teach_case,
        cfg=cfg,
        counters=counters,
    )

    prefilter_profiles = posterior.profiles_for_stage("prefilter", cfg)
    refine_profiles = posterior.profiles_for_stage("refine", cfg)

    prefilter_records: Dict[Tuple[object, ...], Dict[str, object]] = {}
    stage0_start = time.perf_counter()
    for hint in all_candidates:
        sig = candidate_signature(hint)
        stats = prefilter_score_table_under_profiles(candidate_cache[sig], prefilter_profiles, cfg)
        prefilter_records[sig] = _record_for_hint(
            hint=hint,
            stats=stats,
            utility=float(stats.get("prefilter_score", 0.0)),
            stage="prefilter",
            cfg=cfg,
        )
        stage_scores.append(
            _stage_score("prefilter", hint, float(stats.get("prefilter_score", 0.0)))
        )
    if counters is not None:
        counters.stage0_wall_time = float(time.perf_counter() - stage0_start)

    utility_mode = str(getattr(cfg, "utility_mode", "legacy"))
    ranked_prefilter = sorted(
        [record for sig, record in prefilter_records.items() if sig != candidate_signature(None)],
        key=lambda item: float(item["utility"]),
        reverse=True,
    )
    ranked_prefilter = filter_candidate_records_for_stage(ranked_prefilter, cfg, stage="prefilter")
    if getattr(cfg, "prefilter_enabled", True):
        if not (utility_mode in {"advantage_delta", "advantage_transfer", "advantage_mix"} and bool(getattr(cfg, "eval_aware", False))):
            ranked_prefilter = _family_aware_prefilter(ranked_prefilter, cfg)
        ranked_prefilter = _objective_bucketed_records(ranked_prefilter, cfg, stage="prefilter")
    if counters is not None:
        counters.n_candidates_prefiltered = len(ranked_prefilter)

    no_hint_sig = candidate_signature(None)
    proxy_candidates = [None] + [record["hint"] for record in ranked_prefilter]
    proxy_first_reveal_tables = build_first_reveal_tables_for_candidates(
        posterior=posterior,
        candidates=proxy_candidates,
        candidate_cache=candidate_cache,
        teach_case=teach_case,
        profile_weights=prefilter_profiles,
        cfg=cfg,
        counters=counters,
    )
    proxy_records: Dict[Tuple[object, ...], Dict[str, object]] = {}
    stage1_start = time.perf_counter()
    bonus_cfg = _bonus_cfg(cfg) if utility_mode in {"delta_vs_no_tutor_bonus", "advantage_delta", "advantage_fast_success", "advantage_transfer", "advantage_mix"} else None
    proxy_bonus_baseline_stats = None
    if bonus_cfg is not None:
        proxy_bonus_baseline_stats = evaluate_score_table_under_profiles(
            score_table=candidate_cache[no_hint_sig],
            profile_weights=prefilter_profiles,
            cfg=bonus_cfg,
            seed=int(rng.integers(0, 2**31 - 1)),
            stage="proxy",
            first_reveal_tables=proxy_first_reveal_tables.get(no_hint_sig),
            counters=counters,
        )
    for idx, hint in enumerate(proxy_candidates):
        sig = candidate_signature(hint)
        stats = evaluate_score_table_under_profiles(
            score_table=candidate_cache[sig],
            profile_weights=prefilter_profiles,
            cfg=cfg,
            seed=int(rng.integers(0, 2**31 - 1)) + idx,
            stage="proxy",
            first_reveal_tables=proxy_first_reveal_tables.get(sig),
            counters=counters,
        )
        if hint is None and utility_mode in {"delta_vs_no_tutor_bonus", "advantage_delta", "advantage_fast_success", "advantage_transfer", "advantage_mix"}:
            utility = 0.0
        else:
            utility = compute_hint_utility(
                stats,
                cfg,
                no_hint_stats=proxy_records.get(no_hint_sig, {}).get("stats"),
                bonus_baseline_stats=proxy_bonus_baseline_stats,
            )
        proxy_records[sig] = _record_for_hint(hint=hint, stats=stats, utility=utility, stage="proxy", cfg=cfg)
        stage_scores.append(_stage_score("proxy", hint, utility))
    if _transfer_eval_proxy_enabled(cfg):
        _attach_transfer_eval_proxy(
            posterior=posterior,
            records=proxy_records,
            teach_case=teach_case,
            eval_items=eval_items,
            cfg=cfg,
            rng=rng,
            stage="proxy",
            counters=counters,
        )
        proxy_bonus_baseline_stats = _update_bonus_eval_fields(
            proxy_bonus_baseline_stats,
            proxy_records[no_hint_sig]["stats"],
        )
        _recompute_record_utilities(
            proxy_records,
            cfg,
            no_hint_sig=no_hint_sig,
            bonus_baseline_stats=proxy_bonus_baseline_stats,
        )
    if counters is not None:
        counters.n_candidates_proxy_evaluated = max(0, len(proxy_candidates) - 1)
        counters.stage1_wall_time = float(time.perf_counter() - stage1_start)

    final_records: Dict[Tuple[object, ...], Dict[str, object]] = dict(proxy_records)

    ranked_proxy = sorted(
        [record for sig, record in proxy_records.items() if sig != no_hint_sig],
        key=lambda item: float(item["utility"]),
        reverse=True,
    )
    ranked_proxy = filter_candidate_records_for_stage(ranked_proxy, cfg, stage="proxy")
    ranked_proxy = _objective_bucketed_records(ranked_proxy, cfg, stage="refine")
    refine_pool = ranked_proxy[: max(1, int(getattr(cfg, "refine_top_k", len(ranked_proxy))))]

    if getattr(cfg, "refine_enabled", False) and refine_pool:
        stage2_start = time.perf_counter()
        refine_candidates = [None] + [record["hint"] for record in refine_pool]
        refined_records: Dict[Tuple[object, ...], Dict[str, object]] = {}
        refine_mode = str(getattr(cfg, "refine_update_mode", "proxy"))
        refine_bonus_baseline_stats = None
        if bonus_cfg is not None:
            if refine_mode in {"full_cls", "lazy_cls"}:
                refine_bonus_stats_cfg = copy.deepcopy(bonus_cfg)
                refine_bonus_stats_cfg.rollout_mode = "mc"
                refine_bonus_stats_cfg.n_rollouts = int(getattr(cfg, "refine_n_rollouts", cfg.n_rollouts))
                refine_bonus_stats_cfg.beam_top_b = int(getattr(cfg, "refine_beam_top_b", cfg.beam_top_b))
                refine_bonus_stats_cfg.beam_keep_l = int(getattr(cfg, "refine_beam_keep_l", cfg.beam_keep_l))
                refine_bonus_stats_cfg.planning_update_mode = refine_mode
                refine_bonus_stats_cfg.profile_top_mass = float(getattr(cfg, "refine_profile_top_mass", cfg.profile_top_mass))
                refine_bonus_stats_cfg.profile_min_keep = int(getattr(cfg, "refine_profile_min_keep", cfg.profile_min_keep))
                refine_bonus_baseline_stats = evaluate_hint_under_posterior(
                    posterior=posterior,
                    hint=None,
                    teach_case=teach_case,
                    eval_items=eval_items,
                    cfg=refine_bonus_stats_cfg,
                    seed=int(rng.integers(0, 2**31 - 1)),
                    counters=counters,
                )
            else:
                refine_bonus_baseline_stats = evaluate_score_table_under_profiles(
                    score_table=candidate_cache[no_hint_sig],
                    profile_weights=refine_profiles,
                    cfg=bonus_cfg,
                    seed=int(rng.integers(0, 2**31 - 1)),
                    stage="refine",
                    counters=counters,
                )
        if refine_mode in {"full_cls", "lazy_cls"}:
            refine_cfg = copy.deepcopy(cfg)
            refine_cfg.rollout_mode = "mc"
            refine_cfg.n_rollouts = int(getattr(cfg, "refine_n_rollouts", cfg.n_rollouts))
            refine_cfg.beam_top_b = int(getattr(cfg, "refine_beam_top_b", cfg.beam_top_b))
            refine_cfg.beam_keep_l = int(getattr(cfg, "refine_beam_keep_l", cfg.beam_keep_l))
            refine_cfg.planning_update_mode = refine_mode
            refine_cfg.profile_top_mass = float(getattr(cfg, "refine_profile_top_mass", cfg.profile_top_mass))
            refine_cfg.profile_min_keep = int(getattr(cfg, "refine_profile_min_keep", cfg.profile_min_keep))
            for idx, hint in enumerate(refine_candidates):
                stats = evaluate_hint_under_posterior(
                    posterior=posterior,
                    hint=hint,
                    teach_case=teach_case,
                    eval_items=eval_items,
                    cfg=refine_cfg,
                    seed=int(rng.integers(0, 2**31 - 1)) + idx,
                    counters=counters,
                )
                if hint is None and utility_mode in {"delta_vs_no_tutor_bonus", "advantage_delta", "advantage_fast_success", "advantage_transfer", "advantage_mix"}:
                    utility = 0.0
                else:
                    utility = compute_hint_utility(
                        stats,
                        cfg,
                        no_hint_stats=final_records.get(no_hint_sig, proxy_records[no_hint_sig])["stats"],
                        bonus_baseline_stats=refine_bonus_baseline_stats,
                    )
                sig = candidate_signature(hint)
                refined_records[sig] = _record_for_hint(
                    hint=hint,
                    stats=stats,
                    utility=utility,
                    stage="refine",
                    cfg=cfg,
                )
                stage_scores.append(_stage_score("refine", hint, utility))
        else:
            refine_first_reveal_tables = build_first_reveal_tables_for_candidates(
                posterior=posterior,
                candidates=refine_candidates,
                candidate_cache=candidate_cache,
                teach_case=teach_case,
                profile_weights=refine_profiles,
                cfg=cfg,
                counters=counters,
            ) if refine_mode == "first_reveal_cached_cls" else {}
            for idx, hint in enumerate(refine_candidates):
                sig = candidate_signature(hint)
                stats = evaluate_score_table_under_profiles(
                    score_table=candidate_cache[sig],
                    profile_weights=refine_profiles,
                    cfg=cfg,
                    seed=int(rng.integers(0, 2**31 - 1)) + idx,
                    stage="refine",
                    first_reveal_tables=refine_first_reveal_tables.get(sig) if refine_mode == "first_reveal_cached_cls" else None,
                    counters=counters,
                )
                if hint is None and utility_mode in {"delta_vs_no_tutor_bonus", "advantage_delta", "advantage_fast_success", "advantage_transfer", "advantage_mix"}:
                    utility = 0.0
                else:
                    utility = compute_hint_utility(
                        stats,
                        cfg,
                        no_hint_stats=final_records.get(no_hint_sig, proxy_records[no_hint_sig])["stats"],
                        bonus_baseline_stats=refine_bonus_baseline_stats,
                    )
                refined_records[sig] = _record_for_hint(
                    hint=hint,
                    stats=stats,
                    utility=utility,
                    stage="refine",
                    cfg=cfg,
                )
                stage_scores.append(_stage_score("refine", hint, utility))
        if _transfer_eval_proxy_enabled(cfg):
            _attach_transfer_eval_proxy(
                posterior=posterior,
                records=refined_records,
                teach_case=teach_case,
                eval_items=eval_items,
                cfg=cfg,
                rng=rng,
                stage="refine",
                counters=counters,
            )
            refine_bonus_baseline_stats = _update_bonus_eval_fields(
                refine_bonus_baseline_stats,
                refined_records.get(no_hint_sig, proxy_records[no_hint_sig])["stats"],
            )
            _recompute_record_utilities(
                refined_records,
                cfg,
                no_hint_sig=no_hint_sig,
                bonus_baseline_stats=refine_bonus_baseline_stats,
            )
        final_records.update(refined_records)
        if counters is not None:
            counters.n_candidates_refined = max(0, len(refine_candidates) - 1)
            counters.stage2_wall_time = float(time.perf_counter() - stage2_start)

    no_hint_record = final_records[no_hint_sig]
    reference_stats = no_hint_record["stats"]
    if utility_mode in {"delta_vs_no_tutor_bonus", "advantage_delta", "advantage_fast_success", "advantage_transfer", "advantage_mix"}:
        if "refined_records" in locals():
            reference_stats = refine_bonus_baseline_stats if 'refine_bonus_baseline_stats' in locals() and refine_bonus_baseline_stats is not None else proxy_bonus_baseline_stats
        else:
            reference_stats = proxy_bonus_baseline_stats if proxy_bonus_baseline_stats is not None else no_hint_record["stats"]
    candidate_records = filter_candidate_records_for_stage(
        [record for sig, record in final_records.items() if sig != no_hint_sig],
        cfg,
        stage="final",
    )
    candidate_records = _apply_normalized_mix_selection_scores(
        candidate_records,
        no_hint_record,
        cfg,
        reference_stats=reference_stats,
    )
    candidate_records = apply_reranker(candidate_records, context, teach_case, cfg)
    selected_record = _choose_best_candidate(no_hint_record, candidate_records, cfg, reference_stats=reference_stats)
    selected_hint = selected_record["hint"]
    selected_stats = selected_record["stats"]
    selected_utility = (
        float(selected_record["utility"]) if selected_hint is not None else float(no_hint_record["utility"])
    )
    delta = max(0.0, selected_utility - float(no_hint_record["utility"])) if selected_hint is not None else 0.0
    abstain_reason = _infer_abstain_reason(
        no_hint_record,
        candidate_records,
        cfg,
        reference_stats=reference_stats,
    ) if selected_hint is None else None
    hint_quality_tags = _selected_hint_quality_tags(context, teach_case, selected_hint, selected_stats, cfg)

    candidate_scores = sorted(
        candidate_records,
        key=lambda item: float(item.get("selection_score", item["utility"])),
        reverse=True,
    )
    if getattr(cfg, "plan_candidate_limit", None):
        candidate_scores = candidate_scores[: max(1, int(cfg.plan_candidate_limit))]

    if counters is not None:
        counters.select_hint_wall_time = float(time.perf_counter() - t_select)

    planner_prediction = PlannerPrediction(
        pred_p_success_T6=float(selected_stats.get("success_prob", 0.0)),
        pred_tau_mean=selected_stats.get("mean_first_correct_attempt"),
        pred_tau_mode=_tau_mode(selected_stats),
        pred_p_tau_1_to_6=[float(x) for x in selected_stats.get("pred_p_tau_1_to_6", [])],
        pred_p_tau_band=float(band_success_prob(selected_stats, cfg)),
        pred_p_tau_early=float(early_success_prob(selected_stats, cfg)),
        pred_attempt_correct_prob_mean=list(selected_stats.get("pred_attempt_correct_prob_mean", [])),
        pred_attempt_correct_rank_mean=list(selected_stats.get("pred_attempt_correct_rank_mean", [])),
        pred_correct_prob_no_hint_mean=no_hint_record["stats"].get("initial_correct_prob_mean"),
        pred_correct_prob_after_hint_mean=selected_stats.get("initial_correct_prob_mean"),
        pred_correct_rank_no_hint_mean=no_hint_record["stats"].get("initial_correct_rank_mean"),
        pred_correct_rank_after_hint_mean=selected_stats.get("initial_correct_rank_mean"),
        abstained=selected_hint is None,
        abstain_reason=abstain_reason,
        hint_quality_tags=hint_quality_tags,
        kept_profiles=_profiles_payload(
            refine_profiles if selected_record.get("stage") == "refine" else prefilter_profiles
        ),
    )

    return HintPlanResult(
        selected_hint=selected_hint,
        selected_utility=float(selected_utility),
        no_hint_utility=float(no_hint_record["utility"]),
        delta_vs_no_hint=float(delta),
        candidate_scores=candidate_scores,
        planner_prediction=planner_prediction,
        planner_counters=counters,
        stage_scores=stage_scores,
    )
