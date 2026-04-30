from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .metrics import soft_tau_score


def candidate_feature_map(
    record: dict,
    context,
    teach_case,
    cfg,
) -> Dict[str, float]:
    hint = record.get("hint")
    metadata = dict(record.get("metadata", {}) or {})
    words = [] if hint is None else list(getattr(hint.example, "words", []) or [])
    family = str(metadata.get("family", record.get("kind", "none") or "none"))
    difficulty = str(record.get("difficulty", "none") or "none")
    operator_names = {spec.name for spec in getattr(context, "operator_specs", [])}
    teach_words = list(getattr(teach_case.example, "words", []) or [])
    teach_ops = {tok for tok in teach_words if tok in operator_names}
    hint_ops = {tok for tok in words if tok in operator_names}
    transfer_proxy = float(record.get("eval_cell_acc", 0.0)) if bool(getattr(cfg, "eval_aware", False)) else 0.0

    features: Dict[str, float] = {
        "base_utility": float(record.get("utility", 0.0)),
        "pred_success_prob": float(record.get("success_prob", 0.0)),
        "pred_band_prob": float(record.get("band_success_prob", 0.0)),
        "pred_early_prob": float(record.get("early_success_prob", 0.0)),
        "pred_eval_exact": float(record.get("eval_exact_acc", 0.0)),
        "pred_eval_cell": float(record.get("eval_cell_acc", 0.0)),
        "pred_soft_tau": float(soft_tau_score(record.get("stats", {}), cfg)) if record.get("stats") is not None else 0.0,
        "pred_mean_tau": 0.0 if record.get("mean_first_correct_attempt") is None else float(record.get("mean_first_correct_attempt")),
        "pred_initial_correct_prob": 0.0 if record.get("initial_correct_prob_mean") is None else float(record.get("initial_correct_prob_mean")),
        "pred_initial_correct_rank": 0.0 if record.get("initial_correct_rank_mean") is None else float(record.get("initial_correct_rank_mean")),
        "pred_safe_wrong_mean": float(record.get("safe_wrong_mean", 0.0)),
        "pred_conservative_reveal_penalty": float(record.get("conservative_reveal_penalty", 0.0)),
        "pred_collapse_adjustment_mean": float(record.get("collapse_adjustment_mean", 0.0)),
        "pred_first_reveal_cache_hit_prob": float(record.get("first_reveal_cache_hit_prob", 0.0)),
        "transfer_proxy": transfer_proxy,
        "word_count": float(len(words)),
        "operator_overlap": float(metadata.get("operator_overlap", 0.0)),
        "atom_overlap": float(metadata.get("atom_overlap", 0.0)),
        "quality_score": float(metadata.get("quality_score", 0.0)),
        "teach_operator_overlap": float(len(teach_ops & hint_ops)),
        "is_operator_relevant": 1.0 if (hint_ops & teach_ops) else 0.0,
        "is_menu_based": 1.0 if family.startswith("menu_") else 0.0,
        "is_target_family": 1.0 if family.startswith("target_neighborhood") else 0.0,
        "is_operator_probe": 1.0 if family == "operator_probe" else 0.0,
        "is_free": 1.0 if family == "free" else 0.0,
    }
    features[f"family::{family}"] = 1.0
    features[f"difficulty::{difficulty}"] = 1.0
    return features


@lru_cache(maxsize=16)
def load_reranker_model(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid reranker model at {path}: expected JSON object")
    return payload


def _linear_score(features: Dict[str, float], model: dict) -> float:
    weights = dict(model.get("weights", {}) or {})
    means = dict(model.get("feature_means", {}) or {})
    scales = dict(model.get("feature_scales", {}) or {})
    bias = float(model.get("bias", 0.0))
    score = bias
    for name, value in features.items():
        weight = float(weights.get(name, 0.0))
        if weight == 0.0:
            continue
        centered = float(value) - float(means.get(name, 0.0))
        scale = float(scales.get(name, 1.0))
        normalized = centered / scale if abs(scale) > 1e-12 else centered
        score += weight * normalized
    return float(score)


def apply_reranker(
    records: Sequence[dict],
    context,
    teach_case,
    cfg,
) -> List[dict]:
    if not bool(getattr(cfg, "reranker_enabled", False)):
        return [dict(record, selection_score=float(record.get("utility", 0.0)), reranker_score=0.0) for record in records]
    model_path = str(getattr(cfg, "reranker_model_path", "") or "").strip()
    if not model_path:
        raise ValueError("reranker_enabled=True but reranker_model_path is empty")
    model = load_reranker_model(model_path)
    alpha = float(getattr(cfg, "reranker_alpha", model.get("alpha", 1.0)))
    scored: List[dict] = []
    for record in records:
        features = candidate_feature_map(record, context, teach_case, cfg)
        reranker_score = _linear_score(features, model)
        base = float(record.get("utility", 0.0))
        scored.append(
            dict(
                record,
                reranker_target=str(model.get("target", getattr(cfg, "reranker_target", "search"))),
                reranker_score=float(reranker_score),
                selection_score=float(base + alpha * reranker_score),
                reranker_features=features,
            )
        )
    return scored


def collect_feature_names(rows: Iterable[dict]) -> List[str]:
    names = set()
    for row in rows:
        for name in dict(row.get("reranker_features", {}) or {}):
            names.add(str(name))
    return sorted(names)
