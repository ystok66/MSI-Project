from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..interfaces import Example
from .interfaces import EvalItem, HintCandidate, PlannerCounters, TeachCase
from .metrics import attempt_time_reward


@dataclass
class BeamState:
    prob: float
    shadow: object
    active_menu: list
    attempts: int = 0
    success: bool = False
    first_correct_attempt: Optional[int] = None
    safe_wrong_count: int = 0
    risky_wrong_count: int = 0
    risk_count: int = 0
    damage_sum: int = 0


@dataclass(frozen=True)
class PlannerMenuMeta:
    option_indices: Tuple[int, ...]
    correct_index: Optional[int]
    correct_pos: Optional[int]
    text_ids: Tuple[int, ...]
    output_ids: Tuple[int, ...]
    risk_classes: Tuple[int, ...]


@dataclass
class CandidateScoreTable:
    hint: Optional[HintCandidate]
    menu_meta: PlannerMenuMeta
    sem: np.ndarray
    mu_d: np.ndarray
    u_d: np.ndarray


@dataclass(frozen=True)
class DeltaPath:
    active_mask: int
    revealed_text_ids: frozenset[int]
    revealed_output_ids: frozenset[int]
    prev_correct_prob: Optional[float] = None
    current_table_key: Optional[int] = None
    used_first_reveal_cache: bool = False
    attempts: int = 0
    success: bool = False
    first_correct_attempt: Optional[int] = None
    safe_wrong_count: int = 0
    risky_wrong_count: int = 0
    risk_count: int = 0
    damage_sum: int = 0
    logprob: float = 0.0
    picks: Tuple[int, ...] = ()


def candidate_signature(hint: Optional[HintCandidate]) -> Tuple[object, ...]:
    if hint is None:
        return ("none",)
    return (
        hint.kind,
        hint.difficulty,
        hint.source_index,
        tuple(hint.example.words),
        tuple(hint.example.output),
    )


def _option_key(values: Sequence[str]) -> Tuple[str, ...]:
    return tuple(values)


def _build_menu_meta(teach_case: TeachCase) -> PlannerMenuMeta:
    text_lookup: Dict[Tuple[str, ...], int] = {}
    output_lookup: Dict[Tuple[str, ...], int] = {}
    text_ids: List[int] = []
    output_ids: List[int] = []
    option_indices: List[int] = []
    risk_classes: List[int] = []
    correct_pos = None
    correct_index = None

    for pos, opt in enumerate(teach_case.menu):
        tkey = _option_key(opt.text)
        okey = _option_key(opt.rendered_output or [])
        if tkey not in text_lookup:
            text_lookup[tkey] = len(text_lookup)
        if okey not in output_lookup:
            output_lookup[okey] = len(output_lookup)
        text_ids.append(text_lookup[tkey])
        output_ids.append(output_lookup[okey])
        option_indices.append(int(opt.index))
        risk_classes.append(int(getattr(opt, "risk_class", 0) or 0))
        if opt.is_correct:
            correct_pos = pos
            correct_index = int(opt.index)

    return PlannerMenuMeta(
        option_indices=tuple(option_indices),
        correct_index=correct_index,
        correct_pos=correct_pos,
        text_ids=tuple(text_ids),
        output_ids=tuple(output_ids),
        risk_classes=tuple(risk_classes),
    )


def _apply_hint_to_shadow(shadow, hint: Optional[HintCandidate]) -> None:
    if hint is None:
        return
    scorer = getattr(shadow, "scorer", None)
    if scorer is None or not hasattr(scorer, "incremental_study"):
        return
    scorer.incremental_study(
        [Example(words=list(hint.example.words), output=list(hint.example.output))]
    )


def _apply_correct_update(shadow, option, target_output) -> None:
    scorer = getattr(shadow, "scorer", None)
    if scorer is None or not hasattr(scorer, "incremental_study"):
        return
    scorer.incremental_study(
        [Example(words=list(option.text), output=list(target_output))],
        n_em_override=1,
    )


def _apply_planning_correct_update(shadow, option, target_output, cfg) -> None:
    if getattr(cfg, "planning_update_mode", "proxy") == "full_cls":
        _apply_correct_update(shadow, option, target_output)


def _apply_planning_wrong_update(shadow, option, target_output, cfg) -> None:
    scorer = getattr(shadow, "scorer", None)
    damage = int(getattr(option, "risk_class", 0))
    if getattr(cfg, "planning_update_mode", "proxy") == "proxy":
        if scorer is not None and hasattr(scorer, "add_negative_evidence"):
            scorer.add_negative_evidence(list(option.text), list(target_output), weight=1.0)
        shadow.update_from_reveal(
            wrong_text=list(option.text),
            revealed_output=list(option.rendered_output or []),
            danger_vec=np.asarray(option.danger_vec),
            damage=damage,
            update_semantic=False,
            update_risk=damage > 0,
        )
        return
    shadow.update_from_reveal(
        wrong_text=list(option.text),
        revealed_output=list(option.rendered_output or []),
        danger_vec=np.asarray(option.danger_vec),
        damage=damage,
        update_semantic=True,
        update_risk=damage > 0,
    )


def _apply_lazy_prefix_wrong_updates(
    shadow,
    wrong_prefix: Tuple[int, ...],
    option_by_index: Dict[int, object],
    target_output: List[str],
    cfg,
) -> None:
    if not wrong_prefix:
        return
    proxy_cfg = copy.copy(cfg)
    proxy_cfg.planning_update_mode = "proxy"
    for idx in wrong_prefix:
        option = option_by_index.get(int(idx))
        if option is None:
            continue
        _apply_planning_wrong_update(
            shadow=shadow,
            option=option,
            target_output=target_output,
            cfg=proxy_cfg,
        )


def _apply_first_reveal_semantic_update(shadow, option) -> None:
    damage = int(getattr(option, "risk_class", 0))
    shadow.update_from_reveal(
        wrong_text=list(option.text),
        revealed_output=list(option.rendered_output or []),
        danger_vec=np.asarray(option.danger_vec),
        damage=damage,
        update_semantic=True,
        update_risk=damage > 0,
    )


def _shadow_eval_metrics(shadow, eval_items: Iterable[EvalItem]) -> Tuple[float, float]:
    scorer = getattr(shadow, "scorer", None)
    items = list(eval_items)
    if scorer is None or not items:
        return 0.0, 0.0
    exact = 0
    total_cells = 0
    correct_cells = 0
    for item in items:
        pred = scorer.predict_output(list(item.words))
        if pred == list(item.output):
            exact += 1
        total_cells += max(len(item.output), len(pred))
        overlap = min(len(item.output), len(pred))
        correct_cells += sum(1 for i in range(overlap) if pred[i] == item.output[i])
    return exact / max(len(items), 1), correct_cells / max(total_cells, 1)


def eval_proxy_subset_items(eval_items: Iterable[EvalItem], cfg) -> List[EvalItem]:
    items = list(eval_items)
    if not items:
        return []
    per_diff = max(1, int(getattr(cfg, "transfer_eval_proxy_n_per_diff", 3)))
    max_items = max(1, int(getattr(cfg, "transfer_eval_proxy_max_items", per_diff * 3)))
    grouped: Dict[str, List[EvalItem]] = {}
    for item in items:
        grouped.setdefault(str(getattr(item, "difficulty", "unknown")), []).append(item)

    selected: List[EvalItem] = []
    for difficulty in sorted(grouped):
        selected.extend(grouped[difficulty][:per_diff])
        if len(selected) >= max_items:
            return selected[:max_items]
    if len(selected) < max_items:
        seen = {id(item) for item in selected}
        for item in items:
            if id(item) in seen:
                continue
            selected.append(item)
            if len(selected) >= max_items:
                break
    return selected[:max_items]


def evaluate_static_hint_eval_proxies(
    posterior,
    hints: Sequence[Optional[HintCandidate]],
    eval_items: Iterable[EvalItem],
    cfg,
    counters: Optional[PlannerCounters] = None,
) -> Dict[Tuple[object, ...], dict]:
    subset = eval_proxy_subset_items(eval_items, cfg)
    out: Dict[Tuple[object, ...], dict] = {}
    base_shadow = posterior.cloned_shadow()
    if counters is not None:
        counters.n_cls_deepcopy_calls += 1
    for hint in hints:
        sig = candidate_signature(hint)
        model = base_shadow.deep_copy()
        if counters is not None:
            counters.n_cls_deepcopy_calls += 1
        _apply_hint_to_shadow(model, hint)
        exact_acc, cell_acc = _shadow_eval_metrics(model, subset)
        _aggregate_scorer_debug(getattr(model, "scorer", None), counters)
        out[sig] = {
            "eval_exact_acc": float(exact_acc),
            "eval_cell_acc": float(cell_acc),
            "eval_proxy_exact_acc": float(exact_acc),
            "eval_proxy_cell_acc": float(cell_acc),
            "eval_proxy_n_items": int(len(subset)),
            "eval_proxy_mode": "static_subset",
        }
    return out


def _aggregate_scorer_debug(scorer, counters: Optional[PlannerCounters]) -> None:
    if counters is None or scorer is None or not hasattr(scorer, "debug_counters"):
        return
    stats = scorer.debug_counters()
    counters.n_cls_predict_calls += int(stats.get("predict_calls", 0))
    counters.n_cls_score_calls += int(stats.get("score_calls", 0))
    counters.n_incremental_study_calls += int(stats.get("incremental_study_calls", 0))


def _initial_hint_stats_shadow(
    shadow,
    profile,
    hint: Optional[HintCandidate],
    teach_case: TeachCase,
    counters: Optional[PlannerCounters] = None,
) -> dict:
    model = shadow.deep_copy()
    if counters is not None:
        counters.n_cls_deepcopy_calls += 1
    scorer = getattr(model, "scorer", None)
    if scorer is not None and hasattr(scorer, "reset_debug_counters"):
        scorer.reset_debug_counters()
    _apply_hint_to_shadow(model, hint)
    probs = model.predict_pick_probs(
        target_output=list(teach_case.example.output),
        option_texts=[list(opt.text) for opt in teach_case.menu],
        option_danger_vecs=[np.asarray(opt.danger_vec) for opt in teach_case.menu],
        profile=profile,
        spec={"action": "WAIT"},
        banned_indices=set(),
        highlighted_cells=(),
        option_indices=[opt.index for opt in teach_case.menu],
    )
    _aggregate_scorer_debug(getattr(model, "scorer", None), counters)
    correct_pos = next((idx for idx, opt in enumerate(teach_case.menu) if opt.is_correct), None)
    correct_prob, correct_rank, correct_margin = _correct_prob_rank_margin_from_probs(
        np.asarray(probs, dtype=float),
        correct_pos,
    )
    return {
        "initial_correct_prob_mean": float(correct_prob),
        "initial_correct_rank_mean": None if correct_rank is None else float(correct_rank),
        "initial_correct_margin_mean": float(correct_margin),
    }


def _score_table_from_shadow(
    shadow,
    teach_case: TeachCase,
    menu_meta: PlannerMenuMeta,
    counters: Optional[PlannerCounters] = None,
) -> CandidateScoreTable:
    scorer = getattr(shadow, "scorer", None)
    if scorer is not None and hasattr(scorer, "reset_debug_counters"):
        scorer.reset_debug_counters()
    components = shadow.score_option_components(
        target_output=list(teach_case.example.output),
        option_texts=[list(opt.text) for opt in teach_case.menu],
        option_danger_vecs=[np.asarray(opt.danger_vec) for opt in teach_case.menu],
        spec={"action": "WAIT"},
        banned_indices=set(),
        highlighted_cells=(),
        option_indices=list(menu_meta.option_indices),
    )
    _aggregate_scorer_debug(getattr(shadow, "scorer", None), counters)
    return CandidateScoreTable(
        hint=None,
        menu_meta=menu_meta,
        sem=np.asarray(components["sem"], dtype=float),
        mu_d=np.asarray(components["mu_d"], dtype=float),
        u_d=np.asarray(components["u_d"], dtype=float),
    )


def build_candidate_score_table(
    base_shadow,
    hint: Optional[HintCandidate],
    teach_case: TeachCase,
    cfg,
    menu_meta: Optional[PlannerMenuMeta] = None,
    counters: Optional[PlannerCounters] = None,
) -> CandidateScoreTable:
    menu_meta = menu_meta or _build_menu_meta(teach_case)
    shadow = base_shadow.deep_copy()
    if counters is not None:
        counters.n_cls_deepcopy_calls += 1
    scorer = getattr(shadow, "scorer", None)
    if scorer is not None and hasattr(scorer, "reset_debug_counters"):
        scorer.reset_debug_counters()
    _apply_hint_to_shadow(shadow, hint)
    table = _score_table_from_shadow(
        shadow=shadow,
        teach_case=teach_case,
        menu_meta=menu_meta,
        counters=counters,
    )
    table.hint = hint
    return table


def build_score_tables_for_candidates(
    posterior,
    candidates: Sequence[Optional[HintCandidate]],
    teach_case: TeachCase,
    cfg,
    counters: Optional[PlannerCounters] = None,
) -> Dict[Tuple[object, ...], CandidateScoreTable]:
    cache: Dict[Tuple[object, ...], CandidateScoreTable] = {}
    base_shadow = posterior.cloned_shadow()
    if counters is not None:
        counters.n_cls_deepcopy_calls += 1
    menu_meta = _build_menu_meta(teach_case)

    for hint in candidates:
        sig = candidate_signature(hint)
        if getattr(cfg, "cache_hint_profile_scores", True) and sig in cache:
            if counters is not None:
                counters.n_score_table_hits += 1
            continue
        if counters is not None:
            counters.n_score_table_misses += 1
        cache[sig] = build_candidate_score_table(
            base_shadow=base_shadow,
            hint=hint,
            teach_case=teach_case,
            cfg=cfg,
            menu_meta=menu_meta,
            counters=counters,
        )
    return cache


def _first_reveal_weighted_wrong_probs(
    score_table: CandidateScoreTable,
    profile_weights: Sequence[Tuple[object, float]],
    cfg,
) -> np.ndarray:
    probs = np.zeros(len(score_table.sem), dtype=float)
    active_mask = _full_active_mask(len(score_table.sem))
    for profile, weight in profile_weights:
        if float(weight) <= 0.0:
            continue
        cur = _score_table_probs(
            score_table=score_table,
            profile=profile,
            active_mask=active_mask,
            revealed_text_ids=frozenset(),
            revealed_output_ids=frozenset(),
            cfg=cfg,
        )
        probs += float(weight) * np.asarray(cur, dtype=float)
    if score_table.menu_meta.correct_pos is not None:
        probs[int(score_table.menu_meta.correct_pos)] = 0.0
    return probs


def build_first_reveal_tables_for_candidates(
    posterior,
    candidates: Sequence[Optional[HintCandidate]],
    candidate_cache: Dict[Tuple[object, ...], CandidateScoreTable],
    teach_case: TeachCase,
    profile_weights: Sequence[Tuple[object, float]],
    cfg,
    counters: Optional[PlannerCounters] = None,
) -> Dict[Tuple[object, ...], Dict[int, CandidateScoreTable]]:
    if not getattr(cfg, "first_reveal_refine_enabled", True):
        return {}

    base_shadow = posterior.cloned_shadow()
    if counters is not None:
        counters.n_cls_deepcopy_calls += 1
    menu_meta = _build_menu_meta(teach_case)
    first_reveal_cache: Dict[Tuple[object, ...], Dict[int, CandidateScoreTable]] = {}
    top_b = max(0, int(getattr(cfg, "first_reveal_top_b", 0)))
    if top_b <= 0:
        return first_reveal_cache

    for hint in candidates:
        sig = candidate_signature(hint)
        score_table = candidate_cache[sig]
        weighted_wrong = _first_reveal_weighted_wrong_probs(score_table, profile_weights, cfg)
        ranked_pos = [int(pos) for pos in np.argsort(-weighted_wrong) if float(weighted_wrong[pos]) > 0.0]
        selected_pos = ranked_pos[:top_b]
        if not selected_pos:
            continue

        hint_model = base_shadow.deep_copy()
        if counters is not None:
            counters.n_cls_deepcopy_calls += 1
        _apply_hint_to_shadow(hint_model, hint)
        _aggregate_scorer_debug(getattr(hint_model, "scorer", None), counters)

        branch_tables: Dict[int, CandidateScoreTable] = {}
        for pos in selected_pos:
            branch_model = hint_model.deep_copy()
            if counters is not None:
                counters.n_cls_deepcopy_calls += 1
            option = teach_case.menu[pos]
            _apply_first_reveal_semantic_update(branch_model, option)
            branch_tables[int(option.index)] = _score_table_from_shadow(
                shadow=branch_model,
                teach_case=teach_case,
                menu_meta=menu_meta,
                counters=counters,
            )
        if branch_tables:
            first_reveal_cache[sig] = branch_tables
    return first_reveal_cache


def _full_active_mask(K: int) -> int:
    return (1 << K) - 1


def _active_positions_from_mask(mask: int, K: int) -> List[int]:
    return [idx for idx in range(K) if (mask >> idx) & 1]


def _correct_rank_from_probs(probs: np.ndarray, correct_pos: Optional[int]) -> Optional[int]:
    if correct_pos is None or correct_pos >= len(probs):
        return None
    target = float(probs[correct_pos])
    if target <= 0.0:
        return None
    rank = 1 + sum(1 for value in probs if value > target + 1e-12)
    return int(rank)


def _correct_prob_rank_margin_from_probs(
    probs: np.ndarray,
    correct_pos: Optional[int],
) -> Tuple[float, Optional[int], float]:
    if correct_pos is None or correct_pos >= len(probs):
        return 0.0, None, 0.0
    correct_prob = float(probs[correct_pos])
    correct_rank = _correct_rank_from_probs(probs, correct_pos)
    masked = np.asarray(probs, dtype=float).copy()
    masked[correct_pos] = -1.0
    top_wrong = float(masked.max()) if len(masked) > 1 else 0.0
    margin = float(correct_prob - top_wrong)
    return correct_prob, correct_rank, margin


def exactish_tau_le2_from_score_table(
    score_table: CandidateScoreTable,
    first_reveal_tables: Optional[Dict[int, CandidateScoreTable]],
    profile_weights: Sequence[Tuple[object, float]],
    cfg,
) -> float:
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


def _apply_collapse_prior_to_probs(
    probs: np.ndarray,
    correct_pos: Optional[int],
    prev_correct_prob: Optional[float],
    cfg,
) -> Tuple[np.ndarray, float]:
    if not getattr(cfg, "use_reveal_collapse_prior", False):
        return probs, 0.0
    if correct_pos is None or prev_correct_prob is None:
        return probs, 0.0
    if correct_pos >= len(probs):
        return probs, 0.0

    raw = float(probs[correct_pos])
    ratio = float(getattr(cfg, "collapse_ratio_median", 1.0))
    strength = float(getattr(cfg, "collapse_prior_strength", 1.0))
    cap = max(0.0, min(1.0, float(prev_correct_prob) * ratio))
    if raw <= cap + 1e-12:
        return probs, 0.0

    new_correct = raw - strength * max(0.0, raw - cap)
    new_correct = max(0.0, min(raw, new_correct))
    if abs(new_correct - raw) <= 1e-12:
        return probs, 0.0

    adjusted = np.asarray(probs, dtype=float).copy()
    other_mass = max(0.0, 1.0 - raw)
    adjusted[correct_pos] = new_correct
    redistributed = max(0.0, 1.0 - new_correct)
    if other_mass > 1e-12:
        scale = redistributed / other_mass
        for idx in range(len(adjusted)):
            if idx == correct_pos:
                continue
            adjusted[idx] = max(0.0, adjusted[idx] * scale)
    else:
        for idx in range(len(adjusted)):
            if idx != correct_pos:
                adjusted[idx] = 0.0
    total = float(adjusted.sum())
    if total > 0.0:
        adjusted /= total
    return adjusted, max(0.0, raw - new_correct)


def _score_table_probs(
    score_table: CandidateScoreTable,
    profile,
    active_mask: int,
    revealed_text_ids: frozenset[int],
    revealed_output_ids: frozenset[int],
    cfg,
) -> np.ndarray:
    K = len(score_table.sem)
    active_positions = _active_positions_from_mask(active_mask, K)
    if not active_positions:
        return np.zeros(K)

    sem = score_table.sem.copy()
    text_penalty = float(getattr(cfg, "planner_text_reveal_penalty", 0.0))
    output_penalty = float(getattr(cfg, "planner_output_reveal_penalty", 0.0))

    if revealed_text_ids and text_penalty > 0.0:
        for pos, text_id in enumerate(score_table.menu_meta.text_ids):
            if text_id in revealed_text_ids:
                sem[pos] -= text_penalty
    if revealed_output_ids and output_penalty > 0.0:
        for pos, output_id in enumerate(score_table.menu_meta.output_ids):
            if output_id in revealed_output_ids:
                sem[pos] -= output_penalty

    from ..tutor.learner_model import ShadowLearnerModel

    return ShadowLearnerModel.probs_from_components(
        sem=sem,
        mu_d=score_table.mu_d,
        u_d=score_table.u_d,
        profile=profile,
        active_positions=active_positions,
        K_full=K,
    )


def prefilter_score_table_under_profiles(
    score_table: CandidateScoreTable,
    profile_weights: Sequence[Tuple[object, float]],
    cfg,
) -> dict:
    weighted_prob = 0.0
    weighted_rank = 0.0
    weighted_rank_mass = 0.0
    weighted_margin = 0.0
    weighted_score = 0.0

    active_mask = _full_active_mask(len(score_table.sem))
    for profile, weight in profile_weights:
        probs = _score_table_probs(
            score_table=score_table,
            profile=profile,
            active_mask=active_mask,
            revealed_text_ids=frozenset(),
            revealed_output_ids=frozenset(),
            cfg=cfg,
        )
        correct_pos = score_table.menu_meta.correct_pos
        correct_prob, correct_rank, correct_margin = _correct_prob_rank_margin_from_probs(probs, correct_pos)
        exposure_proxy = 0.0
        if correct_rank is not None:
            exposure_proxy = min(max(correct_rank - 1, 0), int(cfg.max_attempts_main) - 1) / max(
                int(cfg.max_attempts_main) - 1,
                1,
            )
        time_proxy = attempt_time_reward(
            correct_rank if correct_rank is not None and correct_rank <= int(cfg.max_attempts_main) else None,
            cfg.target_attempt,
            cfg.sigma_tau,
        )
        early_penalty = 0.0
        if correct_rank is not None and correct_rank <= int(cfg.max_attempts_main):
            early_penalty = max(0.0, float(cfg.target_attempt) - float(correct_rank)) / max(float(cfg.target_attempt), 1.0)
        stage0_score = (
            4.0 * correct_prob
            + 2.0 * (0.0 if correct_rank is None else 1.0 / max(correct_rank, 1))
            + 0.75 * exposure_proxy
            + 0.5 * time_proxy
            - 0.25 * early_penalty
        )
        weighted_prob += float(weight) * correct_prob
        weighted_margin += float(weight) * correct_margin
        if correct_rank is not None:
            weighted_rank += float(weight) * float(correct_rank)
            weighted_rank_mass += float(weight)
        weighted_score += float(weight) * stage0_score
    return {
        "prefilter_score": float(weighted_score),
        "initial_correct_prob_mean": float(weighted_prob),
        "initial_correct_rank_mean": (weighted_rank / weighted_rank_mass) if weighted_rank_mass > 0 else None,
        "initial_correct_margin_mean": float(weighted_margin),
    }


def _mc_rollout_score_table(
    score_table: CandidateScoreTable,
    first_reveal_tables: Optional[Dict[int, CandidateScoreTable]],
    profile,
    cfg,
    rng: np.random.Generator,
    n_rollouts: int,
    counters: Optional[PlannerCounters] = None,
) -> dict:
    T = int(cfg.max_attempts_main)
    K = len(score_table.sem)
    success = 0.0
    safe_wrong = 0.0
    risk_any = 0.0
    risk_count = 0.0
    damage_sum = 0.0
    tau_sum = 0.0
    tau_n = 0
    time_reward_sum = 0.0
    tau_probs = np.zeros(T, dtype=float)
    attempt_correct_prob_sum = np.zeros(T, dtype=float)
    attempt_correct_rank_sum = np.zeros(T, dtype=float)
    attempt_correct_rank_count = np.zeros(T, dtype=float)
    attempt_reach_count = np.zeros(T, dtype=float)
    collapse_adjustment_sum = 0.0
    first_reveal_cache_hit = 0.0

    for _ in range(max(int(n_rollouts), 1)):
        if counters is not None:
            counters.n_rollout_paths += 1
        active_mask = _full_active_mask(K)
        revealed_text_ids = frozenset()
        revealed_output_ids = frozenset()
        current_table = score_table
        first_correct = None
        safe_ct = 0
        risk_ct = 0
        dmg = 0
        prev_correct_prob: Optional[float] = None
        used_first_reveal_cache = False

        for attempt in range(1, T + 1):
            probs = _score_table_probs(
                score_table=current_table,
                profile=profile,
                active_mask=active_mask,
                revealed_text_ids=revealed_text_ids,
                revealed_output_ids=revealed_output_ids,
                cfg=cfg,
            )
            if probs.sum() <= 0.0:
                break
            attempt_reach_count[attempt - 1] += 1.0
            correct_pos = current_table.menu_meta.correct_pos
            probs, collapse_adjustment = _apply_collapse_prior_to_probs(
                probs=probs,
                correct_pos=correct_pos,
                prev_correct_prob=prev_correct_prob if attempt > 1 else None,
                cfg=cfg,
            )
            collapse_adjustment_sum += float(collapse_adjustment)
            if correct_pos is not None and correct_pos < len(probs):
                correct_prob = float(probs[correct_pos])
                attempt_correct_prob_sum[attempt - 1] += correct_prob
                correct_rank = _correct_rank_from_probs(probs, correct_pos)
                if correct_rank is not None:
                    attempt_correct_rank_sum[attempt - 1] += float(correct_rank)
                    attempt_correct_rank_count[attempt - 1] += 1.0
                prev_correct_prob = correct_prob
            else:
                prev_correct_prob = None
            pos = int(rng.choice(K, p=probs))
            if pos == current_table.menu_meta.correct_pos:
                first_correct = attempt
                break
            damage = int(current_table.menu_meta.risk_classes[pos])
            dmg += damage
            if damage > 0:
                risk_ct += 1
            else:
                safe_ct += 1
            active_mask &= ~(1 << pos)
            revealed_text_ids = frozenset(set(revealed_text_ids) | {current_table.menu_meta.text_ids[pos]})
            revealed_output_ids = frozenset(set(revealed_output_ids) | {current_table.menu_meta.output_ids[pos]})
            if not used_first_reveal_cache and first_reveal_tables:
                option_index = int(current_table.menu_meta.option_indices[pos])
                next_table = first_reveal_tables.get(option_index)
                if next_table is not None:
                    current_table = next_table
                    used_first_reveal_cache = True
            if active_mask == 0:
                break

        if counters is not None:
            counters.n_terminal_paths += 1
        if first_correct is not None:
            success += 1.0
            tau_probs[first_correct - 1] += 1.0
            tau_sum += float(first_correct)
            tau_n += 1
        if risk_ct > 0:
            risk_any += 1.0
        risk_count += float(risk_ct)
        safe_wrong += float(safe_ct)
        damage_sum += float(dmg)
        if used_first_reveal_cache:
            first_reveal_cache_hit += 1.0
        time_reward_sum += attempt_time_reward(first_correct, cfg.target_attempt, cfg.sigma_tau)

    n = max(int(n_rollouts), 1)
    return {
        "success_prob": success / n,
        "eval_exact_acc": 0.0,
        "eval_cell_acc": 0.0,
        "safe_wrong_mean": safe_wrong / n,
        "risk_any_prob": risk_any / n,
        "risk_count_mean": risk_count / n,
        "damage_mean": damage_sum / n,
        "mean_first_correct_attempt": (tau_sum / tau_n) if tau_n > 0 else None,
        "time_reward_mean": time_reward_sum / n,
        "pred_p_tau_1_to_6": (tau_probs / n).tolist(),
        "pred_attempt_reach_prob": (attempt_reach_count / n).tolist(),
        "collapse_adjustment_mean": collapse_adjustment_sum / n,
        "first_reveal_cache_hit_prob": first_reveal_cache_hit / n,
        "pred_attempt_correct_prob_mean": [
            (attempt_correct_prob_sum[idx] / attempt_reach_count[idx]) if attempt_reach_count[idx] > 0 else None
            for idx in range(T)
        ],
        "pred_attempt_correct_rank_mean": [
            (attempt_correct_rank_sum[idx] / attempt_correct_rank_count[idx]) if attempt_correct_rank_count[idx] > 0 else None
            for idx in range(T)
        ],
    }


def _beam_rollout_score_table(
    score_table: CandidateScoreTable,
    first_reveal_tables: Optional[Dict[int, CandidateScoreTable]],
    profile,
    cfg,
    beam_top_b: int,
    beam_keep_l: int,
    counters: Optional[PlannerCounters] = None,
) -> dict:
    T = int(cfg.max_attempts_main)
    K = len(score_table.sem)
    frontier: List[DeltaPath] = [
        DeltaPath(
            active_mask=_full_active_mask(K),
            revealed_text_ids=frozenset(),
            revealed_output_ids=frozenset(),
        )
    ]
    attempt_reach_prob = np.zeros(T, dtype=float)
    attempt_prob_num = np.zeros(T, dtype=float)
    attempt_prob_den = np.zeros(T, dtype=float)
    attempt_rank_num = np.zeros(T, dtype=float)
    attempt_rank_den = np.zeros(T, dtype=float)
    collapse_adjustment_num = 0.0
    collapse_adjustment_den = 0.0
    for attempt_idx in range(T):
        expanded: List[DeltaPath] = []
        for path in frontier:
            if path.success or path.active_mask == 0 or path.attempts >= T:
                expanded.append(path)
                continue
            path_mass = math.exp(path.logprob)
            attempt_reach_prob[attempt_idx] += path_mass
            current_table = score_table
            if path.current_table_key is not None and first_reveal_tables:
                current_table = first_reveal_tables.get(path.current_table_key, score_table)
            probs = _score_table_probs(
                score_table=current_table,
                profile=profile,
                active_mask=path.active_mask,
                revealed_text_ids=path.revealed_text_ids,
                revealed_output_ids=path.revealed_output_ids,
                cfg=cfg,
            )
            probs, collapse_adjustment = _apply_collapse_prior_to_probs(
                probs=probs,
                correct_pos=current_table.menu_meta.correct_pos,
                prev_correct_prob=path.prev_correct_prob if path.attempts >= 1 else None,
                cfg=cfg,
            )
            collapse_adjustment_num += path_mass * float(collapse_adjustment)
            collapse_adjustment_den += path_mass
            correct_pos = current_table.menu_meta.correct_pos
            if correct_pos is not None and correct_pos < len(probs):
                correct_prob = float(probs[correct_pos])
                attempt_prob_num[attempt_idx] += path_mass * correct_prob
                attempt_prob_den[attempt_idx] += path_mass
                correct_rank = _correct_rank_from_probs(probs, correct_pos)
                if correct_rank is not None:
                    attempt_rank_num[attempt_idx] += path_mass * float(correct_rank)
                    attempt_rank_den[attempt_idx] += path_mass
            ranked = np.argsort(-probs)[: max(1, int(beam_top_b))]
            any_child = False
            for pos in ranked:
                p = float(probs[pos])
                if p <= 0.0:
                    continue
                any_child = True
                next_prev_correct = None
                if current_table.menu_meta.correct_pos is not None and current_table.menu_meta.correct_pos < len(probs):
                    next_prev_correct = float(probs[current_table.menu_meta.correct_pos])
                if pos == current_table.menu_meta.correct_pos:
                    expanded.append(
                        DeltaPath(
                            active_mask=path.active_mask,
                            revealed_text_ids=path.revealed_text_ids,
                            revealed_output_ids=path.revealed_output_ids,
                            prev_correct_prob=next_prev_correct,
                            current_table_key=path.current_table_key,
                            used_first_reveal_cache=path.used_first_reveal_cache,
                            attempts=path.attempts + 1,
                            success=True,
                            first_correct_attempt=path.attempts + 1,
                            safe_wrong_count=path.safe_wrong_count,
                            risky_wrong_count=path.risky_wrong_count,
                            risk_count=path.risk_count,
                            damage_sum=path.damage_sum,
                            logprob=path.logprob + math.log(max(p, 1e-30)),
                            picks=path.picks + (int(current_table.menu_meta.option_indices[pos]),),
                        )
                    )
                    continue
                damage = int(current_table.menu_meta.risk_classes[pos])
                option_index = int(current_table.menu_meta.option_indices[pos])
                next_table_key = path.current_table_key
                used_first_reveal_cache = path.used_first_reveal_cache
                if path.current_table_key is None and first_reveal_tables and option_index in first_reveal_tables:
                    next_table_key = option_index
                    used_first_reveal_cache = True
                expanded.append(
                    DeltaPath(
                        active_mask=path.active_mask & ~(1 << pos),
                        revealed_text_ids=frozenset(set(path.revealed_text_ids) | {current_table.menu_meta.text_ids[pos]}),
                        revealed_output_ids=frozenset(set(path.revealed_output_ids) | {current_table.menu_meta.output_ids[pos]}),
                        prev_correct_prob=next_prev_correct,
                        current_table_key=next_table_key,
                        used_first_reveal_cache=used_first_reveal_cache,
                        attempts=path.attempts + 1,
                        success=False,
                        first_correct_attempt=path.first_correct_attempt,
                        safe_wrong_count=path.safe_wrong_count + (0 if damage > 0 else 1),
                        risky_wrong_count=path.risky_wrong_count + (1 if damage > 0 else 0),
                        risk_count=path.risk_count + (1 if damage > 0 else 0),
                        damage_sum=path.damage_sum + damage,
                        logprob=path.logprob + math.log(max(p, 1e-30)),
                        picks=path.picks + (option_index,),
                    )
                )
            if not any_child:
                expanded.append(path)
        expanded.sort(key=lambda item: item.logprob, reverse=True)
        frontier = expanded[: max(1, int(beam_keep_l))]
        if all(path.success or path.active_mask == 0 or path.attempts >= T for path in frontier):
            break

    if counters is not None:
        counters.n_rollout_paths += len(frontier)
        counters.n_terminal_paths += len(frontier)

    if not frontier:
        return {
            "success_prob": 0.0,
            "eval_exact_acc": 0.0,
            "eval_cell_acc": 0.0,
            "safe_wrong_mean": 0.0,
            "risk_any_prob": 0.0,
            "risk_count_mean": 0.0,
            "damage_mean": 0.0,
            "mean_first_correct_attempt": None,
            "time_reward_mean": 0.0,
            "pred_p_tau_1_to_6": [0.0] * T,
            "pred_attempt_reach_prob": [0.0] * T,
            "pred_attempt_correct_prob_mean": [None] * T,
            "pred_attempt_correct_rank_mean": [None] * T,
        }

    max_logprob = max(path.logprob for path in frontier)
    weights = np.array([math.exp(path.logprob - max_logprob) for path in frontier], dtype=float)
    total = float(weights.sum()) or 1.0
    tau_probs = np.zeros(T, dtype=float)
    success_prob = 0.0
    safe_wrong_mean = 0.0
    risk_any_prob = 0.0
    risk_count_mean = 0.0
    damage_mean = 0.0
    tau_sum = 0.0
    tau_mass = 0.0
    time_reward_mean = 0.0
    first_reveal_cache_hit_prob = 0.0
    for weight, path in zip(weights, frontier):
        w = float(weight / total)
        if path.success and path.first_correct_attempt is not None:
            success_prob += w
            tau_probs[path.first_correct_attempt - 1] += w
            tau_sum += w * float(path.first_correct_attempt)
            tau_mass += w
        safe_wrong_mean += w * float(path.safe_wrong_count)
        if path.risk_count > 0:
            risk_any_prob += w
        risk_count_mean += w * float(path.risk_count)
        damage_mean += w * float(path.damage_sum)
        if path.used_first_reveal_cache:
            first_reveal_cache_hit_prob += w
        time_reward_mean += w * attempt_time_reward(path.first_correct_attempt, cfg.target_attempt, cfg.sigma_tau)
    return {
        "success_prob": float(success_prob),
        "eval_exact_acc": 0.0,
        "eval_cell_acc": 0.0,
        "safe_wrong_mean": float(safe_wrong_mean),
        "risk_any_prob": float(risk_any_prob),
        "risk_count_mean": float(risk_count_mean),
        "damage_mean": float(damage_mean),
        "mean_first_correct_attempt": (tau_sum / tau_mass) if tau_mass > 0.0 else None,
        "time_reward_mean": float(time_reward_mean),
        "pred_p_tau_1_to_6": tau_probs.tolist(),
        "pred_attempt_reach_prob": attempt_reach_prob.tolist(),
        "collapse_adjustment_mean": (
            float(collapse_adjustment_num / collapse_adjustment_den)
            if collapse_adjustment_den > 0.0
            else 0.0
        ),
        "first_reveal_cache_hit_prob": float(first_reveal_cache_hit_prob),
        "pred_attempt_correct_prob_mean": [
            (attempt_prob_num[idx] / attempt_prob_den[idx]) if attempt_prob_den[idx] > 0.0 else None
            for idx in range(T)
        ],
        "pred_attempt_correct_rank_mean": [
            (attempt_rank_num[idx] / attempt_rank_den[idx]) if attempt_rank_den[idx] > 0.0 else None
            for idx in range(T)
        ],
    }


def evaluate_score_table_under_profiles(
    score_table: CandidateScoreTable,
    profile_weights: Sequence[Tuple[object, float]],
    cfg,
    seed: int,
    stage: str,
    first_reveal_tables: Optional[Dict[int, CandidateScoreTable]] = None,
    counters: Optional[PlannerCounters] = None,
) -> dict:
    if stage == "refine":
        rollout_mode = getattr(cfg, "proxy_rollout_mode", "mc")
        n_rollouts = int(getattr(cfg, "refine_n_rollouts", cfg.n_rollouts))
        beam_top_b = int(getattr(cfg, "refine_beam_top_b", cfg.beam_top_b))
        beam_keep_l = int(getattr(cfg, "refine_beam_keep_l", cfg.beam_keep_l))
    else:
        rollout_mode = getattr(cfg, "proxy_rollout_mode", "mc")
        n_rollouts = int(getattr(cfg, "proxy_n_rollouts", cfg.n_rollouts))
        beam_top_b = int(getattr(cfg, "proxy_beam_top_b", cfg.beam_top_b))
        beam_keep_l = int(getattr(cfg, "proxy_beam_keep_l", cfg.beam_keep_l))

    initial = prefilter_score_table_under_profiles(score_table, profile_weights, cfg)
    weighted = {
        "success_prob": 0.0,
        "eval_exact_acc": 0.0,
        "eval_cell_acc": 0.0,
        "safe_wrong_mean": 0.0,
        "risk_any_prob": 0.0,
        "risk_count_mean": 0.0,
        "damage_mean": 0.0,
        "time_reward_mean": 0.0,
        "collapse_adjustment_mean": 0.0,
        "first_reveal_cache_hit_prob": 0.0,
    }
    tau_acc = 0.0
    tau_weight = 0.0
    tau_probs = np.zeros(int(cfg.max_attempts_main), dtype=float)
    attempt_prob_num = np.zeros(int(cfg.max_attempts_main), dtype=float)
    attempt_prob_den = np.zeros(int(cfg.max_attempts_main), dtype=float)
    attempt_rank_num = np.zeros(int(cfg.max_attempts_main), dtype=float)
    attempt_rank_den = np.zeros(int(cfg.max_attempts_main), dtype=float)

    for idx, (profile, weight) in enumerate(profile_weights):
        if rollout_mode == "score_table_beam":
            stats = _beam_rollout_score_table(
                score_table=score_table,
                first_reveal_tables=first_reveal_tables,
                profile=profile,
                cfg=cfg,
                beam_top_b=beam_top_b,
                beam_keep_l=beam_keep_l,
                counters=counters,
            )
        else:
            stats = _mc_rollout_score_table(
                score_table=score_table,
                first_reveal_tables=first_reveal_tables,
                profile=profile,
                cfg=cfg,
                rng=np.random.default_rng(seed + 101 * (idx + 1)),
                n_rollouts=n_rollouts,
                counters=counters,
            )
        for key in weighted:
            weighted[key] += float(weight) * float(stats.get(key, 0.0))
        tau_vec = np.asarray(stats.get("pred_p_tau_1_to_6", [0.0] * int(cfg.max_attempts_main)), dtype=float)
        tau_probs[: len(tau_vec)] += float(weight) * tau_vec[: int(cfg.max_attempts_main)]
        attempt_reach_vec = stats.get("pred_attempt_reach_prob", [None] * int(cfg.max_attempts_main))
        attempt_prob_vec = stats.get("pred_attempt_correct_prob_mean", [None] * int(cfg.max_attempts_main))
        attempt_rank_vec = stats.get("pred_attempt_correct_rank_mean", [None] * int(cfg.max_attempts_main))
        for attempt in range(int(cfg.max_attempts_main)):
            reach_prob = None if attempt >= len(attempt_reach_vec) else attempt_reach_vec[attempt]
            if reach_prob is None:
                continue
            reach_prob = float(reach_prob)
            if attempt < len(attempt_prob_vec) and attempt_prob_vec[attempt] is not None:
                attempt_prob_num[attempt] += float(weight) * reach_prob * float(attempt_prob_vec[attempt])
                attempt_prob_den[attempt] += float(weight) * reach_prob
            if attempt < len(attempt_rank_vec) and attempt_rank_vec[attempt] is not None:
                attempt_rank_num[attempt] += float(weight) * reach_prob * float(attempt_rank_vec[attempt])
                attempt_rank_den[attempt] += float(weight) * reach_prob
        if stats.get("mean_first_correct_attempt") is not None:
            tau_acc += float(weight) * float(stats["mean_first_correct_attempt"])
            tau_weight += float(weight)

    weighted["mean_first_correct_attempt"] = (tau_acc / tau_weight) if tau_weight > 0.0 else None
    weighted["pred_p_tau_1_to_6"] = tau_probs.tolist()
    weighted["pred_attempt_correct_prob_mean"] = [
        (attempt_prob_num[idx] / attempt_prob_den[idx]) if attempt_prob_den[idx] > 0.0 else None
        for idx in range(int(cfg.max_attempts_main))
    ]
    weighted["pred_attempt_correct_rank_mean"] = [
        (attempt_rank_num[idx] / attempt_rank_den[idx]) if attempt_rank_den[idx] > 0.0 else None
        for idx in range(int(cfg.max_attempts_main))
    ]
    weighted["initial_correct_prob_mean"] = initial["initial_correct_prob_mean"]
    weighted["initial_correct_rank_mean"] = initial["initial_correct_rank_mean"]
    weighted["initial_correct_margin_mean"] = initial.get("initial_correct_margin_mean", 0.0)
    weighted["pred_tau_le2_exact"] = exactish_tau_le2_from_score_table(
        score_table=score_table,
        first_reveal_tables=first_reveal_tables,
        profile_weights=profile_weights,
        cfg=cfg,
    )
    weighted["wrong_before_correct_mean"] = float(weighted["safe_wrong_mean"]) + float(weighted["risk_count_mean"])
    weighted["prefilter_score"] = initial["prefilter_score"]
    return weighted


def _expand_beam_state(
    state: BeamState,
    profile,
    teach_example,
    beam_top_b: int,
    cfg,
) -> List[BeamState]:
    if state.success or not state.active_menu:
        return [state]
    probs = state.shadow.predict_pick_probs(
        target_output=list(teach_example.output),
        option_texts=[list(opt.text) for opt in state.active_menu],
        option_danger_vecs=[np.asarray(opt.danger_vec) for opt in state.active_menu],
        profile=profile,
        spec={"action": "WAIT"},
        banned_indices=set(),
        highlighted_cells=(),
        option_indices=[opt.index for opt in state.active_menu],
    )
    if len(probs) == 0:
        return [state]
    ranked = np.argsort(-probs)[: max(1, beam_top_b)]
    children: List[BeamState] = []
    for pos in ranked:
        p = float(probs[pos])
        if p <= 0.0:
            continue
        child = BeamState(
            prob=state.prob * p,
            shadow=state.shadow.deep_copy(),
            active_menu=copy.deepcopy(state.active_menu),
            attempts=state.attempts + 1,
            success=state.success,
            first_correct_attempt=state.first_correct_attempt,
            safe_wrong_count=state.safe_wrong_count,
            risky_wrong_count=state.risky_wrong_count,
            risk_count=state.risk_count,
            damage_sum=state.damage_sum,
        )
        picked = child.active_menu[pos]
        if picked.is_correct:
            child.success = True
            child.first_correct_attempt = child.attempts
            _apply_planning_correct_update(child.shadow, picked, teach_example.output, cfg)
        else:
            damage = int(getattr(picked, "risk_class", 0))
            if damage > 0:
                child.risk_count += 1
                child.risky_wrong_count += 1
            else:
                child.safe_wrong_count += 1
            child.damage_sum += damage
            _apply_planning_wrong_update(child.shadow, picked, teach_example.output, cfg)
            child.active_menu.pop(pos)
        children.append(child)
    return children or [state]


def _beam_rollout_profile(
    shadow,
    profile,
    hint: Optional[HintCandidate],
    teach_case: TeachCase,
    eval_items: Iterable[EvalItem],
    cfg,
) -> dict:
    eval_items = list(eval_items)
    should_eval = bool(getattr(cfg, "eval_aware", False)) and bool(eval_items)
    root_shadow = shadow.deep_copy()
    _apply_hint_to_shadow(root_shadow, hint)
    frontier: List[BeamState] = [
        BeamState(
            prob=1.0,
            shadow=root_shadow,
            active_menu=copy.deepcopy(teach_case.menu),
        )
    ]
    for _ in range(int(cfg.max_attempts_main)):
        expanded: List[BeamState] = []
        for state in frontier:
            if state.success or state.attempts >= int(cfg.max_attempts_main) or not state.active_menu:
                expanded.append(state)
                continue
            expanded.extend(
                _expand_beam_state(
                    state,
                    profile,
                    teach_case.example,
                    int(cfg.beam_top_b),
                    cfg,
                )
            )
        expanded.sort(key=lambda item: item.prob, reverse=True)
        frontier = expanded[: max(1, int(cfg.beam_keep_l))]
        if all(state.success or state.attempts >= int(cfg.max_attempts_main) or not state.active_menu for state in frontier):
            break

    total_prob = sum(state.prob for state in frontier) or 1.0
    success_prob = sum(state.prob for state in frontier if state.success) / total_prob
    if should_eval:
        eval_exact_num = 0.0
        eval_cell_num = 0.0
        for state in frontier:
            exact_acc, cell_acc = _shadow_eval_metrics(state.shadow, eval_items)
            eval_exact_num += state.prob * exact_acc
            eval_cell_num += state.prob * cell_acc
        eval_exact = eval_exact_num / total_prob
        eval_cell = eval_cell_num / total_prob
    else:
        eval_exact = 0.0
        eval_cell = 0.0
    safe_wrong_mean = sum(state.prob * state.safe_wrong_count for state in frontier) / total_prob
    risk_any_prob = sum(state.prob for state in frontier if state.risk_count > 0) / total_prob
    risk_count_mean = sum(state.prob * state.risk_count for state in frontier) / total_prob
    damage_mean = sum(state.prob * state.damage_sum for state in frontier) / total_prob
    tau_probs = np.zeros(int(cfg.max_attempts_main), dtype=float)
    tau_weight = 0.0
    tau_acc = 0.0
    for state in frontier:
        if state.first_correct_attempt is not None:
            tau_probs[state.first_correct_attempt - 1] += state.prob / total_prob
            tau_acc += state.prob * float(state.first_correct_attempt)
            tau_weight += state.prob
    mean_tau = (tau_acc / tau_weight) if tau_weight > 0.0 else None
    time_reward_mean = sum(
        state.prob * attempt_time_reward(state.first_correct_attempt, cfg.target_attempt, cfg.sigma_tau)
        for state in frontier
    ) / total_prob
    return {
        "success_prob": float(success_prob),
        "eval_exact_acc": float(eval_exact),
        "eval_cell_acc": float(eval_cell),
        "safe_wrong_mean": float(safe_wrong_mean),
        "risk_any_prob": float(risk_any_prob),
        "risk_count_mean": float(risk_count_mean),
        "damage_mean": float(damage_mean),
        "mean_first_correct_attempt": None if mean_tau is None else float(mean_tau),
        "time_reward_mean": float(time_reward_mean),
        "pred_p_tau_1_to_6": tau_probs.tolist(),
    }


def _mc_rollout_profile(
    shadow,
    profile,
    hint: Optional[HintCandidate],
    teach_case: TeachCase,
    eval_items: Iterable[EvalItem],
    cfg,
    rng: np.random.Generator,
) -> dict:
    eval_items = list(eval_items)
    should_eval = bool(getattr(cfg, "eval_aware", False)) and bool(eval_items)
    success = 0
    eval_exact_sum = 0.0
    eval_cell_sum = 0.0
    safe_wrong = 0.0
    risk_any = 0.0
    risk_count = 0.0
    damage_sum = 0.0
    tau_sum = 0.0
    tau_n = 0
    time_reward_sum = 0.0
    tau_probs = np.zeros(int(cfg.max_attempts_main), dtype=float)
    base_model = shadow.deep_copy()
    _apply_hint_to_shadow(base_model, hint)
    for _ in range(int(cfg.n_rollouts)):
        model = base_model.deep_copy()
        active = copy.deepcopy(teach_case.menu)
        first_correct = None
        safe_ct = 0
        risk_ct = 0
        dmg = 0
        for attempt in range(1, int(cfg.max_attempts_main) + 1):
            probs = model.predict_pick_probs(
                target_output=list(teach_case.example.output),
                option_texts=[list(opt.text) for opt in active],
                option_danger_vecs=[np.asarray(opt.danger_vec) for opt in active],
                profile=profile,
                spec={"action": "WAIT"},
                banned_indices=set(),
                highlighted_cells=(),
                option_indices=[opt.index for opt in active],
            )
            if len(probs) == 0:
                break
            pos = int(rng.choice(len(active), p=probs))
            picked = active[pos]
            if picked.is_correct:
                first_correct = attempt
                _apply_planning_correct_update(model, picked, teach_case.example.output, cfg)
                break
            damage = int(getattr(picked, "risk_class", 0))
            dmg += damage
            if damage > 0:
                risk_ct += 1
            else:
                safe_ct += 1
            _apply_planning_wrong_update(model, picked, teach_case.example.output, cfg)
            active.pop(pos)
            if not active:
                break
        if first_correct is not None:
            success += 1
            tau_probs[first_correct - 1] += 1.0
            tau_sum += first_correct
            tau_n += 1
        if risk_ct > 0:
            risk_any += 1.0
        risk_count += risk_ct
        safe_wrong += safe_ct
        damage_sum += dmg
        if should_eval:
            exact_acc, cell_acc = _shadow_eval_metrics(model, eval_items)
            eval_exact_sum += exact_acc
            eval_cell_sum += cell_acc
        time_reward_sum += attempt_time_reward(first_correct, cfg.target_attempt, cfg.sigma_tau)
    n = max(int(cfg.n_rollouts), 1)
    return {
        "success_prob": success / n,
        "eval_exact_acc": (eval_exact_sum / n) if should_eval else 0.0,
        "eval_cell_acc": (eval_cell_sum / n) if should_eval else 0.0,
        "safe_wrong_mean": safe_wrong / n,
        "risk_any_prob": risk_any / n,
        "risk_count_mean": risk_count / n,
        "damage_mean": damage_sum / n,
        "mean_first_correct_attempt": (tau_sum / tau_n) if tau_n > 0 else None,
        "time_reward_mean": time_reward_sum / n,
        "pred_p_tau_1_to_6": (tau_probs / n).tolist(),
    }


def _mc_rollout_profile_lazy_cls(
    shadow,
    profile,
    hint: Optional[HintCandidate],
    teach_case: TeachCase,
    eval_items: Iterable[EvalItem],
    cfg,
    rng: np.random.Generator,
    counters: Optional[PlannerCounters] = None,
) -> dict:
    eval_items = list(eval_items)
    should_eval = bool(getattr(cfg, "eval_aware", False)) and bool(eval_items)
    success = 0.0
    eval_exact_sum = 0.0
    eval_cell_sum = 0.0
    safe_wrong = 0.0
    risk_any = 0.0
    risk_count = 0.0
    damage_sum = 0.0
    tau_sum = 0.0
    tau_n = 0
    time_reward_sum = 0.0
    T = int(cfg.max_attempts_main)
    tau_probs = np.zeros(T, dtype=float)
    attempt_correct_prob_sum = np.zeros(T, dtype=float)
    attempt_correct_rank_sum = np.zeros(T, dtype=float)
    attempt_correct_rank_count = np.zeros(T, dtype=float)
    attempt_reach_count = np.zeros(T, dtype=float)

    option_by_index = {int(opt.index): opt for opt in teach_case.menu}

    base_model = shadow.deep_copy()
    if counters is not None:
        counters.n_cls_deepcopy_calls += 1
    scorer = getattr(base_model, "scorer", None)
    if scorer is not None and hasattr(scorer, "reset_debug_counters"):
        scorer.reset_debug_counters()
    _apply_hint_to_shadow(base_model, hint)
    _aggregate_scorer_debug(getattr(base_model, "scorer", None), counters)

    prefix_cache: Dict[Tuple[int, ...], dict] = {}

    def policy_for_prefix(prefix: Tuple[int, ...]) -> dict:
        if prefix in prefix_cache:
            return prefix_cache[prefix]
        model = base_model.deep_copy()
        if counters is not None:
            counters.n_cls_deepcopy_calls += 1
        scorer_local = getattr(model, "scorer", None)
        if scorer_local is not None and hasattr(scorer_local, "reset_debug_counters"):
            scorer_local.reset_debug_counters()
        _apply_lazy_prefix_wrong_updates(
            shadow=model,
            wrong_prefix=prefix,
            option_by_index=option_by_index,
            target_output=list(teach_case.example.output),
            cfg=cfg,
        )
        active = [opt for opt in teach_case.menu if int(opt.index) not in prefix]
        probs = model.predict_pick_probs(
            target_output=list(teach_case.example.output),
            option_texts=[list(opt.text) for opt in active],
            option_danger_vecs=[np.asarray(opt.danger_vec) for opt in active],
            profile=profile,
            spec={"action": "WAIT"},
            banned_indices=set(),
            highlighted_cells=(),
            option_indices=[opt.index for opt in active],
        )
        _aggregate_scorer_debug(getattr(model, "scorer", None), counters)
        prefix_cache[prefix] = {
            "model": model,
            "active": active,
            "probs": np.asarray(probs, dtype=float),
        }
        return prefix_cache[prefix]

    for _ in range(max(int(cfg.n_rollouts), 1)):
        if counters is not None:
            counters.n_rollout_paths += 1
        wrong_prefix: Tuple[int, ...] = ()
        first_correct = None
        safe_ct = 0
        risk_ct = 0
        dmg = 0

        for attempt in range(1, T + 1):
            state = policy_for_prefix(wrong_prefix)
            active = state["active"]
            probs = np.asarray(state["probs"], dtype=float)
            if len(active) == 0 or probs.sum() <= 0.0:
                break

            attempt_reach_count[attempt - 1] += 1.0
            correct_pos = next((idx for idx, opt in enumerate(active) if opt.is_correct), None)
            if correct_pos is not None and correct_pos < len(probs):
                correct_prob = float(probs[correct_pos])
                attempt_correct_prob_sum[attempt - 1] += correct_prob
                correct_rank = _correct_rank_from_probs(probs, correct_pos)
                if correct_rank is not None:
                    attempt_correct_rank_sum[attempt - 1] += float(correct_rank)
                    attempt_correct_rank_count[attempt - 1] += 1.0

            pos = int(rng.choice(len(active), p=probs))
            picked = active[pos]
            if picked.is_correct:
                first_correct = attempt
                break

            damage = int(getattr(picked, "risk_class", 0))
            dmg += damage
            if damage > 0:
                risk_ct += 1
            else:
                safe_ct += 1
            wrong_prefix = wrong_prefix + (int(picked.index),)

        if counters is not None:
            counters.n_terminal_paths += 1
        if first_correct is not None:
            success += 1.0
            tau_probs[first_correct - 1] += 1.0
            tau_sum += float(first_correct)
            tau_n += 1
        if risk_ct > 0:
            risk_any += 1.0
        risk_count += float(risk_ct)
        safe_wrong += float(safe_ct)
        damage_sum += float(dmg)

        if should_eval:
            leaf_state = policy_for_prefix(wrong_prefix)
            eval_model = leaf_state["model"].deep_copy()
            if counters is not None:
                counters.n_cls_deepcopy_calls += 1
            if first_correct is not None:
                correct_option = next((opt for opt in teach_case.menu if opt.is_correct), None)
                if correct_option is not None:
                    _apply_correct_update(eval_model, correct_option, teach_case.example.output)
            exact_acc, cell_acc = _shadow_eval_metrics(eval_model, eval_items)
            eval_exact_sum += exact_acc
            eval_cell_sum += cell_acc
        time_reward_sum += attempt_time_reward(first_correct, cfg.target_attempt, cfg.sigma_tau)

    n = max(int(cfg.n_rollouts), 1)
    return {
        "success_prob": success / n,
        "eval_exact_acc": (eval_exact_sum / n) if should_eval else 0.0,
        "eval_cell_acc": (eval_cell_sum / n) if should_eval else 0.0,
        "safe_wrong_mean": safe_wrong / n,
        "risk_any_prob": risk_any / n,
        "risk_count_mean": risk_count / n,
        "damage_mean": damage_sum / n,
        "mean_first_correct_attempt": (tau_sum / tau_n) if tau_n > 0 else None,
        "time_reward_mean": time_reward_sum / n,
        "pred_p_tau_1_to_6": (tau_probs / n).tolist(),
        "pred_attempt_reach_prob": (attempt_reach_count / n).tolist(),
        "pred_attempt_correct_prob_mean": [
            (attempt_correct_prob_sum[idx] / attempt_reach_count[idx]) if attempt_reach_count[idx] > 0 else None
            for idx in range(T)
        ],
        "pred_attempt_correct_rank_mean": [
            (attempt_correct_rank_sum[idx] / attempt_correct_rank_count[idx]) if attempt_correct_rank_count[idx] > 0 else None
            for idx in range(T)
        ],
    }


def evaluate_hint_under_posterior(
    posterior,
    hint: Optional[HintCandidate],
    teach_case: TeachCase,
    eval_items: Iterable[EvalItem],
    cfg,
    seed: int,
    counters: Optional[PlannerCounters] = None,
) -> dict:
    weighted = {
        "success_prob": 0.0,
        "eval_exact_acc": 0.0,
        "eval_cell_acc": 0.0,
        "safe_wrong_mean": 0.0,
        "risk_any_prob": 0.0,
        "risk_count_mean": 0.0,
        "damage_mean": 0.0,
        "time_reward_mean": 0.0,
    }
    tau_acc = 0.0
    tau_weight = 0.0
    tau_probs = np.zeros(int(cfg.max_attempts_main), dtype=float)
    attempt_prob_num = np.zeros(int(cfg.max_attempts_main), dtype=float)
    attempt_prob_den = np.zeros(int(cfg.max_attempts_main), dtype=float)
    attempt_rank_num = np.zeros(int(cfg.max_attempts_main), dtype=float)
    attempt_rank_den = np.zeros(int(cfg.max_attempts_main), dtype=float)
    initial_prob_acc = 0.0
    initial_rank_acc = 0.0
    initial_rank_weight = 0.0
    initial_margin_acc = 0.0
    eval_items = list(eval_items)
    base_shadow = posterior.cloned_shadow()
    profile_weights = posterior.profiles_for_stage("prefilter", cfg)
    for idx, (profile, weight) in enumerate(profile_weights):
        if weight <= 0.0:
            continue
        initial_stats = _initial_hint_stats_shadow(
            shadow=base_shadow,
            profile=profile,
            hint=hint,
            teach_case=teach_case,
            counters=counters,
        )
        initial_prob_acc += float(weight) * float(initial_stats.get("initial_correct_prob_mean", 0.0))
        initial_margin_acc += float(weight) * float(initial_stats.get("initial_correct_margin_mean", 0.0))
        if initial_stats.get("initial_correct_rank_mean") is not None:
            initial_rank_acc += float(weight) * float(initial_stats["initial_correct_rank_mean"])
            initial_rank_weight += float(weight)
        if str(getattr(cfg, "planning_update_mode", "proxy")) == "lazy_cls":
            stats = _mc_rollout_profile_lazy_cls(
                base_shadow,
                profile,
                hint,
                teach_case,
                eval_items,
                cfg,
                np.random.default_rng(seed + 101 * (idx + 1)),
                counters=counters,
            )
        elif cfg.rollout_mode == "mc":
            stats = _mc_rollout_profile(
                base_shadow,
                profile,
                hint,
                teach_case,
                eval_items,
                cfg,
                np.random.default_rng(seed + 101 * (idx + 1)),
            )
        else:
            stats = _beam_rollout_profile(base_shadow, profile, hint, teach_case, eval_items, cfg)
        for key in weighted:
            weighted[key] += float(weight) * float(stats.get(key, 0.0))
        tau_vec = np.asarray(stats.get("pred_p_tau_1_to_6", [0.0] * int(cfg.max_attempts_main)), dtype=float)
        tau_probs[: len(tau_vec)] += float(weight) * tau_vec[: int(cfg.max_attempts_main)]
        attempt_reach_vec = stats.get("pred_attempt_reach_prob", [None] * int(cfg.max_attempts_main))
        attempt_prob_vec = stats.get("pred_attempt_correct_prob_mean", [None] * int(cfg.max_attempts_main))
        attempt_rank_vec = stats.get("pred_attempt_correct_rank_mean", [None] * int(cfg.max_attempts_main))
        for attempt in range(int(cfg.max_attempts_main)):
            reach_prob = None if attempt >= len(attempt_reach_vec) else attempt_reach_vec[attempt]
            if reach_prob is None:
                continue
            reach_prob = float(reach_prob)
            if attempt < len(attempt_prob_vec) and attempt_prob_vec[attempt] is not None:
                attempt_prob_num[attempt] += float(weight) * reach_prob * float(attempt_prob_vec[attempt])
                attempt_prob_den[attempt] += float(weight) * reach_prob
            if attempt < len(attempt_rank_vec) and attempt_rank_vec[attempt] is not None:
                attempt_rank_num[attempt] += float(weight) * reach_prob * float(attempt_rank_vec[attempt])
                attempt_rank_den[attempt] += float(weight) * reach_prob
        if stats.get("mean_first_correct_attempt") is not None:
            tau_acc += float(weight) * float(stats["mean_first_correct_attempt"])
            tau_weight += float(weight)
    weighted["mean_first_correct_attempt"] = (tau_acc / tau_weight) if tau_weight > 0.0 else None
    weighted["pred_p_tau_1_to_6"] = tau_probs.tolist()
    weighted["pred_attempt_correct_prob_mean"] = [
        (attempt_prob_num[idx] / attempt_prob_den[idx]) if attempt_prob_den[idx] > 0.0 else None
        for idx in range(int(cfg.max_attempts_main))
    ]
    weighted["pred_attempt_correct_rank_mean"] = [
        (attempt_rank_num[idx] / attempt_rank_den[idx]) if attempt_rank_den[idx] > 0.0 else None
        for idx in range(int(cfg.max_attempts_main))
    ]
    weighted["initial_correct_prob_mean"] = float(initial_prob_acc)
    weighted["initial_correct_rank_mean"] = (initial_rank_acc / initial_rank_weight) if initial_rank_weight > 0.0 else None
    weighted["initial_correct_margin_mean"] = float(initial_margin_acc)
    weighted["pred_tau_le2_exact"] = float(sum(tau_probs[:2]))
    weighted["wrong_before_correct_mean"] = float(weighted["safe_wrong_mean"]) + float(weighted["risk_count_mean"])
    return weighted
