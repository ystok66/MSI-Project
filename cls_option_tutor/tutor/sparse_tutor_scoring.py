from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..config import FullConfig
from ..env.state import QueryState
from ..interfaces import Option


def _clip_pick_probs(probs, n: int) -> np.ndarray:
    raw = np.asarray(probs, dtype=float)
    if len(raw) > n:
        raw = raw[:n]
    if len(raw) < n:
        out = np.zeros(n, dtype=float)
        out[:len(raw)] = raw
        return out
    return raw


def _correct_rank_and_margin(active: List[Option], probs) -> tuple[int, float, float]:
    raw = _clip_pick_probs(probs, len(active))
    if len(raw) == 0:
        return (10**6, 0.0, 0.0)

    correct_idx = None
    for i, opt in enumerate(active):
        if opt.is_correct:
            correct_idx = i
            break
    if correct_idx is None:
        return (10**6, 0.0, 0.0)

    p_correct = float(raw[correct_idx])
    others = [float(raw[i]) for i, _ in enumerate(active) if i != correct_idx]
    top_other = max(others) if others else 0.0
    rank = 1 + sum(1 for p in others if p > p_correct)
    margin = p_correct - top_other
    return rank, margin, p_correct


def _label_is_far_wrong(label: str) -> bool:
    return label in ("safe_far", "safe_random_wrong", "risky_far")


def _option_badness(
    opt: Option,
    *,
    label: str,
    last_wrong_index: Optional[int],
    hp_scale: float,
) -> float:
    """Low-parameter badness proxy for trajectory value.

    We deliberately keep this compact:
    - normalized damage captures safety pressure
    - repeated-wrong penalty captures post-reveal confusion loops
    - far-wrong penalty captures low-information drift / boredom
    """
    hp_scale = max(float(hp_scale), 1.0)
    damage_norm = min(1.0, float(getattr(opt, "risk_class", 0)) / hp_scale)
    repeat_pen = 0.5 if (last_wrong_index is not None and opt.index == last_wrong_index) else 0.0
    far_pen = 0.25 if _label_is_far_wrong(label) else 0.0
    return damage_norm + repeat_pen + far_pen


def _option_infovalue(
    opt: Option,
    *,
    label: str,
) -> float:
    """Low-parameter productive-exploration proxy.

    Safe diagnostic wrongs are potentially useful because they can trigger
    reveal-driven learning. Bounded diagnostic wrongs are weaker but still
    sometimes useful. Everything else defaults to 0 for the first pass.
    """
    if label == "safe_diagnostic_wrong":
        return 1.0
    if label == "bounded_diagnostic_wrong":
        return 0.5
    return 0.0


def build_option_mass_records(
    active: List[Option],
    probs,
    diag_labels: Optional[Dict[int, str]] = None,
    *,
    last_wrong_index: Optional[int] = None,
    hp_scale: float = 1.0,
) -> List[Dict[str, float]]:
    """Build per-option probability / harm / info records without renormalizing picks.

    `probs` may be a pick-only vector or a full action vector with refresh in the
    final slot. `_clip_pick_probs()` deliberately truncates refresh without
    renormalizing so downstream masses remain unconditional over the real action
    support.
    """
    labels = diag_labels or {}
    raw = _clip_pick_probs(probs, len(active))
    records: List[Dict[str, float]] = []
    max_wrong_prob = max(
        [float(raw[i]) for i, opt in enumerate(active) if not opt.is_correct] or [0.0]
    )
    for i, opt in enumerate(active):
        label = labels.get(opt.index, "")
        harm = _option_badness(
            opt,
            label=label,
            last_wrong_index=last_wrong_index,
            hp_scale=hp_scale,
        )
        info = _option_infovalue(opt, label=label)
        p = float(raw[i])
        records.append({
            "index": opt.index,
            "position": i,
            "prob": p,
            "label": label,
            "risk_class": float(getattr(opt, "risk_class", 0)),
            "is_correct": bool(opt.is_correct),
            "is_highrisk": float(label == "high_risk_lure"),
            "is_safe_diag": float(label == "safe_diagnostic_wrong"),
            "is_bounded_diag": float(label == "bounded_diagnostic_wrong"),
            "is_far_wrong": float(_label_is_far_wrong(label)),
            "is_last_wrong": float(last_wrong_index is not None and opt.index == last_wrong_index),
            "is_top_prob_wrong": float((not opt.is_correct) and abs(p - max_wrong_prob) <= 1e-12),
            "harm": float(harm),
            "info": float(info),
            "removed_harm_mass": float(p * harm),
            "removed_info_mass": float(p * info),
        })
    return records


def summarize_option_mass_records(records: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate HarmMass / InfoMass and common wrong-mass diagnostics."""
    harm_mass = 0.0
    info_mass = 0.0
    expected_damage = 0.0
    p_correct = 0.0
    p_highrisk = 0.0
    p_samewrong = 0.0
    p_farwrong = 0.0
    for rec in records:
        p = float(rec["prob"])
        harm_mass += p * float(rec["harm"])
        info_mass += p * float(rec["info"])
        if rec["is_correct"]:
            p_correct = p
        else:
            expected_damage += p * float(rec["risk_class"])
            p_highrisk += p * float(rec["is_highrisk"])
            p_samewrong += p * float(rec["is_last_wrong"])
            p_farwrong += p * float(rec["is_far_wrong"])
    return {
        "harm_mass": float(harm_mass),
        "info_mass": float(info_mass),
        "expected_damage": float(expected_damage),
        "p_correct": float(p_correct),
        "p_highrisk": float(p_highrisk),
        "p_samewrong": float(p_samewrong),
        "p_farwrong": float(p_farwrong),
    }


def compute_ban_oracle_stats(
    active: List[Option],
    wait_probs,
    ban_probs_by_index: Dict[int, np.ndarray],
    diag_labels: Optional[Dict[int, str]] = None,
    *,
    last_wrong_index: Optional[int] = None,
    hp_scale: float = 1.0,
) -> Dict[str, Any]:
    """Compute removed-harm and net-harm BAN oracles on a common frozen state."""
    labels = diag_labels or {}
    wait_records = build_option_mass_records(
        active,
        wait_probs,
        labels,
        last_wrong_index=last_wrong_index,
        hp_scale=hp_scale,
    )
    wait_summary = summarize_option_mass_records(wait_records)
    wait_record_by_idx = {int(rec["index"]): rec for rec in wait_records}
    per_target: Dict[int, Dict[str, float]] = {}
    removed_oracle_index = None
    removed_oracle_mass = float("-inf")
    net_oracle_index = None
    net_oracle_drop = float("-inf")

    for target_idx, after_probs in ban_probs_by_index.items():
        after_records = build_option_mass_records(
            active,
            after_probs,
            labels,
            last_wrong_index=last_wrong_index,
            hp_scale=hp_scale,
        )
        after_summary = summarize_option_mass_records(after_records)
        wait_rec = wait_record_by_idx.get(int(target_idx))
        if wait_rec is None or bool(wait_rec["is_correct"]):
            continue
        removed_harm_mass = float(wait_rec["removed_harm_mass"])
        removed_prob_mass = float(wait_rec["prob"])
        net_harm_drop = float(wait_summary["harm_mass"] - after_summary["harm_mass"])
        info_delta = float(after_summary["info_mass"] - wait_summary["info_mass"])
        rec = {
            "target_index": float(target_idx),
            "removed_prob_mass": removed_prob_mass,
            "removed_harm_mass": removed_harm_mass,
            "net_harm_drop": net_harm_drop,
            "info_delta": info_delta,
            "target_harm": float(wait_rec["harm"]),
            "target_info": float(wait_rec["info"]),
            "target_prob": float(wait_rec["prob"]),
            "target_label": wait_rec["label"],
            "target_is_highrisk": float(wait_rec["is_highrisk"]),
            "target_is_safe_diag": float(wait_rec["is_safe_diag"]),
            "target_is_bounded_diag": float(wait_rec["is_bounded_diag"]),
            "target_is_far_wrong": float(wait_rec["is_far_wrong"]),
            "target_is_last_wrong": float(wait_rec["is_last_wrong"]),
            "target_is_top_prob_wrong": float(wait_rec["is_top_prob_wrong"]),
            "target_is_correct": float(wait_rec["is_correct"]),
            "after_harm_mass": float(after_summary["harm_mass"]),
            "after_info_mass": float(after_summary["info_mass"]),
        }
        per_target[int(target_idx)] = rec
        if removed_harm_mass > removed_oracle_mass:
            removed_oracle_mass = removed_harm_mass
            removed_oracle_index = int(target_idx)
        if net_harm_drop > net_oracle_drop:
            net_oracle_drop = net_harm_drop
            net_oracle_index = int(target_idx)

    return {
        "wait_summary": wait_summary,
        "wait_records": wait_records,
        "per_target": per_target,
        "removed_oracle_index": removed_oracle_index,
        "removed_oracle_mass": 0.0 if removed_oracle_mass == float("-inf") else float(removed_oracle_mass),
        "net_oracle_index": net_oracle_index,
        "net_oracle_drop": 0.0 if net_oracle_drop == float("-inf") else float(net_oracle_drop),
    }


def _log_correct_margin(active: List[Option], probs, eps: float = 1e-6) -> float:
    raw = _clip_pick_probs(probs, len(active))
    if len(raw) == 0:
        return 0.0

    correct_idx = None
    for i, opt in enumerate(active):
        if opt.is_correct:
            correct_idx = i
            break
    if correct_idx is None:
        return 0.0

    p_correct = float(raw[correct_idx])
    others = [float(raw[i]) for i, _ in enumerate(active) if i != correct_idx]
    top_other = max(others) if others else 0.0
    return float(np.log(p_correct + eps) - np.log(top_other + eps))


def compute_postreveal_cue_terms(
    qs: QueryState,
    active: List[Option],
    wait_probs,
    spec_probs,
    *,
    d_shift: float,
) -> Dict[str, float]:
    """Compute post-reveal cue-value terms on learner pick distributions.

    These terms are designed for post-reveal pedagogical scoring and Q
    decomposition. They intentionally reuse already available quantities
    rather than introducing a separate new model.
    """
    labels = getattr(qs, "option_diag_labels", {})
    wait_arr = _clip_pick_probs(wait_probs, len(active))
    spec_arr = _clip_pick_probs(spec_probs, len(active))

    p_hr_wait = 0.0
    p_hr_spec = 0.0
    p_same_wait = 0.0
    p_same_spec = 0.0
    last_wrong_idx = getattr(qs, "last_reveal_option_index", None)

    for i, opt in enumerate(active):
        label = labels.get(opt.index, "")
        if label == "high_risk_lure":
            p_hr_wait += float(wait_arr[i])
            p_hr_spec += float(spec_arr[i])
        if last_wrong_idx is not None and opt.index == last_wrong_idx:
            p_same_wait = float(wait_arr[i])
            p_same_spec = float(spec_arr[i])

    rank_wait, margin_wait, p_corr_wait = _correct_rank_and_margin(active, wait_arr)
    rank_spec, margin_spec, p_corr_spec = _correct_rank_and_margin(active, spec_arr)

    delta_p = p_corr_spec - p_corr_wait
    highrisk_drop = max(0.0, p_hr_wait - p_hr_spec)
    samewrong_drop = max(0.0, p_same_wait - p_same_spec)
    top1_flip = 1.0 if (rank_wait > 1 and rank_spec == 1) else 0.0
    margin_gain = margin_spec - margin_wait

    beneficial_shift = (
        max(0.0, delta_p)
        + max(0.0, margin_gain)
        + 0.5 * top1_flip
        + highrisk_drop
        + samewrong_drop
    )
    harmful_shift = max(0.0, float(d_shift) - beneficial_shift)

    return {
        "delta_p_correct": round(float(delta_p), 6),
        "p_correct_wait": round(float(p_corr_wait), 6),
        "p_correct_spec": round(float(p_corr_spec), 6),
        "correct_rank_wait": int(rank_wait),
        "correct_rank_spec": int(rank_spec),
        "top1_flip": round(float(top1_flip), 6),
        "correct_margin_wait": round(float(margin_wait), 6),
        "correct_margin_spec": round(float(margin_spec), 6),
        "correct_margin_gain": round(float(margin_gain), 6),
        "p_highrisk_wait": round(float(p_hr_wait), 6),
        "p_highrisk_spec": round(float(p_hr_spec), 6),
        "highrisk_drop": round(float(highrisk_drop), 6),
        "p_samewrong_wait": round(float(p_same_wait), 6),
        "p_samewrong_spec": round(float(p_same_spec), 6),
        "samewrong_drop": round(float(samewrong_drop), 6),
        "beneficial_shift": round(float(beneficial_shift), 6),
        "harmful_shift": round(float(harmful_shift), 6),
    }


def compute_sparse_g_exp(
    cfg: FullConfig,
    qs: QueryState,
    active: List[Option],
    tier_probs: np.ndarray,
    *,
    compute_wait_tier_probs,
    spec: Optional[Dict[str, Any]] = None,
) -> float:
    """Safe exposure gain G_exp for SparseTutor.

    This is a direct extraction of the pedagogical shaping logic from
    SparseTutor._compute_g_exp(). The formulas are intentionally preserved so
    refactors do not silently change exploration incentives.
    """
    lg_mode = getattr(cfg.tutor, "tutor_lg_mode", "off")

    if lg_mode == "safety_only":
        return 0.0

    if len(tier_probs) == 0 or len(tier_probs) != len(active):
        return 0.0

    action = (spec or {}).get("action", "WAIT")
    is_hl_action = action in ("HIGHLIGHT", "MIX")
    hp = qs.hp

    if is_hl_action:
        wait_probs = compute_wait_tier_probs()
        p_correct_wait = 0.0
        safe_wait_total = 0.0
        for i, opt in enumerate(active):
            if opt.is_correct:
                p_correct_wait = float(wait_probs[i])
            elif opt.risk_class < hp:
                safe_wait_total += float(wait_probs[i])

        base_hl = max(0.0, (1.0 - p_correct_wait) * safe_wait_total)

        if lg_mode in ("self_correct", "horizon_self_correct") and getattr(qs, "post_reveal_phase", False):
            g_time = 1.0 if (qs.max_rounds - qs.rounds_used) >= 2 else 0.0
            spec_probs = (spec or {}).get("_spec_probs", None)
            if spec_probs is not None and len(spec_probs) == len(active):
                p_correct_spec = 0.0
                for i, opt in enumerate(active):
                    if opt.is_correct:
                        raw = np.asarray(spec_probs, dtype=float)
                        if len(raw) > len(active):
                            raw = raw[:len(active)]
                        if i < len(raw):
                            p_correct_spec = float(raw[i])
                delta_p_correct = p_correct_spec - p_correct_wait
            else:
                p_correct_tier = 0.0
                if len(tier_probs) == len(active):
                    for i, opt in enumerate(active):
                        if opt.is_correct:
                            p_correct_tier = float(tier_probs[i])
                delta_p_correct = p_correct_tier - p_correct_wait

            b_highlight = max(0.0, delta_p_correct) * g_time
            base_hl += b_highlight * getattr(cfg.learner, "rho_assist", 0.3)

        return base_hl

    total = 0.0
    diag_labels = getattr(qs, "option_diag_labels", {})

    p_safe_diag = 0.0
    p_bounded_diag = 0.0
    p_high_risk = 0.0
    p_correct = 0.0
    e_damage_wait = 0.0

    for i, opt in enumerate(active):
        p_j = float(tier_probs[i])
        if opt.is_correct:
            p_correct = p_j
            continue

        label = diag_labels.get(opt.index, "")
        if label == "safe_diagnostic_wrong":
            p_safe_diag += p_j
        elif label == "bounded_diagnostic_wrong":
            p_bounded_diag += p_j
        elif label == "high_risk_lure":
            p_high_risk += p_j
        e_damage_wait += p_j * opt.risk_class

        if lg_mode == "learning_only":
            total += p_j
            continue

        is_lethal = opt.risk_class >= hp
        if is_lethal:
            continue

        total += p_j

    if lg_mode == "diagnostic":
        lg_bonus = 0.0
        for i, opt in enumerate(active):
            if opt.is_correct:
                continue
            label = diag_labels.get(opt.index, "")
            p_j = float(tier_probs[i])
            if label in ("safe_diagnostic_wrong", "bounded_diagnostic_wrong"):
                lg_bonus += p_j
            elif label == "high_risk_lure":
                lg_bonus -= p_j * 0.5
        total += lg_bonus

    elif lg_mode in ("self_correct", "horizon_self_correct"):
        n_safe_reveals = getattr(qs, "n_safe_diag_wrong_reveals", 0)
        post_reveal = getattr(qs, "post_reveal_phase", False)

        g_survive = max(0.0, (hp - e_damage_wait) / max(1, hp))
        g_time = 1.0 if (qs.max_rounds - qs.rounds_used) >= 2 else 0.0

        if not post_reveal and n_safe_reveals == 0:
            if action == "WAIT":
                b_allow = (p_safe_diag * g_survive * g_time
                           - p_high_risk * e_damage_wait * 0.1)
                total += max(0.0, b_allow)
        elif post_reveal and not qs.success:
            if action == "WAIT":
                repeat_penalty = n_safe_reveals * (1.0 - p_correct) * 0.3
                total -= repeat_penalty
            elif action == "BAN":
                if p_high_risk > p_safe_diag:
                    total += p_high_risk * 0.5

    return max(0.0, total)


# ── Phase 6I-D: horizon self-correct G_exp ───────────────────────────────

def compute_horizon_g_exp(
    cfg: FullConfig,
    qs: QueryState,
    active: List[Option],
    tier_probs: np.ndarray,
    *,
    compute_wait_tier_probs,
    spec: Optional[Dict[str, Any]] = None,
    phase: str = "DEFAULT",
) -> float:
    """G_exp for horizon_self_correct mode.

    Extends compute_sparse_g_exp with trajectory value from
    sparse_tutor_horizon helpers. Falls back to base self_correct
    G_exp and adds horizon terms based on pedagogical phase.
    """
    from .sparse_tutor_horizon import (
        compute_pre_reveal_allow_value,
        compute_post_reveal_cue_value,
    )

    # Base G_exp (reuse self_correct logic)
    base = compute_sparse_g_exp(
        cfg, qs, active, tier_probs,
        compute_wait_tier_probs=compute_wait_tier_probs,
        spec=spec,
    )

    action = (spec or {}).get("action", "WAIT")
    hp = qs.hp
    rounds_left = max(0, qs.max_rounds - qs.rounds_used)
    diag_labels = getattr(qs, "option_diag_labels", {})
    eta_traj = 0.5

    probs_arr = np.asarray(tier_probs, dtype=float)
    p_safe_diag = 0.0
    p_bounded_diag = 0.0
    p_high_risk = 0.0
    p_correct = 0.0
    e_damage = 0.0
    for i, opt in enumerate(active):
        if i >= len(probs_arr):
            break
        p_j = float(probs_arr[i])
        if opt.is_correct:
            p_correct = p_j
            continue
        label = diag_labels.get(opt.index, "")
        if label == "safe_diagnostic_wrong":
            p_safe_diag += p_j
        elif label == "bounded_diagnostic_wrong":
            p_bounded_diag += p_j
        elif label == "high_risk_lure":
            p_high_risk += p_j
        e_damage += p_j * opt.risk_class

    mass_records = build_option_mass_records(
        active,
        tier_probs,
        diag_labels,
        last_wrong_index=getattr(qs, "last_reveal_option_index", None),
        hp_scale=max(qs.hp, 1),
    )
    mass_summary = summarize_option_mass_records(mass_records)
    contrastive_ticket_available = not bool(getattr(qs, "contrastive_update_used", False))
    positive_ticket_available = not bool(getattr(qs, "positive_update_used", False))

    horizon_bonus = 0.0

    if phase == "PRE_REVEAL_ALLOW" and action == "WAIT":
        p_survive = max(0.0, (hp - e_damage) / max(1, hp))
        p_time = 1.0 if rounds_left >= 3 else 0.0  # need 1 wrong + 1 cue + 1 sc
        # best_cue_cate: from oracle or estimate
        # For non-oracle: use a conservative estimate (0.03 = typical BAN CATE)
        best_cue_cate = getattr(cfg.tutor, '_best_cue_cate_estimate', 0.03)
        if getattr(cfg.tutor, 'oracle_horizon', False):
            best_cue_cate = getattr(cfg.tutor, '_oracle_cate', best_cue_cate)

        horizon_bonus = compute_pre_reveal_allow_value(
            p_safe_diag,
            p_bounded_diag,
            p_survive,
            p_time,
            best_cue_cate,
            p_correct,
            harm_mass_wait=float(mass_summary.get("harm_mass", 0.0)),
            contrastive_ticket_available=contrastive_ticket_available,
            positive_ticket_available=positive_ticket_available,
            eta_traj=eta_traj,
        )

    elif phase == "POST_REVEAL_CONSOLIDATE" and action in ("HIGHLIGHT", "MIX"):
        # Use spec's _spec_probs to compute delta_p
        spec_probs = (spec or {}).get("_spec_probs", None)
        wait_probs = compute_wait_tier_probs()
        p_correct_wait = 0.0
        for i, opt in enumerate(active):
            if opt.is_correct:
                p_correct_wait = float(wait_probs[i]) if i < len(wait_probs) else 0.0

        delta_p = 0.0
        if spec_probs is not None:
            raw = np.asarray(spec_probs, dtype=float)
            if len(raw) > len(active):
                raw = raw[:len(active)]  # drop refresh slot
            for i, opt in enumerate(active):
                if opt.is_correct and i < len(raw):
                    delta_p = float(raw[i]) - p_correct_wait
                    break

        horizon_bonus = compute_post_reveal_cue_value(
            delta_p, p_high_risk_after=p_high_risk,
        )

    return base + horizon_bonus


# ── Phase 6I.4 Step C: post-reveal shift decomposition ──────────────

def compute_postreveal_shift_decomp(
    active: List[Option],
    wait_probs,
    action_probs,
    diag_labels: Optional[Dict[int, str]] = None,
    last_wrong_index: Optional[int] = None,
    hp_scale: float = 1.0,
    ban_target_index: Optional[int] = None,
) -> Dict[str, float]:
    """Decompose policy shift into pedagogically meaningful components.

    Args:
        last_wrong_index: index of the specific option that was just revealed
            wrong. If provided, SameWrongDrop tracks only this option.
            Falls back to label-based sum if None.

    Returns dict with:
      delta_p_correct:    P_correct(a) - P_correct(WAIT)
      correct_margin_gain: margin improvement (p_correct - p_max_wrong)
      top1_flip:          1 if correct flips to rank-1, else 0
      high_risk_drop:     max(0, P_hr(WAIT) - P_hr(a))
      same_wrong_drop:    max(0, P_sw(WAIT) - P_sw(a))
      good_shift:         sum of beneficial components
      bad_shift:          sum of harmful components
    """
    labels = diag_labels or {}
    K = len(active)
    w = _clip_pick_probs(wait_probs, K)
    a = _clip_pick_probs(action_probs, K)

    wait_records = build_option_mass_records(
        active,
        w,
        labels,
        last_wrong_index=last_wrong_index,
        hp_scale=hp_scale,
    )
    action_records = build_option_mass_records(
        active,
        a,
        labels,
        last_wrong_index=last_wrong_index,
        hp_scale=hp_scale,
    )
    wait_summary = summarize_option_mass_records(wait_records)
    action_summary = summarize_option_mass_records(action_records)

    p_correct_w = float(wait_summary["p_correct"])
    p_correct_a = float(action_summary["p_correct"])
    p_hr_w = float(wait_summary["p_highrisk"])
    p_hr_a = float(action_summary["p_highrisk"])
    p_sw_w = float(wait_summary["p_samewrong"])
    p_sw_a = float(action_summary["p_samewrong"])
    max_wrong_w = 0.0
    max_wrong_a = 0.0
    harm_mass_w = float(wait_summary["harm_mass"])
    harm_mass_a = float(action_summary["harm_mass"])
    info_mass_w = float(wait_summary["info_mass"])
    info_mass_a = float(action_summary["info_mass"])
    removed_prob_mass = 0.0
    removed_bad_mass = 0.0
    removed_info_mass = 0.0
    removed_high_risk_mass = 0.0
    removed_same_wrong_mass = 0.0

    for i, opt in enumerate(active):
        if i >= K:
            break
        lbl = labels.get(opt.index, "")
        if opt.is_correct:
            pass
        else:
            max_wrong_w = max(max_wrong_w, float(w[i]))
            max_wrong_a = max(max_wrong_a, float(a[i]))

        if ban_target_index is not None and opt.index == ban_target_index:
            removed_prob_mass = float(w[i])
            bad_cost = _option_badness(
                opt,
                label=lbl,
                last_wrong_index=last_wrong_index,
                hp_scale=hp_scale,
            )
            removed_bad_mass = float(w[i]) * bad_cost
            removed_info_mass = float(w[i]) * _option_infovalue(opt, label=lbl)
            if lbl == "high_risk_lure":
                removed_high_risk_mass = float(w[i])
            if last_wrong_index is not None and opt.index == last_wrong_index:
                removed_same_wrong_mass = float(w[i])

    delta_p = p_correct_a - p_correct_w
    margin_w = p_correct_w - max_wrong_w
    margin_a = p_correct_a - max_wrong_a
    margin_gain = margin_a - margin_w
    log_margin_w = _log_correct_margin(active, w)
    log_margin_a = _log_correct_margin(active, a)
    log_margin_gain = log_margin_a - log_margin_w

    # Top1 flip
    rank_w, _, _ = _correct_rank_and_margin(active, w)
    rank_a, _, _ = _correct_rank_and_margin(active, a)
    top1_flip = 1.0 if (rank_w > 1 and rank_a == 1) else 0.0
    correct_rank_gain = max(0.0, float(rank_w - rank_a))

    hr_drop = max(0.0, p_hr_w - p_hr_a)
    sw_drop = max(0.0, p_sw_w - p_sw_a)
    harm_mass_drop = max(0.0, harm_mass_w - harm_mass_a)
    info_mass_delta = info_mass_a - info_mass_w

    good_shift = (max(0.0, delta_p) +
                  max(0.0, log_margin_gain) +
                  harm_mass_drop)

    bad_shift = (max(0.0, -delta_p) +
                 max(0.0, -log_margin_gain) +
                 max(0.0, harm_mass_a - harm_mass_w))

    return {
        "delta_p_correct": delta_p,
        "correct_margin_gain": margin_gain,
        "log_margin_wait": log_margin_w,
        "log_margin_action": log_margin_a,
        "log_margin_gain": log_margin_gain,
        "correct_rank_wait": float(rank_w),
        "correct_rank_action": float(rank_a),
        "correct_rank_gain": correct_rank_gain,
        "top1_flip": top1_flip,
        "high_risk_drop": hr_drop,
        "same_wrong_drop": sw_drop,
        "harm_mass_wait": harm_mass_w,
        "harm_mass_action": harm_mass_a,
        "harm_mass_drop": harm_mass_drop,
        "info_mass_wait": info_mass_w,
        "info_mass_action": info_mass_a,
        "info_mass_delta": info_mass_delta,
        # Backward-compatible aliases: "bad mass" now maps to HarmMass.
        "bad_mass_wait": harm_mass_w,
        "bad_mass_action": harm_mass_a,
        "bad_mass_drop": harm_mass_drop,
        "removed_prob_mass": removed_prob_mass,
        "removed_bad_mass": removed_bad_mass,
        "removed_info_mass": removed_info_mass,
        "removed_high_risk_mass": removed_high_risk_mass,
        "removed_same_wrong_mass": removed_same_wrong_mass,
        "good_shift": good_shift,
        "bad_shift": bad_shift,
        "beneficial_shift": good_shift,
        "harmful_shift": bad_shift,
        "p_correct_wait": p_correct_w,
        "p_correct_action": p_correct_a,
        "p_hr_wait": p_hr_w,
        "p_hr_action": p_hr_a,
        "p_sw_wait": p_sw_w,
        "p_sw_action": p_sw_a,
    }


def compute_outcome_conditioned_grace_conversion(
    active: List[Option],
    action_probs,
    diag_labels: Optional[Dict[int, str]] = None,
    *,
    last_wrong_index: Optional[int] = None,
    rounds_left: int = 0,
    p_terminal: float = 0.0,
    hp_scale: float = 1.0,
) -> float:
    """Approximate grace conversion using immediate outcome taxonomy.

    This is intentionally low-parameter. We treat grace as mainly useful for
    recoverable non-terminal outcomes and avoid giving the same credit to
    repeated-wrong, far-wrong, or clearly high-risk outcomes.
    """
    if rounds_left < 2:
        return 0.0

    labels = diag_labels or {}
    pick_probs = _clip_pick_probs(action_probs, len(active))
    records = build_option_mass_records(
        active,
        pick_probs,
        labels,
        last_wrong_index=last_wrong_index,
        hp_scale=hp_scale,
    )
    summary = summarize_option_mass_records(records)
    p_correct = float(summary["p_correct"])
    p_same = float(summary["p_samewrong"])
    p_highrisk = float(summary["p_highrisk"])
    p_far = float(summary["p_farwrong"])
    p_info = float(summary["info_mass"])

    raw = np.asarray(action_probs, dtype=float)
    refresh_prob = 0.0
    if len(raw) > len(active):
        refresh_prob = max(0.0, float(raw[len(active)]))
    else:
        refresh_prob = max(0.0, 1.0 - float(np.sum(pick_probs)))

    # Recoverable mass keeps productive reveal and refresh available for grace,
    # while excluding terminal / repeated / obviously harmful outcomes.
    recoverable = max(
        0.0,
        min(
            1.0,
            p_info + refresh_prob - p_same - p_highrisk - p_far,
        ),
    )
    nonterminal_not_correct = max(0.0, 1.0 - p_correct - max(0.0, p_terminal))
    return max(0.0, min(recoverable, nonterminal_not_correct) * p_correct)


def compute_postreveal_consolidation_value(
    qs: Optional[QueryState],
    *,
    p_correct_action: float,
    action_name: str = "WAIT",
    incidental_correct_credit: float = 0.5,
) -> Dict[str, Any]:
    """Approximate the value of converting post-reveal progress into correct consolidation.

    This is intentionally low-parameter and reuses the learner-side two-ticket
    semantics instead of introducing a new tuned reward family.
    """
    positive_ticket_available = bool(qs is not None and not getattr(qs, "positive_update_used", False))
    reason = "no_query_state"
    source_weight = 0.0
    pedagogical_context = False

    if qs is None:
        return {
            "consolidation_value": 0.0,
            "positive_ticket_available": False,
            "source_weight": 0.0,
            "pedagogical_context": False,
            "reason": reason,
        }

    if not positive_ticket_available:
        return {
            "consolidation_value": 0.0,
            "positive_ticket_available": False,
            "source_weight": 0.0,
            "pedagogical_context": False,
            "reason": "ticket_spent",
        }

    if bool(getattr(qs, "after_highlight_grace_round", False)):
        pedagogical_context = True
        source_weight = 1.0
        reason = "after_grace"
    elif action_name in ("HIGHLIGHT", "MIX"):
        pedagogical_context = True
        source_weight = 1.0
        reason = "after_cue"
    elif (
        bool(getattr(qs, "post_reveal_phase", False))
        or int(getattr(qs, "n_safe_diag_wrong_reveals", 0)) > 0
        or int(getattr(qs, "n_bounded_diag_wrong_reveals", 0)) > 0
        or bool(getattr(qs, "reveal_history", None))
    ):
        pedagogical_context = True
        source_weight = 1.0
        reason = "after_reveal"
    else:
        source_weight = float(incidental_correct_credit)
        reason = "incidental_correct"

    value = max(0.0, float(p_correct_action)) * max(0.0, float(source_weight))
    return {
        "consolidation_value": value,
        "positive_ticket_available": True,
        "source_weight": float(source_weight),
        "pedagogical_context": bool(pedagogical_context),
        "reason": reason,
    }


def compute_postreveal_q(
    decomp: Dict[str, float],
    *,
    action_name: str = "HIGHLIGHT",
    value_mode: str = "legacy",
    eta_p: float = 1.0,
    eta_m: float = 1.0,
    eta_f: float = 0.5,
    eta_hr: float = 1.0,
    eta_rw: float = 1.0,
    lambda_bad: float = 1.0,
    lambda_info_post: float = 0.0,
    grace_conversion: float = 0.0,
    consolidation_value: float = 0.0,
    overteach_cost: float = 0.0,
    cost: float = 0.0,
) -> float:
    """Compute post-reveal Q for a cue action from shift decomposition.

    Q_post = eta_p * max(0, ΔP_correct)
           + eta_m * max(0, CorrectMarginGain)
           + eta_f * Top1Flip
           + eta_hr * HighRiskDrop
           + eta_rw * SameWrongDrop
           - lambda_bad * BadShift
           - cost
    """
    margin_gain = decomp.get("log_margin_gain")
    if margin_gain is None:
        margin_gain = decomp.get("correct_margin_gain", 0.0)
    harmful_shift = decomp.get("harmful_shift")
    if harmful_shift is None:
        harmful_shift = decomp.get("bad_shift", 0.0)

    if value_mode == "traj_v1":
        q = (
            max(0.0, decomp["delta_p_correct"])
            + max(0.0, margin_gain)
            + max(0.0, decomp.get("bad_mass_drop", 0.0))
            + max(0.0, grace_conversion)
            + max(0.0, consolidation_value)
            - lambda_bad * harmful_shift
            - max(0.0, overteach_cost)
            - cost
        )
        return q

    if value_mode == "traj_v2":
        q = (
            max(0.0, decomp["delta_p_correct"])
            + max(0.0, margin_gain)
            + max(0.0, decomp.get("harm_mass_drop", decomp.get("bad_mass_drop", 0.0)))
            + lambda_info_post * max(0.0, decomp.get("info_mass_delta", 0.0))
            + max(0.0, grace_conversion)
            + max(0.0, consolidation_value)
            - lambda_bad * harmful_shift
            - max(0.0, overteach_cost)
            - cost
        )
        return q

    q = (
        eta_p * max(0.0, decomp["delta_p_correct"])
        + eta_m * max(0.0, decomp["correct_margin_gain"])
        + eta_f * decomp["top1_flip"]
        + eta_hr * decomp["high_risk_drop"]
        + eta_rw * decomp["same_wrong_drop"]
        - lambda_bad * decomp["bad_shift"]
        - cost
    )
    return q
