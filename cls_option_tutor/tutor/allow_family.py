from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import numpy as np

from ..env.state import QueryState
from ..interfaces import Option

FAMILY_NATIVE_LIKE_ALLOW = "NATIVE_LIKE_ALLOW"
FAMILY_MIXED_PROD_HARM = "MIXED_PROD_HARM"
FAMILY_PROTECT_CRITICAL = "PROTECT_CRITICAL"
FAMILY_BORING_MASTERY = "BORING_MASTERY"
FAMILY_NO_PRODUCTIVE_OPPORTUNITY = "NO_PRODUCTIVE_OPPORTUNITY"
FAMILY_ROUND_BLOCKED = "ROUND_BLOCKED"
FAMILY_TICKET_BLOCKED = "TICKET_BLOCKED"
FAMILY_NOT_PREREVEAL = "NOT_PREREVEAL"

PREREVEAL_FAMILY_LABELS = (
    FAMILY_NATIVE_LIKE_ALLOW,
    FAMILY_MIXED_PROD_HARM,
    FAMILY_PROTECT_CRITICAL,
    FAMILY_BORING_MASTERY,
    FAMILY_NO_PRODUCTIVE_OPPORTUNITY,
    FAMILY_ROUND_BLOCKED,
    FAMILY_TICKET_BLOCKED,
    FAMILY_NOT_PREREVEAL,
)


def label_is_far_wrong(label: str) -> bool:
    return label in ("safe_far", "safe_random_wrong", "risky_far")


def build_prereveal_allow_features(
    *,
    post_reveal_phase: bool,
    success: bool,
    hp: float,
    n_safe_diag_wrong_reveals: int,
    both_tickets_available: bool,
    rounds_left: int,
    p_safe_diag: float,
    p_bounded_diag: float,
    p_farwrong: float,
    p_highrisk: float,
    p_correct_wait: float,
    harm_mass: float,
    expected_damage_wait: float = 0.0,
    productive_mass: Optional[float] = None,
) -> Dict[str, float | bool | int]:
    productive_mass_val = (
        float(productive_mass)
        if productive_mass is not None
        else max(0.0, float(p_safe_diag)) + 0.5 * max(0.0, float(p_bounded_diag))
    )
    safe_diag_quality_gap = max(0.0, float(p_safe_diag)) - (
        max(0.0, float(p_farwrong)) + max(0.0, float(p_highrisk))
    )
    pre_reveal = bool((not post_reveal_phase) and (not success) and float(hp) > 0.0)
    return {
        "pre_reveal": pre_reveal,
        "post_reveal_phase": bool(post_reveal_phase),
        "success": bool(success),
        "hp": float(hp),
        "n_safe_diag_wrong_reveals": int(n_safe_diag_wrong_reveals),
        "both_tickets_available": bool(both_tickets_available),
        "rounds_left": int(rounds_left),
        "p_safe_diag": max(0.0, float(p_safe_diag)),
        "p_bounded_diag": max(0.0, float(p_bounded_diag)),
        "p_farwrong": max(0.0, float(p_farwrong)),
        "p_highrisk": max(0.0, float(p_highrisk)),
        "p_correct_wait": max(0.0, float(p_correct_wait)),
        "productive_mass": max(0.0, productive_mass_val),
        "harm_mass": max(0.0, float(harm_mass)),
        "expected_damage_wait": max(0.0, float(expected_damage_wait)),
        "safe_diag_quality_gap": float(safe_diag_quality_gap),
        "harm_competition_gap": float(productive_mass_val - float(harm_mass)),
    }


def compute_prereveal_allow_features_from_probs(
    qs: QueryState,
    active: List[Option],
    probs,
) -> Dict[str, float | bool | int]:
    probs_arr = np.asarray(probs, dtype=float) if probs is not None else np.array([])
    diag_labels = getattr(qs, "option_diag_labels", {}) or {}
    last_wrong_index = getattr(qs, "last_reveal_option_index", None)

    p_safe_diag = 0.0
    p_bounded_diag = 0.0
    p_highrisk = 0.0
    p_farwrong = 0.0
    p_correct_wait = 0.0
    harm_mass = 0.0
    expected_damage_wait = 0.0

    for i, opt in enumerate(active):
        if i >= len(probs_arr):
            break
        p_j = float(probs_arr[i])
        if opt.is_correct:
            p_correct_wait = p_j
            continue

        label = str(diag_labels.get(opt.index, ""))
        if label == "safe_diagnostic_wrong":
            p_safe_diag += p_j
        elif label == "bounded_diagnostic_wrong":
            p_bounded_diag += p_j
        elif label == "high_risk_lure":
            p_highrisk += p_j
        elif label_is_far_wrong(label):
            p_farwrong += p_j

        harm_j = (
            float(getattr(opt, "risk_class", 0)) / 4.0
            + (
                1.0
                if last_wrong_index is not None
                and int(opt.index) == int(last_wrong_index)
                else 0.0
            )
            + (1.0 if label_is_far_wrong(label) else 0.0)
        )
        harm_mass += p_j * harm_j
        expected_damage_wait += p_j * float(getattr(opt, "risk_class", 0))

    return build_prereveal_allow_features(
        post_reveal_phase=bool(getattr(qs, "post_reveal_phase", False)),
        success=bool(getattr(qs, "success", False)),
        hp=float(getattr(qs, "hp", 0)),
        n_safe_diag_wrong_reveals=int(getattr(qs, "n_safe_diag_wrong_reveals", 0)),
        both_tickets_available=bool(
            not bool(getattr(qs, "contrastive_update_used", False))
            and not bool(getattr(qs, "positive_update_used", False))
        ),
        rounds_left=max(0, int(getattr(qs, "max_rounds", 0)) - int(getattr(qs, "rounds_used", 0))),
        p_safe_diag=p_safe_diag,
        p_bounded_diag=p_bounded_diag,
        p_farwrong=p_farwrong,
        p_highrisk=p_highrisk,
        p_correct_wait=p_correct_wait,
        harm_mass=harm_mass,
        expected_damage_wait=expected_damage_wait,
    )


def compute_prereveal_allow_features_from_trace(
    entry: Mapping[str, object],
) -> Dict[str, float | bool | int]:
    p_safe_diag = float(entry.get("pre_p_safe_diag_wait", 0.0) or 0.0)
    p_bounded_diag = float(entry.get("pre_p_bounded_diag_wait", 0.0) or 0.0)
    p_farwrong = float(entry.get("pre_p_farwrong_wait", 0.0) or 0.0)
    p_highrisk = float(
        entry.get("pre_p_highrisk_wait", entry.get("pre_p_high_risk_wait", 0.0)) or 0.0
    )
    productive_mass = entry.get("pre_productive_mass_wait", None)
    if productive_mass is None:
        productive_mass = p_safe_diag + 0.5 * p_bounded_diag
    return build_prereveal_allow_features(
        post_reveal_phase=bool(entry.get("pre_post_reveal_phase", False)),
        success=bool(entry.get("pre_success", entry.get("success", False))),
        hp=float(entry.get("pre_hp", entry.get("hp", 2.0)) or 0.0),
        n_safe_diag_wrong_reveals=int(entry.get("pre_n_safe_diag_wrong_reveals", 0) or 0),
        both_tickets_available=bool(entry.get("pre_both_tickets_available", False)),
        rounds_left=int(entry.get("pre_rounds_left", 0) or 0),
        p_safe_diag=p_safe_diag,
        p_bounded_diag=p_bounded_diag,
        p_farwrong=p_farwrong,
        p_highrisk=p_highrisk,
        p_correct_wait=float(entry.get("p_correct_wait", 0.0) or 0.0),
        harm_mass=float(entry.get("pre_harm_mass_wait", 0.0) or 0.0),
        expected_damage_wait=float(entry.get("pre_expected_damage_wait", 0.0) or 0.0),
        productive_mass=float(productive_mass or 0.0),
    )


def is_native_phase_allow_candidate(features: Mapping[str, object]) -> bool:
    return bool(
        bool(features.get("pre_reveal", False))
        and not bool(features.get("success", False))
        and int(features.get("n_safe_diag_wrong_reveals", 0) or 0) == 0
        and float(features.get("p_safe_diag", 0.0) or 0.0) > 0.25
        and float(features.get("p_highrisk", 0.0) or 0.0) <= 0.25
        and int(features.get("rounds_left", 0) or 0) >= 2
        and float(features.get("hp", 0.0) or 0.0) > 1.0
    )


def is_phasecalib_allow_candidate(features: Mapping[str, object]) -> bool:
    p_safe_diag = float(features.get("p_safe_diag", 0.0) or 0.0)
    p_bounded_diag = float(features.get("p_bounded_diag", 0.0) or 0.0)
    p_highrisk = float(features.get("p_highrisk", 0.0) or 0.0)
    p_farwrong = float(features.get("p_farwrong", 0.0) or 0.0)
    p_correct_wait = float(features.get("p_correct_wait", 0.0) or 0.0)
    productive_mass = float(features.get("productive_mass", 0.0) or 0.0)
    harm_mass = float(features.get("harm_mass", 0.0) or 0.0)

    return bool(
        bool(features.get("pre_reveal", False))
        and int(features.get("n_safe_diag_wrong_reveals", 0) or 0) == 0
        and not bool(features.get("success", False))
        and float(features.get("hp", 0.0) or 0.0) > 1.0
        and bool(features.get("both_tickets_available", False))
        and int(features.get("rounds_left", 0) or 0) >= 3
        and productive_mass > 0.0
        and productive_mass >= (0.5 * harm_mass)
        and p_highrisk <= (p_safe_diag + p_bounded_diag)
        and p_farwrong <= (p_safe_diag + p_bounded_diag)
        and p_correct_wait < 0.75
    )


def classify_prereveal_family(features: Mapping[str, object]) -> str:
    p_safe_diag = float(features.get("p_safe_diag", 0.0) or 0.0)
    p_bounded_diag = float(features.get("p_bounded_diag", 0.0) or 0.0)
    p_highrisk = float(features.get("p_highrisk", 0.0) or 0.0)
    p_correct_wait = float(features.get("p_correct_wait", 0.0) or 0.0)
    productive_mass = float(features.get("productive_mass", 0.0) or 0.0)
    harm_mass = float(features.get("harm_mass", 0.0) or 0.0)
    safe_diag_quality_gap = float(features.get("safe_diag_quality_gap", 0.0) or 0.0)

    if not bool(features.get("pre_reveal", False)):
        return FAMILY_NOT_PREREVEAL
    if not bool(features.get("both_tickets_available", False)):
        return FAMILY_TICKET_BLOCKED
    if int(features.get("rounds_left", 0) or 0) < 3:
        return FAMILY_ROUND_BLOCKED
    if p_correct_wait >= 0.75:
        return FAMILY_BORING_MASTERY
    if productive_mass <= 0.0:
        return FAMILY_NO_PRODUCTIVE_OPPORTUNITY
    if p_highrisk > (p_safe_diag + p_bounded_diag):
        return FAMILY_PROTECT_CRITICAL
    if safe_diag_quality_gap > 0.0 and harm_mass <= productive_mass:
        return FAMILY_NATIVE_LIKE_ALLOW
    if productive_mass > 0.0:
        return FAMILY_MIXED_PROD_HARM
    return FAMILY_NO_PRODUCTIVE_OPPORTUNITY


def is_native_like_allow(features: Mapping[str, object]) -> bool:
    return classify_prereveal_family(features) == FAMILY_NATIVE_LIKE_ALLOW
