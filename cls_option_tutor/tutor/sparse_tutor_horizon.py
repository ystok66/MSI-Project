"""Phase 6I-D: horizon-2/3 pedagogical value helpers.

Provides simplified trajectory-value computations for the
horizon_self_correct tutor_lg_mode, keeping rollout logic
out of sparse_tutor.py.

Two main functions:
  - compute_pre_reveal_allow_value: WAIT value in PRE_REVEAL_ALLOW phase
  - compute_post_reveal_cue_value: HIGHLIGHT/MIX value in POST_REVEAL_CONSOLIDATE

Parameters are fixed constants (not sweep targets).
"""
from __future__ import annotations


def compute_pre_reveal_allow_value(
    p_safe_diag: float,
    p_bounded_diag: float,
    p_survive_after_wrong: float,
    p_time_for_sc: float,
    best_cue_cate: float,
    p_correct_wait: float,
    *,
    harm_mass_wait: float = 0.0,
    contrastive_ticket_available: bool = True,
    positive_ticket_available: bool = True,
    eta_traj: float = 0.5,
) -> float:
    """Trajectory value of WAIT in PRE_REVEAL_ALLOW phase.

    Estimates the expected pedagogical gain from allowing a safe
    diagnostic wrong pick now, then cueing self-correction later.

    V_traj(WAIT) = eta_traj
                 * P_productive
                 * P_survive_after_wrong
                 * P_time_for_sc
                 * max(0, best_cue_CATE)
                 * L_local
                 * (1 - HarmMass_WAIT)

    where L_local = 1 - P_correct_wait (higher when learner is uncertain).

    Args:
        p_safe_diag: Probability mass on safe diagnostic wrong options.
        p_bounded_diag: Probability mass on bounded diagnostic wrong options.
        p_survive_after_wrong: Probability of surviving damage after wrong pick.
        p_time_for_sc: 1.0 if rounds_left >= 2 after wrong, else 0.0.
        best_cue_cate: max(0, best HIGHLIGHT/MIX delta P from CATE map).
        p_correct_wait: Current P(correct) under WAIT.
        harm_mass_wait: Expected harmful mass under WAIT. This is a separate
            trajectory-value discount and should not be reused as the physical
            survival probability input.
        contrastive_ticket_available: Whether the contrastive ticket is still available.
        positive_ticket_available: Whether the positive ticket is still available.
        eta_traj: Trajectory value discount (default 0.5).
    """
    if not contrastive_ticket_available or not positive_ticket_available:
        return 0.0
    if p_time_for_sc <= 0.0:
        return 0.0

    p_productive = max(0.0, p_safe_diag) + 0.5 * max(0.0, p_bounded_diag)
    l_local = max(0.0, 1.0 - p_correct_wait)
    raw_v = (p_productive
             * p_survive_after_wrong
             * p_time_for_sc
             * max(0.0, best_cue_cate)
             * l_local)
    harm_discount = max(0.0, 1.0 - max(0.0, harm_mass_wait))
    return eta_traj * raw_v * harm_discount


def compute_post_reveal_cue_value(
    delta_p_correct_full: float,
    p_high_risk_after: float = 0.0,
    p_repeat_wrong_after: float = 0.0,
    *,
    c_hr: float = 0.2,
    c_rw: float = 0.1,
) -> float:
    """Trajectory value of HIGHLIGHT/MIX in POST_REVEAL_CONSOLIDATE phase.

    V_traj(cue) = max(0, delta_p_correct)
                - c_hr * P_high_risk_after
                - c_rw * P_repeat_wrong_after

    Args:
        delta_p_correct_full: DeltaP(correct) from causal audit.
        p_high_risk_after: Probability mass on high-risk options after action.
        p_repeat_wrong_after: Probability of repeating the revealed wrong.
        c_hr: Penalty weight for high-risk exposure (default 0.2).
        c_rw: Penalty weight for repeat-wrong tendency (default 0.1).
    """
    v = max(0.0, delta_p_correct_full)
    v -= c_hr * p_high_risk_after
    v -= c_rw * p_repeat_wrong_after
    return max(0.0, v)
