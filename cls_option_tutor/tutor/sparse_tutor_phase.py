from __future__ import annotations

import copy
from typing import Callable, List, Optional, Tuple

from ..config import FullConfig
from ..env.state import BlockState, QueryState
from ..interfaces import Option
from ..learner.learner_agent import LearnerAgent
from .allow_family import (
    compute_prereveal_allow_features_from_probs,
    is_phasecalib_allow_candidate,
)
from .sparse_tutor_grace import ensure_grace_metrics


def maybe_record_audit_candidate(
    block: BlockState,
    qs: QueryState,
    active: List[Option],
    learner: LearnerAgent,
    select_highlight_cells: Callable[[QueryState, List[Option], LearnerAgent], Optional[Tuple[int, ...]]],
) -> None:
    """Store a decision-time audit snapshot for post-reveal causal analysis."""
    n_safe_reveals = getattr(qs, "n_safe_diag_wrong_reveals", 0)
    post_reveal = getattr(qs, "post_reveal_phase", False)
    rounds_left = max(0, qs.max_rounds - qs.rounds_used)
    is_audit_eligible = (
        post_reveal
        and n_safe_reveals >= 1
        and not qs.success
        and rounds_left >= 1
        and qs.hp > 0
    )
    if not is_audit_eligible:
        return

    if not hasattr(block, "_audit_candidates"):
        block._audit_candidates = []

    hl_diag = None
    try:
        selected = select_highlight_cells(qs, active, learner)
        if selected is not None:
            hl_diag = tuple(selected)
    except Exception:
        hl_diag = None

    hl_cf = None
    cfg_max_cells = None
    try:
        from .highlight_selection import select_counterfactual_highlight_cells

        max_cells = getattr(getattr(block, "profile_state", None), "max_highlight_cells", None)
        try:
            cfg_max_cells = getattr(getattr(learner, "cfg", None), "tutor", None)
            cfg_max_cells = getattr(cfg_max_cells, "max_highlight_cells", None)
        except Exception:
            cfg_max_cells = None
        cf_cells = select_counterfactual_highlight_cells(
            qs,
            active,
            learner,
            max_cells=int(cfg_max_cells or max_cells or 2),
            m_candidates=4,
        )
        if cf_cells is not None:
            hl_cf = tuple(cf_cells)
    except Exception:
        hl_cf = None

    target_len = len(qs.target_output) if qs.target_output else 0
    fixed_k = min(int(cfg_max_cells or 2), target_len)
    hl_fixed = tuple(range(fixed_k))
    block._audit_candidates.append({
        "query_id": qs.query_id,
        "round_t": qs.rounds_used,
        "learner_snapshot": copy.deepcopy(learner),
        "qs_snapshot": copy.deepcopy(qs),
        "active": [copy.deepcopy(o) for o in active],
        "labels": dict(getattr(qs, "option_diag_labels", {})),
        "hl_cells_diagnostic": hl_diag,
        "hl_cells_fixed": hl_fixed,
        "hl_cells_counterfactual": hl_cf,
    })


def update_post_action_phase_flags(
    block: BlockState,
    cfg: FullConfig,
    qs: QueryState,
    action: str,
) -> None:
    """Update query-level post-intervention phase flags."""
    if action in ("HIGHLIGHT", "MIX"):
        lg_mode = getattr(cfg.tutor, "tutor_lg_mode", "off")
        if lg_mode in ("self_correct", "horizon_self_correct") and getattr(qs, "post_reveal_phase", False):
            grace_metrics = ensure_grace_metrics(block)
            grace_metrics["set"] += 1
            rounds_left = max(0, qs.max_rounds - qs.rounds_used)
            if rounds_left >= 2:
                grace_metrics["eligible_next_round"] += 1
            qs.after_highlight_grace_round = True
        return

    if action in ("WAIT", "BAN") and getattr(qs, "after_highlight_grace_round", False):
        grace_metrics = ensure_grace_metrics(block)
        grace_metrics["flag_reset_without_consumption"] += 1
        qs.after_highlight_grace_round = False


PHASE_PRE_REVEAL_ALLOW = "PRE_REVEAL_ALLOW"
PHASE_POST_REVEAL_CONSOLIDATE = "POST_REVEAL_CONSOLIDATE"
PHASE_POST_REVEAL_PROTECT_AND_CUE = "POST_REVEAL_PROTECT_AND_CUE"
PHASE_PROTECT = "PROTECT"
PHASE_BORING_ESCAPE = "BORING_ESCAPE"
PHASE_DEFAULT = "DEFAULT"

_P_SD_THRESH = 0.25
_P_HR_THRESH = 0.25


def infer_pedagogical_phase(
    qs: QueryState,
    active: List[Option],
    tier_probs,
    cfg: FullConfig,
) -> str:
    """Infer the current query-level pedagogical phase."""
    lg_mode = getattr(cfg.tutor, "tutor_lg_mode", "off")
    if lg_mode not in ("self_correct", "horizon_self_correct"):
        return PHASE_DEFAULT

    n_safe_reveals = getattr(qs, "n_safe_diag_wrong_reveals", 0)
    post_reveal = getattr(qs, "post_reveal_phase", False)
    hp = qs.hp
    features = compute_prereveal_allow_features_from_probs(qs, active, tier_probs)
    rounds_left = int(features["rounds_left"])
    p_safe_diag = float(features["p_safe_diag"])
    p_high_risk = float(features["p_highrisk"])
    p_correct = float(features["p_correct_wait"])

    if post_reveal and n_safe_reveals >= 1 and not qs.success and rounds_left >= 1:
        if p_high_risk > _P_HR_THRESH:
            return PHASE_POST_REVEAL_PROTECT_AND_CUE
        return PHASE_POST_REVEAL_CONSOLIDATE

    if p_high_risk > _P_HR_THRESH:
        return PHASE_PROTECT

    if (
        getattr(cfg.tutor, "phase_allow_family_override", False)
        and is_phasecalib_allow_candidate(features)
    ):
        return PHASE_PRE_REVEAL_ALLOW

    if (
        not post_reveal
        and n_safe_reveals == 0
        and p_safe_diag > _P_SD_THRESH
        and p_high_risk <= _P_HR_THRESH
        and rounds_left >= 2
        and hp > 1
    ):
        return PHASE_PRE_REVEAL_ALLOW

    if p_correct < 0.15 and rounds_left <= 1:
        return PHASE_BORING_ESCAPE

    return PHASE_DEFAULT


def phase_action_prior(phase: str) -> dict:
    """Return additive action priors for the current pedagogical phase."""
    if phase == PHASE_PRE_REVEAL_ALLOW:
        return {"WAIT": +0.05, "BAN": -0.03, "HIGHLIGHT": -0.02, "MIX": -0.02}
    if phase == PHASE_POST_REVEAL_CONSOLIDATE:
        return {"WAIT": -0.02, "BAN": +0.01, "HIGHLIGHT": +0.05, "MIX": +0.06}
    if phase == PHASE_POST_REVEAL_PROTECT_AND_CUE:
        return {"WAIT": -0.03, "BAN": +0.03, "HIGHLIGHT": +0.03, "MIX": +0.09}
    if phase == PHASE_PROTECT:
        return {"WAIT": -0.01, "BAN": +0.08, "HIGHLIGHT": 0.0, "MIX": +0.04}
    if phase == PHASE_BORING_ESCAPE:
        return {"WAIT": -0.03, "BAN": +0.02, "HIGHLIGHT": +0.01, "MIX": +0.02}
    return {}
