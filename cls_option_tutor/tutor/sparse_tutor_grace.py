"""Grace-round helpers for SparseTutorAgent.

Phase 6I.4: Grace is a hard-priority phase, not a soft preference.
After a cue (HIGHLIGHT/MIX), the next tutor decision should consume
grace as WAIT unless a true hard override is required.
"""

from __future__ import annotations


def ensure_grace_metrics(block) -> dict:
    """Return the block-level grace metric dict, creating it if needed."""
    grace_metrics = getattr(block, "_grace_metrics", None)
    if grace_metrics is None:
        block._grace_metrics = _default_grace_dict()
        grace_metrics = block._grace_metrics
    else:
        for k, v in _default_grace_dict().items():
            grace_metrics.setdefault(k, v)
    return grace_metrics


def _default_grace_dict() -> dict:
    return {
        "set": 0,
        "eligible_next_round": 0,
        "next_tutor_called": 0,
        "count": 0,
        "chosen_wait": 0,
        "chosen_override": 0,
        "consumed": 0,
        "override": 0,
        "blocked_by_protect": 0,
        "blocked_by_deadline": 0,
        "did_not_reach_tutor_decision": 0,
        "lost_query_succeeded": 0,
        "lost_wrong_terminal": 0,
        "lost_max_round": 0,
        "flag_reset_without_consumption": 0,
    }


def handle_grace_round(block, qs) -> dict:
    """Consume or override a pending post-highlight grace round.

    Phase 6I.4: Returns a structured dict with reason codes instead of
    a bare string, enabling richer trace and metric accounting.

    Returns dict with keys:
        status:   "none" | "wait" | "override"
        reason:   human-readable reason string
        protect_override:  bool
        deadline_override: bool
    """
    result = {
        "status": "none",
        "reason": "no_grace_flag",
        "protect_override": False,
        "deadline_override": False,
    }

    if not getattr(qs, "after_highlight_grace_round", False):
        return result

    grace_metrics = ensure_grace_metrics(block)
    grace_metrics["next_tutor_called"] += 1

    # ── Hard override conditions ──
    # HP critical: hp <= 1 means next wrong pick is lethal
    hp_critical = (qs.hp <= 1)

    # Phase 6I.4 fix: narrow deadline override.
    # rounds_used == max_rounds - 1 means there is still ONE learner action left
    # after this tutor turn. A grace WAIT here is legal and important.
    # True terminal = rounds_used >= max_rounds (no more learner actions possible).
    # This is intentionally very conservative.
    rounds_left = max(0, qs.max_rounds - qs.rounds_used)
    deadline_critical = (rounds_left <= 0)  # was: rounds_left <= 1

    hard_override = (hp_critical or deadline_critical)

    grace_metrics["count"] += 1
    if not hard_override:
        grace_metrics["chosen_wait"] += 1
        grace_metrics["consumed"] += 1
        qs.after_highlight_grace_round = False
        result["status"] = "wait"
        result["reason"] = "grace_consumed"
        return result

    grace_metrics["chosen_override"] += 1
    grace_metrics["override"] += 1
    if hp_critical:
        grace_metrics["blocked_by_protect"] += 1
        result["protect_override"] = True
        result["reason"] = "hp_critical"
    if deadline_critical:
        grace_metrics["blocked_by_deadline"] += 1
        result["deadline_override"] = True
        result["reason"] = "true_terminal_deadline"
    # One grace opportunity should not persist forever after being overridden.
    qs.after_highlight_grace_round = False
    result["status"] = "override"
    return result
