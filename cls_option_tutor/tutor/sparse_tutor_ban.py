"""BAN pool helpers for SparseTutorAgent."""

from __future__ import annotations


def is_hard_protected_safe_diag(qs, opt, diag_labels) -> bool:
    """Return whether an option is the first protectable safe diagnostic wrong."""
    label = diag_labels.get(opt.index, "")
    if label != "safe_diagnostic_wrong":
        return False
    if getattr(qs, "n_safe_diag_wrong_reveals", 0) > 0:
        return False

    hp_after = qs.hp - getattr(opt, "risk_class", 0)
    if hp_after <= 0:
        return False

    rounds_left = qs.max_rounds - qs.rounds_used
    rounds_after = rounds_left - 1
    if rounds_after < 1:
        return False

    return True


def build_learning_ban_pool(qs, non_correct, diag_labels, lg_mode: str, hard_guard_enabled: bool):
    """Return prioritized lure list and candidate pool for learning-mode BAN.

    The returned tuple is `(lures, pool)` where:
    - `lures` are preferred HIGH_RISK_LURE candidates
    - `pool` is the fallback pool for confusion-based selection
    """
    if lg_mode in ("self_correct", "horizon_self_correct") and hard_guard_enabled:
        hard_protected_indices = {
            o.index for o in non_correct if is_hard_protected_safe_diag(qs, o, diag_labels)
        }
        lures = [o for o in non_correct if diag_labels.get(o.index, "") == "high_risk_lure"]
        non_diag = [
            o
            for o in non_correct
            if diag_labels.get(o.index, "") not in {"safe_diagnostic_wrong", "bounded_diagnostic_wrong"}
        ]
        if non_diag:
            pool = non_diag
        else:
            pool = [o for o in non_correct if o.index not in hard_protected_indices]
            if not pool:
                pool = non_correct
        return lures, pool

    if lg_mode in ("self_correct", "horizon_self_correct", "diagnostic"):
        protected = {"safe_diagnostic_wrong", "bounded_diagnostic_wrong"}
        non_protected = [o for o in non_correct if diag_labels.get(o.index, "") not in protected]
        lures = [o for o in non_protected if diag_labels.get(o.index, "") == "high_risk_lure"]
        pool = non_protected if non_protected else non_correct
        return lures, pool

    return [], non_correct
