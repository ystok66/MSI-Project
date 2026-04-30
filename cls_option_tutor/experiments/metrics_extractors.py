"""Block-level metric extraction helpers for learning-increment experiments."""

from __future__ import annotations

from collections import defaultdict

from ..tutor.allow_family import (
    FAMILY_MIXED_PROD_HARM,
    FAMILY_NATIVE_LIKE_ALLOW,
    PREREVEAL_FAMILY_LABELS,
    classify_prereveal_family,
    compute_prereveal_allow_features_from_trace,
    is_native_phase_allow_candidate,
    is_phasecalib_allow_candidate,
)


def _build_decision_trace_lookup(block):
    """Index decision trace entries by (query_id, round_t)."""
    trace = getattr(block, "_decision_trace", None) or []
    lookup = {}
    for entry in trace:
        key = (entry.get("query_id"), entry.get("round_t"))
        if key[0] is None or key[1] is None:
            continue
        lookup[key] = entry
    return lookup


def _next_pick_after_round(learner_steps, round_t):
    """Return the first learner pick at or after a tutor round.

    Tutor and learner actions in the same round share the same `round_t`,
    with tutor acting first. So post-cue immediate outcomes must be read
    using `>=`, not `>`.
    """
    for ls in learner_steps:
        if ls.action == "pick" and getattr(ls, "round_t", -1) >= round_t:
            return ls
    return None


def _first_learner_step_at_or_after_round(learner_steps, round_t):
    """Return the first learner step at or after a tutor round."""
    for ls in learner_steps:
        if getattr(ls, "round_t", -1) >= round_t:
            return ls
    return None


def _learner_steps_within_round_window(learner_steps, start_round_t, horizon_rounds=2):
    """Return learner steps in [start_round_t, start_round_t + horizon_rounds - 1]."""
    end_round_t = start_round_t + max(horizon_rounds - 1, 0)
    return [
        ls for ls in learner_steps
        if start_round_t <= getattr(ls, "round_t", -1) <= end_round_t
    ]


def _label_is_far_wrong(label: str) -> bool:
    return label in ("safe_far", "safe_random_wrong", "risky_far")


def _classify_postcue_wrong(qs, pick_index, last_wrong_index):
    """Classify an immediate wrong pick after cue."""
    if pick_index is None:
        return "OtherWrong"

    labels = getattr(qs, "option_diag_labels", {}) or {}
    label = labels.get(pick_index, "")

    if last_wrong_index is not None and pick_index == last_wrong_index:
        return "SameWrong"
    if label == "high_risk_lure":
        return "HighRisk"
    if label == "bounded_diagnostic_wrong":
        return "BoundedDiag"
    if label == "safe_diagnostic_wrong":
        return "DifferentSafeDiag"
    if _label_is_far_wrong(label):
        return "FarWrong"
    return "OtherWrong"


def _best_q_for_action(candidates, action_name):
    vals = [
        float(c.get("q_use", float("-inf")))
        for c in (candidates or [])
        if c.get("action") == action_name
    ]
    return max(vals) if vals else None


def _best_detail_for_action(candidates, action_name):
    matches = [c for c in (candidates or []) if c.get("action") == action_name]
    if not matches:
        return None
    return max(matches, key=lambda c: float(c.get("q_use", float("-inf"))))


def _detail_metric(detail, key, default=0.0):
    if not detail:
        return default
    if key in detail:
        return detail.get(key, default)
    nested = detail.get("postreveal_decomp", {}) or {}
    return nested.get(key, default)


def _fallback_opp_flags(qs):
    labels = getattr(qs, "option_diag_labels", {})
    return (
        any(v == "safe_diagnostic_wrong" for v in labels.values()),
        any(v == "high_risk_lure" for v in labels.values()),
        getattr(qs, "post_reveal_phase", False),
    )


def _candidate_identity(cand):
    return (
        cand.get("action", "WAIT"),
        cand.get("ban_index"),
        tuple(cand.get("highlight_cells") or ()),
    )


def _chosen_candidate_detail(entry, candidates):
    chosen_action = entry.get("chosen_action")
    chosen_ban = entry.get("chosen_ban_index")
    chosen_cells = tuple(entry.get("chosen_highlight_cells") or ())
    exact = [
        cand for cand in candidates
        if cand.get("action") == chosen_action
        and cand.get("ban_index") == chosen_ban
        and tuple(cand.get("highlight_cells") or ()) == chosen_cells
    ]
    if exact:
        return max(exact, key=lambda cand: float(cand.get("q_use", float("-inf"))))
    same_action = [cand for cand in candidates if cand.get("action") == chosen_action]
    if same_action:
        return max(same_action, key=lambda cand: float(cand.get("q_use", float("-inf"))))
    return None


def build_learning_loop_ledger(block):
    """Build a query-level ledger for the productive self-correct loop.

    The ledger intentionally stays at query granularity so later audit code can
    answer "where does the loop break?" without re-deriving the state machine
    from aggregate rates.
    """
    obs_q = block.obs_phase_queries
    teach_q = block.teach_phase_queries
    teach_queries = block.queries[obs_q: obs_q + teach_q]
    query_by_id = {
        int(getattr(qs, "query_id", qi)): qs
        for qi, qs in enumerate(teach_queries)
    }

    decision_trace = getattr(block, "_decision_trace", None) or []
    learner_trace = getattr(block, "learner_trace", None) or []
    decisions_by_qid = defaultdict(list)
    learner_by_qid = defaultdict(list)

    for entry in decision_trace:
        qid = entry.get("query_id")
        if qid in query_by_id:
            decisions_by_qid[int(qid)].append(entry)
    for qid in decisions_by_qid:
        decisions_by_qid[qid] = sorted(
            decisions_by_qid[qid], key=lambda e: int(e.get("round_t", 10**9))
        )

    for step in learner_trace:
        qid = getattr(step, "query_id", None)
        if qid in query_by_id:
            learner_by_qid[int(qid)].append(step)
    for qid in learner_by_qid:
        learner_by_qid[qid] = sorted(
            learner_by_qid[qid], key=lambda s: int(getattr(s, "round_t", 10**9))
        )

    ledger = []
    for qid in sorted(query_by_id):
        qs = query_by_id[qid]
        decisions = decisions_by_qid.get(qid, [])
        steps = learner_by_qid.get(qid, [])
        first_decision = decisions[0] if decisions else {}
        allow_entry = next(
            (d for d in decisions if d.get("wait_reason") == "WAIT_ALLOW_SAFE_DIAG"),
            None,
        )
        productive_reveal_step = next(
            (
                ls for ls in steps
                if getattr(ls, "raw_feedback_kind", "none") == "wrong_reveal"
                and getattr(ls, "feedback_category", "none") in ("safe_diag", "bounded_diag")
            ),
            None,
        )
        contrastive_step = next(
            (
                ls for ls in steps
                if bool(getattr(ls, "contrastive_ticket_consumed", False))
            ),
            None,
        )
        postreveal_cue_entry = next(
            (
                d for d in decisions
                if bool(d.get("pre_post_reveal_phase", False))
                and d.get("chosen_action") in ("HIGHLIGHT", "MIX")
            ),
            None,
        )
        grace_entry = next(
            (d for d in decisions if d.get("wait_reason") == "WAIT_GRACE"),
            None,
        )
        correct_after_feedback_step = next(
            (
                ls for ls in steps
                if getattr(ls, "raw_feedback_kind", "none") == "correct_pick"
                and getattr(ls, "semantic_credit_reason", "none")
                in ("after_reveal", "after_cue", "after_grace")
            ),
            None,
        )
        positive_step = next(
            (
                ls for ls in steps
                if bool(getattr(ls, "positive_ticket_consumed", False))
            ),
            None,
        )

        both_tickets = bool(first_decision.get("pre_both_tickets_available", False))
        allow_eligible = bool(first_decision.get("pre_allow_eligible", False))
        allow_preserved = allow_entry is not None
        productive_reveal = productive_reveal_step is not None
        contrastive_used = contrastive_step is not None
        postreveal_cue = postreveal_cue_entry is not None
        grace_consumed = grace_entry is not None
        correct_after_feedback = correct_after_feedback_step is not None
        positive_used = positive_step is not None

        if not both_tickets:
            break_stage = "start"
        elif not allow_preserved:
            break_stage = "allow"
        elif not productive_reveal:
            break_stage = "reveal"
        elif not contrastive_used:
            break_stage = "contrastive"
        elif not postreveal_cue:
            break_stage = "cue"
        elif not (grace_consumed or correct_after_feedback):
            break_stage = "grace"
        elif not correct_after_feedback:
            break_stage = "correct"
        elif not positive_used:
            break_stage = "positive_ticket"
        else:
            break_stage = "complete"

        contrastive_credit_total = sum(
            float(getattr(ls, "semantic_credit", 0.0))
            for ls in steps
            if getattr(ls, "semantic_credit_type", "none") == "contrastive"
        )
        positive_credit_total = sum(
            float(getattr(ls, "semantic_credit", 0.0))
            for ls in steps
            if getattr(ls, "semantic_credit_type", "none") == "positive"
        )

        ledger.append({
            "query_id": int(qid),
            "both_tickets_available": both_tickets,
            "allow_eligible": allow_eligible,
            "allow_preserved": allow_preserved,
            "productive_reveal": productive_reveal,
            "contrastive_ticket_used": contrastive_used,
            "postreveal_cue_or_mix": postreveal_cue,
            "grace_consumed": grace_consumed,
            "correct_after_reveal_cue_grace": correct_after_feedback,
            "positive_ticket_used": positive_used,
            "loop_complete": break_stage == "complete",
            "loop_break_stage": break_stage,
            "allow_loop_value": float(first_decision.get("pre_allow_loop_value", 0.0)),
            "allow_productive_mass": float(first_decision.get("pre_productive_mass_wait", 0.0)),
            "allow_info_mass": float(first_decision.get("pre_info_mass_wait", 0.0)),
            "allow_harm_mass": float(first_decision.get("pre_harm_mass_wait", 0.0)),
            "allow_expected_damage": float(first_decision.get("pre_expected_damage_wait", 0.0)),
            "allow_p_survive": float(first_decision.get("pre_allow_p_survive", 0.0)),
            "allow_reason": str(first_decision.get("pre_productive_allow_reason", "none")),
            "allow_reject_reason": str(first_decision.get("pre_allow_reject_reason", "none")),
            "allow_post_reveal_best_value_estimate": float(
                first_decision.get("pre_allow_post_reveal_best_value_estimate", 0.0)
            ),
            "contrastive_credit_total": round(contrastive_credit_total, 4),
            "positive_credit_total": round(positive_credit_total, 4),
            "retained_gain_proxy": round(positive_credit_total, 4),
            "final_success": bool(getattr(qs, "success", False)),
            "final_hp": int(getattr(qs, "hp", 0)),
            "final_rounds_used": int(getattr(qs, "rounds_used", 0)),
            "n_reveals": len(getattr(qs, "reveal_history", []) or []),
            "post_reveal_phase_reached": bool(getattr(qs, "post_reveal_phase", False)),
        })

    return ledger


def build_postreveal_candidate_audit(block):
    """Return per-candidate audit rows for post-reveal tutor decisions."""
    decision_trace = getattr(block, "_decision_trace", None) or []
    audit_rows = []

    for entry in decision_trace:
        if not bool(entry.get("pre_post_reveal_phase", False)):
            continue
        candidates = (entry.get("scoring") or {}).get("candidates") or []
        chosen_detail = _chosen_candidate_detail(entry, candidates)
        best_with = None
        best_without = None
        for cand in candidates:
            q_with = float(cand.get("q_use_with_consolidate", cand.get("q_use", float("-inf"))))
            q_without = float(
                cand.get(
                    "q_use_without_consolidate",
                    q_with - float(cand.get("q_use_consolidate_delta", 0.0)),
                )
            )
            if best_with is None or q_with > float(best_with.get("q_use_with_consolidate", best_with.get("q_use", float("-inf")))):
                best_with = cand
            if best_without is None or q_without > float(
                best_without.get(
                    "q_use_without_consolidate",
                    float(best_without.get("q_use_with_consolidate", best_without.get("q_use", float("-inf"))))
                    - float(best_without.get("q_use_consolidate_delta", 0.0)),
                )
            ):
                best_without = cand

        best_with_id = _candidate_identity(best_with) if best_with is not None else None
        best_without_id = _candidate_identity(best_without) if best_without is not None else None
        chosen_id = _candidate_identity(chosen_detail) if chosen_detail is not None else None

        for idx, cand in enumerate(candidates):
            q_with = float(cand.get("q_use_with_consolidate", cand.get("q_use", float("-inf"))))
            q_without = float(
                cand.get(
                    "q_use_without_consolidate",
                    q_with - float(cand.get("q_use_consolidate_delta", 0.0)),
                )
            )
            audit_rows.append({
                "query_id": int(entry.get("query_id", -1)),
                "round_t": int(entry.get("round_t", -1)),
                "candidate_idx": int(idx),
                "action": str(cand.get("action", "WAIT")),
                "ban_index": cand.get("ban_index"),
                "highlight_cells": tuple(cand.get("highlight_cells") or ()),
                "chosen": _candidate_identity(cand) == chosen_id,
                "is_best_with_consolidate": _candidate_identity(cand) == best_with_id,
                "is_best_without_consolidate": _candidate_identity(cand) == best_without_id,
                "q_total": q_with,
                "q_without_consolidate": q_without,
                "q_with_consolidate": q_with,
                "q_consolidate_delta": float(q_with - q_without),
                "g_eval": float(cand.get("g_eval", 0.0)),
                "g_exp": float(cand.get("g_exp", 0.0)),
                "g_consolidate": float(cand.get("g_consolidate", cand.get("postreveal_consolidation_effective", 0.0))),
                "g_consolidate_raw": float(cand.get("g_consolidate_raw", cand.get("postreveal_consolidation_value", 0.0))),
                "p_correct_next": float(cand.get("postreveal_p_correct_next", 0.0)),
                "p_correct_2r": float(cand.get("postreveal_p_correct_2r", 0.0)),
                "positive_ticket_available": bool(cand.get("postreveal_positive_ticket_available", False)),
                "source_weight": float(cand.get("postreveal_consolidation_source_weight", 0.0)),
                "delta_correct_mass": float(_detail_metric(cand, "delta_p_correct", 0.0)),
                "delta_margin": float(_detail_metric(cand, "log_margin_gain", 0.0)),
                "delta_harm": float(_detail_metric(cand, "harm_mass_drop", _detail_metric(cand, "bad_mass_drop", 0.0))),
                "harmful_shift": float(_detail_metric(cand, "harmful_shift", _detail_metric(cand, "bad_shift", 0.0))),
                "cost": float(cand.get("cost", 0.0)),
                "d_shift": float(cand.get("d_shift", 0.0)),
                "effective_d_shift": float(cand.get("effective_d_shift", cand.get("d_shift", 0.0))),
            })

    return audit_rows


def _allow_loop_upper_bound_proxy(entry):
    productive_mass = max(0.0, float(entry.get("pre_productive_mass_wait", 0.0)))
    p_survive = max(0.0, float(entry.get("pre_allow_p_survive", 0.0)))
    p_time = 1.0 if int(entry.get("pre_rounds_left", 0)) >= 3 else 0.0
    post_value = max(0.0, float(entry.get("pre_allow_post_reveal_best_value_estimate", 0.0)))
    contrastive_ok = 1.0 if bool(entry.get("pre_contrastive_ticket_available", False)) else 0.0
    positive_ok = 1.0 if bool(entry.get("pre_positive_ticket_available", False)) else 0.0
    return productive_mass * p_survive * p_time * contrastive_ok * positive_ok * post_value


def _phase_reject_reason(entry):
    phase = str(entry.get("phase", "DEFAULT") or "DEFAULT")
    if bool(entry.get("pre_post_reveal_phase", False)):
        return "NOT_PRE_REVEAL_POST_REVEAL_ALREADY"

    p_safe = max(0.0, float(entry.get("pre_p_safe_diag_wait", 0.0)))
    p_highrisk = max(
        0.0,
        float(entry.get("pre_p_highrisk_wait", entry.get("pre_p_high_risk_wait", 0.0))),
    )
    rounds_left = int(entry.get("pre_rounds_left", 0))

    if phase == "PRE_REVEAL_ALLOW":
        return "ALLOW_PHASE_ACTIVE"
    if phase == "PROTECT":
        return "NOT_PRE_REVEAL_PROTECT_PHASE"
    if phase == "BORING_ESCAPE":
        return "NOT_PRE_REVEAL_BORING_PHASE"
    if phase == "POST_REVEAL_CONSOLIDATE":
        return "NOT_PRE_REVEAL_POST_REVEAL_ALREADY"
    if phase == "POST_REVEAL_PROTECT_AND_CUE":
        return "NOT_PRE_REVEAL_POST_REVEAL_ALREADY"
    if phase == "GRACE_WAIT":
        return "NOT_PRE_REVEAL_POST_REVEAL_ALREADY"

    if p_safe <= 0.0:
        return "NOT_PRE_REVEAL_NO_SAFE_DIAG_IN_MENU"
    if p_highrisk > 0.25:
        return "NOT_PRE_REVEAL_PROTECT_PHASE"
    if rounds_left < 2:
        return "NOT_PRE_REVEAL_PHASE_INFER_DEFAULT"
    return "NOT_PRE_REVEAL_PHASE_INFER_DEFAULT"


def _is_native_prereveal_allow_phase_candidate(entry):
    return bool(
        is_native_phase_allow_candidate(
            compute_prereveal_allow_features_from_trace(entry)
        )
    )


def _classify_allow_family_phase_blind(entry):
    features = compute_prereveal_allow_features_from_trace(entry)
    both_tickets = bool(features.get("both_tickets_available", False))
    rounds_ok = int(features.get("rounds_left", 0)) >= 3
    p_prod = max(0.0, float(features.get("productive_mass", 0.0)))
    p_safe = max(0.0, float(features.get("p_safe_diag", 0.0)))
    p_bounded = max(0.0, float(features.get("p_bounded_diag", 0.0)))
    p_far = max(0.0, float(features.get("p_farwrong", 0.0)))
    p_highrisk = max(0.0, float(features.get("p_highrisk", 0.0)))
    harm = max(0.0, float(features.get("harm_mass", 0.0)))
    p_correct = max(0.0, float(features.get("p_correct_wait", 0.0)))

    if is_phasecalib_allow_candidate(features):
        return "ALLOW_CRITICAL_STAR"

    if not both_tickets:
        return "TICKET_BLOCKED"
    if not rounds_ok:
        return "ROUND_BLOCKED"
    if p_correct >= 0.75:
        return "BORING_MASTERY"
    if p_prod <= 0.0:
        return "NO_PRODUCTIVE_OPPORTUNITY"
    if p_highrisk > (p_safe + p_bounded):
        return "HIGHRISK_DOMINATED"
    if p_far > (p_safe + p_bounded):
        return "FAR_DOMINATED"
    if p_prod < 0.5 * harm:
        return "HARM_DOMINATED"
    return "ALLOW_CRITICAL_STAR"


def _metric_token(label):
    chars = []
    for ch in str(label):
        if ch.isalnum():
            chars.append(ch.upper())
        else:
            chars.append("_")
    token = "".join(chars)
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "UNKNOWN"


def build_allow_gate_replay(block):
    """Replay lightweight allow gates on the same frozen pre-reveal states."""
    decision_trace = getattr(block, "_decision_trace", None) or []
    replay_rows = []

    for entry in decision_trace:
        if bool(entry.get("pre_post_reveal_phase", False)):
            continue

        p_prod = max(0.0, float(entry.get("pre_productive_mass_wait", 0.0)))
        harm = max(0.0, float(entry.get("pre_harm_mass_wait", 0.0)))
        p_safe = max(0.0, float(entry.get("pre_p_safe_diag_wait", 0.0)))
        p_bounded = max(0.0, float(entry.get("pre_p_bounded_diag_wait", 0.0)))
        p_far = max(0.0, float(entry.get("pre_p_farwrong_wait", 0.0)))
        p_highrisk = max(
            0.0,
            float(entry.get("pre_p_highrisk_wait", entry.get("pre_p_high_risk_wait", 0.0))),
        )
        expected_damage = max(0.0, float(entry.get("pre_expected_damage_wait", 0.0)))
        p_timeout = max(0.0, float(entry.get("p_timeout_wait", 0.0)))
        post_value = max(
            0.0,
            float(
                entry.get(
                    "pre_allow_post_reveal_best_value_estimate",
                    entry.get("pre_allow_best_cue_cate_estimate", 0.0),
                )
            ),
        )
        both_tickets = bool(entry.get("pre_both_tickets_available", False))
        rounds_ok = int(entry.get("pre_rounds_left", 0)) >= 3
        phase_allows = bool(entry.get("pre_allow_phase_eligible", False))
        current_allow = bool(entry.get("pre_productive_allow_preserved", False))
        loop_upper = _allow_loop_upper_bound_proxy(entry)

        gates = {
            "G0_current_controlled_v2": current_allow,
            "G1_permissive_prod": bool(phase_allows and both_tickets and rounds_ok and p_prod > 0.0),
            "G2_ratio_gate": bool(phase_allows and p_prod > 0.0 and p_prod >= 0.5 * harm),
            "G3_highrisk_guard": bool(
                phase_allows
                and p_prod > 0.0
                and p_highrisk <= (p_safe + p_bounded)
            ),
            "G4_postvalue_gate": bool(phase_allows and p_prod > 0.0 and post_value > 0.0),
            "G5_combined": bool(
                phase_allows
                and both_tickets
                and rounds_ok
                and p_prod > 0.0
                and p_prod >= 0.5 * harm
                and post_value > 0.0
            ),
        }

        replay_rows.append({
            "query_id": int(entry.get("query_id", -1)),
            "round_t": int(entry.get("round_t", -1)),
            "phase": str(entry.get("phase", "none")),
            "allow_eligible": bool(entry.get("pre_allow_eligible", False)),
            "allow_reject_reason": str(entry.get("pre_allow_reject_reason", "none")),
            "p_prod": p_prod,
            "harm_mass": harm,
            "expected_damage": expected_damage,
            "p_safe_diag": p_safe,
            "p_bounded_diag": p_bounded,
            "p_farwrong": p_far,
            "p_highrisk": p_highrisk,
            "p_timeout_wait": p_timeout,
            "post_reveal_best_value_estimate": post_value,
            "loop_upper_bound_proxy": loop_upper,
            **gates,
        })

    return replay_rows


def _classify_allow_family(entry):
    phase_allows = bool(entry.get("pre_allow_phase_eligible", False))
    both_tickets = bool(entry.get("pre_both_tickets_available", False))
    rounds_ok = int(entry.get("pre_rounds_left", 0)) >= 3
    p_prod = max(0.0, float(entry.get("pre_productive_mass_wait", 0.0)))
    p_safe = max(0.0, float(entry.get("pre_p_safe_diag_wait", 0.0)))
    p_bounded = max(0.0, float(entry.get("pre_p_bounded_diag_wait", 0.0)))
    p_far = max(0.0, float(entry.get("pre_p_farwrong_wait", 0.0)))
    p_highrisk = max(
        0.0,
        float(entry.get("pre_p_highrisk_wait", entry.get("pre_p_high_risk_wait", 0.0))),
    )
    harm = max(0.0, float(entry.get("pre_harm_mass_wait", 0.0)))
    p_correct = max(0.0, float(entry.get("p_correct_wait", 0.0)))

    if not phase_allows:
        return "NOT_PRE_REVEAL"
    if not both_tickets:
        return "TICKET_BLOCKED"
    if not rounds_ok:
        return "ROUND_BLOCKED"
    if p_correct >= 0.75:
        return "BORING_MASTERY"
    if p_prod <= 0.0:
        return "NO_PRODUCTIVE_OPPORTUNITY"
    if p_highrisk > (p_safe + p_bounded):
        return "HIGHRISK_DOMINATED"
    if p_far > (p_safe + p_bounded):
        return "FAR_DOMINATED"
    if p_prod < 0.5 * harm:
        return "HARM_DOMINATED"
    return "ALLOW_CRITICAL"


def build_allow_family_audit(block):
    """Return per-state audit rows for pre-reveal allow opportunities."""
    obs_q = block.obs_phase_queries
    teach_q = block.teach_phase_queries
    teach_queries = block.queries[obs_q: obs_q + teach_q]
    query_by_id = {
        int(getattr(qs, "query_id", qi)): qs
        for qi, qs in enumerate(teach_queries)
    }
    decision_trace = getattr(block, "_decision_trace", None) or []
    learner_trace = getattr(block, "learner_trace", None) or []
    decisions_by_qid = defaultdict(list)
    learner_by_qid = defaultdict(list)

    for entry in decision_trace:
        qid = entry.get("query_id")
        if qid is None:
            continue
        decisions_by_qid[int(qid)].append(entry)
    for qid in decisions_by_qid:
        decisions_by_qid[qid] = sorted(
            decisions_by_qid[qid], key=lambda e: int(e.get("round_t", 10**9))
        )

    for ls in learner_trace:
        qid = getattr(ls, "query_id", None)
        if qid is None:
            continue
        learner_by_qid[int(qid)].append(ls)
    for qid in learner_by_qid:
        learner_by_qid[qid] = sorted(
            learner_by_qid[qid], key=lambda s: int(getattr(s, "round_t", 10**9))
        )

    rows = []
    for entry in decision_trace:
        if bool(entry.get("pre_post_reveal_phase", False)):
            continue

        qid = int(entry.get("query_id", -1))
        round_t = int(entry.get("round_t", -1))
        qs = query_by_id.get(qid)
        steps = learner_by_qid.get(qid, [])
        future_steps = [ls for ls in steps if int(getattr(ls, "round_t", -1)) >= round_t]
        future_decisions = [
            d for d in decisions_by_qid.get(qid, [])
            if int(d.get("round_t", -1)) >= round_t
        ]

        productive_reveal_after = any(
            getattr(ls, "raw_feedback_kind", "none") == "wrong_reveal"
            and getattr(ls, "feedback_category", "none") in ("safe_diag", "bounded_diag")
            for ls in future_steps
        )
        safe_diag_after = any(
            getattr(ls, "raw_feedback_kind", "none") == "wrong_reveal"
            and getattr(ls, "feedback_category", "none") == "safe_diag"
            for ls in future_steps
        )
        contrastive_after = any(
            bool(getattr(ls, "contrastive_ticket_consumed", False))
            for ls in future_steps
        )
        cue_after = any(
            bool(d.get("pre_post_reveal_phase", False))
            and d.get("chosen_action") in ("HIGHLIGHT", "MIX")
            for d in future_decisions
        )
        grace_after = any(
            d.get("wait_reason") == "WAIT_GRACE"
            for d in future_decisions
        )
        correct_after = any(
            getattr(ls, "raw_feedback_kind", "none") == "correct_pick"
            and getattr(ls, "semantic_credit_reason", "none") in ("after_reveal", "after_cue", "after_grace")
            for ls in future_steps
        )
        positive_after = any(
            bool(getattr(ls, "positive_ticket_consumed", False))
            for ls in future_steps
        )
        damage_after_state = sum(float(getattr(ls, "damage", 0.0) or 0.0) for ls in future_steps)
        death_before_correct_after_state = bool(
            qs is not None
            and float(getattr(qs, "hp", 0.0)) <= 0.0
            and not correct_after
        )

        p_safe = max(0.0, float(entry.get("pre_p_safe_diag_wait", 0.0)))
        p_bounded = max(0.0, float(entry.get("pre_p_bounded_diag_wait", 0.0)))
        p_prod_safe_component = p_safe
        p_prod_bounded_component = 0.5 * p_bounded
        p_prod_total = p_prod_safe_component + p_prod_bounded_component
        competing_harm = max(
            0.0,
            float(entry.get("pre_p_farwrong_wait", 0.0))
            + float(entry.get("pre_p_highrisk_wait", entry.get("pre_p_high_risk_wait", 0.0))),
        )
        safe_diag_quality_gap = p_safe - competing_harm
        harm_mass = max(0.0, float(entry.get("pre_harm_mass_wait", 0.0)))
        harm_competition_gap = p_prod_total - harm_mass
        family_features = compute_prereveal_allow_features_from_trace(entry)
        family_split = classify_prereveal_family(family_features)

        rows.append({
            "query_id": qid,
            "round_t": round_t,
            "family": _classify_allow_family(entry),
            "family_split": family_split,
            "phase_blind_family": _classify_allow_family_phase_blind(entry),
            "phase_reject_reason": _phase_reject_reason(entry),
            "native_phase_allow_candidate": bool(
                _is_native_prereveal_allow_phase_candidate(entry)
            ),
            "allow_preserved": bool(entry.get("pre_productive_allow_preserved", False)),
            "allow_eligible": bool(entry.get("pre_allow_eligible", False)),
            "allow_reject_reason": str(entry.get("pre_allow_reject_reason", "none")),
            "phase": str(entry.get("phase", "DEFAULT")),
            "p_prod_total": p_prod_total,
            "p_prod_safe_component": p_prod_safe_component,
            "p_prod_bounded_component": p_prod_bounded_component,
            "p_prod_safe_share": (
                p_prod_safe_component / max(p_prod_total, 1e-9)
                if p_prod_total > 0.0 else 0.0
            ),
            "safe_diag_quality_gap": safe_diag_quality_gap,
            "competing_harm_mass": competing_harm,
            "harm_competition_gap": harm_competition_gap,
            "harm_mass": harm_mass,
            "expected_damage": max(0.0, float(entry.get("pre_expected_damage_wait", 0.0))),
            "p_correct_wait": max(0.0, float(entry.get("p_correct_wait", 0.0))),
            "p_timeout_wait": max(0.0, float(entry.get("p_timeout_wait", 0.0))),
            "both_tickets_available": bool(entry.get("pre_both_tickets_available", False)),
            "rounds_left": int(entry.get("pre_rounds_left", 0)),
            "post_reveal_best_value_estimate": max(
                0.0,
                float(entry.get("pre_allow_post_reveal_best_value_estimate", 0.0)),
            ),
            "missed_allow_critical": bool(
                _classify_allow_family_phase_blind(entry) == "ALLOW_CRITICAL_STAR"
                and _classify_allow_family(entry) != "ALLOW_CRITICAL"
            ),
            "productive_reveal_after_state": productive_reveal_after,
            "safe_diag_reveal_after_state": safe_diag_after,
            "contrastive_after_state": contrastive_after,
            "cue_after_state": cue_after,
            "grace_after_state": grace_after,
            "correct_after_state": correct_after,
            "positive_ticket_after_state": positive_after,
            "damage_after_state": damage_after_state,
            "death_before_correct_after_state": death_before_correct_after_state,
            "loop_complete_after_state": bool(
                productive_reveal_after
                and contrastive_after
                and cue_after
                and correct_after
                and positive_after
            ),
        })

    return rows


def compute_6fg_metrics(block) -> dict:
    """Compute HIGHLIGHT, tutor decision, and trajectory metrics."""
    obs_q = block.obs_phase_queries
    teach_q = block.teach_phase_queries
    teach_queries = block.queries[obs_q: obs_q + teach_q]

    tt = block.tutor_trace
    lt = block.learner_trace
    wait_count = sum(1 for ts in tt if ts.action == "WAIT")
    ban_count = sum(1 for ts in tt if ts.action == "BAN")
    hl_count = sum(1 for ts in tt if ts.action == "HIGHLIGHT")
    mix_count = sum(1 for ts in tt if ts.action == "MIX")
    total_actions = max(len(tt), 1)

    hl_self_correct = 0
    hl_total = 0
    for ts in tt:
        if ts.action in ("HIGHLIGHT", "MIX"):
            hl_total += 1
            qid = getattr(ts, "query_id", None)
            if qid is not None:
                for ls in lt:
                    ls_qid = getattr(ls, "query_id", None)
                    if ls_qid == qid and ls.action == "pick":
                        if ls.correct:
                            hl_self_correct += 1
                        break

    safe_diag_total = 0
    high_risk_total = 0
    high_risk_banned = 0
    safe_diag_banned = 0
    safe_diag_protectable_total = 0
    safe_diag_protectable_banned = 0
    safe_diag_late_banned = 0

    for qs in teach_queries:
        labels = getattr(qs, "option_diag_labels", {})
        n_safe_reveals = getattr(qs, "n_safe_diag_wrong_reveals", 0)
        hp = qs.hp
        rounds_left = qs.max_rounds - qs.rounds_used

        for opt in qs.menu:
            label = labels.get(opt.index, "")
            if label == "safe_diagnostic_wrong":
                safe_diag_total += 1
                hp_after = hp - getattr(opt, "risk_class", 0)
                rounds_after = rounds_left - 1
                if n_safe_reveals == 0 and hp_after > 0 and rounds_after >= 1:
                    safe_diag_protectable_total += 1
            elif label == "high_risk_lure":
                high_risk_total += 1

    for ts in tt:
        if ts.action == "BAN":
            ban_idx = getattr(ts, "ban_index", None)
            qid = getattr(ts, "query_id", None)
            if qid is not None and ban_idx is not None and qid < len(block.queries):
                qs_ban = block.queries[qid]
                labels = getattr(qs_ban, "option_diag_labels", {})
                lbl = labels.get(ban_idx, "")
                if lbl == "high_risk_lure":
                    high_risk_banned += 1
                elif lbl == "safe_diagnostic_wrong":
                    safe_diag_banned += 1
                    n_safe_at_ban = getattr(qs_ban, "n_safe_diag_wrong_reveals", 0)
                    hp_at_ban = qs_ban.hp
                    rounds_left_at_ban = qs_ban.max_rounds - qs_ban.rounds_used
                    ban_opt = next((o for o in qs_ban.menu if o.index == ban_idx), None)
                    if ban_opt is not None:
                        hp_after = hp_at_ban - getattr(ban_opt, "risk_class", 0)
                        rounds_after = rounds_left_at_ban - 1
                        if n_safe_at_ban == 0 and hp_after > 0 and rounds_after >= 1:
                            safe_diag_protectable_banned += 1
                        else:
                            safe_diag_late_banned += 1

    n_diag_reveal_queries = 0
    n_diag_reveal_then_correct = 0
    n_diag_reveal_then_self_correct = 0
    n_diag_reveal_wasted = 0
    n_post_reveal_hl = 0
    n_post_reveal_mix = 0
    n_post_reveal_wait = 0
    n_post_hl_correct = 0
    n_post_hl_self_correct = 0
    n_repeated_wrong_after_reveal = 0

    query_by_qid = {qs.query_id: qs for qs in teach_queries}
    tutor_by_qid = defaultdict(list)
    learner_by_qid = defaultdict(list)
    tutor_step_by_key = {}
    teach_qids = {qs.query_id for qs in teach_queries}
    trace_entries = [
        tr for tr in (getattr(block, "_decision_trace", None) or [])
        if tr.get("query_id") in teach_qids
    ]
    trace_entries_by_qid = defaultdict(list)
    for ts in tt:
        qid = getattr(ts, "query_id", None)
        if qid is not None:
            tutor_by_qid[qid].append(ts)
            tutor_step_by_key[(qid, getattr(ts, "round_t", None))] = ts
    for ls in lt:
        qid = getattr(ls, "query_id", None)
        if qid is not None:
            learner_by_qid[qid].append(ls)
    for tr in trace_entries:
        qid = tr.get("query_id")
        if qid is not None:
            trace_entries_by_qid[qid].append(tr)
    for qid in list(learner_by_qid.keys()):
        learner_by_qid[qid] = sorted(
            learner_by_qid[qid],
            key=lambda s: (getattr(s, "round_t", -1), getattr(s, "action", "")),
        )
    for qid in list(trace_entries_by_qid.keys()):
        trace_entries_by_qid[qid] = sorted(
            trace_entries_by_qid[qid],
            key=lambda tr: (tr.get("round_t", -1), tr.get("chosen_action", "")),
        )

    for qs in teach_queries:
        n_safe = getattr(qs, "n_safe_diag_wrong_reveals", 0)
        if n_safe == 0:
            continue
        n_diag_reveal_queries += 1

        if qs.success:
            n_diag_reveal_then_correct += 1
            assist = getattr(qs, "assist_level", "none")
            if assist in ("none", "risk_hint"):
                n_diag_reveal_then_self_correct += 1
        else:
            n_diag_reveal_wasted += 1

        qid = qs.query_id
        l_actions = learner_by_qid.get(qid, [])

        for tr in trace_entries:
            if tr.get("query_id") != qid or not bool(tr.get("pre_post_reveal_phase", False)):
                continue
            action = tr.get("chosen_action", "WAIT")
            ts_round = int(tr.get("round_t", -1))
            next_pick = _next_pick_after_round(l_actions, ts_round)
            if action == "HIGHLIGHT":
                n_post_reveal_hl += 1
                if next_pick is not None and next_pick.correct:
                    n_post_hl_correct += 1
                    n_post_hl_self_correct += 1
            elif action == "MIX":
                n_post_reveal_mix += 1
            elif action == "WAIT":
                n_post_reveal_wait += 1

        wrong_after = sum(
            1
            for ls in l_actions
            if ls.action == "pick" and not ls.correct and getattr(ls, "round_t", -1) > 0
        )
        if wrong_after > 1:
            n_repeated_wrong_after_reveal += 1

    n_tq = max(len(teach_queries), 1)
    safe_diag_ban_rate = safe_diag_banned / max(safe_diag_total, 1)
    high_risk_ban_rate = high_risk_banned / max(high_risk_total, 1)
    protectable_ban_rate = safe_diag_protectable_banned / max(safe_diag_protectable_total, 1)
    late_ban_rate = safe_diag_late_banned / max(safe_diag_banned - safe_diag_protectable_banned, 1)
    ped_selectivity = ((1.0 - protectable_ban_rate) + high_risk_ban_rate) / 2.0
    mean_p_safe_diag = safe_diag_total / max(n_tq, 1)
    mean_p_high_risk = high_risk_total / max(n_tq, 1)
    grace_m = getattr(block, "_grace_metrics", {})
    grace_set = grace_m.get("set", 0)
    grace_eligible = grace_m.get("eligible_next_round", 0)
    grace_next_tutor = grace_m.get("next_tutor_called", 0)
    grace_count = grace_m.get("count", 0)
    grace_chosen_wait = grace_m.get("chosen_wait", 0)
    grace_chosen_override = grace_m.get("chosen_override", 0)
    grace_consumed = grace_m.get("consumed", 0)
    grace_override = grace_m.get("override", 0)
    grace_blocked_protect = grace_m.get("blocked_by_protect", 0)
    grace_blocked_deadline = grace_m.get("blocked_by_deadline", 0)
    grace_no_tutor = grace_m.get("did_not_reach_tutor_decision", 0)
    grace_lost_success = grace_m.get("lost_query_succeeded", 0)
    grace_lost_wrong_terminal = grace_m.get("lost_wrong_terminal", 0)
    grace_lost_max_round = grace_m.get("lost_max_round", 0)
    grace_flag_reset = grace_m.get("flag_reset_without_consumption", 0)
    # ── Phase 6I: opportunity-conditioned metrics ──
    # Count tutor actions conditioned on what opportunity exists at decision time
    sd_opp_wait = 0    # WAIT when safe_diag opportunity exists
    sd_opp_ban = 0     # BAN when safe_diag opportunity exists
    sd_opp_total = 0   # total decisions when safe_diag opportunity exists
    hr_opp_wait = 0    # WAIT when high_risk opportunity exists
    hr_opp_ban = 0     # BAN when high_risk opportunity exists
    hr_opp_total = 0   # total decisions when high_risk opportunity exists
    postrev_hl_total = 0   # tutor decisions in post-reveal states
    postrev_mix_total = 0
    postrev_wait_total = 0
    postrev_decisions = 0
    # Q decomposition / routing diagnostics from decision-time trace
    postrev_hl_generated = 0
    postrev_mix_generated = 0
    postrev_hl_beats_wait = 0
    postrev_mix_beats_wait = 0
    postrev_wait_best_by_q = 0
    postrev_wait_when_nonwait_better = 0
    postrev_q_regret = []
    postrev_q_wait = []
    postrev_q_hl = []
    postrev_q_mix = []
    postrev_q_ban = []
    postrev_hl_dshift = []
    postrev_mix_dshift = []
    postrev_hl_eff_dshift = []
    postrev_mix_eff_dshift = []
    postrev_hl_cost = []
    postrev_mix_cost = []
    postrev_hl_top1flip = []
    postrev_mix_top1flip = []
    postrev_hl_margin = []
    postrev_mix_margin = []
    postrev_hl_hrdrop = []
    postrev_mix_hrdrop = []
    postrev_hl_samewrong = []
    postrev_mix_samewrong = []
    postrev_hl_badmass = []
    postrev_mix_badmass = []
    postrev_hl_removedbad = []
    postrev_mix_removedbad = []
    postrev_hl_removedprob = []
    postrev_mix_removedprob = []
    forcebest_selected_hl = 0
    forcebest_selected_mix = 0
    forcebest_selected_none = 0
    postrev_positive_cue_opp = 0
    postrev_positive_mix_opp = 0
    postrev_positive_hl_opp = 0
    postrev_badwait_positive_cue = 0
    postrev_badwait_positive_mix = 0
    postrev_badwait_positive_hl = 0
    mix_chosen_count = 0
    mix_ban_lastwrong = 0
    mix_ban_highrisk = 0
    mix_ban_safe_diag = 0
    mix_ban_topwrong = 0
    mix_ban_farwrong = 0
    mix_ban_correct = 0
    mix_ban_policy_mass = []
    mix_ban_badness = []
    mix_removed_prob = []
    mix_removed_bad = []
    mix_badmass_drop = []
    mix_delta_p = []
    mix_margin_gain = []
    mix_removed_target_regret = []
    mix_net_target_regret = []
    mix_oracle_removed_mass = []
    mix_oracle_net_drop = []
    mix_matches_removed_oracle = 0
    mix_matches_net_oracle = 0
    mix_joint_gate_applied = 0
    mix_joint_gate_replaced = 0
    mix_joint_target_regret = []
    mix_joint_highlight_regret = []
    mix_joint_regret = []
    mix_joint_interaction_regret = []
    mix_direct_selector_applied = 0
    mix_direct_selected_net_harm = []
    mix_direct_oracle_net_harm = []
    mix_direct_net_target_regret = []
    joint_replay_eligible_count = 0
    joint_replay_eligible_regret = []
    joint_replay_triggered_regret = []
    joint_replay_skipped_high_regret = 0
    allow_wait_count = 0
    safe_diag_reveal_after_allow = 0
    safe_diag_reveal_then_cue = 0
    safe_diag_reveal_then_grace = 0
    safe_diag_reveal_then_traj_success = 0
    bounded_reveal_after_allow = 0
    bounded_reveal_then_traj_success = 0

    for tr in trace_entries:
        qid = tr.get("query_id")
        round_t = tr.get("round_t", None)
        step = tutor_step_by_key.get((qid, round_t))
        action = tr.get("chosen_action") or getattr(step, "action", "WAIT")
        has_sd = bool(tr.get("pre_has_safe_diag_opp", False))
        has_hr = bool(tr.get("pre_has_high_risk_opp", False))
        is_post_reveal = bool(tr.get("pre_post_reveal_phase", False))

        if has_sd:
            sd_opp_total += 1
            if action == "WAIT":
                sd_opp_wait += 1
            elif action == "BAN":
                sd_opp_ban += 1

        if has_hr:
            hr_opp_total += 1
            if action == "WAIT":
                hr_opp_wait += 1
            elif action == "BAN":
                hr_opp_ban += 1

        if is_post_reveal:
            postrev_decisions += 1
            joint_info = ((tr.get("generation", {}) or {}).get("mix_joint_replay_gate", {}) or {})
            if bool(joint_info.get("evaluated", False)):
                joint_replay_eligible_count += 1
                joint_replay_eligible_regret.append(float(joint_info.get("joint_regret", 0.0)))
                if bool(joint_info.get("applied", False)):
                    joint_replay_triggered_regret.append(float(joint_info.get("joint_regret", 0.0)))
                elif float(joint_info.get("joint_regret", 0.0)) > 1e-9:
                    joint_replay_skipped_high_regret += 1
            if action == "HIGHLIGHT":
                postrev_hl_total += 1
            elif action == "MIX":
                postrev_mix_total += 1
            elif action == "WAIT":
                postrev_wait_total += 1

            scoring = tr.get("scoring", {}) or {}
            candidates = scoring.get("candidates", []) or []
            q_wait = scoring.get("q_wait", None)
            q_hl = _best_q_for_action(candidates, "HIGHLIGHT")
            q_mix = _best_q_for_action(candidates, "MIX")
            q_ban = _best_q_for_action(candidates, "BAN")
            best_hl = _best_detail_for_action(candidates, "HIGHLIGHT")
            best_mix = _best_detail_for_action(candidates, "MIX")

            if q_wait is not None:
                q_wait = float(q_wait)
                postrev_q_wait.append(q_wait)
            if q_hl is not None:
                q_hl = float(q_hl)
                postrev_q_hl.append(q_hl)
                postrev_hl_generated += 1
                if q_wait is not None and q_hl > q_wait:
                    postrev_hl_beats_wait += 1
                    postrev_positive_hl_opp += 1
            if q_mix is not None:
                q_mix = float(q_mix)
                postrev_q_mix.append(q_mix)
                postrev_mix_generated += 1
                if q_wait is not None and q_mix > q_wait:
                    postrev_mix_beats_wait += 1
                    postrev_positive_mix_opp += 1
            if q_ban is not None:
                postrev_q_ban.append(float(q_ban))
            any_positive_cue = (
                q_wait is not None and (
                    (q_hl is not None and q_hl > q_wait) or
                    (q_mix is not None and q_mix > q_wait)
                )
            )
            if any_positive_cue:
                postrev_positive_cue_opp += 1
                if action == "WAIT":
                    postrev_badwait_positive_cue += 1
            if q_wait is not None and q_mix is not None and q_mix > q_wait and action == "WAIT":
                postrev_badwait_positive_mix += 1
            if q_wait is not None and q_hl is not None and q_hl > q_wait and action == "WAIT":
                postrev_badwait_positive_hl += 1

            if best_hl is not None:
                postrev_hl_dshift.append(float(_detail_metric(best_hl, "d_shift", 0.0)))
                postrev_hl_eff_dshift.append(float(_detail_metric(best_hl, "effective_d_shift", _detail_metric(best_hl, "d_shift", 0.0))))
                postrev_hl_cost.append(float(_detail_metric(best_hl, "cost", 0.0)))
                postrev_hl_top1flip.append(float(_detail_metric(best_hl, "top1_flip", 0.0)))
                postrev_hl_margin.append(float(_detail_metric(best_hl, "correct_margin_gain", 0.0)))
                postrev_hl_hrdrop.append(float(_detail_metric(best_hl, "highrisk_drop", 0.0)))
                postrev_hl_samewrong.append(float(_detail_metric(best_hl, "samewrong_drop", 0.0)))
                postrev_hl_badmass.append(float(_detail_metric(best_hl, "bad_mass_drop", 0.0)))
                postrev_hl_removedbad.append(float(_detail_metric(best_hl, "removed_bad_mass", 0.0)))
                postrev_hl_removedprob.append(float(_detail_metric(best_hl, "removed_prob_mass", 0.0)))
            if best_mix is not None:
                postrev_mix_dshift.append(float(_detail_metric(best_mix, "d_shift", 0.0)))
                postrev_mix_eff_dshift.append(float(_detail_metric(best_mix, "effective_d_shift", _detail_metric(best_mix, "d_shift", 0.0))))
                postrev_mix_cost.append(float(_detail_metric(best_mix, "cost", 0.0)))
                postrev_mix_top1flip.append(float(_detail_metric(best_mix, "top1_flip", 0.0)))
                postrev_mix_margin.append(float(_detail_metric(best_mix, "correct_margin_gain", 0.0)))
                postrev_mix_hrdrop.append(float(_detail_metric(best_mix, "highrisk_drop", 0.0)))
                postrev_mix_samewrong.append(float(_detail_metric(best_mix, "samewrong_drop", 0.0)))
                postrev_mix_badmass.append(float(_detail_metric(best_mix, "bad_mass_drop", 0.0)))
                postrev_mix_removedbad.append(float(_detail_metric(best_mix, "removed_bad_mass", 0.0)))
                postrev_mix_removedprob.append(float(_detail_metric(best_mix, "removed_prob_mass", 0.0)))

            if action == "MIX":
                mix_chosen_count += 1
                qs = query_by_qid.get(qid)
                chosen_ban_index = tr.get("chosen_ban_index")
                last_wrong_index = tr.get("pre_last_reveal_option_index")
                labels = getattr(qs, "option_diag_labels", {}) if qs is not None else {}
                ban_label = labels.get(chosen_ban_index, "")
                if chosen_ban_index is not None and last_wrong_index is not None and chosen_ban_index == last_wrong_index:
                    mix_ban_lastwrong += 1
                if ban_label == "high_risk_lure":
                    mix_ban_highrisk += 1
                if ban_label == "safe_diagnostic_wrong":
                    mix_ban_safe_diag += 1
                chosen_detail = (tr.get("scoring", {}) or {}).get("chosen_detail", {}) or best_mix or {}
                if bool(_detail_metric(chosen_detail, "mix_ban_target_was_top_prob_wrong", False)):
                    mix_ban_topwrong += 1
                if bool(_detail_metric(chosen_detail, "mix_ban_target_was_far_wrong", False)):
                    mix_ban_farwrong += 1
                if bool(_detail_metric(chosen_detail, "mix_ban_target_was_correct", False)):
                    mix_ban_correct += 1
                mix_ban_policy_mass.append(float(_detail_metric(chosen_detail, "mix_ban_target_policy_prob_wait", 0.0)))
                mix_ban_badness.append(float(_detail_metric(chosen_detail, "mix_ban_target_badness", 0.0)))
                mix_removed_target_regret.append(float(_detail_metric(chosen_detail, "mix_removed_target_regret", 0.0)))
                mix_net_target_regret.append(float(_detail_metric(chosen_detail, "mix_net_target_regret", 0.0)))
                mix_oracle_removed_mass.append(float(_detail_metric(chosen_detail, "mix_removed_oracle_mass", 0.0)))
                mix_oracle_net_drop.append(float(_detail_metric(chosen_detail, "mix_net_oracle_drop", 0.0)))
                if bool(_detail_metric(chosen_detail, "mix_chosen_matches_removed_oracle", False)):
                    mix_matches_removed_oracle += 1
                if bool(_detail_metric(chosen_detail, "mix_chosen_matches_net_oracle", False)):
                    mix_matches_net_oracle += 1
                if bool(_detail_metric(chosen_detail, "mix_joint_gate_applied", False)):
                    mix_joint_gate_applied += 1
                if bool(_detail_metric(chosen_detail, "mix_joint_gate_replaced", False)):
                    mix_joint_gate_replaced += 1
                if bool(_detail_metric(chosen_detail, "mix_direct_selector_applied", False)):
                    mix_direct_selector_applied += 1
                mix_direct_selected_net_harm.append(float(_detail_metric(chosen_detail, "mix_direct_selected_net_harm_drop", 0.0)))
                mix_direct_oracle_net_harm.append(float(_detail_metric(chosen_detail, "mix_direct_oracle_net_harm_drop", 0.0)))
                mix_direct_net_target_regret.append(float(_detail_metric(chosen_detail, "mix_direct_net_target_regret", 0.0)))
                mix_joint_target_regret.append(float(_detail_metric(chosen_detail, "mix_joint_target_regret", 0.0)))
                mix_joint_highlight_regret.append(float(_detail_metric(chosen_detail, "mix_joint_highlight_regret", 0.0)))
                mix_joint_regret.append(float(_detail_metric(chosen_detail, "mix_joint_regret", 0.0)))
                mix_joint_interaction_regret.append(float(_detail_metric(chosen_detail, "mix_joint_interaction_regret", 0.0)))
                if best_mix is not None:
                    mix_removed_prob.append(float(_detail_metric(best_mix, "removed_prob_mass", 0.0)))
                    mix_removed_bad.append(float(_detail_metric(best_mix, "removed_bad_mass", 0.0)))
                    mix_badmass_drop.append(float(_detail_metric(best_mix, "bad_mass_drop", 0.0)))
                    mix_delta_p.append(float(_detail_metric(best_mix, "delta_p_correct", 0.0)))
                    mix_margin_gain.append(float(_detail_metric(best_mix, "correct_margin_gain", 0.0)))

            if candidates and q_wait is not None:
                valid_qs = [
                    float(c.get("q_use", float("-inf")))
                    for c in candidates if c.get("guard_passed", True)
                ]
                best_q = max(valid_qs) if valid_qs else q_wait
                if q_wait >= best_q:
                    postrev_wait_best_by_q += 1
                if action == "WAIT" and best_q > q_wait:
                    postrev_wait_when_nonwait_better += 1
                postrev_q_regret.append(best_q - q_wait)

        force_type = (tr.get("scoring", {}) or {}).get("force_type")
        if force_type == "best_CATE":
            if action == "HIGHLIGHT":
                forcebest_selected_hl += 1
            elif action == "MIX":
                forcebest_selected_mix += 1
            else:
                forcebest_selected_none += 1

        wait_reason = tr.get("wait_reason")
        if wait_reason == "WAIT_ALLOW_SAFE_DIAG":
            allow_wait_count += 1
            qid = tr.get("query_id")
            ts_round = int(tr.get("round_t", -1))
            qs = query_by_qid.get(qid)
            labels = getattr(qs, "option_diag_labels", {}) if qs is not None else {}
            l_actions = learner_by_qid.get(qid, [])
            first_pick = _next_pick_after_round(l_actions, ts_round)
            subsequent_traces = [
                tr2 for tr2 in trace_entries_by_qid.get(qid, [])
                if int(tr2.get("round_t", -1)) > ts_round
            ]
            has_cue = any(tr2.get("chosen_action") in ("HIGHLIGHT", "MIX") for tr2 in subsequent_traces)
            has_grace = any(tr2.get("wait_reason") == "WAIT_GRACE" for tr2 in subsequent_traces)
            win_steps = _learner_steps_within_round_window(l_actions, ts_round, horizon_rounds=3)
            traj_success = any(ls.action == "pick" and ls.correct for ls in win_steps)
            if first_pick is not None and first_pick.action == "pick" and not first_pick.correct:
                label = labels.get(getattr(first_pick, "pick_index", None), "")
                if label == "safe_diagnostic_wrong":
                    safe_diag_reveal_after_allow += 1
                    if has_cue:
                        safe_diag_reveal_then_cue += 1
                    if has_grace:
                        safe_diag_reveal_then_grace += 1
                    if traj_success:
                        safe_diag_reveal_then_traj_success += 1
                elif label == "bounded_diagnostic_wrong":
                    bounded_reveal_after_allow += 1
                    if traj_success:
                        bounded_reveal_then_traj_success += 1

    # Phase 6I.1 P0: Post-cue outcome classification (four-way split)
    # - unassisted_sc: correct pick after no tutor cue (assist = none/risk_hint/self_correct)
    # - cue_guided_sc: correct pick after HIGHLIGHT (weak structural cue)
    # - structural_protect_correct: correct pick after MIX (ban + highlight)
    # - answer_leakage: correct pick after direct_answer/shortlist
    unassisted_sc = 0
    cue_guided_sc = 0
    structural_protect_correct = 0
    answer_leakage = 0
    cue_total_picks = 0
    cue_wrong_picks = 0
    cue_events = 0
    cue_refresh = 0
    cue_wrong_same = 0
    cue_wrong_diff_safe = 0
    cue_wrong_bounded = 0
    cue_wrong_highrisk = 0
    cue_wrong_far = 0
    cue_wrong_other = 0
    cue_immediate_correct = 0
    cue_immediate_wrong = 0
    cue_grace_wait = 0
    cue_then_grace_correct = 0
    cue_then_grace_wrong = 0
    cue_then_grace_refresh = 0
    cue_traj_success_2rounds = 0
    cue_traj_fail_2rounds = 0
    cue_traj_terminal_fail = 0
    cue_traj_repeated_wrong = 0
    cue_traj_highrisk_fail = 0
    cue_traj_farwrong_fail = 0

    for ts in tt:
        if ts.action not in ("HIGHLIGHT", "MIX"):
            continue
        ts_round = getattr(ts, "round_t", -1)
        qid = getattr(ts, "query_id", None)
        if qid is None:
            continue
        qs = query_by_qid.get(qid)
        learner_steps = learner_by_qid.get(qid, [])
        cue_events += 1

        first_step = _first_learner_step_at_or_after_round(learner_steps, ts_round)
        if first_step is not None:
            if first_step.action == "refresh":
                cue_refresh += 1
            elif first_step.action == "pick":
                cue_total_picks += 1
                if first_step.correct:
                    cue_immediate_correct += 1
                    if ts.action == "HIGHLIGHT":
                        cue_guided_sc += 1
                    elif ts.action == "MIX":
                        structural_protect_correct += 1
                else:
                    cue_immediate_wrong += 1
                    cue_wrong_picks += 1
                    wrong_type = _classify_postcue_wrong(
                        qs,
                        getattr(first_step, "pick_index", None),
                        getattr(qs, "last_reveal_option_index", None),
                    )
                    if wrong_type == "SameWrong":
                        cue_wrong_same += 1
                    elif wrong_type == "DifferentSafeDiag":
                        cue_wrong_diff_safe += 1
                    elif wrong_type == "BoundedDiag":
                        cue_wrong_bounded += 1
                    elif wrong_type == "HighRisk":
                        cue_wrong_highrisk += 1
                    elif wrong_type == "FarWrong":
                        cue_wrong_far += 1
                    else:
                        cue_wrong_other += 1

        next_trace = None
        for tr2 in trace_entries_by_qid.get(qid, []):
            if int(tr2.get("round_t", -1)) > ts_round:
                next_trace = tr2
                break
        if next_trace is not None and next_trace.get("wait_reason") == "WAIT_GRACE":
            cue_grace_wait += 1
            post_grace_step = _first_learner_step_at_or_after_round(
                learner_steps,
                int(next_trace.get("round_t", -1)),
            )
            if post_grace_step is not None:
                if post_grace_step.action == "refresh":
                    cue_then_grace_refresh += 1
                elif post_grace_step.action == "pick":
                    if post_grace_step.correct:
                        cue_then_grace_correct += 1
                    else:
                        cue_then_grace_wrong += 1

        window_steps = _learner_steps_within_round_window(learner_steps, ts_round, horizon_rounds=2)
        has_success_2r = any(ls.action == "pick" and ls.correct for ls in window_steps)
        if has_success_2r:
            cue_traj_success_2rounds += 1
        else:
            cue_traj_fail_2rounds += 1
        wrong_picks_window = [ls for ls in window_steps if ls.action == "pick" and not ls.correct]
        if wrong_picks_window:
            last_wrong_index = getattr(qs, "last_reveal_option_index", None) if qs is not None else None
            wrong_types = {
                _classify_postcue_wrong(qs, getattr(ls, "pick_index", None), last_wrong_index)
                for ls in wrong_picks_window
            }
            if "SameWrong" in wrong_types:
                cue_traj_repeated_wrong += 1
            if "HighRisk" in wrong_types:
                cue_traj_highrisk_fail += 1
            if "FarWrong" in wrong_types:
                cue_traj_farwrong_fail += 1
        if qs is not None and any(ls.action == "pick" and not ls.correct for ls in window_steps) and not qs.success:
            cue_traj_terminal_fail += 1

    pedagogical_success = cue_guided_sc + structural_protect_correct
    postreveal_hl_rate = postrev_hl_total / max(postrev_decisions, 1)
    postreveal_mix_rate = postrev_mix_total / max(postrev_decisions, 1)
    postreveal_wait_rate = postrev_wait_total / max(postrev_decisions, 1)

    teach_qids = {qs.query_id for qs in teach_queries}
    teach_steps = [ls for ls in lt if getattr(ls, "query_id", None) in teach_qids]
    raw_wrong_reveal_count = sum(1 for ls in teach_steps if getattr(ls, "raw_feedback_kind", "none") == "wrong_reveal")
    semantic_wrong_update_count = sum(
        1 for ls in teach_steps
        if getattr(ls, "raw_feedback_kind", "none") == "wrong_reveal"
        and bool(getattr(ls, "semantic_update_applied", False))
    )
    raw_correct_pick_count = sum(1 for ls in teach_steps if getattr(ls, "raw_feedback_kind", "none") == "correct_pick")
    semantic_correct_update_count = sum(
        1 for ls in teach_steps
        if getattr(ls, "raw_feedback_kind", "none") == "correct_pick"
        and bool(getattr(ls, "semantic_update_applied", False))
    )
    contrastive_ticket_used = sum(1 for qs in teach_queries if bool(getattr(qs, "contrastive_update_used", False)))
    positive_ticket_used = sum(1 for qs in teach_queries if bool(getattr(qs, "positive_update_used", False)))
    contrastive_credits = [
        float(getattr(ls, "semantic_credit", 0.0))
        for ls in teach_steps
        if getattr(ls, "semantic_credit_type", "none") == "contrastive"
    ]
    positive_credits = [
        float(getattr(ls, "semantic_credit", 0.0))
        for ls in teach_steps
        if getattr(ls, "semantic_credit_type", "none") == "positive"
    ]
    productive_reveals = [
        ls for ls in teach_steps
        if getattr(ls, "raw_feedback_kind", "none") == "wrong_reveal"
        and getattr(ls, "feedback_category", "none") in ("safe_diag", "bounded_diag")
    ]
    harmful_reveals = [
        ls for ls in teach_steps
        if getattr(ls, "raw_feedback_kind", "none") == "wrong_reveal"
        and getattr(ls, "feedback_category", "none") in ("same_wrong", "high_risk", "far_wrong")
    ]
    productive_credit_values = [float(getattr(ls, "semantic_credit", 0.0)) for ls in productive_reveals]
    harmful_credit_values = [float(getattr(ls, "semantic_credit", 0.0)) for ls in harmful_reveals]

    def _ticket_spend_rate(category):
        spent = [
            ls for ls in teach_steps
            if bool(getattr(ls, "contrastive_ticket_consumed", False))
            and getattr(ls, "feedback_category", "none") == category
        ]
        return len(spent) / max(contrastive_ticket_used, 1)

    safe_diag_reveal_credit = [
        float(getattr(ls, "semantic_credit", 0.0))
        for ls in teach_steps
        if getattr(ls, "feedback_category", "none") == "safe_diag"
    ]
    far_wrong_reveal_credit = [
        float(getattr(ls, "semantic_credit", 0.0))
        for ls in teach_steps
        if getattr(ls, "feedback_category", "none") == "far_wrong"
    ]
    high_risk_reveal_credit = [
        float(getattr(ls, "semantic_credit", 0.0))
        for ls in teach_steps
        if getattr(ls, "feedback_category", "none") == "high_risk"
    ]
    repeated_wrong_reveal_credit = [
        float(getattr(ls, "semantic_credit", 0.0))
        for ls in teach_steps
        if getattr(ls, "feedback_category", "none") == "same_wrong"
    ]
    correct_pick_steps = [
        ls for ls in teach_steps
        if getattr(ls, "raw_feedback_kind", "none") == "correct_pick"
    ]
    correct_after_reveal_steps = [
        ls for ls in correct_pick_steps
        if getattr(ls, "semantic_credit_reason", "none") == "after_reveal"
    ]
    correct_after_cue_steps = [
        ls for ls in correct_pick_steps
        if getattr(ls, "semantic_credit_reason", "none") == "after_cue"
    ]
    correct_after_grace_steps = [
        ls for ls in correct_pick_steps
        if getattr(ls, "semantic_credit_reason", "none") == "after_grace"
    ]
    correct_incidental_steps = [
        ls for ls in correct_pick_steps
        if getattr(ls, "semantic_credit_reason", "none") == "incidental_correct"
    ]

    def _applied_rate(steps):
        return (
            sum(1 for ls in steps if bool(getattr(ls, "semantic_update_applied", False)))
            / max(len(steps), 1)
        )

    def _ticket_used_rate(steps):
        return (
            sum(1 for ls in steps if bool(getattr(ls, "positive_ticket_consumed", False)))
            / max(len(steps), 1)
        )

    def _mean(values):
        return round(sum(values) / len(values), 4) if values else 0.0

    decision_trace = getattr(block, "_decision_trace", None) or []
    loop_ledger = build_learning_loop_ledger(block)
    postreveal_candidate_audit = build_postreveal_candidate_audit(block)
    allow_gate_replay = build_allow_gate_replay(block)
    allow_family_audit = build_allow_family_audit(block)
    block._learning_loop_ledger = loop_ledger
    block._postreveal_candidate_audit = postreveal_candidate_audit
    block._allow_gate_replay = allow_gate_replay
    block._allow_family_audit = allow_family_audit
    prereveal_entries = [
        d for d in decision_trace
        if not bool(d.get("pre_post_reveal_phase", False))
    ]
    allow_attempt_entries = [
        d for d in decision_trace
        if d.get("phase") == "PRE_REVEAL_ALLOW"
    ]
    allow_preserved_entries = [
        d for d in allow_attempt_entries
        if bool(d.get("pre_productive_allow_preserved", False))
    ]
    allow_eligible_entries = [
        d for d in prereveal_entries
        if bool(d.get("pre_allow_eligible", False))
    ]
    allow_preserved_eligible_entries = [
        d for d in allow_eligible_entries
        if bool(d.get("pre_productive_allow_preserved", False))
    ]
    allow_entries = [
        d for d in decision_trace
        if d.get("wait_reason") == "WAIT_ALLOW_SAFE_DIAG"
    ]
    postreveal_entries = [
        d for d in decision_trace
        if bool(d.get("pre_post_reveal_phase", False))
    ]
    query_by_id = {
        int(getattr(qs, "query_id", qi)): qs
        for qi, qs in enumerate(teach_queries)
    }
    allow_query_ids = {
        int(d.get("query_id"))
        for d in allow_entries
        if d.get("query_id") is not None
    }
    allow_attempt_query_ids = {
        int(d.get("query_id"))
        for d in allow_attempt_entries
        if d.get("query_id") is not None
    }
    allow_queries = [
        query_by_id[qid]
        for qid in allow_query_ids
        if qid in query_by_id
    ]

    first_trace_by_qid = {}
    for d in decision_trace:
        qid = d.get("query_id")
        round_t = d.get("round_t", -1)
        if qid is None:
            continue
        prev = first_trace_by_qid.get(qid)
        if prev is None or round_t < prev.get("round_t", 10**9):
            first_trace_by_qid[qid] = d

    loop_both_ticket_qids = {
        int(qid)
        for qid, d in first_trace_by_qid.items()
        if bool(d.get("pre_both_tickets_available", False))
    }
    productive_reveal_qids = {
        int(getattr(ls, "query_id"))
        for ls in productive_reveals
        if getattr(ls, "query_id", None) is not None
    }
    contrastive_used_qids = {
        int(getattr(ls, "query_id"))
        for ls in teach_steps
        if getattr(ls, "query_id", None) is not None
        and bool(getattr(ls, "contrastive_ticket_consumed", False))
    }
    cue_after_reveal_qids = {
        int(d.get("query_id"))
        for d in postreveal_entries
        if d.get("query_id") is not None
        and d.get("chosen_action") in ("HIGHLIGHT", "MIX")
    }
    grace_consumed_qids = {
        int(d.get("query_id"))
        for d in decision_trace
        if d.get("query_id") is not None
        and d.get("wait_reason") == "WAIT_GRACE"
    }
    correct_after_feedback_qids = {
        int(getattr(ls, "query_id"))
        for ls in correct_pick_steps
        if getattr(ls, "query_id", None) is not None
        and getattr(ls, "semantic_credit_reason", "none") in ("after_reveal", "after_cue", "after_grace")
    }
    positive_used_qids = {
        int(getattr(ls, "query_id"))
        for ls in teach_steps
        if getattr(ls, "query_id", None) is not None
        and bool(getattr(ls, "positive_ticket_consumed", False))
    }
    loop_complete_qids = (
        allow_query_ids
        & productive_reveal_qids
        & contrastive_used_qids
        & cue_after_reveal_qids
        & correct_after_feedback_qids
        & positive_used_qids
    )

    g_consolidate_all = []
    g_consolidate_wait = []
    g_consolidate_hl = []
    g_consolidate_mix = []
    g_consolidate_ban = []
    g_consolidate_chosen = []
    g_consolidate_changes_best_action = 0
    chosen_has_max_gconsolidate = 0
    p_correct_2r_wait = []
    p_correct_2r_hl = []
    p_correct_2r_mix = []
    p_correct_2r_ban = []
    p_correct_2r_chosen = []
    allow_loop_values = [
        float(d.get("pre_allow_loop_value", 0.0))
        for d in allow_entries
    ]
    allow_attempt_loop_values = [
        float(d.get("pre_allow_loop_value", 0.0))
        for d in allow_attempt_entries
    ]
    allow_productive_mass = [
        float(d.get("pre_productive_mass_wait", 0.0))
        for d in allow_entries
    ]
    allow_attempt_productive_mass = [
        float(d.get("pre_productive_mass_wait", 0.0))
        for d in allow_attempt_entries
    ]
    allow_harm_mass = [
        float(d.get("pre_harm_mass_wait", 0.0))
        for d in allow_entries
    ]
    allow_attempt_harm_mass = [
        float(d.get("pre_harm_mass_wait", 0.0))
        for d in allow_attempt_entries
    ]
    allow_attempt_expected_damage = [
        float(d.get("pre_expected_damage_wait", 0.0))
        for d in allow_attempt_entries
    ]
    allow_attempt_p_survive = [
        float(d.get("pre_allow_p_survive", 0.0))
        for d in allow_attempt_entries
    ]

    for entry in postreveal_entries:
        candidates = (entry.get("scoring") or {}).get("candidates") or []
        chosen_detail = _chosen_candidate_detail(entry, candidates)
        max_g = 0.0
        best_with = None
        best_without = None
        for cand in candidates:
            value = float(
                cand.get(
                    "g_consolidate",
                    cand.get(
                        "postreveal_consolidation_effective",
                        cand.get("postreveal_consolidation_value", 0.0),
                    ),
                )
            )
            act = cand.get("action", "WAIT")
            g_consolidate_all.append(value)
            max_g = max(max_g, value)
            p2r = float(cand.get("postreveal_p_correct_2r", 0.0))
            if act == "WAIT":
                g_consolidate_wait.append(value)
                p_correct_2r_wait.append(p2r)
            elif act == "HIGHLIGHT":
                g_consolidate_hl.append(value)
                p_correct_2r_hl.append(p2r)
            elif act == "MIX":
                g_consolidate_mix.append(value)
                p_correct_2r_mix.append(p2r)
            elif act == "BAN":
                g_consolidate_ban.append(value)
                p_correct_2r_ban.append(p2r)
            if (
                best_with is None
                or float(cand.get("q_use", float("-inf"))) > float(best_with.get("q_use", float("-inf")))
            ):
                best_with = cand
            if (
                best_without is None
                or (
                    float(
                        cand.get(
                            "q_use_without_consolidate",
                            float(cand.get("q_use", float("-inf")))
                            - float(cand.get("q_use_consolidate_delta", 0.0)),
                        )
                    )
                ) > (
                    float(
                        best_without.get(
                            "q_use_without_consolidate",
                            float(best_without.get("q_use", float("-inf")))
                            - float(best_without.get("q_use_consolidate_delta", 0.0)),
                        )
                    )
                )
            ):
                best_without = cand
        if chosen_detail is not None:
            chosen_g = float(
                chosen_detail.get(
                    "g_consolidate",
                    chosen_detail.get(
                        "postreveal_consolidation_effective",
                        chosen_detail.get("postreveal_consolidation_value", 0.0),
                    ),
                )
            )
            g_consolidate_chosen.append(chosen_g)
            p_correct_2r_chosen.append(float(chosen_detail.get("postreveal_p_correct_2r", 0.0)))
            if chosen_g >= max_g - 1e-9:
                chosen_has_max_gconsolidate += 1
        if best_with is not None and best_without is not None:
            if _candidate_identity(best_with) != _candidate_identity(best_without):
                g_consolidate_changes_best_action += 1

    allow_attempt_reason_counts = defaultdict(int)
    for d in allow_attempt_entries:
        allow_attempt_reason_counts[str(d.get("pre_productive_allow_reason", "unknown"))] += 1
    allow_reject_reason_counts = defaultdict(int)
    allow_reject_entries = [
        d for d in prereveal_entries
        if str(d.get("pre_allow_reject_reason", "none")) not in ("ALLOW_PRESERVED", "none", "")
    ]
    for d in allow_reject_entries:
        allow_reject_reason_counts[str(d.get("pre_allow_reject_reason", "unknown"))] += 1

    allow_replay_gate_labels = {
        "G0_current_controlled_v2": "G0_Current",
        "G1_permissive_prod": "G1_PermissiveProd",
        "G2_ratio_gate": "G2_RatioGate",
        "G3_highrisk_guard": "G3_HighRiskGuard",
        "G4_postvalue_gate": "G4_PostValueGate",
        "G5_combined": "G5_Combined",
    }
    allow_replay_metrics = {}
    replay_state_count = len(allow_gate_replay)
    for gate_key, gate_label in allow_replay_gate_labels.items():
        allowed_rows = [row for row in allow_gate_replay if bool(row.get(gate_key, False))]
        allow_replay_metrics[f"AllowReplay_{gate_label}_EligibleStateCount"] = replay_state_count
        allow_replay_metrics[f"AllowReplay_{gate_label}_WouldAllowCount"] = len(allowed_rows)
        allow_replay_metrics[f"AllowReplay_{gate_label}_WouldAllowRate"] = (
            len(allowed_rows) / max(replay_state_count, 1)
        )
        allow_replay_metrics[f"AllowReplay_{gate_label}_MeanPProd"] = _mean(
            [float(row.get("p_prod", 0.0)) for row in allowed_rows]
        )
        allow_replay_metrics[f"AllowReplay_{gate_label}_MeanHarm"] = _mean(
            [float(row.get("harm_mass", 0.0)) for row in allowed_rows]
        )
        allow_replay_metrics[f"AllowReplay_{gate_label}_MeanExpectedDamage"] = _mean(
            [float(row.get("expected_damage", 0.0)) for row in allowed_rows]
        )
        allow_replay_metrics[f"AllowReplay_{gate_label}_PredictedSafeDiagRevealRate"] = _mean(
            [float(row.get("p_safe_diag", 0.0)) for row in allowed_rows]
        )
        allow_replay_metrics[f"AllowReplay_{gate_label}_PredictedHighRiskRevealRate"] = _mean(
            [float(row.get("p_highrisk", 0.0)) for row in allowed_rows]
        )
        allow_replay_metrics[f"AllowReplay_{gate_label}_PredictedFarWrongRevealRate"] = _mean(
            [float(row.get("p_farwrong", 0.0)) for row in allowed_rows]
        )
        allow_replay_metrics[f"AllowReplay_{gate_label}_PredictedPostRevealCueValue"] = _mean(
            [float(row.get("post_reveal_best_value_estimate", 0.0)) for row in allowed_rows]
        )
        allow_replay_metrics[f"AllowReplay_{gate_label}_PredictedLoopCompleteUpperBound"] = _mean(
            [float(row.get("loop_upper_bound_proxy", 0.0)) for row in allowed_rows]
        )

    allow_family_labels = [
        "ALLOW_CRITICAL",
        "HARM_DOMINATED",
        "HIGHRISK_DOMINATED",
        "FAR_DOMINATED",
        "NO_PRODUCTIVE_OPPORTUNITY",
        "TICKET_BLOCKED",
        "ROUND_BLOCKED",
        "BORING_MASTERY",
        "NOT_PRE_REVEAL",
    ]
    phase_blind_allow_family_labels = [
        "ALLOW_CRITICAL_STAR",
        "HARM_DOMINATED",
        "HIGHRISK_DOMINATED",
        "FAR_DOMINATED",
        "NO_PRODUCTIVE_OPPORTUNITY",
        "TICKET_BLOCKED",
        "ROUND_BLOCKED",
        "BORING_MASTERY",
    ]
    phase_reject_labels = [
        "ALLOW_PHASE_ACTIVE",
        "NOT_PRE_REVEAL_POST_REVEAL_ALREADY",
        "NOT_PRE_REVEAL_NO_SAFE_DIAG_IN_MENU",
        "NOT_PRE_REVEAL_PHASE_INFER_DEFAULT",
        "NOT_PRE_REVEAL_PROTECT_PHASE",
        "NOT_PRE_REVEAL_BORING_PHASE",
    ]
    allow_family_metrics = {}
    family_state_count = len(allow_family_audit)
    for family_label in allow_family_labels:
        family_rows = [
            row for row in allow_family_audit
            if str(row.get("family", "")) == family_label
        ]
        allow_family_metrics[f"AllowFamily_{family_label}_StateCount"] = len(family_rows)
        allow_family_metrics[f"AllowFamily_{family_label}_Rate"] = (
            len(family_rows) / max(family_state_count, 1)
        )
        allow_family_metrics[f"AllowFamily_{family_label}_PreserveRate"] = _mean(
            [1.0 if bool(row.get("allow_preserved", False)) else 0.0 for row in family_rows]
        )
        allow_family_metrics[f"AllowFamily_{family_label}_ProductiveRevealRate"] = _mean(
            [1.0 if bool(row.get("productive_reveal_after_state", False)) else 0.0 for row in family_rows]
        )
        allow_family_metrics[f"AllowFamily_{family_label}_CueRate"] = _mean(
            [1.0 if bool(row.get("cue_after_state", False)) else 0.0 for row in family_rows]
        )
        allow_family_metrics[f"AllowFamily_{family_label}_CorrectRate"] = _mean(
            [1.0 if bool(row.get("correct_after_state", False)) else 0.0 for row in family_rows]
        )
        allow_family_metrics[f"AllowFamily_{family_label}_LoopCompleteRate"] = _mean(
            [1.0 if bool(row.get("loop_complete_after_state", False)) else 0.0 for row in family_rows]
        )

    prereveal_family_metrics = {}
    for family_label in PREREVEAL_FAMILY_LABELS:
        family_rows = [
            row for row in allow_family_audit
            if str(row.get("family_split", "")) == family_label
        ]
        prereveal_family_metrics[f"PreRevealFamily_{family_label}_StateCount"] = len(family_rows)
        prereveal_family_metrics[f"PreRevealFamily_{family_label}_Rate"] = (
            len(family_rows) / max(family_state_count, 1)
        )
        prereveal_family_metrics[f"PreRevealFamily_{family_label}_MeanPProd"] = _mean(
            [float(row.get("p_prod_total", 0.0)) for row in family_rows]
        )
        prereveal_family_metrics[f"PreRevealFamily_{family_label}_MeanHarmMass"] = _mean(
            [float(row.get("harm_mass", 0.0)) for row in family_rows]
        )
        prereveal_family_metrics[f"PreRevealFamily_{family_label}_MeanSafeDiagQualityGap"] = _mean(
            [float(row.get("safe_diag_quality_gap", 0.0)) for row in family_rows]
        )
        prereveal_family_metrics[f"PreRevealFamily_{family_label}_MeanPCorrectWAIT"] = _mean(
            [float(row.get("p_correct_wait", 0.0)) for row in family_rows]
        )
        prereveal_family_metrics[f"PreRevealFamily_{family_label}_AllowPreserveRate"] = _mean(
            [1.0 if bool(row.get("allow_preserved", False)) else 0.0 for row in family_rows]
        )
        prereveal_family_metrics[f"PreRevealFamily_{family_label}_ProductiveRevealRate"] = _mean(
            [1.0 if bool(row.get("productive_reveal_after_state", False)) else 0.0 for row in family_rows]
        )
        prereveal_family_metrics[f"PreRevealFamily_{family_label}_ContrastiveTicketUsedRate"] = _mean(
            [1.0 if bool(row.get("contrastive_after_state", False)) else 0.0 for row in family_rows]
        )
        prereveal_family_metrics[f"PreRevealFamily_{family_label}_CueAfterRevealRate"] = _mean(
            [1.0 if bool(row.get("cue_after_state", False)) else 0.0 for row in family_rows]
        )
        prereveal_family_metrics[f"PreRevealFamily_{family_label}_GraceConsumedRate"] = _mean(
            [1.0 if bool(row.get("grace_after_state", False)) else 0.0 for row in family_rows]
        )
        prereveal_family_metrics[f"PreRevealFamily_{family_label}_CorrectAfterCueGraceRate"] = _mean(
            [1.0 if bool(row.get("correct_after_state", False)) else 0.0 for row in family_rows]
        )
        prereveal_family_metrics[f"PreRevealFamily_{family_label}_PositiveTicketUsedRate"] = _mean(
            [1.0 if bool(row.get("positive_ticket_after_state", False)) else 0.0 for row in family_rows]
        )
        prereveal_family_metrics[f"PreRevealFamily_{family_label}_LoopCompleteRate"] = _mean(
            [1.0 if bool(row.get("loop_complete_after_state", False)) else 0.0 for row in family_rows]
        )
        prereveal_family_metrics[f"PreRevealFamily_{family_label}_TeachDamageMean"] = _mean(
            [float(row.get("damage_after_state", 0.0)) for row in family_rows]
        )
        prereveal_family_metrics[f"PreRevealFamily_{family_label}_DeathBeforeCorrectRate"] = _mean(
            [1.0 if bool(row.get("death_before_correct_after_state", False)) else 0.0 for row in family_rows]
        )

    for family_label in phase_blind_allow_family_labels:
        family_rows = [
            row for row in allow_family_audit
            if str(row.get("phase_blind_family", "")) == family_label
        ]
        allow_family_metrics[f"PhaseBlindAllowFamily_{family_label}_StateCount"] = len(family_rows)
        allow_family_metrics[f"PhaseBlindAllowFamily_{family_label}_Rate"] = (
            len(family_rows) / max(family_state_count, 1)
        )

    for reject_label in phase_reject_labels:
        reject_rows = [
            row for row in allow_family_audit
            if str(row.get("phase_reject_reason", "")) == reject_label
        ]
        allow_family_metrics[f"AllowPhaseReject_{reject_label}_StateCount"] = len(reject_rows)
        allow_family_metrics[f"AllowPhaseReject_{reject_label}_Rate"] = (
            len(reject_rows) / max(family_state_count, 1)
        )

    missed_allow_rows = [
        row for row in allow_family_audit
        if bool(row.get("missed_allow_critical", False))
    ]
    allow_family_metrics["PhaseBlind_MissedAllowCritical_Count"] = len(missed_allow_rows)
    allow_family_metrics["PhaseBlind_MissedAllowCritical_Rate"] = (
        len(missed_allow_rows) / max(family_state_count, 1)
    )
    allow_family_metrics["PhaseBlind_MissedAllowMeanPProd"] = _mean(
        [float(row.get("p_prod_total", 0.0)) for row in missed_allow_rows]
    )
    allow_family_metrics["PhaseBlind_MissedAllowMeanHarm"] = _mean(
        [float(row.get("harm_mass", 0.0)) for row in missed_allow_rows]
    )
    allow_family_metrics["PhaseBlind_MissedAllowMeanSafeDiagQualityGap"] = _mean(
        [float(row.get("safe_diag_quality_gap", 0.0)) for row in missed_allow_rows]
    )
    allow_family_metrics["PhaseBlind_MissedAllowMeanRoundsLeft"] = _mean(
        [float(row.get("rounds_left", 0.0)) for row in missed_allow_rows]
    )
    allow_family_metrics["PhaseBlind_MissedAllowMeanBothTickets"] = _mean(
        [1.0 if bool(row.get("both_tickets_available", False)) else 0.0 for row in missed_allow_rows]
    )

    confusion_phases = [
        "PRE_REVEAL_ALLOW",
        "PROTECT",
        "BORING_ESCAPE",
        "DEFAULT",
        "POST_REVEAL_CONSOLIDATE",
        "POST_REVEAL_PROTECT_AND_CUE",
        "GRACE_WAIT",
    ]
    confusion_cols = ["ALLOW_CRITICAL_STAR"] + [
        label for label in phase_blind_allow_family_labels
        if label != "ALLOW_CRITICAL_STAR"
    ]
    for phase_label in confusion_phases:
        phase_rows = [
            row for row in allow_family_audit
            if str(row.get("phase", "")) == phase_label
        ]
        allow_family_metrics[f"AllowPhase_{phase_label}_StateCount"] = len(phase_rows)
        allow_family_metrics[f"AllowPhase_{phase_label}_Rate"] = (
            len(phase_rows) / max(family_state_count, 1)
        )
        allow_family_metrics[f"AllowPhase_{phase_label}_PhaseBlindAllowRate"] = _mean(
            [
                1.0 if str(row.get("phase_blind_family", "")) == "ALLOW_CRITICAL_STAR" else 0.0
                for row in phase_rows
            ]
        )
        for col_label in confusion_cols:
            col_token = _metric_token(col_label)
            allow_family_metrics[
                f"AllowPhaseConfusion_{phase_label}__{col_token}_Count"
            ] = sum(
                1 for row in phase_rows
                if str(row.get("phase_blind_family", "")) == col_label
            )

    pprod_proxy_metrics = {
        "AllowFamilyAuditStateCount": family_state_count,
        "PProdTotalMean": _mean([float(row.get("p_prod_total", 0.0)) for row in allow_family_audit]),
        "PProdSafeComponentMean": _mean([float(row.get("p_prod_safe_component", 0.0)) for row in allow_family_audit]),
        "PProdBoundedComponentMean": _mean([float(row.get("p_prod_bounded_component", 0.0)) for row in allow_family_audit]),
        "PProdSafeShareMean": _mean([float(row.get("p_prod_safe_share", 0.0)) for row in allow_family_audit]),
        "AllowSafeDiagQualityGapMean": _mean([float(row.get("safe_diag_quality_gap", 0.0)) for row in allow_family_audit]),
        "AllowCompetingHarmMassMean": _mean([float(row.get("competing_harm_mass", 0.0)) for row in allow_family_audit]),
        "AllowHarmCompetitionGapMean": _mean([float(row.get("harm_competition_gap", 0.0)) for row in allow_family_audit]),
        "AllowCritical_PProdMean": _mean([
            float(row.get("p_prod_total", 0.0))
            for row in allow_family_audit
            if str(row.get("family", "")) == "ALLOW_CRITICAL"
        ]),
        "AllowCritical_HarmMean": _mean([
            float(row.get("harm_mass", 0.0))
            for row in allow_family_audit
            if str(row.get("family", "")) == "ALLOW_CRITICAL"
        ]),
        "AllowCritical_SafeDiagQualityGapMean": _mean([
            float(row.get("safe_diag_quality_gap", 0.0))
            for row in allow_family_audit
            if str(row.get("family", "")) == "ALLOW_CRITICAL"
        ]),
        "AllowCritical_CompetingHarmMassMean": _mean([
            float(row.get("competing_harm_mass", 0.0))
            for row in allow_family_audit
            if str(row.get("family", "")) == "ALLOW_CRITICAL"
        ]),
        "PhaseBlind_ALLOW_CRITICAL_StateCount": sum(
            1 for row in allow_family_audit
            if str(row.get("phase_blind_family", "")) == "ALLOW_CRITICAL_STAR"
        ),
        "PhaseBlind_ALLOW_CRITICAL_Rate": _mean([
            1.0 if str(row.get("phase_blind_family", "")) == "ALLOW_CRITICAL_STAR" else 0.0
            for row in allow_family_audit
        ]),
        "NativeLikeAllow_StateCount": sum(
            1 for row in allow_family_audit
            if str(row.get("family_split", "")) == FAMILY_NATIVE_LIKE_ALLOW
        ),
        "NativeLikeAllow_Rate": _mean([
            1.0 if str(row.get("family_split", "")) == FAMILY_NATIVE_LIKE_ALLOW else 0.0
            for row in allow_family_audit
        ]),
        "NativeLikeAllow_LoopCompleteRate": _mean([
            1.0 if bool(row.get("loop_complete_after_state", False)) else 0.0
            for row in allow_family_audit
            if str(row.get("family_split", "")) == FAMILY_NATIVE_LIKE_ALLOW
        ]),
        "NativeLikeAllow_Damage": _mean([
            float(row.get("damage_after_state", 0.0))
            for row in allow_family_audit
            if str(row.get("family_split", "")) == FAMILY_NATIVE_LIKE_ALLOW
        ]),
        "NativeLikeAllow_DeathBeforeCorrectRate": _mean([
            1.0 if bool(row.get("death_before_correct_after_state", False)) else 0.0
            for row in allow_family_audit
            if str(row.get("family_split", "")) == FAMILY_NATIVE_LIKE_ALLOW
        ]),
        "MixedProdHarm_StateCount": sum(
            1 for row in allow_family_audit
            if str(row.get("family_split", "")) == FAMILY_MIXED_PROD_HARM
        ),
        "MixedProdHarm_Rate": _mean([
            1.0 if str(row.get("family_split", "")) == FAMILY_MIXED_PROD_HARM else 0.0
            for row in allow_family_audit
        ]),
        "MixedProdHarm_LoopCompleteRate": _mean([
            1.0 if bool(row.get("loop_complete_after_state", False)) else 0.0
            for row in allow_family_audit
            if str(row.get("family_split", "")) == FAMILY_MIXED_PROD_HARM
        ]),
        "MixedProdHarm_Damage": _mean([
            float(row.get("damage_after_state", 0.0))
            for row in allow_family_audit
            if str(row.get("family_split", "")) == FAMILY_MIXED_PROD_HARM
        ]),
        "MixedProdHarm_DeathBeforeCorrectRate": _mean([
            1.0 if bool(row.get("death_before_correct_after_state", False)) else 0.0
            for row in allow_family_audit
            if str(row.get("family_split", "")) == FAMILY_MIXED_PROD_HARM
        ]),
    }

    ledger_break_counts = defaultdict(int)
    for row in loop_ledger:
        ledger_break_counts[str(row.get("loop_break_stage", "unknown"))] += 1
    chosen_candidate_audit = [
        row for row in postreveal_candidate_audit
        if bool(row.get("chosen", False))
    ]
    q_consolidate_delta_all = [
        float(row.get("q_consolidate_delta", 0.0))
        for row in postreveal_candidate_audit
    ]
    q_consolidate_delta_chosen = [
        float(row.get("q_consolidate_delta", 0.0))
        for row in chosen_candidate_audit
    ]
    q_without_chosen = [
        float(row.get("q_without_consolidate", 0.0))
        for row in chosen_candidate_audit
    ]
    q_with_chosen = [
        float(row.get("q_with_consolidate", 0.0))
        for row in chosen_candidate_audit
    ]

    result = {
        "WAITRate": wait_count / total_actions,
        "BANRate": ban_count / total_actions,
        "HIGHLIGHTRate": hl_count / total_actions,
        "MIXRate": mix_count / total_actions,
        "HLCount": hl_count,
        "MIXCount": mix_count,
        "PostRevHLCount": n_post_reveal_hl,
        "PostHLCorrectCount": n_post_hl_correct,
        "PostHLSelfCorrectCount": n_post_hl_self_correct,
        "HighlightSelfCorrectRate": hl_self_correct / max(hl_total, 1),
        "HighlightTotal": hl_total,
        "HighRiskBanRate": high_risk_ban_rate,
        "SafeDiagBanRate": safe_diag_ban_rate,
        "ProtectableSafeDiagBanRate": round(protectable_ban_rate, 4),
        "LateOrExhaustedSafeDiagBanRate": round(late_ban_rate, 4),
        "SafeDiagTotal": safe_diag_total,
        "SafeDiagBanned": safe_diag_banned,
        "SafeDiagProtectableTotal": safe_diag_protectable_total,
        "SafeDiagProtectableBanned": safe_diag_protectable_banned,
        "SafeDiagLateBanned": safe_diag_late_banned,
        "HighRiskTotal": high_risk_total,
        "HighRiskBanned": high_risk_banned,
        "PedagogicalSelectivity": round(ped_selectivity, 4),
        "MeanPSafeDiag": round(mean_p_safe_diag, 4),
        "MeanPHighRisk": round(mean_p_high_risk, 4),
        "DiagRevealQueries": n_diag_reveal_queries,
        "SafeDiagRevealThenCorrectRate": n_diag_reveal_then_correct / max(n_diag_reveal_queries, 1),
        "SafeDiagRevealThenSelfCorrectRate": n_diag_reveal_then_self_correct / max(n_diag_reveal_queries, 1),
        "RevThenNaturalSelfCorrectRate": n_diag_reveal_then_self_correct / max(n_diag_reveal_queries, 1),
        "RevThenUnassistedCorrectRate": n_diag_reveal_then_self_correct / max(n_diag_reveal_queries, 1),
        "DiagRevealWastedRate": n_diag_reveal_wasted / max(n_diag_reveal_queries, 1),
        "PostRevealHighlightRate": postreveal_hl_rate,
        "PostHighlightCorrectRate": n_post_hl_correct / max(n_post_reveal_hl, 1),
        "PostHighlightSelfCorrectRate": n_post_hl_self_correct / max(n_post_reveal_hl, 1),
        "RepeatedWrongAfterRevealRate": n_repeated_wrong_after_reveal / max(n_diag_reveal_queries, 1),
        "GraceRoundCount": grace_count,
        "GraceSetCount": grace_set,
        "GraceEligibleNextRoundCount": grace_eligible,
        "GraceNextTutorCalledCount": grace_next_tutor,
        "GraceChosenWAITCount": grace_chosen_wait,
        "GraceChosenOverrideCount": grace_chosen_override,
        "GraceConsumedCount": grace_consumed,
        "GraceOverrideCount": grace_override,
        "GraceBlockedByProtectCount": grace_blocked_protect,
        "GraceBlockedByDeadlineCount": grace_blocked_deadline,
        "GraceDidNotReachTutorDecisionCount": grace_no_tutor,
        "GraceLostBecauseQuerySucceededCount": grace_lost_success,
        "GraceLostBecauseWrongTerminalCount": grace_lost_wrong_terminal,
        "GraceLostBecauseMaxRoundCount": grace_lost_max_round,
        "GraceFlagResetWithoutConsumptionCount": grace_flag_reset,
        "GraceConsumedRate": grace_consumed / max(grace_count, 1) if grace_count > 0 else 0.0,
        # Phase 6I: opportunity-conditioned metrics
        "SafeDiagOpp_WaitRate": sd_opp_wait / max(sd_opp_total, 1),
        "SafeDiagOpp_BanRate": sd_opp_ban / max(sd_opp_total, 1),
        "HighRiskOpp_BanRate": hr_opp_ban / max(hr_opp_total, 1),
        "HighRiskOpp_WaitRate": hr_opp_wait / max(hr_opp_total, 1),
        "PostReveal_HLRate": postreveal_hl_rate,
        "PostReveal_MIXRate": postreveal_mix_rate,
        "PostReveal_WAITRate": postreveal_wait_rate,
        "PostReveal_HLGeneratedRate": postrev_hl_generated / max(postrev_decisions, 1),
        "PostReveal_MIXGeneratedRate": postrev_mix_generated / max(postrev_decisions, 1),
        "PostReveal_HLBeatsWAITRate": postrev_hl_beats_wait / max(postrev_decisions, 1),
        "PostReveal_MIXBeatsWAITRate": postrev_mix_beats_wait / max(postrev_decisions, 1),
        "PostReveal_WAITBestByQRate": postrev_wait_best_by_q / max(postrev_decisions, 1),
        "PostReveal_WAITChosenWhenNonWaitBetterRate": (
            postrev_wait_when_nonwait_better / max(postrev_decisions, 1)
        ),
        "PostReveal_QRegretToBestNonWaitMean": (
            round(sum(postrev_q_regret) / len(postrev_q_regret), 4)
            if postrev_q_regret else 0.0
        ),
        "PostReveal_QWAITMean": round(sum(postrev_q_wait) / len(postrev_q_wait), 4) if postrev_q_wait else 0.0,
        "PostReveal_QHLMean": round(sum(postrev_q_hl) / len(postrev_q_hl), 4) if postrev_q_hl else 0.0,
        "PostReveal_QMIXMean": round(sum(postrev_q_mix) / len(postrev_q_mix), 4) if postrev_q_mix else 0.0,
        "PostReveal_QBANMean": round(sum(postrev_q_ban) / len(postrev_q_ban), 4) if postrev_q_ban else 0.0,
        "PostReveal_HL_DShiftMean": round(sum(postrev_hl_dshift) / len(postrev_hl_dshift), 4) if postrev_hl_dshift else 0.0,
        "PostReveal_MIX_DShiftMean": round(sum(postrev_mix_dshift) / len(postrev_mix_dshift), 4) if postrev_mix_dshift else 0.0,
        "PostReveal_HL_EffectiveDShiftMean": round(sum(postrev_hl_eff_dshift) / len(postrev_hl_eff_dshift), 4) if postrev_hl_eff_dshift else 0.0,
        "PostReveal_MIX_EffectiveDShiftMean": round(sum(postrev_mix_eff_dshift) / len(postrev_mix_eff_dshift), 4) if postrev_mix_eff_dshift else 0.0,
        "PostReveal_HL_CostMean": round(sum(postrev_hl_cost) / len(postrev_hl_cost), 4) if postrev_hl_cost else 0.0,
        "PostReveal_MIX_CostMean": round(sum(postrev_mix_cost) / len(postrev_mix_cost), 4) if postrev_mix_cost else 0.0,
        "PostReveal_HL_Top1FlipMean": round(sum(postrev_hl_top1flip) / len(postrev_hl_top1flip), 4) if postrev_hl_top1flip else 0.0,
        "PostReveal_MIX_Top1FlipMean": round(sum(postrev_mix_top1flip) / len(postrev_mix_top1flip), 4) if postrev_mix_top1flip else 0.0,
        "PostReveal_HL_MarginGainMean": round(sum(postrev_hl_margin) / len(postrev_hl_margin), 4) if postrev_hl_margin else 0.0,
        "PostReveal_MIX_MarginGainMean": round(sum(postrev_mix_margin) / len(postrev_mix_margin), 4) if postrev_mix_margin else 0.0,
        "PostReveal_HL_HighRiskDropMean": round(sum(postrev_hl_hrdrop) / len(postrev_hl_hrdrop), 4) if postrev_hl_hrdrop else 0.0,
        "PostReveal_MIX_HighRiskDropMean": round(sum(postrev_mix_hrdrop) / len(postrev_mix_hrdrop), 4) if postrev_mix_hrdrop else 0.0,
        "PostReveal_HL_SameWrongDropMean": round(sum(postrev_hl_samewrong) / len(postrev_hl_samewrong), 4) if postrev_hl_samewrong else 0.0,
        "PostReveal_MIX_SameWrongDropMean": round(sum(postrev_mix_samewrong) / len(postrev_mix_samewrong), 4) if postrev_mix_samewrong else 0.0,
        "PostReveal_HL_BadMassDropMean": round(sum(postrev_hl_badmass) / len(postrev_hl_badmass), 4) if postrev_hl_badmass else 0.0,
        "PostReveal_MIX_BadMassDropMean": round(sum(postrev_mix_badmass) / len(postrev_mix_badmass), 4) if postrev_mix_badmass else 0.0,
        "PostReveal_HL_RemovedBadMassMean": round(sum(postrev_hl_removedbad) / len(postrev_hl_removedbad), 4) if postrev_hl_removedbad else 0.0,
        "PostReveal_MIX_RemovedBadMassMean": round(sum(postrev_mix_removedbad) / len(postrev_mix_removedbad), 4) if postrev_mix_removedbad else 0.0,
        "PostReveal_HL_RemovedProbMassMean": round(sum(postrev_hl_removedprob) / len(postrev_hl_removedprob), 4) if postrev_hl_removedprob else 0.0,
        "PostReveal_MIX_RemovedProbMassMean": round(sum(postrev_mix_removedprob) / len(postrev_mix_removedprob), 4) if postrev_mix_removedprob else 0.0,
        # Phase 6I.1 P0: cue outcome four-way classification
        "PostCueGuidedSCCount": cue_guided_sc,
        "PostCueGuidedSCRate": cue_guided_sc / max(cue_total_picks, 1),
        "PostCueStructProtectCount": structural_protect_correct,
        "PostCueStructProtectRate": structural_protect_correct / max(cue_total_picks, 1),
        "PostCuePedagogicalSuccessCount": pedagogical_success,
        "PostCuePedagogicalSuccessRate": pedagogical_success / max(cue_total_picks, 1),
        "PostCueAnswerLeakageCount": answer_leakage,
        "PostCueAnswerLeakageRate": answer_leakage / max(cue_total_picks, 1),
        "PostCueWrongPickRate": cue_wrong_picks / max(cue_total_picks, 1),
        "PostCueTotalPicks": cue_total_picks,
        "PostCueImmediateCorrectRate": cue_immediate_correct / max(cue_events, 1),
        "PostCueImmediateWrongRate": cue_immediate_wrong / max(cue_events, 1),
        "PostCueRefreshRate": cue_refresh / max(cue_events, 1),
        "CueThenGraceWAITRate": cue_grace_wait / max(cue_events, 1),
        "CueThenGraceCorrectRate": cue_then_grace_correct / max(cue_events, 1),
        "CueThenGraceWrongRate": cue_then_grace_wrong / max(cue_events, 1),
        "CueThenGraceRefreshRate": cue_then_grace_refresh / max(cue_events, 1),
        "CueTrajectorySuccessWithin2RoundsRate": cue_traj_success_2rounds / max(cue_events, 1),
        "CueTrajectoryFailureWithin2RoundsRate": cue_traj_fail_2rounds / max(cue_events, 1),
        "CueTrajectoryTerminalFailureRate": cue_traj_terminal_fail / max(cue_events, 1),
        "CueTrajectoryRepeatedWrongRate": cue_traj_repeated_wrong / max(cue_events, 1),
        "CueTrajectoryHighRiskFailureRate": cue_traj_highrisk_fail / max(cue_events, 1),
        "CueTrajectoryFarWrongFailureRate": cue_traj_farwrong_fail / max(cue_events, 1),
        "PostCueWrongPick_SameWrongRate": cue_wrong_same / max(cue_total_picks, 1),
        "PostCueWrongPick_DifferentSafeDiagRate": cue_wrong_diff_safe / max(cue_total_picks, 1),
        "PostCueWrongPick_BoundedDiagRate": cue_wrong_bounded / max(cue_total_picks, 1),
        "PostCueWrongPick_HighRiskRate": cue_wrong_highrisk / max(cue_total_picks, 1),
        "PostCueWrongPick_FarWrongRate": cue_wrong_far / max(cue_total_picks, 1),
        "PostCueWrongPick_OtherRate": cue_wrong_other / max(cue_total_picks, 1),
        "ForceBest_SelectedHLCount": forcebest_selected_hl,
        "ForceBest_SelectedMIXCount": forcebest_selected_mix,
        "ForceBest_SelectedNoneCount": forcebest_selected_none,
        "PostReveal_PositiveCueOppCount": postrev_positive_cue_opp,
        "PostReveal_PositiveMixOppCount": postrev_positive_mix_opp,
        "PostReveal_PositiveHLOppCount": postrev_positive_hl_opp,
        "MIXChosenCount": mix_chosen_count,
        "MIXBanTargetWasLastWrongRate": mix_ban_lastwrong / max(mix_chosen_count, 1),
        "MIXBanTargetWasHighRiskRate": mix_ban_highrisk / max(mix_chosen_count, 1),
        "MIXBanTargetWasSafeDiagRate": mix_ban_safe_diag / max(mix_chosen_count, 1),
        "MIXBanTargetWasTopProbWrongRate": mix_ban_topwrong / max(mix_chosen_count, 1),
        "MIXBanTargetWasFarWrongRate": mix_ban_farwrong / max(mix_chosen_count, 1),
        "MIXBanTargetWasCorrectRate": mix_ban_correct / max(mix_chosen_count, 1),
        "MIXBanTargetMeanPolicyMass": round(sum(mix_ban_policy_mass) / len(mix_ban_policy_mass), 4) if mix_ban_policy_mass else 0.0,
        "MIXBanTargetMeanBadness": round(sum(mix_ban_badness) / len(mix_ban_badness), 4) if mix_ban_badness else 0.0,
        "MIXRemovedProbMassMean": round(sum(mix_removed_prob) / len(mix_removed_prob), 4) if mix_removed_prob else 0.0,
        "MIXRemovedBadMassMean": round(sum(mix_removed_bad) / len(mix_removed_bad), 4) if mix_removed_bad else 0.0,
        "MIXBadMassDropMean": round(sum(mix_badmass_drop) / len(mix_badmass_drop), 4) if mix_badmass_drop else 0.0,
        "MIXDeltaPcorrectMean": round(sum(mix_delta_p) / len(mix_delta_p), 4) if mix_delta_p else 0.0,
        "MIXMarginGainMean": round(sum(mix_margin_gain) / len(mix_margin_gain), 4) if mix_margin_gain else 0.0,
        "MIXRemovedTargetRegretMean": round(sum(mix_removed_target_regret) / len(mix_removed_target_regret), 4) if mix_removed_target_regret else 0.0,
        "MIXNetTargetRegretMean": round(sum(mix_net_target_regret) / len(mix_net_target_regret), 4) if mix_net_target_regret else 0.0,
        "MIXOracleRemovedMassMean": round(sum(mix_oracle_removed_mass) / len(mix_oracle_removed_mass), 4) if mix_oracle_removed_mass else 0.0,
        "MIXOracleNetBadMassDropMean": round(sum(mix_oracle_net_drop) / len(mix_oracle_net_drop), 4) if mix_oracle_net_drop else 0.0,
        "MIXTargetMatchesRemovedOracleRate": mix_matches_removed_oracle / max(mix_chosen_count, 1),
        "MIXTargetMatchesNetOracleRate": mix_matches_net_oracle / max(mix_chosen_count, 1),
        "MIXDirectSelectorAppliedRate": mix_direct_selector_applied / max(mix_chosen_count, 1),
        "MIXDirectSelectedNetHarmDropMean": round(sum(mix_direct_selected_net_harm) / len(mix_direct_selected_net_harm), 4) if mix_direct_selected_net_harm else 0.0,
        "MIXDirectOracleNetHarmDropMean": round(sum(mix_direct_oracle_net_harm) / len(mix_direct_oracle_net_harm), 4) if mix_direct_oracle_net_harm else 0.0,
        "MIXDirectNetTargetRegretMean": round(sum(mix_direct_net_target_regret) / len(mix_direct_net_target_regret), 4) if mix_direct_net_target_regret else 0.0,
        "MIXJointGateAppliedRate": mix_joint_gate_applied / max(mix_chosen_count, 1),
        "MIXJointGateReplacedRate": mix_joint_gate_replaced / max(mix_chosen_count, 1),
        "MIXJointTargetRegretMean": round(sum(mix_joint_target_regret) / len(mix_joint_target_regret), 4) if mix_joint_target_regret else 0.0,
        "MIXJointHighlightRegretMean": round(sum(mix_joint_highlight_regret) / len(mix_joint_highlight_regret), 4) if mix_joint_highlight_regret else 0.0,
        "MIXJointRegretMean": round(sum(mix_joint_regret) / len(mix_joint_regret), 4) if mix_joint_regret else 0.0,
        "MIXJointInteractionRegretMean": round(sum(mix_joint_interaction_regret) / len(mix_joint_interaction_regret), 4) if mix_joint_interaction_regret else 0.0,
        "JointReplayEligibleCount": joint_replay_eligible_count,
        "JointReplayEligibleRegretMean": round(sum(joint_replay_eligible_regret) / len(joint_replay_eligible_regret), 4) if joint_replay_eligible_regret else 0.0,
        "JointReplayTriggeredRegretMean": round(sum(joint_replay_triggered_regret) / len(joint_replay_triggered_regret), 4) if joint_replay_triggered_regret else 0.0,
        "JointReplaySkippedHighRegretCount": joint_replay_skipped_high_regret,
        "LoopLedgerCount": len(loop_ledger),
        "AllowLedger_AllTeachQueries": n_tq,
        "AllowLedger_PreRevealStates": len(prereveal_entries),
        "AllowLedger_BothTicketsAvailable": sum(
            1 for d in prereveal_entries
            if bool(d.get("pre_both_tickets_available", False))
        ),
        "AllowLedger_BothTicketsAndRoundsOK": sum(
            1 for d in prereveal_entries
            if bool(d.get("pre_both_tickets_available", False))
            and int(d.get("pre_rounds_left", 0)) >= 3
        ),
        "AllowLedger_EligibleForAllow": len(allow_eligible_entries),
        "AllowLedger_AllowPreserved": len(allow_preserved_eligible_entries),
        "AllowEligibleRate": len(allow_eligible_entries) / max(len(prereveal_entries), 1),
        "AllowPreserveGivenEligibleRate": (
            len(allow_preserved_eligible_entries) / max(len(allow_eligible_entries), 1)
        ),
        "LoopBreak_AtAllow_GivenEligible": (
            len(allow_eligible_entries) - len(allow_preserved_eligible_entries)
        ) / max(len(allow_eligible_entries), 1),
        "Loop_BothTicketsAvailableCount": len(loop_both_ticket_qids),
        "Loop_AllowCount": len(allow_query_ids),
        "Loop_ProductiveRevealCount": len(allow_query_ids & productive_reveal_qids),
        "Loop_ContrastiveUsedCount": len(allow_query_ids & contrastive_used_qids),
        "Loop_CueAfterRevealCount": len(allow_query_ids & cue_after_reveal_qids),
        "Loop_GraceConsumedCount": len(allow_query_ids & grace_consumed_qids),
        "Loop_CorrectAfterCueGraceCount": len(allow_query_ids & correct_after_feedback_qids),
        "Loop_PositiveTicketUsedCount": len(allow_query_ids & positive_used_qids),
        "Loop_CompleteCount": len(loop_complete_qids),
        "Loop_BothTicketsAvailableRate": len(loop_both_ticket_qids) / max(n_tq, 1),
        "Loop_AllowRate": len(allow_query_ids) / max(len(loop_both_ticket_qids), 1),
        "Loop_ProductiveRevealRate": len(allow_query_ids & productive_reveal_qids) / max(len(allow_query_ids), 1),
        "Loop_ContrastiveUsedRate": len(allow_query_ids & contrastive_used_qids) / max(len(allow_query_ids), 1),
        "Loop_CueAfterRevealRate": len(allow_query_ids & cue_after_reveal_qids) / max(len(allow_query_ids), 1),
        "Loop_GraceConsumedRate": len(allow_query_ids & grace_consumed_qids) / max(len(allow_query_ids), 1),
        "Loop_CorrectAfterCueGraceRate": len(allow_query_ids & correct_after_feedback_qids) / max(len(allow_query_ids), 1),
        "Loop_PositiveTicketUsedRate": len(allow_query_ids & positive_used_qids) / max(len(allow_query_ids), 1),
        "Loop_CompleteRate": len(loop_complete_qids) / max(len(allow_query_ids), 1),
        "LoopBreak_AtAllowRate": len(loop_both_ticket_qids - allow_query_ids) / max(len(loop_both_ticket_qids), 1),
        "LoopBreak_AtRevealRate": len(allow_query_ids - productive_reveal_qids) / max(len(allow_query_ids), 1),
        "LoopBreak_AtCueRate": len((allow_query_ids & productive_reveal_qids) - cue_after_reveal_qids) / max(len(allow_query_ids & productive_reveal_qids), 1),
        "LoopBreak_AtGraceRate": len((allow_query_ids & cue_after_reveal_qids) - (grace_consumed_qids | correct_after_feedback_qids)) / max(len(allow_query_ids & cue_after_reveal_qids), 1),
        "LoopBreak_AtCorrectRate": len(((allow_query_ids & cue_after_reveal_qids) | (allow_query_ids & grace_consumed_qids)) - correct_after_feedback_qids) / max(len((allow_query_ids & cue_after_reveal_qids) | (allow_query_ids & grace_consumed_qids)), 1),
        "LoopBreak_AtPositiveTicketRate": len((allow_query_ids & correct_after_feedback_qids) - positive_used_qids) / max(len(allow_query_ids & correct_after_feedback_qids), 1),
        "LoopLedgerBreak_AtAllowRate": ledger_break_counts.get("allow", 0) / max(len(loop_ledger), 1),
        "LoopLedgerBreak_AtRevealRate": ledger_break_counts.get("reveal", 0) / max(len(loop_ledger), 1),
        "LoopLedgerBreak_AtContrastiveRate": ledger_break_counts.get("contrastive", 0) / max(len(loop_ledger), 1),
        "LoopLedgerBreak_AtCueRate": ledger_break_counts.get("cue", 0) / max(len(loop_ledger), 1),
        "LoopLedgerBreak_AtGraceRate": ledger_break_counts.get("grace", 0) / max(len(loop_ledger), 1),
        "LoopLedgerBreak_AtCorrectRate": ledger_break_counts.get("correct", 0) / max(len(loop_ledger), 1),
        "LoopLedgerBreak_AtPositiveTicketRate": ledger_break_counts.get("positive_ticket", 0) / max(len(loop_ledger), 1),
        "AllowSafeDiagDecisionCount": allow_wait_count,
        "SafeDiagRevealAfterAllowRate": safe_diag_reveal_after_allow / max(allow_wait_count, 1),
        "SafeDiagRevealThenCueRate": safe_diag_reveal_then_cue / max(safe_diag_reveal_after_allow, 1),
        "SafeDiagRevealThenGraceRate": safe_diag_reveal_then_grace / max(safe_diag_reveal_after_allow, 1),
        "SafeDiagRevealThenTrajectorySuccessRate": safe_diag_reveal_then_traj_success / max(safe_diag_reveal_after_allow, 1),
        "BoundedRevealThenTrajectorySuccessRate": bounded_reveal_then_traj_success / max(bounded_reveal_after_allow, 1),
        # Phase 6I.8: budgeted pedagogical feedback metrics
        "RawWrongRevealCount": raw_wrong_reveal_count,
        "SemanticWrongUpdateCount": semantic_wrong_update_count,
        "RawCorrectPickCount": raw_correct_pick_count,
        "SemanticCorrectUpdateCount": semantic_correct_update_count,
        "ContrastiveTicketUsedRate": contrastive_ticket_used / max(n_tq, 1),
        "PositiveTicketUsedRate": positive_ticket_used / max(n_tq, 1),
        "ContrastiveCreditMean": round(sum(contrastive_credits) / len(contrastive_credits), 4) if contrastive_credits else 0.0,
        "PositiveCreditMean": round(sum(positive_credits) / len(positive_credits), 4) if positive_credits else 0.0,
        "CorrectAfterRevealRate": len(correct_after_reveal_steps) / max(raw_correct_pick_count, 1),
        "CorrectAfterCueRate": len(correct_after_cue_steps) / max(raw_correct_pick_count, 1),
        "CorrectAfterGraceRate": len(correct_after_grace_steps) / max(raw_correct_pick_count, 1),
        "SemanticCorrectUpdateAfterRevealRate": _applied_rate(correct_after_reveal_steps),
        "SemanticCorrectUpdateAfterCueRate": _applied_rate(correct_after_cue_steps),
        "SemanticCorrectUpdateAfterGraceRate": _applied_rate(correct_after_grace_steps),
        "PositiveTicketUsedAfterCueRate": _ticket_used_rate(correct_after_cue_steps),
        "PositiveTicketUsedAfterGraceRate": _ticket_used_rate(correct_after_grace_steps),
        "PositiveTicketUsedAfterIncidentalCorrectRate": _ticket_used_rate(correct_incidental_steps),
        "ContrastiveTicketAvailableAtAllowRate": (
            sum(1 for d in allow_entries if bool(d.get("pre_contrastive_ticket_available", False)))
            / max(len(allow_entries), 1)
        ),
        "PositiveTicketAvailableAtAllowRate": (
            sum(1 for d in allow_entries if bool(d.get("pre_positive_ticket_available", False)))
            / max(len(allow_entries), 1)
        ),
        "PositiveTicketAvailablePostRevealRate": (
            sum(1 for d in postreveal_entries if bool(d.get("pre_positive_ticket_available", False)))
            / max(len(postreveal_entries), 1)
        ),
        "AllowAttemptCount": len(allow_attempt_entries),
        "AllowPreservedCount": len(allow_preserved_entries),
        "AllowAttemptPreserveRate": len(allow_preserved_entries) / max(len(allow_attempt_entries), 1),
        "AllowWithBothTicketsRate": (
            sum(1 for d in allow_entries if bool(d.get("pre_both_tickets_available", False)))
            / max(len(allow_entries), 1)
        ),
        "AllowWithoutPositiveTicketRate": (
            sum(1 for d in allow_entries if not bool(d.get("pre_positive_ticket_available", False)))
            / max(len(allow_entries), 1)
        ),
        "AllowWithoutEnoughRoundsRate": (
            sum(1 for d in allow_entries if int(d.get("pre_rounds_left", 0)) < 3)
            / max(len(allow_entries), 1)
        ),
        "AllowThenContrastiveTicketUsedRate": (
            sum(1 for qs in allow_queries if bool(getattr(qs, "contrastive_update_used", False)))
            / max(len(allow_queries), 1)
        ),
        "AllowThenPositiveTicketUsedRate": (
            sum(1 for qs in allow_queries if bool(getattr(qs, "positive_update_used", False)))
            / max(len(allow_queries), 1)
        ),
        "AllowThenBothTicketsUsedRate": (
            sum(
                1 for qs in allow_queries
                if bool(getattr(qs, "contrastive_update_used", False))
                and bool(getattr(qs, "positive_update_used", False))
            )
            / max(len(allow_queries), 1)
        ),
        "AllowAttemptedLoopValueMean": round(sum(allow_attempt_loop_values) / len(allow_attempt_loop_values), 4) if allow_attempt_loop_values else 0.0,
        "AllowAttemptedProductiveMassMean": round(sum(allow_attempt_productive_mass) / len(allow_attempt_productive_mass), 4) if allow_attempt_productive_mass else 0.0,
        "AllowAttemptedHarmMassMean": round(sum(allow_attempt_harm_mass) / len(allow_attempt_harm_mass), 4) if allow_attempt_harm_mass else 0.0,
        "AllowAttemptedExpectedDamageMean": round(sum(allow_attempt_expected_damage) / len(allow_attempt_expected_damage), 4) if allow_attempt_expected_damage else 0.0,
        "AllowAttemptedPSurviveMean": round(sum(allow_attempt_p_survive) / len(allow_attempt_p_survive), 4) if allow_attempt_p_survive else 0.0,
        "AllowAttemptControlledV2MissingTicketRate": allow_attempt_reason_counts.get("controlled_v2_missing_ticket", 0) / max(len(allow_attempt_entries), 1),
        "AllowAttemptControlledV2NotEnoughRoundsRate": allow_attempt_reason_counts.get("controlled_v2_not_enough_rounds", 0) / max(len(allow_attempt_entries), 1),
        "AllowAttemptControlledV2ZeroLoopValueRate": allow_attempt_reason_counts.get("controlled_v2_zero_loop_value", 0) / max(len(allow_attempt_entries), 1),
        "AllowAttemptControlledV2BaseBlockedRate": allow_attempt_reason_counts.get("controlled_v2_base_blocked", 0) / max(len(allow_attempt_entries), 1),
        "AllowReject_NoContrastiveTicketRate": allow_reject_reason_counts.get("NO_CONTRASTIVE_TICKET", 0) / max(len(allow_reject_entries), 1),
        "AllowReject_NoPositiveTicketRate": allow_reject_reason_counts.get("NO_POSITIVE_TICKET", 0) / max(len(allow_reject_entries), 1),
        "AllowReject_NoRoundsRate": allow_reject_reason_counts.get("NOT_ENOUGH_ROUNDS", 0) / max(len(allow_reject_entries), 1),
        "AllowReject_NoProductiveMassRate": allow_reject_reason_counts.get("NO_PRODUCTIVE_MASS", 0) / max(len(allow_reject_entries), 1),
        "AllowReject_HarmDominatesRate": allow_reject_reason_counts.get("HARM_DOMINATES", 0) / max(len(allow_reject_entries), 1),
        "AllowReject_HighRiskDominatesRate": allow_reject_reason_counts.get("HIGH_RISK_DOMINATES", 0) / max(len(allow_reject_entries), 1),
        "AllowReject_FarWrongDominatesRate": allow_reject_reason_counts.get("FAR_WRONG_DOMINATES", 0) / max(len(allow_reject_entries), 1),
        "AllowReject_PostValueLowRate": allow_reject_reason_counts.get("POST_REVEAL_VALUE_LOW", 0) / max(len(allow_reject_entries), 1),
        "AllowReject_MasteryRate": allow_reject_reason_counts.get("MASTERY_ALREADY_HIGH", 0) / max(len(allow_reject_entries), 1),
        "AllowReject_ProtectRequiredRate": allow_reject_reason_counts.get("PROTECT_REQUIRED", 0) / max(len(allow_reject_entries), 1),
        "AllowReject_NotPreRevealRate": allow_reject_reason_counts.get("NOT_PRE_REVEAL", 0) / max(len(allow_reject_entries), 1),
        "AllowLoopValueMean": round(sum(allow_loop_values) / len(allow_loop_values), 4) if allow_loop_values else 0.0,
        "AllowProductiveMassMean": round(sum(allow_productive_mass) / len(allow_productive_mass), 4) if allow_productive_mass else 0.0,
        "AllowHarmMassMean": round(sum(allow_harm_mass) / len(allow_harm_mass), 4) if allow_harm_mass else 0.0,
        "CandidateAuditPostRevealDecisionCount": len(postreveal_entries),
        "CandidateAuditEntryCount": len(postreveal_candidate_audit),
        "GConsolidateMean": round(sum(g_consolidate_all) / len(g_consolidate_all), 4) if g_consolidate_all else 0.0,
        "GConsolidateChosenMean": round(sum(g_consolidate_chosen) / len(g_consolidate_chosen), 4) if g_consolidate_chosen else 0.0,
        "GConsolidateWAITMean": round(sum(g_consolidate_wait) / len(g_consolidate_wait), 4) if g_consolidate_wait else 0.0,
        "GConsolidateHLMean": round(sum(g_consolidate_hl) / len(g_consolidate_hl), 4) if g_consolidate_hl else 0.0,
        "GConsolidateMIXMean": round(sum(g_consolidate_mix) / len(g_consolidate_mix), 4) if g_consolidate_mix else 0.0,
        "GConsolidateBANMean": round(sum(g_consolidate_ban) / len(g_consolidate_ban), 4) if g_consolidate_ban else 0.0,
        "GConsolidateChangesBestActionRate": g_consolidate_changes_best_action / max(len(postreveal_entries), 1),
        "ChosenHasMaxGConsolidateRate": chosen_has_max_gconsolidate / max(len(g_consolidate_chosen), 1),
        "QConsolidateDeltaMean": round(sum(q_consolidate_delta_all) / len(q_consolidate_delta_all), 4) if q_consolidate_delta_all else 0.0,
        "QConsolidateChosenDeltaMean": round(sum(q_consolidate_delta_chosen) / len(q_consolidate_delta_chosen), 4) if q_consolidate_delta_chosen else 0.0,
        "QWithoutConsolidateChosenMean": round(sum(q_without_chosen) / len(q_without_chosen), 4) if q_without_chosen else 0.0,
        "QWithConsolidateChosenMean": round(sum(q_with_chosen) / len(q_with_chosen), 4) if q_with_chosen else 0.0,
        "PostReveal_PCorrect2RChosenMean": round(sum(p_correct_2r_chosen) / len(p_correct_2r_chosen), 4) if p_correct_2r_chosen else 0.0,
        "PostReveal_PCorrect2RWAITMean": round(sum(p_correct_2r_wait) / len(p_correct_2r_wait), 4) if p_correct_2r_wait else 0.0,
        "PostReveal_PCorrect2RHLMean": round(sum(p_correct_2r_hl) / len(p_correct_2r_hl), 4) if p_correct_2r_hl else 0.0,
        "PostReveal_PCorrect2RMIXMean": round(sum(p_correct_2r_mix) / len(p_correct_2r_mix), 4) if p_correct_2r_mix else 0.0,
        "PostReveal_PCorrect2RBANMean": round(sum(p_correct_2r_ban) / len(p_correct_2r_ban), 4) if p_correct_2r_ban else 0.0,
        "ProductiveRevealCount": len(productive_reveals),
        "ProductiveRevealRate": len(productive_reveals) / max(raw_wrong_reveal_count, 1),
        "HarmfulRevealCount": len(harmful_reveals),
        "HarmfulRevealRate": len(harmful_reveals) / max(raw_wrong_reveal_count, 1),
        "ProductiveRevealCreditMean": round(sum(productive_credit_values) / len(productive_credit_values), 4) if productive_credit_values else 0.0,
        "HarmfulRevealCreditMean": round(sum(harmful_credit_values) / len(harmful_credit_values), 4) if harmful_credit_values else 0.0,
        "SafeDiagRevealCreditMean": round(sum(safe_diag_reveal_credit) / len(safe_diag_reveal_credit), 4) if safe_diag_reveal_credit else 0.0,
        "FarWrongRevealCreditMean": round(sum(far_wrong_reveal_credit) / len(far_wrong_reveal_credit), 4) if far_wrong_reveal_credit else 0.0,
        "HighRiskRevealCreditMean": round(sum(high_risk_reveal_credit) / len(high_risk_reveal_credit), 4) if high_risk_reveal_credit else 0.0,
        "RepeatedWrongRevealCreditMean": round(sum(repeated_wrong_reveal_credit) / len(repeated_wrong_reveal_credit), 4) if repeated_wrong_reveal_credit else 0.0,
        "ContrastiveTicketSpentOnSafeDiagRate": _ticket_spend_rate("safe_diag"),
        "ContrastiveTicketSpentOnBoundedDiagRate": _ticket_spend_rate("bounded_diag"),
        "ContrastiveTicketSpentOnFarWrongRate": _ticket_spend_rate("far_wrong"),
        "ContrastiveTicketSpentOnHighRiskRate": _ticket_spend_rate("high_risk"),
        "ContrastiveTicketSpentOnSameWrongRate": _ticket_spend_rate("same_wrong"),
    }
    result.update(allow_replay_metrics)
    result.update(allow_family_metrics)
    result.update(prereveal_family_metrics)
    result.update(pprod_proxy_metrics)

    # 6I.5: WAIT reason code distribution from decision trace
    wait_reason_counts = {}
    total_wait_decisions = 0
    total_decisions = 0
    for d in decision_trace:
        total_decisions += 1
        wr = d.get("wait_reason")
        if wr is not None:
            total_wait_decisions += 1
            wait_reason_counts[wr] = wait_reason_counts.get(wr, 0) + 1
    # Standard reason names
    for reason_name in [
        "WAIT_ALLOW_SAFE_DIAG", "WAIT_GRACE", "WAIT_GRACE_OVERRIDE",
        "WAIT_NO_GOOD_CUE", "WAIT_Q_BEATS_CUE", "WAIT_MISSED_POSITIVE_CUE",
        "WAIT_BORING_ESCAPE", "WAIT_BORING_MASTERY", "WAIT_NO_PED_OPPORTUNITY",
        "WAIT_LOW_LEVERAGE", "WAIT_BLOCKED_BY_PROTECT", "WAIT_BLOCKED_BY_DEADLINE",
        "WAIT_GENERIC",
    ]:
        count = wait_reason_counts.get(reason_name, 0)
        result[f"WR_{reason_name}_Count"] = count
        result[f"WR_{reason_name}_Rate"] = count / max(total_wait_decisions, 1)
    result["WR_WAIT_BORING_ESCAPE_Narrow_Count"] = wait_reason_counts.get("WAIT_BORING_ESCAPE", 0)
    result["WR_WAIT_BORING_ESCAPE_Narrow_Rate"] = wait_reason_counts.get("WAIT_BORING_ESCAPE", 0) / max(total_wait_decisions, 1)
    legacy_boring_escape = (
        wait_reason_counts.get("WAIT_BORING_ESCAPE", 0)
        + wait_reason_counts.get("WAIT_BORING_MASTERY", 0)
        + wait_reason_counts.get("WAIT_NO_PED_OPPORTUNITY", 0)
        + wait_reason_counts.get("WAIT_LOW_LEVERAGE", 0)
    )
    result["WR_WAIT_BORING_ESCAPE_Count"] = legacy_boring_escape
    result["WR_WAIT_BORING_ESCAPE_Rate"] = legacy_boring_escape / max(total_wait_decisions, 1)
    # Aggregates
    bad_wait = (wait_reason_counts.get("WAIT_MISSED_POSITIVE_CUE", 0) +
                wait_reason_counts.get("WAIT_Q_BEATS_CUE", 0))
    good_wait = (wait_reason_counts.get("WAIT_ALLOW_SAFE_DIAG", 0) +
                 wait_reason_counts.get("WAIT_GRACE", 0))
    result["BadWAIT_AllReasonCount"] = bad_wait
    result["BadWAIT_PostReveal_PositiveCueCount"] = postrev_badwait_positive_cue
    result["BadWAIT_PostReveal_PositiveMixCount"] = postrev_badwait_positive_mix
    result["BadWAIT_PostReveal_PositiveHLCount"] = postrev_badwait_positive_hl
    result["BadWAIT_PostReveal_PositiveCueRate"] = postrev_badwait_positive_cue / max(postrev_positive_cue_opp, 1)
    result["BadWAIT_PostReveal_PositiveMixRate"] = postrev_badwait_positive_mix / max(postrev_positive_mix_opp, 1)
    result["BadWAIT_PostReveal_PositiveHLRate"] = postrev_badwait_positive_hl / max(postrev_positive_hl_opp, 1)
    result["BadWAIT_AmongWaitRate"] = bad_wait / max(total_wait_decisions, 1)
    result["GoodWAIT_PedagogicalCount"] = good_wait
    result["GoodWAIT_PedagogicalRate"] = good_wait / max(total_wait_decisions, 1)

    return result


def compute_6e_metrics(block) -> dict:
    """Compute diagnostic menu metrics from a block."""
    obs_q = block.obs_phase_queries
    teach_q = block.teach_phase_queries
    teach_queries = block.queries[obs_q: obs_q + teach_q]

    n_teach = max(len(teach_queries), 1)
    quota_met = 0
    wr_diag = 0
    wr_random = 0
    wr_total = 0

    for qs in teach_queries:
        labels = getattr(qs, "option_diag_labels", {})
        if not labels:
            continue
        has_safe_diag = any(v == "safe_diagnostic_wrong" for v in labels.values())
        has_bounded = any(v == "bounded_diagnostic_wrong" for v in labels.values())
        has_lure = any(v == "high_risk_lure" for v in labels.values())
        if has_safe_diag and has_bounded and has_lure:
            quota_met += 1

    for ls in block.learner_trace:
        if ls.action == "pick" and not ls.correct:
            pick_idx = getattr(ls, "pick_index", None)
            qid = getattr(ls, "query_id", None)
            if qid is not None and qid < len(block.queries):
                qs = block.queries[qid]
                labels = getattr(qs, "option_diag_labels", {})
                label = labels.get(pick_idx, "")
                wr_total += 1
                if label in ("safe_diagnostic_wrong", "bounded_diagnostic_wrong", "high_risk_lure"):
                    wr_diag += 1
                elif label in ("safe_far", "safe_random_wrong", "risky_far"):
                    wr_random += 1

    return {
        "ConfoundCoverage": quota_met / n_teach,
        "QuotaFailRate": 1.0 - (quota_met / n_teach),
        "WrongRevealDiagRate": wr_diag / max(wr_total, 1),
        "WrongRevealRandRate": wr_random / max(wr_total, 1),
        "WrongRevealTotal": wr_total,
    }
