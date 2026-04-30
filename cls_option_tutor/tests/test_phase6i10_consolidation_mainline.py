import numpy as np
import pytest

from cls_option_tutor.env.state import BlockState, QueryState
from cls_option_tutor.experiments.condition_overrides import resolve_condition_alias
from cls_option_tutor.experiments.metrics_extractors import (
    build_allow_gate_replay,
    build_allow_family_audit,
    build_learning_loop_ledger,
    build_postreveal_candidate_audit,
    compute_6fg_metrics,
)
from cls_option_tutor.interfaces import LearnerStep, Option
from cls_option_tutor.tutor.sparse_tutor_scoring import (
    compute_postreveal_consolidation_value,
    compute_postreveal_q,
)
from cls_option_tutor.tutor.sparse_tutor_horizon import (
    compute_pre_reveal_allow_value,
)


def _make_qs():
    qs = QueryState(
        query_id=0,
        target_output=["red", "blue"],
        true_program=["prog"],
        hp=5,
        max_rounds=5,
    )
    qs.menu = [
        Option(
            index=0,
            text=["correct"],
            danger_vec=np.zeros(4, dtype=float),
            is_correct=True,
            risk_class=0,
            rendered_output=["red", "blue"],
        ),
        Option(
            index=1,
            text=["wrong"],
            danger_vec=np.zeros(4, dtype=float),
            is_correct=False,
            risk_class=1,
            rendered_output=["blue", "red"],
        ),
    ]
    return qs


def test_postreveal_consolidation_value_prefers_cue_context():
    qs = _make_qs()
    qs.post_reveal_phase = True

    meta = compute_postreveal_consolidation_value(
        qs,
        p_correct_action=0.6,
        action_name="MIX",
        incidental_correct_credit=0.5,
    )

    assert meta["positive_ticket_available"] is True
    assert meta["reason"] == "after_cue"
    assert meta["source_weight"] == 1.0
    assert meta["consolidation_value"] == 0.6


def test_postreveal_consolidation_value_zero_when_positive_ticket_spent():
    qs = _make_qs()
    qs.post_reveal_phase = True
    qs.positive_update_used = True

    meta = compute_postreveal_consolidation_value(
        qs,
        p_correct_action=0.9,
        action_name="HIGHLIGHT",
        incidental_correct_credit=0.5,
    )

    assert meta["positive_ticket_available"] is False
    assert meta["reason"] == "ticket_spent"
    assert meta["consolidation_value"] == 0.0


def test_postreveal_q_traj_v2_includes_consolidation_value():
    decomp = {
        "delta_p_correct": 0.1,
        "log_margin_gain": 0.2,
        "harm_mass_drop": 0.3,
        "harmful_shift": 0.05,
    }
    q_without = compute_postreveal_q(
        decomp,
        action_name="MIX",
        value_mode="traj_v2",
        lambda_info_post=0.0,
        grace_conversion=0.1,
        consolidation_value=0.0,
        cost=0.0,
    )
    q_with = compute_postreveal_q(
        decomp,
        action_name="MIX",
        value_mode="traj_v2",
        lambda_info_post=0.0,
        grace_conversion=0.1,
        consolidation_value=0.4,
        cost=0.0,
    )

    assert q_with > q_without
    assert round(q_with - q_without, 6) == 0.4


def test_budget_metrics_expose_positive_ticket_and_consolidation_paths():
    qs = _make_qs()
    qs.post_reveal_phase = True
    qs.contrastive_update_used = True
    qs.positive_update_used = True

    block = BlockState(
        queries=[qs],
        current_query_idx=0,
        obs_phase_queries=0,
        teach_phase_queries=1,
        eval_phase_queries=0,
    )
    block.tutor_trace = []
    block._grace_metrics = {}
    block._decision_trace = [
        {
            "query_id": 0,
            "round_t": 0,
            "pre_post_reveal_phase": True,
            "pre_contrastive_ticket_available": True,
            "pre_positive_ticket_available": True,
            "pre_both_tickets_available": True,
            "pre_productive_allow_preserved": True,
            "pre_rounds_left": 4,
            "phase": "PRE_REVEAL_ALLOW",
            "wait_reason": "WAIT_ALLOW_SAFE_DIAG",
            "chosen_action": "MIX",
            "scoring": {
                "candidates": [
                    {
                        "action": "WAIT",
                        "q_use": 0.3,
                        "q_use_with_consolidate": 0.3,
                        "q_use_without_consolidate": 0.3,
                        "q_use_consolidate_delta": 0.0,
                        "g_consolidate": 0.0,
                        "postreveal_consolidation_value": 0.2,
                        "postreveal_p_correct_2r": 0.2,
                    },
                    {
                        "action": "MIX",
                        "q_use": 0.9,
                        "q_use_with_consolidate": 0.9,
                        "q_use_without_consolidate": 0.1,
                        "q_use_consolidate_delta": 0.8,
                        "g_consolidate": 0.8,
                        "postreveal_consolidation_value": 0.8,
                        "postreveal_p_correct_2r": 0.8,
                    },
                ]
            },
        }
    ]
    block.learner_trace = [
        LearnerStep(
            round_t=0,
            query_id=0,
            action="pick",
            pick_index=1,
            correct=False,
            hp_before=5,
            hp_after=4,
            raw_feedback_kind="wrong_reveal",
            feedback_category="safe_diag",
            semantic_credit=0.8,
            semantic_credit_type="contrastive",
            semantic_update_attempted=True,
            semantic_update_applied=True,
            contrastive_ticket_consumed=True,
        ),
        LearnerStep(
            round_t=1,
            query_id=0,
            action="pick",
            pick_index=0,
            correct=True,
            hp_before=4,
            hp_after=4,
            raw_feedback_kind="correct_pick",
            feedback_category="correct_after_feedback",
            semantic_credit=1.0,
            semantic_credit_type="positive",
            semantic_credit_reason="after_cue",
            semantic_update_attempted=True,
            semantic_update_applied=True,
            positive_ticket_consumed=True,
        ),
    ]

    metrics = compute_6fg_metrics(block)
    assert metrics["PositiveTicketAvailableAtAllowRate"] == 1.0
    assert metrics["AllowWithBothTicketsRate"] == 1.0
    assert metrics["AllowThenBothTicketsUsedRate"] == 1.0
    assert metrics["AllowAttemptCount"] == 1
    assert metrics["AllowPreservedCount"] == 1
    assert metrics["AllowAttemptPreserveRate"] == 1.0
    assert metrics["CorrectAfterCueRate"] == 1.0
    assert metrics["PositiveTicketUsedAfterCueRate"] == 1.0
    assert metrics["GConsolidateMIXMean"] == 0.8
    assert metrics["GConsolidateWAITMean"] == 0.0
    assert metrics["GConsolidateChosenMean"] == 0.8
    assert metrics["ChosenHasMaxGConsolidateRate"] == 1.0
    assert metrics["GConsolidateChangesBestActionRate"] == 1.0
    assert metrics["PostReveal_PCorrect2RChosenMean"] == 0.8
    assert metrics["QConsolidateChosenDeltaMean"] == 0.8
    assert metrics["QWithoutConsolidateChosenMean"] == 0.1
    assert metrics["QWithConsolidateChosenMean"] == 0.9
    assert metrics["Loop_AllowRate"] == 1.0
    assert metrics["Loop_ProductiveRevealRate"] == 1.0
    assert metrics["Loop_CueAfterRevealRate"] == 1.0
    assert metrics["Loop_PositiveTicketUsedRate"] == 1.0
    assert metrics["Loop_CompleteRate"] == 1.0
    assert metrics["LoopBreak_AtAllowRate"] == 0.0
    assert metrics["AllowLoopValueMean"] == 0.0
    ledger = build_learning_loop_ledger(block)
    assert len(ledger) == 1
    assert ledger[0]["loop_complete"] is True
    assert ledger[0]["loop_break_stage"] == "complete"
    audit = build_postreveal_candidate_audit(block)
    assert len(audit) == 2
    chosen = [row for row in audit if row["chosen"]]
    assert len(chosen) == 1
    assert chosen[0]["action"] == "MIX"
    assert chosen[0]["q_consolidate_delta"] == 0.8


def test_candidate_audit_uses_explicit_q_without_consolidate_not_raw_wait_telemetry():
    qs = _make_qs()
    qs.post_reveal_phase = True
    block = BlockState(
        queries=[qs],
        current_query_idx=0,
        obs_phase_queries=0,
        teach_phase_queries=1,
        eval_phase_queries=0,
    )
    block.tutor_trace = []
    block._grace_metrics = {}
    block._decision_trace = [
        {
            "query_id": 0,
            "round_t": 0,
            "pre_post_reveal_phase": True,
            "pre_both_tickets_available": True,
            "chosen_action": "WAIT",
            "scoring": {
                "candidates": [
                    {
                        "action": "WAIT",
                        "q_use": 1.0,
                        "q_use_with_consolidate": 1.0,
                        "q_use_without_consolidate": 1.0,
                        "q_use_consolidate_delta": 0.0,
                        "g_consolidate": 0.0,
                        "postreveal_consolidation_value": 0.9,
                    },
                    {
                        "action": "MIX",
                        "q_use": 0.9,
                        "q_use_with_consolidate": 0.9,
                        "q_use_without_consolidate": 0.4,
                        "q_use_consolidate_delta": 0.5,
                        "g_consolidate": 0.5,
                        "postreveal_consolidation_value": 0.5,
                    },
                ]
            },
        }
    ]
    block.learner_trace = []

    metrics = compute_6fg_metrics(block)
    assert metrics["GConsolidateChangesBestActionRate"] == 0.0


def test_allow_entry_calibration_metrics_use_eligible_denominator_and_replay():
    qs0 = _make_qs()
    qs0.query_id = 0
    qs1 = _make_qs()
    qs1.query_id = 1
    qs2 = _make_qs()
    qs2.query_id = 2

    block = BlockState(
        queries=[qs0, qs1, qs2],
        current_query_idx=0,
        obs_phase_queries=0,
        teach_phase_queries=3,
        eval_phase_queries=0,
    )
    block.tutor_trace = []
    block._grace_metrics = {}
    block.learner_trace = []
    block._decision_trace = [
        {
            "query_id": 0,
            "round_t": 0,
            "pre_post_reveal_phase": False,
            "pre_contrastive_ticket_available": True,
            "pre_positive_ticket_available": True,
            "pre_both_tickets_available": True,
            "pre_allow_eligible": True,
            "pre_allow_phase_eligible": True,
            "pre_productive_allow_preserved": True,
            "pre_productive_allow_reason": "controlled_v2_allow",
            "pre_allow_reject_reason": "ALLOW_PRESERVED",
            "pre_rounds_left": 4,
            "pre_productive_mass_wait": 0.4,
            "pre_info_mass_wait": 0.4,
            "pre_harm_mass_wait": 0.2,
            "pre_expected_damage_wait": 0.4,
            "pre_allow_p_survive": 0.9,
            "pre_allow_loop_value": 0.06,
            "pre_allow_post_reveal_best_value_estimate": 0.2,
            "pre_p_safe_diag_wait": 0.3,
            "pre_p_bounded_diag_wait": 0.2,
            "pre_p_farwrong_wait": 0.05,
            "pre_p_highrisk_wait": 0.05,
            "phase": "PRE_REVEAL_ALLOW",
            "wait_reason": "WAIT_ALLOW_SAFE_DIAG",
            "p_timeout_wait": 0.1,
            "chosen_action": "WAIT",
            "scoring": {"candidates": []},
        },
        {
            "query_id": 1,
            "round_t": 0,
            "pre_post_reveal_phase": False,
            "pre_contrastive_ticket_available": True,
            "pre_positive_ticket_available": True,
            "pre_both_tickets_available": True,
            "pre_allow_eligible": True,
            "pre_allow_phase_eligible": True,
            "pre_productive_allow_preserved": False,
            "pre_productive_allow_reason": "controlled_v2_base_blocked",
            "pre_allow_reject_reason": "HARM_DOMINATES",
            "pre_rounds_left": 4,
            "pre_productive_mass_wait": 0.2,
            "pre_info_mass_wait": 0.2,
            "pre_harm_mass_wait": 0.6,
            "pre_expected_damage_wait": 1.1,
            "pre_allow_p_survive": 0.7,
            "pre_allow_loop_value": 0.0,
            "pre_allow_post_reveal_best_value_estimate": 0.1,
            "pre_p_safe_diag_wait": 0.1,
            "pre_p_bounded_diag_wait": 0.2,
            "pre_p_farwrong_wait": 0.1,
            "pre_p_highrisk_wait": 0.1,
            "phase": "PRE_REVEAL_ALLOW",
            "wait_reason": "WAIT_GENERIC",
            "p_timeout_wait": 0.2,
            "chosen_action": "WAIT",
            "scoring": {"candidates": []},
        },
        {
            "query_id": 2,
            "round_t": 0,
            "pre_post_reveal_phase": False,
            "pre_contrastive_ticket_available": True,
            "pre_positive_ticket_available": False,
            "pre_both_tickets_available": False,
            "pre_allow_eligible": False,
            "pre_allow_phase_eligible": True,
            "pre_productive_allow_preserved": False,
            "pre_productive_allow_reason": "controlled_v2_missing_ticket",
            "pre_allow_reject_reason": "NO_POSITIVE_TICKET",
            "pre_rounds_left": 4,
            "pre_productive_mass_wait": 0.3,
            "pre_info_mass_wait": 0.3,
            "pre_harm_mass_wait": 0.1,
            "pre_expected_damage_wait": 0.2,
            "pre_allow_p_survive": 0.95,
            "pre_allow_loop_value": 0.04,
            "pre_allow_post_reveal_best_value_estimate": 0.2,
            "pre_p_safe_diag_wait": 0.2,
            "pre_p_bounded_diag_wait": 0.2,
            "pre_p_farwrong_wait": 0.0,
            "pre_p_highrisk_wait": 0.0,
            "phase": "PRE_REVEAL_ALLOW",
            "wait_reason": "WAIT_GENERIC",
            "p_timeout_wait": 0.05,
            "chosen_action": "WAIT",
            "scoring": {"candidates": []},
        },
    ]

    metrics = compute_6fg_metrics(block)
    replay = build_allow_gate_replay(block)

    assert metrics["AllowLedger_PreRevealStates"] == 3
    assert metrics["AllowLedger_BothTicketsAvailable"] == 2
    assert metrics["AllowLedger_BothTicketsAndRoundsOK"] == 2
    assert metrics["AllowLedger_EligibleForAllow"] == 2
    assert metrics["AllowLedger_AllowPreserved"] == 1
    assert metrics["AllowEligibleRate"] == 2 / 3
    assert metrics["AllowPreserveGivenEligibleRate"] == 0.5
    assert metrics["LoopBreak_AtAllow_GivenEligible"] == 0.5
    assert metrics["AllowReject_HarmDominatesRate"] == 0.5
    assert metrics["AllowReject_NoPositiveTicketRate"] == 0.5
    assert len(replay) == 3
    assert metrics["AllowReplay_G0_Current_WouldAllowRate"] == 1 / 3
    assert metrics["AllowReplay_G1_PermissiveProd_WouldAllowRate"] == 2 / 3
    assert metrics["AllowReplay_G5_Combined_WouldAllowRate"] == 1 / 3


def test_allow_family_audit_splits_allow_critical_vs_harm_dominated_and_decomposes_pprod():
    qs0 = _make_qs()
    qs0.query_id = 0
    qs1 = _make_qs()
    qs1.query_id = 1

    block = BlockState(
        queries=[qs0, qs1],
        current_query_idx=0,
        obs_phase_queries=0,
        teach_phase_queries=2,
        eval_phase_queries=0,
    )
    block.tutor_trace = []
    block._grace_metrics = {}
    block._decision_trace = [
        {
            "query_id": 0,
            "round_t": 0,
            "pre_post_reveal_phase": False,
            "pre_allow_phase_eligible": True,
            "pre_both_tickets_available": True,
            "pre_rounds_left": 4,
            "pre_productive_allow_preserved": True,
            "pre_allow_reject_reason": "ALLOW_PRESERVED",
            "pre_productive_mass_wait": 0.40,
            "pre_p_safe_diag_wait": 0.30,
            "pre_p_bounded_diag_wait": 0.20,
            "pre_p_farwrong_wait": 0.05,
            "pre_p_highrisk_wait": 0.05,
            "pre_harm_mass_wait": 0.30,
            "p_correct_wait": 0.20,
            "pre_expected_damage_wait": 0.40,
            "p_timeout_wait": 0.10,
            "pre_allow_post_reveal_best_value_estimate": 0.20,
            "wait_reason": "WAIT_ALLOW_SAFE_DIAG",
            "chosen_action": "WAIT",
            "scoring": {"candidates": []},
        },
        {
            "query_id": 1,
            "round_t": 0,
            "pre_post_reveal_phase": False,
            "pre_allow_phase_eligible": True,
            "pre_both_tickets_available": True,
            "pre_rounds_left": 4,
            "pre_productive_allow_preserved": False,
            "pre_allow_reject_reason": "HARM_DOMINATES",
            "pre_productive_mass_wait": 0.20,
            "pre_p_safe_diag_wait": 0.10,
            "pre_p_bounded_diag_wait": 0.20,
            "pre_p_farwrong_wait": 0.10,
            "pre_p_highrisk_wait": 0.10,
            "pre_harm_mass_wait": 0.60,
            "p_correct_wait": 0.20,
            "pre_expected_damage_wait": 1.00,
            "p_timeout_wait": 0.20,
            "pre_allow_post_reveal_best_value_estimate": 0.10,
            "wait_reason": "WAIT_GENERIC",
            "chosen_action": "WAIT",
            "scoring": {"candidates": []},
        },
        {
            "query_id": 0,
            "round_t": 1,
            "pre_post_reveal_phase": True,
            "chosen_action": "MIX",
            "scoring": {"candidates": []},
        },
    ]
    block.learner_trace = [
        LearnerStep(
            round_t=0,
            query_id=0,
            action="pick",
            pick_index=1,
            correct=False,
            hp_before=5,
            hp_after=4,
            raw_feedback_kind="wrong_reveal",
            feedback_category="safe_diag",
            semantic_credit_type="contrastive",
            contrastive_ticket_consumed=True,
        ),
        LearnerStep(
            round_t=1,
            query_id=0,
            action="pick",
            pick_index=0,
            correct=True,
            hp_before=4,
            hp_after=4,
            raw_feedback_kind="correct_pick",
            feedback_category="correct_after_feedback",
            semantic_credit_type="positive",
            semantic_credit_reason="after_cue",
            positive_ticket_consumed=True,
        ),
    ]

    audit = build_allow_family_audit(block)
    metrics = compute_6fg_metrics(block)

    assert len(audit) == 2
    families = {row["query_id"]: row["family"] for row in audit}
    assert families[0] == "ALLOW_CRITICAL"
    assert families[1] == "HARM_DOMINATED"

    q0 = next(row for row in audit if row["query_id"] == 0)
    q1 = next(row for row in audit if row["query_id"] == 1)
    assert q0["family_split"] == "NATIVE_LIKE_ALLOW"
    assert q1["family_split"] == "MIXED_PROD_HARM"
    assert q0["p_prod_safe_component"] == 0.30
    assert q0["p_prod_bounded_component"] == 0.10
    assert q0["p_prod_total"] == 0.40
    assert q0["p_prod_safe_share"] == pytest.approx(0.75)
    assert q0["safe_diag_quality_gap"] == pytest.approx(0.20)
    assert q0["harm_competition_gap"] == pytest.approx(0.10)
    assert q0["productive_reveal_after_state"] is True
    assert q0["cue_after_state"] is True
    assert q0["correct_after_state"] is True
    assert q0["loop_complete_after_state"] is True

    assert metrics["AllowFamilyAuditStateCount"] == 2
    assert metrics["AllowFamily_ALLOW_CRITICAL_StateCount"] == 1
    assert metrics["AllowFamily_HARM_DOMINATED_StateCount"] == 1
    assert metrics["AllowFamily_ALLOW_CRITICAL_LoopCompleteRate"] == 1.0
    assert metrics["AllowFamily_HARM_DOMINATED_LoopCompleteRate"] == 0.0
    assert metrics["PreRevealFamily_NATIVE_LIKE_ALLOW_StateCount"] == 1
    assert metrics["PreRevealFamily_MIXED_PROD_HARM_StateCount"] == 1
    assert metrics["NativeLikeAllow_StateCount"] == 1
    assert metrics["MixedProdHarm_StateCount"] == 1
    assert metrics["NativeLikeAllow_LoopCompleteRate"] == 1.0
    assert metrics["MixedProdHarm_LoopCompleteRate"] == 0.0
    assert metrics["PProdSafeComponentMean"] == pytest.approx(0.20)
    assert metrics["PProdBoundedComponentMean"] == pytest.approx(0.10)
    assert metrics["AllowSafeDiagQualityGapMean"] == pytest.approx(0.05)
    assert metrics["AllowCritical_PProdMean"] == pytest.approx(0.40)
    assert metrics["AllowCritical_HarmMean"] == pytest.approx(0.30)


def test_allow_family_audit_splits_not_pre_reveal_and_finds_phase_blind_missed_allow():
    qs = _make_qs()
    block = BlockState(
        queries=[qs],
        current_query_idx=0,
        obs_phase_queries=0,
        teach_phase_queries=1,
        eval_phase_queries=0,
    )
    block.tutor_trace = []
    block._grace_metrics = {}
    block.learner_trace = []
    block._decision_trace = [
        {
            "query_id": 0,
            "round_t": 0,
            "phase": "DEFAULT",
            "pre_post_reveal_phase": False,
            "pre_allow_phase_eligible": False,
            "pre_productive_allow_preserved": False,
            "pre_allow_eligible": False,
            "pre_allow_reject_reason": "NOT_PRE_REVEAL",
            "pre_both_tickets_available": True,
            "pre_rounds_left": 4,
            "pre_productive_mass_wait": 0.40,
            "pre_p_safe_diag_wait": 0.30,
            "pre_p_bounded_diag_wait": 0.20,
            "pre_p_farwrong_wait": 0.05,
            "pre_p_highrisk_wait": 0.10,
            "pre_harm_mass_wait": 0.30,
            "pre_expected_damage_wait": 0.10,
            "pre_allow_post_reveal_best_value_estimate": 0.08,
            "p_correct_wait": 0.20,
            "p_timeout_wait": 0.05,
        },
        {
            "query_id": 0,
            "round_t": 1,
            "phase": "PROTECT",
            "pre_post_reveal_phase": False,
            "pre_allow_phase_eligible": False,
            "pre_productive_allow_preserved": False,
            "pre_allow_eligible": False,
            "pre_allow_reject_reason": "PROTECT_REQUIRED",
            "pre_both_tickets_available": True,
            "pre_rounds_left": 4,
            "pre_productive_mass_wait": 0.15,
            "pre_p_safe_diag_wait": 0.10,
            "pre_p_bounded_diag_wait": 0.00,
            "pre_p_farwrong_wait": 0.05,
            "pre_p_highrisk_wait": 0.40,
            "pre_harm_mass_wait": 0.45,
            "pre_expected_damage_wait": 0.25,
            "pre_allow_post_reveal_best_value_estimate": 0.02,
            "p_correct_wait": 0.20,
            "p_timeout_wait": 0.10,
        },
    ]

    audit = build_allow_family_audit(block)
    metrics = compute_6fg_metrics(block)

    assert len(audit) == 2
    r0 = next(row for row in audit if row["round_t"] == 0)
    r1 = next(row for row in audit if row["round_t"] == 1)

    assert r0["family"] == "NOT_PRE_REVEAL"
    assert r0["phase_reject_reason"] == "NOT_PRE_REVEAL_PHASE_INFER_DEFAULT"
    assert r0["native_phase_allow_candidate"] is True
    assert r0["phase_blind_family"] == "ALLOW_CRITICAL_STAR"
    assert r0["missed_allow_critical"] is True

    assert r1["family"] == "NOT_PRE_REVEAL"
    assert r1["phase_reject_reason"] == "NOT_PRE_REVEAL_PROTECT_PHASE"
    assert r1["native_phase_allow_candidate"] is False
    assert r1["phase_blind_family"] == "HIGHRISK_DOMINATED"
    assert r1["missed_allow_critical"] is False

    assert metrics["AllowFamily_NOT_PRE_REVEAL_StateCount"] == 2
    assert metrics["PhaseBlindAllowFamily_ALLOW_CRITICAL_STAR_StateCount"] == 1
    assert metrics["PhaseBlind_ALLOW_CRITICAL_StateCount"] == 1
    assert metrics["PhaseBlind_MissedAllowCritical_Count"] == 1
    assert metrics["PhaseBlind_MissedAllowCritical_Rate"] == pytest.approx(0.5)
    assert metrics["AllowPhaseReject_NOT_PRE_REVEAL_PHASE_INFER_DEFAULT_StateCount"] == 1
    assert metrics["AllowPhaseReject_NOT_PRE_REVEAL_PROTECT_PHASE_StateCount"] == 1
    assert metrics["AllowPhase_DEFAULT_PhaseBlindAllowRate"] == pytest.approx(1.0)
    assert metrics["AllowPhaseConfusion_DEFAULT__ALLOW_CRITICAL_STAR_Count"] == 1
    assert metrics["AllowPhaseConfusion_PROTECT__HIGHRISK_DOMINATED_Count"] == 1


def test_loop_v1_condition_alias_resolves_to_frozen_mainline():
    resolved = resolve_condition_alias("SIS_cf_mix_loop_v1")
    assert resolved == (
        "SIS_horizon_self_correct_cf_mix_netharm_direct_allow_"
        "budgeted_allowctl2_consolidate_tmax5"
    )


def test_loop_v1_phasecalib_alias_appends_runtime_tag():
    resolved = resolve_condition_alias("SIS_cf_mix_loop_v1_phasecalib")
    assert resolved == (
        "SIS_horizon_self_correct_cf_mix_netharm_direct_allow_"
        "budgeted_allowctl2_consolidate_tmax5_phasecalib"
    )


def test_loop_v1_nativeallow_alias_appends_runtime_tag():
    resolved = resolve_condition_alias("SIS_cf_mix_loop_v1_nativeallow")
    assert resolved == (
        "SIS_horizon_self_correct_cf_mix_netharm_direct_allow_"
        "budgeted_allowctl2_consolidate_tmax5_nativeallow"
    )


def test_loop_v2_condition_alias_resolves_to_promoted_consolidation_mainline():
    resolved = resolve_condition_alias("SIS_cf_mix_loop_v2")
    assert resolved == (
        "SIS_horizon_self_correct_cf_mix_netharm_direct_allow_"
        "budgeted_allowctl2_consolidateq_tmax5"
    )


def test_pre_reveal_allow_value_requires_both_tickets():
    value = compute_pre_reveal_allow_value(
        0.4,
        0.2,
        0.9,
        1.0,
        0.1,
        0.2,
        harm_mass_wait=0.05,
        contrastive_ticket_available=False,
        positive_ticket_available=True,
    )
    assert value == 0.0


def test_pre_reveal_allow_value_penalizes_harm_mass():
    low_harm = compute_pre_reveal_allow_value(
        0.5,
        0.1,
        1.0,
        1.0,
        0.2,
        0.1,
        harm_mass_wait=0.01,
        contrastive_ticket_available=True,
        positive_ticket_available=True,
    )
    high_harm = compute_pre_reveal_allow_value(
        0.5,
        0.1,
        1.0,
        1.0,
        0.2,
        0.1,
        harm_mass_wait=0.25,
        contrastive_ticket_available=True,
        positive_ticket_available=True,
    )
    assert low_harm > high_harm
