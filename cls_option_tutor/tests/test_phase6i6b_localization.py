import numpy as np

from cls_option_tutor.env.state import BlockState, QueryState
from cls_option_tutor.interfaces import LearnerStep, Option, TutorStep
from cls_option_tutor.experiments.condition_overrides import (
    apply_condition_overrides,
    extract_scripted_protocol_name,
    resolve_condition_alias,
)
from cls_option_tutor.experiments.metrics_extractors import compute_6fg_metrics
from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
from cls_option_tutor.tutor.sparse_tutor_scoring import (
    compute_ban_oracle_stats,
    compute_postreveal_shift_decomp,
)


def _make_option(index, *, is_correct=False, risk_class=0):
    return Option(
        index=index,
        text=["tok", str(index)],
        danger_vec=np.zeros(4),
        is_correct=is_correct,
        risk_class=risk_class,
        rendered_output=["out", str(index)],
    )


def _make_qs(query_id=0):
    qs = QueryState(
        query_id=query_id,
        target_output=["red", "blue"],
        true_program=["prog"],
        hp=4,
    )
    qs.menu = [
        _make_option(10, is_correct=True, risk_class=0),
        _make_option(20, is_correct=False, risk_class=4),
        _make_option(30, is_correct=False, risk_class=2),
    ]
    qs.option_diag_labels = {
        10: "",
        20: "safe_far",
        30: "safe_diagnostic_wrong",
    }
    qs.post_reveal_phase = True
    qs.n_safe_diag_wrong_reveals = 1
    qs.last_reveal_option_index = 30
    return qs


def test_condition_routing_supports_traj_v2_and_net_badmass():
    class Dummy:
        pass

    cfg = Dummy()
    cfg.env = Dummy()
    cfg.tutor = Dummy()
    cfg.env.highlight_mode = "diagnostic"
    cfg.env.highlight_strength = 1.0
    cfg.env.diagnostic_quota_strict = False
    cfg.tutor.tutor_lg_mode = "off"
    cfg.tutor.postreveal_value_mode = "legacy"
    cfg.tutor.mix_target_mode = "current"
    cfg.tutor.postreveal_info_weight = 0.0
    cfg.tutor.use_bayesian_postreveal_value = False
    cfg.tutor.joint_mix_replay_gate = False
    cfg.tutor.max_highlight_cells = 2
    cfg.tutor.force_postreveal_action = "none"
    cfg.tutor.protect_safe_diag_hard_guard = False

    apply_condition_overrides(cfg, "SIS_horizon_self_correct_cf_mix_netbadmass_traj_v2_harminfo_bayes_jointmix")
    assert cfg.tutor.tutor_lg_mode == "horizon_self_correct"
    assert cfg.tutor.postreveal_value_mode == "traj_v2"
    assert cfg.tutor.mix_target_mode == "net_badmass"
    assert cfg.tutor.postreveal_info_weight == 0.25
    assert cfg.tutor.use_bayesian_postreveal_value is True
    assert cfg.tutor.joint_mix_replay_gate is True


def test_condition_routing_supports_direct_netharm_and_allow():
    class Dummy:
        pass

    cfg = Dummy()
    cfg.env = Dummy()
    cfg.tutor = Dummy()
    cfg.env.highlight_mode = "diagnostic"
    cfg.env.highlight_strength = 1.0
    cfg.env.diagnostic_quota_strict = False
    cfg.tutor.tutor_lg_mode = "off"
    cfg.tutor.postreveal_value_mode = "legacy"
    cfg.tutor.mix_target_mode = "current"
    cfg.tutor.postreveal_info_weight = 0.0
    cfg.tutor.use_bayesian_postreveal_value = False
    cfg.tutor.joint_mix_replay_gate = False
    cfg.tutor.direct_mix_selector = False
    cfg.tutor.productive_allow_planning = False
    cfg.tutor.max_highlight_cells = 2
    cfg.tutor.force_postreveal_action = "none"
    cfg.tutor.protect_safe_diag_hard_guard = False

    apply_condition_overrides(cfg, "SIS_horizon_self_correct_cf_mix_netharm_direct_allow")
    assert cfg.tutor.tutor_lg_mode == "horizon_self_correct"
    assert cfg.tutor.postreveal_value_mode == "traj_v2"
    assert cfg.tutor.mix_target_mode == "net_badmass"
    assert cfg.tutor.direct_mix_selector is True
    assert cfg.tutor.productive_allow_planning is True


def test_resolve_condition_alias_preserves_runtime_suffixes():
    resolved = resolve_condition_alias("w1_sc_diagnostic_safe_tmax4_er05_cpoff")
    assert resolved == "script_wrong1_self_correct_diagnostic_safe_tmax4_er05_cpoff"


def test_extract_scripted_protocol_name_strips_runtime_override_suffixes():
    protocol = extract_scripted_protocol_name("script_wrong1_self_correct_diagnostic_safe_tmax5_er0_cpoff")
    assert protocol == "script_wrong1_self_correct_diagnostic_safe"

    no_tutor_protocol = extract_scripted_protocol_name("no_tutor_reveal_tmax4_er05")
    assert no_tutor_protocol == "no_tutor_reveal"


def test_condition_routing_supports_tmax_reveal_and_correct_pick_ablation_tags():
    class Dummy:
        pass

    cfg = Dummy()
    cfg.env = Dummy()
    cfg.learner = Dummy()
    cfg.tutor = Dummy()
    cfg.env.highlight_mode = "diagnostic"
    cfg.env.highlight_strength = 1.0
    cfg.env.diagnostic_quota_strict = False
    cfg.env.T_max = 3
    cfg.learner.eta_reveal = 1.0
    cfg.learner.correct_pick_learning_mode = "cortex_em"
    cfg.tutor.tutor_lg_mode = "off"
    cfg.tutor.postreveal_value_mode = "legacy"
    cfg.tutor.mix_target_mode = "current"
    cfg.tutor.postreveal_info_weight = 0.0
    cfg.tutor.use_bayesian_postreveal_value = False
    cfg.tutor.joint_mix_replay_gate = False
    cfg.tutor.direct_mix_selector = False
    cfg.tutor.productive_allow_planning = False
    cfg.tutor.max_highlight_cells = 2
    cfg.tutor.force_postreveal_action = "none"
    cfg.tutor.protect_safe_diag_hard_guard = False

    apply_condition_overrides(cfg, "no_tutor_reveal_tmax4_er05_cpoff")
    assert cfg.env.T_max == 4
    assert cfg.learner.eta_reveal == 0.5
    assert cfg.learner.correct_pick_learning_mode == "off"

    apply_condition_overrides(cfg, "script_wrong1_self_correct_diagnostic_safe_tmax5_er0")
    assert cfg.env.T_max == 5
    assert cfg.learner.eta_reveal == 0.0


def test_postreveal_shift_decomp_keeps_unconditional_pick_mass_when_refresh_slot_present():
    active = [
        _make_option(0, is_correct=True, risk_class=0),
        _make_option(1, is_correct=False, risk_class=1),
    ]
    labels = {0: "", 1: "safe_diagnostic_wrong"}
    wait_probs_full = np.array([0.40, 0.20, 0.40])   # refresh=0.40
    action_probs_full = np.array([0.50, 0.10, 0.40])  # refresh unchanged

    decomp = compute_postreveal_shift_decomp(
        active,
        wait_probs_full,
        action_probs_full,
        labels,
        last_wrong_index=1,
        hp_scale=4,
        ban_target_index=1,
    )

    assert abs(decomp["p_correct_wait"] - 0.40) < 1e-9
    assert abs(decomp["p_correct_action"] - 0.50) < 1e-9
    assert abs(decomp["removed_prob_mass"] - 0.20) < 1e-9


def test_ban_oracle_stats_can_prefer_net_badmass_over_removed_badmass():
    active = [
        _make_option(0, is_correct=True, risk_class=0),
        _make_option(1, is_correct=False, risk_class=4),
        _make_option(2, is_correct=False, risk_class=2),
    ]
    labels = {
        0: "",
        1: "safe_far",
        2: "safe_diagnostic_wrong",
    }
    wait_probs = np.array([0.30, 0.40, 0.30])
    ban_probs_by_index = {
        1: np.array([0.25, 0.00, 0.75]),  # redistribution mostly to repeated wrong -> poor net gain
        2: np.array([0.60, 0.40, 0.00]),  # redistribution to correct + far wrong -> better net gain
    }

    audit = compute_ban_oracle_stats(
        active,
        wait_probs,
        ban_probs_by_index,
        labels,
        last_wrong_index=2,
        hp_scale=4,
    )

    assert audit["removed_oracle_index"] == 1
    assert audit["net_oracle_index"] == 2
    assert audit["per_target"][1]["removed_harm_mass"] > audit["per_target"][2]["removed_harm_mass"]
    assert audit["per_target"][2]["net_harm_drop"] > audit["per_target"][1]["net_harm_drop"]


def test_postreveal_decomp_tracks_option_identity_not_active_position():
    active = [
        _make_option(10, is_correct=True, risk_class=0),
        _make_option(20, is_correct=False, risk_class=1),
        _make_option(30, is_correct=False, risk_class=0),
    ]
    labels = {10: "", 20: "safe_diagnostic_wrong", 30: "safe_far"}
    wait_probs = np.array([0.25, 0.45, 0.20])
    action_probs = np.array([0.40, 0.00, 0.35])

    decomp = compute_postreveal_shift_decomp(
        active,
        wait_probs,
        action_probs,
        labels,
        last_wrong_index=20,
        hp_scale=4,
        ban_target_index=20,
    )

    assert abs(decomp["removed_prob_mass"] - 0.45) < 1e-9
    assert decomp["same_wrong_drop"] > 0.44


def test_mix_target_audit_metrics_read_chosen_detail_fields():
    qs = _make_qs()
    block = BlockState(
        queries=[qs],
        obs_phase_queries=0,
        teach_phase_queries=1,
        eval_phase_queries=0,
    )
    block.tutor_trace = [
        TutorStep(round_t=0, query_id=0, action="MIX", ban_index=30, highlight_cells=(0,))
    ]
    block.learner_trace = [
        LearnerStep(
            round_t=0,
            query_id=0,
            action="pick",
            pick_index=30,
            correct=False,
            damage=2,
            hp_before=4,
            hp_after=2,
            menu_size=3,
        ),
        LearnerStep(
            round_t=1,
            query_id=0,
            action="pick",
            pick_index=10,
            correct=True,
            damage=0,
            hp_before=2,
            hp_after=2,
            menu_size=3,
        ),
    ]
    block._decision_trace = [{
        "query_id": 0,
        "round_t": 0,
        "pre_post_reveal_phase": True,
        "pre_has_safe_diag_opp": False,
        "pre_has_high_risk_opp": True,
        "pre_last_reveal_option_index": 30,
        "chosen_action": "MIX",
        "chosen_ban_index": 30,
        "scoring": {
            "q_wait": 0.0,
            "candidates": [{
                "action": "WAIT",
                "q_use": 0.0,
            }, {
                "action": "MIX",
                "q_use": 0.4,
                "mix_ban_target_was_top_prob_wrong": True,
                "mix_ban_target_was_far_wrong": False,
                "mix_ban_target_was_correct": False,
                "mix_ban_target_policy_prob_wait": 0.30,
                "mix_ban_target_badness": 1.0,
                "mix_removed_target_regret": 0.05,
                "mix_net_target_regret": 0.02,
                "mix_removed_oracle_mass": 0.35,
                "mix_net_oracle_drop": 0.10,
                "mix_joint_gate_applied": True,
                "mix_joint_gate_replaced": True,
                "mix_joint_target_regret": 0.07,
                "mix_joint_highlight_regret": 0.00,
                "mix_joint_regret": 0.09,
                "mix_joint_interaction_regret": 0.02,
                "removed_prob_mass": 0.30,
                "removed_bad_mass": 0.30,
                "bad_mass_drop": 0.08,
                "delta_p_correct": 0.05,
                "correct_margin_gain": 0.10,
            }],
            "chosen_detail": {
                "action": "MIX",
                "q_use": 0.4,
                "mix_ban_target_was_top_prob_wrong": True,
                "mix_ban_target_was_far_wrong": False,
                "mix_ban_target_was_correct": False,
                "mix_ban_target_policy_prob_wait": 0.30,
                "mix_ban_target_badness": 1.0,
                "mix_removed_target_regret": 0.05,
                "mix_net_target_regret": 0.02,
                "mix_removed_oracle_mass": 0.35,
                "mix_net_oracle_drop": 0.10,
                "mix_joint_gate_applied": True,
                "mix_joint_gate_replaced": True,
                "mix_joint_target_regret": 0.07,
                "mix_joint_highlight_regret": 0.00,
                "mix_joint_regret": 0.09,
                "mix_joint_interaction_regret": 0.02,
                "removed_prob_mass": 0.30,
                "removed_bad_mass": 0.30,
                "bad_mass_drop": 0.08,
                "delta_p_correct": 0.05,
                "correct_margin_gain": 0.10,
            },
        },
    }, {
        "query_id": 0,
        "round_t": 1,
        "pre_post_reveal_phase": True,
        "pre_has_safe_diag_opp": False,
        "pre_has_high_risk_opp": False,
        "chosen_action": "WAIT",
        "wait_reason": "WAIT_GRACE",
        "scoring": {"q_wait": 0.0, "candidates": []},
    }]

    metrics = compute_6fg_metrics(block)
    assert metrics["MIXChosenCount"] == 1
    assert metrics["MIXBanTargetWasLastWrongRate"] == 1.0
    assert metrics["MIXBanTargetWasTopProbWrongRate"] == 1.0
    assert metrics["MIXBanTargetMeanPolicyMass"] == 0.3
    assert metrics["MIXNetTargetRegretMean"] == 0.02
    assert metrics["MIXJointGateAppliedRate"] == 1.0
    assert metrics["MIXJointGateReplacedRate"] == 1.0
    assert metrics["MIXJointInteractionRegretMean"] == 0.02
    assert metrics["CueThenGraceCorrectRate"] == 1.0


def test_joint_mix_replay_gate_can_replace_mix_candidate(monkeypatch):
    class Dummy:
        pass

    tutor = SparseTutorAgent.__new__(SparseTutorAgent)
    tutor.cfg = Dummy()
    tutor.cfg.tutor = Dummy()
    tutor.cfg.tutor.joint_mix_replay_gate = True
    tutor.cfg.tutor.max_highlight_cells = 2
    tutor.cfg.tutor.protect_safe_diag_hard_guard = False
    tutor.cfg.tutor.tutor_lg_mode = "horizon_self_correct"
    tutor._mix_target_audit_cache = {}

    qs = _make_qs(query_id=1)
    active = list(qs.menu)
    candidates = [
        {"action": "WAIT"},
        {"action": "MIX", "ban_index": 20, "highlight_cells": (0,)},
    ]

    monkeypatch.setattr(
        "cls_option_tutor.tutor.sparse_tutor.build_learning_ban_pool",
        lambda qs, non_correct, diag_labels, lg_mode, hard_guard_enabled: (None, [o for o in non_correct]),
    )
    monkeypatch.setattr(
        tutor,
        "_candidate_joint_mix_highlight_sets",
        lambda qs, active, learner, current_hl_cells: [tuple(current_hl_cells), (1,)],
    )
    monkeypatch.setattr(tutor, "_compute_learner_probs", lambda *args, **kwargs: np.array([0.5, 0.3, 0.2]))

    def _fake_q(qs, active, spec, learner, **kwargs):
        ban = spec.get("ban_index")
        cells = tuple(spec.get("highlight_cells") or ())
        q = 0.10
        if ban == 30:
            q += 0.35
        if cells == (1,):
            q += 0.20
        return q, {}

    monkeypatch.setattr(tutor, "_compute_q_use", _fake_q)
    monkeypatch.setattr(tutor, "_compute_mix_target_audit", lambda *args, **kwargs: {"per_target": {20: {}, 30: {}}})
    chosen = {}
    monkeypatch.setattr(tutor, "_record_mix_target_audit", lambda qs, audit, chosen_index: chosen.setdefault("index", chosen_index))

    updated_candidates, info = tutor._apply_joint_mix_replay_gate(
        qs,
        active,
        learner=None,
        candidates=candidates,
        wait_probs_lc=np.array([0.6, 0.2, 0.2]),
        p_death_wait=0.0,
        p_timeout_wait=0.0,
    )

    updated_mix = updated_candidates[1]
    assert info["evaluated"] is True
    assert info["replaced"] is True
    assert updated_mix["ban_index"] == 30
    assert updated_mix["highlight_cells"] == (1,)
    assert updated_mix["mix_joint_gate_applied"] is True
    assert updated_mix["mix_joint_gate_replaced"] is True
    assert updated_mix["mix_joint_target_regret"] > 0.0
    assert updated_mix["mix_joint_interaction_regret"] >= 0.0
    assert chosen["index"] == 30


def test_direct_mix_selector_can_replace_mix_candidate(monkeypatch):
    class Dummy:
        pass

    tutor = SparseTutorAgent.__new__(SparseTutorAgent)
    tutor.cfg = Dummy()
    tutor.cfg.tutor = Dummy()
    tutor.cfg.tutor.direct_mix_selector = True
    tutor.cfg.tutor.direct_mix_top_k = 2
    tutor.cfg.tutor.direct_mix_top_m = 2
    tutor.cfg.tutor.max_highlight_cells = 2
    tutor.cfg.tutor.protect_safe_diag_hard_guard = False
    tutor.cfg.tutor.tutor_lg_mode = "horizon_self_correct"
    tutor._mix_target_audit_cache = {}

    qs = _make_qs(query_id=2)
    active = list(qs.menu)
    candidates = [
        {"action": "WAIT"},
        {"action": "MIX", "ban_index": 20, "highlight_cells": (0,)},
    ]

    monkeypatch.setattr(
        "cls_option_tutor.tutor.sparse_tutor.build_learning_ban_pool",
        lambda qs, non_correct, diag_labels, lg_mode, hard_guard_enabled: (None, [o for o in non_correct]),
    )
    monkeypatch.setattr(
        tutor,
        "_candidate_joint_mix_highlight_sets",
        lambda qs, active, learner, current_hl_cells: [tuple(current_hl_cells), (1,)],
    )
    monkeypatch.setattr(tutor, "_compute_learner_probs", lambda *args, **kwargs: np.array([0.5, 0.3, 0.2]))

    audit = {
        "net_oracle_drop": 0.12,
        "per_target": {
            20: {"target_index": 20, "net_harm_drop": 0.03, "removed_harm_mass": 0.04, "target_is_correct": 0.0},
            30: {"target_index": 30, "net_harm_drop": 0.12, "removed_harm_mass": 0.03, "target_is_correct": 0.0},
        },
    }
    monkeypatch.setattr(tutor, "_compute_mix_target_audit", lambda *args, **kwargs: audit)
    chosen = {}
    monkeypatch.setattr(tutor, "_record_mix_target_audit", lambda qs, audit, chosen_index: chosen.setdefault("index", chosen_index))

    def _fake_q(qs, active, spec, learner, **kwargs):
        ban = spec.get("ban_index")
        cells = tuple(spec.get("highlight_cells") or ())
        q = 0.10
        if ban == 30:
            q += 0.25
        if cells == (1,):
            q += 0.15
        return q, {}

    monkeypatch.setattr(tutor, "_compute_q_use", _fake_q)

    updated_candidates, info = tutor._apply_direct_mix_selector(
        qs,
        active,
        learner=None,
        candidates=candidates,
        wait_probs_lc=np.array([0.6, 0.2, 0.2]),
        p_death_wait=0.0,
        p_timeout_wait=0.0,
    )

    updated_mix = updated_candidates[1]
    assert info["evaluated"] is True
    assert info["applied"] is True
    assert info["suppressed"] is False
    assert updated_mix["ban_index"] == 30
    assert updated_mix["highlight_cells"] == (1,)
    assert updated_mix["mix_direct_selector_applied"] is True
    assert updated_mix["mix_direct_selected_ban_index"] == 30
    assert updated_mix["mix_direct_selected_net_harm_drop"] == 0.12
    assert chosen["index"] == 30


def test_select_ban_target_returns_none_when_productive_allow_guard_triggers(monkeypatch):
    class Dummy:
        pass

    tutor = SparseTutorAgent.__new__(SparseTutorAgent)
    tutor.cfg = Dummy()
    tutor.cfg.tutor = Dummy()
    tutor.cfg.tutor.productive_allow_planning = True
    tutor.cfg.tutor.tutor_lg_mode = "horizon_self_correct"
    tutor.cfg.tutor.mix_target_mode = "current"
    tutor.cfg.tutor.protect_safe_diag_hard_guard = False
    tutor._mix_target_audit_cache = {}

    qs = _make_qs(query_id=3)
    qs.post_reveal_phase = False
    qs.n_safe_diag_wrong_reveals = 0
    active = list(qs.menu)
    non_correct = [o for o in active if not o.is_correct]

    monkeypatch.setattr(tutor, "_should_preserve_productive_allow", lambda *args, **kwargs: True)
    chosen = tutor._select_ban_target(qs, non_correct, learner=None)
    assert chosen is None


def test_rescue_detail_keeps_postreveal_decomp_for_forced_mix_trace():
    qs = _make_qs()
    active = list(qs.menu)

    class Dummy:
        pass

    tutor = SparseTutorAgent.__new__(SparseTutorAgent)
    tutor.cfg = Dummy()
    tutor.cfg.tutor = Dummy()
    tutor.cfg.tutor.postreveal_value_mode = "traj_v2"
    tutor.cfg.tutor.postreveal_info_weight = 0.0
    tutor._mix_target_audit_cache = {
        (int(qs.query_id), int(qs.rounds_used)): {
            "target_mode": "net_badmass",
            "removed_oracle_index": 20,
            "net_oracle_index": 30,
            "chosen_matches_removed_oracle": False,
            "chosen_matches_net_oracle": True,
            "removed_target_regret": 0.05,
            "net_target_regret": 0.0,
            "removed_oracle_mass": 0.40,
            "net_oracle_drop": 0.12,
            "chosen_record": {
                "target_prob": 0.45,
                "target_harm": 1.50,
                "target_label": "safe_diagnostic_wrong",
                "target_is_last_wrong": 1.0,
                "target_is_highrisk": 0.0,
                "target_is_safe_diag": 1.0,
                "target_is_far_wrong": 0.0,
                "target_is_top_prob_wrong": 1.0,
                "target_is_correct": 0.0,
            },
        }
    }

    wait_probs = np.array([0.20, 0.35, 0.25, 0.20])
    mix_probs = np.array([0.40, 0.15, 0.20, 0.25])
    detail = {"action": "MIX", "mode": "rescue"}

    tutor._attach_postreveal_diagnostics_to_detail(
        detail,
        qs,
        active,
        {"action": "MIX", "ban_index": 30, "highlight_cells": (0,)},
        wait_probs,
        mix_probs,
        p_death=0.0,
        p_timeout=0.1,
    )

    assert "postreveal_decomp" in detail
    assert detail["removed_bad_mass"] > 0.0
    assert detail["bad_mass_drop"] > 0.0
    assert detail["mix_target_mode"] == "net_badmass"
    assert detail["mix_ban_target_was_last_wrong"] is True
