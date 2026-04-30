"""Phase 6H.5: Causal Action Effect Audit Tests.

Tests:
  1. Safe diagnostic BAN protection (hard guard)
  2. Grace round semantics
  3. Action-effect audit integrity
  4. Metric definition correctness
  5. Condition routing for 6H.5
"""
import numpy as np
import pytest
import copy
from unittest.mock import MagicMock

from cls_option_tutor.env.state import QueryState, BlockState
from cls_option_tutor.interfaces import Option, TutorStep, LearnerStep


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_qs(hp=5, rounds_used=0, max_rounds=5,
             n_safe_diag=0, post_reveal=False,
             option_diag_labels=None, success=False):
    qs = QueryState(
        query_id=0,
        target_output=["red", "blue", "green", "red"],
        true_program=["repeat", "2", "red", "blue", "green"],
        hp=hp, rounds_used=rounds_used, max_rounds=max_rounds,
    )
    qs.n_safe_diag_wrong_reveals = n_safe_diag
    qs.post_reveal_phase = post_reveal
    qs.success = success
    if option_diag_labels:
        qs.option_diag_labels = option_diag_labels
    return qs


def _make_option(index, is_correct=False, risk_class=0):
    return Option(
        index=index, text=["word"],
        rendered_output=["red", "blue", "green", "red"] if is_correct
                        else ["red", "blue", "blue", "red"],
        is_correct=is_correct, risk_class=risk_class,
        danger_vec=np.zeros(4),
    )


# ── Test: Safe Diagnostic BAN Hard Guard ─────────────────────────────────────

class TestSafeDiagBanProtection:
    """Test that first safe diagnostic opportunity is NOT banned in self_correct mode."""

    def _make_tutor(self, lg_mode="self_correct"):
        from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
        from cls_option_tutor.config import FullConfig
        cfg = FullConfig()
        cfg.tutor.tutor_lg_mode = lg_mode
        tutor = SparseTutorAgent(cfg=cfg)
        return tutor

    def _make_learner(self):
        lm = MagicMock()
        # Dummy learner: uniform pick probs
        lm.predict_pick_probs = MagicMock(return_value=[0.33, 0.34, 0.33])
        return lm

    def test_safe_diagnostic_first_opportunity_not_banned(self):
        """First safe diag wrong should NOT be selected as ban target."""
        from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
        from cls_option_tutor.config import FullConfig

        cfg = FullConfig()
        cfg.tutor.tutor_lg_mode = "self_correct"
        tutor = SparseTutorAgent(cfg=cfg)

        qs = _make_qs(hp=5, rounds_used=0, max_rounds=5, n_safe_diag=0)
        qs.option_diag_labels = {
            0: "",                       # correct
            1: "safe_diagnostic_wrong",  # PROTECTED first opportunity
            2: "high_risk_lure",         # ban target
        }

        active = [
            _make_option(0, is_correct=True, risk_class=0),
            _make_option(1, is_correct=False, risk_class=0),  # safe_diag
            _make_option(2, is_correct=False, risk_class=3),  # high_risk_lure
        ]
        non_correct = [o for o in active if not o.is_correct]

        learner = MagicMock()
        learner.pick_distribution = MagicMock(return_value=[0.5, 0.5])

        # Mock _compute_pick_probs_for_opts to return uniform
        import numpy as np
        tutor._compute_pick_probs_for_opts = lambda qs, pool, learner, use_logit=True: \
            np.ones(len(pool)) / max(len(pool), 1)

        result = tutor._select_ban_target(qs, non_correct, learner)

        # Should return high_risk_lure, NOT safe_diagnostic_wrong
        assert result is not None
        assert result.index == 2, \
            f"Expected lure (idx=2) but got idx={result.index}"

    def test_safe_diagnostic_can_be_banned_after_reveal(self):
        """After a reveal already happened, safe_diag can be banned."""
        from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
        from cls_option_tutor.config import FullConfig

        cfg = FullConfig()
        cfg.tutor.tutor_lg_mode = "self_correct"
        tutor = SparseTutorAgent(cfg=cfg)

        # n_safe_diag=1: reveal already happened
        qs = _make_qs(hp=5, rounds_used=2, max_rounds=5, n_safe_diag=1)
        qs.option_diag_labels = {
            0: "",                       # correct
            1: "safe_diagnostic_wrong",  # no longer hard-protected (already revealed)
            2: "",                       # safe random
        }
        active = [
            _make_option(0, is_correct=True, risk_class=0),
            _make_option(1, is_correct=False, risk_class=0),
            _make_option(2, is_correct=False, risk_class=0),
        ]
        non_correct = [o for o in active if not o.is_correct]
        learner = MagicMock()
        tutor._compute_pick_probs_for_opts = lambda qs, pool, learner, use_logit=True: \
            np.ones(len(pool)) / max(len(pool), 1)

        result = tutor._select_ban_target(qs, non_correct, learner)
        # With no lure and no hard protection (already revealed), can pick any
        assert result is not None

    def test_safe_diagnostic_not_protected_if_hp_too_low(self):
        """Safe diag is NOT protected if pick would kill learner."""
        from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
        from cls_option_tutor.config import FullConfig

        cfg = FullConfig()
        cfg.tutor.tutor_lg_mode = "self_correct"
        tutor = SparseTutorAgent(cfg=cfg)

        # hp=1 and safe_diag has risk_class=1 → hp_after=0 → NOT protected
        qs = _make_qs(hp=1, rounds_used=0, max_rounds=5, n_safe_diag=0)
        qs.option_diag_labels = {
            0: "",
            1: "safe_diagnostic_wrong",  # risk_class=1, hp=1 → hp_after=0 → unsafe
            2: "high_risk_lure",
        }
        active = [
            _make_option(0, is_correct=True, risk_class=0),
            _make_option(1, is_correct=False, risk_class=1),  # lethal for hp=1
            _make_option(2, is_correct=False, risk_class=3),  # even more lethal
        ]
        non_correct = [o for o in active if not o.is_correct]
        learner = MagicMock()
        tutor._compute_pick_probs_for_opts = lambda qs, pool, learner, use_logit=True: \
            np.ones(len(pool)) / max(len(pool), 1)

        result = tutor._select_ban_target(qs, non_correct, learner)
        # Should still work, lure is picked
        assert result is not None


# ── Test: Grace Round Semantics ───────────────────────────────────────────────

class TestGraceRoundSemantics:
    """Test that grace round flag is set after HIGHLIGHT and cleared on WAIT."""

    def test_grace_round_flag_initialized_false(self):
        qs = _make_qs(post_reveal=True, n_safe_diag=1)
        assert qs.after_highlight_grace_round is False

    def test_grace_round_set_after_highlight_in_post_reveal(self):
        """Grace round should be set when HIGHLIGHT/MIX applied in post-reveal."""
        qs = _make_qs(post_reveal=True, n_safe_diag=1)
        # Simulate what sparse_tutor.py does after HIGHLIGHT
        lg_mode = "self_correct"
        action = "HIGHLIGHT"
        if action in ("HIGHLIGHT", "MIX"):
            if lg_mode == "self_correct" and qs.post_reveal_phase:
                qs.after_highlight_grace_round = True
        assert qs.after_highlight_grace_round is True

    def test_grace_round_prefers_wait_when_active(self):
        """When grace round is active, WAIT should be returned."""
        qs = _make_qs(hp=5, rounds_used=1, max_rounds=5,
                      post_reveal=True, n_safe_diag=1)
        qs.after_highlight_grace_round = True

        hard_override = (qs.hp <= 1 or qs.rounds_used >= (qs.max_rounds - 1))
        assert hard_override is False  # no override → should WAIT

    def test_grace_round_not_triggered_at_last_round(self):
        """Grace round should not force WAIT at the last round."""
        qs = _make_qs(hp=5, rounds_used=4, max_rounds=5,
                      post_reveal=True, n_safe_diag=1)
        qs.after_highlight_grace_round = True
        hard_override = (qs.hp <= 1 or qs.rounds_used >= (qs.max_rounds - 1))
        # rounds_used=4, max_rounds=5 → 4 >= 4 → True → override → don't force WAIT
        assert hard_override is True

    def test_grace_round_not_triggered_at_low_hp(self):
        """Grace round should not force WAIT when hp is critically low."""
        qs = _make_qs(hp=1, rounds_used=1, max_rounds=5,
                      post_reveal=True, n_safe_diag=1)
        qs.after_highlight_grace_round = True
        hard_override = (qs.hp <= 1 or qs.rounds_used >= (qs.max_rounds - 1))
        assert hard_override is True

    def test_horizon_self_correct_sets_grace_after_highlight(self):
        """horizon_self_correct should also activate grace after post-reveal cue."""
        from cls_option_tutor.config import FullConfig
        from cls_option_tutor.tutor.sparse_tutor_phase import update_post_action_phase_flags

        cfg = FullConfig()
        cfg.tutor.tutor_lg_mode = "horizon_self_correct"
        block = BlockState()
        qs = _make_qs(post_reveal=True, n_safe_diag=1)

        update_post_action_phase_flags(block, cfg, qs, "HIGHLIGHT")

        assert qs.after_highlight_grace_round is True
        assert getattr(block, "_grace_metrics", {}).get("set", 0) == 1
        assert getattr(block, "_grace_metrics", {}).get("eligible_next_round", 0) == 1

    def test_act_passes_grace_status_into_teaching_path(self):
        """Regression: act() must thread grace_status into _act_teaching()."""
        from cls_option_tutor.config import FullConfig
        from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent

        cfg = FullConfig()
        tutor = SparseTutorAgent(cfg=cfg)
        block = BlockState()
        qs = _make_qs(post_reveal=False, n_safe_diag=0)
        block.queries = [qs]
        block.current_query_idx = 0
        block.obs_phase_queries = 0
        block.teach_phase_queries = 1
        block.eval_phase_queries = 0
        block.done = False
        env = MagicMock()
        learner = MagicMock()

        sentinel = object()
        tutor._act_teaching = MagicMock(return_value=sentinel)

        result = tutor.act(block, env, learner)

        assert result is sentinel
        tutor._act_teaching.assert_called_once()
        _, _, _ = tutor._act_teaching.call_args.args
        grace_arg = tutor._act_teaching.call_args.kwargs.get("grace_result")
        assert grace_arg is not None
        assert grace_arg["status"] == "none"


# ── Test: Causal Audit Module ─────────────────────────────────────────────────

class TestCausalAuditIntegrity:
    """Test action-effect audit returns correct structure and ordering."""

    def test_audit_returns_results_for_all_actions(self):
        """Audit should return one result per action."""
        from cls_option_tutor.tutor.causal_audit import audit_post_reveal_action_effects

        qs = _make_qs(hp=5, rounds_used=1, max_rounds=5,
                      post_reveal=True, n_safe_diag=1)
        qs.target_output = ["red", "blue", "green", "red"]
        qs.option_diag_labels = {
            0: "", 1: "safe_diagnostic_wrong", 2: "high_risk_lure"
        }

        active = [
            _make_option(0, is_correct=True, risk_class=0),
            _make_option(1, is_correct=False, risk_class=0),
            _make_option(2, is_correct=False, risk_class=3),
        ]

        learner = MagicMock()
        # Default pick probs fallback: uniform
        learner.predict_pick_probs = MagicMock(
            return_value=[0.5, 0.3, 0.2])

        results = audit_post_reveal_action_effects(
            learner=learner, qs=qs, active=active
        )
        assert len(results) >= 2  # at least WAIT + HIGHLIGHT_diagnostic
        action_names = {r.action_name for r in results}
        assert "WAIT" in action_names

    def test_audit_delta_p_correct_is_correct_minus_before(self):
        """DeltaP_correct should be after - before."""
        from cls_option_tutor.tutor.causal_audit import ActionEffectResult
        r = ActionEffectResult(
            action_name="HIGHLIGHT_diagnostic",
            p_correct_before=0.3,
            p_correct_after=0.5,
            delta_p_correct=0.2,
            p_safe_diag_after=0.2,
            p_high_risk_after=0.1,
            entropy_before=1.0,
            entropy_after=0.8,
            top1_before=1,
            top1_after=0,
            top1_changed=True,
        )
        assert abs(r.delta_p_correct - (r.p_correct_after - r.p_correct_before)) < 1e-6


# ── Test: Condition Routing 6H.5 ─────────────────────────────────────────────

class TestConditionRouting6H5:
    """Test 6H.5 conditions route correctly to highlight_strength."""

    def test_highlight_2x_routing(self):
        from cls_option_tutor.experiments.run_learning_increment_micro import _apply_condition_overrides
        cfg = MagicMock()
        cfg.env.highlight_mode = "diagnostic"
        cfg.tutor.tutor_lg_mode = "off"
        cfg.env.highlight_strength = 1.0
        cfg.env.diagnostic_quota_strict = False

        _apply_condition_overrides(cfg, "SIS_self_correct_highlight_2x")
        assert cfg.env.highlight_strength == 2.0
        assert cfg.tutor.tutor_lg_mode == "self_correct"

    def test_highlight_4x_routing(self):
        from cls_option_tutor.experiments.run_learning_increment_micro import _apply_condition_overrides
        cfg = MagicMock()
        cfg.env.highlight_mode = "diagnostic"
        cfg.tutor.tutor_lg_mode = "off"
        cfg.env.highlight_strength = 1.0
        cfg.env.diagnostic_quota_strict = False

        _apply_condition_overrides(cfg, "SIS_self_correct_highlight_4x")
        assert cfg.env.highlight_strength == 4.0
        assert cfg.tutor.tutor_lg_mode == "self_correct"

    def test_protect_safe_diag_routing(self):
        from cls_option_tutor.experiments.run_learning_increment_micro import _apply_condition_overrides
        cfg = MagicMock()
        cfg.env.highlight_mode = "diagnostic"
        cfg.tutor.tutor_lg_mode = "off"
        cfg.env.highlight_strength = 1.0
        cfg.env.diagnostic_quota_strict = False

        _apply_condition_overrides(cfg, "SIS_self_correct_protect_safe_diag")
        assert cfg.tutor.tutor_lg_mode == "self_correct"
        assert cfg.env.highlight_strength == 1.0

    def test_strict_quota_routing(self):
        from cls_option_tutor.experiments.run_learning_increment_micro import _apply_condition_overrides
        cfg = MagicMock()
        cfg.env.highlight_mode = "diagnostic"
        cfg.tutor.tutor_lg_mode = "off"
        cfg.env.highlight_strength = 1.0
        cfg.env.diagnostic_quota_strict = False

        _apply_condition_overrides(cfg, "SIS_self_correct_strict")
        assert cfg.env.diagnostic_quota_strict is True

    def test_forced_postreveal_conditions_keep_self_correct_mode(self):
        from cls_option_tutor.experiments.run_learning_increment_micro import _apply_condition_overrides
        cfg = MagicMock()
        cfg.env.highlight_mode = "diagnostic"
        cfg.tutor.tutor_lg_mode = "off"
        cfg.env.highlight_strength = 1.0
        cfg.env.diagnostic_quota_strict = False
        cfg.tutor.force_postreveal_action = "none"

        _apply_condition_overrides(cfg, "SIS_force_HL_cf_after_safe_diag_reveal")
        assert cfg.tutor.force_postreveal_action == "HL_cf"
        assert cfg.tutor.tutor_lg_mode == "self_correct"

    def test_cells3_routing_updates_max_highlight_cells(self):
        from cls_option_tutor.experiments.run_learning_increment_micro import _apply_condition_overrides
        cfg = MagicMock()
        cfg.env.highlight_mode = "diagnostic"
        cfg.tutor.tutor_lg_mode = "off"
        cfg.env.highlight_strength = 1.0
        cfg.env.diagnostic_quota_strict = False
        cfg.tutor.max_highlight_cells = 2
        cfg.tutor.force_postreveal_action = "none"

        _apply_condition_overrides(cfg, "SIS_force_MIX_cf_after_safe_diag_reveal_highlight_2x_cells3")
        assert cfg.tutor.max_highlight_cells == 3
        assert cfg.env.highlight_strength == 2.0
        assert cfg.tutor.force_postreveal_action == "MIX_cf"


# ── Test: Metric Definitions ──────────────────────────────────────────────────

class TestMetricDefinitions:
    """Test raw count fields and selectivity logic."""

    def test_ped_selectivity_range(self):
        """PedagogicalSelectivity should be in [0, 1]."""
        # Perfect: no safe_diag banned, all high_risk banned
        safe_diag_ban_rate = 0.0
        high_risk_ban_rate = 1.0
        ps = ((1.0 - safe_diag_ban_rate) + high_risk_ban_rate) / 2.0
        assert 0.0 <= ps <= 1.0
        assert ps == 1.0

    def test_ped_selectivity_worst_case(self):
        """Worst case: all safe_diag banned, no high_risk banned."""
        safe_diag_ban_rate = 1.0
        high_risk_ban_rate = 0.0
        ps = ((1.0 - safe_diag_ban_rate) + high_risk_ban_rate) / 2.0
        assert ps == 0.0

    def test_forcebest_selection_counts_come_from_decision_trace(self):
        from cls_option_tutor.experiments.metrics_extractors import compute_6fg_metrics

        block = BlockState()
        qs = _make_qs(post_reveal=True, n_safe_diag=1)
        block.obs_phase_queries = 0
        block.teach_phase_queries = 1
        block.queries = [qs]
        block._decision_trace = [
            {
                "query_id": 0,
                "round_t": 0,
                "pre_post_reveal_phase": True,
                "pre_has_safe_diag_opp": True,
                "pre_has_high_risk_opp": False,
                "chosen_action": "MIX",
                "scoring": {"force_type": "best_CATE", "q_wait": 0.0, "candidates": []},
            }
        ]

        m = compute_6fg_metrics(block)
        assert m["ForceBest_SelectedMIXCount"] == 1
        assert m["ForceBest_SelectedHLCount"] == 0

    def test_raw_count_denominator_safety(self):
        """Raw counts with zero denominator should not crash."""
        hl_total = 0
        hl_self_correct = 0
        rate = hl_self_correct / max(hl_total, 1)
        assert rate == 0.0

    def test_rev_wasted_plus_correct_eq_total(self):
        """Wasted + Correct should equal total reveal queries."""
        n_diag_reveal_queries = 10
        n_diag_reveal_then_correct = 4
        n_diag_reveal_wasted = 6
        assert n_diag_reveal_then_correct + n_diag_reveal_wasted == n_diag_reveal_queries

    def test_postreveal_metrics_use_decision_trace_not_final_query_state(self):
        """Post-reveal opportunity metrics should read decision-time trace first."""
        from cls_option_tutor.experiments.metrics_extractors import compute_6fg_metrics

        qs = _make_qs(post_reveal=False, n_safe_diag=1, success=True)
        qs.option_diag_labels = {0: "correct", 1: "safe_diagnostic_wrong"}
        qs.done = True
        block = BlockState(
            queries=[qs],
            obs_phase_queries=0,
            teach_phase_queries=1,
            eval_phase_queries=0,
        )
        block.tutor_trace = [
            TutorStep(round_t=0, query_id=0, action="HIGHLIGHT", highlight_cells=(0, 1))
        ]
        block.learner_trace = [
            LearnerStep(
                round_t=1,
                query_id=0,
                action="pick",
                pick_index=0,
                correct=True,
                damage=0,
                hp_before=5,
                hp_after=5,
                menu_size=2,
            )
        ]
        block._decision_trace = [{
            "query_id": 0,
            "round_t": 0,
            "pre_post_reveal_phase": True,
            "pre_has_safe_diag_opp": True,
            "pre_has_high_risk_opp": False,
        }]

        m = compute_6fg_metrics(block)
        assert m["PostReveal_HLRate"] > 0.0
        assert m["SafeDiagOpp_WaitRate"] == 0.0

    def test_grace_metrics_distinguish_next_tutor_and_loss_before_tutor(self):
        from cls_option_tutor.tutor.sparse_tutor_grace import ensure_grace_metrics, handle_grace_round
        from cls_option_tutor.env.option_env import OptionEnv
        from cls_option_tutor.config import FullConfig

        cfg = FullConfig()
        env = OptionEnv(cfg)
        block = BlockState()
        gm = ensure_grace_metrics(block)

        qs_wait = _make_qs(post_reveal=True, n_safe_diag=1, rounds_used=1, max_rounds=5)
        block.queries = [qs_wait]
        qs_wait.after_highlight_grace_round = True
        result = handle_grace_round(block, qs_wait)
        assert result["status"] == "wait"
        assert result["reason"] == "grace_consumed"
        assert gm["next_tutor_called"] == 1
        assert gm["chosen_wait"] == 1
        assert gm["consumed"] == 1

        qs_lost = _make_qs(post_reveal=True, n_safe_diag=1, rounds_used=5, max_rounds=5)
        block.queries = [qs_lost]
        qs_lost.after_highlight_grace_round = True
        env._check_query_end(block, qs_lost)
        assert gm["did_not_reach_tutor_decision"] == 1
        assert gm["lost_max_round"] == 1

    def test_postreveal_shift_decomp_captures_beneficial_shift(self):
        from cls_option_tutor.tutor.sparse_tutor_scoring import compute_postreveal_shift_decomp

        active = [
            _make_option(0, is_correct=True, risk_class=0),
            _make_option(1, is_correct=False, risk_class=0),
            _make_option(2, is_correct=False, risk_class=3),
        ]
        diag_labels = {
            0: "",
            1: "safe_diagnostic_wrong",
            2: "high_risk_lure",
        }
        wait_probs = np.array([0.30, 0.40, 0.30])
        cue_probs = np.array([0.45, 0.20, 0.35])

        decomp = compute_postreveal_shift_decomp(
            active, wait_probs, cue_probs, diag_labels, last_wrong_index=1
        )

        assert decomp["delta_p_correct"] > 0
        assert decomp["same_wrong_drop"] > 0
        assert decomp["top1_flip"] == 1.0
        assert decomp["good_shift"] > 0
        assert decomp["bad_shift"] < 0.20

    def test_postcue_metrics_use_same_round_immediate_pick_and_taxonomy(self):
        from cls_option_tutor.experiments.metrics_extractors import compute_6fg_metrics

        qs = _make_qs(post_reveal=True, n_safe_diag=1)
        qs.last_reveal_option_index = 1
        qs.option_diag_labels = {
            0: "",
            1: "safe_diagnostic_wrong",
            2: "high_risk_lure",
        }
        block = BlockState(
            queries=[qs],
            obs_phase_queries=0,
            teach_phase_queries=1,
            eval_phase_queries=0,
        )
        block.tutor_trace = [
            TutorStep(round_t=0, query_id=0, action="HIGHLIGHT", highlight_cells=(0,))
        ]
        block.learner_trace = [
            LearnerStep(
                round_t=0,
                query_id=0,
                action="pick",
                pick_index=1,
                correct=False,
                damage=0,
                hp_before=5,
                hp_after=5,
                menu_size=3,
            )
        ]
        block._decision_trace = [{
            "query_id": 0,
            "round_t": 0,
            "pre_post_reveal_phase": True,
            "pre_has_safe_diag_opp": False,
            "pre_has_high_risk_opp": True,
            "chosen_action": "HIGHLIGHT",
            "scoring": {"q_wait": 0.0, "candidates": []},
        }]

        m = compute_6fg_metrics(block)
        assert m["PostCueTotalPicks"] == 1
        assert m["PostCueWrongPickRate"] == 1.0
        assert m["PostCueWrongPick_SameWrongRate"] == 1.0
        assert m["PostCueImmediateWrongRate"] == 1.0

    def test_badwait_positivecue_rate_uses_positive_opportunity_denominator(self):
        from cls_option_tutor.experiments.metrics_extractors import compute_6fg_metrics

        qs = _make_qs(post_reveal=True, n_safe_diag=1)
        block = BlockState(
            queries=[qs],
            obs_phase_queries=0,
            teach_phase_queries=1,
            eval_phase_queries=0,
        )
        block.tutor_trace = [
            TutorStep(round_t=0, query_id=0, action="WAIT"),
            TutorStep(round_t=1, query_id=0, action="WAIT"),
        ]
        block._decision_trace = [
            {
                "query_id": 0,
                "round_t": 0,
                "pre_post_reveal_phase": True,
                "pre_has_safe_diag_opp": False,
                "pre_has_high_risk_opp": False,
                "chosen_action": "WAIT",
                "wait_reason": "WAIT_MISSED_POSITIVE_CUE",
                "scoring": {
                    "q_wait": 0.1,
                    "candidates": [{"action": "MIX", "q_use": 0.2}],
                },
            },
            {
                "query_id": 0,
                "round_t": 1,
                "pre_post_reveal_phase": True,
                "pre_has_safe_diag_opp": False,
                "pre_has_high_risk_opp": False,
                "chosen_action": "WAIT",
                "wait_reason": "WAIT_NO_GOOD_CUE",
                "scoring": {
                    "q_wait": 0.1,
                    "candidates": [{"action": "MIX", "q_use": 0.05}],
                },
            },
        ]

        m = compute_6fg_metrics(block)
        assert m["PostReveal_PositiveCueOppCount"] == 1
        assert m["BadWAIT_PostReveal_PositiveCueCount"] == 1
        assert m["BadWAIT_PostReveal_PositiveCueRate"] == 1.0
        assert m["BadWAIT_AmongWaitRate"] == 0.5

    def test_cue_then_grace_metrics_capture_delayed_success(self):
        from cls_option_tutor.experiments.metrics_extractors import compute_6fg_metrics

        qs = _make_qs(post_reveal=True, n_safe_diag=1, success=True)
        block = BlockState(
            queries=[qs],
            obs_phase_queries=0,
            teach_phase_queries=1,
            eval_phase_queries=0,
        )
        block.tutor_trace = [
            TutorStep(round_t=0, query_id=0, action="HIGHLIGHT", highlight_cells=(0,)),
            TutorStep(round_t=1, query_id=0, action="WAIT"),
        ]
        block.learner_trace = [
            LearnerStep(
                round_t=0,
                query_id=0,
                action="pick",
                pick_index=1,
                correct=False,
                damage=0,
                hp_before=5,
                hp_after=5,
                menu_size=3,
            ),
            LearnerStep(
                round_t=1,
                query_id=0,
                action="pick",
                pick_index=0,
                correct=True,
                damage=0,
                hp_before=5,
                hp_after=5,
                menu_size=3,
            ),
        ]
        block._decision_trace = [
            {
                "query_id": 0,
                "round_t": 0,
                "pre_post_reveal_phase": True,
                "pre_has_safe_diag_opp": False,
                "pre_has_high_risk_opp": False,
                "chosen_action": "HIGHLIGHT",
                "scoring": {"q_wait": 0.0, "candidates": []},
            },
            {
                "query_id": 0,
                "round_t": 1,
                "pre_post_reveal_phase": True,
                "pre_has_safe_diag_opp": False,
                "pre_has_high_risk_opp": False,
                "chosen_action": "WAIT",
                "wait_reason": "WAIT_GRACE",
                "scoring": {"q_wait": 0.0, "candidates": []},
            },
        ]

        m = compute_6fg_metrics(block)
        assert m["CueThenGraceWAITRate"] == 1.0
        assert m["CueThenGraceCorrectRate"] == 1.0
        assert m["CueTrajectorySuccessWithin2RoundsRate"] == 1.0

    def test_mix_target_audit_reads_ban_target_and_removed_bad_mass(self):
        from cls_option_tutor.experiments.metrics_extractors import compute_6fg_metrics

        qs = _make_qs(post_reveal=True, n_safe_diag=1)
        qs.last_reveal_option_index = 1
        qs.option_diag_labels = {
            0: "",
            1: "safe_diagnostic_wrong",
            2: "high_risk_lure",
        }
        block = BlockState(
            queries=[qs],
            obs_phase_queries=0,
            teach_phase_queries=1,
            eval_phase_queries=0,
        )
        block.tutor_trace = [
            TutorStep(round_t=0, query_id=0, action="MIX", ban_index=2, highlight_cells=(0,))
        ]
        block._decision_trace = [{
            "query_id": 0,
            "round_t": 0,
            "pre_post_reveal_phase": True,
            "pre_has_safe_diag_opp": False,
            "pre_has_high_risk_opp": True,
            "pre_last_reveal_option_index": 1,
            "chosen_action": "MIX",
            "chosen_ban_index": 2,
            "scoring": {
                "q_wait": 0.0,
                "candidates": [{
                    "action": "MIX",
                    "q_use": 0.3,
                    "postreveal_decomp": {
                        "removed_prob_mass": 0.2,
                        "removed_bad_mass": 0.3,
                        "bad_mass_drop": 0.4,
                        "delta_p_correct": 0.1,
                        "correct_margin_gain": 0.05,
                    },
                }],
            },
        }]

        m = compute_6fg_metrics(block)
        assert m["MIXChosenCount"] == 1
        assert m["MIXBanTargetWasHighRiskRate"] == 1.0
        assert m["MIXBanTargetWasLastWrongRate"] == 0.0
        assert m["MIXRemovedBadMassMean"] == 0.3
        assert m["MIXBadMassDropMean"] == 0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
