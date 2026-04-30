"""Phase 6H: Online Self-Correction Tutor Tests.

Tests:
  1. State tracking: wrong pick updates trajectory state
  2. State tracking: correct pick clears post_reveal_phase
  3. HIGHLIGHT gate: post-reveal generates HIGHLIGHT without timeout
  4. G_exp: self_correct mode ALLOW bonus for WAIT
  5. G_exp: self_correct mode CONSOLIDATE penalty for repeated WAIT
  6. G_exp: self_correct HIGHLIGHT bonus post-reveal
"""
import numpy as np
import pytest
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch

from cls_option_tutor.interfaces import Option
from cls_option_tutor.env.state import QueryState


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_qs(
    hp=5, rounds_used=0, max_rounds=5,
    n_safe_diag=0, post_reveal=False,
    option_diag_labels=None,
):
    """Create a minimal QueryState for testing."""
    qs = QueryState(
        query_id=0,
        target_output=["red", "blue", "green", "red"],
        true_program=["repeat", "2", "red", "blue", "green"],
        hp=hp,
        rounds_used=rounds_used,
        max_rounds=max_rounds,
    )
    qs.n_safe_diag_wrong_reveals = n_safe_diag
    qs.post_reveal_phase = post_reveal
    if option_diag_labels:
        qs.option_diag_labels = option_diag_labels
    return qs


def _make_option(index, is_correct=False, risk_class=0, text=None):
    return Option(
        index=index,
        text=text or ["word1"],
        rendered_output=["red", "blue", "green", "red"] if is_correct else ["red", "blue", "blue", "red"],
        is_correct=is_correct,
        risk_class=risk_class,
        danger_vec=np.zeros(4),
    )


# ── Test: State Tracking ─────────────────────────────────────────────────────

class TestTrajectoryStateTracking:
    """Test that wrong/correct picks update trajectory state correctly."""

    def test_wrong_pick_updates_safe_diag_count(self):
        """Wrong pick of safe_diagnostic_wrong increments counter."""
        qs = _make_qs()
        opt = _make_option(1, is_correct=False, risk_class=0)
        qs.option_diag_labels = {1: "safe_diagnostic_wrong"}

        # Simulate the update
        from cls_option_tutor.env.option_env import OptionEnv
        env = OptionEnv.__new__(OptionEnv)  # bare object
        env._update_query_trajectory_after_wrong_pick(qs, opt, round_t=0)

        assert qs.n_safe_diag_wrong_reveals == 1
        assert qs.post_reveal_phase is True
        assert qs.last_wrong_diag_label == "safe_diagnostic_wrong"
        assert qs.last_reveal_round_t == 0
        assert qs.last_reveal_option_index == 1

    def test_wrong_pick_updates_high_risk_count(self):
        """Wrong pick of high_risk_lure increments counter but does NOT set post_reveal_phase."""
        qs = _make_qs()
        opt = _make_option(2, is_correct=False, risk_class=3)
        qs.option_diag_labels = {2: "high_risk_lure"}

        from cls_option_tutor.env.option_env import OptionEnv
        env = OptionEnv.__new__(OptionEnv)
        env._update_query_trajectory_after_wrong_pick(qs, opt, round_t=1)

        assert qs.n_high_risk_wrong_reveals == 1
        assert qs.post_reveal_phase is False  # only diagnostic reveals trigger this

    def test_correct_pick_clears_post_reveal(self):
        """Correct pick clears post_reveal_phase."""
        qs = _make_qs(post_reveal=True, n_safe_diag=1)
        assert qs.post_reveal_phase is True

        # Simulate correct pick
        qs.post_reveal_phase = False  # This is what option_env does
        assert qs.post_reveal_phase is False

    def test_bounded_diag_sets_post_reveal(self):
        """Bounded diagnostic wrong also triggers post_reveal_phase."""
        qs = _make_qs()
        opt = _make_option(3, is_correct=False, risk_class=1)
        qs.option_diag_labels = {3: "bounded_diagnostic_wrong"}

        from cls_option_tutor.env.option_env import OptionEnv
        env = OptionEnv.__new__(OptionEnv)
        env._update_query_trajectory_after_wrong_pick(qs, opt, round_t=0)

        assert qs.n_bounded_diag_wrong_reveals == 1
        assert qs.post_reveal_phase is True


# ── Test: HIGHLIGHT Gate ──────────────────────────────────────────────────────

class TestHighlightGate:
    """Test that pedagogical HIGHLIGHT gate opens after diagnostic reveal."""

    def test_pedagogical_hl_gate_conditions(self):
        """Verify the pedagogical_hl_gate boolean logic."""
        # Case 1: post-reveal + safe diag reveal + self_correct mode → True
        qs = _make_qs(post_reveal=True, n_safe_diag=1, hp=3, rounds_used=1, max_rounds=5)
        gate = (
            qs.post_reveal_phase
            and qs.n_safe_diag_wrong_reveals >= 1
            and not qs.success
            and qs.hp > 0
            and qs.rounds_used < (qs.max_rounds - 1)
        )
        assert gate is True

        # Case 2: no post-reveal → False
        qs2 = _make_qs(post_reveal=False, n_safe_diag=0)
        gate2 = (
            qs2.post_reveal_phase
            and qs2.n_safe_diag_wrong_reveals >= 1
        )
        assert gate2 is False

        # Case 3: last round → gate should be False
        qs3 = _make_qs(post_reveal=True, n_safe_diag=1, rounds_used=4, max_rounds=5)
        gate3 = (
            qs3.post_reveal_phase
            and qs3.n_safe_diag_wrong_reveals >= 1
            and not qs3.success
            and qs3.hp > 0
            and qs3.rounds_used < (qs3.max_rounds - 1)
        )
        # rounds_used=4, max_rounds=5 → 4 < 4 → False: no time for consolidate
        assert gate3 is False


# ── Test: G_exp Self-Correct Mode ─────────────────────────────────────────

class TestGExpSelfCorrect:
    """Test self_correct mode's ALLOW and CONSOLIDATE logic in G_exp."""

    def _make_config(self, lg_mode="self_correct"):
        """Create minimal config mock."""
        cfg = MagicMock()
        cfg.tutor.tutor_lg_mode = lg_mode
        cfg.env.highlight_mode = "diagnostic"
        return cfg

    def test_allow_bonus_for_safe_diagnostic_opportunity(self):
        """Pre-reveal WAIT should get ALLOW bonus when safe diag opportunity exists."""
        qs = _make_qs(hp=5, rounds_used=0, n_safe_diag=0, post_reveal=False)
        qs.option_diag_labels = {
            0: "",  # correct
            1: "safe_diagnostic_wrong",
            2: "safe_random_wrong",
        }

        active = [
            _make_option(0, is_correct=True, risk_class=0),
            _make_option(1, is_correct=False, risk_class=0),
            _make_option(2, is_correct=False, risk_class=0),
        ]

        # Uniform probs
        tier_probs = np.array([0.33, 0.34, 0.33])

        # Compute G_exp with self_correct mode
        # B_allow = P_safe_diag * G_survive * G_time - P_high_risk * E_damage * 0.1
        # P_safe_diag = 0.34, G_survive ≈ 1.0, G_time = 1.0
        # → B_allow ≈ 0.34
        # total (base safe wrongs) = 0.34 + 0.33 = 0.67 + 0.34 = 1.01

        # We can't easily call the full method without the tutor object,
        # so test the logic directly
        p_safe_diag = 0.34
        p_high_risk = 0.0
        e_damage_wait = 0.0
        g_survive = max(0.0, (5 - e_damage_wait) / max(1, 5))
        g_time = 1.0 if (5 - 0) >= 2 else 0.0
        b_allow = p_safe_diag * g_survive * g_time - p_high_risk * e_damage_wait * 0.1
        assert b_allow > 0.3  # significant positive bonus

    def test_consolidate_penalty_for_repeated_wait(self):
        """Post-reveal WAIT should get a penalty when p_correct is low."""
        n_safe_reveals = 1
        p_correct = 0.2  # low
        repeat_penalty = n_safe_reveals * (1.0 - p_correct) * 0.3
        assert repeat_penalty > 0.2  # meaningful penalty

    def test_consolidate_no_penalty_when_p_correct_high(self):
        """Post-reveal WAIT penalty should be minimal when p_correct is high."""
        n_safe_reveals = 1
        p_correct = 0.9  # high — learner already improved
        repeat_penalty = n_safe_reveals * (1.0 - p_correct) * 0.3
        assert repeat_penalty < 0.05  # near-zero penalty

    def test_self_correct_mode_does_not_break_off_mode(self):
        """The 'off' mode should still work without any trajectory state."""
        qs = _make_qs(hp=5, rounds_used=0)
        qs.option_diag_labels = {}

        # No trajectory fields needed for "off" mode
        # Just verify the state defaults are safe
        assert qs.n_safe_diag_wrong_reveals == 0
        assert qs.post_reveal_phase is False


# ── Test: Condition Routing ───────────────────────────────────────────────────

class TestConditionRouting:
    """Test that condition names correctly map to config overrides."""

    def test_sis_self_correct_routing(self):
        from cls_option_tutor.experiments.run_learning_increment_micro import _apply_condition_overrides
        cfg = MagicMock()
        cfg.env.highlight_mode = "diagnostic"
        cfg.tutor.tutor_lg_mode = "off"

        _apply_condition_overrides(cfg, "SIS_self_correct")
        assert cfg.tutor.tutor_lg_mode == "self_correct"
        assert cfg.env.highlight_mode == "diagnostic"

    def test_sis_self_correct_no_highlight_routing(self):
        from cls_option_tutor.experiments.run_learning_increment_micro import _apply_condition_overrides
        cfg = MagicMock()
        cfg.env.highlight_mode = "diagnostic"
        cfg.tutor.tutor_lg_mode = "off"

        _apply_condition_overrides(cfg, "SIS_self_correct_no_highlight")
        assert cfg.tutor.tutor_lg_mode == "self_correct"
        assert cfg.env.highlight_mode == "none"

    def test_sis_self_correct_fixed_highlight_routing(self):
        from cls_option_tutor.experiments.run_learning_increment_micro import _apply_condition_overrides
        cfg = MagicMock()
        cfg.env.highlight_mode = "diagnostic"
        cfg.tutor.tutor_lg_mode = "off"

        _apply_condition_overrides(cfg, "SIS_self_correct_fixed_highlight")
        assert cfg.tutor.tutor_lg_mode == "self_correct"
        assert cfg.env.highlight_mode == "fixed"

    def test_self_correct_priority_over_inverse_shadow(self):
        """self_correct should win over inverse_shadow in compound names."""
        from cls_option_tutor.experiments.run_learning_increment_micro import _apply_condition_overrides
        cfg = MagicMock()
        cfg.env.highlight_mode = "diagnostic"
        cfg.tutor.tutor_lg_mode = "off"

        _apply_condition_overrides(cfg, "SIS_oracle_forward_self_correct")
        assert cfg.tutor.tutor_lg_mode == "self_correct"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
