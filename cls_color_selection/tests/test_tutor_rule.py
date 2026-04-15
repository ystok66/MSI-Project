"""
test_tutor_rule.py — Tests for Rule Tutor (T0).

Covers:
  - Warning correctness (truthful)
  - Hint timing (only after confirm fail)
  - Courage trigger conditions
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..')))

from cls_color_selection.config import TutorConfig
from cls_color_selection.constants import TutorActionType
from cls_color_selection.interfaces import CandidateBall
from cls_color_selection.environment.state import QueryState
from cls_color_selection.tutor_api.tutor_state import TutorBelief
from cls_color_selection.tutor_api.tutor_rule import RuleTutor


def _make_ball(idx, color='RED', is_danger=False, dim=5):
    return CandidateBall(
        index=idx, color=color,
        danger_vec=np.zeros(dim), observed_vec=np.zeros(dim),
        is_danger=is_danger, danger_type=1 if is_danger else 0,
    )


def _make_state(target=None, gt=None, completion=None, n_confirm_max=5):
    target = target or ['RED', 'BLUE']
    gt = gt or list(target)
    completion = completion or [None] * len(target)
    pool = [_make_ball(i) for i in range(4)]
    return QueryState(
        query_id=0, query_words=['dax'],
        target_output=target, ground_truth=gt,
        grammar_colors=['RED', 'BLUE', 'GREEN'],
        completion=completion, candidate_pool=pool,
        n_confirm_max=n_confirm_max,
    )


class TestRuleTutorWarning:
    def test_warns_on_danger(self):
        cfg = TutorConfig()
        tutor = RuleTutor(cfg)
        state = _make_state()
        selected = [_make_ball(0, is_danger=True)]
        action = tutor.on_select(state, selected)
        assert action.action_type == TutorActionType.WARNING

    def test_no_warn_on_safe(self):
        cfg = TutorConfig()
        tutor = RuleTutor(cfg)
        state = _make_state()
        selected = [_make_ball(0, is_danger=False)]
        action = tutor.on_select(state, selected)
        assert action.action_type == TutorActionType.WAIT

    def test_warning_always_truthful(self):
        """Warning only fires when there IS danger."""
        cfg = TutorConfig()
        tutor = RuleTutor(cfg)
        state = _make_state()
        for _ in range(20):
            safe_balls = [_make_ball(i) for i in range(3)]
            action = tutor.on_select(state, safe_balls)
            assert action.action_type == TutorActionType.WAIT


class TestRuleTutorHint:
    def test_hint_after_confirm_fail_timeout_risk(self):
        """Hint should fire when timeout risk is high."""
        cfg = TutorConfig(hint_after_confirm_fail=True)
        belief = TutorBelief()
        belief.sem.success_rate.alpha = 1.0
        belief.sem.success_rate.beta = 10.0  # low grammar → high timeout risk
        tutor = RuleTutor(cfg, belief=belief)

        state = _make_state(n_confirm_max=3)
        state.confirm_count = 2  # only 1 confirm left → high risk

        feedback = {'mode': 'wrong_only', 'mask': [False, False]}
        action = tutor.on_confirm_fail(state, feedback)
        assert action.action_type == TutorActionType.HINT

    def test_no_hint_when_plenty_confirms(self):
        """No hint when many confirms remain."""
        cfg = TutorConfig(hint_after_confirm_fail=True)
        belief = TutorBelief()
        tutor = RuleTutor(cfg, belief=belief)

        state = _make_state(n_confirm_max=5)
        state.confirm_count = 0

        feedback = {'mode': 'wrong_only', 'mask': [False]}
        action = tutor.on_confirm_fail(state, feedback)
        assert action.action_type == TutorActionType.WAIT

    def test_hint_disabled(self):
        """No hint when hint_after_confirm_fail=False."""
        cfg = TutorConfig(hint_after_confirm_fail=False)
        tutor = RuleTutor(cfg)

        state = _make_state(n_confirm_max=2)
        state.confirm_count = 1

        feedback = {}
        action = tutor.on_confirm_fail(state, feedback)
        assert action.action_type == TutorActionType.WAIT


class TestRuleTutorCourage:
    def test_courage_when_stuck(self):
        cfg = TutorConfig(n_retry_courage=3)
        tutor = RuleTutor(cfg)

        state = _make_state()
        state.consecutive_retries = 5
        # Add safe needed ball to pool
        state.candidate_pool = [_make_ball(0, color='RED', is_danger=False)]

        action = tutor.on_courage_check(state)
        assert action.action_type == TutorActionType.COURAGE

    def test_no_courage_when_not_stuck(self):
        cfg = TutorConfig(n_retry_courage=5)
        tutor = RuleTutor(cfg)

        state = _make_state()
        state.consecutive_retries = 2

        action = tutor.on_courage_check(state)
        assert action.action_type == TutorActionType.WAIT


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
