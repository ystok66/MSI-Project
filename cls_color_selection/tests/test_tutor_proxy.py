"""
test_tutor_proxy.py — Tests for Proxy Utility Tutor (T1).

Covers:
  - Warning always wins over WAIT when danger present
  - Hint utility correctly trades off over-help vs timeout reduction
  - Best action selection works
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..')))

from cls_color_selection.config import TutorConfig
from cls_color_selection.constants import TutorActionType
from cls_color_selection.interfaces import TutorAction, CandidateBall
from cls_color_selection.environment.state import QueryState
from cls_color_selection.tutor_api.tutor_state import TutorBelief
from cls_color_selection.tutor_api.tutor_proxy import ProxyTutor
from cls_color_selection.tutor_api.utility import compute_action_utility


def _make_ball(idx, color='RED', is_danger=False, dim=5):
    return CandidateBall(
        index=idx, color=color,
        danger_vec=np.zeros(dim), observed_vec=np.zeros(dim),
        is_danger=is_danger, danger_type=1 if is_danger else 0,
    )


def _make_state(target=None, n_confirm_max=5):
    target = target or ['RED', 'BLUE']
    pool = [_make_ball(i) for i in range(4)]
    return QueryState(
        query_id=0, query_words=['dax'],
        target_output=target, ground_truth=target,
        grammar_colors=['RED', 'BLUE', 'GREEN'],
        completion=[None] * len(target), candidate_pool=pool,
        n_confirm_max=n_confirm_max,
    )


class TestProxyTutorWarning:
    def test_always_warns_on_danger(self):
        """Proxy tutor must always warn when danger present."""
        cfg = TutorConfig()
        belief = TutorBelief()
        tutor = ProxyTutor(cfg, belief=belief)

        state = _make_state()
        selected = [_make_ball(0, is_danger=True)]
        action = tutor.on_select(state, selected)
        assert action.action_type == TutorActionType.WARNING

    def test_no_warn_on_safe(self):
        cfg = TutorConfig()
        belief = TutorBelief()
        tutor = ProxyTutor(cfg, belief=belief)

        state = _make_state()
        selected = [_make_ball(0)]
        action = tutor.on_select(state, selected)
        assert action.action_type != TutorActionType.WARNING


class TestProxyTutorHint:
    def test_hint_utility_positive_when_near_timeout(self):
        """Hint should have positive utility when timeout is imminent."""
        cfg = TutorConfig(lambda_teach=2.0, lambda_over=0.5)
        belief = TutorBelief()
        belief.sem.success_rate.beta = 10.0  # low grammar → high timeout risk

        state = _make_state(n_confirm_max=3)
        state.confirm_count = 2  # 1 confirm left

        hint_1 = TutorAction(
            action_type=TutorActionType.HINT,
            hint_positions=[(0, 'RED')],
        )
        u_hint = compute_action_utility(
            hint_1, state, belief, None, cfg, 'post_confirm')
        u_wait = compute_action_utility(
            TutorAction(), state, belief, None, cfg, 'post_confirm')
        assert u_hint > u_wait, f"Hint should beat WAIT: {u_hint:.3f} vs {u_wait:.3f}"

    def test_hint_costly_when_learner_competent(self):
        """Over-help penalty should make hints expensive for strong learners."""
        cfg = TutorConfig(lambda_over=3.0, lambda_teach=0.5)
        belief = TutorBelief()
        belief.sem.success_rate.alpha = 20.0  # high grammar
        belief.sem.success_rate.beta = 1.0

        state = _make_state()

        hint_1 = TutorAction(
            action_type=TutorActionType.HINT,
            hint_positions=[(0, 'RED')],
        )
        u_hint = compute_action_utility(
            hint_1, state, belief, None, cfg, 'post_confirm')
        # With high competence and heavy over-help penalty, hint should be costly
        u_wait = compute_action_utility(
            TutorAction(), state, belief, None, cfg, 'post_confirm')
        assert u_hint < u_wait, \
            f"Hint should be costly for competent learner: {u_hint:.3f} vs {u_wait:.3f}"


class TestProxyTutorPostConfirm:
    def test_returns_valid_action(self):
        cfg = TutorConfig()
        belief = TutorBelief()
        tutor = ProxyTutor(cfg, belief=belief)

        state = _make_state()
        feedback = {'mode': 'wrong_only', 'mask': [False]}
        action = tutor.on_confirm_fail(state, feedback)
        assert action.action_type in (TutorActionType.WAIT, TutorActionType.HINT)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
