"""
test_feedback_update.py — Tests for confirm feedback grammar update.

Covers:
  - wrong_only likelihood computation
  - wrong_positions likelihood computation
  - Posterior reweighting correctly shifts mass away from wrong traces
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..')))

from cls_color_selection.config import LearnerConfig
from cls_color_selection.learner.feedback_update import FeedbackUpdater


def _make_updater(**kwargs):
    cfg = LearnerConfig(**kwargs)
    return FeedbackUpdater(cfg)


class TestFeedbackLikelihood:
    def test_wrong_only_exact_match(self):
        """Exact match should get tiny likelihood (suppressed)."""
        updater = _make_updater(eps_wrong=0.01)
        # Y_hat == Y_k → likelihood should be ε
        lik = updater.compute_feedback_likelihood_wrong_only(
            ['RED', 'BLUE'], ['RED', 'BLUE'])
        assert abs(lik - 0.01) < 1e-6

    def test_wrong_only_mismatch(self):
        """Mismatch should get high likelihood."""
        updater = _make_updater(eps_wrong=0.01)
        lik = updater.compute_feedback_likelihood_wrong_only(
            ['RED', 'BLUE'], ['RED', 'GREEN'])
        assert abs(lik - 0.99) < 1e-6

    def test_wrong_positions_all_correct(self):
        """All positions correct → high likelihood for matching trace."""
        updater = _make_updater(eps_eq=0.05)
        lik = updater.compute_feedback_likelihood_wrong_positions(
            ['RED', 'BLUE'], ['RED', 'BLUE'], [True, True])
        # Should be (1-ε)^2 = 0.95^2 = 0.9025
        assert lik > 0.8

    def test_wrong_positions_all_wrong(self):
        """All positions wrong → high likelihood for non-matching trace."""
        updater = _make_updater(eps_eq=0.05)
        lik = updater.compute_feedback_likelihood_wrong_positions(
            ['RED', 'BLUE'], ['GREEN', 'GREEN'], [False, False])
        # Mismatching Y_k: s = ε = 0.05
        # P = Π(1-s) = (0.95)^2 = 0.9025
        assert lik > 0.8

    def test_wrong_positions_mixed(self):
        """Mixed positions: first correct, second wrong."""
        updater = _make_updater(eps_eq=0.05)
        # Y_hat = [RED, BLUE], Y_k = [RED, GREEN]
        # mask = [True, False]
        # Position 0: Y_k[0]==Y_hat[0]=RED, mask=True → s=0.95 → contribute s=0.95
        # Position 1: Y_k[1]=GREEN≠Y_hat[1]=BLUE, mask=False → s=0.05 → contribute (1-s)=0.95
        lik = updater.compute_feedback_likelihood_wrong_positions(
            ['RED', 'BLUE'], ['RED', 'GREEN'], [True, False])
        expected = 0.95 * 0.95
        assert abs(lik - expected) < 0.01


class TestPosteriorReweighting:
    def test_wrong_trace_suppressed(self):
        """wrong_only: trace matching Y_hat should be suppressed."""
        updater = _make_updater(eps_wrong=0.01)

        # Beam: two candidates
        beam = [
            (0.0, [], ['RED', 'BLUE']),   # matches Y_hat — should be penalized
            (0.0, [], ['RED', 'GREEN']),   # doesn't match — should be boosted
        ]
        Y_hat = ['RED', 'BLUE']
        feedback = {'mode': 'wrong_only'}

        q_old, q_new = updater.reweight_beam_posterior(beam, Y_hat, feedback)

        # After wrong_only, matching trace should have much lower weight
        assert q_new[0] < q_old[0], \
            f"Matching trace should be suppressed: {q_old[0]:.4f} → {q_new[0]:.4f}"
        assert q_new[1] > q_old[1], \
            f"Non-matching trace should be boosted: {q_old[1]:.4f} → {q_new[1]:.4f}"

    def test_probabilities_sum_to_one(self):
        """Reweighted posterior should be valid probability distribution."""
        updater = _make_updater()
        beam = [
            (np.log(0.3), [], ['RED']),
            (np.log(0.5), [], ['GREEN']),
            (np.log(0.2), [], ['BLUE']),
        ]
        q_old, q_new = updater.reweight_beam_posterior(
            beam, ['RED'], {'mode': 'wrong_only'})
        assert abs(q_new.sum() - 1.0) < 1e-6

    def test_empty_beam(self):
        """Empty beam should return empty arrays."""
        updater = _make_updater()
        q_old, q_new = updater.reweight_beam_posterior([], ['RED'], {})
        assert len(q_old) == 0
        assert len(q_new) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
