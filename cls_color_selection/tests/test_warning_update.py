"""
test_warning_update.py — Tests for set-conditional Bayesian warning update.

Covers:
  - Basic posterior shift after warning
  - Edge cases: all safe, all danger, single ball
  - Numerical stability
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..')))

from cls_color_selection.learner.risk_belief import DangerTypeBelief
from cls_color_selection.learner.warning_update import warning_set_bayes_update
from cls_color_selection.interfaces import CandidateBall


def _make_belief(n_danger=2, dim=5, obs_sigma=0.1):
    belief = DangerTypeBelief(
        n_danger_types=n_danger,
        danger_dim=dim,
        obs_sigma=obs_sigma,
        prior_safe=0.7,
    )
    # Set easy-to-distinguish prototypes
    protos = np.zeros((1 + n_danger, dim))
    protos[0] = np.array([1, 0, 0, 0, 0])  # safe
    if n_danger >= 1:
        protos[1] = np.array([-1, 0, 0, 0, 0])  # danger-1
    if n_danger >= 2:
        protos[2] = np.array([0, -1, 0, 0, 0])  # danger-2
    belief.set_prototypes(protos)
    return belief


def _make_ball(idx, vec, color='RED', is_danger=False, danger_type=0):
    return CandidateBall(
        index=idx, color=color,
        danger_vec=vec, observed_vec=vec,
        is_danger=is_danger, danger_type=danger_type,
    )


class TestWarningUpdate:
    def test_warning_increases_danger_posterior(self):
        """After warning, P(danger) should increase for all balls."""
        belief = _make_belief()
        # Mix of safe-like and danger-like balls
        balls = [
            _make_ball(0, np.array([0.9, 0, 0, 0, 0])),  # safe-like
            _make_ball(1, np.array([-0.8, 0, 0, 0, 0])),  # danger-like
        ]

        before = belief.batch_posterior(
            np.stack([b.observed_vec for b in balls]))
        updated = warning_set_bayes_update(belief, balls)

        # P(safe) should decrease for safe-like ball after warning
        assert updated[0, 0] < before[0, 0], \
            f"Safe posterior should decrease: {before[0,0]:.4f} → {updated[0,0]:.4f}"

    def test_single_ball_warning(self):
        """With single ball, warning means it's definitely danger."""
        belief = _make_belief()
        balls = [
            _make_ball(0, np.array([0.5, 0, 0, 0, 0])),  # ambiguous
        ]
        updated = warning_set_bayes_update(belief, balls)
        # With one ball, P(safe | warning) should be 0
        assert updated[0, 0] < 0.01, \
            f"Single ball warning: P(safe) should be ~0, got {updated[0,0]:.4f}"

    def test_probabilities_sum_to_one(self):
        """Updated posteriors should be valid probability distributions."""
        belief = _make_belief()
        balls = [
            _make_ball(0, np.random.randn(5)),
            _make_ball(1, np.random.randn(5)),
            _make_ball(2, np.random.randn(5)),
        ]
        updated = warning_set_bayes_update(belief, balls)
        for i in range(len(balls)):
            assert abs(updated[i].sum() - 1.0) < 1e-6, \
                f"Row {i} sums to {updated[i].sum()}"

    def test_no_nan_no_inf(self):
        """No NaN or Inf in output, even with extreme inputs."""
        belief = _make_belief()
        balls = [
            _make_ball(0, np.ones(5) * 100),  # extreme
            _make_ball(1, np.ones(5) * -100),  # extreme
        ]
        updated = warning_set_bayes_update(belief, balls)
        assert not np.any(np.isnan(updated)), "NaN in output"
        assert not np.any(np.isinf(updated)), "Inf in output"


class TestDangerTypeBelief:
    def test_safe_ball_classified_safe(self):
        belief = _make_belief()
        x = np.array([1.0, 0, 0, 0, 0])  # matches safe prototype
        post = belief.single_ball_posterior(x)
        assert post[0] > 0.5, f"Safe ball should be classified safe, got P(safe)={post[0]:.3f}"

    def test_danger_ball_classified_danger(self):
        belief = _make_belief()
        x = np.array([-1.0, 0, 0, 0, 0])  # matches danger-1 prototype
        post = belief.single_ball_posterior(x)
        assert post[0] < 0.5, f"Danger ball should be classified danger, got P(safe)={post[0]:.3f}"

    def test_set_danger_probability(self):
        belief = _make_belief()
        # Two safe-like balls
        X = np.array([[1.0, 0, 0, 0, 0], [0.8, 0.1, 0, 0, 0]])
        p = belief.set_danger_probability(X)
        assert p < 0.5, f"Two safe balls: P(∃danger) should be low, got {p:.3f}"

        # One danger ball
        X2 = np.array([[-1.0, 0, 0, 0, 0]])
        p2 = belief.set_danger_probability(X2)
        assert p2 > 0.5, f"One danger ball: P(∃danger) should be high, got {p2:.3f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
