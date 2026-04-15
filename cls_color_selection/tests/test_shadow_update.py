"""
test_shadow_update.py — Tests for shadow update consistency with real learner.
"""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))

from cls_color_selection.tutor_api.shadow_snapshot import (
    ShadowLearnerSnapshot, RiskSnapshot,
)
from cls_color_selection.tutor_api.shadow_update import (
    shadow_warning_update, shadow_safe_observation_update,
    shadow_death_update,
)
from cls_color_selection.tutor_api.shadow_clone import write_shadow_to_real_risk
from cls_color_selection.learner.risk_belief import DangerTypeBelief
from cls_color_selection.interfaces import CandidateBall


def _make_risk_snapshot(n_types=4, dim=5):
    return RiskSnapshot(
        n_danger_types=3, n_types=n_types, danger_dim=dim, obs_sigma=0.3,
        type_prior=np.array([0.7, 0.1, 0.1, 0.1]),
        proto_mu=np.random.randn(n_types, dim),
        proto_var=np.ones((n_types, dim)),
        _counts=np.zeros(n_types), _sum_x=np.zeros((n_types, dim)),
        _sum_x2=np.zeros((n_types, dim)),
    )


def _make_ball(idx, color='RED', is_danger=False, dim=5):
    return CandidateBall(
        index=idx, color=color,
        danger_vec=np.random.randn(dim),
        observed_vec=np.random.randn(dim),
        is_danger=is_danger, danger_type=1 if is_danger else 0,
    )


class TestShadowWarningUpdate:
    def test_warning_changes_risk(self):
        """Shadow warning should shift risk posteriors."""
        np.random.seed(42)
        rs = _make_risk_snapshot()
        snap = ShadowLearnerSnapshot(risk=rs)
        old_mu = snap.risk.proto_mu.copy()

        balls = [_make_ball(0, is_danger=True), _make_ball(1)]
        shadow_warning_update(snap, balls)

        # After warning, prototypes should have shifted
        assert not np.allclose(snap.risk.proto_mu, old_mu), \
            "Warning should change risk prototypes"


class TestShadowSafeUpdate:
    def test_safe_changes_proto(self):
        np.random.seed(42)
        rs = _make_risk_snapshot()
        snap = ShadowLearnerSnapshot(risk=rs)
        old_mu = snap.risk.proto_mu.copy()

        x = np.random.randn(5)
        shadow_safe_observation_update(snap, x)

        assert not np.allclose(snap.risk.proto_mu[0], old_mu[0]), \
            "Safe obs should update safe proto"


class TestShadowDeathUpdate:
    def test_death_changes_proto(self):
        np.random.seed(42)
        rs = _make_risk_snapshot()
        snap = ShadowLearnerSnapshot(risk=rs)
        old_counts = snap.risk._counts.copy()

        # Use x near danger prototype 1 so death update is clearly assigned
        x = snap.risk.proto_mu[1] + 0.1 * np.random.randn(5)
        shadow_death_update(snap, x)

        # Death should increase danger type counts
        assert snap.risk._counts[1:].sum() > old_counts[1:].sum(), \
            "Death should increase danger type counts"


class TestShadowRiskConsistency:
    def test_shadow_matches_real_warning(self):
        """Shadow warning update should match real learner warning update."""
        np.random.seed(42)
        rs = _make_risk_snapshot()

        # Create shadow
        snap = ShadowLearnerSnapshot(risk=rs.clone())

        # Create real risk belief matching same initial state
        real_risk = write_shadow_to_real_risk(snap)

        balls = [_make_ball(0, is_danger=True), _make_ball(1)]

        # Apply warning to shadow
        shadow_warning_update(snap, balls)

        # Apply warning to real
        from cls_color_selection.learner.warning_update import warning_set_bayes_update
        warning_set_bayes_update(real_risk, balls)

        # They should be nearly identical
        np.testing.assert_allclose(
            snap.risk.proto_mu, real_risk.proto_mu, atol=1e-10,
            err_msg="Shadow and real risk should match after same warning")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
