"""
test_joint_debug.py — Tests for divergence and counterfactual tracking.
"""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))

from cls_color_selection.tutor_api.joint_debug import (
    JointDebugLog, DivergenceRecord, CounterfactualRecord,
    measure_risk_divergence,
)
from cls_color_selection.tutor_api.shadow_snapshot import (
    ShadowLearnerSnapshot, RiskSnapshot,
)
from cls_color_selection.tutor_api.shadow_clone import write_shadow_to_real_risk
from cls_color_selection.learner.risk_belief import DangerTypeBelief


class TestJointDebugLog:
    def test_empty_summary(self):
        log = JointDebugLog()
        s = log.summary()
        assert s['n_divergence_records'] == 0
        assert s['n_counterfactual_records'] == 0

    def test_add_records(self):
        log = JointDebugLog()
        log.add_divergence(DivergenceRecord(step=0, top1_agreement=True))
        log.add_divergence(DivergenceRecord(step=1, top1_agreement=False))
        log.add_counterfactual(CounterfactualRecord(
            q_predicted=0.5, return_realized=0.3, error=0.2))

        s = log.summary()
        assert s['n_divergence_records'] == 2
        assert s['n_counterfactual_records'] == 1
        assert s['D_gram_top1_agreement'] == 0.5  # 1/2 agreed
        assert abs(s['CF_abs_error'] - 0.2) < 1e-6


class TestRiskDivergence:
    def test_identical_risk_zero_divergence(self):
        """Zero divergence when shadow matches real."""
        rs = RiskSnapshot(
            n_danger_types=1, n_types=2, danger_dim=3, obs_sigma=0.3,
            type_prior=np.array([0.7, 0.3]),
            proto_mu=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            proto_var=np.ones((2, 3)),
            _counts=np.zeros(2), _sum_x=np.zeros((2, 3)), _sum_x2=np.zeros((2, 3)),
        )
        snap = ShadowLearnerSnapshot(risk=rs)
        real_risk = write_shadow_to_real_risk(snap)

        test_vecs = [np.array([0.5, 0.5, 0.5])]
        l1 = measure_risk_divergence(snap, real_risk, test_vecs)
        assert l1 < 1e-6, f"Identical risk should have zero divergence, got {l1}"

    def test_different_risk_nonzero_divergence(self):
        """Nonzero divergence when shadow differs from real."""
        rs = RiskSnapshot(
            n_danger_types=1, n_types=2, danger_dim=3, obs_sigma=0.3,
            type_prior=np.array([0.7, 0.3]),
            proto_mu=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            proto_var=np.ones((2, 3)),
            _counts=np.zeros(2), _sum_x=np.zeros((2, 3)), _sum_x2=np.zeros((2, 3)),
        )
        snap = ShadowLearnerSnapshot(risk=rs)

        # Create real risk with different prototypes
        real_risk = DangerTypeBelief(n_danger_types=1, danger_dim=3, obs_sigma=0.3)
        real_risk.proto_mu = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])  # shifted
        real_risk.proto_var = np.ones((2, 3))
        real_risk.type_prior = np.array([0.7, 0.3])

        test_vecs = [np.array([0.5, 0.5, 0.5])]
        l1 = measure_risk_divergence(snap, real_risk, test_vecs)
        assert l1 > 0.01, f"Different risk should have nonzero divergence, got {l1}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
