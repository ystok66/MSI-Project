"""
test_shadow_clone.py — Tests for shadow snapshot and clone.
"""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))

from cls_color_selection.tutor_api.shadow_snapshot import (
    ShadowLearnerSnapshot, ConceptSnapshot, RiskSnapshot, PolicySnapshot,
)


class TestConceptSnapshot:
    def test_clone_independence(self):
        c = ConceptSnapshot(
            name='dax',
            role_counts={'EMIT': 5.0, 'REPEAT': 1.0},
            emit_stats={'sum_w': 3.0, 'sum_wx': np.ones(3), 'sum_wx2': np.ones(3)},
        )
        c2 = c.clone()
        c2.role_counts['EMIT'] = 99.0
        c2.emit_stats['sum_wx'][0] = 99.0
        assert c.role_counts['EMIT'] == 5.0
        assert c.emit_stats['sum_wx'][0] == 1.0


class TestRiskSnapshot:
    def test_clone_independence(self):
        r = RiskSnapshot(
            n_danger_types=3, n_types=4, danger_dim=5,
            obs_sigma=0.3,
            type_prior=np.array([0.7, 0.1, 0.1, 0.1]),
            proto_mu=np.zeros((4, 5)),
            proto_var=np.ones((4, 5)),
            _counts=np.zeros(4),
            _sum_x=np.zeros((4, 5)),
            _sum_x2=np.zeros((4, 5)),
        )
        r2 = r.clone()
        r2.proto_mu[0, 0] = 99.0
        r2.type_prior[0] = 0.1
        assert r.proto_mu[0, 0] == 0.0
        assert r.type_prior[0] == 0.7


class TestShadowLearnerSnapshot:
    def test_clone_deep(self):
        snap = ShadowLearnerSnapshot(
            grammar={'dax': ConceptSnapshot(
                name='dax',
                role_counts={'EMIT': 3.0},
                emit_stats={'sum_w': 1.0, 'sum_wx': np.ones(3), 'sum_wx2': np.ones(3)},
            )},
            risk=RiskSnapshot(
                n_danger_types=1, n_types=2, danger_dim=3, obs_sigma=0.3,
                type_prior=np.array([0.7, 0.3]),
                proto_mu=np.zeros((2, 3)),
                proto_var=np.ones((2, 3)),
                _counts=np.zeros(2), _sum_x=np.zeros((2, 3)), _sum_x2=np.zeros((2, 3)),
            ),
            policy=PolicySnapshot(),
        )
        snap2 = snap.clone()
        snap2.grammar['dax'].role_counts['EMIT'] = 99.0
        snap2.risk.proto_mu[0, 0] = 99.0
        assert snap.grammar['dax'].role_counts['EMIT'] == 3.0
        assert snap.risk.proto_mu[0, 0] == 0.0

    def test_grammar_hash_changes(self):
        snap = ShadowLearnerSnapshot(
            grammar={'dax': ConceptSnapshot(
                name='dax',
                role_counts={'EMIT': 3.0},
                emit_stats={'sum_w': 1.0, 'sum_wx': np.ones(3), 'sum_wx2': np.ones(3)},
            )},
        )
        h1 = snap.grammar_hash()
        snap.grammar['dax'].role_counts['EMIT'] = 10.0
        h2 = snap.grammar_hash()
        assert h1 != h2

    def test_risk_hash_changes(self):
        snap = ShadowLearnerSnapshot(
            risk=RiskSnapshot(
                n_danger_types=1, n_types=2, danger_dim=3, obs_sigma=0.3,
                type_prior=np.array([0.7, 0.3]),
                proto_mu=np.zeros((2, 3)),
                proto_var=np.ones((2, 3)),
                _counts=np.zeros(2), _sum_x=np.zeros((2, 3)), _sum_x2=np.zeros((2, 3)),
            ),
        )
        h1 = snap.risk_hash()
        snap.risk.proto_mu[0, 0] = 5.0
        h2 = snap.risk_hash()
        assert h1 != h2


class TestPolicySnapshot:
    def test_clone(self):
        p = PolicySnapshot(alpha_risk=0.8, epsilon_policy=0.1)
        p2 = p.clone()
        p2.alpha_risk = 99.0
        assert p.alpha_risk == 0.8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
