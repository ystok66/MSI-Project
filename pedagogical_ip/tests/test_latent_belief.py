"""
Tests for feature belief in latent context — Phase 4.

Verifies:
1. Feature belief exposes latent belief
2. Latent update changes belief
3. Copy/reset still work
4. Visited cells get stronger supervision
5. CellBelief protocol still satisfied
"""

import numpy as np
import pytest

from src.agents.feature_belief import FeatureBeliefMap
from src.agents.belief_protocol import CellBelief


def test_feature_belief_exposes_latent_belief():
    """Each cell has mu_z and Sigma_z accessible."""
    fb = FeatureBeliefMap(5, 5, d=4)
    mean, var = fb.get_belief(2, 2)
    assert mean.shape == (4,)
    assert var.shape == (4,)
    assert np.allclose(mean, 0.5), "Prior mean should be 0.5"


def test_latent_update_changes_belief():
    """Observation updates the latent belief."""
    fb = FeatureBeliefMap(5, 5, d=4)
    mean_before, var_before = fb.get_belief(2, 2)
    fb.update(2, 2, np.array([0.1, 0.0, 0.9, 0.8]), 0.01)
    mean_after, var_after = fb.get_belief(2, 2)
    assert not np.allclose(mean_before, mean_after), "Mean should change"
    assert np.all(var_after < var_before), "Variance should decrease"


def test_latent_copy_reset_still_work():
    """Copy and reset preserve protocol under latent usage."""
    fb = FeatureBeliefMap(4, 4, d=4)
    fb.update(1, 1, np.array([0.9, 0.0, 0.1, 0.1]), 0.01)

    fb2 = fb.copy()
    fb2.update(1, 1, np.zeros(4), 0.01)
    mean_orig, _ = fb.get_belief(1, 1)
    mean_copy, _ = fb2.get_belief(1, 1)
    assert not np.allclose(mean_orig, mean_copy), "Copy should be independent"

    fb.reset()
    mean_reset, var_reset = fb.get_belief(1, 1)
    assert np.allclose(mean_reset, 0.5), "Should reset to prior"
    assert np.allclose(var_reset, 0.25), "Should reset variance"


def test_visited_cell_stronger_supervision():
    """Self-observation (low noise) updates more than neighbor (high noise)."""
    fb = FeatureBeliefMap(5, 5, d=4)
    z_true = np.array([0.0, 0.0, 0.9, 0.8])

    # Self-observation: very low noise
    fb.update(2, 2, z_true, 0.01)
    mean_self, var_self = fb.get_belief(2, 2)

    # Neighbor observation: higher noise (same true z)
    fb2 = FeatureBeliefMap(5, 5, d=4)
    fb2.update(2, 2, z_true, 0.08)
    mean_nbr, var_nbr = fb2.get_belief(2, 2)

    assert np.all(var_self < var_nbr), "Self-obs should give lower variance"


def test_belief_protocol_still_satisfied():
    """FeatureBeliefMap still satisfies CellBelief protocol."""
    fb = FeatureBeliefMap(5, 5, d=4)
    assert isinstance(fb, CellBelief)
