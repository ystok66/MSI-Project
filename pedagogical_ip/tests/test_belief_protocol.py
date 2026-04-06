"""
Tests for CellBelief protocol — post-cleanup.

Verifies:
1. BeliefMap and FeatureBeliefMap satisfy CellBelief protocol
2. Protocol methods exist and are callable
3. copy() and reset() have correct behavior
"""

import numpy as np
import pytest

from src.agents.belief_protocol import CellBelief
from src.agents.belief import BeliefMap
from src.agents.feature_belief import FeatureBeliefMap
from src.envs.lattice_v2 import FEATURE_DIM


def test_belief_map_satisfies_protocol():
    """BeliefMap satisfies the CellBelief protocol."""
    bm = BeliefMap.from_prior(8, 8)
    assert isinstance(bm, CellBelief)


def test_feature_belief_satisfies_protocol():
    """FeatureBeliefMap satisfies the CellBelief protocol."""
    fb = FeatureBeliefMap(7, 15, d=FEATURE_DIM)
    assert isinstance(fb, CellBelief)


def test_protocol_methods_exist():
    """All CellBelief protocol methods are callable on both types."""
    for belief in [BeliefMap.from_prior(5, 5), FeatureBeliefMap(5, 5)]:
        # H, W exist
        assert belief.H == 5
        assert belief.W == 5

        # get_belief returns (mean, var) tuple of arrays
        mean, var = belief.get_belief(2, 2)
        assert isinstance(mean, np.ndarray)
        assert isinstance(var, np.ndarray)
        assert mean.shape == var.shape

        # copy returns a new instance
        belief_copy = belief.copy()
        assert belief_copy is not belief

        # reset is callable
        belief.reset()


def test_protocol_copy_reset_behavior():
    """copy() is independent; reset() restores prior values."""
    bm = BeliefMap.from_prior(4, 4, prior_risk_mean=0.1, prior_risk_var=0.25)
    # Modify one cell
    bm.risk_mean[2, 2] = 0.9
    bm.cost_mean[1, 1] = 10.0

    # Copy should be independent
    bm2 = bm.copy()
    bm2.risk_mean[2, 2] = 0.0
    assert bm.risk_mean[2, 2] == 0.9, "Original should be unaffected"

    # Reset should restore priors
    bm.reset()
    assert bm.risk_mean[2, 2] == pytest.approx(0.1), f"Should reset to prior: {bm.risk_mean[2,2]}"
    assert bm.cost_mean[1, 1] == pytest.approx(1.5), f"Should reset to prior: {bm.cost_mean[1,1]}"
    assert not bm.visited_mask.any(), "Visited mask should be cleared"

    # FeatureBeliefMap reset
    fb = FeatureBeliefMap(4, 4, prior_mean=0.5, prior_var=0.25)
    fb.mean[1, 1] = [0.9, 0.1, 0.8, 0.7]
    fb.reset()
    assert fb.mean[1, 1, 0] == pytest.approx(0.5), f"Should reset to prior: {fb.mean[1,1,0]}"
