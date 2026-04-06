"""Tests for belief map and Bayesian update."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.agents.belief import (
    BeliefMap,
    bayesian_update,
    update_belief_cell,
    apply_warning_to_belief,
)


def test_bayesian_update_moves_toward_observation():
    mean, var = bayesian_update(
        prior_mean=1.0, prior_var=4.0,
        obs=5.0, obs_var=1.0,
    )
    # Posterior mean should be between prior and observation
    assert 1.0 < mean < 5.0
    # Posterior variance should be smaller than both
    assert var < 4.0
    assert var < 1.0


def test_bayesian_update_low_noise_follows_obs():
    mean, var = bayesian_update(
        prior_mean=1.0, prior_var=4.0,
        obs=3.0, obs_var=0.01,
    )
    # Very low obs noise → posterior close to obs
    assert abs(mean - 3.0) < 0.1
    assert var < 0.02


def test_belief_from_prior():
    b = BeliefMap.from_prior(8, 8, 1.5, 4.0, 0.1, 0.25)
    assert b.cost_mean.shape == (8, 8)
    assert np.allclose(b.cost_mean, 1.5)
    assert np.allclose(b.cost_var, 4.0)
    assert np.allclose(b.risk_mean, 0.1)
    assert not b.visited_mask.any()


def test_update_belief_cell_converges():
    b = BeliefMap.from_prior(8, 8, 1.5, 4.0, 0.1, 0.25)
    # Simulate multiple precise observations at (2,3)
    true_cost, true_risk = 3.0, 0.5
    for _ in range(10):
        update_belief_cell(b, 2, 3, true_cost, true_risk, 0.01, 0.01)

    assert abs(b.cost_mean[2, 3] - true_cost) < 0.1
    assert abs(b.risk_mean[2, 3] - true_risk) < 0.1
    assert b.cost_var[2, 3] < 0.01


def test_unvisited_cells_maintain_prior():
    b = BeliefMap.from_prior(8, 8, 1.5, 4.0, 0.1, 0.25)
    # Update only (0,0)
    update_belief_cell(b, 0, 0, 5.0, 0.5, 0.01, 0.01)
    # Other cells should be unchanged
    assert np.isclose(b.cost_mean[4, 4], 1.5)
    assert np.isclose(b.cost_var[4, 4], 4.0)


def test_warning_updates_risk():
    b = BeliefMap.from_prior(8, 8, 1.5, 4.0, 0.1, 0.25)
    old_risk_right = b.risk_mean[0, 7].copy()
    apply_warning_to_belief(b, "RIGHT_AREA_RISKY")
    # Right half should have higher risk mean
    assert b.risk_mean[0, 7] > old_risk_right
    # Left half should be unchanged
    assert np.isclose(b.risk_mean[0, 0], 0.1)


def test_belief_copy_is_independent():
    b = BeliefMap.from_prior(4, 4, 1.0, 2.0, 0.1, 0.5)
    b2 = b.copy()
    b2.cost_mean[0, 0] = 999.0
    assert b.cost_mean[0, 0] == 1.0  # original unaffected


def test_total_variance():
    b = BeliefMap.from_prior(4, 4, 1.0, 2.0, 0.1, 0.5)
    expected = 2.0 * 16 + 0.5 * 16  # cost_var + risk_var
    assert abs(b.total_variance() - expected) < 1e-6
