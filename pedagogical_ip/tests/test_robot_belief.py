"""
Tests for RobotBelief surrogate — Phase 7.

Verifies init, sync, copy modes, competence tracking, and state isolation.
"""

import numpy as np
import pytest
from copy import deepcopy

from src.teachers.robot_belief import (
    RobotBelief, init_robot_belief, sync_robot_belief, build_surrogate_predictor,
)
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.feature_belief import FeatureBeliefMap


def _setup():
    H, W, d = 5, 15, 4
    fb = FeatureBeliefMap(H, W, d=d)
    lp = LatentCostRiskHead(d=d)
    return H, W, d, fb, lp


def test_robot_belief_init_runs():
    """RobotBelief can be initialized."""
    H, W, d, fb, lp = _setup()
    rb = init_robot_belief(fb.mean, fb.var, latent_predictor=lp)
    assert isinstance(rb, RobotBelief)
    assert rb.agent_belief_mean.shape == (H, W, d)


def test_robot_belief_can_copy_agent_belief_exact():
    """Exact mode correctly copies agent belief."""
    H, W, d, fb, lp = _setup()
    fb.mean[2, 3] = np.array([1.0, 2.0, 3.0, 4.0])
    rb = init_robot_belief(fb.mean, fb.var, latent_predictor=lp, copy_mode="exact")
    np.testing.assert_array_equal(rb.agent_belief_mean[2, 3], fb.mean[2, 3])


def test_robot_belief_noisy_mode_differs_from_exact():
    """Noisy mode produces different belief than exact."""
    H, W, d, fb, lp = _setup()
    rng = np.random.default_rng(42)
    rb_exact = init_robot_belief(fb.mean, fb.var, latent_predictor=lp, copy_mode="exact")
    rb_noisy = init_robot_belief(fb.mean, fb.var, latent_predictor=lp,
                                  copy_mode="noisy", belief_noise_std=0.5, rng=rng)
    assert not np.allclose(rb_exact.agent_belief_mean, rb_noisy.agent_belief_mean)


def test_robot_belief_stale_mode_updates_less_frequently():
    """Stale mode skips sync when interval not reached."""
    H, W, d, fb, lp = _setup()
    rb = init_robot_belief(fb.mean, fb.var, latent_predictor=lp,
                           copy_mode="stale", stale_interval=3)
    old_mean = rb.agent_belief_mean.copy()
    fb.mean[2, 3] = np.array([9.0, 9.0, 9.0, 9.0])
    # t=1: should NOT sync (interval=3)
    sync_robot_belief(rb, fb.mean, fb.var, latent_predictor=lp, t=1)
    np.testing.assert_array_equal(rb.agent_belief_mean, old_mean)
    # t=3: should sync
    sync_robot_belief(rb, fb.mean, fb.var, latent_predictor=lp, t=3)
    np.testing.assert_array_equal(rb.agent_belief_mean[2, 3], [9.0, 9.0, 9.0, 9.0])


def test_robot_belief_tracks_boundedness_params():
    """Competence knobs are stored in robot belief."""
    H, W, d, fb, lp = _setup()
    rb = init_robot_belief(fb.mean, fb.var, latent_predictor=lp,
                           agent_search_budget=20, budget_mismatch=-5,
                           agent_risk_weight=4.0, risk_weight_mismatch=1.0)
    assert rb.agent_search_budget == 15  # 20 + (-5)
    # agent_risk_weight is now a PlannerWeights property (canonical, not mismatch-adjusted)
    assert rb.agent_risk_weight == 4.0   # legacy param → PlannerWeights.lambda_risk
    assert rb.risk_weight_mismatch == 1.0  # mismatch stored separately


def test_robot_belief_does_not_mutate_agent_state():
    """Sync does not mutate original agent belief arrays."""
    H, W, d, fb, lp = _setup()
    original_mean = fb.mean.copy()
    rb = init_robot_belief(fb.mean, fb.var, latent_predictor=lp)
    sync_robot_belief(rb, fb.mean, fb.var, latent_predictor=lp, t=5)
    np.testing.assert_array_equal(fb.mean, original_mean)


def test_robot_belief_config_switch():
    """robot_belief_mode can be toggled in runner."""
    from src.envs.lattice_v2_runner import LatticeV2Runner
    runner = LatticeV2Runner()
    s_off = runner.reset(seed=42, latent_mode=True, robot_belief_mode=False)
    assert s_off.robot_belief is None
    s_on = runner.reset(seed=42, latent_mode=True, robot_belief_mode=True)
    assert s_on.robot_belief is not None


def test_surrogate_predictor_builds():
    """build_surrogate_predictor returns a functional predictor."""
    H, W, d, fb, lp = _setup()
    rb = init_robot_belief(fb.mean, fb.var, latent_predictor=lp)
    surr = build_surrogate_predictor(rb)
    assert surr is not None
    # Should predict without error
    x = np.array([0.5, 0.5, 0.5, 0.5])
    cost = surr.predict_cost(x)
    risk = surr.predict_risk(x)
    assert np.isfinite(float(cost))
    assert np.isfinite(float(risk))
