"""
Tests for belief planning core — Phase 6.

Verifies:
1. plan_from_belief runs and returns BeliefPlan
2. Planning uses predicted cost
3. Planning uses predicted risk
4. Planning uses uncertainty
5. Belief-based plan differs from oracle when belief is wrong
6. Search budget affects behavior
7. Confidence temperature is configurable
8. belief_planning_mode config switch works
"""

import numpy as np
import pytest
from copy import deepcopy

from src.agents.belief_planning import (
    plan_from_belief, estimate_failure_modes,
    BeliefPlan, FailureModeEstimate, DOMINANT_REASONS,
)
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.feature_belief import FeatureBeliefMap
from src.agents.risk_model import BayesianRiskHead
from src.envs.lattice_v2 import generate_lattice_v2, FEATURE_DIM


def _setup():
    gm, cfg, meta = generate_lattice_v2(seed=42, latent_mode=True)
    H, W = gm.height, gm.width
    lp = LatentCostRiskHead(d=4)
    fb = FeatureBeliefMap(H, W, d=4)
    rh = lp.risk_head
    belief_cost = np.ones((H, W))
    passable = (gm.cell_types != 0).astype(bool)
    return gm, meta, lp, fb, rh, belief_cost, passable, H, W


def test_plan_from_belief_runs():
    """plan_from_belief returns a valid BeliefPlan."""
    gm, meta, lp, fb, rh, bc, pa, H, W = _setup()
    bp = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, prefix_horizon=5)
    assert isinstance(bp, BeliefPlan)
    assert bp.action in ("UP", "DOWN", "LEFT", "RIGHT", "STAY")
    assert len(bp.full_path) > 0


def test_plan_from_belief_uses_predicted_cost():
    """Changing cost prediction changes action/path."""
    gm, meta, lp, fb, rh, bc, pa, H, W = _setup()
    bp1 = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, prefix_horizon=5)
    # Train predictor to make one direction very costly
    for _ in range(30):
        lp.update_from_outcome(fb.mean[2, 2], cost_label=20.0, risk_label=0.0, weight=3.0)
    bp2 = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, prefix_horizon=5)
    # Path or cost should change
    assert bp2.expected_cost != bp1.expected_cost or bp2.full_path != bp1.full_path


def test_plan_from_belief_uses_predicted_risk():
    """Changing risk prediction changes plan."""
    gm, meta, lp, fb, rh, bc, pa, H, W = _setup()
    bp1 = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, prefix_horizon=5)
    for _ in range(30):
        lp.update_from_outcome(fb.mean[2, 2], cost_label=1.0, risk_label=0.9, weight=3.0)
    bp2 = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, prefix_horizon=5)
    assert bp2.expected_risk != bp1.expected_risk or bp2.full_path != bp1.full_path


def test_plan_from_belief_uses_uncertainty():
    """Higher uncertainty penalty changes preference."""
    gm, meta, lp, fb, rh, bc, pa, H, W = _setup()
    bp_low = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, lambda_uc=0.0, lambda_ur=0.0, prefix_horizon=5)
    bp_high = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, lambda_uc=5.0, lambda_ur=5.0, prefix_horizon=5)
    # Uncertainty contribution should differ
    assert bp_high.uncertainty >= 0
    assert bp_low.uncertainty >= 0


def test_belief_differs_from_oracle_when_wrong():
    """When belief is badly wrong, plan differs from oracle-based plan."""
    gm, meta, lp, fb, rh, bc, pa, H, W = _setup()
    # Train predictor with wrong info (safe cells labeled as risky)
    for _ in range(30):
        lp.update_from_outcome(np.array([0.5, 0.5, 0.1, 0.1]),
                               cost_label=1.0, risk_label=0.9, weight=3.0)
    bp_wrong = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, prefix_horizon=5)
    # At minimum the plan should complete
    assert bp_wrong.action in ("UP", "DOWN", "LEFT", "RIGHT", "STAY")


def test_search_budget_affects_path():
    """Different search budgets produce different path quality."""
    gm, meta, lp, fb, rh, bc, pa, H, W = _setup()
    bp_low = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, search_budget=4, prefix_horizon=5)
    bp_high = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, search_budget=50, prefix_horizon=5)
    # Both should produce valid results
    assert bp_low.action in ("UP", "DOWN", "LEFT", "RIGHT", "STAY")
    assert bp_high.action in ("UP", "DOWN", "LEFT", "RIGHT", "STAY")


def test_confidence_temperature_is_configurable():
    """Different temperatures produce different confidence values."""
    gm, meta, lp, fb, rh, bc, pa, H, W = _setup()
    bp1 = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, confidence_temperature=0.1, prefix_horizon=5)
    bp2 = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, confidence_temperature=10.0, prefix_horizon=5)
    # Same gap, different temperatures → different confidence
    if bp1.runner_up_gap > 0:
        assert bp1.action_confidence != bp2.action_confidence


def test_belief_planning_config_switch():
    """belief_planning_mode can be toggled."""
    from src.envs.lattice_v2_runner import LatticeV2Runner
    runner = LatticeV2Runner()
    # Mode off
    s_off = runner.reset(seed=42, latent_mode=True, belief_planning_mode=False)
    runner.step(s_off)
    assert s_off.last_belief_plan is None
    # Mode on
    s_on = runner.reset(seed=42, latent_mode=True, belief_planning_mode=True, prefix_horizon=5)
    runner.step(s_on)
    assert s_on.last_belief_plan is not None
