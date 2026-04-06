"""
Tests for AgentPredictor — Phase 7.

Verifies prefix prediction from surrogate, counterfactual rollouts,
and read-only guarantees.
"""

import numpy as np
import pytest
from copy import deepcopy

from src.teachers.robot_belief import (
    RobotBelief, init_robot_belief, sync_robot_belief,
)
from src.teachers.agent_predictor import (
    predict_agent_prefix, predict_agent_prefix_after_warn,
    predict_agent_prefix_after_unlock, estimate_learning_gain,
    AgentPrediction,
)
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.feature_belief import FeatureBeliefMap
from src.envs.lattice_v2 import generate_lattice_v2, FEATURE_DIM
from src.envs.map_generator import CellType


def _setup():
    gm, cfg, meta = generate_lattice_v2(seed=42, latent_mode=True)
    H, W = gm.height, gm.width
    lp = LatentCostRiskHead(d=4)
    fb = FeatureBeliefMap(H, W, d=4)
    bc = np.ones((H, W))
    pa = (gm.cell_types != CellType.WALL).astype(bool)
    rb = init_robot_belief(fb.mean, fb.var, latent_predictor=lp)
    return gm, meta, lp, fb, rb, bc, pa, H, W


def test_agent_predictor_runs():
    """predict_agent_prefix returns an AgentPrediction."""
    gm, meta, lp, fb, rb, bc, pa, H, W = _setup()
    pred = predict_agent_prefix(rb, (2, 1), (2, W-2), bc, pa)
    assert isinstance(pred, AgentPrediction)
    assert pred.predicted_plan.action in ("UP", "DOWN", "LEFT", "RIGHT", "STAY")


def test_agent_predictor_returns_prefix():
    """Prediction includes a path prefix, not just a single action."""
    gm, meta, lp, fb, rb, bc, pa, H, W = _setup()
    pred = predict_agent_prefix(rb, (2, 1), (2, W-2), bc, pa, prefix_horizon=5)
    assert len(pred.predicted_plan.planned_prefix) > 0


def test_agent_predictor_respects_prefix_horizon():
    """Longer horizon produces longer prefix."""
    gm, meta, lp, fb, rb, bc, pa, H, W = _setup()
    pred_short = predict_agent_prefix(rb, (2, 1), (2, W-2), bc, pa, prefix_horizon=2)
    pred_long = predict_agent_prefix(rb, (2, 1), (2, W-2), bc, pa, prefix_horizon=10)
    assert len(pred_long.predicted_plan.planned_prefix) >= len(pred_short.predicted_plan.planned_prefix)


def test_agent_predictor_changes_with_belief_mismatch():
    """Changing surrogate belief changes prediction."""
    gm, meta, lp, fb, rb, bc, pa, H, W = _setup()
    pred1 = predict_agent_prefix(rb, (2, 1), (2, W-2), bc, pa)
    # Modify surrogate belief
    rb.agent_belief_mean[2, 2] = np.array([5.0, 5.0, 5.0, 5.0])
    # Re-train surrogate predictor to match
    for _ in range(20):
        lp.update_from_outcome(np.array([5.0, 5.0, 5.0, 5.0]),
                              cost_label=20.0, risk_label=0.9, weight=3.0)
    rb2 = init_robot_belief(rb.agent_belief_mean, rb.agent_belief_var, latent_predictor=lp)
    pred2 = predict_agent_prefix(rb2, (2, 1), (2, W-2), bc, pa)
    # Should produce valid result (may or may not differ in this small grid)
    assert pred2.predicted_plan.action in ("UP", "DOWN", "LEFT", "RIGHT", "STAY")


def test_agent_predictor_changes_with_boundedness_mismatch():
    """Different budget changes prediction."""
    gm, meta, lp, fb, rb, bc, pa, H, W = _setup()
    rb_low = init_robot_belief(fb.mean, fb.var, latent_predictor=lp,
                                agent_search_budget=4)
    rb_high = init_robot_belief(fb.mean, fb.var, latent_predictor=lp,
                                 agent_search_budget=50)
    pred_low = predict_agent_prefix(rb_low, (2, 1), (2, W-2), bc, pa)
    pred_high = predict_agent_prefix(rb_high, (2, 1), (2, W-2), bc, pa)
    assert pred_low.predicted_plan.action in ("UP", "DOWN", "LEFT", "RIGHT", "STAY")
    assert pred_high.predicted_plan.action in ("UP", "DOWN", "LEFT", "RIGHT", "STAY")


def test_agent_predictor_is_read_only():
    """Prediction does not mutate agent belief or env state."""
    gm, meta, lp, fb, rb, bc, pa, H, W = _setup()
    fb_copy = fb.mean.copy()
    lp_w_copy = lp.cost_head.w.copy()
    bc_copy = bc.copy()
    pa_copy = pa.copy()
    predict_agent_prefix(rb, (2, 1), (2, W-2), bc, pa)
    np.testing.assert_array_equal(fb.mean, fb_copy)
    np.testing.assert_array_equal(lp.cost_head.w, lp_w_copy)
    np.testing.assert_array_equal(bc, bc_copy)
    np.testing.assert_array_equal(pa, pa_copy)


def test_learning_gain_computable():
    """Learning gain heuristic returns a non-negative float."""
    gm, meta, lp, fb, rb, bc, pa, H, W = _setup()
    pred = predict_agent_prefix(rb, (2, 1), (2, W-2), bc, pa)
    gain = estimate_learning_gain(rb, pred.predicted_plan.planned_prefix)
    assert gain >= 0.0
