"""
Tests for prefix prediction — Phase 5.

Verifies:
1. Returns path prefix
2. Respects horizon
3. Cost computable
4. Risk computable
5. Uncertainty computable
6. risky_prefix_cells reported
7. Warning changes prefix
8. Existing planner still runs
9. Prefix prediction is read-only
"""

import numpy as np
import pytest
from copy import deepcopy

from src.agents.prefix_prediction import compute_prefix_predictions, PrefixPrediction
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.feature_belief import FeatureBeliefMap
from src.agents.planner_astar import plan_next_action_v2
from src.agents.risk_model import BayesianRiskHead
from src.envs.lattice_v2 import generate_lattice_v2, FEATURE_DIM


def _train_predictor():
    """Create a trained LatentCostRiskHead for testing."""
    lp = LatentCostRiskHead(d=4)
    rng = np.random.default_rng(42)
    for _ in range(20):
        x = rng.uniform(0, 1, size=4)
        lp.update_from_outcome(x, cost_label=2.0+x[2]*3, risk_label=x[2]*0.8, weight=1.0)
    return lp


def test_prefix_prediction_returns_path_prefix():
    """compute_prefix_predictions returns a PrefixPrediction with prefix cells."""
    lp = _train_predictor()
    path = [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5)]
    fbm = np.full((5, 10, 4), 0.5)
    pred = compute_prefix_predictions(path, fbm, lp, horizon=3)
    assert isinstance(pred, PrefixPrediction)
    assert len(pred.prefix_cells) > 0
    assert len(pred.prefix_cells) <= 3


def test_prefix_length_respects_horizon():
    """Prefix length does not exceed horizon."""
    lp = _train_predictor()
    path = [(2, i) for i in range(10)]
    fbm = np.full((5, 10, 4), 0.5)
    for h in [1, 3, 5, 8]:
        pred = compute_prefix_predictions(path, fbm, lp, horizon=h)
        assert len(pred.prefix_cells) <= h


def test_prefix_expected_cost_computable():
    """cumulative_cost is computed and positive."""
    lp = _train_predictor()
    path = [(2, i) for i in range(6)]
    fbm = np.full((5, 10, 4), 0.5)
    pred = compute_prefix_predictions(path, fbm, lp, horizon=4)
    assert pred.cumulative_cost > 0


def test_prefix_expected_risk_computable():
    """cumulative_risk is computed and in [0, 1)."""
    lp = _train_predictor()
    path = [(2, i) for i in range(6)]
    fbm = np.full((5, 10, 4), 0.5)
    pred = compute_prefix_predictions(path, fbm, lp, horizon=4)
    assert 0.0 <= pred.cumulative_risk < 1.0


def test_prefix_uncertainty_computable():
    """cost and risk uncertainties are lists of valid floats."""
    lp = _train_predictor()
    path = [(2, i) for i in range(6)]
    fbm = np.full((5, 10, 4), 0.5)
    pred = compute_prefix_predictions(path, fbm, lp, horizon=4)
    assert len(pred.cost_uncertainties) == len(pred.prefix_cells)
    assert len(pred.risk_uncertainties) == len(pred.prefix_cells)
    assert all(u >= 0 for u in pred.cost_uncertainties)
    assert all(u >= 0 for u in pred.risk_uncertainties)


def test_risky_prefix_cells_reported():
    """Cells with risk above threshold are flagged."""
    lp = LatentCostRiskHead(d=4)
    # Train to make texture dims predict high risk
    for _ in range(30):
        lp.update_from_outcome(np.array([0, 0, 0.9, 0.9]),
                               cost_label=1.0, risk_label=0.8, weight=2.0)
        lp.update_from_outcome(np.array([0, 0, 0.1, 0.1]),
                               cost_label=1.0, risk_label=0.05, weight=2.0)
    path = [(2, 0), (2, 1), (2, 2), (2, 3)]
    fbm = np.zeros((5, 5, 4))
    fbm[2, 1] = [0, 0, 0.9, 0.9]  # risky
    fbm[2, 2] = [0, 0, 0.1, 0.1]  # safe
    fbm[2, 3] = [0, 0, 0.9, 0.9]  # risky
    pred = compute_prefix_predictions(path, fbm, lp, horizon=5, risk_threshold=0.3)
    assert len(pred.risky_prefix_cells) >= 1, "Should flag at least one risky cell"


def test_warning_changes_prefix_score_or_belief():
    """Warning should influence prefix predictions via belief changes."""
    gm, cfg, meta = generate_lattice_v2(seed=42, latent_mode=True)
    H, W = gm.height, gm.width
    lp = LatentCostRiskHead(d=4)
    fb = FeatureBeliefMap(H, W, d=4)
    # Simulate some observations
    for c in range(1, 6):
        fb.update(2, c, meta.cell_features[2, c], 0.01)

    path = [(2, c) for c in range(1, 8)]
    pred_before = compute_prefix_predictions(path, fb.mean, lp, horizon=5)

    # Simulate warning: train predictor with risky info for cells ahead
    for _ in range(10):
        lp.update_from_outcome(fb.mean[2, 4], cost_label=5.0, risk_label=0.7, weight=3.0)
    pred_after = compute_prefix_predictions(path, fb.mean, lp, horizon=5)
    assert pred_after.cumulative_risk != pred_before.cumulative_risk, (
        "Warning info should change prefix risk assessment")


def test_existing_planner_path_still_runs():
    """Legacy planner without prefix mode still works."""
    gm, cfg, meta = generate_lattice_v2(seed=42)
    H, W = gm.height, gm.width
    cost_mean = np.ones((H, W))
    feature_mean = np.full((H, W, FEATURE_DIM), 0.5)
    passable = (gm.cell_types != 0).astype(bool)  # rough approx
    rh = BayesianRiskHead(d=FEATURE_DIM)
    action, pos, path = plan_next_action_v2(
        (2, 1), (2, W-2), cost_mean, feature_mean, rh,
        budget=30, passable_mask=passable)
    assert action in ("UP", "DOWN", "LEFT", "RIGHT", "STAY")


def test_prefix_prediction_is_read_only():
    """compute_prefix_predictions does not alter predictor, belief, or env state."""
    lp = _train_predictor()
    fbm = np.full((5, 10, 4), 0.5, dtype=np.float64)
    path = [(2, i) for i in range(6)]

    # Snapshot state before
    lp_copy = deepcopy(lp)
    fbm_copy = fbm.copy()

    pred = compute_prefix_predictions(path, fbm, lp, horizon=4)

    # Verify nothing changed
    np.testing.assert_array_equal(fbm, fbm_copy)
    np.testing.assert_array_equal(lp.cost_head.w, lp_copy.cost_head.w)
    np.testing.assert_array_equal(lp.risk_head.w, lp_copy.risk_head.w)
    assert lp.cost_head.b == lp_copy.cost_head.b
    assert lp.risk_head.b == lp_copy.risk_head.b
    assert lp.cost_head.n_updates == lp_copy.cost_head.n_updates
