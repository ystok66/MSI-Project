"""
Tests for cost_risk_model.py — joint latent heads.

Verifies:
1. Cost head linear response
2. Risk head sigmoid response
3. Same latent affects both cost and risk
4. Uncertainty propagation
5. Shared latent dimension
6. World weights reproducible from seed
7. Configurable supervision
"""

import numpy as np
import pytest

from src.agents.cost_risk_model import (
    BayesianCostHead, LatentCostRiskHead, WorldWeights, generate_world_weights,
)


def test_cost_head_linear_response():
    """Cost head responds linearly to feature changes."""
    ch = BayesianCostHead(d=4)
    # Train on a few examples
    x_high = np.array([0.0, 0.0, 0.9, 0.8])
    x_low = np.array([0.0, 0.0, 0.1, 0.1])
    for _ in range(20):
        ch.update_from_label(x_high, 5.0, weight=1.0)
        ch.update_from_label(x_low, 1.0, weight=1.0)
    assert ch.predict_cost(x_high) > ch.predict_cost(x_low), (
        "High-cost features should predict higher cost")


def test_risk_head_sigmoid_response():
    """Risk head output is in (0, 1) and responds to features."""
    lp = LatentCostRiskHead(d=4)
    x = np.array([0.0, 0.0, 0.5, 0.5])
    risk = lp.predict_risk(x)
    assert 0.0 < risk < 1.0, f"Risk should be in (0,1), got {risk}"

    # After training on risky features
    x_risky = np.array([0.0, 0.0, 0.9, 0.9])
    for _ in range(30):
        lp.risk_head.update_from_label(x_risky, 1.0, weight=2.0)
    assert lp.predict_risk(x_risky) > 0.5, "Risky features should predict high risk"


def test_same_latent_affects_both_cost_and_risk():
    """One latent vector change affects both cost_hat and risk_hat."""
    lp = LatentCostRiskHead(d=4)
    x_a = np.array([0.0, 0.0, 0.1, 0.1])
    x_b = np.array([0.0, 0.0, 0.9, 0.9])

    # Train: high-texture = high cost + high risk
    for _ in range(30):
        lp.update_from_outcome(x_b, cost_label=5.0, risk_label=0.8, weight=1.0)
        lp.update_from_outcome(x_a, cost_label=1.0, risk_label=0.05, weight=1.0)

    # Both cost and risk should differ for the two z vectors
    assert lp.predict_cost(x_b) > lp.predict_cost(x_a), "Cost should differ"
    assert lp.predict_risk(x_b) > lp.predict_risk(x_a), "Risk should differ"


def test_uncertainty_propagates_to_predictions():
    """Uncertainty decreases with more observations."""
    lp = LatentCostRiskHead(d=4)
    x = np.array([0.5, 0.0, 0.3, 0.3])
    unc_before = lp.predict_cost_uncertainty(x)
    risk_unc_before = lp.predict_risk_uncertainty(x)

    for _ in range(20):
        lp.update_from_outcome(x, cost_label=1.5, risk_label=0.1, weight=1.0)

    unc_after = lp.predict_cost_uncertainty(x)
    risk_unc_after = lp.predict_risk_uncertainty(x)
    assert unc_after < unc_before, f"Cost uncertainty should decrease: {unc_before} -> {unc_after}"
    assert risk_unc_after < risk_unc_before, f"Risk uncertainty should decrease"


def test_world_weights_reproducible_from_seed():
    """Same seed produces identical world weights."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    ww1 = generate_world_weights(rng1, d=4)
    ww2 = generate_world_weights(rng2, d=4)
    np.testing.assert_array_equal(ww1.w_cost, ww2.w_cost)
    np.testing.assert_array_equal(ww1.w_risk, ww2.w_risk)
    assert ww1.b_cost == ww2.b_cost
    assert ww1.b_risk == ww2.b_risk

    # Different seed should differ
    rng3 = np.random.default_rng(99)
    ww3 = generate_world_weights(rng3, d=4)
    assert not np.allclose(ww1.w_risk, ww3.w_risk), "Different seed should differ"


def test_world_weights_derive_cost_and_risk():
    """WorldWeights derives cost and risk from the same z."""
    ww = WorldWeights(
        w_cost=np.array([0.1, 0.0, 0.5, 0.3]),
        b_cost=1.0,
        w_risk=np.array([0.0, 0.0, 3.0, 2.5]),
        b_risk=-2.0,
    )
    z_safe = np.array([0.5, 0.0, 0.05, 0.05])
    z_risky = np.array([0.0, 0.0, 0.9, 0.85])
    assert ww.true_cost(z_risky) > ww.true_cost(z_safe)
    assert ww.true_risk(z_risky) > ww.true_risk(z_safe)
    assert 0.0 < ww.true_risk(z_safe) < 1.0
    assert 0.0 < ww.true_risk(z_risky) < 1.0


def test_cost_risk_model_configurable():
    """Supervision mode is configurable."""
    lp_oracle = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lp_binary = LatentCostRiskHead(d=4, risk_supervision="binary_outcome")
    assert lp_oracle.risk_supervision == "oracle_visited"
    assert lp_binary.risk_supervision == "binary_outcome"
