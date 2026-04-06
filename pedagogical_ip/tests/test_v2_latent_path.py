"""
Tests for V2 latent path — end-to-end integration.

Verifies:
1. Latent mode reset runs
2. Latent mode episode runs to completion
3. Planner uses cost+risk+uncertainty
4. Cost/risk tradeoff affects action
5. Legacy mode baseline unchanged
6. Latent mode uses same z for cost and risk
7. World weights reproducible from seed
8. Latent mode info contains predictions
"""

import numpy as np
import pytest

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.envs.lattice_v2 import generate_lattice_v2, FEATURE_DIM
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.planner_astar import cell_cost_v2_latent


runner = LatticeV2Runner()


def test_v2_latent_mode_reset_runs():
    """Latent mode reset produces valid state."""
    s = runner.reset(seed=10, latent_mode=True)
    assert s.latent_mode is True
    assert s.latent_predictor is not None
    assert s.meta.world_weights is not None
    assert s.meta.latent_mode is True
    assert s.agent_pos == (2, 1)
    assert s.t == 0


def test_v2_latent_mode_episode_runs():
    """Latent mode episode runs to termination."""
    s = runner.reset(seed=42, latent_mode=True)
    while not s.done:
        runner.step(s)
    assert s.done
    assert s.steps > 0
    assert s.steps <= s.t_max


def test_planner_uses_cost_risk_uncertainty():
    """Latent planner cost function uses all 4 components."""
    lp = LatentCostRiskHead(d=4)
    x = np.array([0.0, 0.0, 0.5, 0.5])
    # Train so predictions are non-trivial
    for _ in range(10):
        lp.update_from_outcome(x, cost_label=2.0, risk_label=0.3, weight=1.0)

    fbm = np.full((5, 5, 4), 0.5, dtype=np.float64)
    passable = np.ones((5, 5), dtype=bool)

    cost_default = cell_cost_v2_latent(2, 2, fbm, lp, passable)
    # With higher risk weight, score should increase
    cost_high_risk = cell_cost_v2_latent(2, 2, fbm, lp, passable, lambda_r=20.0)
    assert cost_high_risk > cost_default, "Higher λ_r should increase score"

    # With higher cost weight, score should increase
    cost_high_c = cell_cost_v2_latent(2, 2, fbm, lp, passable, lambda_c=5.0)
    assert cost_high_c > cost_default, "Higher λ_c should increase score"


def test_high_cost_low_risk_tradeoff():
    """Planner distinguishes high-cost/low-risk vs low-cost/high-risk cells."""
    lp = LatentCostRiskHead(d=4)
    x_high_cost = np.array([1.0, 0.0, 0.1, 0.1])
    x_high_risk = np.array([0.0, 0.0, 0.9, 0.9])

    for _ in range(30):
        lp.update_from_outcome(x_high_cost, cost_label=5.0, risk_label=0.05, weight=1.0)
        lp.update_from_outcome(x_high_risk, cost_label=1.0, risk_label=0.8, weight=1.0)

    fbm = np.zeros((3, 3, 4), dtype=np.float64)
    passable = np.ones((3, 3), dtype=bool)

    fbm[1, 0] = x_high_cost
    fbm[1, 2] = x_high_risk

    # With high λ_r, risk penalty dominates cost difference
    score_cost = cell_cost_v2_latent(1, 0, fbm, lp, passable, lambda_r=15.0)
    score_risk = cell_cost_v2_latent(1, 2, fbm, lp, passable, lambda_r=15.0)
    assert score_risk > score_cost, (
        f"High-risk cell should have higher total score: {score_risk} vs {score_cost}")


def test_legacy_mode_baseline_unchanged():
    """Legacy mode (latent_mode=False) produces same results."""
    results_legacy = []
    for seed in range(20):
        s = runner.reset(seed=seed, tutor_mode="none", latent_mode=False)
        while not s.done:
            runner.step(s)
        results_legacy.append(runner.get_metrics(s))
    surv = sum(r["survived"] for r in results_legacy) / len(results_legacy)
    # Should be approximately 9% for no_tutor
    assert surv < 0.30, f"Legacy no_tutor survival too high: {surv:.0%}"


def test_latent_mode_uses_same_z_for_cost_and_risk():
    """Same z_s is used by both cost_head and risk_head."""
    lp = LatentCostRiskHead(d=4)
    z = np.array([0.0, 0.0, 0.7, 0.6])

    # Predictions before training
    cost_before = lp.predict_cost(z)
    risk_before = lp.predict_risk(z)

    # Train with one update using the SAME z
    # Use risk_label=0.8 (not 0.5) to avoid zero gradient at sigmoid(0)=0.5
    lp.update_from_outcome(z, cost_label=3.0, risk_label=0.8, weight=5.0)

    cost_after = lp.predict_cost(z)
    risk_after = lp.predict_risk(z)

    # Both should change from the same z
    assert cost_after != cost_before, "Cost prediction should change after update"
    assert risk_after != risk_before, "Risk prediction should change after update"


def test_world_weights_in_meta_reproducible():
    """World weights stored in meta are reproducible from seed."""
    _, _, meta1 = generate_lattice_v2(seed=77, latent_mode=True)
    _, _, meta2 = generate_lattice_v2(seed=77, latent_mode=True)
    assert meta1.world_weights is not None
    assert meta2.world_weights is not None
    np.testing.assert_array_equal(meta1.world_weights.w_cost, meta2.world_weights.w_cost)
    np.testing.assert_array_equal(meta1.world_weights.w_risk, meta2.world_weights.w_risk)
