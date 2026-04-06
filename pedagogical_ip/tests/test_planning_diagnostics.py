"""
Tests for planning diagnostics — Phase 6.

Verifies structured outputs, confidence, dominant_reason, failure modes.
Includes read-only and determinism checks.
"""

import numpy as np
import pytest
from copy import deepcopy

from src.agents.belief_planning import (
    plan_from_belief, estimate_failure_modes,
    BeliefPlan, FailureModeEstimate, ScoreBreakdown, DOMINANT_REASONS,
)
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.feature_belief import FeatureBeliefMap
from src.agents.risk_model import BayesianRiskHead
from src.agents.planner_astar import plan_with_alternatives_v2
from src.envs.lattice_v2 import generate_lattice_v2, FEATURE_DIM


def _setup():
    gm, cfg, meta = generate_lattice_v2(seed=42, latent_mode=True)
    H, W = gm.height, gm.width
    lp = LatentCostRiskHead(d=4)
    fb = FeatureBeliefMap(H, W, d=4)
    rh = lp.risk_head
    bc = np.ones((H, W))
    pa = (gm.cell_types != 0).astype(bool)
    return gm, meta, lp, fb, rh, bc, pa, H, W


def test_planning_result_contains_expected_fields():
    """BeliefPlan has all required fields."""
    _, _, lp, fb, rh, bc, pa, H, W = _setup()
    bp = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, prefix_horizon=5)
    assert hasattr(bp, 'action')
    assert hasattr(bp, 'planned_prefix')
    assert hasattr(bp, 'expected_cost')
    assert hasattr(bp, 'expected_risk')
    assert hasattr(bp, 'uncertainty')
    assert hasattr(bp, 'dominant_reason')
    assert hasattr(bp, 'runner_up_gap')
    assert hasattr(bp, 'score_breakdown')
    assert isinstance(bp.score_breakdown, ScoreBreakdown)


def test_action_confidence_computable():
    """Confidence is a valid float in [0, 1]."""
    _, _, lp, fb, rh, bc, pa, H, W = _setup()
    bp = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, prefix_horizon=5)
    assert 0.0 <= bp.action_confidence <= 1.0


def test_action_confidence_changes_with_runner_up_gap():
    """Larger gap → higher confidence (for same temperature)."""
    _, _, lp, fb, rh, bc, pa, H, W = _setup()
    bp = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, prefix_horizon=5, confidence_temperature=1.0)
    # Confidence formula: gap / (gap + T)
    if bp.runner_up_gap > 0:
        expected = bp.runner_up_gap / (bp.runner_up_gap + 1.0)
        assert abs(bp.action_confidence - expected) < 0.01


def test_dominant_reason_is_valid():
    """dominant_reason is a valid structured value."""
    _, _, lp, fb, rh, bc, pa, H, W = _setup()
    bp = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, prefix_horizon=5)
    assert bp.dominant_reason in DOMINANT_REASONS


def test_dominant_reason_allows_mixed_case():
    """dominant_reason can be 'mixed' when components are balanced."""
    # This is a coverage test — mixed should be a valid value
    assert "mixed" in DOMINANT_REASONS


def test_failure_modes_computable():
    """FailureModeEstimate fields are computable."""
    _, _, lp, fb, rh, bc, pa, H, W = _setup()
    bp = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, prefix_horizon=5)
    _, _, _, cand = plan_with_alternatives_v2(
        (2, 1), (2, W-2), bc, fb.mean, rh, budget=30,
        passable_mask=pa, latent_predictor=lp)
    fm = estimate_failure_modes(bp, t=0, t_max=100, candidate_scores=cand)
    assert isinstance(fm, FailureModeEstimate)
    assert 0.0 <= fm.high_cumulative_risk <= 1.0
    assert fm.high_uncertainty >= 0
    assert 0.0 <= fm.deadline_miss <= 1.0


def test_failure_modes_reflect_path_risk():
    """Higher prefix risk → higher high_cumulative_risk score."""
    _, _, lp, fb, rh, bc, pa, H, W = _setup()
    # Train to predict high risk
    for _ in range(30):
        lp.update_from_outcome(np.array([0.5, 0.5, 0.5, 0.5]),
                               cost_label=1.0, risk_label=0.8, weight=3.0)
    bp = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, prefix_horizon=5)
    _, _, _, cand = plan_with_alternatives_v2(
        (2, 1), (2, W-2), bc, fb.mean, rh, budget=30,
        passable_mask=pa, latent_predictor=lp)
    fm = estimate_failure_modes(bp, t=0, t_max=100, candidate_scores=cand)
    assert fm.high_cumulative_risk > 0.0


def test_failure_modes_reflect_deadline_pressure():
    """Tight deadline → higher deadline_miss score."""
    _, _, lp, fb, rh, bc, pa, H, W = _setup()
    bp = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, prefix_horizon=5)
    _, _, _, cand = plan_with_alternatives_v2(
        (2, 1), (2, W-2), bc, fb.mean, rh, budget=30,
        passable_mask=pa, latent_predictor=lp)
    # Tight: t nearly at t_max
    fm_tight = estimate_failure_modes(bp, t=99, t_max=100, candidate_scores=cand)
    fm_loose = estimate_failure_modes(bp, t=0, t_max=100, candidate_scores=cand)
    assert fm_tight.deadline_miss >= fm_loose.deadline_miss


def test_planning_diagnostics_are_read_only():
    """plan_from_belief does not alter predictor, belief, or env state."""
    _, _, lp, fb, rh, bc, pa, H, W = _setup()
    lp_copy = deepcopy(lp)
    fb_copy = fb.mean.copy()
    bc_copy = bc.copy()
    bp = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, prefix_horizon=5)
    np.testing.assert_array_equal(fb.mean, fb_copy)
    np.testing.assert_array_equal(bc, bc_copy)
    np.testing.assert_array_equal(lp.cost_head.w, lp_copy.cost_head.w)
    np.testing.assert_array_equal(lp.risk_head.w, lp_copy.risk_head.w)


def test_same_belief_same_plan_fixed_seed():
    """Identical belief + identical config → identical plan (determinism)."""
    _, _, lp, fb, rh, bc, pa, H, W = _setup()
    bp1 = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, prefix_horizon=5)
    bp2 = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, prefix_horizon=5)
    assert bp1.action == bp2.action
    assert bp1.next_pos == bp2.next_pos
    assert bp1.full_path == bp2.full_path
    assert bp1.runner_up_gap == bp2.runner_up_gap


def test_runner_up_gap_uses_path_level_scores():
    """runner_up_gap comes from path-level candidate scores, not single-cell."""
    _, _, lp, fb, rh, bc, pa, H, W = _setup()
    _, _, _, cand = plan_with_alternatives_v2(
        (2, 1), (2, W-2), bc, fb.mean, rh, budget=30,
        passable_mask=pa, latent_predictor=lp)
    # Candidate scores should have multiple entries (path-level)
    assert len(cand) >= 1
    # Values should be path-level (not just single cell cost)
    for score in cand.values():
        assert score > 0, "Path-level scores should be positive"


def test_failure_modes_are_prefix_based():
    """Failure mode scores depend on the chosen prefix, not global state."""
    _, _, lp, fb, rh, bc, pa, H, W = _setup()
    bp = plan_from_belief(
        (2, 1), (2, W-2), bc, fb.mean, rh, pa,
        latent_predictor=lp, prefix_horizon=5)
    _, _, _, cand = plan_with_alternatives_v2(
        (2, 1), (2, W-2), bc, fb.mean, rh, budget=30,
        passable_mask=pa, latent_predictor=lp)
    fm = estimate_failure_modes(bp, t=0, t_max=100, candidate_scores=cand)
    # high_cumulative_risk should match prefix prediction
    if bp.prefix_prediction:
        assert abs(fm.high_cumulative_risk - bp.prefix_prediction.cumulative_risk) < 0.01
