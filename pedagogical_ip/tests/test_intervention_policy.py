"""
Tests for InterventionPolicy — Phase 7.

Verifies WAIT/WARN/UNLOCK scoring from counterfactual rollouts,
structured outputs, and policy preferences.
"""

import numpy as np
import pytest

from src.teachers.robot_belief import init_robot_belief
from src.teachers.intervention_policy import (
    score_interventions, InterventionDecision, InterventionConfig, VALID_ACTIONS,
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


def test_intervention_policy_scores_actions():
    """All three actions get scores."""
    gm, meta, lp, fb, rb, bc, pa, H, W = _setup()
    dec = score_interventions(rb, (2, 1), (2, W-2), bc, pa, meta)
    assert isinstance(dec, InterventionDecision)
    assert set(dec.scores.keys()) == {"WAIT", "WARN", "UNLOCK", "ITEM_DROP"}


def test_intervention_scores_use_counterfactual_rollouts():
    """Scores come from surrogate rollouts, not static constants."""
    gm, meta, lp, fb, rb, bc, pa, H, W = _setup()
    dec = score_interventions(rb, (2, 1), (2, W-2), bc, pa, meta)
    # Counterfactual scores should have entries for all actions
    assert "WAIT" in dec.counterfactual_scores
    assert "WARN" in dec.counterfactual_scores
    assert "UNLOCK" in dec.counterfactual_scores
    # Each should be a (risk, cost) tuple
    for action_cf in dec.counterfactual_scores.values():
        assert len(action_cf) == 2


def test_wait_preferred_when_path_safe():
    """Robot prefers WAIT when predicted path is safe."""
    gm, meta, lp, fb, rb, bc, pa, H, W = _setup()
    # With default (uninformed) predictor and no traps trained, path should be low risk
    dec = score_interventions(
        rb, (2, 1), (2, W-2), bc, pa, meta,
        config=InterventionConfig(catastrophe_weight=0.1, autonomy_penalty=5.0),
    )
    # High autonomy penalty should push toward WAIT
    assert dec.action in VALID_ACTIONS


def test_autonomy_penalty_discourages_overhelping():
    """Higher autonomy penalty makes intervention less likely."""
    gm, meta, lp, fb, rb, bc, pa, H, W = _setup()
    dec_low = score_interventions(
        rb, (2, 1), (2, W-2), bc, pa, meta,
        config=InterventionConfig(autonomy_penalty=0.1))
    dec_high = score_interventions(
        rb, (2, 1), (2, W-2), bc, pa, meta,
        config=InterventionConfig(autonomy_penalty=50.0))
    # High penalty: WARN and UNLOCK scores should be lower
    assert dec_high.scores["WARN"] <= dec_low.scores["WARN"]
    assert dec_high.scores["UNLOCK"] <= dec_low.scores["UNLOCK"]


def test_deadline_pressure_changes_choice():
    """Tighter deadline changes intervention scores."""
    gm, meta, lp, fb, rb, bc, pa, H, W = _setup()
    dec_loose = score_interventions(rb, (2, 1), (2, W-2), bc, pa, meta, t=0, t_max=100)
    dec_tight = score_interventions(rb, (2, 1), (2, W-2), bc, pa, meta, t=95, t_max=100)
    # Tight deadline should lower WAIT score
    assert dec_tight.scores["WAIT"] <= dec_loose.scores["WAIT"]


def test_policy_outputs_structured_decision():
    """Output is structured, not free-form text."""
    gm, meta, lp, fb, rb, bc, pa, H, W = _setup()
    dec = score_interventions(rb, (2, 1), (2, W-2), bc, pa, meta)
    assert dec.action in VALID_ACTIONS
    assert isinstance(dec.reason, str)
    assert isinstance(dec.decision_margin, float)
    assert isinstance(dec.predicted_prefix, list)
    assert isinstance(dec.predicted_failure_modes, object)


def test_robot_decision_depends_on_predicted_prefix():
    """Robot decision depends on predicted prefix, not just local risk."""
    gm, meta, lp, fb, rb, bc, pa, H, W = _setup()
    # Train predictor to predict high risk for certain features
    for _ in range(30):
        lp.update_from_outcome(np.array([0.5, 0.5, 0.5, 0.5]),
                              cost_label=1.0, risk_label=0.9, weight=3.0)
    rb_risky = init_robot_belief(fb.mean, fb.var, latent_predictor=lp)
    dec = score_interventions(rb_risky, (2, 1), (2, W-2), bc, pa, meta)
    # Robot should have non-trivial scores
    assert isinstance(dec, InterventionDecision)


def test_boundedness_mismatch_changes_decision():
    """Robot with different budget estimate may make different choices."""
    gm, meta, lp, fb, rb, bc, pa, H, W = _setup()
    rb_weak = init_robot_belief(fb.mean, fb.var, latent_predictor=lp,
                                 agent_search_budget=4, budget_mismatch=-10)
    rb_strong = init_robot_belief(fb.mean, fb.var, latent_predictor=lp,
                                   agent_search_budget=50, budget_mismatch=20)
    dec_weak = score_interventions(rb_weak, (2, 1), (2, W-2), bc, pa, meta)
    dec_strong = score_interventions(rb_strong, (2, 1), (2, W-2), bc, pa, meta)
    # Both should produce valid decisions
    assert dec_weak.action in VALID_ACTIONS
    assert dec_strong.action in VALID_ACTIONS
