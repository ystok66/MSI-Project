"""
Tests for PragmaticWarner protocol — post-abstraction.

Verifies:
1. RSAWarner and LaneWarner satisfy the PragmaticWarner protocol
2. select_utterance() returns valid utterances
3. listener_update() changes beliefs
4. Lane bias is computable and reasonable
"""

import numpy as np
import pytest

from src.agents.pragmatic_warning import PragmaticWarner
from src.teachers.rsa_warning import RSAWarner, UTTERANCE_VOCAB
from src.agents.warning_update import LaneWarner, Utterance
from src.agents.belief import BeliefMap
from src.agents.feature_belief import FeatureBeliefMap
from src.agents.risk_model import BayesianRiskHead
from src.envs.lattice_v2 import FEATURE_DIM


# ── Protocol compliance ──────────────────────────────────────────────

def test_rsa_warner_implements_protocol():
    """RSAWarner satisfies the PragmaticWarner protocol."""
    warner = RSAWarner()
    assert isinstance(warner, PragmaticWarner)


def test_lane_warner_implements_protocol():
    """LaneWarner satisfies the PragmaticWarner protocol."""
    warner = LaneWarner()
    assert isinstance(warner, PragmaticWarner)


# ── RSAWarner ────────────────────────────────────────────────────────

def test_rsa_select_returns_valid_utterance():
    """RSAWarner.select_utterance() returns a valid utterance or None."""
    H, W = 8, 8
    rng = np.random.default_rng(42)
    risk_mean = np.full((H, W), 0.1)
    risk_var = np.full((H, W), 0.2)
    true_risk = np.zeros((H, W))
    # Put risk on the right side
    true_risk[:, 5:] = 0.3

    warner = RSAWarner(alpha=5.0, beta=0.1, tau=1.0)
    result = warner.select_utterance({
        "learner_belief_risk_mean": risk_mean,
        "learner_belief_risk_var": risk_var,
        "true_risk": true_risk,
        "agent_pos": (3, 3),
        "current_plan": None,
    })
    if result is not None:
        assert result in UTTERANCE_VOCAB, f"Got invalid utterance: {result}"


# ── LaneWarner ───────────────────────────────────────────────────────

def test_lane_select_returns_valid_utterance():
    """LaneWarner.select_utterance() returns a valid utterance string or None."""
    H, W = 7, 15
    fb = FeatureBeliefMap(H, W, d=FEATURE_DIM)
    risk_head = BayesianRiskHead(d=FEATURE_DIM)

    # Set some cells to have trap-like features
    trap_cells = [(1, 3), (1, 4), (1, 5)]
    for r, c in trap_cells:
        fb.update(r, c, np.array([0.0, 0.0, 0.85, 0.78]), obs_var=0.01)

    warner = LaneWarner(tau=0.3, lambda_lane_warn=5.0)
    result = warner.select_utterance({
        "candidate_cells": trap_cells,
        "feature_belief": fb,
        "risk_head": risk_head,
    })

    valid_values = [u.value for u in Utterance]
    if result is not None:
        assert result in valid_values, f"Got invalid utterance: {result}"


def test_lane_listener_update_changes_belief():
    """LaneWarner.listener_update() modifies risk head predictions."""
    H, W = 7, 15
    fb = FeatureBeliefMap(H, W, d=FEATURE_DIM)
    risk_head = BayesianRiskHead(d=FEATURE_DIM)
    warned_bias = {}

    cells = [(1, 3), (1, 4)]
    for r, c in cells:
        fb.update(r, c, np.array([0.0, 0.0, 0.85, 0.78]), obs_var=0.01)

    # Get risk prediction before warning
    x_before = fb.get_mean(1, 3)
    p_before = risk_head.predict_risk(x_before)

    warner = LaneWarner(tau=0.3, lambda_lane_warn=5.0, weight=5.0)
    effect = warner.listener_update(
        "risky_texture_ahead",
        fb,
        upcoming_cells=cells,
        risk_head=risk_head,
        warned_lane_bias=warned_bias,
        segment_index=0,
    )

    # Risk head should have learned something
    p_after = risk_head.predict_risk(x_before)
    assert p_after > p_before, (
        f"Warning should increase risk prediction: {p_after} vs {p_before}"
    )
    assert effect is not None
    assert 0 in warned_bias, "Lane bias should be stored for segment 0"


def test_lane_bias_computable():
    """compute_lane_bias() produces reasonable numerical values."""
    from src.agents.warning_update import compute_lane_bias

    H, W = 7, 15
    fb = FeatureBeliefMap(H, W, d=FEATURE_DIM)
    cells = [(1, 3), (1, 4), (1, 5)]

    # Set trap-like features
    for r, c in cells:
        fb.update(r, c, np.array([0.0, 0.0, 0.88, 0.82]), obs_var=0.01)

    bias = compute_lane_bias(Utterance.RISKY_TEXTURE_AHEAD, cells, fb, tau=0.3)
    assert isinstance(bias, float)
    assert bias > 0, f"Bias should be positive for trap-like features: {bias}"
    assert np.isfinite(bias), f"Bias should be finite: {bias}"
