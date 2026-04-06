"""Tests for symbolic RSA warning module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.teachers.rsa_warning import (
    UTTERANCE_VOCAB,
    score_utterances,
    select_best_warning,
    _build_region_masks,
)
from src.agents.belief import BeliefMap, apply_rsa_warning


def test_utterance_vocab_size():
    assert len(UTTERANCE_VOCAB) == 6


def test_region_masks_cover_grid():
    """Each region mask should cover cells in the expected area."""
    masks = _build_region_masks(8, 8)
    assert masks["LEFT_RISKY"][:, :4].all()     # left half
    assert not masks["LEFT_RISKY"][:, 4:].any()  # not right half
    assert masks["RIGHT_RISKY"][:, 4:].all()
    assert masks["UPPER_RISKY"][:4, :].all()
    assert masks["LOWER_RISKY"][4:, :].all()
    assert masks["DOOR_PATH_SAFE"].sum() > 0     # non-empty


def test_score_utterances_returns_all():
    """Should return a score for every utterance."""
    true_risk = np.random.default_rng(42).random((8, 8)) * 0.5
    belief = BeliefMap.from_prior(8, 8)
    scores = score_utterances(
        belief.risk_mean, belief.risk_var,
        true_risk, (0, 0),
    )
    assert set(scores.keys()) == set(UTTERANCE_VOCAB)


def test_select_best_warning_returns_valid():
    """Best warning should be from vocabulary."""
    true_risk = np.zeros((8, 8))
    true_risk[:, 4:] = 0.8  # right side very risky
    belief = BeliefMap.from_prior(8, 8)
    utt, score = select_best_warning(
        belief.risk_mean, belief.risk_var,
        true_risk, (0, 0),
    )
    assert utt in UTTERANCE_VOCAB


def test_rsa_prefers_risky_side():
    """When right side is risky and learner doesn't know, RSA should prefer RIGHT_RISKY."""
    true_risk = np.zeros((8, 8))
    true_risk[:, 4:] = 0.9  # right is dangerous

    belief = BeliefMap.from_prior(8, 8, prior_risk_mean=0.1)
    scores = score_utterances(
        belief.risk_mean, belief.risk_var,
        true_risk, (0, 0),
    )
    # RIGHT_RISKY should have high score (region has high true risk, learner underestimates)
    # LEFT_RISKY should have lower score (no actual risk there)
    assert scores["RIGHT_RISKY"] > scores["LEFT_RISKY"]


def test_apply_rsa_warning_changes_belief():
    """Warning should update risk belief in the target region."""
    belief = BeliefMap.from_prior(8, 8, prior_risk_mean=0.1, prior_risk_var=0.25)
    old_right_risk = belief.risk_mean[0, 6]

    apply_rsa_warning(belief, "RIGHT_RISKY", warn_sensitivity=0.5)

    # Right-side risk should increase
    assert belief.risk_mean[0, 6] > old_right_risk
    # Left-side should be unchanged
    assert np.isclose(belief.risk_mean[0, 0], 0.1)


def test_apply_safe_warning_reduces_risk():
    """DOOR_PATH_SAFE should reduce risk belief in center region."""
    belief = BeliefMap.from_prior(8, 8, prior_risk_mean=0.5, prior_risk_var=0.25)
    old_center_risk = belief.risk_mean[3, 3]

    apply_rsa_warning(belief, "DOOR_PATH_SAFE", warn_sensitivity=0.5)

    # Center risk should decrease toward 0
    assert belief.risk_mean[3, 3] < old_center_risk


def test_warn_sensitivity_affects_update_strength():
    """Higher warn_sensitivity should produce stronger belief update."""
    belief_low = BeliefMap.from_prior(8, 8, prior_risk_mean=0.1, prior_risk_var=0.25)
    belief_high = BeliefMap.from_prior(8, 8, prior_risk_mean=0.1, prior_risk_var=0.25)

    apply_rsa_warning(belief_low, "RIGHT_RISKY", warn_sensitivity=0.25)
    apply_rsa_warning(belief_high, "RIGHT_RISKY", warn_sensitivity=1.0)

    # Higher sensitivity → bigger shift
    shift_low = belief_low.risk_mean[0, 6] - 0.1
    shift_high = belief_high.risk_mean[0, 6] - 0.1
    assert shift_high > shift_low
