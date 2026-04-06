"""
Tests for shield item-drop semantics — Phase 8.

Verifies binary inventory, shield construction, consumption,
risk effect, counterfactual evaluation, and read-only guarantee.
"""

import numpy as np
import pytest

from src.teachers.interventions import (
    Intervention, InterventionType, ItemType, ItemEffect,
    InventoryState, SHIELD_DEFAULT_RISK_REDUCTION,
)
from src.teachers.robot_belief import init_robot_belief
from src.teachers.agent_predictor import (
    predict_agent_prefix, predict_agent_prefix_after_item_drop,
)
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.feature_belief import FeatureBeliefMap
from src.envs.lattice_v2 import generate_lattice_v2, FEATURE_DIM
from src.envs.map_generator import CellType


def test_item_drop_shield_constructible():
    """Shield item can be constructed."""
    a = Intervention.item_drop(item_type="shield", location="current_cell")
    assert a.type == InterventionType.DROP_SHIELD


def test_inventory_binary_state_supported():
    """InventoryState supports binary 0/1 shield."""
    inv = InventoryState()
    assert inv.shield == 0
    assert not inv.has_shield()
    inv.add_shield()
    assert inv.shield == 1
    assert inv.has_shield()


def test_item_drop_adds_inventory():
    """Adding shield changes inventory from 0 to 1."""
    inv = InventoryState()
    assert inv.add_shield() is True
    assert inv.shield == 1


def test_shield_consumes_once():
    """Shield consumed once returns to 0."""
    inv = InventoryState(shield=1)
    assert inv.consume_shield() is True
    assert inv.shield == 0
    assert inv.consume_shield() is False


def test_shield_reduces_risk_effect():
    """Shield risk reduction matches SHIELD_DEFAULT_RISK_REDUCTION."""
    inv = InventoryState(shield=1)
    base_risk = 0.8
    effective = base_risk * (1.0 - inv.shield_risk_reduction)
    assert effective == pytest.approx(0.4)  # 0.8 * 0.5
    assert inv.shield_risk_reduction == SHIELD_DEFAULT_RISK_REDUCTION


def test_shield_does_not_stack():
    """Adding shield when already have one returns False."""
    inv = InventoryState(shield=1)
    assert inv.add_shield() is False
    assert inv.shield == 1  # still 1, not 2


def test_item_drop_location_scope_restricted():
    """Only current_cell is allowed in Phase 8."""
    with pytest.raises(ValueError):
        Intervention.item_drop(item_type="shield", location="far_away_cell")


def test_item_drop_is_counterfactually_evaluable():
    """Item-drop counterfactual rollout runs without error."""
    gm, cfg, meta = generate_lattice_v2(seed=42, latent_mode=True)
    H, W = gm.height, gm.width
    lp = LatentCostRiskHead(d=4)
    fb = FeatureBeliefMap(H, W, d=4)
    bc = np.ones((H, W))
    pa = (gm.cell_types != CellType.WALL).astype(bool)
    rb = init_robot_belief(fb.mean, fb.var, latent_predictor=lp)
    inv = InventoryState(shield=0)

    pred = predict_agent_prefix_after_item_drop(
        rb, (2, 1), (2, W-2), bc, pa,
        inventory_state=inv, prefix_horizon=5,
    )
    assert pred.predicted_plan.action in ("UP", "DOWN", "LEFT", "RIGHT", "STAY")


def test_item_drop_read_only_in_counterfactual_mode():
    """Counterfactual item rollout does not mutate real inventory."""
    inv = InventoryState(shield=0)
    gm, cfg, meta = generate_lattice_v2(seed=42, latent_mode=True)
    H, W = gm.height, gm.width
    lp = LatentCostRiskHead(d=4)
    fb = FeatureBeliefMap(H, W, d=4)
    bc = np.ones((H, W))
    pa = (gm.cell_types != CellType.WALL).astype(bool)
    rb = init_robot_belief(fb.mean, fb.var, latent_predictor=lp)

    predict_agent_prefix_after_item_drop(
        rb, (2, 1), (2, W-2), bc, pa,
        inventory_state=inv, prefix_horizon=5,
    )
    assert inv.shield == 0  # unchanged
