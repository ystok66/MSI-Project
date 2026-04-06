"""
Tests for unified intervention API — Phase 8.

Verifies schema constructibility, validation, and family membership.
"""

import pytest

from src.teachers.interventions import (
    InterventionType, Intervention, WARNING_VOCAB,
    ItemType, ItemEffect, InventoryState,
    MAIN_INTERVENTION_FAMILY, VALID_ITEM_LOCATIONS,
    SHIELD_DEFAULT_RISK_REDUCTION,
)


def test_wait_action_constructible():
    """WAIT can be constructed via unified API."""
    a = Intervention.wait()
    assert a.type == InterventionType.WAIT


def test_warn_action_constructible():
    """WARN(payload) can be constructed via unified API."""
    a = Intervention.warn("LEFT_RISKY")
    assert a.type == InterventionType.WARN
    assert a.param == "LEFT_RISKY"


def test_unlock_action_constructible():
    """UNLOCK(target) can be constructed via unified API."""
    a = Intervention.unlock_door("0")
    assert a.type == InterventionType.UNLOCK_DOOR


def test_item_drop_action_constructible():
    """ITEM_DROP(shield, current_cell) can be constructed via unified API."""
    a = Intervention.item_drop(item_type="shield", location="current_cell")
    assert a.type == InterventionType.DROP_SHIELD
    assert "shield" in a.param
    assert "current_cell" in a.param


def test_block_legacy_action_still_available():
    """BLOCK_PATH is available but NOT in main intervention family."""
    a = Intervention.block_path()
    assert a.type == InterventionType.BLOCK_PATH
    assert InterventionType.BLOCK_PATH not in MAIN_INTERVENTION_FAMILY


def test_intervention_enum_stable():
    """InterventionType has exactly 5 members."""
    assert len(InterventionType) == 5
    assert InterventionType.WAIT in MAIN_INTERVENTION_FAMILY
    assert InterventionType.WARN in MAIN_INTERVENTION_FAMILY
    assert InterventionType.UNLOCK_DOOR in MAIN_INTERVENTION_FAMILY
    assert InterventionType.DROP_SHIELD in MAIN_INTERVENTION_FAMILY


def test_invalid_item_type_rejected():
    """Invalid item type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown item_type"):
        Intervention.item_drop(item_type="laser_gun")


def test_invalid_location_rejected():
    """Invalid location raises ValueError."""
    with pytest.raises(ValueError, match="Invalid location"):
        Intervention.item_drop(item_type="shield", location="other_side_of_map")
