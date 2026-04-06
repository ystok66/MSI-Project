"""Tests for InterventionSemantics."""
import sys
sys.path.insert(0, ".")
import pytest
import numpy as np

from src.teachers.intervention_semantics import (
    WarnSemantics, UnlockSemantics, ItemDropSemantics,
    InterventionSemantics, WarnEffect, UnlockEffect, ItemDropEffect,
)


class TestWarnSemantics:
    def test_warn_does_not_change_world(self):
        ws = WarnSemantics()
        effect = ws.apply(np.array([0.5, 0.3]), np.array([0.1, 0.1]),
                         np.array([1.0, 0.0]))
        assert effect.world_changed == False

    def test_warn_updates_belief(self):
        ws = WarnSemantics(alpha_warn=0.5)
        effect = ws.apply(np.array([0.0, 0.0]), np.array([0.1, 0.1]),
                         np.array([1.0, 0.0]))
        assert np.linalg.norm(effect.belief_delta) > 0

    def test_warn_reduces_uncertainty(self):
        ws = WarnSemantics(beta_warn=0.3)
        mean = np.array([0.5, 0.3, 0.2, 0.1])
        var = np.array([0.2, 0.2, 0.2, 0.2])
        direction = np.array([1.0, 0.0, 0.0, 0.0])
        new_mean, new_var = ws.predicted_belief_after_warn(mean, var, direction)
        assert np.all(new_var < var)

    def test_warn_zero_direction(self):
        ws = WarnSemantics()
        effect = ws.apply(np.zeros(4), np.ones(4), np.zeros(4))
        assert np.allclose(effect.belief_delta, 0.0)


class TestUnlockSemantics:
    def test_unlock_changes_topology(self):
        us = UnlockSemantics()
        passable = np.ones((5, 5), dtype=bool)
        passable[2, 3] = False
        effect = us.apply(passable, [(2, 3)])
        assert effect.topology_changed == True
        assert (2, 3) in effect.cells_unlocked

    def test_unlock_does_not_change_risk(self):
        us = UnlockSemantics()
        passable = np.ones((5, 5), dtype=bool)
        passable[1, 1] = False
        effect = us.apply(passable, [(1, 1)])
        assert effect.risk_mean_changed == False

    def test_unlock_already_passable(self):
        us = UnlockSemantics()
        passable = np.ones((5, 5), dtype=bool)
        effect = us.apply(passable, [(2, 2)])
        assert len(effect.cells_unlocked) == 0

    def test_apply_to_passable(self):
        us = UnlockSemantics()
        passable = np.zeros((5, 5), dtype=bool)
        new_p = us.apply_to_passable(passable, [(1, 1), (2, 2)])
        assert new_p[1, 1] == True
        assert new_p[2, 2] == True
        assert passable[1, 1] == False  # original not modified


class TestItemDropSemantics:
    def test_item_does_not_change_belief(self):
        ids = ItemDropSemantics()
        effect = ids.apply(0.3)
        assert effect.belief_changed == False

    def test_item_does_not_change_topology(self):
        ids = ItemDropSemantics()
        effect = ids.apply(0.3)
        assert effect.world_topology_changed == False

    def test_shield_reduces_cost(self):
        ids = ItemDropSemantics(gamma_shield=0.5)
        shielded = ids.shielded_risk_cost(0.3)
        unshielded = ids.unshielded_risk_cost(0.3)
        assert shielded < unshielded

    def test_shield_reduction_ratio(self):
        ids = ItemDropSemantics(gamma_shield=0.5)
        ratio = ids.cost_reduction_ratio(0.3)
        assert 0.45 < ratio < 0.55  # should be ~0.5

    def test_zero_risk_no_reduction(self):
        ids = ItemDropSemantics()
        ratio = ids.cost_reduction_ratio(0.0)
        assert ratio == 0.0

    def test_matches_existing_shield_constant(self):
        from src.teachers.interventions import SHIELD_DEFAULT_RISK_REDUCTION
        ids = ItemDropSemantics(gamma_shield=SHIELD_DEFAULT_RISK_REDUCTION)
        assert ids.gamma_shield == 0.5


class TestInterventionSemantics:
    def test_bundle_construction(self):
        sem = InterventionSemantics()
        assert isinstance(sem.warn, WarnSemantics)
        assert isinstance(sem.unlock, UnlockSemantics)
        assert isinstance(sem.item_drop, ItemDropSemantics)
