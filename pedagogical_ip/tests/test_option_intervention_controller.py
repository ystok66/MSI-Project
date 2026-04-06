"""Tests for OptionInterventionController."""
import sys
sys.path.insert(0, ".")
import pytest
import numpy as np

from src.teachers.option_intervention_controller import (
    OptionInterventionController, OptionConfig, OptionDecision,
)


class TestOptionInterventionController:

    def _m_hat(self, nu=0.1, gamma_gen=0.05):
        return {"tau": 0.5, "nu": nu, "gamma_gen": gamma_gen,
                "gamma_spec": 0.5, "kappa": 0.3}

    def test_construction(self):
        ctrl = OptionInterventionController()
        assert ctrl.config.lambda_teach == 1.5

    def test_fork_trap_prefers_warn(self):
        """fork_trap primary lever is WARN."""
        ctrl = OptionInterventionController()
        d = ctrl.select_option(
            scenario_family="fork_trap",
            primary_intervention="WARN",
            m_hat=self._m_hat(),
            p_blind=0.5, p_timeout=0.1)
        assert d.chosen == "WARN"

    def test_hazard_belt_prefers_item_drop(self):
        """hazard_belt primary lever is ITEM_DROP."""
        # Reduce teaching advantage so base family-aligned score dominates
        cfg = OptionConfig(shield_cost=0.5, lambda_teach=0.5)
        ctrl = OptionInterventionController(config=cfg)
        d = ctrl.select_option(
            scenario_family="hazard_belt",
            primary_intervention="ITEM_DROP",
            m_hat=self._m_hat(),
            has_shield=False,
            p_blind=0.1, p_timeout=0.1)
        assert d.chosen == "ITEM_DROP"

    def test_deadline_gate_prefers_unlock(self):
        """deadline_gate primary lever is UNLOCK."""
        ctrl = OptionInterventionController()
        d = ctrl.select_option(
            scenario_family="deadline_gate",
            primary_intervention="UNLOCK",
            m_hat=self._m_hat(),
            has_locked_doors=True,
            p_timeout=0.7, p_blind=0.1)
        assert d.chosen == "UNLOCK"

    def test_inflation_penalty_reduces_overuse(self):
        """High ν̂ should penalize WARN more."""
        ctrl = OptionInterventionController()
        # Low ν
        d1 = ctrl.select_option(
            "fork_trap", "WARN", self._m_hat(nu=0.05),
            p_blind=0.3)
        ctrl.reset()
        # High ν
        d2 = ctrl.select_option(
            "fork_trap", "WARN", self._m_hat(nu=0.8),
            p_blind=0.3)
        # WARN score should be lower with high ν
        assert d2.scores["WARN"] < d1.scores["WARN"]

    def test_resource_cost_penalizes_depleted(self):
        """Already having shield should penalize ITEM_DROP."""
        ctrl = OptionInterventionController()
        d = ctrl.select_option(
            "hazard_belt", "ITEM_DROP", self._m_hat(),
            has_shield=True)  # already has shield
        assert d.scores["ITEM_DROP"] < d.scores["NONE"]

    def test_no_locked_doors_disables_unlock(self):
        ctrl = OptionInterventionController()
        d = ctrl.select_option(
            "deadline_gate", "UNLOCK", self._m_hat(),
            has_locked_doors=False)
        assert d.chosen != "UNLOCK"

    def test_none_wins_when_no_bottleneck(self):
        """Low urgency → NONE should win when teaching suppressed."""
        ctrl = OptionInterventionController(
            config=OptionConfig(lambda_teach=0.0, lambda_time=0.0,
                               lambda_infl=3.0))
        d = ctrl.select_option(
            "baseline_v2", "WARN", self._m_hat(nu=0.5, gamma_gen=0.3),
            p_blind=0.0, p_timeout=0.0)
        assert d.chosen == "NONE"

    def test_option_decision_structure(self):
        ctrl = OptionInterventionController()
        d = ctrl.select_option("fork_trap", "WARN", self._m_hat())
        assert isinstance(d, OptionDecision)
        assert d.scenario_family == "fork_trap"
        assert d.primary_lever == "WARN"
        assert len(d.scores) == 4

    def test_history_tracking(self):
        ctrl = OptionInterventionController()
        ctrl.select_option("fork_trap", "WARN", self._m_hat())
        ctrl.select_option("hazard_belt", "ITEM_DROP", self._m_hat())
        assert len(ctrl.get_history()) == 2

    def test_reset(self):
        ctrl = OptionInterventionController()
        ctrl.select_option("fork_trap", "WARN", self._m_hat())
        ctrl.reset()
        assert len(ctrl.get_history()) == 0

    def test_warn_count_escalation(self):
        """Repeated WARNs should have increasing cost."""
        ctrl = OptionInterventionController()
        d1 = ctrl.select_option("fork_trap", "WARN", self._m_hat())
        s1 = d1.resource_cost["WARN"]
        d2 = ctrl.select_option("fork_trap", "WARN", self._m_hat())
        s2 = d2.resource_cost["WARN"]
        assert s2 > s1  # escalating cost
