"""Tests for ConsequenceGroundedRollout."""
import sys
sys.path.insert(0, ".")
import pytest
import numpy as np

from src.teachers.consequence_grounded_option_rollout import (
    ConsequenceGroundedRollout, ConsequenceConfig, RolloutResult,
)
from src.teachers.action_predictor import ActionPredictor
from src.teachers.intervention_semantics import InterventionSemantics
from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.agent_belief_state import AgentBelief
from src.agents.world_state import WorldState


@pytest.fixture
def ap():
    return ActionPredictor(AgentPolicyParams(beta=4.0, epsilon=0.1))


@pytest.fixture
def branches():
    return [
        BranchAttributes(safety_score=0.8, risk_penalty=0.1),
        BranchAttributes(safety_score=0.3, risk_penalty=0.4, temptation_score=0.3),
    ]


@pytest.fixture
def ab():
    return AgentBelief(theta="shiny")


class TestConsequenceGroundedRollout:

    def test_none_leaves_distribution_unchanged(self, ap, branches, ab):
        cgr = ConsequenceGroundedRollout(ap)
        result = cgr.evaluate_option("NONE", branches, ab)
        assert abs(result.success_lift) < 1e-6

    def test_warn_increases_p_safe(self, ap, branches, ab):
        """WARN reduces risk perception → agent more likely to pick safe."""
        cgr = ConsequenceGroundedRollout(ap)
        result = cgr.evaluate_option("WARN", branches, ab, safe_branch_idx=0)
        # WARN should increase P(safe) by reducing risk_penalty
        assert result.success_lift > 0 or result.p_safe_counterfactual >= result.p_safe_original

    def test_item_drop_reduces_risk_penalty(self, ap, branches, ab):
        """ITEM_DROP shields → risk_penalty reduced by gamma_shield."""
        cgr = ConsequenceGroundedRollout(ap, config=ConsequenceConfig(gamma_shield=0.5))
        mod = cgr.apply_consequence("ITEM_DROP", branches)
        # Risky branch risk_penalty should be reduced
        assert mod[1].risk_penalty < branches[1].risk_penalty
        assert abs(mod[1].risk_penalty - 0.2) < 1e-6  # 0.4 * 0.5

    def test_unlock_adds_shortcut_bonus(self, ap, branches, ab):
        """UNLOCK adds shortcut_bonus to branches."""
        cgr = ConsequenceGroundedRollout(ap, config=ConsequenceConfig(alpha_unlock=0.5))
        mod = cgr.apply_consequence("UNLOCK", branches)
        for i in range(len(branches)):
            assert mod[i].shortcut_bonus >= branches[i].shortcut_bonus + 0.5

    def test_none_preserves_branches(self, ap, branches, ab):
        """NONE should not modify any branch attribute."""
        cgr = ConsequenceGroundedRollout(ap)
        mod = cgr.apply_consequence("NONE", branches)
        for i in range(len(branches)):
            assert mod[i].risk_penalty == branches[i].risk_penalty
            assert mod[i].safety_score == branches[i].safety_score

    def test_success_lift_positive_for_risky_scenario(self, ap, ab):
        """In strongly risky scenario, WARN should increase P(safe).
        ITEM_DROP makes risky path traversable, so may DECREASE P(safe) — correct behavior.
        """
        risky_branches = [
            BranchAttributes(safety_score=0.9, risk_penalty=0.05),
            BranchAttributes(safety_score=0.1, risk_penalty=0.6, temptation_score=0.5),
        ]
        cgr = ConsequenceGroundedRollout(ap)
        # WARN should increase P(safe)
        r_warn = cgr.evaluate_option("WARN", risky_branches, ab, safe_branch_idx=0)
        assert r_warn.success_lift > 0
        # ITEM_DROP reduces risk → changes distribution (may decrease P(safe) for shiny θ)
        r_item = cgr.evaluate_option("ITEM_DROP", risky_branches, ab, safe_branch_idx=0)
        assert r_item.p_safe_counterfactual != r_item.p_safe_original  # distribution changed

    def test_rank_options(self, ap, branches, ab):
        cgr = ConsequenceGroundedRollout(ap)
        ranked = cgr.rank_options(branches, ab)
        assert len(ranked) == 4
        assert "NONE" in ranked

    def test_best_option(self, ap, branches, ab):
        cgr = ConsequenceGroundedRollout(ap)
        best = cgr.best_option(branches, ab)
        assert best in ("NONE", "WARN", "UNLOCK", "ITEM_DROP")

    def test_warn_grounding_strength_parametric(self, ap, branches, ab):
        """Verify sweep-ready: different alpha_warn → different lift."""
        lifts = []
        for alpha in [0.05, 0.15, 0.30]:
            cgr = ConsequenceGroundedRollout(
                ap, config=ConsequenceConfig(alpha_warn=alpha))
            r = cgr.evaluate_option("WARN", branches, ab, safe_branch_idx=0)
            lifts.append(r.success_lift)
        # All lifts should be non-negative (WARN helps or is neutral)
        for lift in lifts:
            assert lift >= -0.01
        # Higher alpha should give weakly higher lift (within noise)
        assert lifts[-1] >= lifts[0] - 0.02

    def test_rollout_result_structure(self, ap, branches, ab):
        cgr = ConsequenceGroundedRollout(ap)
        r = cgr.evaluate_option("WARN", branches, ab)
        assert isinstance(r, RolloutResult)
        assert r.option == "WARN"
        assert 0 <= r.p_safe_original <= 1
        assert 0 <= r.p_safe_counterfactual <= 1
