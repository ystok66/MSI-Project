"""Tests for GoalTemptationPosterior."""
import sys
sys.path.insert(0, ".")
import pytest
import numpy as np

from src.teachers.goal_temptation_posterior import (
    GoalTemptationPosterior, HiddenState, DEFAULT_TEMPT_GRID,
)
from src.teachers.action_predictor import ActionPredictor
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
        BranchAttributes(safety_score=0.3, risk_penalty=0.4, temptation_score=0.2),
    ]


class TestGoalTemptationPosterior:

    def test_construction(self, ap):
        gtp = GoalTemptationPosterior(action_predictor=ap)
        assert len(gtp.get_weights()) == 8  # 2 goals × 4 tempt levels

    def test_prior_sums_to_one(self, ap):
        gtp = GoalTemptationPosterior(action_predictor=ap)
        total = sum(gtp.get_weights().values())
        assert abs(total - 1.0) < 1e-6

    def test_prior_has_low_tempt_bias(self, ap):
        gtp = GoalTemptationPosterior(action_predictor=ap)
        mt = gtp.marginal_tempt()
        # Prior should favor low temptation
        assert mt[0.0] > mt[0.9]

    def test_update_changes_posterior(self, ap, branches):
        gtp = GoalTemptationPosterior(action_predictor=ap)
        ws = WorldState()
        ab = AgentBelief(theta="shiny")

        pre_entropy = gtp.entropy()
        # Agent picks risky branch (action=1) → suggests temptation
        gtp.update(ws, branches, observed_action=1, agent_belief=ab)
        post_entropy = gtp.entropy()

        # Entropy should change after informative action
        assert pre_entropy != post_entropy

    def test_posterior_sharpens_with_repeated_risky(self, ap, branches):
        gtp = GoalTemptationPosterior(action_predictor=ap)
        ws = WorldState()
        ab = AgentBelief(theta="shiny")

        entropies = [gtp.entropy()]
        for _ in range(5):
            gtp.update(ws, branches, observed_action=1, agent_belief=ab)
            entropies.append(gtp.entropy())

        # Posterior should generally sharpen with consistent evidence
        assert entropies[-1] < entropies[0]

    def test_safe_action_reduces_tempt_estimate(self, ap, branches):
        gtp = GoalTemptationPosterior(action_predictor=ap)
        ws = WorldState()
        ab = AgentBelief(theta="safe")

        # Repeatedly choosing safe branch suggests low temptation
        for _ in range(5):
            gtp.update(ws, branches, observed_action=0, agent_belief=ab)

        mt = gtp.marginal_tempt()
        # Low temptation should dominate after safe actions
        assert mt[0.0] > mt[0.9]

    def test_risky_action_increases_tempt_estimate(self, ap, branches):
        gtp = GoalTemptationPosterior(action_predictor=ap)
        ws = WorldState()
        ab = AgentBelief(theta="shiny")

        for _ in range(5):
            gtp.update(ws, branches, observed_action=1, agent_belief=ab)

        e_tempt = gtp.expected_tempt()
        # Expected temptation should be elevated after risky choices
        assert e_tempt > 0.2  # above low-tempt prior mean

    def test_marginal_goal_sums_to_one(self, ap, branches):
        gtp = GoalTemptationPosterior(action_predictor=ap)
        ws = WorldState()
        ab = AgentBelief(theta="safe")
        gtp.update(ws, branches, observed_action=0, agent_belief=ab)

        mg = gtp.marginal_goal()
        assert abs(sum(mg.values()) - 1.0) < 1e-6

    def test_map_hypothesis(self, ap):
        gtp = GoalTemptationPosterior(action_predictor=ap)
        h = gtp.map_hypothesis()
        assert isinstance(h, HiddenState)
        assert h.goal in ("true_goal", "decoy_goal")
        assert h.z_tempt in DEFAULT_TEMPT_GRID

    def test_history_tracking(self, ap, branches):
        gtp = GoalTemptationPosterior(action_predictor=ap)
        ws = WorldState()
        ab = AgentBelief(theta="safe")
        gtp.update(ws, branches, observed_action=0, agent_belief=ab)
        gtp.update(ws, branches, observed_action=1, agent_belief=ab)
        assert len(gtp.get_history()) == 2

    def test_reset(self, ap, branches):
        gtp = GoalTemptationPosterior(action_predictor=ap)
        ws = WorldState()
        ab = AgentBelief(theta="safe")
        gtp.update(ws, branches, observed_action=0, agent_belief=ab)
        gtp.reset()
        assert len(gtp.get_history()) == 0
        # Prior should be restored
        mt = gtp.marginal_tempt()
        assert mt[0.0] > mt[0.9]
