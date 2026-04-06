"""Tests for JointGoalPrefPosterior."""
import sys
sys.path.insert(0, ".")
import pytest
import numpy as np

from src.teachers.joint_goal_pref_posterior import (
    JointGoalPrefPosterior, JointHypothesis, THETA_2, THETA_K,
    DEFAULT_TEMPT_GRID, DEFAULT_TEMPT_PRIOR,
)
from src.teachers.compositional_goal_hypotheses import (
    GoalHypothesisSpace, DEFAULT_GOAL_SPACE, ATOMIC_ONLY_GOAL_SPACE,
)
from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.agent_belief_state import AgentBelief
from src.agents.world_state import WorldState


@pytest.fixture
def branches():
    return [
        BranchAttributes(safety_score=0.8, risk_penalty=0.1),
        BranchAttributes(safety_score=0.2, risk_penalty=0.4, temptation_score=0.5),
    ]


class TestJointGoalPrefPosterior:

    def test_construction_theta2(self):
        jgpp = JointGoalPrefPosterior(pref_types=THETA_2)
        # 8 goals × 2 prefs × 1 z = 16
        mg = jgpp.marginal_goal()
        assert len(mg) == 8
        mp = jgpp.marginal_pref()
        assert len(mp) == 2

    def test_construction_theta_k(self):
        jgpp = JointGoalPrefPosterior(pref_types=THETA_K)
        mp = jgpp.marginal_pref()
        assert len(mp) == 5

    def test_prior_sums_to_one(self):
        jgpp = JointGoalPrefPosterior()
        mg = jgpp.marginal_goal()
        assert abs(sum(mg.values()) - 1.0) < 1e-6

    def test_update_changes_posterior(self, branches):
        jgpp = JointGoalPrefPosterior()
        ws = WorldState()
        pre_ent = jgpp.entropy()
        jgpp.update(ws, branches, observed_action=0)
        post_ent = jgpp.entropy()
        assert pre_ent != post_ent

    def test_posterior_sharpens(self, branches):
        jgpp = JointGoalPrefPosterior(forgetting_rate=0.0)
        ws = WorldState()
        ents = [jgpp.entropy()]
        for _ in range(5):
            jgpp.update(ws, branches, observed_action=0)
            ents.append(jgpp.entropy())
        # Entropy should decrease with consistent evidence
        assert ents[-1] < ents[0]

    def test_safe_actions_recover_safe_pref(self, branches):
        jgpp = JointGoalPrefPosterior(forgetting_rate=0.0)
        ws = WorldState()
        for _ in range(8):
            jgpp.update(ws, branches, observed_action=0)
        # Safe actions → safe pref should dominate
        mp = jgpp.marginal_pref()
        assert mp["safe"] > mp["shiny"]

    def test_risky_actions_recover_shiny_pref(self, branches):
        jgpp = JointGoalPrefPosterior(forgetting_rate=0.0)
        ws = WorldState()
        for _ in range(8):
            jgpp.update(ws, branches, observed_action=1)
        mp = jgpp.marginal_pref()
        assert mp["shiny"] > mp["safe"]

    def test_marginal_goal_identifies_correct_goal(self):
        """Safe actions should favor safe-aligned goals."""
        branches = [
            BranchAttributes(safety_score=0.9, risk_penalty=0.05),
            BranchAttributes(safety_score=0.1, risk_penalty=0.5, temptation_score=0.8),
        ]
        jgpp = JointGoalPrefPosterior(forgetting_rate=0.0)
        ws = WorldState()
        for _ in range(8):
            jgpp.update(ws, branches, observed_action=0)
        mg = jgpp.marginal_goal()
        # Safe goals (use_safe, avoid_blue) should be higher than collect_red
        assert mg["use_safe"] > mg["collect_red"]

    def test_map_hypothesis(self):
        jgpp = JointGoalPrefPosterior()
        h = jgpp.map_hypothesis()
        assert isinstance(h, JointHypothesis)
        assert h.goal_label in DEFAULT_GOAL_SPACE.labels
        assert h.theta in THETA_2

    def test_goal_conditional_pref(self, branches):
        jgpp = JointGoalPrefPosterior()
        ws = WorldState()
        jgpp.update(ws, branches, observed_action=0)
        gcp = jgpp.goal_conditional_pref("use_safe")
        assert abs(sum(gcp.values()) - 1.0) < 1e-6

    def test_with_temptation(self, branches):
        jgpp = JointGoalPrefPosterior(
            tempt_grid=DEFAULT_TEMPT_GRID,
            tempt_prior=DEFAULT_TEMPT_PRIOR)
        ws = WorldState()
        jgpp.update(ws, branches, observed_action=1)
        mt = jgpp.marginal_tempt()
        assert len(mt) == 4

    def test_k_type_doesnt_collapse(self, branches):
        """K-type posterior should still be informative."""
        jgpp = JointGoalPrefPosterior(pref_types=THETA_K, forgetting_rate=0.0)
        ws = WorldState()
        for _ in range(5):
            jgpp.update(ws, branches, observed_action=0)
        mp = jgpp.marginal_pref()
        # Should not be completely uniform
        assert max(mp.values()) > 1.0 / len(THETA_K) + 0.01

    def test_history_tracking(self, branches):
        jgpp = JointGoalPrefPosterior()
        ws = WorldState()
        jgpp.update(ws, branches, observed_action=0)
        jgpp.update(ws, branches, observed_action=1)
        assert len(jgpp.get_history()) == 2

    def test_reset(self, branches):
        jgpp = JointGoalPrefPosterior()
        ws = WorldState()
        jgpp.update(ws, branches, observed_action=0)
        jgpp.reset()
        assert len(jgpp.get_history()) == 0

    def test_atomic_only_space(self, branches):
        jgpp = JointGoalPrefPosterior(goal_space=ATOMIC_ONLY_GOAL_SPACE)
        mg = jgpp.marginal_goal()
        assert len(mg) == 4  # only atomics
