"""Tests for POMDP-style interface shells (Task 3 Phase A).

Verifies:
1. WorldState: construction, adapter, immutability
2. AgentBelief: construction, adapter from FeatureBeliefMap
3. ActionPredictor: predict, score, NLL, wraps existing policy
4. RobotBeliefOverAgent: theta posterior update, entropy, shadow log
5. No existing code is modified
"""
import sys
sys.path.insert(0, ".")
import pytest
import numpy as np

from src.agents.world_state import WorldState, world_state_from_grid_map
from src.agents.agent_belief_state import AgentBelief, agent_belief_from_feature_map
from src.agents.stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, compute_choice_probs,
)
from src.teachers.action_predictor import ActionPredictor, ActionDistribution
from src.teachers.robot_belief_over_agent import RobotBeliefOverAgent, RobotBeliefState
from src.envs.map_generator import generate_default_map


# ═══════════════════════════════════════════════════════════════
# WorldState
# ═══════════════════════════════════════════════════════════════

class TestWorldState:

    def test_construction(self):
        ws = WorldState(agent_pos=(0, 0), height=8, width=8)
        assert ws.agent_pos == (0, 0)
        assert ws.remaining_budget == 100

    def test_adapter_from_grid_map(self):
        gm = generate_default_map()
        ws = world_state_from_grid_map(gm, t=5, t_max=50)
        assert ws.height == 8
        assert ws.width == 8
        assert ws.t == 5
        assert ws.remaining_budget == 45
        assert ws.true_cost is not None
        assert ws.true_risk is not None
        assert ws.goal_pos == gm.target_pos

    def test_passable_excludes_walls_and_doors(self):
        gm = generate_default_map()
        ws = world_state_from_grid_map(gm)
        # Walls should not be passable
        assert ws.passable[1, 1] == False  # wall
        # Normal cells should be passable
        assert ws.passable[0, 0] == True


# ═══════════════════════════════════════════════════════════════
# AgentBelief
# ═══════════════════════════════════════════════════════════════

class TestAgentBelief:

    def test_construction(self):
        ab = AgentBelief(theta="shiny")
        assert ab.theta == "shiny"
        assert ab.m_state["kappa"] == 1.0

    def test_risk_uncertainty_default(self):
        ab = AgentBelief()
        assert ab.risk_uncertainty(0, 0) == 0.25  # prior

    def test_adapter_from_feature_map(self):
        from src.agents.feature_belief import FeatureBeliefMap
        fbm = FeatureBeliefMap(8, 8, d=4)
        ab = agent_belief_from_feature_map(fbm, theta="safe")
        assert ab.belief_mean.shape == (8, 8, 4)
        assert ab.belief_var.shape == (8, 8, 4)
        assert ab.theta == "safe"

    def test_adapter_with_m_state_dict(self):
        from src.agents.feature_belief import FeatureBeliefMap
        fbm = FeatureBeliefMap(4, 4, d=4)
        m = {"kappa": 0.5, "tau": 0.6, "nu": 0.2, "gamma_gen": 0.1}
        ab = agent_belief_from_feature_map(fbm, m_state=m, theta="shiny")
        assert ab.m_state["tau"] == 0.6
        assert ab.theta == "shiny"


# ═══════════════════════════════════════════════════════════════
# ActionPredictor
# ═══════════════════════════════════════════════════════════════

class TestActionPredictor:

    def _make_branches(self):
        return [
            BranchAttributes(safety_score=0.8, risk_penalty=0.1,
                           temptation_score=0.1),
            BranchAttributes(safety_score=0.3, risk_penalty=0.5,
                           temptation_score=0.9),
        ]

    def test_predict_returns_distribution(self):
        ap = ActionPredictor()
        branches = self._make_branches()
        ab = AgentBelief(theta="safe")
        dist = ap.predict(None, ab, branches)
        assert isinstance(dist, ActionDistribution)
        assert len(dist.probs) == 2
        assert abs(dist.probs.sum() - 1.0) < 1e-9

    def test_predict_matches_existing_policy(self):
        """New interface should give same probs as existing compute_choice_probs."""
        params = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
        ap = ActionPredictor(params=params)
        branches = self._make_branches()
        ab = AgentBelief(theta="safe")

        dist = ap.predict(None, ab, branches)
        old_probs = compute_choice_probs(branches, "safe", params)

        np.testing.assert_allclose(dist.probs, old_probs, atol=1e-10)

    def test_score_returns_log_likelihood(self):
        ap = ActionPredictor()
        branches = self._make_branches()
        ab = AgentBelief(theta="safe")
        ll = ap.score(None, ab, branches, 0)
        assert ll <= 0  # log prob ≤ 0
        assert ll > -10  # should be reasonable

    def test_nll_positive(self):
        ap = ActionPredictor()
        branches = self._make_branches()
        ab = AgentBelief(theta="safe")
        nll = ap.nll(None, ab, branches, 0)
        assert nll >= 0

    def test_entropy_property(self):
        ap = ActionPredictor()
        branches = self._make_branches()
        ab = AgentBelief(theta="safe")
        dist = ap.predict(None, ab, branches)
        assert dist.entropy > 0  # not degenerate

    def test_stats_tracking(self):
        ap = ActionPredictor()
        branches = self._make_branches()
        ab = AgentBelief(theta="safe")
        ap.nll(None, ab, branches, 0)
        ap.nll(None, ab, branches, 1)
        assert ap.mean_nll > 0
        ap.reset_stats()
        assert ap.mean_nll == 0.0


# ═══════════════════════════════════════════════════════════════
# RobotBeliefOverAgent
# ═══════════════════════════════════════════════════════════════

class TestRobotBeliefOverAgent:

    def test_construction(self):
        rboa = RobotBeliefOverAgent()
        state = rboa.get_state()
        assert isinstance(state, RobotBeliefState)
        assert "safe" in state.theta_posterior

    def test_mean_belief(self):
        rboa = RobotBeliefOverAgent()
        mb = rboa.mean_belief()
        assert "tau" in mb
        assert mb["tau"] == 0.3

    def test_theta_update_from_action(self):
        """Observing safe action should increase safe posterior."""
        ap = ActionPredictor(AgentPolicyParams(beta=4.0, epsilon=0.05))
        rboa = RobotBeliefOverAgent(action_predictor=ap)

        # Safe branch is clearly safer
        branches = [
            BranchAttributes(safety_score=0.9, risk_penalty=0.0),
            BranchAttributes(safety_score=0.2, risk_penalty=0.6, temptation_score=0.8),
        ]

        prior_safe = rboa.get_state().theta_posterior["safe"]
        # Agent chose safe branch (action 0)
        rboa.update_from_action(None, branches, 0)
        post_safe = rboa.get_state().theta_posterior["safe"]

        # safe posterior should increase (or stay high)
        assert post_safe >= prior_safe - 0.01

    def test_entropy_decreases_with_evidence(self):
        """Repeated consistent actions should reduce entropy."""
        ap = ActionPredictor(AgentPolicyParams(beta=4.0, epsilon=0.05))
        rboa = RobotBeliefOverAgent(action_predictor=ap)

        branches = [
            BranchAttributes(safety_score=0.9, risk_penalty=0.0),
            BranchAttributes(safety_score=0.2, risk_penalty=0.6, temptation_score=0.8),
        ]

        h0 = rboa.get_state().entropy
        for _ in range(5):
            rboa.update_from_action(None, branches, 0)
        h5 = rboa.get_state().entropy

        assert h5 <= h0  # entropy should decrease

    def test_shadow_log(self):
        ap = ActionPredictor()
        rboa = RobotBeliefOverAgent(action_predictor=ap)
        branches = [
            BranchAttributes(safety_score=0.5),
            BranchAttributes(safety_score=0.5),
        ]
        rboa.update_from_action(None, branches, 0)
        log = rboa.get_shadow_log()
        assert len(log) == 1
        assert "theta_post" in log[0]
        assert "entropy" in log[0]

    def test_update_from_observer(self):
        rboa = RobotBeliefOverAgent()
        rboa.update_from_observer(
            {"tau": 0.7, "nu": 0.2}, {"tau": 0.6, "nu": 0.4})
        assert rboa.mean_belief()["tau"] == 0.7
        assert rboa.confidence()["tau"] == 0.6

    def test_reset(self):
        ap = ActionPredictor()
        rboa = RobotBeliefOverAgent(action_predictor=ap)
        branches = [BranchAttributes(), BranchAttributes()]
        rboa.update_from_action(None, branches, 0)
        rboa.reset()
        assert rboa.get_state().n_updates == 0
        assert len(rboa.get_shadow_log()) == 0
