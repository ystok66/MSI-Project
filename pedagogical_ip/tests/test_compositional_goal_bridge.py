"""Tests for CompositionalGoalBridge + GoalConditionalCurriculumHook."""
import sys
sys.path.insert(0, ".")
import pytest
import numpy as np

from src.teachers.compositional_goal_bridge import (
    CompositionalGoalBridge, GoalConditionalOptionScore,
)
from src.teachers.goal_conditional_curriculum_hook import (
    GoalConditionalCurriculumHook, CurriculumConfig, CurriculumDecision,
)
from src.teachers.joint_goal_pref_posterior import JointGoalPrefPosterior, THETA_2
from src.teachers.compositional_goal_hypotheses import DEFAULT_GOAL_SPACE
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
        BranchAttributes(safety_score=0.8, risk_penalty=0.15),
        BranchAttributes(safety_score=0.2, risk_penalty=0.4, temptation_score=0.4),
    ]


@pytest.fixture
def ab():
    return AgentBelief(theta="safe")


@pytest.fixture
def posterior():
    return JointGoalPrefPosterior(pref_types=THETA_2)


class TestCompositionalGoalBridge:

    def test_construction(self, ap):
        bridge = CompositionalGoalBridge(ap)
        assert bridge is not None

    def test_update_posterior(self, ap, branches, ab, posterior):
        bridge = CompositionalGoalBridge(ap)
        ws = WorldState()
        pre_ent = posterior.entropy()
        bridge.update_posterior(posterior, ws, branches, observed_action=0, agent_belief=ab)
        post_ent = posterior.entropy()
        assert pre_ent != post_ent

    def test_score_options(self, ap, branches, ab, posterior):
        bridge = CompositionalGoalBridge(ap)
        scores = bridge.score_options(posterior, branches, ab)
        assert len(scores) == 4
        assert "NONE" in scores
        assert "WARN" in scores

    def test_score_structure(self, ap, branches, ab, posterior):
        bridge = CompositionalGoalBridge(ap)
        scores = bridge.score_options(posterior, branches, ab)
        for opt, score in scores.items():
            assert isinstance(score, GoalConditionalOptionScore)
            assert isinstance(score.expected_success_lift, float)
            assert isinstance(score.goal_weighted_lifts, dict)

    def test_best_option(self, ap, branches, ab, posterior):
        bridge = CompositionalGoalBridge(ap)
        best = bridge.best_option(posterior, branches, ab)
        assert best in ("NONE", "WARN", "UNLOCK", "ITEM_DROP")

    def test_none_has_zero_lift(self, ap, branches, ab, posterior):
        bridge = CompositionalGoalBridge(ap)
        scores = bridge.score_options(posterior, branches, ab)
        assert abs(scores["NONE"].expected_success_lift) < 1e-6


class TestGoalConditionalCurriculumHook:

    def test_construction(self, ap):
        hook = GoalConditionalCurriculumHook(ap)
        assert hook is not None

    def test_decide(self, ap, branches, ab, posterior):
        hook = GoalConditionalCurriculumHook(ap)
        ws = WorldState()
        d = hook.decide(posterior, branches, ab, ws, kappa_hat=0.5)
        assert isinstance(d, CurriculumDecision)
        assert d.chosen_option in ("NONE", "WARN", "UNLOCK", "ITEM_DROP")

    def test_scores_are_finite(self, ap, branches, ab, posterior):
        hook = GoalConditionalCurriculumHook(ap)
        ws = WorldState()
        d = hook.decide(posterior, branches, ab, ws)
        for opt, score in d.scores.items():
            assert np.isfinite(score)

    def test_kappa_bonus_is_additive(self, ap, branches, ab, posterior):
        hook = GoalConditionalCurriculumHook(ap)
        ws = WorldState()
        d0 = hook.decide(posterior, branches, ab, ws, kappa_hat=0.0)
        d1 = hook.decide(posterior, branches, ab, ws, kappa_hat=1.0)
        assert d1.kappa_bonus > d0.kappa_bonus

    def test_inflation_penalizes_high_nu(self, ap, branches, ab, posterior):
        hook = GoalConditionalCurriculumHook(ap)
        ws = WorldState()
        d_low = hook.decide(posterior, branches, ab, ws, nu_hat=0.0)
        d_high = hook.decide(posterior, branches, ab, ws, nu_hat=0.8)
        # WARN score should be lower with higher nu
        assert d_high.scores.get("WARN", 0) <= d_low.scores.get("WARN", 0) + 0.01

    def test_goal_alignment_populated(self, ap, branches, ab, posterior):
        hook = GoalConditionalCurriculumHook(ap)
        ws = WorldState()
        d = hook.decide(posterior, branches, ab, ws)
        assert len(d.goal_alignment) > 0
        for gl, opt in d.goal_alignment.items():
            assert opt in ("NONE", "WARN", "UNLOCK", "ITEM_DROP")
