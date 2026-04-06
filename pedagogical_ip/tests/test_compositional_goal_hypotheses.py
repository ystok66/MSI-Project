"""Tests for CompositionalGoalHypotheses."""
import sys
sys.path.insert(0, ".")
import pytest
import numpy as np

from src.teachers.compositional_goal_hypotheses import (
    GoalHypothesisSpace, GoalHypothesis, ATOMIC_GOALS, VALID_COMPOSITES,
    DEFAULT_GOAL_SPACE, ATOMIC_ONLY_GOAL_SPACE, ATOMIC_GOAL_WEIGHTS,
)
from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams


class TestGoalHypothesisSpace:

    def test_default_space_has_8_hypotheses(self):
        """4 atomic + 4 composite = 8."""
        assert DEFAULT_GOAL_SPACE.n_goals == 8

    def test_atomic_only_space_has_4(self):
        assert ATOMIC_ONLY_GOAL_SPACE.n_goals == 4

    def test_atomic_goals_present(self):
        labels = DEFAULT_GOAL_SPACE.labels
        for a in ATOMIC_GOALS:
            assert a in labels

    def test_composite_labels(self):
        labels = DEFAULT_GOAL_SPACE.labels
        assert "collect_red+avoid_blue" in labels
        assert "collect_red+use_safe" in labels

    def test_is_composite_flag(self):
        gh_atomic = DEFAULT_GOAL_SPACE.get("collect_red")
        assert not gh_atomic.is_composite
        gh_comp = DEFAULT_GOAL_SPACE.get("collect_red+avoid_blue")
        assert gh_comp.is_composite

    def test_reward_weights_shape(self):
        for gh in DEFAULT_GOAL_SPACE.hypotheses:
            assert gh.reward_weights.shape == (4,)

    def test_composite_reward_is_average(self):
        gh = DEFAULT_GOAL_SPACE.get("collect_red+avoid_blue")
        expected = (ATOMIC_GOAL_WEIGHTS["collect_red"] +
                    ATOMIC_GOAL_WEIGHTS["avoid_blue"]) / 2
        np.testing.assert_allclose(gh.reward_weights, expected)

    def test_goal_conditioned_utility(self):
        branch = BranchAttributes(safety_score=0.8, risk_penalty=0.1,
                                  temptation_score=0.3)
        gh = DEFAULT_GOAL_SPACE.get("use_safe")
        u = DEFAULT_GOAL_SPACE.goal_conditioned_utility(branch, gh, "safe")
        assert isinstance(u, float)

    def test_choice_probs_sum_to_one(self):
        branches = [
            BranchAttributes(safety_score=0.8, risk_penalty=0.1),
            BranchAttributes(safety_score=0.3, risk_penalty=0.4),
        ]
        gh = DEFAULT_GOAL_SPACE.get("collect_red")
        probs = DEFAULT_GOAL_SPACE.compute_choice_probs(branches, gh, "shiny")
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_safe_goal_prefers_safe_branch(self):
        branches = [
            BranchAttributes(safety_score=0.9, risk_penalty=0.05),
            BranchAttributes(safety_score=0.1, risk_penalty=0.5, temptation_score=0.8),
        ]
        gh = DEFAULT_GOAL_SPACE.get("use_safe")
        probs = DEFAULT_GOAL_SPACE.compute_choice_probs(branches, gh, "safe")
        assert probs[0] > probs[1]

    def test_collect_goal_prefers_tempting_branch(self):
        branches = [
            BranchAttributes(safety_score=0.9, risk_penalty=0.05),
            BranchAttributes(safety_score=0.1, risk_penalty=0.1, temptation_score=0.8),
        ]
        gh = DEFAULT_GOAL_SPACE.get("collect_red")
        probs = DEFAULT_GOAL_SPACE.compute_choice_probs(branches, gh, "shiny")
        assert probs[1] > probs[0]

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            DEFAULT_GOAL_SPACE.get("nonexistent")
