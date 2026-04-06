"""Tests for CompositeGoalCompatibility."""
import sys
sys.path.insert(0, ".")
import pytest
import numpy as np

from src.teachers.composite_goal_compatibility import (
    CompositeGoalCompatibility, CompatibilityConfig,
)
from src.teachers.compositional_goal_hypotheses import DEFAULT_GOAL_SPACE
from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams


@pytest.fixture
def branches():
    return [
        BranchAttributes(safety_score=0.8, risk_penalty=0.1),
        BranchAttributes(safety_score=0.2, risk_penalty=0.4, temptation_score=0.5),
    ]


class TestCompositeGoalCompatibility:

    def test_construction(self):
        cgc = CompositeGoalCompatibility()
        assert cgc is not None

    def test_observe_accumulates(self, branches):
        cgc = CompositeGoalCompatibility()
        cgc.observe(branches, observed_action=0)
        assert cgc._n_obs == 1
        cgc.observe(branches, observed_action=1)
        assert cgc._n_obs == 2

    def test_subgoal_progress_initial(self):
        cgc = CompositeGoalCompatibility()
        gh = DEFAULT_GOAL_SPACE.get("collect_red")
        assert cgc.subgoal_progress(gh) == 0.0

    def test_subgoal_progress_after_obs(self, branches):
        cgc = CompositeGoalCompatibility()
        cgc.observe(branches, observed_action=0)
        gh = DEFAULT_GOAL_SPACE.get("use_safe")
        prog = cgc.subgoal_progress(gh)
        assert isinstance(prog, float)

    def test_composite_progress_is_mean(self, branches):
        cgc = CompositeGoalCompatibility()
        for _ in range(5):
            cgc.observe(branches, observed_action=0)
        gh_comp = DEFAULT_GOAL_SPACE.get("collect_red+avoid_blue")
        gh_a = DEFAULT_GOAL_SPACE.get("collect_red")
        gh_b = DEFAULT_GOAL_SPACE.get("avoid_blue")
        p_comp = cgc.subgoal_progress(gh_comp)
        p_mean = (cgc.subgoal_progress(gh_a) + cgc.subgoal_progress(gh_b)) / 2
        assert abs(p_comp - p_mean) < 1e-6

    def test_complexity_penalty_atomic_is_zero(self):
        cgc = CompositeGoalCompatibility()
        gh = DEFAULT_GOAL_SPACE.get("collect_red")
        assert cgc.complexity_penalty(gh) == 0.0

    def test_complexity_penalty_composite_positive(self):
        cgc = CompositeGoalCompatibility()
        gh = DEFAULT_GOAL_SPACE.get("collect_red+avoid_blue")
        assert cgc.complexity_penalty(gh) > 0

    def test_redundancy_atomic_is_zero(self, branches):
        cgc = CompositeGoalCompatibility()
        gh = DEFAULT_GOAL_SPACE.get("collect_red")
        assert cgc.redundancy_score(gh, branches) == 0.0

    def test_redundancy_composite_finite(self, branches):
        cgc = CompositeGoalCompatibility()
        gh = DEFAULT_GOAL_SPACE.get("collect_red+avoid_blue")
        r = cgc.redundancy_score(gh, branches)
        assert 0.0 <= r <= 1.0

    def test_compatibility_score_runs(self, branches):
        cgc = CompositeGoalCompatibility()
        cgc.observe(branches, observed_action=0)
        gh = DEFAULT_GOAL_SPACE.get("collect_red+avoid_blue")
        s = cgc.compatibility_score(gh, branches)
        assert isinstance(s, float)

    def test_log_bonus(self, branches):
        cgc = CompositeGoalCompatibility()
        cgc.observe(branches, observed_action=0)
        gh = DEFAULT_GOAL_SPACE.get("use_safe")
        b = cgc.log_compatibility_bonus(gh, branches)
        assert isinstance(b, float)

    def test_subgoal_marginals(self):
        # Fake marginal goal
        mg = {"use_safe": 0.3, "collect_red": 0.2, "avoid_blue": 0.1,
              "reach_fast": 0.05, "collect_red+avoid_blue": 0.15,
              "collect_red+use_safe": 0.1, "avoid_blue+use_safe": 0.05,
              "reach_fast+avoid_blue": 0.05}
        cgc = CompositeGoalCompatibility()
        sm = cgc.subgoal_marginals(mg)
        # collect_red should appear in atomic + composites containing it
        assert sm["collect_red"] > mg["collect_red"]

    def test_reset(self, branches):
        cgc = CompositeGoalCompatibility()
        cgc.observe(branches, observed_action=0)
        cgc.reset()
        assert cgc._n_obs == 0
