"""Tests for InterventionRiskHead and MacroPredictiveHook."""
import sys
sys.path.insert(0, ".")
import pytest
import numpy as np

from src.agents.world_state import WorldState
from src.agents.agent_belief_state import AgentBelief
from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.teachers.action_predictor import ActionPredictor
from src.teachers.robot_belief_over_agent import RobotBeliefOverAgent
from src.teachers.intervention_risk_head import InterventionRiskHead, InterventionRisk
from src.teachers.macro_predictive_hook import MacroPredictiveHook, PredictiveScore


# ═══════════════════════════════════════════════════════════════
# InterventionRiskHead
# ═══════════════════════════════════════════════════════════════

class TestInterventionRiskHead:

    def _setup(self):
        ap = ActionPredictor(AgentPolicyParams(beta=4.0, epsilon=0.1))
        rboa = RobotBeliefOverAgent(action_predictor=ap)
        irh = InterventionRiskHead()
        branches = [
            BranchAttributes(safety_score=0.8, risk_penalty=0.1),
            BranchAttributes(safety_score=0.3, risk_penalty=0.5),
        ]
        return ap, rboa, irh, branches

    def test_construction(self):
        irh = InterventionRiskHead()
        assert irh.threshold == 0.5

    def test_predict_returns_risk(self):
        ap, rboa, irh, branches = self._setup()
        ws = WorldState(t=5, t_max=50)
        ab = AgentBelief(theta="safe")
        result = irh.predict(ws, rboa, ap, branches, ab,
                           d_commit=3, d_reveal=2, path_length_estimate=10)
        assert isinstance(result, InterventionRisk)
        assert 0 <= result.p_timeout <= 1
        assert 0 <= result.p_blind <= 1

    def test_timeout_high_when_path_exceeds_budget(self):
        ap, rboa, irh, branches = self._setup()
        ws = WorldState(t=95, t_max=100)  # only 5 steps left
        ab = AgentBelief(theta="safe")
        result = irh.predict(ws, rboa, ap, branches, ab,
                           path_length_estimate=20)  # need 20 steps
        assert result.p_timeout > 0.5

    def test_timeout_low_when_budget_ample(self):
        ap, rboa, irh, branches = self._setup()
        ws = WorldState(t=0, t_max=100)  # 100 steps left
        ab = AgentBelief(theta="safe")
        result = irh.predict(ws, rboa, ap, branches, ab,
                           path_length_estimate=5)
        assert result.p_timeout < 0.3

    def test_blind_commit_increases_with_reveal_gap(self):
        ap, rboa, irh, branches = self._setup()
        ws = WorldState(t=10, t_max=100)
        ab = AgentBelief(theta="safe")
        # d_reveal >> d_commit: high blind risk
        r1 = irh.predict(ws, rboa, ap, branches, ab,
                        d_commit=1, d_reveal=5, path_length_estimate=10)
        # d_reveal < d_commit: low blind risk
        r2 = irh.predict(ws, rboa, ap, branches, ab,
                        d_commit=5, d_reveal=1, path_length_estimate=10)
        assert r1.p_blind > r2.p_blind

    def test_urgency_combines_both(self):
        ap, rboa, irh, branches = self._setup()
        ws = WorldState(t=90, t_max=100)
        ab = AgentBelief(theta="safe")
        result = irh.predict(ws, rboa, ap, branches, ab,
                           d_commit=1, d_reveal=5, path_length_estimate=20)
        assert result.u_int > 0

    def test_flag_above_threshold(self):
        ap, rboa, irh, branches = self._setup()
        irh.threshold = 0.3
        ws = WorldState(t=95, t_max=100)
        ab = AgentBelief(theta="safe")
        result = irh.predict(ws, rboa, ap, branches, ab,
                           d_commit=1, d_reveal=5, path_length_estimate=30)
        assert result.flagged == True

    def test_history_tracking(self):
        ap, rboa, irh, branches = self._setup()
        ws = WorldState(t=10, t_max=100)
        ab = AgentBelief(theta="safe")
        irh.predict(ws, rboa, ap, branches, ab)
        irh.predict(ws, rboa, ap, branches, ab)
        assert len(irh.get_history()) == 2
        irh.reset()
        assert len(irh.get_history()) == 0


# ═══════════════════════════════════════════════════════════════
# MacroPredictiveHook
# ═══════════════════════════════════════════════════════════════

class TestMacroPredictiveHook:

    def _branches(self):
        return [
            BranchAttributes(safety_score=0.8, risk_penalty=0.1),
            BranchAttributes(safety_score=0.3, risk_penalty=0.5),
        ]

    def test_construction(self):
        hook = MacroPredictiveHook()
        assert hook.beta_pred == 0.5

    def test_predictive_gain_with_improvement(self):
        hook = MacroPredictiveHook()
        ab = AgentBelief(theta="safe")
        branches = [self._branches()]
        # Post-lesson belief with higher safety should improve action prediction
        ab_post = AgentBelief(theta="safe",
                             m_state={"kappa": 1.0, "tau": 0.8, "nu": 0.05})
        gain = hook.score_predictive_gain(
            "test_lesson", ab, branches, [0], ab_post)
        assert isinstance(gain, float)

    def test_predictive_gain_no_probes(self):
        hook = MacroPredictiveHook()
        ab = AgentBelief(theta="safe")
        gain = hook.score_predictive_gain("test", ab, [], [])
        assert gain == 0.0

    def test_rerank_shadow(self):
        hook = MacroPredictiveHook(beta_pred=1.0)
        names = ["lesson_a", "lesson_b", "lesson_c"]
        base = [0.5, 0.8, 0.3]  # b is best
        gains = [0.4, 0.0, 0.1]  # a has best gain
        results = hook.rerank_lessons_shadow(names, base, gains)
        assert len(results) == 3
        assert all(isinstance(r, PredictiveScore) for r in results)

    def test_rerank_changes_top1(self):
        """Large predictive gain should change top-1."""
        hook = MacroPredictiveHook(beta_pred=2.0)
        names = ["a", "b"]
        base = [0.5, 0.6]  # b is top-1
        gains = [0.5, 0.0]  # a has large gain
        # shadow: a=0.5+2*0.5=1.5, b=0.6+0=0.6 → a becomes top-1
        results = hook.rerank_lessons_shadow(names, base, gains)
        top1_shadow = min(results, key=lambda r: r.rank_shadow)
        assert top1_shadow.lesson_name == "a"
        assert top1_shadow.rank_changed == True

    def test_report(self):
        hook = MacroPredictiveHook(beta_pred=1.0)
        names = ["a", "b", "c"]
        base = [0.5, 0.8, 0.3]
        gains = [0.4, 0.0, 0.1]
        hook.rerank_lessons_shadow(names, base, gains)
        report = hook.get_report()
        assert report["n_calls"] == 1
        assert report["n_scores"] == 3
        assert "rank_change_rate" in report
        assert "top1_agreement" in report

    def test_reset(self):
        hook = MacroPredictiveHook()
        hook.rerank_lessons_shadow(["a"], [0.5], [0.1])
        hook.reset()
        assert hook.get_report()["n_calls"] == 0
