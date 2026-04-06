"""Tests for ShadowBridge adapter (Task 3 Phase B/C)."""
import sys
sys.path.insert(0, ".")
import pytest
import numpy as np

from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.agent_belief_state import AgentBelief
from src.teachers.shadow_bridge import ShadowBridge, ShadowReport


class TestShadowBridge:

    def _branches(self):
        return [
            BranchAttributes(safety_score=0.8, risk_penalty=0.1,
                           temptation_score=0.1),
            BranchAttributes(safety_score=0.3, risk_penalty=0.5,
                           temptation_score=0.9),
        ]

    def test_construction(self):
        sb = ShadowBridge(theta="safe")
        assert sb.theta == "safe"

    def test_observe_step(self):
        sb = ShadowBridge(theta="safe")
        ab = AgentBelief(theta="safe")
        sb.observe_step(None, ab, self._branches(), 0)
        assert len(sb._log) == 1

    def test_old_new_nll_match(self):
        """Old and new path NLLs must be exactly equal."""
        sb = ShadowBridge(theta="safe")
        ab = AgentBelief(theta="safe")
        for _ in range(5):
            sb.observe_step(None, ab, self._branches(), 0)
        report = sb.get_report()
        assert abs(report.mean_old_nll - report.mean_new_nll) < 1e-10

    def test_nll_parity_zero(self):
        sb = ShadowBridge(theta="safe")
        ab = AgentBelief(theta="safe")
        for _ in range(5):
            sb.observe_step(None, ab, self._branches(), 0)
        report = sb.get_report()
        assert report.nll_parity < 1e-10

    def test_top1_agreement_perfect(self):
        sb = ShadowBridge(theta="safe")
        ab = AgentBelief(theta="safe")
        for _ in range(10):
            sb.observe_step(None, ab, self._branches(), 0)
        report = sb.get_report()
        assert report.top1_agreement == 1.0

    def test_brier_match(self):
        sb = ShadowBridge(theta="safe")
        ab = AgentBelief(theta="safe")
        for _ in range(5):
            sb.observe_step(None, ab, self._branches(), 1)
        report = sb.get_report()
        assert abs(report.brier_old - report.brier_new) < 1e-10

    def test_ece_match(self):
        sb = ShadowBridge(theta="safe")
        ab = AgentBelief(theta="safe")
        for _ in range(10):
            sb.observe_step(None, ab, self._branches(), 0)
        report = sb.get_report()
        assert abs(report.ece_old - report.ece_new) < 1e-10

    def test_entropy_decreases_with_consistent_actions(self):
        sb = ShadowBridge(theta="safe")
        ab = AgentBelief(theta="safe")
        branches = [
            BranchAttributes(safety_score=0.9, risk_penalty=0.0),
            BranchAttributes(safety_score=0.2, risk_penalty=0.6,
                           temptation_score=0.8),
        ]
        sb.observe_step(None, ab, branches, 0)
        h0 = sb.belief_tracker.get_state().entropy
        for _ in range(10):
            sb.observe_step(None, ab, branches, 0)
        h_final = sb.belief_tracker.get_state().entropy
        assert h_final <= h0

    def test_theta_map_recovery(self):
        """After several safe actions, θ_MAP should be 'safe'."""
        sb = ShadowBridge(theta="safe")
        ab = AgentBelief(theta="safe")
        branches = [
            BranchAttributes(safety_score=0.9, risk_penalty=0.0),
            BranchAttributes(safety_score=0.2, risk_penalty=0.6,
                           temptation_score=0.8),
        ]
        for _ in range(15):
            sb.observe_step(None, ab, branches, 0)  # always pick safe
        report = sb.get_report()
        theta_map = max(report.final_theta_posterior,
                       key=report.final_theta_posterior.get)
        assert theta_map == "safe"

    def test_reset(self):
        sb = ShadowBridge(theta="safe")
        ab = AgentBelief(theta="safe")
        sb.observe_step(None, ab, self._branches(), 0)
        sb.reset()
        report = sb.get_report()
        assert report.n_steps == 0

    def test_report_structure(self):
        sb = ShadowBridge(theta="safe")
        ab = AgentBelief(theta="safe")
        sb.observe_step(None, ab, self._branches(), 0)
        report = sb.get_report()
        assert isinstance(report, ShadowReport)
        assert report.n_steps == 1
        assert report.mean_old_nll > 0
        assert report.final_theta_posterior is not None
