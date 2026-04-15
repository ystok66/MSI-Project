"""
test_belief_update.py — Tests for tutor belief state and updates.

Covers:
  - BetaPosterior basic math
  - BSem, BRisk initialization and update
  - BType type inference
  - Observation → belief initialization
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..')))

from cls_color_selection.config import BeliefConfig
from cls_color_selection.tutor_api.tutor_state import (
    BetaPosterior, BSem, BRisk, BType, TutorBelief,
)
from cls_color_selection.tutor_api.observation import ObservationSummary
from cls_color_selection.tutor_api.belief_update import (
    initialize_belief_from_observation, update_belief_from_query_result,
    compute_timeout_risk,
)


class TestBetaPosterior:
    def test_initial_mean(self):
        b = BetaPosterior(1.0, 1.0)
        assert abs(b.mean - 0.5) < 1e-6

    def test_update_success(self):
        b = BetaPosterior(1.0, 1.0)
        b.update_success(10)
        assert b.mean > 0.5  # should shift toward 1

    def test_update_failure(self):
        b = BetaPosterior(1.0, 1.0)
        b.update_failure(10)
        assert b.mean < 0.5  # should shift toward 0

    def test_mean_after_data(self):
        b = BetaPosterior(1.0, 1.0)
        b.update_success(8)
        b.update_failure(2)
        # (1+8) / (1+8+1+2) = 9/12 = 0.75
        assert abs(b.mean - 9.0 / 12.0) < 1e-6


class TestBType:
    def test_initial_uniform(self):
        bt = BType()
        p = bt.posterior
        assert len(p) == 3
        assert all(abs(pi - 1.0 / 3) < 1e-6 for pi in p)

    def test_update_shifts_posterior(self):
        bt = BType()
        # Give high log-likelihood to 'balanced' (index 0)
        log_lik = np.array([5.0, 0.0, 0.0])
        bt.update_log_likelihood(log_lik)
        p = bt.posterior
        assert p[0] > p[1], f"balanced should dominate: {p}"
        assert p[0] > p[2], f"balanced should dominate: {p}"
        assert bt.map_type == 'balanced'

    def test_map_type_risk_averse(self):
        bt = BType()
        log_lik = np.array([0.0, 10.0, 0.0])
        bt.update_log_likelihood(log_lik)
        assert bt.map_type == 'risk_averse'


class TestTutorBelief:
    def test_from_config(self):
        cfg = BeliefConfig()
        belief = TutorBelief.from_config(cfg)
        assert belief.sem.a_probe == 0.5  # Beta(1,1) mean
        assert belief.risk.detect_rate == 0.5
        assert len(belief.type.type_names) == 3

    def test_summary_dict(self):
        belief = TutorBelief()
        d = belief.summary_dict()
        assert 'B_sem_a_probe' in d
        assert 'B_risk_detect' in d
        assert 'B_type_map' in d


class TestBeliefInitialization:
    def test_obs_success_updates_sem(self):
        cfg = BeliefConfig()
        belief = TutorBelief.from_config(cfg)

        obs = ObservationSummary(
            n_queries=4,
            n_success=3,
            n_death=0,
            n_timeout=1,
        )
        initialize_belief_from_observation(belief, obs, cfg)

        # After 3 successes and 1 failure: prior Beta(1,1) → Beta(4, 2)
        assert belief.sem.success_rate.alpha == 4.0  # 1 + 3
        assert belief.sem.success_rate.beta == 2.0   # 1 + 1
        assert belief.sem.a_probe == 4.0 / 6.0

    def test_obs_danger_updates_risk(self):
        cfg = BeliefConfig()
        belief = TutorBelief.from_config(cfg)

        obs = ObservationSummary(
            n_queries=4,
            n_success=2,
            n_death=0,
            n_timeout=2,
            total_danger_selects=3,
        )
        initialize_belief_from_observation(belief, obs, cfg)

        # 3 danger selects → detect failures, but many safe selections too
        # detect_rate should be < 1.0 (not perfect detection)
        assert belief.risk.p_detect.beta > 1.0  # prior + failures > prior
        assert belief.risk.detect_rate < 1.0    # imperfect detection


class TestTimeoutRisk:
    def test_high_competence_low_timeout(self):
        belief = TutorBelief()
        belief.sem.success_rate.alpha = 10.0
        belief.sem.success_rate.beta = 1.0  # high grammar acc

        # Mock state with many confirms left
        from cls_color_selection.environment.state import QueryState
        state = QueryState(
            query_id=0, query_words=['dax'],
            target_output=['RED'], ground_truth=['RED'],
            grammar_colors=['RED'], completion=['RED'],
            candidate_pool=[], n_confirm_max=5,
        )
        p_to = compute_timeout_risk(belief, state)
        assert p_to < 0.5  # low timeout risk when competent and filled


class TestIncrementalUpdate:
    def test_success_updates_belief(self):
        belief = TutorBelief()
        from cls_color_selection.interfaces import QueryResult
        from cls_color_selection.constants import Outcome

        result = QueryResult(
            query_id=0, query_words=['dax'],
            target_output=['RED'], ground_truth=['RED'],
            outcome=Outcome.SUCCESS,
        )
        old_a = belief.sem.success_rate.alpha
        update_belief_from_query_result(belief, result)
        assert belief.sem.success_rate.alpha == old_a + 1.0
        assert belief.n_queries_seen == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
