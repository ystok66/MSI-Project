"""
observation.py — Observation phase: collect n_obs frozen-learner statistics.

Runs the learner with frozen long-term parameters (no grammar/risk updates)
in immortal mode (warnings without death). Tutor only observes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import copy
import numpy as np

from ..config import FullConfig
from ..interfaces import QueryResult, Example
from ..constants import Outcome


@dataclass
class ObservationSummary:
    """Statistics collected during observation phase.

    These are the raw observations the tutor uses to initialize beliefs.
    """
    n_queries: int = 0
    n_success: int = 0
    n_death: int = 0
    n_timeout: int = 0
    total_confirms: int = 0
    total_retries: int = 0
    total_danger_selects: int = 0
    total_stuck_retries: int = 0
    # Per-query beam entropy (if computed)
    beam_entropies: List[float] = field(default_factory=list)
    # Counterfactual: how many times would danger have killed without warning
    counterfactual_death_count: int = 0
    # Per-query results for detailed analysis
    query_results: List[QueryResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.n_success / max(self.n_queries, 1)

    @property
    def death_rate(self) -> float:
        return self.n_death / max(self.n_queries, 1)

    @property
    def timeout_rate(self) -> float:
        return self.n_timeout / max(self.n_queries, 1)

    @property
    def mean_confirms(self) -> float:
        return self.total_confirms / max(self.n_queries, 1)

    @property
    def mean_retries(self) -> float:
        return self.total_retries / max(self.n_queries, 1)

    @property
    def mean_danger_selects(self) -> float:
        return self.total_danger_selects / max(self.n_queries, 1)

    @property
    def mean_beam_entropy(self) -> float:
        return float(np.mean(self.beam_entropies)) if self.beam_entropies else 0.0

    def to_dict(self) -> Dict[str, float]:
        """Flat dict for logging and belief initialization."""
        return {
            'ObsN': self.n_queries,
            'ObsSuccessRate': self.success_rate,
            'ObsDeathRate': self.death_rate,
            'ObsTimeoutRate': self.timeout_rate,
            'ObsMeanConfirms': self.mean_confirms,
            'ObsMeanRetries': self.mean_retries,
            'ObsMeanDangerSelects': self.mean_danger_selects,
            'ObsMeanBeamEntropy': self.mean_beam_entropy,
            'ObsCounterfactualDeaths': self.counterfactual_death_count,
            'ObsStuckRetries': self.total_stuck_retries,
        }


def run_observation_phase(
    env,
    obs_queries: List[Example],
    policy,
    risk_belief,
    feedback_updater,
    predictor,
    target_pred,
    rng: np.random.Generator,
    cfg: FullConfig,
) -> ObservationSummary:
    """Run observation phase with frozen learner parameters.

    The learner is run in immortal+warning mode so we always get full
    query trajectories. Long-term parameters are NOT updated.

    Args:
        env: GrammarTaskEnv with loaded task
        obs_queries: queries to observe
        policy: ColorSelectionPolicy
        risk_belief: DangerTypeBelief (will be deep-copied, not modified)
        feedback_updater: FeedbackUpdater
        predictor: CLSSequencePredictor
        target_pred: TargetPredictor
        rng: random generator
        cfg: full config

    Returns:
        ObservationSummary with collected statistics
    """
    from ..tutor_api.dummy_tutor import NoTutorImmortalWarnlike
    from ..learner.memory import QueryMemory

    # Deep-copy risk belief so observation doesn't modify teaching state
    frozen_risk = copy.deepcopy(risk_belief)

    # Use immortal warning tutor — observe but don't kill
    obs_tutor = NoTutorImmortalWarnlike()

    summary = ObservationSummary()

    for qi, query in enumerate(obs_queries):
        y_star = target_pred.predict_target(query.words)
        state = env.init_query(query, query_id=qi, target_output=y_star)
        memory = QueryMemory()

        # Run with the main loop from run_phase1 (import lazily)
        from ..experiments.run_phase1 import run_single_query

        result = run_single_query(
            env, state, policy, frozen_risk, feedback_updater,
            predictor, target_pred, obs_tutor, memory, rng, cfg,
            immortal=True,         # never die during observation
            enable_feedback=False,  # no grammar updates
        )

        # Collect statistics
        summary.n_queries += 1
        if result.outcome == Outcome.SUCCESS:
            summary.n_success += 1
        elif result.outcome == Outcome.DEATH:
            summary.n_death += 1  # shouldn't happen in immortal mode
        elif result.outcome == Outcome.TIMEOUT:
            summary.n_timeout += 1

        summary.total_confirms += result.confirm_count
        summary.total_retries += result.retry_count
        summary.total_danger_selects += result.danger_select_count
        summary.total_stuck_retries += result.stuck_retry_events

        # Count counterfactual deaths (danger selects = would have been fatal)
        summary.counterfactual_death_count += result.danger_select_count

        # Compute beam entropy for this query (if surrogate mode)
        if cfg.belief.sem_estimator in ('surrogate', 'both'):
            try:
                beam = predictor.beam_posterior(query.words)
                if beam:
                    scores = np.array([b[0] for b in beam])
                    from scipy.special import logsumexp
                    log_q = scores - logsumexp(scores)
                    q = np.exp(log_q)
                    h = -np.sum(q * log_q)  # entropy
                    summary.beam_entropies.append(float(h))
            except Exception:
                pass

        summary.query_results.append(result)

    return summary
