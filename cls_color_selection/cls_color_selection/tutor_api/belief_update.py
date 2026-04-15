"""
belief_update.py — Update tutor belief from observation and teaching events.

Updates B_sem, B_risk, B_type from:
  1. Observation summary (batch initialization)
  2. Per-query teaching events (incremental)
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from .tutor_state import TutorBelief, BSem, BRisk, BType
from .observation import ObservationSummary
from ..config import BeliefConfig


# ── Type-conditional likelihoods (hand-designed Gaussians) ─────

# For each type, define expected (mean, sigma) for key observation statistics
# Format: {stat_name: (μ, σ)} for each type
TYPE_PROFILES = {
    'balanced': {
        'success_rate':     (0.45, 0.20),
        'timeout_rate':     (0.40, 0.15),
        'mean_retries':     (30.0, 20.0),
        'mean_danger_selects': (1.0, 1.0),
    },
    'risk_averse': {
        'success_rate':     (0.35, 0.20),
        'timeout_rate':     (0.50, 0.15),
        'mean_retries':     (60.0, 25.0),   # more retries (over-cautious)
        'mean_danger_selects': (0.2, 0.3),  # rarely selects danger
    },
    'slow_uncertain': {
        'success_rate':     (0.25, 0.20),
        'timeout_rate':     (0.60, 0.15),
        'mean_retries':     (50.0, 25.0),
        'mean_danger_selects': (1.5, 1.5),  # more risk exposure
    },
}


def initialize_belief_from_observation(
    belief: TutorBelief,
    obs: ObservationSummary,
    cfg: BeliefConfig,
) -> TutorBelief:
    """Initialize/update belief from observation phase summary.

    Updates:
      - B_sem.success_rate with obs success/fail counts
      - B_sem.beam_entropy with observed beam entropies
      - B_risk from danger selection patterns
      - B_type from observation summary likelihood

    Args:
        belief: TutorBelief to update (modified in place)
        obs: ObservationSummary from observation phase
        cfg: BeliefConfig

    Returns:
        Updated TutorBelief
    """
    # ── B_sem: grammar competence ──
    belief.sem.success_rate.update_success(obs.n_success)
    belief.sem.success_rate.update_failure(obs.n_timeout + obs.n_death)
    belief.sem.confirm_timeout_rate = obs.timeout_rate

    # Beam entropy
    for h in obs.beam_entropies:
        belief.sem.update_beam_entropy(h)

    # ── B_risk: risk competence ──
    # Danger detection: each danger_select is a failure to avoid
    # Each safe selection is a success (assume many safe selections per query)
    n_safe_success = max(obs.n_queries * 3 - obs.total_danger_selects, 0)
    belief.risk.p_detect.update_success(n_safe_success)
    belief.risk.p_detect.update_failure(obs.total_danger_selects)
    belief.risk.n_danger_encountered += obs.total_danger_selects
    belief.risk.n_danger_avoided += 0  # in obs phase, warnings handle this

    # Over-avoidance: stuck retries suggest over-caution
    belief.risk.p_overavoid.update_success(obs.total_stuck_retries)
    belief.risk.p_overavoid.update_failure(
        max(obs.n_queries - obs.total_stuck_retries, 0))

    # ── B_type: learner type ──
    if cfg.enable_type_inference and obs.n_queries > 0:
        _update_type_from_obs(belief.type, obs)

    return belief


def _update_type_from_obs(btype: BType, obs: ObservationSummary):
    """Update type posterior from observation summary using Gaussian likelihood."""
    obs_stats = {
        'success_rate': obs.success_rate,
        'timeout_rate': obs.timeout_rate,
        'mean_retries': obs.mean_retries,
        'mean_danger_selects': obs.mean_danger_selects,
    }

    log_liks = np.zeros(len(btype.type_names))
    for t_idx, t_name in enumerate(btype.type_names):
        profile = TYPE_PROFILES.get(t_name, {})
        ll = 0.0
        for stat_name, observed in obs_stats.items():
            if stat_name in profile:
                mu, sigma = profile[stat_name]
                # Gaussian log-likelihood
                ll += -0.5 * ((observed - mu) / sigma) ** 2
                ll += -0.5 * np.log(2 * np.pi * sigma ** 2)
        log_liks[t_idx] = ll

    btype.update_log_likelihood(log_liks)


def update_belief_from_query_result(
    belief: TutorBelief,
    result,  # QueryResult
    beam_entropy: Optional[float] = None,
    had_warning: bool = False,
    had_hint: bool = False,
    had_courage: bool = False,
    danger_encountered: bool = False,
    danger_avoided: bool = False,
    safe_skipped: bool = False,
):
    """Incremental belief update from a single teaching query.

    Called after each query in the teaching phase.
    """
    from ..constants import Outcome

    belief.n_queries_seen += 1

    # ── B_sem ──
    if result.outcome == Outcome.SUCCESS:
        belief.sem.success_rate.update_success(1.0)
    else:
        belief.sem.success_rate.update_failure(1.0)

    if beam_entropy is not None:
        belief.sem.update_beam_entropy(beam_entropy)

    # ── B_risk ──
    if danger_encountered:
        belief.risk.n_danger_encountered += 1
        if danger_avoided:
            belief.risk.p_detect.update_success(1.0)
            belief.risk.n_danger_avoided += 1
        else:
            belief.risk.p_detect.update_failure(1.0)

    if safe_skipped:
        belief.risk.p_overavoid.update_success(1.0)
        belief.risk.n_safe_skipped += 1
    else:
        belief.risk.p_overavoid.update_failure(1.0)
        belief.risk.n_safe_selected += 1

    # ── Counters ──
    if had_warning:
        belief.n_warnings_issued += 1
    if had_hint:
        belief.n_hints_issued += 1
    if had_courage:
        belief.n_courage_issued += 1


def compute_timeout_risk(
    belief: TutorBelief,
    state,  # QueryState
) -> float:
    """Estimate P(timeout | current state, belief).

    Crude proxy: based on confirm count vs max and grammar success rate.
    """
    confirms_left = state.n_confirm_max - state.confirm_count
    grammar_acc = belief.sem.a_probe

    if confirms_left <= 0:
        return 1.0
    if state.is_complete:
        return 0.1  # likely to succeed on confirm

    # Rough model: P(timeout) ≈ (1 - grammar_acc)^confirms_left
    p_each_fail = max(1.0 - grammar_acc, 0.01)
    p_timeout = p_each_fail ** confirms_left
    return float(p_timeout)


def compute_death_risk(
    belief: TutorBelief,
    state,  # QueryState, with candidate_pool
    risk_belief,  # DangerTypeBelief (learner's)
) -> float:
    """Estimate P(death | WAIT) for current state.

    Based on learner's risk detection rate and pool danger density.
    """
    pool = state.candidate_pool
    if not pool:
        return 0.0

    n_danger = sum(1 for b in pool if b.is_danger)
    danger_frac = n_danger / len(pool)

    # P(learner selects danger | WAIT) ≈ danger_frac × (1 - detect_rate)
    detect = belief.risk.detect_rate
    p_death = danger_frac * (1.0 - detect)
    return float(np.clip(p_death, 0, 1))
