"""
Tests for Batch C: Canonical boredom surfacing to episode/aggregate metrics.
"""
import pytest
import numpy as np

from src.metrics.phase9_metrics import (
    EpisodeSummary, AggregateMetrics,
    compute_episode_summary, aggregate_summaries,
)


# ── Mock state for compute_episode_summary ──────────────────────────

class _MockState:
    """Minimal mock of V2EpisodeState for testing."""
    def __init__(self, steps=10, survived=True, reached_goal=True,
                 traps_hit=0, warn_count=1, unlock_count=0,
                 risky_entered=2, t_max=30, boredom_trace=None):
        self.steps = steps
        self.survived = survived
        self.reached_goal = reached_goal
        self.traps_hit = traps_hit
        self.warn_count = warn_count
        self.unlock_count = unlock_count
        self.risky_entered = risky_entered
        self.t_max = t_max
        self._boredom_trace = boredom_trace or []


# ══════════════════════════════════════════════════════════════════════
# 1. EpisodeSummary canonical boredom fields
# ══════════════════════════════════════════════════════════════════════

def test_episode_summary_has_canonical_boredom_fields():
    """EpisodeSummary dataclass has the canonical boredom fields."""
    es = EpisodeSummary()
    assert hasattr(es, 'boredom_canonical_mean')
    assert hasattr(es, 'boredom_canonical_max')
    assert hasattr(es, 'learning_gain_mean')
    assert hasattr(es, 'avg_prefix_cost_mean')
    # Defaults should be 0.0
    assert es.boredom_canonical_mean == 0.0
    assert es.boredom_canonical_max == 0.0


def test_aggregate_metrics_has_canonical_boredom_fields():
    """AggregateMetrics has canonical boredom statistics."""
    am = AggregateMetrics()
    assert hasattr(am, 'boredom_canonical_mean')
    assert hasattr(am, 'boredom_canonical_std')
    assert am.boredom_canonical_mean == 0.0
    assert am.boredom_canonical_std == 0.0


# ══════════════════════════════════════════════════════════════════════
# 2. compute_episode_summary reads boredom trace
# ══════════════════════════════════════════════════════════════════════

def test_episode_summary_from_boredom_trace():
    """compute_episode_summary correctly reads _boredom_trace from state."""
    trace = [
        {"boredom_penalty": 1.0, "learning_gain": 0.5, "avg_prefix_cost": 2.0},
        {"boredom_penalty": 3.0, "learning_gain": 0.1, "avg_prefix_cost": 4.0},
        {"boredom_penalty": 5.0, "learning_gain": 0.3, "avg_prefix_cost": 1.5},
    ]
    state = _MockState(boredom_trace=trace)
    es = compute_episode_summary(state, seed=42)

    assert abs(es.boredom_canonical_mean - 3.0) < 1e-6
    assert abs(es.boredom_canonical_max - 5.0) < 1e-6
    assert abs(es.learning_gain_mean - 0.3) < 1e-6
    assert abs(es.avg_prefix_cost_mean - 2.5) < 1e-6


def test_episode_summary_explicit_boredom_trace_parameter():
    """Explicit boredom_trace parameter takes precedence over state."""
    state_trace = [
        {"boredom_penalty": 100.0, "learning_gain": 100.0, "avg_prefix_cost": 100.0},
    ]
    explicit_trace = [
        {"boredom_penalty": 2.0, "learning_gain": 0.2, "avg_prefix_cost": 1.0},
    ]
    state = _MockState(boredom_trace=state_trace)
    es = compute_episode_summary(state, seed=42, boredom_trace=explicit_trace)

    # Should use explicit trace, not state trace
    assert abs(es.boredom_canonical_mean - 2.0) < 1e-6


def test_episode_summary_empty_trace_stays_zero():
    """Empty boredom trace → canonical boredom fields stay 0."""
    state = _MockState(boredom_trace=[])
    es = compute_episode_summary(state, seed=42)
    assert es.boredom_canonical_mean == 0.0
    assert es.boredom_canonical_max == 0.0
    assert es.learning_gain_mean == 0.0
    assert es.avg_prefix_cost_mean == 0.0


def test_episode_summary_no_trace_attribute():
    """State without _boredom_trace at all → graceful fallback to 0."""
    class _BareState:
        steps = 5
        survived = True
        reached_goal = True
        traps_hit = 0
        warn_count = 0
        unlock_count = 0
        risky_entered = 0
        t_max = 20

    es = compute_episode_summary(_BareState(), seed=42)
    assert es.boredom_canonical_mean == 0.0


# ══════════════════════════════════════════════════════════════════════
# 3. aggregate_summaries includes canonical boredom
# ══════════════════════════════════════════════════════════════════════

def test_aggregate_canonical_boredom():
    """aggregate_summaries computes mean and std of canonical boredom."""
    s1 = EpisodeSummary(boredom_canonical_mean=2.0)
    s2 = EpisodeSummary(boredom_canonical_mean=4.0)
    s3 = EpisodeSummary(boredom_canonical_mean=6.0)

    agg = aggregate_summaries([s1, s2, s3])
    assert abs(agg.boredom_canonical_mean - 4.0) < 1e-6
    expected_std = float(np.std([2.0, 4.0, 6.0]))
    assert abs(agg.boredom_canonical_std - expected_std) < 1e-6


def test_aggregate_single_episode_zero_std():
    """Single episode → std should be 0."""
    s1 = EpisodeSummary(boredom_canonical_mean=3.0)
    agg = aggregate_summaries([s1])
    assert agg.boredom_canonical_mean == 3.0
    assert agg.boredom_canonical_std == 0.0


# ══════════════════════════════════════════════════════════════════════
# 4. Boredom monotonicity sanity
# ══════════════════════════════════════════════════════════════════════

def test_boredom_monotonicity_low_lg_high_cost():
    """High cost + low LG → higher canonical boredom than low cost + high LG."""
    trace_boring = [
        {"boredom_penalty": 10.0, "learning_gain": 0.01, "avg_prefix_cost": 5.0},
    ] * 5
    trace_engaging = [
        {"boredom_penalty": 0.1, "learning_gain": 2.0, "avg_prefix_cost": 0.5},
    ] * 5

    es_boring = compute_episode_summary(_MockState(boredom_trace=trace_boring), seed=1)
    es_engaging = compute_episode_summary(_MockState(boredom_trace=trace_engaging), seed=2)

    assert es_boring.boredom_canonical_mean > es_engaging.boredom_canonical_mean


# ══════════════════════════════════════════════════════════════════════
# 5. Legacy proxy still works (backward compatibility)
# ══════════════════════════════════════════════════════════════════════

def test_legacy_boredom_proxy_still_populated():
    """Old boredom_proxy field is still populated for backward compatibility."""
    state = _MockState(boredom_trace=[])
    es = compute_episode_summary(state, seed=42)
    # boredom_proxy uses the legacy normalized formula — should still be a float
    assert isinstance(es.boredom_proxy, float)
