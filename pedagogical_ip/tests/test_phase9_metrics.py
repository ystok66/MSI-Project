"""
Tests for Phase 9 metrics — unified metrics layer.
"""

import numpy as np
import pytest

from src.metrics.phase9_metrics import (
    EpisodeSummary, TransferSummary, AggregateMetrics,
    compute_episode_summary, compute_transfer_summary,
    aggregate_summaries, aggregate_transfer_summaries,
)
from src.envs.lattice_v2_runner import LatticeV2Runner


runner = LatticeV2Runner()


def _run_episode(seed=42, **kw):
    s = runner.reset(seed=seed, latent_mode=True, **kw)
    while not s.done:
        runner.step(s)
    return s


def test_task_metrics_computable():
    """success/death/timeout/cost/risk/intervention_count from episode."""
    s = _run_episode()
    summary = compute_episode_summary(s, seed=42)
    assert isinstance(summary.success, bool)
    assert isinstance(summary.death, bool)
    assert isinstance(summary.timeout, bool)
    assert summary.cumulative_cost >= 0
    assert summary.cumulative_risk >= 0
    assert summary.intervention_count >= 0


def test_learning_metrics_computable():
    """cost_prediction_error and risk_calibration_gap are floats."""
    s = _run_episode()
    summary = compute_episode_summary(s, seed=42)
    assert isinstance(summary.cost_prediction_error, float)
    assert isinstance(summary.risk_calibration_gap, float)


def test_pedagogical_metrics_computable():
    """boredom/frustration/timing/info_gain are floats."""
    s = _run_episode()
    summary = compute_episode_summary(s, seed=42)
    assert isinstance(summary.boredom_proxy, float)
    assert isinstance(summary.frustration_proxy, float)
    assert isinstance(summary.intervention_timing_quality, float)
    assert isinstance(summary.information_gain, float)


def test_online_and_transfer_metrics_separated():
    """EpisodeSummary and TransferSummary are distinct types."""
    assert EpisodeSummary != TransferSummary
    es = EpisodeSummary()
    ts = TransferSummary()
    assert hasattr(es, 'boredom_proxy')
    assert not hasattr(ts, 'boredom_proxy')


def test_episode_summary_vs_aggregate_schema_distinct():
    """Episode has success:bool, aggregate has success_rate:float."""
    es = EpisodeSummary(success=True)
    assert isinstance(es.success, bool)
    agg = AggregateMetrics(success_rate=0.5)
    assert isinstance(agg.success_rate, float)
    assert not hasattr(es, 'success_rate')
    assert not hasattr(agg, 'success')


def test_boredom_proxy_monotonic_in_low_info_high_cost():
    """Higher cost + lower info gain → higher boredom."""
    from src.metrics.phase9_metrics import _boredom_proxy
    low_boredom = _boredom_proxy(info_gain=10.0, cumulative_cost=5, steps=5)
    high_boredom = _boredom_proxy(info_gain=0.01, cumulative_cost=50, steps=10)
    assert high_boredom > low_boredom


def test_frustration_proxy_monotonic_in_failures():
    """More traps hit → higher frustration."""
    from src.metrics.phase9_metrics import _frustration_proxy
    low_frust = _frustration_proxy(uncertainty_reduction=0.5, traps_hit=0, steps=10, t_max=50)
    high_frust = _frustration_proxy(uncertainty_reduction=0.0, traps_hit=5, steps=45, t_max=50)
    assert high_frust > low_frust


def test_aggregate_outputs_mean_std_n_sem():
    """Aggregate includes mean, std, n, sem."""
    summaries = [EpisodeSummary(success=True, cumulative_cost=10),
                 EpisodeSummary(success=False, cumulative_cost=20),
                 EpisodeSummary(success=True, cumulative_cost=15)]
    agg = aggregate_summaries(summaries)
    assert agg.n == 3
    assert 0 < agg.success_rate < 1
    assert agg.success_rate_sem > 0
    assert agg.cost_mean > 0
    assert agg.cost_std >= 0


def test_metrics_schema_stable():
    """EpisodeSummary.to_dict() has expected keys."""
    es = EpisodeSummary()
    d = es.to_dict()
    required = {"success", "death", "timeout", "steps",
                "cumulative_cost", "cumulative_risk",
                "cost_prediction_error", "risk_calibration_gap",
                "boredom_proxy", "frustration_proxy",
                "intervention_timing_quality", "information_gain"}
    assert required.issubset(set(d.keys()))


def test_timing_quality_prefers_timely_intervention():
    """Early interventions get higher timing quality."""
    from src.metrics.phase9_metrics import _timing_quality
    early = _timing_quality(warnings_sent=2, first_risky_step=30, total_steps=50)
    late = _timing_quality(warnings_sent=2, first_risky_step=5, total_steps=50)
    assert early > late
