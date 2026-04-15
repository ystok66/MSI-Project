"""
Phase 9 Metrics — unified episode-level and aggregate-level metrics.

Three metric groups:
1. Task metrics  — success/death/timeout/cost/risk/interventions
2. Learning metrics — calibration, cost error, uncertainty reduction
3. Pedagogical metrics — boredom, frustration, timing quality, info gain

IMPORTANT:
- Episode-level fields are raw values (success: bool, cost: float)
- Aggregate-level fields are statistics (success_rate: float, cost_mean: float)
- These two schemas are NEVER mixed.

All computations are POST-EPISODE only — never called inside step().
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# ── Episode-Level Summary ───────────────────────────────────────────


@dataclass
class EpisodeSummary:
    """Per-episode metrics. Raw values, not rates."""
    # Identity
    seed: int = 0
    agent_level: str = ""
    teacher_condition: str = ""
    env_condition: str = ""

    # Task metrics (raw booleans / counts)
    success: bool = False
    death: bool = False
    timeout: bool = False
    steps: int = 0
    cumulative_cost: float = 0.0
    cumulative_risk: float = 0.0
    intervention_count: int = 0
    intervention_types_used: list = field(default_factory=list)

    # Learning metrics
    cost_prediction_error: float = 0.0    # MAE on visited cells
    risk_calibration_gap: float = 0.0     # |mean(risk_hat) - mean(true_risk)|
    uncertainty_reduction_visited: float = 0.0
    uncertainty_reduction_nearby: float = 0.0

    # Pedagogical metrics (heuristic proxies)
    information_gain: float = 0.0
    boredom_proxy: float = 0.0            # legacy normalized proxy [0,1]
    frustration_proxy: float = 0.0
    intervention_timing_quality: float = 0.0

    # Canonical boredom (from decision-layer B_wait = avg_cost / (ε + LG))
    boredom_canonical_mean: float = 0.0   # mean B_wait over tutor-active steps
    boredom_canonical_max: float = 0.0    # max B_wait spike
    learning_gain_mean: float = 0.0       # mean LG constituent
    avg_prefix_cost_mean: float = 0.0     # mean avg_cost constituent

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TransferSummary:
    """Per-episode transfer metrics. Separate from online."""
    seed: int = 0
    agent_level: str = ""
    teacher_condition: str = ""   # what tutor was used DURING TRAINING
    env_condition: str = ""

    # Task metrics (no-tutor performance)
    success: bool = False
    death: bool = False
    timeout: bool = False
    steps: int = 0
    cumulative_cost: float = 0.0
    cumulative_risk: float = 0.0

    # Learning retention
    cost_prediction_error: float = 0.0
    risk_calibration_gap: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ── Aggregate-Level Summary ─────────────────────────────────────────


@dataclass
class AggregateMetrics:
    """Aggregated statistics over multiple episodes. mean/std/n/sem."""
    agent_level: str = ""
    teacher_condition: str = ""
    env_condition: str = ""
    n: int = 0

    # Task statistics
    success_rate: float = 0.0
    success_rate_sem: float = 0.0
    death_rate: float = 0.0
    timeout_rate: float = 0.0
    cost_mean: float = 0.0
    cost_std: float = 0.0
    risk_mean: float = 0.0
    risk_std: float = 0.0
    intervention_count_mean: float = 0.0

    # Learning statistics
    cost_error_mean: float = 0.0
    cost_error_std: float = 0.0
    calibration_gap_mean: float = 0.0
    calibration_gap_std: float = 0.0
    uncertainty_reduction_mean: float = 0.0

    # Pedagogical statistics
    info_gain_mean: float = 0.0
    boredom_mean: float = 0.0             # legacy normalized proxy
    frustration_mean: float = 0.0
    timing_quality_mean: float = 0.0

    # Canonical boredom statistics
    boredom_canonical_mean: float = 0.0
    boredom_canonical_std: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ── Computation Functions ───────────────────────────────────────────


def compute_episode_summary(
    state,
    feature_belief_before: Optional[np.ndarray] = None,
    feature_belief_after: Optional[np.ndarray] = None,
    true_features: Optional[np.ndarray] = None,
    true_risk: Optional[np.ndarray] = None,
    visited_cells: Optional[set] = None,
    observed_cells: Optional[set] = None,
    intervention_log: Optional[list] = None,
    first_risky_step: Optional[int] = None,
    boredom_trace: Optional[list] = None,
    seed: int = 0,
    agent_level: str = "",
    teacher_condition: str = "",
    env_condition: str = "",
) -> EpisodeSummary:
    """Compute unified episode summary from V2EpisodeState.

    Called AFTER episode completes. Never called inside step().
    """
    s = state
    summary = EpisodeSummary(
        seed=seed,
        agent_level=agent_level,
        teacher_condition=teacher_condition,
        env_condition=env_condition,
        success=s.reached_goal and s.survived,
        death=not s.survived,
        timeout=not s.reached_goal and s.survived,
        steps=s.steps,
        cumulative_cost=float(s.steps),  # each step costs 1 in current env
        cumulative_risk=float(s.risky_entered),
        intervention_count=s.warn_count + s.unlock_count,
    )

    # Intervention types used
    if intervention_log:
        summary.intervention_types_used = list(set(intervention_log))

    # Learning metrics — only if belief arrays provided
    if (feature_belief_before is not None and feature_belief_after is not None
            and visited_cells):
        summary.uncertainty_reduction_visited = _uncertainty_reduction(
            feature_belief_before, feature_belief_after, visited_cells)

    if observed_cells and feature_belief_before is not None and feature_belief_after is not None:
        summary.uncertainty_reduction_nearby = _uncertainty_reduction(
            feature_belief_before, feature_belief_after, observed_cells)

    if true_features is not None and feature_belief_after is not None and visited_cells:
        summary.cost_prediction_error = _cost_prediction_error(
            feature_belief_after, true_features, visited_cells, s)

    if true_risk is not None and feature_belief_after is not None and visited_cells:
        summary.risk_calibration_gap = _risk_calibration_gap(
            feature_belief_after, true_risk, visited_cells, s)

    # Pedagogical metrics
    if feature_belief_before is not None and feature_belief_after is not None:
        summary.information_gain = _information_gain(
            feature_belief_before, feature_belief_after)

    summary.boredom_proxy = _boredom_proxy(
        summary.information_gain, summary.cumulative_cost, s.steps)
    summary.frustration_proxy = _frustration_proxy(
        summary.uncertainty_reduction_visited, s.traps_hit, s.steps, s.t_max)
    summary.intervention_timing_quality = _timing_quality(
        s.warn_count, first_risky_step, s.steps)

    # Canonical boredom from accumulated decision-layer trace
    _bt = boredom_trace
    if _bt is None:
        _bt = getattr(s, '_boredom_trace', None)
    if _bt:
        bp = [d["boredom_penalty"] for d in _bt]
        summary.boredom_canonical_mean = float(np.mean(bp))
        summary.boredom_canonical_max = float(np.max(bp))
        summary.learning_gain_mean = float(np.mean(
            [d["learning_gain"] for d in _bt]))
        summary.avg_prefix_cost_mean = float(np.mean(
            [d["avg_prefix_cost"] for d in _bt]))

    return summary


def compute_transfer_summary(
    state,
    seed: int = 0,
    agent_level: str = "",
    teacher_condition: str = "",
    env_condition: str = "",
    true_features: Optional[np.ndarray] = None,
    true_risk: Optional[np.ndarray] = None,
    feature_belief_after: Optional[np.ndarray] = None,
    visited_cells: Optional[set] = None,
) -> TransferSummary:
    """Compute transfer summary. Called AFTER no-tutor episode."""
    s = state
    ts = TransferSummary(
        seed=seed,
        agent_level=agent_level,
        teacher_condition=teacher_condition,
        env_condition=env_condition,
        success=s.reached_goal and s.survived,
        death=not s.survived,
        timeout=not s.reached_goal and s.survived,
        steps=s.steps,
        cumulative_cost=float(s.steps),
        cumulative_risk=float(s.risky_entered),
    )

    if true_features is not None and feature_belief_after is not None and visited_cells:
        ts.cost_prediction_error = _cost_prediction_error(
            feature_belief_after, true_features, visited_cells, s)
    if true_risk is not None and feature_belief_after is not None and visited_cells:
        ts.risk_calibration_gap = _risk_calibration_gap(
            feature_belief_after, true_risk, visited_cells, s)

    return ts


def aggregate_summaries(
    summaries: list[EpisodeSummary],
    agent_level: str = "",
    teacher_condition: str = "",
    env_condition: str = "",
) -> AggregateMetrics:
    """Aggregate episode summaries into mean/std/n/sem statistics."""
    n = len(summaries)
    if n == 0:
        return AggregateMetrics(
            agent_level=agent_level,
            teacher_condition=teacher_condition,
            env_condition=env_condition,
        )

    successes = [s.success for s in summaries]
    sr = float(np.mean(successes))

    return AggregateMetrics(
        agent_level=agent_level,
        teacher_condition=teacher_condition,
        env_condition=env_condition,
        n=n,
        success_rate=sr,
        success_rate_sem=float(np.std(successes) / np.sqrt(n)) if n > 1 else 0.0,
        death_rate=float(np.mean([s.death for s in summaries])),
        timeout_rate=float(np.mean([s.timeout for s in summaries])),
        cost_mean=float(np.mean([s.cumulative_cost for s in summaries])),
        cost_std=float(np.std([s.cumulative_cost for s in summaries])),
        risk_mean=float(np.mean([s.cumulative_risk for s in summaries])),
        risk_std=float(np.std([s.cumulative_risk for s in summaries])),
        intervention_count_mean=float(np.mean([s.intervention_count for s in summaries])),
        cost_error_mean=float(np.mean([s.cost_prediction_error for s in summaries])),
        cost_error_std=float(np.std([s.cost_prediction_error for s in summaries])),
        calibration_gap_mean=float(np.mean([s.risk_calibration_gap for s in summaries])),
        calibration_gap_std=float(np.std([s.risk_calibration_gap for s in summaries])),
        uncertainty_reduction_mean=float(np.mean(
            [s.uncertainty_reduction_visited for s in summaries])),
        info_gain_mean=float(np.mean([s.information_gain for s in summaries])),
        boredom_mean=float(np.mean([s.boredom_proxy for s in summaries])),
        frustration_mean=float(np.mean([s.frustration_proxy for s in summaries])),
        timing_quality_mean=float(np.mean(
            [s.intervention_timing_quality for s in summaries])),
        boredom_canonical_mean=float(np.mean(
            [s.boredom_canonical_mean for s in summaries])),
        boredom_canonical_std=float(np.std(
            [s.boredom_canonical_mean for s in summaries])),
    )


def aggregate_transfer_summaries(
    summaries: list[TransferSummary],
    agent_level: str = "",
    teacher_condition: str = "",
    env_condition: str = "",
) -> AggregateMetrics:
    """Aggregate transfer summaries into statistics."""
    n = len(summaries)
    if n == 0:
        return AggregateMetrics(
            agent_level=agent_level,
            teacher_condition=teacher_condition,
            env_condition=env_condition,
        )

    successes = [s.success for s in summaries]
    sr = float(np.mean(successes))

    return AggregateMetrics(
        agent_level=agent_level,
        teacher_condition=teacher_condition,
        env_condition=env_condition,
        n=n,
        success_rate=sr,
        success_rate_sem=float(np.std(successes) / np.sqrt(n)) if n > 1 else 0.0,
        death_rate=float(np.mean([s.death for s in summaries])),
        timeout_rate=float(np.mean([s.timeout for s in summaries])),
        cost_mean=float(np.mean([s.cumulative_cost for s in summaries])),
        cost_std=float(np.std([s.cumulative_cost for s in summaries])),
        risk_mean=float(np.mean([s.cumulative_risk for s in summaries])),
        risk_std=float(np.std([s.cumulative_risk for s in summaries])),
        cost_error_mean=float(np.mean([s.cost_prediction_error for s in summaries])),
        cost_error_std=float(np.std([s.cost_prediction_error for s in summaries])),
        calibration_gap_mean=float(np.mean([s.risk_calibration_gap for s in summaries])),
        calibration_gap_std=float(np.std([s.risk_calibration_gap for s in summaries])),
    )


# ── Internal helpers ────────────────────────────────────────────────


def _uncertainty_reduction(
    belief_before: np.ndarray,
    belief_after: np.ndarray,
    cells: set,
) -> float:
    """Mean variance reduction over a set of cells."""
    if not cells:
        return 0.0
    total = 0.0
    for r, c in cells:
        if 0 <= r < belief_before.shape[0] and 0 <= c < belief_before.shape[1]:
            var_b = float(np.mean(np.abs(belief_before[r, c])))
            var_a = float(np.mean(np.abs(belief_after[r, c])))
            total += max(0.0, var_b - var_a)
    return total / max(len(cells), 1)


def _cost_prediction_error(
    belief_after: np.ndarray,
    true_features: np.ndarray,
    visited_cells: set,
    state,
) -> float:
    """MAE of predicted cost vs true cost on visited cells."""
    if not visited_cells:
        return 0.0
    errors = []
    lp = getattr(state, 'latent_predictor', None)
    for r, c in visited_cells:
        if 0 <= r < belief_after.shape[0] and 0 <= c < belief_after.shape[1]:
            if lp is not None:
                cost_hat = lp.predict_cost(belief_after[r, c])
                cost_true = lp.predict_cost(true_features[r, c])
                errors.append(abs(float(cost_hat) - float(cost_true)))
            else:
                errors.append(0.0)
    return float(np.mean(errors)) if errors else 0.0


def _risk_calibration_gap(
    belief_after: np.ndarray,
    true_risk: np.ndarray,
    visited_cells: set,
    state,
) -> float:
    """|mean(risk_hat) - mean(true_risk)| over visited cells.

    Named 'calibration_gap', not ECE. Simple mean-level check.
    """
    if not visited_cells:
        return 0.0
    risk_hats = []
    risk_trues = []
    lp = getattr(state, 'latent_predictor', None)
    for r, c in visited_cells:
        if 0 <= r < belief_after.shape[0] and 0 <= c < belief_after.shape[1]:
            if lp is not None:
                rh = float(lp.predict_risk(belief_after[r, c]))
                risk_hats.append(rh)
            risk_trues.append(float(true_risk[r, c]))
    if not risk_hats:
        return 0.0
    return abs(float(np.mean(risk_hats)) - float(np.mean(risk_trues)))


def _information_gain(
    belief_before: np.ndarray,
    belief_after: np.ndarray,
) -> float:
    """Total variance reduction (scalar)."""
    var_b = float(np.sum(np.var(belief_before, axis=-1)))
    var_a = float(np.sum(np.var(belief_after, axis=-1)))
    return max(0.0, var_b - var_a)


def _boredom_proxy(
    info_gain: float,
    cumulative_cost: float,
    steps: int,
) -> float:
    """Heuristic boredom: low info gain + high cost per step."""
    if steps == 0:
        return 0.0
    cost_per_step = cumulative_cost / max(steps, 1)
    # Normalize: boredom grows when info_gain is low and cost is high
    ig_component = 1.0 / (1.0 + info_gain)         # high when gain is low
    cost_component = min(1.0, cost_per_step / 5.0)  # high when costly
    return float(np.clip(0.5 * ig_component + 0.5 * cost_component, 0.0, 1.0))


def _frustration_proxy(
    uncertainty_reduction: float,
    traps_hit: int,
    steps: int,
    t_max: int,
) -> float:
    """Heuristic frustration: high uncertainty + failures + time pressure."""
    if steps == 0:
        return 0.0
    trap_rate = min(1.0, traps_hit / max(steps, 1) * 10.0)
    time_pressure = max(0.0, 1.0 - (t_max - steps) / max(t_max, 1))
    low_learning = 1.0 / (1.0 + uncertainty_reduction * 10.0)
    return float(np.clip(
        0.3 * trap_rate + 0.3 * time_pressure + 0.4 * low_learning, 0.0, 1.0))


def _timing_quality(
    warnings_sent: int,
    first_risky_step: Optional[int],
    total_steps: int,
) -> float:
    """Fraction of interventions that came before first risky entry.

    1.0 = all interventions were timely (before danger).
    0.0 = no interventions or all after danger.
    """
    if warnings_sent == 0:
        return 0.0  # no interventions to judge
    if first_risky_step is None or first_risky_step == 0:
        return 0.5  # ambiguous
    # Simple proxy: did warnings come early enough?
    # If first risky entry was late relative to total, interventions were timely
    timing = first_risky_step / max(total_steps, 1)
    return float(np.clip(timing, 0.0, 1.0))
