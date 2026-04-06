"""
DEPRECATED — V0 metrics. Replaced by phase9_metrics.py for V2.

v1a Evaluation Metrics.

Four core metrics:
1. Zero-shot Transfer Success Rate
2. Epistemic-Cost Efficiency (ECE)
3. Counterfactual Frustration Avoidance (CFA)
4. ToM Estimation MSE

Uses V0 BeliefMap, not V2 FeatureBeliefMap.
Not imported by Phase 9 metrics, the runner, or any script.
"""

from __future__ import annotations

import numpy as np

from ..agents.belief import BeliefMap, log_det_risk_var


def zero_shot_success_rate(outcomes: list[bool]) -> float:
    """
    Zero-shot transfer success rate.

    ZS = (1/N) Σ 1[success_i]
    """
    if not outcomes:
        return 0.0
    return float(np.mean(outcomes))


def epistemic_cost_efficiency(
    belief_initial: BeliefMap,
    belief_final: BeliefMap,
    cumulative_cost: float,
    cumulative_intervention_cost: float,
    lambda_int: float = 0.5,
) -> float:
    """
    Epistemic-Cost Efficiency.

    ECE = (log|Σ₀| - log|Σ_T|) / (Σcost + λ·Σintervention_cost)

    Higher = more learning per unit cost.
    """
    log_det_0 = log_det_risk_var(belief_initial)
    log_det_T = log_det_risk_var(belief_final)
    numerator = log_det_0 - log_det_T  # positive = variance reduced
    denominator = cumulative_cost + lambda_int * cumulative_intervention_cost + 1e-6
    return float(numerator / denominator)


def counterfactual_frustration_avoidance(
    frustration_wait: list[float],
    frustration_actual: list[float],
) -> float:
    """
    Counterfactual Frustration Avoidance.

    CFA = (1/|K|) Σ_k [F(WAIT) - F(actual intervention)]

    Positive = intervention reduced frustration vs WAIT.
    """
    if not frustration_wait or not frustration_actual:
        return 0.0
    assert len(frustration_wait) == len(frustration_actual)
    diffs = [fw - fa for fw, fa in zip(frustration_wait, frustration_actual)]
    return float(np.mean(diffs))


def tom_estimation_mse(
    teacher_risk_mean: np.ndarray,
    oracle_risk_mean: np.ndarray,
) -> float:
    """
    Theory-of-Mind Estimation Error.

    ToM-MSE = (1/|S|) Σ_c (μ̂_risk(c) - μ_risk_true(c))²

    Lower = better teacher inference.
    """
    return float(np.mean((teacher_risk_mean - oracle_risk_mean) ** 2))


# ── Convenience: compute all metrics at once ─────────────────────────
def compute_episode_metrics(
    belief_initial: BeliefMap,
    belief_final: BeliefMap,
    cumulative_cost: float,
    cumulative_intervention_cost: float,
    frustration_wait_at_interventions: list[float],
    frustration_actual_at_interventions: list[float],
    teacher_risk_mean: np.ndarray | None = None,
    oracle_risk_mean: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all v1a metrics for one episode."""
    metrics = {
        "ece": epistemic_cost_efficiency(
            belief_initial, belief_final,
            cumulative_cost, cumulative_intervention_cost,
        ),
        "cfa": counterfactual_frustration_avoidance(
            frustration_wait_at_interventions,
            frustration_actual_at_interventions,
        ),
    }
    if teacher_risk_mean is not None and oracle_risk_mean is not None:
        metrics["tom_mse"] = tom_estimation_mse(teacher_risk_mean, oracle_risk_mean)
    return metrics
