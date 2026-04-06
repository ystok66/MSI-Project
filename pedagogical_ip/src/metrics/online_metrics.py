"""
DEPRECATED — V0 online metrics using BeliefMap. Replaced by phase9_metrics.py.

epistemic_gain()  → reimplemented as _information_gain() in phase9_metrics.py
frustration_score() → reimplemented as _frustration_proxy() in phase9_metrics.py

Uses V0 BeliefMap, not V2 FeatureBeliefMap.
This file is kept for backward compatibility and reference only.
"""

from __future__ import annotations

import numpy as np

from ..agents.belief import BeliefMap


def epistemic_gain(
    belief_before: BeliefMap,
    belief_after: BeliefMap,
) -> float:
    """
    Compute epistemic gain = reduction in total variance.

    Positive means the agent learned something.
    """
    var_before = belief_before.total_variance()
    var_after = belief_after.total_variance()
    return max(0.0, var_before - var_after)


def frustration_score(
    replan_count: int,
    time_left: int,
    max_steps: int,
    risk_budget_left: float,
    initial_risk_budget: float,
) -> float:
    """
    Heuristic frustration score in [0, 1].

    Higher when:
    - Agent has replanned many times (stuck/oscillating)
    - Time is running out
    - Risk budget is nearly exhausted
    """
    time_pressure = 1.0 - (time_left / max(max_steps, 1))
    risk_pressure = 1.0 - (risk_budget_left / max(initial_risk_budget, 0.001))
    replan_pressure = min(1.0, replan_count / 30.0)

    score = 0.4 * time_pressure + 0.3 * risk_pressure + 0.3 * replan_pressure
    return float(np.clip(score, 0.0, 1.0))
