"""
Utility functions for evaluating teacher interventions.
"""

from __future__ import annotations

import numpy as np

from ..agents.belief import BeliefMap
from ..agents.planner_astar import bounded_astar


def pedagogical_utility(
    success_gain: float,
    learning_gain: float,
    intervention_cost: float,
    takeover_amount: float = 0.0,
    w_success: float = 1.0,
    w_learn: float = 0.3,
    w_cost: float = 0.2,
    w_takeover: float = 0.1,
) -> float:
    """
    Compute pedagogical utility for a candidate intervention.

    U = w_success * success_gain
      + w_learn  * learning_gain
      - w_cost   * intervention_cost
      - w_take   * takeover_amount
    """
    return (
        w_success * success_gain
        + w_learn * learning_gain
        - w_cost * intervention_cost
        - w_takeover * takeover_amount
    )


def estimate_success_prob(
    agent_pos: tuple[int, int],
    goal: tuple[int, int],
    belief_cost_mean: np.ndarray,
    belief_risk_mean: np.ndarray,
    belief_cost_var: np.ndarray,
    time_left: int,
    risk_budget_left: float,
    passable_mask: np.ndarray,
    budget: int = 30,
    lambda_risk: float = 3.0,
) -> float:
    """
    Estimate probability of success via a simple heuristic:
    1. Try to find a path with A*
    2. Check if path length <= time_left
    3. Check if expected cumulative risk along path <= risk_budget_left
    """
    path = bounded_astar(
        agent_pos, goal,
        belief_cost_mean, belief_risk_mean, belief_cost_var,
        budget=budget,
        lambda_risk=lambda_risk,
        passable_mask=passable_mask,
    )

    if not path or path[-1] != goal:
        return 0.1  # low but non-zero (partial info)

    path_len = len(path) - 1  # number of steps
    if path_len > time_left:
        return 0.2  # might time out

    # Estimate cumulative risk along path
    cum_risk = 0.0
    for r, c in path[1:]:  # skip start
        cum_risk += belief_risk_mean[r, c]
    if cum_risk > risk_budget_left:
        return max(0.1, 1.0 - cum_risk / (risk_budget_left + 1e-6))

    # Looks good
    time_margin = (time_left - path_len) / max(time_left, 1)
    risk_margin = max(0, risk_budget_left - cum_risk)
    return min(1.0, 0.5 + 0.3 * time_margin + 0.2 * risk_margin)


def estimate_learning_gain(
    belief_before: BeliefMap,
    belief_after: BeliefMap,
) -> float:
    """
    Estimate learning gain as reduction in total variance.
    Positive = agent learned more.
    """
    var_before = belief_before.total_variance()
    var_after = belief_after.total_variance()
    return max(0.0, var_before - var_after)
