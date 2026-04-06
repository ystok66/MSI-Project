"""
Agent Predictor — Phase 7+8.

Predicts the agent's likely behaviour from the robot's surrogate model.
Uses counterfactual rollout: runs plan_from_belief on the surrogate
to produce predicted prefix and failure modes.

Phase 8: adds ITEM_DROP counterfactual (shield in inventory).

All operations are READ-ONLY — never mutates real agent or env state.
Counterfactual item-drop uses cloned surrogate inventory only.

CRITICAL: Must NOT access hidden true trap or latent values.
Only uses robot's surrogate belief copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .robot_belief import RobotBelief, build_surrogate_predictor
from ..agents.belief_planning import (
    plan_from_belief, estimate_failure_modes,
    BeliefPlan, FailureModeEstimate,
)
from ..agents.planner_astar import plan_with_alternatives_v2


@dataclass
class AgentPrediction:
    """Robot's prediction of what the agent will do."""
    predicted_plan: BeliefPlan
    predicted_failure_modes: FailureModeEstimate
    candidate_scores: dict


def predict_agent_prefix(
    rb: RobotBelief,
    agent_pos: tuple[int, int],
    goal: tuple[int, int],
    belief_cost: np.ndarray,
    passable: np.ndarray,
    warned_cell_extra: Optional[dict] = None,
    prefix_horizon: int = 5,
    t: int = 0,
    t_max: int = 100,
    inventory_state=None,
) -> AgentPrediction:
    """Predict agent's likely plan from the robot's surrogate model.

    This is a counterfactual rollout on the surrogate — it does NOT
    touch the real agent's state or the real environment.
    """
    surrogate_lp = build_surrogate_predictor(rb)
    if surrogate_lp is None:
        raise ValueError("Cannot predict agent without latent predictor snapshot")

    # Rollout using surrogate belief + surrogate competence
    bp = plan_from_belief(
        agent_pos, goal, belief_cost, rb.agent_belief_mean,
        surrogate_lp.risk_head, passable,
        latent_predictor=surrogate_lp,
        warned_cell_extra=warned_cell_extra,
        search_budget=rb.agent_search_budget,
        prefix_horizon=prefix_horizon,
        lambda_risk=rb.agent_risk_weight,
        lambda_uncertainty=rb.agent_uncertainty_weight,
        lambda_c=rb.agent_lambda_c,
        lambda_uc=rb.agent_lambda_uc,
        lambda_ur=rb.agent_lambda_ur,
        t=t, t_max=t_max,
    )

    # Get candidate scores for failure modes
    _, _, _, cand_scores = plan_with_alternatives_v2(
        agent_pos, goal, belief_cost, rb.agent_belief_mean,
        surrogate_lp.risk_head, budget=rb.agent_search_budget,
        passable_mask=passable, warned_cell_extra_cost=warned_cell_extra,
        latent_predictor=surrogate_lp,
        lambda_risk=rb.agent_risk_weight,
        lambda_uncertainty=rb.agent_uncertainty_weight,
        lambda_c=rb.agent_lambda_c,
        lambda_uc=rb.agent_lambda_uc,
        lambda_ur=rb.agent_lambda_ur,
        inventory_state=inventory_state,
    )

    fm = estimate_failure_modes(bp, t, t_max, cand_scores)

    return AgentPrediction(
        predicted_plan=bp,
        predicted_failure_modes=fm,
        candidate_scores=cand_scores,
    )


def predict_agent_prefix_after_warn(
    rb: RobotBelief,
    agent_pos: tuple[int, int],
    goal: tuple[int, int],
    belief_cost: np.ndarray,
    passable: np.ndarray,
    warned_cell_extra: Optional[dict] = None,
    warn_extra_cost: Optional[dict] = None,
    prefix_horizon: int = 5,
    t: int = 0,
    t_max: int = 100,
) -> AgentPrediction:
    """Counterfactual: predict agent prefix IF robot sends warning.

    Simulates the effect of warning by adding warn_extra_cost to
    the surrogate's warned cells. Read-only.
    """
    merged_extra = dict(warned_cell_extra or {})
    if warn_extra_cost:
        for k, v in warn_extra_cost.items():
            merged_extra[k] = merged_extra.get(k, 0.0) + v

    return predict_agent_prefix(
        rb, agent_pos, goal, belief_cost, passable,
        warned_cell_extra=merged_extra,
        prefix_horizon=prefix_horizon, t=t, t_max=t_max,
    )


def predict_agent_prefix_after_unlock(
    rb: RobotBelief,
    agent_pos: tuple[int, int],
    goal: tuple[int, int],
    belief_cost: np.ndarray,
    passable: np.ndarray,
    unlock_cells: list[tuple[int, int]],
    warned_cell_extra: Optional[dict] = None,
    prefix_horizon: int = 5,
    t: int = 0,
    t_max: int = 100,
) -> AgentPrediction:
    """Counterfactual: predict agent prefix IF robot unlocks a door.

    Simulates unlock by setting unlock_cells to passable. Read-only.
    """
    passable_cf = passable.copy()
    belief_cost_cf = belief_cost.copy()
    for r, c in unlock_cells:
        passable_cf[r, c] = True
        belief_cost_cf[r, c] = 1.0

    return predict_agent_prefix(
        rb, agent_pos, goal, belief_cost_cf, passable_cf,
        warned_cell_extra=warned_cell_extra,
        prefix_horizon=prefix_horizon, t=t, t_max=t_max,
    )


def estimate_learning_gain(
    rb: RobotBelief,
    predicted_prefix: list[tuple[int, int]],
) -> float:
    """Heuristic learning gain if robot waits.

    Local heuristic: mean uncertainty reduction along predicted prefix
    cells — how much the agent would learn by exploring those cells.
    """
    if not predicted_prefix:
        return 0.0

    total_unc = 0.0
    for r, c in predicted_prefix:
        if 0 <= r < rb.agent_belief_var.shape[0] and 0 <= c < rb.agent_belief_var.shape[1]:
            total_unc += float(np.mean(rb.agent_belief_var[r, c]))

    return total_unc / max(len(predicted_prefix), 1)


def predict_agent_prefix_after_item_drop(
    rb: RobotBelief,
    agent_pos: tuple[int, int],
    goal: tuple[int, int],
    belief_cost: np.ndarray,
    passable: np.ndarray,
    inventory_state,
    warned_cell_extra: Optional[dict] = None,
    prefix_horizon: int = 5,
    t: int = 0,
    t_max: int = 100,
) -> AgentPrediction:
    """Counterfactual: predict agent prefix IF robot drops a shield.

    Simulates item-drop by cloning inventory and adding shield.
    Uses the CLONED inventory — never mutates real state.
    Read-only.
    """
    cf_inventory = inventory_state.clone() if inventory_state is not None else None
    if cf_inventory is not None:
        cf_inventory.add_shield()
    else:
        from ..teachers.interventions import InventoryState
        cf_inventory = InventoryState(shield=1)

    return predict_agent_prefix(
        rb, agent_pos, goal, belief_cost, passable,
        warned_cell_extra=warned_cell_extra,
        prefix_horizon=prefix_horizon, t=t, t_max=t_max,
        inventory_state=cf_inventory,
    )
