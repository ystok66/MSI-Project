"""
Belief-Conditioned Bounded Planning — Phase 6.

This module implements belief-conditioned bounded planning.
NOT a full belief-tree planner. NOT an exact POMDP solver.

The agent plans on its MAP / expected world using bounded A*.
What changes versus Phase 5 is:
- Structured planning result (BeliefPlan)
- Action confidence from path-level runner-up gap
- Structured failure-mode estimates
- Explicit score breakdown and dominant reason

All diagnostics are READ-ONLY. They do not alter planner, belief, or env state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .prefix_prediction import compute_prefix_predictions, PrefixPrediction


# ── Structured Outputs ──────────────────────────────────────────────

DOMINANT_REASONS = frozenset({
    "lower_risk", "lower_cost", "lower_uncertainty",
    "deadline_pressure", "mixed",
})


@dataclass
class ScoreBreakdown:
    """Per-component breakdown of a path's planning score."""
    cost_term: float
    risk_term: float
    uncertainty_term: float


@dataclass
class BeliefPlan:
    """Structured planning result from belief-conditioned bounded planner."""
    action: str
    next_pos: tuple[int, int]
    planned_prefix: list[tuple[int, int]]
    full_path: list[tuple[int, int]]
    # Aggregate scores over prefix
    expected_cost: float
    expected_risk: float          # cumulative independence approx
    uncertainty: float            # mean uncertainty over prefix
    # Confidence
    runner_up_gap: float          # path-level score difference (best vs 2nd)
    action_confidence: float      # normalized: gap / (gap + temperature)
    # Reasoning
    dominant_reason: str          # from DOMINANT_REASONS
    score_breakdown: ScoreBreakdown
    # Optional prefix prediction
    prefix_prediction: Optional[PrefixPrediction] = None


@dataclass
class FailureModeEstimate:
    """Heuristic failure-mode analysis over the chosen prefix.

    All scores are prefix-based, not global.
    """
    high_cumulative_risk: float     # cumulative risk along prefix
    high_uncertainty: float         # mean uncertainty along prefix
    deadline_miss: float            # 0-1: time pressure vs path length
    no_safe_route: float            # 0-1: fraction of candidates with high risk
    warning_insufficient: float     # 0-1: risky prefix cells not influenced by warning


# ── Core Functions ──────────────────────────────────────────────────

def plan_from_belief(
    agent_pos: tuple[int, int],
    goal: tuple[int, int],
    belief_cost: np.ndarray,           # (H, W)
    feature_belief_mean: np.ndarray,   # (H, W, d)
    risk_model,                        # BayesianRiskHead
    passable: np.ndarray,              # (H, W) bool
    latent_predictor=None,             # LatentCostRiskHead
    warned_cell_extra: Optional[dict] = None,
    search_budget: int = 30,
    prefix_horizon: int = 5,
    confidence_temperature: float = 1.0,
    lambda_risk: float = 3.0,
    lambda_uncertainty: float = 0.5,
    lambda_c: float = 1.0,
    lambda_uc: float = 0.1,
    lambda_ur: float = 0.1,
    t: int = 0,
    t_max: int = 100,
    risk_threshold: float = 0.3,
    feature_belief_var: Optional[np.ndarray] = None,
    route_necessity: float = 0.0,
    inventory_state=None,              # InventoryState or None
) -> BeliefPlan:
    """Plan from belief using bounded A*, returning structured diagnostics.

    This is a belief-conditioned bounded planner:
    - Plans on MAP/expected world (not belief tree)
    - Computes path-level alternative scores for confidence
    - Produces structured score breakdown and dominant reason
    - All computations are read-only
    """
    from .planner_astar import plan_with_alternatives_v2

    # Plan with path-level alternatives
    best_action, best_next, best_path, candidate_scores = plan_with_alternatives_v2(
        agent_pos, goal, belief_cost, feature_belief_mean,
        risk_model, budget=search_budget, passable_mask=passable,
        warned_cell_extra_cost=warned_cell_extra,
        latent_predictor=latent_predictor,
        lambda_risk=lambda_risk, lambda_uncertainty=lambda_uncertainty,
        lambda_c=lambda_c, lambda_uc=lambda_uc, lambda_ur=lambda_ur,
        inventory_state=inventory_state,
        feature_belief_var=feature_belief_var,
        route_necessity=route_necessity,
    )

    # Compute prefix prediction if latent predictor available
    prefix_pred = None
    if latent_predictor is not None and best_path:
        prefix_pred = compute_prefix_predictions(
            best_path, feature_belief_mean, latent_predictor,
            horizon=prefix_horizon, risk_threshold=risk_threshold,
        )

    # Score breakdown from prefix
    cost_term = prefix_pred.cumulative_cost if prefix_pred else 0.0
    risk_term = prefix_pred.cumulative_risk if prefix_pred else 0.0
    unc_term = (
        float(np.mean(prefix_pred.cost_uncertainties + prefix_pred.risk_uncertainties))
        if prefix_pred and prefix_pred.cost_uncertainties else 0.0
    )

    breakdown = ScoreBreakdown(
        cost_term=cost_term,
        risk_term=risk_term,
        uncertainty_term=unc_term,
    )

    # Runner-up gap from path-level candidate scores
    sorted_scores = sorted(candidate_scores.values())
    if len(sorted_scores) >= 2:
        runner_up_gap = sorted_scores[1] - sorted_scores[0]
    else:
        runner_up_gap = 0.0

    # Confidence normalized by temperature
    action_confidence = runner_up_gap / (runner_up_gap + confidence_temperature) if (runner_up_gap + confidence_temperature) > 0 else 0.0

    # Dominant reason
    dominant_reason = _compute_dominant_reason(
        candidate_scores, best_action, feature_belief_mean,
        latent_predictor, passable, t, t_max, best_path, goal,
    )

    prefix_cells = best_path[1:prefix_horizon + 1] if len(best_path) > 1 else []

    return BeliefPlan(
        action=best_action,
        next_pos=best_next,
        planned_prefix=prefix_cells,
        full_path=best_path,
        expected_cost=cost_term,
        expected_risk=risk_term,
        uncertainty=unc_term,
        runner_up_gap=runner_up_gap,
        action_confidence=action_confidence,
        dominant_reason=dominant_reason,
        score_breakdown=breakdown,
        prefix_prediction=prefix_pred,
    )


def estimate_failure_modes(
    belief_plan: BeliefPlan,
    t: int,
    t_max: int,
    candidate_scores: dict,
    warned_cells: Optional[set] = None,
    risk_threshold: float = 0.3,
) -> FailureModeEstimate:
    """Heuristic failure-mode analysis based on prefix diagnostics.

    All scores are prefix-based, not global. Read-only.
    """
    pp = belief_plan.prefix_prediction

    # 1. high_cumulative_risk: directly from prefix
    high_risk = pp.cumulative_risk if pp else 0.0

    # 2. high_uncertainty: mean over prefix
    high_unc = belief_plan.uncertainty

    # 3. deadline_miss: remaining time vs estimated path length
    path_len = len(belief_plan.full_path)
    remaining = max(t_max - t, 1)
    deadline_miss = max(0.0, min(1.0, (path_len - remaining) / max(remaining, 1)))

    # 4. no_safe_route: fraction of candidate paths with high total score
    if candidate_scores:
        scores = list(candidate_scores.values())
        median_score = float(np.median(scores))
        # If best score is still very high, no safe route
        best_score = min(scores)
        no_safe = min(1.0, best_score / (median_score + 1e-6)) if median_score > 0 else 0.0
        # Alternative: fraction of candidates above a threshold
        high_score_threshold = best_score * 2.0
        frac_bad = sum(1 for s in scores if s > high_score_threshold) / max(len(scores), 1)
        no_safe = frac_bad
    else:
        no_safe = 0.0

    # 5. warning_insufficient: risky prefix cells not influenced by warning
    if pp and pp.risky_prefix_cells:
        warned = warned_cells or set()
        risky_not_warned = [c for c in pp.risky_prefix_cells if c not in warned]
        warning_insuff = len(risky_not_warned) / max(len(pp.risky_prefix_cells), 1)
    else:
        warning_insuff = 0.0

    return FailureModeEstimate(
        high_cumulative_risk=high_risk,
        high_uncertainty=high_unc,
        deadline_miss=deadline_miss,
        no_safe_route=no_safe,
        warning_insufficient=warning_insuff,
    )


def _compute_dominant_reason(
    candidate_scores: dict,
    best_action: str,
    feature_belief_mean: np.ndarray,
    latent_predictor,
    passable: np.ndarray,
    t: int,
    t_max: int,
    best_path: list,
    goal: tuple[int, int],
) -> str:
    """Determine dominant reason for path choice.

    Heuristic: compare cost, risk, uncertainty contributions.
    If no clear winner (margin < 20% of total), return "mixed".
    """
    if not latent_predictor or len(best_path) < 2:
        # Cannot decompose without latent predictor
        remaining = max(t_max - t, 1)
        if len(best_path) > remaining * 0.8:
            return "deadline_pressure"
        return "lower_cost"

    # Get predictions for the best next cell
    best_next = best_path[1]
    x = feature_belief_mean[best_next[0], best_next[1]]
    cost_hat = latent_predictor.predict_cost(x)
    risk_hat = latent_predictor.predict_risk(x)
    cost_unc = latent_predictor.predict_cost_uncertainty(x)
    risk_unc = latent_predictor.predict_risk_uncertainty(x)

    # Deadline pressure check
    remaining = max(t_max - t, 1)
    if len(best_path) > remaining * 0.8:
        return "deadline_pressure"

    # Compare magnitudes of the three components
    components = {
        "lower_cost": cost_hat,
        "lower_risk": risk_hat * 3.0,  # weight-adjusted
        "lower_uncertainty": (cost_unc + risk_unc),
    }
    total = sum(components.values()) + 1e-10
    fracs = {k: v / total for k, v in components.items()}

    # Find the dominant component
    sorted_fracs = sorted(fracs.items(), key=lambda x: x[1], reverse=True)
    top_frac = sorted_fracs[0][1]
    second_frac = sorted_fracs[1][1] if len(sorted_fracs) > 1 else 0.0

    # If margin is less than 20% of total, it's mixed
    if top_frac - second_frac < 0.20:
        return "mixed"

    return sorted_fracs[0][0]
