"""
Prefix Prediction — multi-cell diagnostics over the planned path.

Phase 5: read-only diagnostics computed from the existing A* path.
Does NOT alter planner, belief, or environment state.

Usage:
    path = [current_pos, next_pos, ...]  # from A*
    pred = compute_prefix_predictions(path, belief_mean, latent_predictor, horizon=5)
    print(pred.cumulative_risk, pred.risky_prefix_cells)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PrefixPrediction:
    """Multi-cell prediction over the first H cells of the planned path.

    cumulative_risk uses an independence approximation:
      P(at least one hazard) = 1 - ∏(1 - ρ_i)
    This is NOT a fully modeled joint hazard process.
    """
    prefix_cells: list[tuple[int, int]]
    cost_predictions: list[float]
    risk_predictions: list[float]
    cost_uncertainties: list[float]
    risk_uncertainties: list[float]
    cumulative_cost: float
    cumulative_risk: float  # independence approximation
    risky_prefix_cells: list[tuple[int, int]] = field(default_factory=list)


def compute_prefix_predictions(
    path: list[tuple[int, int]],
    feature_belief_mean: np.ndarray,   # (H, W, d)
    latent_predictor,                   # LatentCostRiskHead (latent_predictor protocol)
    horizon: int = 5,
    risk_threshold: float = 0.3,
) -> PrefixPrediction:
    """Compute cost/risk/uncertainty predictions over path prefix.

    This is a READ-ONLY diagnostic function. It does not modify any state.

    Args:
        path: planned path from A* (list of (row, col) tuples)
        feature_belief_mean: agent's current belief mean (H, W, d)
        latent_predictor: provides predict_cost/risk/uncertainty methods
        horizon: max number of future cells to predict over
        risk_threshold: cells with risk > threshold are flagged

    Returns:
        PrefixPrediction with per-cell and aggregate predictions
    """
    # Take the first `horizon` cells from path (skip current position)
    prefix = path[1:horizon + 1] if len(path) > 1 else []

    cost_preds = []
    risk_preds = []
    cost_uncs = []
    risk_uncs = []
    risky = []

    for r, c in prefix:
        x = feature_belief_mean[r, c]
        cp = latent_predictor.predict_cost(x)
        rp = latent_predictor.predict_risk(x)
        cu = latent_predictor.predict_cost_uncertainty(x)
        ru = latent_predictor.predict_risk_uncertainty(x)

        cost_preds.append(cp)
        risk_preds.append(rp)
        cost_uncs.append(cu)
        risk_uncs.append(ru)

        if rp > risk_threshold:
            risky.append((r, c))

    cum_cost = sum(cost_preds) if cost_preds else 0.0
    # Independence approximation: P(≥1 hazard) = 1 - ∏(1 - ρ_i)
    survival_prod = 1.0
    for rp in risk_preds:
        survival_prod *= (1.0 - min(rp, 0.999))
    cum_risk = 1.0 - survival_prod

    return PrefixPrediction(
        prefix_cells=list(prefix),
        cost_predictions=cost_preds,
        risk_predictions=risk_preds,
        cost_uncertainties=cost_uncs,
        risk_uncertainties=risk_uncs,
        cumulative_cost=cum_cost,
        cumulative_risk=cum_risk,
        risky_prefix_cells=risky,
    )
