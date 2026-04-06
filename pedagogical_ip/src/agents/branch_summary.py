"""V5.1 — Branch Semantic Summary Layer.

Deterministic aggregator that lifts per-cell predictions to branch-level
semantic summaries. No neural network — just principled feature engineering.

The 8D summary vector is the basis for V5.2 (Gaussian concepts),
V5.3 (familiarity), and V5.6 (branch scorer).

Interface
---------
    summarize_branch(cells, belief_mean, belief_var, cost_risk_head)
        -> np.ndarray (SUMMARY_DIM,)
"""

from __future__ import annotations

import numpy as np

SUMMARY_DIM = 8

# Summary vector indices
S_MEAN_RISK   = 0   # mean predicted risk across branch
S_MAX_RISK    = 1   # max predicted risk
S_MEAN_COST   = 2   # mean predicted cost
S_RISK_UNC    = 3   # mean risk uncertainty
S_COST_UNC    = 4   # mean cost uncertainty
S_CUE_COUNT   = 5   # number of high-texture cells (potential cues)
S_CUE_VAR     = 6   # variance of texture features (cue diversity)
S_LENGTH      = 7   # branch length (normalized)


def summarize_branch(
    cells: list[tuple[int, int]],
    belief_mean: np.ndarray,
    belief_var: np.ndarray | None,
    cost_risk_head,
    *,
    cue_threshold: float = 0.3,
    max_branch_len: float = 10.0,
) -> np.ndarray:
    """Compute semantic summary for a branch.

    Parameters
    ----------
    cells : list of (row, col)
        Cells belonging to this branch.
    belief_mean : (H, W, d) array
        Agent's current feature belief mean.
    belief_var : (H, W, d) array or None
        Agent's current feature belief variance. If None uses Hessian.
    cost_risk_head : LatentCostRiskHead
        Current learned predictor.
    cue_threshold : float
        Texture value above which a cell is counted as a "cue".
    max_branch_len : float
        Normalizer for branch length.

    Returns
    -------
    summary : (SUMMARY_DIM,) ndarray
    """
    if not cells:
        return np.zeros(SUMMARY_DIM, dtype=np.float64)

    n = len(cells)
    risks = np.empty(n)
    costs = np.empty(n)
    risk_uncs = np.empty(n)
    cost_uncs = np.empty(n)
    textures = []

    for i, (r, c) in enumerate(cells):
        x = belief_mean[r, c]
        risks[i] = cost_risk_head.predict_risk(x)
        costs[i] = cost_risk_head.predict_cost(x)

        if belief_var is not None:
            xv = belief_var[r, c]
            risk_uncs[i] = cost_risk_head.predict_risk_uncertainty_from_var(xv)
            cost_uncs[i] = cost_risk_head.predict_cost_uncertainty_from_var(xv)
        else:
            risk_uncs[i] = cost_risk_head.predict_risk_uncertainty(x)
            cost_uncs[i] = cost_risk_head.predict_cost_uncertainty(x)

        # Texture features (indices 2, 3 are texture dims)
        textures.append(x[2:4] if len(x) >= 4 else x[-2:])

    textures = np.array(textures)  # (n, 2)

    # Cue count: cells where any texture > threshold
    cue_count = int(np.sum(np.any(textures > cue_threshold, axis=1)))

    # Cue variance: how diverse are the texture patterns
    cue_var = float(np.mean(np.var(textures, axis=0))) if n > 1 else 0.0

    summary = np.array([
        float(np.mean(risks)),                    # S_MEAN_RISK
        float(np.max(risks)),                     # S_MAX_RISK
        float(np.mean(costs)),                    # S_MEAN_COST
        float(np.mean(risk_uncs)),                # S_RISK_UNC
        float(np.mean(cost_uncs)),                # S_COST_UNC
        cue_count / max(n, 1),                    # S_CUE_COUNT (normalized)
        cue_var,                                  # S_CUE_VAR
        min(n / max_branch_len, 1.0),             # S_LENGTH (normalized)
    ], dtype=np.float64)

    return summary


def summarize_branches(
    branch_cells_list: list[list[tuple[int, int]]],
    belief_mean: np.ndarray,
    belief_var: np.ndarray | None,
    cost_risk_head,
    **kwargs,
) -> np.ndarray:
    """Summarize multiple branches at once.

    Returns (n_branches, SUMMARY_DIM) array.
    """
    return np.array([
        summarize_branch(cells, belief_mean, belief_var, cost_risk_head, **kwargs)
        for cells in branch_cells_list
    ])
