"""Observation Mask — realistic partial observability for branch summaries.

At each timestep, the agent can only observe cells within its observation
radius. For branch decision-making at the fork point, this means:
  - Early cells (near fork): VISIBLE → use actual features
  - Deep cells (far from fork): UNOBSERVED → use prior features

Masked summary: x̃ = m ⊙ x + (1-m) ⊙ x_prior

This creates genuine epistemic uncertainty at decision time,
which the tutor can reduce via warning.
"""

from __future__ import annotations

import numpy as np


DEFAULT_PRIOR = np.array([0.5, 0.5, 0.25, 0.25])  # neutral prior feature


def make_observation_mask(
    branch_cells: list[tuple[int, int]],
    observer_pos: tuple[int, int],
    obs_radius: int = 2,
) -> np.ndarray:
    """Binary mask: 1 = visible, 0 = unobserved.

    Uses Manhattan distance from observer position.
    """
    mask = np.zeros(len(branch_cells), dtype=np.float64)
    for i, (r, c) in enumerate(branch_cells):
        dist = abs(r - observer_pos[0]) + abs(c - observer_pos[1])
        if dist <= obs_radius:
            mask[i] = 1.0
    return mask


def mask_branch_features(
    branch_cells: list[tuple[int, int]],
    features: np.ndarray,
    mask: np.ndarray,
    prior: np.ndarray | None = None,
) -> np.ndarray:
    """Apply observation mask to branch features.

    Returns masked feature array: visible cells use real features,
    unobserved cells use prior.
    """
    if prior is None:
        prior = DEFAULT_PRIOR

    n = len(branch_cells)
    d = features.shape[-1] if features.ndim > 2 else len(prior)
    masked = np.zeros((n, d), dtype=np.float64)

    for i, (r, c) in enumerate(branch_cells):
        if mask[i] > 0.5:
            masked[i] = features[r, c]
        else:
            masked[i] = prior

    return masked


def summarize_branch_masked(
    branch_cells: list[tuple[int, int]],
    features: np.ndarray,
    feature_var: np.ndarray | None,
    cost_risk_head,
    mask: np.ndarray,
    prior: np.ndarray | None = None,
) -> np.ndarray:
    """Compute branch summary using only visible cells.

    Unobserved cells contribute prior values to the summary.
    This creates genuine partial observability in the branch representation.
    """
    if prior is None:
        prior = DEFAULT_PRIOR

    n = len(branch_cells)
    if n == 0:
        from ..agents.branch_summary import SUMMARY_DIM
        return np.zeros(SUMMARY_DIM)

    # Collect per-cell predictions, mixing real and prior
    risks = []
    costs = []
    risk_uncs = []
    cost_uncs = []
    cue_vals = []

    for i, (r, c) in enumerate(branch_cells):
        if mask[i] > 0.5:
            z = features[r, c]
        else:
            z = prior

        r_pred = cost_risk_head.predict_risk(z)
        c_pred = cost_risk_head.predict_cost(z)
        r_unc = cost_risk_head.predict_risk_uncertainty(z) if mask[i] > 0.5 else 0.5
        c_unc = cost_risk_head.predict_cost_uncertainty(z) if mask[i] > 0.5 else 0.5

        risks.append(r_pred)
        costs.append(c_pred)
        risk_uncs.append(r_unc)
        cost_uncs.append(c_unc)
        cue_vals.append(z[2] if len(z) > 2 else 0.0)

    risks = np.array(risks)
    costs = np.array(costs)
    risk_uncs = np.array(risk_uncs)
    cost_uncs = np.array(cost_uncs)
    cue_vals = np.array(cue_vals)

    # 8D summary: [mean_r, max_r, mean_c, unc_r, unc_c, cue_count, cue_var, length]
    summary = np.array([
        np.mean(risks),
        np.max(risks),
        np.mean(costs),
        np.mean(risk_uncs),
        np.mean(cost_uncs),
        float(np.sum(cue_vals > 0.3)),  # count of notable cues
        float(np.var(cue_vals)) if len(cue_vals) > 1 else 0.0,
        float(n) / 10.0,
    ])

    return summary


def branch_entropy(
    concept_lib,
    summary: np.ndarray,
    obs_var: float = 0.01,
    tau: float = 1.0,
) -> float:
    """Compute semantic entropy H(π) = -Σ P(k|π) log P(k|π).

    Uses concept match scores as log-probabilities.
    """
    scores = concept_lib.score_all(summary, obs_var=obs_var, tau=tau)
    if not scores:
        return 0.0

    vals = np.array(list(scores.values()))
    # Softmax to get probabilities
    vals = vals - np.max(vals)
    probs = np.exp(vals) / np.sum(np.exp(vals))

    # Entropy
    H = -np.sum(probs * np.log(probs + 1e-10))
    return float(H)
