"""
warning_update.py — Set-conditional Bayesian update after WARNING.

WARNING semantics: "at least one ball in your selection is dangerous."

P(Z | X, warning) ∝ 1[∃i: z_i ≠ 0] · P(Z | X)

Uses independence approximation to compute per-ball marginal posteriors.
"""
from __future__ import annotations
from typing import List
import numpy as np

from .risk_belief import DangerTypeBelief
from ..interfaces import CandidateBall


def warning_set_bayes_update(
    risk_belief: DangerTypeBelief,
    selected_balls: List[CandidateBall],
) -> np.ndarray:
    """Update per-ball posteriors given WARNING signal.

    The warning tells us: ∃i in selected set s.t. z_i ≠ 0.
    Equivalently: NOT(all balls are safe).

    Using independence approximation:
    P(z_i=k | X, warning) for k≠0:
      = P(z_i=k|x_i) · [1 - P_all_safe_without_i] / (1 - P_all_safe)

    where P_all_safe = Π_j P(z_j=0|x_j)
    and P_all_safe_without_i = P_all_safe / P(z_i=0|x_i)

    Args:
        risk_belief: current DangerTypeBelief model
        selected_balls: the balls that were selected (and warned about)

    Returns:
        (m, n_types) updated posterior matrix for the selected balls.
        Also updates the risk_belief prototypes using the posterior shift.
    """
    m = len(selected_balls)
    if m == 0:
        return np.array([])

    n_types = risk_belief.n_types

    # Collect observed vectors
    X = np.stack([b.observed_vec for b in selected_balls])  # (m, d)

    # Get current per-ball posteriors
    posteriors = risk_belief.batch_posterior(X)  # (m, n_types)

    # P(z_i = 0 | x_i) for each ball
    p_safe = posteriors[:, 0]  # (m,)

    # P(all safe) = Π P(z_i=0|x_i)
    # Use log for numerical stability
    log_p_safe = np.log(np.clip(p_safe, 1e-30, 1.0))
    log_p_all_safe = np.sum(log_p_safe)
    p_all_safe = np.exp(log_p_all_safe)

    # Conditioning: P(Z|X, warning) ∝ (1 - 1[all safe]) · P(Z|X)
    # The normalizer is (1 - P_all_safe)
    denom = 1.0 - p_all_safe
    if denom < 1e-30:
        # Edge case: model thinks all are safe but warning says otherwise
        # Fall back to uniform danger shift
        updated = posteriors.copy()
        updated[:, 0] *= 0.5  # reduce safe confidence
        updated /= updated.sum(axis=1, keepdims=True) + 1e-30
        return updated

    # Compute updated posteriors using marginal formula
    updated = np.zeros_like(posteriors)

    for i in range(m):
        # P(all safe except i) = P_all_safe / P(z_i=0|x_i)
        p_safe_i = max(p_safe[i], 1e-30)
        p_rest_safe = p_all_safe / p_safe_i

        for k in range(n_types):
            if k == 0:
                # P(z_i=0 | warning) = P(z_i=0) · P(rest has danger | z_i=0)
                # = P(z_i=0) · (1 - P_rest_all_safe) / denom ... but this
                # simplifies to:
                # P(z_i=0 | X, warning) = P(z_i=0|x_i) · (1 - p_rest_safe) / denom
                updated[i, 0] = posteriors[i, 0] * (1.0 - p_rest_safe) / denom
            else:
                # P(z_i=k | X, warning) = P(z_i=k|x_i) · 1.0 / denom
                # Because if z_i≠0, the warning constraint is automatically satisfied
                updated[i, k] = posteriors[i, k] * 1.0 / denom

        # Renormalize
        row_sum = updated[i].sum()
        if row_sum > 1e-30:
            updated[i] /= row_sum
        else:
            # Fallback
            updated[i] = posteriors[i]

    # Update risk_belief prototypes using the posterior shift
    for i, ball in enumerate(selected_balls):
        risk_belief._accumulate_update(ball.observed_vec, updated[i])

    return updated
