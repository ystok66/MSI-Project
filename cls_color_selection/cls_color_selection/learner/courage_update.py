"""
courage_update.py — Minimal courage mechanism (Phase 1).

COURAGE semantics: "there exists a ball that is safe AND needed."

P(Z | X, courage) ∝ 1[∃i: z_i=0 ∧ color_i ∈ needed] · P(Z | X)

Default: disabled (enable_courage=False in config).
"""
from __future__ import annotations
from typing import List, Set
import numpy as np

from .risk_belief import DangerTypeBelief
from ..interfaces import CandidateBall


def courage_literal_update(
    risk_belief: DangerTypeBelief,
    selected_balls: List[CandidateBall],
    needed_colors: Set[str],
) -> np.ndarray:
    """Update per-ball posteriors given COURAGE signal.

    Courage tells us: ∃i in candidate set s.t. (z_i=0 AND color_i ∈ needed).

    Args:
        risk_belief: current DangerTypeBelief model
        selected_balls: balls in the current candidate pool
        needed_colors: colors still needed for completion

    Returns:
        (m, n_types) updated posterior matrix.
    """
    m = len(selected_balls)
    if m == 0:
        return np.array([])

    # Collect observed vectors
    X = np.stack([b.observed_vec for b in selected_balls])
    posteriors = risk_belief.batch_posterior(X)  # (m, n_types)

    # Identify which balls are "eligible" (needed color)
    eligible = np.array([b.color in needed_colors for b in selected_balls])

    # P(all eligible balls are danger) = Π_{i: eligible} (1 - P(z_i=0|x_i))
    p_safe = posteriors[:, 0]  # (m,)

    eligible_safe_probs = p_safe[eligible]
    if len(eligible_safe_probs) == 0:
        # No eligible balls — courage is vacuous
        return posteriors

    # P(no eligible ball is safe) = Π (1 - p_safe_i) for eligible i
    p_none_safe_eligible = np.prod(1.0 - eligible_safe_probs)

    denom = 1.0 - p_none_safe_eligible
    if denom < 1e-30:
        return posteriors  # Can't condition

    updated = posteriors.copy()

    for i in range(m):
        if not eligible[i]:
            continue  # Non-eligible balls' posteriors unchanged

        # For eligible ball i:
        # P(z_i=0 | courage) = P(z_i=0) · 1.0 / denom (courage satisfied if z_i=0)
        # P(z_i=k | courage) for k>0: need rest to have ∃j safe&needed
        p_rest_none = p_none_safe_eligible / max(1.0 - p_safe[i], 1e-30)

        updated[i, 0] = posteriors[i, 0] * 1.0 / denom
        for k in range(1, risk_belief.n_types):
            updated[i, k] = posteriors[i, k] * (1.0 - p_rest_none) / denom

        row_sum = updated[i].sum()
        if row_sum > 1e-30:
            updated[i] /= row_sum

    return updated
