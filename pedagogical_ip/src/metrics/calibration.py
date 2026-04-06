"""H2 — Calibration Metrics for Selectivity Law.

Provides:
  1. Expected Calibration Error (ECE) for p_self
  2. Continuous WarnRate(Δ) curve fitting
  3. Self-discovery empirical frequency computation
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def expected_calibration_error(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict]]:
    """Compute Expected Calibration Error.

    Args:
        predicted: predicted probabilities (e.g., p_self)
        actual: binary outcomes (e.g., did agent self-discover?)

    Returns:
        (ECE, list of bin details)
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    n = len(predicted)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    weighted_error = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (predicted >= lo) & (predicted < hi + (1e-9 if i == n_bins - 1 else 0))
        count = int(np.sum(mask))
        if count == 0:
            continue
        avg_pred = float(np.mean(predicted[mask]))
        avg_actual = float(np.mean(actual[mask]))
        error = abs(avg_pred - avg_actual)
        weighted_error += (count / n) * error
        bins.append({
            "bin": "[{:.1f},{:.1f})".format(lo, hi),
            "count": count,
            "avg_predicted": round(avg_pred, 3),
            "avg_actual": round(avg_actual, 3),
            "error": round(error, 3),
        })

    return round(weighted_error, 4), bins


def compute_empirical_self_discovery(
    obs_radius: int,
    reveal_depth: int,
    branch_len: int,
) -> float:
    """Deterministic empirical self-discovery: fraction of strong cues visible.

    If obs_radius >= reveal_depth, agent can see at least one strong-cue cell.
    """
    n_strong = branch_len - reveal_depth
    if n_strong <= 0:
        return 0.0
    n_visible_strong = max(0, min(obs_radius, branch_len) - reveal_depth)
    return min(n_visible_strong / n_strong, 1.0)


def compute_warm_rate_curve(
    deltas: list[int],
    warn_rates: list[float],
) -> dict:
    """Analyze WarnRate(Δ) curve properties.

    Returns:
        monotonic: bool — is WarnRate monotonically decreasing?
        transition_zone: (Δ_start, Δ_end) where WarnRate transitions
        slope_at_zero: approximate slope at Δ=0
    """
    is_monotonic = all(warn_rates[i] >= warn_rates[i + 1]
                       for i in range(len(warn_rates) - 1))

    # Find transition zone (first and last Δ where 0 < WR < 1)
    trans = [(d, wr) for d, wr in zip(deltas, warn_rates) if 0 < wr < 1]
    if trans:
        trans_start = trans[0][0]
        trans_end = trans[-1][0]
    else:
        trans_start = trans_end = None

    # Slope at Δ=0
    idx0 = None
    for i, d in enumerate(deltas):
        if d == 0:
            idx0 = i
            break
    if idx0 is not None and idx0 > 0 and idx0 < len(deltas) - 1:
        slope = (warn_rates[idx0 + 1] - warn_rates[idx0 - 1]) / 2.0
    else:
        slope = 0.0

    return {
        "monotonic": is_monotonic,
        "transition_zone": (trans_start, trans_end),
        "slope_at_zero": round(slope, 3),
        "min_warn_rate": round(min(warn_rates), 3),
        "max_warn_rate": round(max(warn_rates), 3),
    }
