"""Curriculum Metrics: MPG, Stop Efficiency."""

from __future__ import annotations
import numpy as np

PROBE_NAMES = ["RC", "TR", "EP", "VA", "IA"]
W_PROBE = {"RC": 1.0, "TR": 1.2, "EP": 2.5, "VA": 1.5, "IA": 2.5}


def mastery_progress_gain(mastery_history: list, weights=None) -> float:
    """MPG = (1/T) Σ Σ_k w_k (u_{k,t+1} - u_{k,t})."""
    if weights is None:
        weights = W_PROBE
    if len(mastery_history) < 2:
        return 0.0
    total = 0.0
    for t in range(len(mastery_history) - 1):
        for p in PROBE_NAMES:
            total += weights[p] * (mastery_history[t + 1].get(p, 0.5)
                                   - mastery_history[t].get(p, 0.5))
    return round(total / (len(mastery_history) - 1), 4)


def stop_efficiency(transfer_gain: float, n_lessons: int) -> float:
    """SE = transfer_gain / #lessons."""
    if n_lessons == 0:
        return 0.0
    return round(transfer_gain / n_lessons, 4)
