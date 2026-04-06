"""Overteaching detection and penalty.

R_over(a) = α_κ·[κ'-κ_max_safe]²₊ + α_γ·[γ'-γ_max_safe]²₊ + α_η·[η_min_useful-η']²₊
"""

from __future__ import annotations
import numpy as np


SAFE_BOUNDS = {
    "kappa_max_safe": 2.2,
    "gamma_max_safe": 0.55,
    "eta_min_useful": 0.15,
}

OVER_WEIGHTS = {"kappa": 2.0, "gamma": 1.5, "eta": 1.0}


def overteach_penalty(
    kappa_next: float, eta_next: float, gamma_next: float,
    bounds: dict = None,
) -> float:
    """R_over = weighted sum of safe-bound violations."""
    b = bounds or SAFE_BOUNDS
    r_k = max(kappa_next - b["kappa_max_safe"], 0.0) ** 2
    r_g = max(gamma_next - b["gamma_max_safe"], 0.0) ** 2
    r_e = max(b["eta_min_useful"] - eta_next, 0.0) ** 2
    return float(
        OVER_WEIGHTS["kappa"] * r_k
        + OVER_WEIGHTS["gamma"] * r_g
        + OVER_WEIGHTS["eta"] * r_e)


def overteach_rate(kappa_hist, eta_hist, gamma_hist, bounds=None):
    """Fraction of timesteps where any component exceeds safe bounds."""
    b = bounds or SAFE_BOUNDS
    if not kappa_hist:
        return 0.0
    violations = 0
    for k, e, g in zip(kappa_hist, eta_hist, gamma_hist):
        if (k > b["kappa_max_safe"] or g > b["gamma_max_safe"]
                or e < b["eta_min_useful"]):
            violations += 1
    return violations / len(kappa_hist)


def overteach_decomposed(kappa_hist, eta_hist, gamma_hist, bounds=None):
    """Per-component overteach rates."""
    b = bounds or SAFE_BOUNDS
    n = max(len(kappa_hist), 1)
    return {
        "kappa_over": sum(1 for k in kappa_hist if k > b["kappa_max_safe"]) / n,
        "gamma_over": sum(1 for g in gamma_hist if g > b["gamma_max_safe"]) / n,
        "eta_under": sum(1 for e in eta_hist if e < b["eta_min_useful"]) / n,
    }
