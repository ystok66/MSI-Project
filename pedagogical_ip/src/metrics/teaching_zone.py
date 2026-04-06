"""Teaching Zone: target intervals for internalization state.

Z_θ = [κ_lo, κ_hi] × [η_lo, η_hi] × [γ_lo, γ_hi]
Band loss: φ(x; [l,u]) = max(l-x,0)² + max(x-u,0)²
L_teach(m, θ) = w_κ·φ(κ) + w_η·φ(η) + w_γ·φ(γ)
"""

from __future__ import annotations
import numpy as np


# Per-θ target zones (empirically derived from TIC results)
TARGET_ZONES = {
    "safe":     {"kappa": (0.8, 1.8), "eta": (0.2, 0.7), "gamma": (0.0, 0.3)},
    "shiny":    {"kappa": (1.0, 2.0), "eta": (0.2, 0.7), "gamma": (0.2, 0.5)},
    "neutral":  {"kappa": (0.8, 1.5), "eta": (0.2, 0.6), "gamma": (0.0, 0.2)},
    "shortcut": {"kappa": (1.0, 2.0), "eta": (0.3, 0.7), "gamma": (0.1, 0.4)},
    "explorer": {"kappa": (0.6, 1.5), "eta": (0.2, 0.6), "gamma": (0.0, 0.2)},
}

ZONE_WEIGHTS = {"kappa": 1.0, "eta": 1.5, "gamma": 1.2}


def band_loss(x: float, lo: float, hi: float) -> float:
    """φ(x; [l,u]) = max(l-x,0)² + max(x-u,0)²."""
    return max(lo - x, 0.0) ** 2 + max(x - hi, 0.0) ** 2


def teaching_loss(kappa, eta, gamma, theta="safe", q_theta=None):
    """L_teach(m, q) = Σ_θ q(θ)·L(m,θ)."""
    if q_theta is not None:
        from ..agents.stochastic_agent_policy import PREFERENCE_TYPES
        loss = 0.0
        for pi, p in enumerate(PREFERENCE_TYPES):
            z = TARGET_ZONES.get(p, TARGET_ZONES["safe"])
            l = (ZONE_WEIGHTS["kappa"] * band_loss(kappa, *z["kappa"])
                 + ZONE_WEIGHTS["eta"] * band_loss(eta, *z["eta"])
                 + ZONE_WEIGHTS["gamma"] * band_loss(gamma, *z["gamma"]))
            loss += q_theta[pi] * l
        return float(loss)
    z = TARGET_ZONES.get(theta, TARGET_ZONES["safe"])
    return float(
        ZONE_WEIGHTS["kappa"] * band_loss(kappa, *z["kappa"])
        + ZONE_WEIGHTS["eta"] * band_loss(eta, *z["eta"])
        + ZONE_WEIGHTS["gamma"] * band_loss(gamma, *z["gamma"]))


def in_zone(kappa, eta, gamma, theta="safe") -> dict:
    """Check which components are inside their target zone."""
    z = TARGET_ZONES.get(theta, TARGET_ZONES["safe"])
    return {
        "kappa_in": z["kappa"][0] <= kappa <= z["kappa"][1],
        "eta_in": z["eta"][0] <= eta <= z["eta"][1],
        "gamma_in": z["gamma"][0] <= gamma <= z["gamma"][1],
        "all_in": (z["kappa"][0] <= kappa <= z["kappa"][1]
                   and z["eta"][0] <= eta <= z["eta"][1]
                   and z["gamma"][0] <= gamma <= z["gamma"][1]),
    }


def zone_hit_rate(kappa_hist, eta_hist, gamma_hist, theta="safe") -> float:
    """Fraction of timesteps where all components are in-zone."""
    if not kappa_hist:
        return 0.0
    hits = sum(1 for k, e, g in zip(kappa_hist, eta_hist, gamma_hist)
               if in_zone(k, e, g, theta)["all_in"])
    return hits / len(kappa_hist)
