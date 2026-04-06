"""Calibrated confidence: temperature scaling + actionable confidence.

q̃(x) = softmax(log q(x) / T)
C_t = (1 - H̄(q̃)) · σ((margin - τ_m) / τ_c)
"""

from __future__ import annotations
import numpy as np


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))


def calibrate_posterior(
    log_table: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """q̃(x) ∝ exp(log q(x) / T). Returns normalized probabilities."""
    if temperature <= 0:
        temperature = 1e-6
    scaled = log_table / temperature
    scaled = scaled - np.max(scaled)
    t = np.exp(scaled)
    return t / (t.sum() + 1e-10)


def calibrated_confidence(
    calibrated_probs: np.ndarray,
    tau_margin: float = 0.15,
    tau_temp: float = 5.0,
) -> float:
    """C_t = (1 - H̄(q̃)) · σ((margin - τ_m) / τ_c)."""
    p = calibrated_probs.ravel()
    H = float(-np.sum(p * np.log(p + 1e-10)))
    H_max = float(np.log(len(p)))
    H_norm = H / max(H_max, 1e-6)

    sorted_p = np.sort(p)[::-1]
    margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) >= 2 else float(sorted_p[0])
    margin_gate = _sigmoid((margin - tau_margin) * tau_temp)

    return float((1.0 - H_norm) * margin_gate)


def calibrated_entropy(calibrated_probs: np.ndarray) -> float:
    p = calibrated_probs.ravel()
    return float(-np.sum(p * np.log(p + 1e-10)))


def calibrated_top1(calibrated_probs: np.ndarray) -> float:
    return float(np.max(calibrated_probs.ravel()))
