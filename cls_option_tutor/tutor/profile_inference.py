"""
profile_inference.py — Bayesian profile posterior update in log-space.

Maintains posterior weights over a discrete profile hypothesis set Ψ.
Update rule:
    log w_t(ψ) = log w_{t-1}(ψ) + η * log π_ψ(a_t | x_t)
    then normalize via logsumexp.

All computations in log-space to avoid underflow on small likelihoods.
"""
from __future__ import annotations

import numpy as np


def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    c = x.max()
    if not np.isfinite(c):
        return float('-inf')
    return float(c + np.log(np.sum(np.exp(x - c))))


def update_profile_posterior(
    log_weights: np.ndarray,
    action_likelihoods: np.ndarray,
    eta_prof: float = 1.0,
    floor: float = 1e-8,
) -> np.ndarray:
    """Bayesian update of profile posterior in log-space.

    Args:
        log_weights:        (M,) current log-posterior weights.
        action_likelihoods: (M,) π_ψ(a_t | x_t) for each profile hypothesis.
                            Must be non-negative; clipped to [floor, 1.0].
        eta_prof:           likelihood temperature.  1.0 = standard Bayes.
        floor:              minimum likelihood to prevent log(0).

    Returns:
        (M,) updated log-posterior weights (normalized: logsumexp = 0).
    """
    ll = np.log(np.clip(action_likelihoods, floor, 1.0))
    log_w = log_weights + eta_prof * ll
    log_w = log_w - _logsumexp(log_w)
    return log_w


def init_uniform_log_weights(m: int) -> np.ndarray:
    """Initialize uniform log-posterior: log(1/M) for each of M profiles."""
    return np.full(m, -np.log(m))


def posterior_entropy(log_weights: np.ndarray) -> float:
    """Shannon entropy of the profile posterior (nats).

    H = -Σ w(ψ) log w(ψ).
    Returns 0 if degenerate (single hypothesis with all mass).
    """
    w = np.exp(log_weights)
    w = w / w.sum()  # ensure normalized
    w = np.clip(w, 1e-15, 1.0)
    return float(-np.sum(w * np.log(w)))


def posterior_probs(log_weights: np.ndarray) -> np.ndarray:
    """Convert log-weights to probability simplex."""
    w = np.exp(log_weights - log_weights.max())
    return w / w.sum()
