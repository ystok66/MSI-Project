"""Change detection: surprisal and drift score.

s_t = -log Σ_x q_{t-1}(x) · P_A(a_t | s_t, x)
D_t = σ((s_t - τ_s) / τ_d)
ρ_t = ρ_min + (ρ_max - ρ_min) · D_t
"""

from __future__ import annotations
import numpy as np


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))


def compute_surprisal(
    q_table: np.ndarray,
    likelihood_per_cell: np.ndarray,
) -> float:
    """s_t = -log Σ_x q(x) · P(a|x).

    Args:
        q_table: flattened posterior [N]
        likelihood_per_cell: P(a_t|x) for each cell [N]
    """
    q = q_table.ravel()
    lik = likelihood_per_cell.ravel()
    pred = float(np.dot(q, lik))
    return float(-np.log(max(pred, 1e-10)))


def compute_drift_score(
    surprisal: float,
    tau_s: float = 1.5,
    tau_d: float = 2.0,
) -> float:
    """D_t = σ((s_t - τ_s) / τ_d)."""
    return _sigmoid((surprisal - tau_s) / tau_d)


def compute_adaptive_rho(
    drift_score: float,
    rho_min: float = 0.005,
    rho_max: float = 0.15,
) -> float:
    """ρ_t = ρ_min + (ρ_max - ρ_min) · D_t."""
    return rho_min + (rho_max - rho_min) * drift_score


def apply_adaptive_diffusion(
    log_table: np.ndarray,
    rho_t: float,
) -> np.ndarray:
    """q^-_t(x) = (1-ρ_t)·q_{t-1}(x) + ρ_t·u(x).

    Returns updated log_table.
    """
    lt = log_table - np.max(log_table)
    t = np.exp(lt)
    t = t / (t.sum() + 1e-10)
    uniform = np.ones_like(t) / t.size
    t_diffused = (1.0 - rho_t) * t + rho_t * uniform
    new_lt = np.log(t_diffused + 1e-10)
    return new_lt - np.mean(new_lt)
