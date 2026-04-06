"""Pedagogical metrics: Learning Gain, Transfer Improvement, Pedagogical Efficiency.

LG = E[m_post - m_pre]
TI = Perf_post_tutor - Perf_no_tutor_baseline
PE = TI / #warnings
"""

from __future__ import annotations
import numpy as np
from ..agents.internalization_agent import InternalizationState


def learning_gain(m_pre: InternalizationState, m_post: InternalizationState) -> dict:
    """LG = m_post - m_pre, decomposed by component."""
    return {
        "lg_kappa": round(m_post.kappa - m_pre.kappa, 4),
        "lg_eta": round(m_post.eta - m_pre.eta, 4),
        "lg_gamma": round(m_post.gamma - m_pre.gamma, 4),
        "lg_total": round(
            abs(m_post.kappa - m_pre.kappa) +
            abs(m_post.eta - m_pre.eta) +
            abs(m_post.gamma - m_pre.gamma), 4),
    }


def transfer_improvement(
    post_tutor_sbcr: float,
    no_tutor_baseline_sbcr: float,
) -> float:
    """TI = Perf(post tutor) - Perf(no tutor baseline)."""
    return round(post_tutor_sbcr - no_tutor_baseline_sbcr, 4)


def pedagogical_efficiency(
    transfer_improvement_val: float,
    n_warnings: int,
) -> float:
    """PE = TI / max(n_warnings, 1)."""
    return round(transfer_improvement_val / max(n_warnings, 1), 4)


def internalization_trajectory(m: InternalizationState) -> dict:
    """Return full trajectory of (κ,η,γ) for plotting."""
    return {
        "kappa": list(m.kappa_history),
        "eta": list(m.eta_history),
        "gamma": list(m.gamma_history),
    }


def compute_decision_calibration_error(
    confidence_values: list[float],
    oracle_compatible: list[bool],
    n_bins: int = 5,
) -> float:
    """DCE = E[|C_t - 1[oracle-compatible]|]."""
    if not confidence_values:
        return 0.0
    c = np.array(confidence_values)
    o = np.array(oracle_compatible, dtype=float)
    return float(np.mean(np.abs(c - o)))
