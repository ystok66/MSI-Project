"""Factor-to-Action Bridge.

Connects per-factor confidence and utility to tutor action signal.

C_k = 1 - H(q(y_k)) / log(3)
ΔU_k(b) = E[y_k] · x_k(b)
M_factor(b) = Σ_k ω_k · C_k · ΔU_k(b)
R_conflict_factor = Σ_k ω_k · C_k · 1[factor ≠ pref]
"""

from __future__ import annotations
import numpy as np

FACTOR_VALUES = [-1, 0, 1]


def factor_confidence(q_yk: np.ndarray) -> float:
    """C_k = 1 - H(q(y_k)) / log(3)."""
    p = q_yk / (q_yk.sum() + 1e-10)
    h = float(-np.sum(p * np.log(p + 1e-10)))
    return max(0.0, 1.0 - h / np.log(3))


def factor_expected_value(q_yk: np.ndarray) -> float:
    """E[y_k] = Σ v · q(y_k = v)."""
    p = q_yk / (q_yk.sum() + 1e-10)
    return float(np.dot(FACTOR_VALUES, p))


def factor_utility_delta(expected_yk: float, branch_feature_k: float) -> float:
    """ΔU_k(b) = E[y_k] · x_k(b)."""
    return expected_yk * branch_feature_k


def factor_branch_margin(
    factor_marginals: list[np.ndarray],
    branch_a_features: np.ndarray,
    branch_b_features: np.ndarray,
    weights: np.ndarray = None,
) -> float:
    """M_factor = Σ_k ω_k · C_k · (ΔU_k(a) - ΔU_k(b))."""
    K = len(factor_marginals)
    if weights is None:
        weights = np.ones(K)
    margin = 0.0
    for k in range(K):
        ck = factor_confidence(factor_marginals[k])
        ek = factor_expected_value(factor_marginals[k])
        du_a = factor_utility_delta(ek, branch_a_features[k])
        du_b = factor_utility_delta(ek, branch_b_features[k])
        margin += weights[k] * ck * (du_a - du_b)
    return float(margin)


def factor_conflict_mass(
    factor_marginals: list[np.ndarray],
    pref_induced_branch: int,
    branch_a_features: np.ndarray,
    branch_b_features: np.ndarray,
    weights: np.ndarray = None,
) -> float:
    """R_conflict = Σ_k ω_k · C_k · 1[factor→branch ≠ pref→branch]."""
    K = len(factor_marginals)
    if weights is None:
        weights = np.ones(K)
    conflict = 0.0
    for k in range(K):
        ck = factor_confidence(factor_marginals[k])
        ek = factor_expected_value(factor_marginals[k])
        du_a = factor_utility_delta(ek, branch_a_features[k])
        du_b = factor_utility_delta(ek, branch_b_features[k])
        factor_preferred = 0 if du_a >= du_b else 1
        if factor_preferred != pref_induced_branch:
            conflict += weights[k] * ck
    return float(conflict)


def factor_aware_info_value(
    factor_marginals: list[np.ndarray],
    branch_a_features: np.ndarray,
    branch_b_features: np.ndarray,
    weights: np.ndarray = None,
) -> float:
    """How much would revealing hidden factors change branch choice?"""
    K = len(factor_marginals)
    if weights is None:
        weights = np.ones(K)
    value = 0.0
    for k in range(K):
        ck = factor_confidence(factor_marginals[k])
        ek = factor_expected_value(factor_marginals[k])
        du_a = factor_utility_delta(ek, branch_a_features[k])
        du_b = factor_utility_delta(ek, branch_b_features[k])
        # Low confidence + high potential impact = high info value
        value += weights[k] * (1.0 - ck) * abs(du_a - du_b)
    return float(value)
