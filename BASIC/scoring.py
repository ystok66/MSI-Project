"""
Inclusion scoring functions for RSA research project.

Provides:
- log_inc_single: Log-inclusion score for single token
- inc_single: Inclusion score (exp of log_inc)
- log_inc_multi: Log-inclusion for multiple tokens (AND combination)
- log_inc_matrix: Batch computation for all regions

Inclusion score: inc(t,u) = exp(-KL(A_t || B_u) / tau)
where A_t is the object distribution and B_u is the concept distribution.
"""

import numpy as np
from typing import List, Tuple

from gaussian import kl_diag_gaussians, clamp_var


# =============================================================================
# Default Hyperparameters
# =============================================================================

EPS_OBJ = 1e-4    # Object variance (small = point-like observation)
TAU = 1.0         # Temperature for softening KL
MIN_VAR = 1e-8    # Minimum variance for numerical stability


# =============================================================================
# Single Token Scoring
# =============================================================================

def log_inc_single(
    x_t: np.ndarray,
    mu_u: np.ndarray,
    var_u: np.ndarray,
    *,
    eps_obj: float = EPS_OBJ,
    tau: float = TAU,
    min_var: float = MIN_VAR
) -> float:
    """
    Compute log-inclusion score for a single token.
    
    log_inc(t, u) = -KL(A_t || B_u) / tau
    
    where A_t ~ N(x_t, eps_obj * I) is the object distribution
    and B_u ~ N(mu_u, diag(var_u)) is the concept distribution.
    
    Args:
        x_t: Object observation vector, shape (d,)
        mu_u: Concept mean, shape (d,)
        var_u: Concept diagonal variance, shape (d,)
        eps_obj: Object variance (same for all dimensions)
        tau: Temperature parameter
        min_var: Minimum variance for stability
        
    Returns:
        Log-inclusion score as Python float
    """
    d = x_t.shape[0]
    
    # Object distribution: point-like with small variance
    mu_a = x_t
    var_a = eps_obj * np.ones(d, dtype=np.float64)
    
    # Compute KL divergence
    kl = kl_diag_gaussians(mu_a, var_a, mu_u, var_u, min_var=min_var)
    
    # Log-inclusion: -KL / tau
    log_inc = -kl / tau
    
    return log_inc


def inc_single(
    x_t: np.ndarray,
    mu_u: np.ndarray,
    var_u: np.ndarray,
    *,
    eps_obj: float = EPS_OBJ,
    tau: float = TAU,
    min_var: float = MIN_VAR,
    clamp_max: float = 1.0
) -> float:
    """
    Compute inclusion score (exp of log-inclusion).
    
    inc(t, u) = exp(-KL(A_t || B_u) / tau)
    
    Since KL >= 0, theoretically inc <= 1. However, numerical errors
    may cause slight overflow; clamp_max prevents this.
    
    Args:
        x_t: Object observation vector, shape (d,)
        mu_u: Concept mean, shape (d,)
        var_u: Concept diagonal variance, shape (d,)
        eps_obj: Object variance
        tau: Temperature parameter
        min_var: Minimum variance for stability
        clamp_max: Maximum allowed inclusion score
        
    Returns:
        Inclusion score in (0, clamp_max]
    """
    log_inc = log_inc_single(
        x_t, mu_u, var_u,
        eps_obj=eps_obj, tau=tau, min_var=min_var
    )
    
    inc = np.exp(log_inc)
    
    # Clamp to prevent numerical overflow above 1
    if clamp_max is not None:
        inc = min(inc, clamp_max)
    
    return float(inc)


# =============================================================================
# Multi-Token Scoring (AND combination)
# =============================================================================

def log_inc_multi(
    x_t: np.ndarray,
    concepts: List[Tuple[np.ndarray, np.ndarray]],
    *,
    eps_obj: float = EPS_OBJ,
    tau: float = TAU,
    min_var: float = MIN_VAR
) -> float:
    """
    Compute log-inclusion for multiple tokens with AND semantics.
    
    log_inc(t, U) = sum_{u in U} log_inc(t, u)
    
    This is equivalent to requiring the object to be included in ALL concepts.
    
    Args:
        x_t: Object observation vector, shape (d,)
        concepts: List of (mu_u, var_u) tuples for each token
        eps_obj: Object variance
        tau: Temperature parameter
        min_var: Minimum variance for stability
        
    Returns:
        Sum of log-inclusion scores.
        If concepts is empty, returns 0.0 (neutral element for AND).
    """
    if len(concepts) == 0:
        return 0.0
    
    total_log_inc = 0.0
    for mu_u, var_u in concepts:
        total_log_inc += log_inc_single(
            x_t, mu_u, var_u,
            eps_obj=eps_obj, tau=tau, min_var=min_var
        )
    
    return total_log_inc


# =============================================================================
# Batch Computation (for RSA inference)
# =============================================================================

def log_inc_matrix(
    X: np.ndarray,
    mask: np.ndarray,
    mu_u: np.ndarray,
    var_u: np.ndarray,
    *,
    eps_obj: float = EPS_OBJ,
    tau: float = TAU,
    min_var: float = MIN_VAR
) -> np.ndarray:
    """
    Compute log-inclusion for all regions in a scene.
    
    Args:
        X: Feature matrix, shape (4, d)
        mask: Boolean mask, shape (4,) - True if region is non-empty
        mu_u: Concept mean, shape (d,)
        var_u: Concept diagonal variance, shape (d,)
        eps_obj: Object variance
        tau: Temperature parameter
        min_var: Minimum variance for stability
        
    Returns:
        Log-inclusion scores, shape (4,).
        For masked-out (empty) regions, returns -inf so they never win.
    """
    n_regions = X.shape[0]
    result = np.full(n_regions, -np.inf, dtype=np.float64)
    
    for t in range(n_regions):
        if mask[t]:
            result[t] = log_inc_single(
                X[t], mu_u, var_u,
                eps_obj=eps_obj, tau=tau, min_var=min_var
            )
    
    return result


def log_inc_matrix_multi(
    X: np.ndarray,
    mask: np.ndarray,
    concepts: List[Tuple[np.ndarray, np.ndarray]],
    *,
    eps_obj: float = EPS_OBJ,
    tau: float = TAU,
    min_var: float = MIN_VAR
) -> np.ndarray:
    """
    Compute log-inclusion for all regions with multiple tokens (AND).
    
    Args:
        X: Feature matrix, shape (4, d)
        mask: Boolean mask, shape (4,)
        concepts: List of (mu_u, var_u) tuples
        eps_obj: Object variance
        tau: Temperature parameter
        min_var: Minimum variance for stability
        
    Returns:
        Log-inclusion scores, shape (4,).
        For masked-out regions, returns -inf.
    """
    n_regions = X.shape[0]
    result = np.full(n_regions, -np.inf, dtype=np.float64)
    
    for t in range(n_regions):
        if mask[t]:
            result[t] = log_inc_multi(
                X[t], concepts,
                eps_obj=eps_obj, tau=tau, min_var=min_var
            )
    
    return result
