"""
Diagonal Gaussian utilities for RSA research project.

Provides:
- Variance clamping for numerical stability
- KL divergence between diagonal Gaussians
- Log-determinant for volume penalty

All functions operate on diagonal covariances represented as 1D arrays.
"""

import numpy as np
from typing import Union


# =============================================================================
# Numerical Safety Utilities
# =============================================================================

def clamp_var(var: np.ndarray, min_var: float = 1e-8) -> np.ndarray:
    """
    Ensure all variance components are >= min_var.
    
    Prevents division by zero, negative variances, and NaN in log operations.
    
    Args:
        var: Variance vector, shape (d,)
        min_var: Minimum allowed variance per dimension
        
    Returns:
        Clamped variance vector, shape (d,)
    """
    return np.maximum(var, min_var)


# =============================================================================
# KL Divergence
# =============================================================================

def kl_diag_gaussians(
    mu_a: np.ndarray,
    var_a: np.ndarray,
    mu_b: np.ndarray,
    var_b: np.ndarray,
    *,
    min_var: float = 1e-8
) -> float:
    """
    Compute KL divergence KL(A || B) for diagonal Gaussians.
    
    Formula (per dimension i, then summed):
        KL = 0.5 * sum_i [ var_a_i/var_b_i + (mu_b_i - mu_a_i)^2/var_b_i 
                          - 1 + ln(var_b_i/var_a_i) ]
    
    Args:
        mu_a: Mean of distribution A, shape (d,)
        var_a: Diagonal variance of A, shape (d,)
        mu_b: Mean of distribution B, shape (d,)
        var_b: Diagonal variance of B, shape (d,)
        min_var: Minimum variance for numerical stability
        
    Returns:
        KL divergence as a Python float (scalar)
        
    Raises:
        ValueError: If result is NaN or infinite
    """
    # Clamp variances for numerical stability
    var_a = clamp_var(var_a, min_var)
    var_b = clamp_var(var_b, min_var)
    
    # Compute each term vectorized
    # Term 1: var_a / var_b
    ratio = var_a / var_b
    
    # Term 2: (mu_b - mu_a)^2 / var_b
    diff_sq = (mu_b - mu_a) ** 2
    mahal = diff_sq / var_b
    
    # Term 3: -1 (constant)
    
    # Term 4: ln(var_b / var_a) = ln(var_b) - ln(var_a)
    log_ratio = np.log(var_b) - np.log(var_a)
    
    # Sum: 0.5 * sum_i [ ratio_i + mahal_i - 1 + log_ratio_i ]
    kl = 0.5 * np.sum(ratio + mahal - 1.0 + log_ratio)
    
    # Convert to Python float
    kl = float(kl)
    
    # Safety check
    if not np.isfinite(kl):
        raise ValueError(
            f"KL divergence is not finite: {kl}. "
            f"var_a range: [{var_a.min()}, {var_a.max()}], "
            f"var_b range: [{var_b.min()}, {var_b.max()}]"
        )
    
    return kl


# =============================================================================
# Log-Determinant (for volume penalty)
# =============================================================================

def logdet_diag(var: np.ndarray, *, min_var: float = 1e-8) -> float:
    """
    Compute log-determinant of diagonal covariance matrix.
    
    For diagonal covariance: log|Σ| = sum_i log(var_i)
    
    Args:
        var: Diagonal variance vector, shape (d,)
        min_var: Minimum variance for numerical stability
        
    Returns:
        Log-determinant as Python float
        
    Raises:
        ValueError: If result is not finite
    """
    var = clamp_var(var, min_var)
    result = float(np.sum(np.log(var)))
    
    if not np.isfinite(result):
        raise ValueError(f"Log-determinant is not finite: {result}")
    
    return result
