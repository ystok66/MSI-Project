"""
Learner module for RSA concept learning.

Provides:
- scale_by_cardinality: Convert probability to soft counts (w = k * p)
- update_concept_soft_counts: Weighted batch merge update for concepts
- learn_step: One-step learning (infer → scale → update all tokens)

Uses numerically stable weighted-merge formula (order-independent).
"""

from typing import List, Dict, Optional, Union
import numpy as np

from concepts import Concept, ConceptTable
from rsa import infer_posterior


# =============================================================================
# Cardinality Scaling
# =============================================================================

def scale_by_cardinality(
    p: np.ndarray,
    k: int,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert probability distribution to soft counts.
    
    w = k * p, with w[t] = 0 for empty regions.
    
    Args:
        p: Probability distribution, shape (4,), sums to 1 over non-empty
        k: Cardinality (number of objects mentioned)
        mask: Boolean mask, shape (4,). If None, all regions are valid.
        
    Returns:
        Soft counts w, shape (4,). Sum equals k over non-empty regions.
    """
    p = np.asarray(p, dtype=np.float64)
    w = k * p
    
    # Zero out empty regions
    if mask is not None:
        mask = np.asarray(mask)
        w[~mask] = 0.0
    
    return w


# =============================================================================
# Soft-Count Update (Weighted Batch Merge)
# =============================================================================

def update_concept_soft_counts(
    concept: Concept,
    X: np.ndarray,
    mask: np.ndarray,
    w: np.ndarray,
    *,
    min_var: float = 1e-8,
    var_floor: float = 1e-6
) -> None:
    """
    Update concept parameters using soft counts (in-place).
    
    Uses numerically stable weighted batch-merge formula:
    - Computes batch statistics (weighted mean, M2)
    - Merges with existing concept statistics
    - Order-independent (same result regardless of batch order)
    
    Formula:
        kappa' = kappa + W
        mu' = mu + (W / kappa') * (mu_batch - mu)
        M2' = M2 + M2_batch + (kappa * W / kappa') * (mu_batch - mu)^2
        var' = M2' / kappa' + var_floor
    
    Args:
        concept: Concept to update (modified in-place)
        X: Feature matrix, shape (4, d)
        mask: Boolean mask, shape (4,)
        w: Soft counts, shape (4,)
        min_var: Minimum variance for clamping
        var_floor: Floor added to variance to prevent collapse
    """
    X = np.asarray(X, dtype=np.float64)
    mask = np.asarray(mask)
    w = np.asarray(w, dtype=np.float64)
    
    # Filter to non-empty regions with positive weight
    valid = mask & (w > 0)
    
    # Total weight from this batch
    W = np.sum(w[valid])
    
    # If no weight, nothing to update
    if W <= 0:
        return
    
    # Get valid observations and weights
    X_valid = X[valid]  # shape (n_valid, d)
    w_valid = w[valid]  # shape (n_valid,)
    
    # Batch weighted mean: mu_batch = sum(w_i * x_i) / W
    mu_batch = np.sum(w_valid[:, np.newaxis] * X_valid, axis=0) / W
    
    # Batch second central moment: M2_batch = sum(w_i * (x_i - mu_batch)^2)
    diff_batch = X_valid - mu_batch  # shape (n_valid, d)
    M2_batch = np.sum(w_valid[:, np.newaxis] * (diff_batch ** 2), axis=0)
    
    # Current concept statistics
    kappa = concept.kappa
    mu = concept.mu
    var = concept.var
    
    # Convert variance to M2 (second central moment accumulator)
    M2 = kappa * var
    
    # Merge statistics
    kappa_new = kappa + W
    
    # Delta between batch mean and current mean
    delta = mu_batch - mu
    
    # Update mean
    mu_new = mu + (W / kappa_new) * delta
    
    # Update M2 (parallel algorithm formula)
    M2_new = M2 + M2_batch + (kappa * W / kappa_new) * (delta ** 2)
    
    # Convert M2 back to variance
    var_new = M2_new / kappa_new + var_floor
    
    # Clamp variance
    var_new = np.maximum(var_new, min_var)
    
    # Update concept in-place
    concept.kappa = kappa_new
    concept.mu = mu_new
    concept.var = var_new


# =============================================================================
# One-Step Learning Entry Point
# =============================================================================

def learn_step(
    X: np.ndarray,
    mask: np.ndarray,
    k: int,
    tokens: List[str],
    table: ConceptTable,
    *,
    alpha: float = 5.0,
    beta: float = 0.1,
    lam: float = 0.0,
    include_empty_alt: bool = True,
    eps_obj: float = 1e-4,
    tau: float = 1.0,
    min_var: float = 1e-8,
    var_floor: float = 1e-6,
    return_debug: bool = False
) -> Optional[Dict]:
    """
    One-step learning: infer posterior, scale by cardinality, update concepts.
    
    For each token in the utterance, updates the concept's (mu, var, kappa)
    using the soft-count weighted observations.
    
    Args:
        X: Feature matrix, shape (4, d)
        mask: Boolean mask, shape (4,)
        k: Cardinality (number of objects)
        tokens: List of token strings
        table: ConceptTable (concepts will be created/updated)
        alpha: RSA rationality parameter
        beta: Volume penalty weight
        lam: Length cost weight
        include_empty_alt: Include empty utterance in alternatives (default True for learning)
        var_floor: Floor added to variance
        return_debug: Whether to return debug info
        
    Returns:
        If return_debug=False: None
        If return_debug=True: Dict with "p", "w", "tokens", "k", etc.
    """
    # Step 1: Infer posterior
    p = infer_posterior(
        X, mask, tokens, table,
        alpha=alpha, beta=beta, lam=lam,
        include_empty_alt=include_empty_alt,
        eps_obj=eps_obj, tau=tau, min_var=min_var
    )
    
    # Step 2: Scale by cardinality
    w = scale_by_cardinality(p, k, mask)
    
    # Step 3: Update each token's concept
    for token in tokens:
        concept = table.ensure(token)
        update_concept_soft_counts(
            concept, X, mask, w,
            min_var=min_var, var_floor=var_floor
        )
    
    if not return_debug:
        return None
    
    return {
        "p": p,
        "w": w,
        "tokens": tokens,
        "k": k
    }
