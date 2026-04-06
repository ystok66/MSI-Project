"""
RSA (Rational Speech Act) inference module.

Provides:
- Alt construction: Build alternative utterances (subsets)
- L0 scoring: Log-inclusion with volume and length penalties
- S1 speaker: Softmax competition over alternatives
- L1 listener: Posterior over referents

Core formula:
    P_L1(t | U) ∝ P(t) * P_S1(U | t)
    P_S1(U | t) ∝ exp(α * S_L0(t, U))
    S_L0(t, U) = log_inc(t, U) - β * vol(U) - λ * |U|
"""

from typing import List, Tuple, Dict, Optional, Union
from itertools import combinations
import numpy as np

from concepts import Concept, ConceptTable
from scoring import log_inc_matrix
from gaussian import logdet_diag


# Type alias
TokenSet = Tuple[str, ...]


# =============================================================================
# Token Normalization
# =============================================================================

def normalize_tokens(tokens: List[str]) -> TokenSet:
    """
    Normalize token list to canonical form.
    
    - Lowercase all tokens
    - Remove empty strings
    - Remove duplicates
    - Sort alphabetically
    
    Args:
        tokens: List of token strings
        
    Returns:
        Sorted tuple of unique lowercase tokens
    """
    # Lowercase, strip whitespace, filter empty
    cleaned = [t.lower().strip() for t in tokens]
    cleaned = [t for t in cleaned if t]
    
    # Remove duplicates and sort
    unique = sorted(set(cleaned))
    
    return tuple(unique)


# =============================================================================
# Alternative Construction
# =============================================================================

def build_alternatives(
    U: TokenSet,
    *,
    include_empty: bool = False,
    extra_tokens: Optional[List[str]] = None
) -> List[TokenSet]:
    """
    Build all alternative utterances (subsets of U, plus optional extra single-tokens).
    
    For |U| tokens, generates 2^|U| - 1 non-empty subsets (or 2^|U| if including empty).
    If extra_tokens is provided, also includes (tok,) for each extra token not in U.
    
    Args:
        U: Token set (normalized tuple)
        include_empty: Whether to include empty set
        extra_tokens: Additional single-token alternatives to include (for mutual exclusivity)
        
    Returns:
        List of TokenSets (all subsets + extra singles)
    """
    tokens = list(U)
    n = len(tokens)
    alts = []
    
    # Start from size 0 or 1
    start_size = 0 if include_empty else 1
    
    for size in range(start_size, n + 1):
        for combo in combinations(tokens, size):
            alts.append(tuple(sorted(combo)))
    
    # Add extra single-token alternatives (for mutual exclusivity)
    if extra_tokens:
        U_set = set(U)
        for tok in extra_tokens:
            tok_norm = tok.lower().strip()
            if tok_norm and tok_norm not in U_set:
                single = (tok_norm,)
                if single not in alts:
                    alts.append(single)
    
    return alts


# =============================================================================
# Numerical Utilities
# =============================================================================

def logsumexp(x: np.ndarray) -> float:
    """
    Numerically stable log-sum-exp.
    
    log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
    """
    x = np.asarray(x)
    
    # Handle all -inf case
    if np.all(x == -np.inf):
        return -np.inf
    
    max_x = np.max(x[np.isfinite(x)])
    return max_x + np.log(np.sum(np.exp(x - max_x)))


def softmax_from_logits(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    
    Args:
        logits: Array of log-unnormalized probabilities
        
    Returns:
        Normalized probabilities (same shape)
    """
    logits = np.asarray(logits)
    
    # Handle all -inf
    if np.all(logits == -np.inf):
        return np.zeros_like(logits)
    
    max_logit = np.max(logits[np.isfinite(logits)])
    exp_logits = np.exp(logits - max_logit)
    return exp_logits / np.sum(exp_logits)


# =============================================================================
# L0 Scoring
# =============================================================================

def log_inc_for_tokens(
    X: np.ndarray,
    mask: np.ndarray,
    concepts: List[Concept],
    *,
    eps_obj: float = 1e-4,
    tau: float = 1.0,
    min_var: float = 1e-8
) -> np.ndarray:
    """
    Compute combined log-inclusion for multiple tokens (AND semantics).
    
    log_inc(t, U) = sum_{u in U} log_inc(t, u)
    
    Args:
        X: Feature matrix, shape (4, d)
        mask: Boolean mask, shape (4,)
        concepts: List of Concept objects
        
    Returns:
        Log-inclusion scores, shape (4,). Empty regions = -inf.
    """
    if len(concepts) == 0:
        # No tokens: return 0 for non-empty regions, -inf for empty
        result = np.zeros(4, dtype=np.float64)
        result[~mask] = -np.inf
        return result
    
    # Sum log_inc over all tokens
    total = np.zeros(4, dtype=np.float64)
    
    for concept in concepts:
        log_inc = log_inc_matrix(
            X, mask, concept.mu, concept.var,
            eps_obj=eps_obj, tau=tau, min_var=min_var
        )
        total += log_inc
    
    return total


def volume_penalty(concepts: List[Concept], *, min_var: float = 1e-8) -> float:
    """
    Compute total volume penalty for a set of concepts.
    
    vol(U) = sum_{u in U} logdet(var_u)
    
    Args:
        concepts: List of Concept objects
        
    Returns:
        Sum of log-determinants
    """
    if len(concepts) == 0:
        return 0.0
    
    total = 0.0
    for concept in concepts:
        total += logdet_diag(concept.var, min_var=min_var)
    
    return total


def score_L0(
    X: np.ndarray,
    mask: np.ndarray,
    U: TokenSet,
    table: ConceptTable,
    *,
    beta: float = 0.1,
    lam: float = 0.0,
    eps_obj: float = 1e-4,
    tau: float = 1.0,
    min_var: float = 1e-8
) -> np.ndarray:
    """
    Compute L0 scores for an utterance over all regions.
    
    S_L0(t, U) = log_inc(t, U) - β * vol(U) - λ * |U|
    
    Args:
        X: Feature matrix, shape (4, d)
        mask: Boolean mask, shape (4,)
        U: Token set (normalized tuple)
        table: ConceptTable for looking up concepts
        beta: Volume penalty weight
        lam: Length cost weight
        
    Returns:
        L0 scores, shape (4,). Empty regions = -inf.
    """
    # Get concepts for all tokens
    concepts = [table.ensure(token) for token in U]
    
    # Log-inclusion (AND over tokens)
    log_inc = log_inc_for_tokens(
        X, mask, concepts,
        eps_obj=eps_obj, tau=tau, min_var=min_var
    )
    
    # Volume penalty (same for all regions)
    vol = volume_penalty(concepts, min_var=min_var)
    
    # Length cost
    length = len(U)
    
    # L0 score
    score = log_inc - beta * vol - lam * length
    
    return score


# =============================================================================
# S1 Speaker
# =============================================================================

def speaker_S1(
    X: np.ndarray,
    mask: np.ndarray,
    U: TokenSet,
    table: ConceptTable,
    *,
    alpha: float = 5.0,
    beta: float = 0.1,
    lam: float = 0.0,
    include_empty_alt: bool = True,
    alt_extra_tokens: Optional[List[str]] = None,
    auto_alt_from_table: bool = True,
    eps_obj: float = 1e-4,
    tau: float = 1.0,
    min_var: float = 1e-8
) -> np.ndarray:
    """
    Compute S1 speaker likelihood: P(U | t).
    
    P_S1(U | t) = exp(α * S_L0(t, U)) / sum_{U' in Alt} exp(α * S_L0(t, U'))
    
    This implements the counterfactual competition: alternatives that could
    have been said but weren't factor into the denominator.
    
    Args:
        X: Feature matrix, shape (4, d)
        mask: Boolean mask, shape (4,)
        U: Token set (normalized tuple)
        table: ConceptTable
        alpha: Rationality parameter (higher = more deterministic)
        beta: Volume penalty weight
        lam: Length cost weight
        include_empty_alt: Whether to include empty utterance in alternatives
        alt_extra_tokens: Additional tokens to include as single-token alternatives (for ME)
        auto_alt_from_table: If True, automatically add all known tokens from table as alternatives
        
    Returns:
        S1 likelihoods, shape (4,). Empty regions = 0.
    """
    # Auto-generate extra tokens from ConceptTable if enabled
    if auto_alt_from_table:
        known_tokens = list(table._concepts.keys())
        if alt_extra_tokens:
            # Merge with existing
            extra = list(alt_extra_tokens) + [t for t in known_tokens if t not in alt_extra_tokens]
        else:
            extra = known_tokens
        
        # Dynamic include_empty_alt selection:
        # Disable empty alt in two scenarios:
        #
        # Scenario A (Mutual Exclusivity):
        #   1. There are known concepts in the table, AND
        #   2. The query contains tokens NOT in the known concepts (novel word)
        #   This enables ME: "speaker chose novel word instead of known word"
        #
        # Scenario B (Scalar Implicature):
        #   1. There are >=2 known concepts in the table
        #   2. The query uses one of them
        #   This enables contrastive inference: "speaker chose 'blue' not 'solid'"
        #
        # Without disabling empty alt, L0=0 overwhelms all alternatives in softmax
        U_tokens = set(U)
        novel_tokens = U_tokens - set(known_tokens)
        
        # Scenario A: ME - known concepts exist AND query has novel word
        me_condition = len(known_tokens) >= 1 and len(novel_tokens) > 0
        
        # Scenario B: Scalar Implicature - multiple known concepts compete
        scalar_condition = len(known_tokens) >= 2
        
        if me_condition or scalar_condition:
            include_empty_alt = False
    else:
        extra = alt_extra_tokens
    
    # Build alternatives (all subsets + extra tokens)
    alts = build_alternatives(U, include_empty=include_empty_alt, extra_tokens=extra)
    
    # Edge case: if U is empty and we don't include empty, alts is empty
    if len(alts) == 0:
        result = np.zeros(4, dtype=np.float64)
        return result
    
    # Compute L0 scores for all alternatives
    # Shape: (num_alts, 4)
    scores = np.zeros((len(alts), 4), dtype=np.float64)
    target_idx = -1
    
    for i, alt in enumerate(alts):
        scores[i] = score_L0(
            X, mask, alt, table,
            beta=beta, lam=lam,
            eps_obj=eps_obj, tau=tau, min_var=min_var
        )
        if alt == U:
            target_idx = i
    
    # If U not in alts (shouldn't happen), raise
    if target_idx < 0:
        raise ValueError(f"Target utterance {U} not found in alternatives")
    
    # For each region t, compute softmax over alternatives
    result = np.zeros(4, dtype=np.float64)
    
    for t in range(4):
        if not mask[t]:
            result[t] = 0.0
            continue
        
        # Scaled logits for this region
        logits = alpha * scores[:, t]
        
        # Compute P(U | t) via softmax
        log_norm = logsumexp(logits)
        log_prob = logits[target_idx] - log_norm
        
        result[t] = np.exp(log_prob)
    
    return result


# =============================================================================
# L1 Listener
# =============================================================================

def listener_L1(
    X: np.ndarray,
    mask: np.ndarray,
    U: TokenSet,
    table: ConceptTable,
    *,
    alpha: float = 5.0,
    beta: float = 0.1,
    lam: float = 0.0,
    include_empty_alt: bool = True,
    alt_extra_tokens: Optional[List[str]] = None,
    auto_alt_from_table: bool = True,
    eps_obj: float = 1e-4,
    tau: float = 1.0,
    min_var: float = 1e-8
) -> np.ndarray:
    """
    Compute L1 listener posterior: P(t | U).
    
    P_L1(t | U) ∝ P(t) * P_S1(U | t)
    
    Prior P(t) is uniform over non-empty regions.
    
    Args:
        X: Feature matrix, shape (4, d)
        mask: Boolean mask, shape (4,)
        U: Token set (normalized tuple)
        table: ConceptTable
        alpha: Rationality parameter
        beta: Volume penalty weight
        lam: Length cost weight
        include_empty_alt: Include empty utterance in alternatives (enables single-token learning)
        alt_extra_tokens: Additional tokens for alternatives (for mutual exclusivity)
        auto_alt_from_table: If True, auto-add all known tokens as alternatives
        
    Returns:
        Posterior, shape (4,). Sums to 1 over non-empty regions.
        
    Raises:
        ValueError: If all regions are empty
    """
    n_nonempty = np.sum(mask)
    
    if n_nonempty == 0:
        raise ValueError("Cannot compute posterior: all regions are empty")
    
    # Uniform prior over non-empty regions
    prior = np.zeros(4, dtype=np.float64)
    prior[mask] = 1.0 / n_nonempty
    
    # S1 likelihood
    s1 = speaker_S1(
        X, mask, U, table,
        alpha=alpha, beta=beta, lam=lam,
        include_empty_alt=include_empty_alt,
        alt_extra_tokens=alt_extra_tokens,
        auto_alt_from_table=auto_alt_from_table,
        eps_obj=eps_obj, tau=tau, min_var=min_var
    )
    
    # Unnormalized posterior
    unnorm = prior * s1
    
    # Normalize
    total = np.sum(unnorm)
    
    if total == 0 or not np.isfinite(total):
        # Fall back to prior if S1 gives all zeros
        return prior.copy()
    
    posterior = unnorm / total
    
    return posterior


# =============================================================================
# Main Entry Point
# =============================================================================

def infer_posterior(
    X: np.ndarray,
    mask: np.ndarray,
    tokens: List[str],
    table: ConceptTable,
    *,
    alpha: float = 5.0,
    beta: float = 0.1,
    lam: float = 0.0,
    include_empty_alt: bool = True,
    alt_extra_tokens: Optional[List[str]] = None,
    auto_alt_from_table: bool = True,
    eps_obj: float = 1e-4,
    tau: float = 1.0,
    min_var: float = 1e-8,
    return_debug: bool = False
) -> Union[np.ndarray, Dict]:
    """
    Main entry point: compute L1 posterior from tokens.
    
    Args:
        X: Feature matrix, shape (4, d)
        mask: Boolean mask, shape (4,)
        tokens: List of token strings
        table: ConceptTable
        alpha: Rationality parameter (higher = more deterministic)
        beta: Volume penalty weight
        lam: Length cost weight
        include_empty_alt: Include empty utterance in alternatives (enables single-token learning)
        alt_extra_tokens: Additional known tokens as alternatives (for mutual exclusivity)
        auto_alt_from_table: If True, auto-add all known tokens as alternatives
        return_debug: Whether to return debug info
        
    Returns:
        If return_debug=False:
            Posterior, shape (4,), sums to 1 over non-empty regions
        If return_debug=True:
            Dict with keys: "U", "alts", "L0", "S1", "p"
    """
    # Normalize tokens
    U = normalize_tokens(tokens)
    
    if not return_debug:
        return listener_L1(
            X, mask, U, table,
            alpha=alpha, beta=beta, lam=lam,
            include_empty_alt=include_empty_alt,
            alt_extra_tokens=alt_extra_tokens,
            auto_alt_from_table=auto_alt_from_table,
            eps_obj=eps_obj, tau=tau, min_var=min_var
        )
    
    # Build debug info - need to get the actual alts that will be used
    if auto_alt_from_table:
        known_tokens = list(table._concepts.keys())
        if alt_extra_tokens:
            extra = list(alt_extra_tokens) + [t for t in known_tokens if t not in alt_extra_tokens]
        else:
            extra = known_tokens
    else:
        extra = alt_extra_tokens
    
    alts = build_alternatives(U, include_empty=include_empty_alt, extra_tokens=extra)
    
    # L0 scores for all alternatives
    L0_scores = {}
    for alt in alts:
        L0_scores[alt] = score_L0(
            X, mask, alt, table,
            beta=beta, lam=lam,
            eps_obj=eps_obj, tau=tau, min_var=min_var
        )
    
    # S1 likelihood for target
    S1 = speaker_S1(
        X, mask, U, table,
        alpha=alpha, beta=beta, lam=lam,
        include_empty_alt=include_empty_alt,
        alt_extra_tokens=alt_extra_tokens,
        auto_alt_from_table=auto_alt_from_table,
        eps_obj=eps_obj, tau=tau, min_var=min_var
    )
    
    # L1 posterior
    p = listener_L1(
        X, mask, U, table,
        alpha=alpha, beta=beta, lam=lam,
        include_empty_alt=include_empty_alt,
        alt_extra_tokens=alt_extra_tokens,
        auto_alt_from_table=auto_alt_from_table,
        eps_obj=eps_obj, tau=tau, min_var=min_var
    )
    
    return {
        "U": U,
        "alts": alts,
        "L0": L0_scores,
        "S1": S1,
        "p": p
    }


# =============================================================================
# Multi-Referent RSA (Set-Valued Intent)
# =============================================================================

def enumerate_subsets(mask: np.ndarray, k: int) -> List[Tuple[int, ...]]:
    """
    Enumerate all subsets of size k from valid (non-empty) regions.
    
    Args:
        mask: Boolean mask, shape (4,), True = valid region
        k: Subset size (cardinality)
        
    Returns:
        List of tuples, each tuple contains indices of the subset
    """
    valid_indices = [i for i in range(len(mask)) if mask[i]]
    
    if k > len(valid_indices):
        return []  # Not enough objects
    
    return list(combinations(valid_indices, k))


def score_L0_set(
    X: np.ndarray,
    mask: np.ndarray,
    U: TokenSet,
    table: ConceptTable,
    T: Tuple[int, ...],
    *,
    beta: float = 0.1,
    lam: float = 0.0,
    min_var: float = 1e-8
) -> float:
    """
    L0 score for a SET of referents T.
    
    log_inc(T, U) = sum_{t in T} sum_{u in U} log_inc(t, u)
    
    This is the AND semantics: every object in T should match U.
    
    Args:
        X: Feature matrix, shape (4, d)
        mask: Boolean mask, shape (4,)
        U: Token set (utterance)
        table: ConceptTable
        T: Tuple of object indices (the target set)
        beta: Volume penalty weight
        lam: Length cost weight
        min_var: Minimum variance
        
    Returns:
        S_L0(T, U) = log_inc(T, U) - beta * vol(U) - lam * |U|
    """
    if len(U) == 0:
        # Empty utterance: return 0 (neutral)
        return 0.0
    
    # Sum over all objects in T and all tokens in U
    # log_inc(T, U) = sum_{t in T} sum_{u in U} log_inc(t, u)
    total_log_inc = 0.0
    
    for t_idx in T:
        if not mask[t_idx]:
            return -np.inf  # Invalid object in set
        
        for u in U:
            c = table.ensure(u)
            # Use log_inc_single from scoring module
            from scoring import log_inc_single
            log_inc_val = log_inc_single(
                X[t_idx], c.mu, c.var,
                min_var=min_var
            )
            total_log_inc += log_inc_val
    
    # Volume penalty: sum of log-determinants
    vol_penalty = 0.0
    for u in U:
        c = table.ensure(u)
        vol_penalty += logdet_diag(c.var)
    
    # Length cost
    length_cost = lam * len(U)
    
    return total_log_inc - beta * vol_penalty - length_cost


def speaker_S1_set(
    X: np.ndarray,
    mask: np.ndarray,
    U: TokenSet,
    table: ConceptTable,
    k: int,
    *,
    alpha: float = 5.0,
    beta: float = 0.1,
    lam: float = 0.0,
    include_empty_alt: bool = True,
    alt_extra_tokens: Optional[List[str]] = None,
    min_var: float = 1e-8
) -> Dict[Tuple[int, ...], float]:
    """
    S1 speaker probability for set-valued referents.
    
    P(U | T) = exp(α * S_L0(T, U)) / Σ_{U'} exp(α * S_L0(T, U'))
    
    Args:
        X: Feature matrix, shape (4, d)
        mask: Boolean mask, shape (4,)
        U: Token set (utterance)
        table: ConceptTable
        k: Cardinality (subset size)
        alpha: Rationality parameter
        beta: Volume penalty weight
        lam: Length cost weight
        include_empty_alt: Include empty utterance
        alt_extra_tokens: Extra tokens for alternatives
        min_var: Minimum variance
        
    Returns:
        Dict mapping each set T (of size k) to P(U | T)
    """
    # Build alternatives
    alts = build_alternatives(U, include_empty=include_empty_alt, extra_tokens=alt_extra_tokens)
    
    # Enumerate all subsets of size k
    subsets = enumerate_subsets(mask, k)
    
    if len(subsets) == 0:
        return {}
    
    # For each subset T, compute P(U | T)
    result = {}
    for T in subsets:
        # Compute L0 scores for all alternatives given T
        logits = []
        for alt in alts:
            score = score_L0_set(X, mask, alt, table, T, beta=beta, lam=lam, min_var=min_var)
            logits.append(alpha * score)
        
        logits = np.array(logits)
        
        # Find index of U in alts
        U_idx = alts.index(U) if U in alts else -1
        
        if U_idx < 0:
            result[T] = 0.0
        else:
            # Softmax to get P(U | T)
            lse = logsumexp(logits)
            result[T] = np.exp(logits[U_idx] - lse)
    
    return result


def listener_L1_set(
    X: np.ndarray,
    mask: np.ndarray,
    U: TokenSet,
    table: ConceptTable,
    k: int,
    *,
    alpha: float = 5.0,
    beta: float = 0.1,
    lam: float = 0.0,
    include_empty_alt: bool = True,
    alt_extra_tokens: Optional[List[str]] = None,
    eps_obj: float = 1e-4,
    min_var: float = 1e-8
) -> Dict[Tuple[int, ...], float]:
    """
    L1 listener posterior for set-valued referents.
    
    P(T | U) ∝ P(T) * P(U | T)
    
    where |T| = k (cardinality constraint).
    
    Args:
        X: Feature matrix, shape (4, d)
        mask: Boolean mask, shape (4,)
        U: Token set (utterance)
        table: ConceptTable
        k: Cardinality (must select exactly k objects)
        alpha: Rationality parameter
        beta: Volume penalty weight
        lam: Length cost weight
        include_empty_alt: Include empty utterance
        alt_extra_tokens: Extra tokens for alternatives
        eps_obj: Prior floor
        min_var: Minimum variance
        
    Returns:
        Dict mapping each set T (of size k) to P(T | U)
    """
    # Get S1 probabilities
    s1 = speaker_S1_set(
        X, mask, U, table, k,
        alpha=alpha, beta=beta, lam=lam,
        include_empty_alt=include_empty_alt,
        alt_extra_tokens=alt_extra_tokens,
        min_var=min_var
    )
    
    if len(s1) == 0:
        return {}
    
    # Uniform prior over subsets of size k
    n_subsets = len(s1)
    prior = 1.0 / n_subsets
    
    # Compute unnormalized posterior
    unnorm = {}
    for T, p_u_given_t in s1.items():
        unnorm[T] = prior * p_u_given_t
    
    # Normalize
    total = sum(unnorm.values())
    if total <= 0:
        # Fallback to uniform
        return {T: 1.0 / n_subsets for T in s1.keys()}
    
    return {T: p / total for T, p in unnorm.items()}


def marginalize_set_posterior(
    set_posterior: Dict[Tuple[int, ...], float],
    n_regions: int = 4
) -> np.ndarray:
    """
    Marginalize set posterior to per-object probabilities.
    
    P(t in T | U) = Σ_{T: t ∈ T} P(T | U)
    
    Args:
        set_posterior: Dict mapping sets to probabilities
        n_regions: Number of regions
        
    Returns:
        Array of shape (n_regions,), P(t selected | U) for each t
    """
    marginal = np.zeros(n_regions)
    
    for T, prob in set_posterior.items():
        for t in T:
            marginal[t] += prob
    
    return marginal


def infer_posterior_set(
    X: np.ndarray,
    mask: np.ndarray,
    tokens: List[str],
    table: ConceptTable,
    k: int,
    *,
    alpha: float = 5.0,
    beta: float = 0.1,
    lam: float = 0.0,
    include_empty_alt: bool = False,
    alt_extra_tokens: Optional[List[str]] = None,
    eps_obj: float = 1e-4,
    min_var: float = 1e-8,
    return_debug: bool = False
) -> Union[Dict[Tuple[int, ...], float], Dict]:
    """
    Multi-referent RSA: Infer posterior over SETS of k objects.
    
    This is the set-valued intent version that supports cardinality constraints.
    For "2 green", use k=2 and tokens=["green"].
    
    Args:
        X: Feature matrix, shape (4, d)
        mask: Boolean mask, shape (4,)
        tokens: List of token strings
        table: ConceptTable
        k: Cardinality (number of objects to select)
        alpha: Rationality parameter
        beta: Volume penalty weight
        lam: Length cost weight
        include_empty_alt: Include empty utterance
        alt_extra_tokens: Extra tokens for alternatives (for ME)
        eps_obj: Prior floor
        min_var: Minimum variance
        return_debug: Return debug info
        
    Returns:
        If return_debug=False:
            Dict mapping each set T (of size k) to P(T | U)
        If return_debug=True:
            Dict with keys: "set_posterior", "marginal", "k", "U"
    """
    U = normalize_tokens(tokens)
    
    set_posterior = listener_L1_set(
        X, mask, U, table, k,
        alpha=alpha, beta=beta, lam=lam,
        include_empty_alt=include_empty_alt,
        alt_extra_tokens=alt_extra_tokens,
        eps_obj=eps_obj, min_var=min_var
    )
    
    if not return_debug:
        return set_posterior
    
    marginal = marginalize_set_posterior(set_posterior, n_regions=len(mask))
    
    return {
        "U": U,
        "k": k,
        "set_posterior": set_posterior,
        "marginal": marginal,
        "subsets": list(set_posterior.keys())
    }


# =============================================================================
# Multi-Intent RSA (e.g., "1 blue, 1 red")
# =============================================================================

# Type alias for intent: (tokens, count)
Intent = Tuple[Tuple[str, ...], int]
# Assignment: tuple of (intent_idx, object_idx) pairs, or just tuple of object indices per intent
Assignment = Tuple[Tuple[int, ...], ...]


def enumerate_multi_intent_assignments(
    mask: np.ndarray,
    intents: List[Intent]
) -> List[Assignment]:
    """
    Enumerate all valid assignments for multiple intents with disjoint objects.
    
    For intents = [(("blue",), 1), (("red",), 1)]:
    - Select 1 object for "blue", 1 object for "red"
    - Objects must be disjoint (same object can't satisfy both)
    
    Args:
        mask: Boolean mask, shape (4,)
        intents: List of (tokens, count) tuples
        
    Returns:
        List of assignments, each assignment is a tuple of object-index tuples.
        assignment[i] = tuple of object indices for intent i
    """
    valid_indices = [i for i in range(len(mask)) if mask[i]]
    n_valid = len(valid_indices)
    
    # Check if we have enough objects
    total_needed = sum(count for _, count in intents)
    if total_needed > n_valid:
        return []
    
    def enumerate_recursive(intent_idx: int, used: set) -> List[Assignment]:
        if intent_idx >= len(intents):
            return [()]  # Base case: empty assignment
        
        tokens, count = intents[intent_idx]
        available = [i for i in valid_indices if i not in used]
        
        if count > len(available):
            return []
        
        # All combinations of 'count' objects from available
        results = []
        for chosen in combinations(available, count):
            new_used = used | set(chosen)
            # Recurse for remaining intents
            for rest in enumerate_recursive(intent_idx + 1, new_used):
                results.append((chosen,) + rest)
        
        return results
    
    return enumerate_recursive(0, set())


def score_L0_multi_intent(
    X: np.ndarray,
    mask: np.ndarray,
    intents: List[Intent],
    assignment: Assignment,
    table: ConceptTable,
    *,
    beta: float = 0.1,
    lam: float = 0.0,
    min_var: float = 1e-8,
    exactness_gamma: float = 0.5
) -> float:
    """
    L0 score for a multi-intent assignment with exactness constraint.
    
    Score = sum over intents of score_L0_set for that intent's objects
            - exactness penalty (soft count should match k)
    
    Args:
        X: Feature matrix
        mask: Boolean mask
        intents: List of (tokens, count) tuples
        assignment: Tuple of object-index tuples for each intent
        table: ConceptTable
        exactness_gamma: Penalty weight for count mismatch
        
    Returns:
        Total L0 score for this assignment
    """
    from scoring import log_inc_single
    
    total_score = 0.0
    
    # Get all objects in the target set (union of all intent assignments)
    all_target_objects = set()
    for obj_tuple in assignment:
        all_target_objects.update(obj_tuple)
    
    for intent_idx, (tokens, count) in enumerate(intents):
        objects = assignment[intent_idx]
        
        # Score each object against the intent's tokens
        for t_idx in objects:
            if not mask[t_idx]:
                return -np.inf
            
            for token in tokens:
                c = table.ensure(token)
                log_inc_val = log_inc_single(
                    X[t_idx], c.mu, c.var,
                    min_var=min_var
                )
                total_score += log_inc_val
        
        # Volume penalty for this intent's tokens (scaled by count k)
        for token in tokens:
            c = table.ensure(token)
            total_score -= beta * count * logdet_diag(c.var)
        
        # Length cost
        total_score -= lam * len(tokens)
        
        # === EXACTNESS CONSTRAINT ===
        # "1 b" should mean exactly 1 b in the target set
        # Compute soft count: sum of inc scores for this intent over ALL target objects
        # Penalize if soft count differs from k
        if exactness_gamma > 0:
            for token in tokens:
                c = table.ensure(token)
                # Soft count: how many objects in target match this token?
                # Use normalized probability-like scores
                soft_count = 0.0
                for t_idx in all_target_objects:
                    inc = log_inc_single(X[t_idx], c.mu, c.var, min_var=min_var)
                    # Convert to probability-like weight (0 to 1)
                    # Sigmoid centered at -38 with scale 0.2 for good discrimination
                    weight = 1.0 / (1.0 + np.exp(-0.2 * (inc + 38)))
                    soft_count += weight
                
                # Penalty: (soft_count - k)^2
                total_score -= exactness_gamma * (soft_count - count) ** 2
    
    return total_score


# =============================================================================
# Pure RSA Competition Mode (≥k semantics)
# =============================================================================

def enumerate_variable_size_assignments(
    mask: np.ndarray,
    intents: List[Intent],
    max_extra: int = 2
) -> List[Assignment]:
    """
    Enumerate assignments where |T_j| >= k_j (variable size).
    
    For pure RSA competition, we need variable-size subsets so the 
    speaker can choose between "2 blue" and "3 blue" alternatives.
    
    Args:
        mask: Boolean mask, shape (4,)
        intents: List of (tokens, count) tuples
        max_extra: Maximum extra objects per intent beyond k_j
        
    Returns:
        List of assignments. Each assignment[i] has len >= k_i.
    """
    valid_indices = [i for i in range(len(mask)) if mask[i]]
    n_valid = len(valid_indices)
    
    total_min = sum(count for _, count in intents)
    if total_min > n_valid:
        return []
    
    def enumerate_recursive(intent_idx: int, used: set) -> List[Assignment]:
        if intent_idx >= len(intents):
            return [()]
        
        tokens, min_k = intents[intent_idx]
        available = [i for i in valid_indices if i not in used]
        
        if min_k > len(available):
            return []
        
        results = []
        max_k = min(min_k + max_extra, len(available))
        
        for size in range(min_k, max_k + 1):
            for chosen in combinations(available, size):
                new_used = used | set(chosen)
                for rest in enumerate_recursive(intent_idx + 1, new_used):
                    results.append((chosen,) + rest)
        
        return results
    
    return enumerate_recursive(0, set())


def score_L0_geq_k(
    X: np.ndarray,
    mask: np.ndarray,
    intents: List[Intent],
    assignment: Assignment,
    table: ConceptTable,
    *,
    beta: float = 0.1,
    lam: float = 0.0,
    min_var: float = 1e-8,
    eta: float = 0.02
) -> float:
    """
    L0 score with >= k semantics (pure RSA competition mode).
    
    No exactness penalty — "exactly k" emerges from scalar implicature
    via numeral alternatives in S1.
    
    Score = Σ_j [log_inc(T_j, W_j) - β·|T_j|·vol(W_j) - λ·|W_j|] - η·|T_total|
    
    The ≥k gate: if |T_j| < k_j → -inf.
    
    Args:
        X: Feature matrix
        mask: Boolean mask
        intents: List of (tokens, count) tuples. count = minimum k.
        assignment: Tuple of object-index tuples for each intent
        table: ConceptTable
        beta: Volume penalty weight
        lam: Length cost weight
        eta: Weak set-size prior (prevents L0 from preferring large sets)
        
    Returns:
        L0 score (float). -inf if any intent violates ≥k constraint.
    """
    from scoring import log_inc_single
    
    total_score = 0.0
    total_objects = 0
    
    for intent_idx, (tokens, min_k) in enumerate(intents):
        objects = assignment[intent_idx]
        
        # ≥k gate
        if len(objects) < min_k:
            return -np.inf
        
        # Score each object against the intent's tokens
        for t_idx in objects:
            if not mask[t_idx]:
                return -np.inf
            
            for token in tokens:
                c = table.ensure(token)
                log_inc_val = log_inc_single(
                    X[t_idx], c.mu, c.var,
                    min_var=min_var
                )
                total_score += log_inc_val
        
        # Volume penalty scaled by actual count
        for token in tokens:
            c = table.ensure(token)
            total_score -= beta * len(objects) * logdet_diag(c.var)
        
        # Length cost
        total_score -= lam * len(tokens)
        
        total_objects += len(objects)
    
    # Weak set-size prior: prevent L0 from preferring large sets
    total_score -= eta * total_objects
    
    return total_score


def _infer_pure_rsa(
    X: np.ndarray,
    mask: np.ndarray,
    norm_intents: List[Intent],
    table: ConceptTable,
    *,
    alpha: float = 5.0,
    beta: float = 0.1,
    lam: float = 0.0,
    min_var: float = 1e-8,
    eta: float = 0.02,
    use_rsa: bool = True,
    return_debug: bool = False
) -> Union[Dict[Assignment, float], Dict]:
    """
    Pure RSA competition inference (internal helper).
    
    "Exactly k" emerges from scalar implicature:
    - L0 uses "≥k" semantics (no exactness penalty)
    - Alt includes different numerals: (1,W), (2,W), ..., (n,W)
    - S1 = softmax over Alt via utility = α·ln P_L0 - Cost
    - L1 = Bayes inversion
    
    Soft counts come from posterior marginalization, not sigmoid gates.
    """
    n_valid = sum(1 for m in mask if m)
    
    # Enumerate variable-size assignments (≥k per intent, disjoint)
    assignments = enumerate_variable_size_assignments(mask, norm_intents)
    
    if len(assignments) == 0:
        return {} if not return_debug else {"assignments": [], "posterior": {}}
    
    if not use_rsa:
        # L0-only mode: just score and normalize, no alternatives
        logits = []
        for assignment in assignments:
            score = score_L0_geq_k(
                X, mask, norm_intents, assignment, table,
                beta=beta, lam=lam, min_var=min_var, eta=eta
            )
            logits.append(alpha * score)
        
        logits_arr = np.array(logits)
        if np.all(logits_arr == -np.inf):
            posterior = {a: 1.0 / len(assignments) for a in assignments}
        else:
            lse = logsumexp(logits_arr)
            posterior = {a: np.exp(logits_arr[i] - lse)
                         for i, a in enumerate(assignments)}
        
        if not return_debug:
            return posterior
        return {"intents": norm_intents, "assignments": assignments,
                "posterior": posterior, "mode": "pure_rsa_l0"}
    
    # ======== Full RSA with numeral alternatives ========
    
    # Build alternative utterances: vary numerals for each token set
    # For "1 blue, 1 solid": Alt includes
    #   (1,blue)+(1,solid), (2,blue), (2,solid),
    #   (1,blue)+(2,solid), etc. — all numeral combinations
    
    known_tokens = list(table._concepts.keys())
    
    # Collect all unique token sets from current intents
    all_token_sets = list(set(tokens for tokens, _ in norm_intents))
    # Add known tokens as single-token alternatives
    for tok in known_tokens:
        ts = (tok,)
        if ts not in all_token_sets:
            all_token_sets.append(ts)
    
    # Build Alt: for each token set, vary numeral from 1 to n_valid
    alt_intents_list: List[List[Intent]] = [list(norm_intents)]  # current first
    
    # Single-intent alternatives: "(k, W)" for each W and each k
    for token_set in all_token_sets:
        for k in range(1, n_valid + 1):
            alt = [(token_set, k)]
            if alt not in alt_intents_list:
                alt_intents_list.append(alt)
    
    # Multi-intent alternatives: "(1, W_a), (1, W_b)" for a != b
    if len(norm_intents) >= 2:
        for ts_a in all_token_sets:
            for ts_b in all_token_sets:
                if ts_a != ts_b:
                    alt = [(ts_a, 1), (ts_b, 1)]
                    if alt not in alt_intents_list:
                        alt_intents_list.append(alt)
    
    num_alts = len(alt_intents_list)
    
    # For each alt, enumerate its own valid assignments
    # and compute L0 scores using ≥k semantics
    
    # Precompute: for each alternative u', compute P_L0(A | u')
    # We need a unified assignment space across all alternatives.
    # Strategy: each alt has its own assignment space; we compute
    # P_L0(A | u') only for assignments compatible with the observed
    # target object set.
    
    # Step 1: Compute L0 scores for observed utterance across all assignments
    obs_l0_logits = []
    for a in assignments:
        score = score_L0_geq_k(
            X, mask, norm_intents, a, table,
            beta=beta, lam=lam, min_var=min_var, eta=eta
        )
        obs_l0_logits.append(alpha * score)
    obs_l0_logits = np.array(obs_l0_logits)
    
    # Normalize P_L0(A | u_obs) over assignments
    if np.all(obs_l0_logits == -np.inf):
        obs_l0_post = np.full(len(assignments), 1.0 / len(assignments))
    else:
        lse = logsumexp(obs_l0_logits)
        obs_l0_post = np.exp(obs_l0_logits - lse)
    
    # Step 2: For each assignment A (= target), compute S1 utility
    # Utility(u, A) = α · ln P_L0(A | u) - Cost(u)
    # P_S1(u | A) = softmax over Alt
    
    s1_log_probs = {}  # assignment -> log P_S1(u_obs | A)
    
    for ai, assignment in enumerate(assignments):
        # Get the target object set for this assignment
        target_set = set()
        for obj_tuple in assignment:
            target_set.update(obj_tuple)
        target_list = sorted(target_set)
        target_size = len(target_list)
        
        # Compute utility for each alternative utterance
        alt_utilities = []
        obs_idx = 0  # index of observed utterance in alt_intents_list
        
        for ui, alt_intent in enumerate(alt_intents_list):
            # For this alternative, enumerate its assignments that
            # match the same target object set
            alt_total_k = sum(cnt for _, cnt in alt_intent)
            
            if alt_total_k > target_size:
                # This alt requires more objects than our target has
                alt_utilities.append(-np.inf)
                continue
            
            # Enumerate assignments for this alt with ≥k semantics
            alt_assignments = enumerate_variable_size_assignments(
                mask, alt_intent
            )
            
            if len(alt_assignments) == 0:
                alt_utilities.append(-np.inf)
                continue
            
            # Compute L0 scores for all alt assignments
            alt_logits = []
            target_logit = -np.inf
            for aa in alt_assignments:
                s = score_L0_geq_k(
                    X, mask, alt_intent, aa, table,
                    beta=beta, lam=lam, min_var=min_var, eta=eta
                )
                alt_logits.append(alpha * s)
                
                # Check if this alt assignment covers same objects
                aa_set = set()
                for obj_tuple in aa:
                    aa_set.update(obj_tuple)
                if aa_set == target_set:
                    target_logit = max(target_logit, alpha * s)
            
            alt_logits_arr = np.array(alt_logits)
            
            if np.all(alt_logits_arr == -np.inf) or target_logit == -np.inf:
                alt_utilities.append(-np.inf)
                continue
            
            # P_L0(target | u') = exp(target_logit) / sum(exp(all_logits))
            lse_alt = logsumexp(alt_logits_arr)
            log_p_l0 = target_logit - lse_alt
            
            # Cost(u') = λ · |u'|  (β·vol is in φ/L0 only)
            cost = 0.0
            for tokens, cnt in alt_intent:
                cost += lam * len(tokens)
            
            utility = alpha * log_p_l0 - cost
            alt_utilities.append(utility)
        
        # P_S1(u_obs | A) via softmax over alternatives
        alt_utilities_arr = np.array(alt_utilities)
        
        if np.all(alt_utilities_arr == -np.inf):
            s1_log_probs[assignment] = -np.inf
        else:
            lse_s1 = logsumexp(alt_utilities_arr)
            s1_log_probs[assignment] = alt_utilities_arr[obs_idx] - lse_s1
    
    # Step 3: L1 posterior
    # P_L1(A | u) ∝ P(A) · P_S1(u | A)
    log_prior = -np.log(len(assignments))
    
    log_posterior = []
    for assignment in assignments:
        lp = log_prior + s1_log_probs.get(assignment, -np.inf)
        log_posterior.append(lp)
    
    log_posterior_arr = np.array(log_posterior)
    
    if np.all(log_posterior_arr == -np.inf):
        posterior = {a: 1.0 / len(assignments) for a in assignments}
    else:
        lse = logsumexp(log_posterior_arr)
        posterior = {a: np.exp(log_posterior_arr[i] - lse)
                     for i, a in enumerate(assignments)}
    
    if not return_debug:
        return posterior
    
    return {
        "intents": norm_intents,
        "assignments": assignments,
        "posterior": posterior,
        "alternatives": alt_intents_list,
        "s1_log_probs": s1_log_probs,
        "mode": "pure_rsa"
    }



def infer_posterior_multi_intent(
    X: np.ndarray,
    mask: np.ndarray,
    intents: List[Tuple[List[str], int]],
    table: ConceptTable,
    *,
    alpha: float = 5.0,
    beta: float = 0.1,
    lam: float = 0.0,
    min_var: float = 1e-8,
    exactness_gamma: float = 2.0,
    exactness_mode: str = "soft_count",
    eta: float = 0.02,
    use_rsa: bool = True,
    return_debug: bool = False
) -> Union[Dict[Assignment, float], Dict]:
    """
    Full Multi-Intent RSA with S1 speaker reasoning.
    
    For "1 blue, 1 red":
        intents = [(["blue"], 1), (["red"], 1)]
    
    Two exactness modes:
        "soft_count": Sigmoid-based soft counting + γ penalty (default)
        "pure_rsa":   ≥k semantics + numeral alternatives; exactness
                      emerges from scalar implicature (no penalty term)
    
    Args:
        X: Feature matrix, shape (4, d)
        mask: Boolean mask, shape (4,)
        intents: List of (tokens, count) tuples
        table: ConceptTable
        alpha: Rationality parameter
        beta: Volume penalty
        lam: Length cost
        min_var: Minimum variance
        exactness_gamma: Penalty weight for soft_count mode
        exactness_mode: "soft_count" or "pure_rsa"
        eta: Weak set-size prior for pure_rsa mode
        use_rsa: If True, use full RSA with alternatives; if False, use L0 only
        return_debug: Return debug info
        
    Returns:
        Dict mapping assignments to posterior probabilities
    """
    # Normalize intents
    norm_intents: List[Intent] = []
    for tokens, count in intents:
        norm_tokens = normalize_tokens(tokens)
        norm_intents.append((norm_tokens, count))
    
    # Total count
    total_k = sum(count for _, count in norm_intents)
    
    # ================================================================
    # Pure RSA Competition Mode
    # ================================================================
    if exactness_mode == "pure_rsa":
        return _infer_pure_rsa(
            X, mask, norm_intents, table,
            alpha=alpha, beta=beta, lam=lam,
            min_var=min_var, eta=eta,
            use_rsa=use_rsa, return_debug=return_debug
        )
    
    # ================================================================
    # Soft Count Mode (existing behavior)
    # ================================================================
    
    # Enumerate all valid target sets (disjoint assignments)
    assignments = enumerate_multi_intent_assignments(mask, norm_intents)
    
    if len(assignments) == 0:
        return {} if not return_debug else {"assignments": [], "posterior": {}}
    
    if not use_rsa:
        # L0-only mode (original behavior)
        logits = []
        for assignment in assignments:
            score = score_L0_multi_intent(
                X, mask, norm_intents, assignment, table,
                beta=beta, lam=lam, min_var=min_var,
                exactness_gamma=exactness_gamma
            )
            logits.append(alpha * score)
        
        logits = np.array(logits)
        if np.all(logits == -np.inf):
            posterior = {a: 1.0 / len(assignments) for a in assignments}
        else:
            lse = logsumexp(logits)
            posterior = {a: np.exp(logits[i] - lse) for i, a in enumerate(assignments)}
        
        if not return_debug:
            return posterior
        return {"intents": norm_intents, "assignments": assignments, "posterior": posterior}
    
    # ========== Full RSA with S1 speaker reasoning ==========
    
    # Build alternative utterances from ALL known concepts in table
    # This allows proper comparison: e.g., "1 b, 1 solid" vs "1 r, 1 solid"
    
    known_tokens = list(table._concepts.keys())
    
    # Build alternative intents
    alt_intents_list = [norm_intents]  # Current utterance first
    
    if len(norm_intents) == 2 and all(cnt == 1 for _, cnt in norm_intents):
        # For "1 X, 1 Y" pattern: generate "1 A, 1 B" for A != B only
        # "1 X, 1 X" is semantically equivalent to "2 X", so we skip it
        for tok1 in known_tokens:
            for tok2 in known_tokens:
                if tok1 == tok2:
                    continue  # Skip "1 X, 1 X" - handled by "2 X" below
                alt = [((tok1,), 1), ((tok2,), 1)]
                if alt not in alt_intents_list:
                    alt_intents_list.append(alt)
    
    # Add single-token alternatives: "k token" for ALL known tokens
    # This covers "2 b", "2 solid", etc.
    for tok in known_tokens:
        alt = [((tok,), total_k)]
        if alt not in alt_intents_list:
            alt_intents_list.append(alt)

    # ========== STEP 1: Compute L0 scores for all (T, U') pairs ==========
    # l0_scores[assignment][alt_idx] = L0 score
    l0_scores = {}
    
    for assignment in assignments:
        target_objects = set()
        for obj_tuple in assignment:
            target_objects.update(obj_tuple)
        target_objects_list = list(sorted(target_objects))
        
        l0_scores[assignment] = []
        
        for alt_intents in alt_intents_list:
            from scoring import log_inc_single
            
            alt_total_k = sum(cnt for _, cnt in alt_intents)
            
            if alt_total_k != len(target_objects_list):
                l0_scores[assignment].append(-np.inf)
                continue
            
            if len(alt_intents) == 1:
                # Single intent scoring
                tokens, count = alt_intents[0]
                score = 0.0
                for t_idx in target_objects_list:
                    for token in tokens:
                        c = table.ensure(token)
                        log_inc = log_inc_single(X[t_idx], c.mu, c.var, min_var=min_var)
                        score += log_inc
                for token in tokens:
                    c = table.ensure(token)
                    score -= beta * count * logdet_diag(c.var)
                # Exactness
                if exactness_gamma > 0:
                    for token in tokens:
                        c = table.ensure(token)
                        soft_count = 0.0
                        for t_idx in target_objects_list:
                            inc = log_inc_single(X[t_idx], c.mu, c.var, min_var=min_var)
                            weight = 1.0 / (1.0 + np.exp(-0.2 * (inc + 38)))
                            soft_count += weight
                        score -= exactness_gamma * (soft_count - count) ** 2
                l0_scores[assignment].append(score)
            else:
                # Multi-intent scoring
                if alt_intents == norm_intents:
                    score = score_L0_multi_intent(
                        X, mask, alt_intents, assignment, table,
                        beta=beta, lam=lam, min_var=min_var,
                        exactness_gamma=exactness_gamma
                    )
                    l0_scores[assignment].append(score)
                else:
                    best_score = -np.inf
                    try:
                        alt_assignments = enumerate_multi_intent_assignments(mask, alt_intents)
                        for alt_a in alt_assignments:
                            alt_objects = set()
                            for obj_tuple in alt_a:
                                alt_objects.update(obj_tuple)
                            if alt_objects == set(target_objects_list):
                                score = score_L0_multi_intent(
                                    X, mask, alt_intents, alt_a, table,
                                    beta=beta, lam=lam, min_var=min_var,
                                    exactness_gamma=exactness_gamma
                                )
                                best_score = max(best_score, score)
                    except:
                        pass
                    l0_scores[assignment].append(best_score)
    
    # ========== STEP 2: Normalize L0 across targets for each utterance ==========
    # P_L0(T | U') = exp(α * L0(T, U')) / Σ_T' exp(α * L0(T', U'))
    num_alts = len(alt_intents_list)
    l0_posterior = {}  # assignment -> list of P_L0(T | U') for each alt
    
    for alt_idx in range(num_alts):
        # Get scores for this utterance across all targets
        alt_logits = [alpha * l0_scores[a][alt_idx] for a in assignments]
        alt_logits_arr = np.array(alt_logits)
        
        if np.all(alt_logits_arr == -np.inf):
            for i, a in enumerate(assignments):
                if a not in l0_posterior:
                    l0_posterior[a] = []
                l0_posterior[a].append(1.0 / len(assignments))
        else:
            lse = logsumexp(alt_logits_arr)
            for i, a in enumerate(assignments):
                if a not in l0_posterior:
                    l0_posterior[a] = []
                l0_posterior[a].append(np.exp(alt_logits_arr[i] - lse))
    
    # ========== STEP 3: Compute S1(U | T) using L0 posterior ==========
    # 
    # CORRECT Informativeness-Aware S1:
    #   S1(U | T) ∝ [P_L0(T | U)]^α
    #
    # This is the key insight: S1 should reflect how well the utterance U
    # distinguishes target T from OTHER targets, not how well U compares
    # to other utterances for describing T.
    #
    # P_L0(T | U) was already computed in Step 2 (normalized across targets).
    # A high P_L0(T | U) means: "given utterance U, target T is likely"
    # This is exactly what informativeness measures!
    #
    # OLD (WRONG) approach: S1 = softmax across utterances
    #   This made S1(Wrong) ≈ S1(Correct) because all alternative utterances
    #   were equally bad for both targets.
    #
    # NEW (CORRECT) approach: S1 = P_L0(T | U) directly (already normalized)
    #   This preserves the 20:1 ratio from L0 posterior.
    
    s1_scores = {}
    
    for assignment in assignments:
        # S1(U | T) = P_L0(T | U) for current utterance (index 0)
        # This is already normalized across targets in Step 2
        p_l0 = l0_posterior[assignment][0]
        
        # Apply alpha exponent for rationality scaling
        # Higher alpha = speaker is more rational about choosing informative utterances
        s1_scores[assignment] = p_l0 ** alpha
    
    # L1: P(T | U) ∝ P(T) * S1(U | T)
    # Uniform prior over assignments
    prior = 1.0 / len(assignments)
    
    unnorm = {}
    for assignment in assignments:
        unnorm[assignment] = prior * s1_scores[assignment]
    
    total = sum(unnorm.values())
    if total <= 0:
        posterior = {a: 1.0 / len(assignments) for a in assignments}
    else:
        posterior = {a: p / total for a, p in unnorm.items()}
    
    if not return_debug:
        return posterior
    
    return {
        "intents": norm_intents,
        "assignments": assignments,
        "posterior": posterior,
        "s1_scores": s1_scores,
        "alternatives": alt_intents_list
    }
