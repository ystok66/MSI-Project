"""Compositional Goal Prior — generative prior P₀(g|c₀).

Replaces the legacy compatibility bonus exp(β_C · C_t(g)) with a proper
prior that depends ONLY on episode-start context, never on observed actions.

Three prior features:
    complexity:  |g| - 1  (Occam penalty for composites)
    redundancy:  R(g; c₀) (penalizes near-duplicate constituent overlap)
    feasibility: F(g; c₀) ∈ {0, 1} (hard mask; goal must be valid in context)

Hyperparameters (only 2 free):
    β_len:  complexity penalty weight
    β_red:  redundancy penalty weight

Formula:
    log P₀(g|c₀) = β_feas·F(g;c₀) − β_len·complexity(g) − β_red·R(g;c₀) − log Z(c₀)

where β_feas is not a free parameter — it's a hard mask (F=0 → log P = -∞).

Does NOT modify any frozen module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np

from .compositional_goal_hypotheses import (
    GoalHypothesis, GoalHypothesisSpace, DEFAULT_GOAL_SPACE,
    ATOMIC_GOALS, ATOMIC_GOAL_WEIGHTS, VALID_COMPOSITES,
)


# ═══════════════════════════════════════════════════════════
# Prior Context (episode-start only)
# ═══════════════════════════════════════════════════════════

@dataclass
class GoalPriorContext:
    """Episode-start context for building the goal prior.

    Must NOT depend on any observed actions.
    """
    family: str = ""
    scenario_type: str = ""         # goal_aligned, goal_conflict, goal_boundary
    available_goals: Optional[List[str]] = None   # None = all valid
    branch_attrs_0: Optional[list] = None         # initial branch attributes (optional)
    n_branches: int = 2
    # CGC-v2 metadata (read-only)
    goal_registry: Optional[Dict] = None


@dataclass
class GoalPriorConfig:
    """Hyperparameters for the structural prior.

    Only 2 free parameters: β_len and β_red.
    β_feas is a hard mask, not a tunable weight.
    """
    beta_len: float = 1.0     # complexity penalty
    beta_red: float = 0.5     # redundancy penalty


# ═══════════════════════════════════════════════════════════
# Prior Feature Functions
# ═══════════════════════════════════════════════════════════

def compute_goal_complexity(goal: GoalHypothesis) -> float:
    """complexity(g) = |g| - 1.

    Atomic → 0, Composite(2 parts) → 1.
    Penalizes composite hypotheses under Occam's razor.
    """
    return float(len(goal.components) - 1)


def compute_redundancy(goal: GoalHypothesis,
                       context: GoalPriorContext) -> float:
    """R(g; c₀) — structural redundancy penalty.

    For atomic goals: 0 (no redundancy possible).
    For composite goals: cosine similarity between constituent reward weights.
    High cosine → observationally near-equivalent → harder to disambiguate.

    Does NOT depend on observed actions — only on goal structure.
    """
    if not goal.is_composite:
        return 0.0

    # Pairwise reward-weight similarity among constituents
    ws = [ATOMIC_GOAL_WEIGHTS[c] for c in goal.components]
    if len(ws) < 2:
        return 0.0

    # Cosine similarity between first two constituents
    w1, w2 = ws[0], ws[1]
    n1 = float(np.linalg.norm(w1))
    n2 = float(np.linalg.norm(w2))
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0

    cos_sim = float(np.dot(w1, w2) / (n1 * n2))
    # Clip to [0, 1] — negative similarity means complementary (good)
    return max(0.0, cos_sim)


def compute_feasibility(goal: GoalHypothesis,
                        context: GoalPriorContext) -> bool:
    """F(g; c₀) ∈ {0, 1} — hard validity mask.

    A goal is feasible if:
    1. All its atomic components are valid goals
    2. If composite, the pair is in VALID_COMPOSITES
    3. If context specifies available_goals, all components must be in it

    This is a HARD mask, not a soft weight.
    """
    # Check all components are valid atomic goals
    for c in goal.components:
        if c not in ATOMIC_GOALS:
            return False

    # Check composite validity
    if goal.is_composite:
        pair = tuple(sorted(goal.components))
        valid_pairs = {tuple(sorted(vc)) for vc in VALID_COMPOSITES}
        if pair not in valid_pairs:
            return False

    # Check context availability constraint
    if context.available_goals is not None:
        for c in goal.components:
            if c not in context.available_goals:
                return False

    return True


# ═══════════════════════════════════════════════════════════
# Log Prior Computation
# ═══════════════════════════════════════════════════════════

def compute_log_goal_prior(
    goal: GoalHypothesis,
    context: GoalPriorContext,
    cfg: GoalPriorConfig,
) -> float:
    """log P₀(g|c₀) (unnormalized).

    log P₀(g|c₀) = -∞     if not feasible
                  = -β_len·complexity(g) - β_red·R(g;c₀)  otherwise
    """
    if not compute_feasibility(goal, context):
        return -np.inf

    complexity = compute_goal_complexity(goal)
    redundancy = compute_redundancy(goal, context)

    return -(cfg.beta_len * complexity + cfg.beta_red * redundancy)


def compute_log_goal_prior_vector(
    goal_space: GoalHypothesisSpace,
    context: GoalPriorContext,
    cfg: GoalPriorConfig,
) -> np.ndarray:
    """Compute log P₀(g|c₀) for all hypotheses.

    Returns:
        (n_goals,) array of unnormalized log-priors
    """
    log_priors = np.array([
        compute_log_goal_prior(gh, context, cfg)
        for gh in goal_space.hypotheses
    ])
    return log_priors


def compute_normalized_goal_prior(
    goal_space: GoalHypothesisSpace,
    context: GoalPriorContext,
    cfg: GoalPriorConfig,
) -> np.ndarray:
    """Normalized P₀(g|c₀) vector.

    Returns:
        (n_goals,) array summing to 1
    """
    log_p = compute_log_goal_prior_vector(goal_space, context, cfg)

    # Handle all -inf case (degenerate)
    if np.all(~np.isfinite(log_p)):
        return np.ones(len(log_p)) / len(log_p)

    # Set -inf to very negative for softmax
    finite_mask = np.isfinite(log_p)
    log_p_safe = np.where(finite_mask, log_p, -100.0)
    log_p_safe -= np.max(log_p_safe)
    priors = np.exp(log_p_safe)
    priors[~finite_mask] = 0.0
    total = priors.sum()
    if total > 0:
        priors /= total
    else:
        priors = np.ones(len(priors)) / len(priors)
    return priors


# ═══════════════════════════════════════════════════════════
# Subgoal Marginal (primary metric)
# ═══════════════════════════════════════════════════════════

def compute_subgoal_marginals(
    goal_marginal: Dict[str, float],
    goal_space: GoalHypothesisSpace,
) -> Dict[str, float]:
    """q(u) = Σ_{g ∋ u} q(g) for each atomic subgoal u.

    This is the Step 4 primary metric — replaces composite top-1.

    Args:
        goal_marginal: {label: P(g)} from posterior
        goal_space: hypothesis space

    Returns:
        {atomic_goal: marginal_prob}
    """
    marginals = {u: 0.0 for u in ATOMIC_GOALS}

    for gh in goal_space.hypotheses:
        prob = goal_marginal.get(gh.label, 0.0)
        for component in gh.components:
            if component in marginals:
                marginals[component] += prob

    return marginals


# ═══════════════════════════════════════════════════════════
# PCFG Prior (Scheme B — paper comparison only)
# ═══════════════════════════════════════════════════════════

@dataclass
class PCFGPriorConfig:
    """PCFG prior parameters.

    Grammar:
        C → P           with prob p_atomic
        C → P ∧ P       with prob p_compose
        P → collect_red | avoid_blue | use_safe | reach_fast  (uniform)

    Only 2 parameters.
    """
    p_atomic: float = 0.7     # prob of atomic derivation
    p_compose: float = 0.3    # prob of composite derivation


def compute_pcfg_log_prior(
    goal: GoalHypothesis,
    pcfg_cfg: PCFGPriorConfig,
) -> float:
    """log P₀(g) under PCFG grammar.

    Atomic: log p_atomic + log(1/4)
    Composite: log p_compose + 2·log(1/4) - log(2)  [order-invariant]
    """
    n_atomic = len(ATOMIC_GOALS)

    if not goal.is_composite:
        return np.log(pcfg_cfg.p_atomic) + np.log(1.0 / n_atomic)
    else:
        # Composite: p_compose × (1/4) × (1/4) / 2 (unordered pair)
        return (np.log(pcfg_cfg.p_compose)
                + 2 * np.log(1.0 / n_atomic)
                - np.log(2.0))


def compute_pcfg_log_prior_vector(
    goal_space: GoalHypothesisSpace,
    pcfg_cfg: PCFGPriorConfig,
) -> np.ndarray:
    """PCFG log-prior for all hypotheses."""
    return np.array([
        compute_pcfg_log_prior(gh, pcfg_cfg)
        for gh in goal_space.hypotheses
    ])


# ═══════════════════════════════════════════════════════════
# Prior Feature Correlation Audit (redundancy check)
# ═══════════════════════════════════════════════════════════

def audit_prior_features(
    goal_space: GoalHypothesisSpace,
    context: GoalPriorContext,
) -> Dict[str, float]:
    """Audit correlation between prior features.

    Returns pairwise correlations to detect redundant features.
    """
    n = goal_space.n_goals
    complexity = np.array([compute_goal_complexity(gh) for gh in goal_space.hypotheses])
    redundancy = np.array([compute_redundancy(gh, context) for gh in goal_space.hypotheses])
    feasibility = np.array([float(compute_feasibility(gh, context))
                            for gh in goal_space.hypotheses])

    result = {
        "n_goals": n,
        "n_feasible": int(feasibility.sum()),
        "mean_complexity": float(complexity.mean()),
        "mean_redundancy": float(redundancy.mean()),
    }

    # Correlation between complexity and redundancy (only among feasible)
    feas_mask = feasibility > 0.5
    if feas_mask.sum() > 2:
        c_sub = complexity[feas_mask]
        r_sub = redundancy[feas_mask]
        if np.std(c_sub) > 1e-8 and np.std(r_sub) > 1e-8:
            result["corr_complexity_redundancy"] = float(
                np.corrcoef(c_sub, r_sub)[0, 1])
        else:
            result["corr_complexity_redundancy"] = 0.0
    else:
        result["corr_complexity_redundancy"] = 0.0

    # Check if feasibility is constant (always 1 or always 0)
    result["feasibility_constant"] = bool(np.std(feasibility) < 1e-8)

    return result
