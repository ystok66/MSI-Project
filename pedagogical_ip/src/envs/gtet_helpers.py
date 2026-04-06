"""GTET-L Helpers: Posterior Factor Ablation Wrapper.

Provides thin wrappers around JointGoalPrefPosterior to compute
factor-restricted posteriors WITHOUT modifying the frozen posterior core.

Factor sets:
  - "full"     : q(g, θ, z)  — use as-is
  - "g_theta"  : q(g, θ) = Σ_z q(g,θ,z)
  - "g_z"      : q(g, z) = Σ_θ q(g,θ,z)
  - "theta_z"  : q(θ, z) = Σ_g q(g,θ,z)
  - "g_only"   : q(g)   = Σ_{θ,z} q(g,θ,z)
  - "theta_only": q(θ)  = Σ_{g,z} q(g,θ,z)
  - "z_only"   : q(z)   = Σ_{g,θ} q(g,θ,z)

The wrapper reads the full posterior's internal weights, marginalizes,
and produces a "lifted" joint table using the factored posterior × uniform
over dropped dimensions. This lets downstream code (tutor, interventions)
use the same interface without changes.

Does NOT modify JointGoalPrefPosterior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

from ..teachers.joint_goal_pref_posterior import JointGoalPrefPosterior


FACTOR_SETS = ("full", "g_theta", "g_z", "theta_z",
               "g_only", "theta_only", "z_only")


@dataclass
class FactorAblationResult:
    """Result of a factor-restricted posterior evaluation."""
    factor_set: str
    marginal_goal: Dict[str, float]
    marginal_pref: Dict[str, float]
    marginal_tempt: Dict[float, float]
    entropy: float
    map_goal: str
    map_pref: str
    # The lifted joint weights (same shape as full posterior)
    lifted_weights: np.ndarray


def marginalize_posterior(
    jgpp: JointGoalPrefPosterior,
    factor_set: str = "full",
) -> FactorAblationResult:
    """Compute factor-restricted posterior from a full joint posterior.

    Args:
        jgpp: A JointGoalPrefPosterior that has already been updated.
        factor_set: Which factors to keep.  One of FACTOR_SETS.

    Returns:
        FactorAblationResult with marginals, entropy, MAP, and lifted weights.
    """
    if factor_set not in FACTOR_SETS:
        raise ValueError(f"Unknown factor_set '{factor_set}'. "
                         f"Available: {FACTOR_SETS}")

    # Get full weights (normalized)
    w = jgpp._weights()  # shape (n_g, n_p, n_z)
    n_g, n_p, n_z = w.shape

    if factor_set == "full":
        lifted = w.copy()
    elif factor_set == "g_theta":
        # q(g,θ) = Σ_z q(g,θ,z)
        marg = w.sum(axis=2)                      # (n_g, n_p)
        lifted = marg[:, :, None] * np.ones(n_z) / n_z  # lift back
    elif factor_set == "g_z":
        # q(g,z) = Σ_θ q(g,θ,z)
        marg = w.sum(axis=1)                      # (n_g, n_z)
        lifted = marg[:, None, :] * np.ones(n_p) / n_p
    elif factor_set == "theta_z":
        # q(θ,z) = Σ_g q(g,θ,z)
        marg = w.sum(axis=0)                      # (n_p, n_z)
        lifted = marg[None, :, :] * np.ones(n_g) / n_g
    elif factor_set == "g_only":
        marg = w.sum(axis=(1, 2))                 # (n_g,)
        lifted = marg[:, None, None] * np.ones((1, n_p, n_z)) / (n_p * n_z)
    elif factor_set == "theta_only":
        marg = w.sum(axis=(0, 2))                 # (n_p,)
        lifted = marg[None, :, None] * np.ones((n_g, 1, n_z)) / (n_g * n_z)
    elif factor_set == "z_only":
        marg = w.sum(axis=(0, 1))                 # (n_z,)
        lifted = marg[None, None, :] * np.ones((n_g, n_p, 1)) / (n_g * n_p)

    # Normalize lifted
    total = lifted.sum()
    if total > 0:
        lifted /= total

    # Compute marginals from lifted
    mg = lifted.sum(axis=(1, 2))
    mp = lifted.sum(axis=(0, 2))
    mz = lifted.sum(axis=(0, 1))

    goal_labels = jgpp._goal_space.labels
    pref_types = jgpp._pref_types
    tempt_grid = jgpp._tempt_grid

    marginal_goal = {goal_labels[i]: float(mg[i]) for i in range(n_g)}
    marginal_pref = {pref_types[i]: float(mp[i]) for i in range(n_p)}
    marginal_tempt = {tempt_grid[i]: float(mz[i]) for i in range(n_z)}

    # Entropy
    flat = lifted.ravel()
    flat = flat[flat > 1e-15]
    entropy = -float(np.sum(flat * np.log(flat)))

    # MAP
    map_goal = max(marginal_goal, key=marginal_goal.get)
    map_pref = max(marginal_pref, key=marginal_pref.get)

    return FactorAblationResult(
        factor_set=factor_set,
        marginal_goal=marginal_goal,
        marginal_pref=marginal_pref,
        marginal_tempt=marginal_tempt,
        entropy=entropy,
        map_goal=map_goal,
        map_pref=map_pref,
        lifted_weights=lifted,
    )


def compute_delta_joint(results: dict[str, float]) -> float:
    """Compute Δ_joint = Perf(full) - max{Perf(factored)}.

    Args:
        results: dict mapping factor_set -> performance metric.
                Must contain "full" and at least one of the 2-factor sets.

    Returns:
        Δ_joint value.  >0 means full joint is better than best factored.
    """
    full_perf = results.get("full", 0.0)
    factored_keys = ["g_theta", "g_z", "theta_z"]
    factored_perfs = [results[k] for k in factored_keys if k in results]
    if not factored_perfs:
        return 0.0
    return full_perf - max(factored_perfs)
