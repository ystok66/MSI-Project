"""GTET Factor Adapter — Posterior View for Tutor Decisions.

Provides a thin adapter that makes factor-restricted posteriors influence
the tutor's intervention decisions via two mechanisms:

1. **Epistemic adjustment**: The posterior entropy (or factor-restricted
   entropy) modifies the bottleneck diagnosis epistemic score. A higher-
   entropy posterior means more goal/preference ambiguity, boosting WARN.

2. **Risk bias from posterior MAP**: The MAP (g,θ,z) estimate generates
   a predicted-risk modifier that adjusts the counterfactual risk
   estimates used in intervention scoring.

Does NOT modify: JointGoalPrefPosterior, RobotBelief, intervention_policy,
bottleneck_diagnosis, planner, observer, or feature dimensionality.

Integration: called from the runner immediately before `score_interventions`.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from ..teachers.joint_goal_pref_posterior import JointGoalPrefPosterior


# ════════════════════════════════════════════════════════════════
# Factor modes and marginalization
# ════════════════════════════════════════════════════════════════

FACTOR_MODES = {
    "FULL":       None,         # keep all dimensions
    "G_THETA":    (0, 1),       # sum out z (axis=2)
    "G_Z":        (0, 2),       # sum out θ (axis=1)
    "THETA_Z":    (1, 2),       # sum out g (axis=0)
    "G_ONLY":     (0,),         # sum out θ, z
    "THETA_ONLY": (1,),         # sum out g, z
    "Z_ONLY":     (2,),         # sum out g, θ
}

# axis mapping: dimension index → marginalization axes
_KEEP_TO_SUM = {
    (0, 1):    (2,),        # keep g,θ → sum z
    (0, 2):    (1,),        # keep g,z → sum θ
    (1, 2):    (0,),        # keep θ,z → sum g
    (0,):      (1, 2),      # keep g   → sum θ,z
    (1,):      (0, 2),      # keep θ   → sum g,z
    (2,):      (0, 1),      # keep g,θ → sum g,θ
}


def build_factor_restricted_view(
    jgpp: JointGoalPrefPosterior,
    factor_mode: str = "FULL",
    lift_mode: str = "uniform",
) -> np.ndarray:
    """Build a factor-restricted posterior view from the full posterior.

    Returns a (G, T, Z) array that represents the lifted factored posterior.

    Args:
        jgpp: Full joint posterior q(g,θ,z).
        factor_mode: One of FACTOR_MODES keys.
        lift_mode: "uniform" (Lift U) or "prior" (Lift P).
            - uniform: dropped dims filled with 1/|X_M̄|
            - prior: dropped dims filled with prior conditional p0(x_M̄ | x_M)

    Returns:
        Lifted posterior array, shape (n_g, n_p, n_z), normalized.
    """
    if factor_mode not in FACTOR_MODES:
        raise ValueError(f"Unknown factor_mode '{factor_mode}'. "
                         f"Available: {list(FACTOR_MODES.keys())}")

    w = jgpp._weights()  # (n_g, n_p, n_z) — normalized
    n_g, n_p, n_z = w.shape

    if factor_mode == "FULL":
        return w.copy()

    keep_dims = FACTOR_MODES[factor_mode]
    sum_axes = _KEEP_TO_SUM[keep_dims]

    # Step 1: Marginalize — sum out dropped dimensions
    marginal = w.sum(axis=sum_axes)

    # Step 2: Lift back to full shape
    if lift_mode == "prior":
        # Lift P: use initial prior structure for fill
        prior_w = _compute_prior_weights(jgpp)
        lifted = _lift_marginal_prior(marginal, keep_dims, n_g, n_p, n_z, prior_w)
    else:
        # Lift U: uniform fill (default)
        lifted = _lift_marginal(marginal, keep_dims, n_g, n_p, n_z, w)

    # Step 3: Normalize
    total = lifted.sum()
    if total > 0:
        lifted /= total

    return lifted


def _compute_prior_weights(jgpp: JointGoalPrefPosterior) -> np.ndarray:
    """Reconstruct the initial prior q0(g,θ,z) = p(g)·p(θ)·p(z).

    Uses the goal_space labels, pref_types, and tempt_grid to compute
    the same prior structure as __init__, without modifying the posterior.
    """
    n_g = jgpp._goal_space.n_goals
    n_p = len(jgpp._pref_types)
    n_z = len(jgpp._tempt_grid)

    # Reconstruct marginal priors (simplified: assume uniform-ish)
    # The actual prior was computed in __init__ but not stored separately.
    # We reconstruct it from the structural prior mode.
    from ..teachers.joint_goal_pref_posterior import (
        compute_normalized_goal_prior, GoalPriorContext, GoalPriorConfig,
    )
    try:
        gp = compute_normalized_goal_prior(
            jgpp._goal_space, jgpp._prior_context, jgpp._prior_config)
    except Exception:
        gp = np.ones(n_g) / n_g

    pp = np.ones(n_p) / n_p
    zp = np.ones(n_z) / n_z
    if hasattr(jgpp, '_tempt_grid') and jgpp._has_tempt:
        from ..teachers.joint_goal_pref_posterior import DEFAULT_TEMPT_PRIOR
        if DEFAULT_TEMPT_PRIOR is not None:
            zp = np.array(DEFAULT_TEMPT_PRIOR[:n_z], dtype=np.float64)
            zp /= zp.sum()

    prior_w = np.einsum('g,p,z->gpz', gp, pp, zp)
    prior_w /= prior_w.sum()
    return prior_w


def _lift_marginal(marginal, keep_dims, n_g, n_p, n_z, full_w):
    """Lift a marginal back to full (G, T, Z) using uniform fill.

    For dropped dimensions, we fill with uniform conditional:
        lifted(x) = marginal(x_M) / |x_M̄|

    This INTENTIONALLY loses the correlation structure present in
    the full posterior, so that KL(full || lifted) > 0 when
    correlations exist. This is the core of the factor ablation:
    the factored posterior cannot represent g×θ×z correlations.
    """
    lifted = np.zeros((n_g, n_p, n_z))

    if keep_dims == (0, 1):
        # marginal shape: (n_g, n_p) — uniform over z
        for gi in range(n_g):
            for pi in range(n_p):
                lifted[gi, pi, :] = marginal[gi, pi] / n_z

    elif keep_dims == (0, 2):
        # marginal shape: (n_g, n_z) — uniform over θ
        for gi in range(n_g):
            for zi in range(n_z):
                lifted[gi, :, zi] = marginal[gi, zi] / n_p

    elif keep_dims == (1, 2):
        # marginal shape: (n_p, n_z) — uniform over g
        for pi in range(n_p):
            for zi in range(n_z):
                lifted[:, pi, zi] = marginal[pi, zi] / n_g

    elif keep_dims == (0,):
        # marginal shape: (n_g,) — uniform over θ, z
        for gi in range(n_g):
            lifted[gi, :, :] = marginal[gi] / (n_p * n_z)

    elif keep_dims == (1,):
        # marginal shape: (n_p,) — uniform over g, z
        for pi in range(n_p):
            lifted[:, pi, :] = marginal[pi] / (n_g * n_z)

    elif keep_dims == (2,):
        # marginal shape: (n_z,) — uniform over g, θ
        for zi in range(n_z):
            lifted[:, :, zi] = marginal[zi] / (n_g * n_p)

    return lifted


def _lift_marginal_prior(marginal, keep_dims, n_g, n_p, n_z, prior_w):
    """Lift a marginal back to full (G, T, Z) using prior-conditional fill.

    For dropped dimensions, we fill with the prior conditional:
        lifted(x) = marginal(x_M) * p0(x_M̄ | x_M)

    Where p0(x_M̄ | x_M) = p0(x) / Σ_{x': x'_M = x_M} p0(x')

    Uses the INITIAL PRIOR (not posterior) to avoid the identity-lift bug.
    """
    lifted = np.zeros((n_g, n_p, n_z))

    if keep_dims == (0, 1):
        for gi in range(n_g):
            for pi in range(n_p):
                denom = prior_w[gi, pi, :].sum()
                if denom > 1e-15:
                    lifted[gi, pi, :] = marginal[gi, pi] * prior_w[gi, pi, :] / denom
                else:
                    lifted[gi, pi, :] = marginal[gi, pi] / n_z

    elif keep_dims == (0, 2):
        for gi in range(n_g):
            for zi in range(n_z):
                denom = prior_w[gi, :, zi].sum()
                if denom > 1e-15:
                    lifted[gi, :, zi] = marginal[gi, zi] * prior_w[gi, :, zi] / denom
                else:
                    lifted[gi, :, zi] = marginal[gi, zi] / n_p

    elif keep_dims == (1, 2):
        for pi in range(n_p):
            for zi in range(n_z):
                denom = prior_w[:, pi, zi].sum()
                if denom > 1e-15:
                    lifted[:, pi, zi] = marginal[pi, zi] * prior_w[:, pi, zi] / denom
                else:
                    lifted[:, pi, zi] = marginal[pi, zi] / n_g

    elif keep_dims == (0,):
        for gi in range(n_g):
            denom = prior_w[gi, :, :].sum()
            if denom > 1e-15:
                lifted[gi, :, :] = marginal[gi] * prior_w[gi, :, :] / denom
            else:
                lifted[gi, :, :] = marginal[gi] / (n_p * n_z)

    elif keep_dims == (1,):
        for pi in range(n_p):
            denom = prior_w[:, pi, :].sum()
            if denom > 1e-15:
                lifted[:, pi, :] = marginal[pi] * prior_w[:, pi, :] / denom
            else:
                lifted[:, pi, :] = marginal[pi] / (n_g * n_z)

    elif keep_dims == (2,):
        for zi in range(n_z):
            denom = prior_w[:, :, zi].sum()
            if denom > 1e-15:
                lifted[:, :, zi] = marginal[zi] * prior_w[:, :, zi] / denom
            else:
                lifted[:, :, zi] = marginal[zi] / (n_g * n_p)

    return lifted


# ════════════════════════════════════════════════════════════════
# Tutor decision integration
# ════════════════════════════════════════════════════════════════

def compute_posterior_epistemic_modifier(
    q_full: np.ndarray,
    q_restricted: np.ndarray,
) -> float:
    """Compute epistemic modifier from posterior KL divergence.

    Returns KL(q_full || q_restricted), which measures how much
    information is lost by dropping factors.

    Higher KL → the dropped factors carry important information
    → the tutor should be MORE cautious (boost epistemic score).
    """
    q_f = q_full.ravel()
    q_r = q_restricted.ravel()

    # Avoid log(0)
    mask = (q_f > 1e-15) & (q_r > 1e-15)
    if not mask.any():
        return 0.0

    kl = np.sum(q_f[mask] * np.log(q_f[mask] / q_r[mask]))
    return float(max(0.0, kl))


def compute_posterior_risk_modifier(
    q_view: np.ndarray,
    tempt_grid: tuple[float, ...],
    z_risk_scale: float = 0.15,
) -> float:
    """Compute risk modifier from the posterior's MAP temptation level.

    Higher MAP z → the tutor believes the agent is more susceptible
    to temptation → boost the estimated risk on temptation routes.

    Returns a risk bias in [0, z_risk_scale].
    """
    n_g, n_p, n_z = q_view.shape
    # P(z) = Σ_{g,θ} q(g,θ,z)
    marg_z = q_view.sum(axis=(0, 1))
    expected_z = sum(marg_z[i] * tempt_grid[i] for i in range(n_z))
    return float(expected_z * z_risk_scale)


def compute_action_divergence(
    actions_full: list[str],
    actions_restricted: list[str],
) -> float:
    """Compute action divergence rate (ADR).

    ADR = fraction of timesteps where full and restricted disagree.
    """
    if not actions_full or not actions_restricted:
        return 0.0
    n = min(len(actions_full), len(actions_restricted))
    disagree = sum(1 for i in range(n) if actions_full[i] != actions_restricted[i])
    return disagree / n


def compute_ambiguity_trajectory(q_view: np.ndarray) -> float:
    """Compute ambiguity A = 1 - max q(g,θ,z)."""
    return 1.0 - float(np.max(q_view))
