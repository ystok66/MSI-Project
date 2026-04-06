"""AgentBelief — Agent's internal belief state for planning.

This is NOT the true world state. It wraps the agent's uncertain,
observation-derived model of the environment + its internal decision state.

POMDP-interface shell (Task 3 Phase A).
Does not change any existing behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import numpy as np


@dataclass
class AgentBelief:
    """Agent's internal belief used for planning.

    b_t^A = (b_env, m_t, θ, h_t)

    This contains:
      - b_env: belief over cell-wise cost/risk (means/variances)
      - m_t: current internalization / control state
      - θ: learner preference type
      - h_t: optional action-observation history summary
    """
    # --- Environment belief ---
    # Per-cell feature belief (from FeatureBeliefMap)
    belief_mean: Optional[np.ndarray] = None      # (H, W, d)
    belief_var: Optional[np.ndarray] = None       # (H, W, d)

    # Derived cost/risk predictions
    predicted_cost: Optional[np.ndarray] = None   # (H, W)
    predicted_risk: Optional[np.ndarray] = None   # (H, W)

    # Visibility / observation state
    observed_mask: Optional[np.ndarray] = None    # (H, W) bool
    visit_count: Optional[np.ndarray] = None      # (H, W) int

    # --- Internal decision state ---
    # Internalization state m_t = (κ, τ, ν, γ_spec, γ_gen)
    m_state: Dict[str, float] = field(default_factory=lambda: {
        "kappa": 1.0, "tau": 0.3, "nu": 0.1,
        "gamma_spec": 0.0, "gamma_gen": 0.0,
    })

    # Learner preference type
    theta: str = "safe"

    # History summary (lightweight)
    n_steps_taken: int = 0
    n_warnings_received: int = 0
    n_self_discoveries: int = 0

    def risk_uncertainty(self, row: int, col: int) -> float:
        """Mean feature variance at (row, col) — proxy for risk uncertainty."""
        if self.belief_var is None:
            return 0.25  # prior
        return float(np.mean(self.belief_var[row, col]))


def agent_belief_from_feature_map(fbm, m_state=None, theta="safe",
                                   latent_predictor=None) -> AgentBelief:
    """Adapter: build AgentBelief from existing FeatureBeliefMap + state."""
    ab = AgentBelief(
        belief_mean=fbm.mean.copy(),
        belief_var=fbm.var.copy(),
        observed_mask=fbm.observed.copy(),
        visit_count=fbm.visit_count.copy(),
        theta=theta,
    )
    if m_state is not None:
        if hasattr(m_state, 'as_dict'):
            ab.m_state = dict(m_state.as_dict)
        elif isinstance(m_state, dict):
            ab.m_state = dict(m_state)

    # Generate predicted cost/risk from latent predictor if available
    if latent_predictor is not None and fbm.mean is not None:
        H, W, d = fbm.mean.shape
        pred_cost = np.ones((H, W))
        pred_risk = np.zeros((H, W))
        for r in range(H):
            for c in range(W):
                z = fbm.mean[r, c]
                pred_cost[r, c] = latent_predictor.predict_cost(z)
                pred_risk[r, c] = latent_predictor.predict_risk(z)
        ab.predicted_cost = pred_cost
        ab.predicted_risk = pred_risk

    return ab


# ═══════════════════════════════════════════════════════════
# Step 2: RSA Warning Belief Methods
# ═══════════════════════════════════════════════════════════

def predict_warning_prior(
    ab: AgentBelief,
    segment_info: dict,
) -> 'np.ndarray':
    """Compute prior b_t⁻(r) from AgentBelief for a segment.

    Uses risk uncertainty as prior: higher uncertainty → more uniform.
    Lower uncertainty + high risk → prior favors hazard hypotheses.

    Args:
        ab: current agent belief
        segment_info: dict with segment_cells, segment_side

    Returns:
        prior over 4 risk hypotheses, shape (4,), sums to 1
    """
    from .rsa_warning_channel import N_HYPOTHESES

    # If no risk predictions available, return uniform
    if ab.predicted_risk is None:
        return np.ones(N_HYPOTHESES) / N_HYPOTHESES

    cells = segment_info.get("segment_cells", [])
    side = segment_info.get("segment_side", "left")

    if not cells:
        return np.ones(N_HYPOTHESES) / N_HYPOTHESES

    # Compute mean risk for this segment
    risks = [float(ab.predicted_risk[r, c]) for r, c in cells
             if 0 <= r < ab.predicted_risk.shape[0]
             and 0 <= c < ab.predicted_risk.shape[1]]
    mean_risk = np.mean(risks) if risks else 0.25

    # Prior: tilted toward the hypothesis matching observed risk pattern
    prior = np.ones(N_HYPOTHESES) * 0.15  # base mass
    if side == "left":
        prior[0] += 0.4 * mean_risk     # left_risky
        prior[2] += 0.4 * (1 - mean_risk)  # both_safe
    else:
        prior[1] += 0.4 * mean_risk     # right_risky
        prior[2] += 0.4 * (1 - mean_risk)  # both_safe
    prior[3] += 0.2 * mean_risk  # hazard_ahead

    prior /= prior.sum()
    return prior


def update_from_warning_rsa(
    ab: AgentBelief,
    rsa_channel: 'RSAWarningChannel',
    belief_state: 'RSABeliefState',
    utterance: 'RSAUtterance',
    context: dict,
    variant: str = "s1",
    tau_hat: float = 0.3,
) -> dict:
    """Update AgentBelief from RSA warning channel.

    This is the Step 2 replacement: warning only modifies belief,
    not the planner directly.

    Returns:
        diagnostics dict from the RSA update
    """
    info = rsa_channel.update_belief(
        belief_state, utterance, context,
        variant=variant, tau_hat=tau_hat)

    # Update n_warnings counter
    ab.n_warnings_received += 1

    return info


def export_local_risk_posterior(
    ab: AgentBelief,
    belief_state: 'RSABeliefState',
    segment_cells: list,
    segment_side: str = "left",
) -> dict:
    """Export risk adjustment for planner from RSA posterior.

    Translates RSA belief state into per-cell risk adjustments
    that the planner can consume without knowing about RSA.

    Returns:
        dict mapping (row, col) → risk_adjustment
    """
    from .rsa_warning_channel import belief_to_risk_update

    delta = belief_to_risk_update(belief_state, segment_side)

    adjustments = {}
    for r, c in segment_cells:
        adjustments[(r, c)] = delta

    return {
        "cell_adjustments": adjustments,
        "risk_delta": delta,
        "belief": belief_state.belief.tolist(),
        "entropy": belief_state.entropy(),
    }
