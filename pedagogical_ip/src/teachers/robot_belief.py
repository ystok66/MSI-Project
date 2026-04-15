"""
Robot Belief — Phase 7.

Approximate surrogate of the agent's internal state, maintained by the robot.
NOT full Bayesian nested inference. This is an approximate copy with
configurable mismatch for both belief state and competence parameters.

CRITICAL: The robot-belief tutor must NOT access hidden true trap cells,
hidden latent vectors, or true future risk values. Only agent-observable
state and segment topology are allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy

import numpy as np

from ..agents.planner_weights import PlannerWeights


@dataclass
class RobotBelief:
    """Approximate surrogate of the agent's belief and competence.

    Belief mismatch (copy_mode) and competence mismatch are configured
    separately, as they represent different sources of robot uncertainty.
    """

    # Surrogate belief state
    agent_belief_mean: np.ndarray       # (H, W, d)
    agent_belief_var: np.ndarray        # (H, W, d)

    # Surrogate competence parameters (robot's estimate of agent)
    agent_search_budget: int = 30
    agent_heuristic_noise_std: float = 0.0

    # Canonical planner weights — single source (Phase D)
    agent_planner_weights: PlannerWeights = field(default_factory=PlannerWeights)

    # Copy config
    copy_mode: str = "exact"            # "exact" | "noisy" | "stale"
    belief_noise_std: float = 0.05      # noise added in "noisy" mode
    stale_interval: int = 3             # sync every N steps in "stale" mode
    last_sync_t: int = 0

    # Competence mismatch config
    budget_mismatch: int = 0            # added to true budget (can be negative)
    risk_weight_mismatch: float = 0.0   # added to true risk weight

    # Surrogate latent predictor snapshot (full deepcopy, read-only)
    _predictor_snapshot: Optional[object] = None

    # ── Deprecated properties (backward compat) ──────────────────
    # DEPRECATED: use agent_planner_weights.lambda_risk instead.
    @property
    def agent_risk_weight(self) -> float:
        """DEPRECATED: use agent_planner_weights.lambda_risk."""
        return self.agent_planner_weights.lambda_risk

    # DEPRECATED: use agent_planner_weights.lambda_uc instead.
    @property
    def agent_uncertainty_weight(self) -> float:
        """DEPRECATED: use agent_planner_weights.lambda_uc (or lambda_ur)."""
        return self.agent_planner_weights.lambda_uc

    # DEPRECATED: use agent_planner_weights.lambda_cost instead.
    @property
    def agent_lambda_c(self) -> float:
        """DEPRECATED: use agent_planner_weights.lambda_cost."""
        return self.agent_planner_weights.lambda_cost

    # DEPRECATED: use agent_planner_weights.lambda_uc instead.
    @property
    def agent_lambda_uc(self) -> float:
        """DEPRECATED: use agent_planner_weights.lambda_uc."""
        return self.agent_planner_weights.lambda_uc

    # DEPRECATED: use agent_planner_weights.lambda_ur instead.
    @property
    def agent_lambda_ur(self) -> float:
        """DEPRECATED: use agent_planner_weights.lambda_ur."""
        return self.agent_planner_weights.lambda_ur


def init_robot_belief(
    agent_belief_mean: np.ndarray,
    agent_belief_var: np.ndarray,
    latent_predictor=None,
    copy_mode: str = "exact",
    belief_noise_std: float = 0.05,
    stale_interval: int = 3,
    agent_search_budget: int = 30,
    budget_mismatch: int = 0,
    risk_weight_mismatch: float = 0.0,
    planner_weights: Optional[PlannerWeights] = None,
    # DEPRECATED individual weight params — kept for backward compat
    agent_risk_weight: Optional[float] = None,
    agent_uncertainty_weight: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> RobotBelief:
    """Create a RobotBelief from current agent state.

    Does NOT read hidden true values — only agent-observable state.

    Parameters
    ----------
    planner_weights : PlannerWeights, optional
        Canonical planner weights. If provided, overrides individual
        weight params. If None, uses PlannerWeights() defaults.
    agent_risk_weight : float, optional
        DEPRECATED. Use planner_weights instead.
    agent_uncertainty_weight : float, optional
        DEPRECATED. Use planner_weights instead.
    """
    rng = rng or np.random.default_rng()

    # Resolve planner weights: explicit PlannerWeights > legacy params > defaults
    if planner_weights is not None:
        _pw = planner_weights
    elif agent_risk_weight is not None or agent_uncertainty_weight is not None:
        # Legacy caller: construct from individual params
        _pw = PlannerWeights(
            lambda_risk=agent_risk_weight if agent_risk_weight is not None else 3.0,
            lambda_uc=agent_uncertainty_weight if agent_uncertainty_weight is not None else 0.1,
        )
    else:
        _pw = PlannerWeights()

    mean_copy = agent_belief_mean.copy()
    var_copy = agent_belief_var.copy()

    if copy_mode == "noisy":
        mean_copy = mean_copy + rng.normal(0, belief_noise_std, mean_copy.shape)

    rb = RobotBelief(
        agent_belief_mean=mean_copy,
        agent_belief_var=var_copy,
        agent_search_budget=agent_search_budget + budget_mismatch,
        agent_planner_weights=_pw,
        copy_mode=copy_mode,
        belief_noise_std=belief_noise_std,
        stale_interval=stale_interval,
        budget_mismatch=budget_mismatch,
        risk_weight_mismatch=risk_weight_mismatch,
    )

    # Snapshot latent predictor (full deepcopy, works with any head type)
    if latent_predictor is not None:
        rb._predictor_snapshot = deepcopy(latent_predictor)

    return rb


def sync_robot_belief(
    rb: RobotBelief,
    agent_belief_mean: np.ndarray,
    agent_belief_var: np.ndarray,
    latent_predictor=None,
    t: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Update surrogate from agent state, respecting copy_mode.

    - exact: full copy every step
    - noisy: full copy + noise every step
    - stale: only sync every stale_interval steps
    """
    rng = rng or np.random.default_rng()

    if rb.copy_mode == "stale":
        if (t - rb.last_sync_t) < rb.stale_interval:
            return  # skip this sync
        rb.last_sync_t = t

    rb.agent_belief_mean = agent_belief_mean.copy()
    rb.agent_belief_var = agent_belief_var.copy()

    if rb.copy_mode == "noisy":
        rb.agent_belief_mean += rng.normal(0, rb.belief_noise_std,
                                            rb.agent_belief_mean.shape)

    # Re-snapshot predictor if provided
    if latent_predictor is not None:
        rb._predictor_snapshot = deepcopy(latent_predictor)


def build_surrogate_predictor(rb: RobotBelief):
    """Build a surrogate predictor from robot's snapshot.

    Returns a deepcopy of the snapshotted predictor. Read-only use.
    Works with any head type (LatentCostRiskHead, StructuredBasisCostRiskHead, etc.)
    """
    if rb._predictor_snapshot is None:
        return None
    return deepcopy(rb._predictor_snapshot)
