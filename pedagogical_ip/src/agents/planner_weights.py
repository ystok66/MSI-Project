"""
Canonical planner weight configuration.

Defines the 4 weights used by the latent planner's cell-level objective:

    J_i = λ_cost · ĉ_i
        + λ_risk · φ(r̂_i)
        + λ_uc  · (1-n) · u^c_i
        + λ_ur  · (1-n) · u^r_i
        + warned_extra_i

This is the SINGLE canonical source of truth for planner weights.
Both the runner (V2EpisodeState) and the tutor surrogate (RobotBelief)
should reference PlannerWeights rather than scattering individual defaults.

Legacy note: older code used λ_r / agent_risk_weight / λ_uncertainty as
aliases. Those names are deprecated; new code should use PlannerWeights.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlannerWeights:
    """Canonical planner weights for the latent cell-cost objective.

    Frozen to prevent accidental mutation during an episode.
    Values here are the de-facto canonical defaults verified in Batch D
    investigation (matching both plan_from_belief and RobotBelief defaults).
    """
    lambda_cost: float = 1.0     # cost weight (λ_c)
    lambda_risk: float = 3.0     # risk weight (λ_r)
    lambda_uc:   float = 0.1     # cost-uncertainty weight
    lambda_ur:   float = 0.1     # risk-uncertainty weight
