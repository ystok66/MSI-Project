"""
Tutor Perceptual Model — Phase 10 Stage 4.

Tracks the tutor's estimate of what the agent has *effectively seen*
(perceptual access posterior) and provides decision-relevant diagnostics
for intervention selection.

Core state variable per cell:
    ρ_{i,t} = P(agent has effectively seen cell i by time t)

Update rule:
    ρ_{i,t+1} = 1 - (1 - ρ_{i,t}) * (1 - p_see_{i,t+1})

where p_see depends on distance to agent, patch radius, and obs noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PerceptualAccessState:
    """Tutor's estimate of what the agent has perceived.

    Maintained by the tutor (not the agent). Updated each step based on
    agent position, patch radius, and observation noise model.
    """

    # ρ[H, W]: probability agent has effectively seen each cell
    seen_prob: np.ndarray

    # Effective observation variance tutor estimates for agent
    # Lower = agent had better observations
    effective_obs_var: np.ndarray  # [H, W]

    # Which interventions tutor has already issued (de-duplication)
    intervention_memory: dict = field(default_factory=dict)

    # Configuration
    patch_radius: int = 2
    lambda_distance: float = 0.8    # distance decay for p_see
    base_self_var: float = 0.01     # σ² at distance 0
    base_neighbor_var: float = 0.08  # σ² at distance 1


def init_perceptual_access(height: int, width: int,
                           patch_radius: int = 2) -> PerceptualAccessState:
    """Initialize perceptual access state (all unseen)."""
    return PerceptualAccessState(
        seen_prob=np.zeros((height, width), dtype=np.float64),
        effective_obs_var=np.ones((height, width), dtype=np.float64),
        patch_radius=patch_radius,
    )


def update_perceptual_access(
    pa: PerceptualAccessState,
    agent_pos: tuple[int, int],
    passable: np.ndarray,
) -> None:
    """Update seen_prob after agent observes at agent_pos.

    For each cell within patch_radius of agent_pos:
        p_see = 1[d ≤ r_patch] · exp(-λ_d · d) · q_obs
        ρ_new = 1 - (1 - ρ_old) * (1 - p_see)

    This is monotonically non-decreasing: once seen, always seen.
    """
    H, W = pa.seen_prob.shape
    ar, ac = agent_pos

    for dr in range(-pa.patch_radius, pa.patch_radius + 1):
        for dc in range(-pa.patch_radius, pa.patch_radius + 1):
            r, c = ar + dr, ac + dc
            if r < 0 or r >= H or c < 0 or c >= W:
                continue
            if not passable[r, c]:
                continue

            d = abs(dr) + abs(dc)  # Manhattan distance
            if d > pa.patch_radius:
                continue

            # p_see = exp(-λ·d) · obs_quality
            if d == 0:
                obs_var = pa.base_self_var
            else:
                obs_var = pa.base_neighbor_var * d
            q_obs = 1.0 / (1.0 + obs_var)

            p_see = np.exp(-pa.lambda_distance * d) * q_obs

            # Monotonic update: ρ_new = 1 - (1-ρ_old)(1-p_see)
            pa.seen_prob[r, c] = 1.0 - (1.0 - pa.seen_prob[r, c]) * (1.0 - p_see)

            # Track effective obs variance (keep best ever)
            pa.effective_obs_var[r, c] = min(pa.effective_obs_var[r, c], obs_var)


def compute_redundancy(
    pa: PerceptualAccessState,
    cells: list[tuple[int, int]],
    risk_uncertainty: Optional[np.ndarray] = None,
    tau_u: float = 0.3,
) -> float:
    """Compute warning redundancy for a set of cells.

    R_warn = (1/|D|) Σ_{i∈D} ρ_i · exp(-u_r_i / τ_u)

    High redundancy → agent already knows about these cells,
    warning would be wasted.
    """
    if not cells:
        return 1.0  # no cells = fully redundant

    total = 0.0
    for r, c in cells:
        rho = pa.seen_prob[r, c]
        if risk_uncertainty is not None:
            u_r = risk_uncertainty[r, c]
            total += rho * np.exp(-u_r / tau_u)
        else:
            total += rho  # treat as fully confident if no uncertainty provided
    return total / len(cells)


def compute_decision_relevant_unseen(
    pa: PerceptualAccessState,
    prefix_cells: list[tuple[int, int]],
) -> float:
    """Fraction of decision-relevant cells the agent hasn't effectively seen.

    Returns mean(1 - ρ_i) over prefix cells.
    """
    if not prefix_cells:
        return 0.0
    unseen = sum(1.0 - pa.seen_prob[r, c] for r, c in prefix_cells)
    return unseen / len(prefix_cells)
