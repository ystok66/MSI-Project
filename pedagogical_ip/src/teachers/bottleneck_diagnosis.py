"""
Bottleneck Diagnosis — Phase 10 Stage 4.

Classifies the agent's current failure bottleneck into three types:
  - epistemic:  agent lacks information about decision-relevant cells
  - structural: agent can't reach goal in time without topology change
  - outcome:    agent faces unavoidable risk that only mitigation can help

Each score is computed as (bottleneck_severity × intervention_gain).
The dominant bottleneck determines which intervention family should be preferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .perceptual_model import PerceptualAccessState


@dataclass
class BottleneckScores:
    """Three-way bottleneck diagnosis scores."""
    epistemic: float = 0.0   # WARN lever
    structural: float = 0.0  # UNLOCK lever
    outcome: float = 0.0     # ITEM_DROP lever

    @property
    def dominant(self) -> str:
        """Return dominant bottleneck type."""
        vals = {"epistemic": self.epistemic,
                "structural": self.structural,
                "outcome": self.outcome}
        return max(vals, key=vals.get)

    @property
    def max_score(self) -> float:
        return max(self.epistemic, self.structural, self.outcome)


def diagnose_bottleneck(
    agent_pos: tuple[int, int],
    goal: tuple[int, int],
    passable: np.ndarray,
    t: int,
    t_max: int,
    pa: Optional[PerceptualAccessState],
    # Counterfactual Q-values from intervention_policy
    q_wait: float = 0.0,
    q_warn: float = 0.0,
    q_unlock: float = 0.0,
    q_item: float = 0.0,
    # Decision-relevant cells (from predicted prefix)
    prefix_cells: Optional[list[tuple[int, int]]] = None,
    # Agent belief uncertainty on decision region
    risk_uncertainty_map: Optional[np.ndarray] = None,
    # Structural info
    has_locked_doors: bool = False,
    slack_steps: int = 999,
    # Minimum risk on any feasible path
    min_path_risk: float = 0.0,
    # Weights
    omega_rho: float = 1.0,    # weight for unseen probability
    omega_u: float = 1.0,      # weight for risk uncertainty
    tau_s: float = 3.0,        # structural urgency temperature
) -> BottleneckScores:
    """Compute three-way bottleneck scores.

    Each score = severity × max(0, intervention_gain).
    """
    prefix = prefix_cells or []

    # ─── Epistemic score ───────────────────────────────────────────
    # How much decision-relevant uncertainty exists?
    if prefix and pa is not None:
        # U_D = (1/|D|) Σ [ω_ρ(1-ρ_i) + ω_u · u_r_i]
        u_d = 0.0
        for r, c in prefix:
            unseen = 1.0 - pa.seen_prob[r, c]
            u_r = 0.0
            if risk_uncertainty_map is not None:
                u_r = float(risk_uncertainty_map[r, c])
            u_d += omega_rho * unseen + omega_u * u_r
        u_d /= max(len(prefix), 1)
    elif prefix:
        # No perceptual model: use uncertainty map only
        u_d = 0.0
        if risk_uncertainty_map is not None:
            for r, c in prefix:
                u_d += float(risk_uncertainty_map[r, c])
            u_d /= max(len(prefix), 1)
        else:
            u_d = 0.5  # moderate default
    else:
        u_d = 0.0

    delta_q_warn = max(0.0, q_warn - q_wait)
    s_epi = u_d * (1.0 + delta_q_warn)

    # ─── Structural score ──────────────────────────────────────────
    # Is the agent running out of time / can't reach goal without unlock?
    time_left = max(1, t_max - t)

    # Structural urgency: exponential decay with slack
    if slack_steps <= 0 or has_locked_doors:
        g_t = 1.0  # critical: no feasible path or very tight
    else:
        g_t = np.exp(-slack_steps / tau_s)

    delta_q_unlock = max(0.0, q_unlock - q_wait)
    s_str = g_t * (1.0 + delta_q_unlock)

    # ─── Outcome score ─────────────────────────────────────────────
    # Is there unavoidable risk on any feasible path?
    delta_q_item = max(0.0, q_item - q_wait)
    s_out = min_path_risk * (1.0 + delta_q_item)

    return BottleneckScores(
        epistemic=s_epi,
        structural=s_str,
        outcome=s_out,
    )


def match_intervention_to_bottleneck(
    action: str,
    scores: BottleneckScores,
) -> float:
    """Compute bottleneck-intervention match bonus.

    M(action, bottleneck) aligns each intervention with its natural lever:
      WARN  ↔ epistemic
      UNLOCK ↔ structural
      ITEM_DROP ↔ outcome
      WAIT ↔ 0
    """
    mapping = {
        "WARN": scores.epistemic,
        "UNLOCK": scores.structural,
        "ITEM_DROP": scores.outcome,
        "WAIT": 0.0,
    }
    return mapping.get(action, 0.0)
