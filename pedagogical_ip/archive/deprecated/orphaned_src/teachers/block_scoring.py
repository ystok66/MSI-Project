"""
Pedagogical Blocking — trigger conditions and timeout estimation.

BLOCK is a Semantic+Safety extension. It temporarily closes a hazard
entry cell when the learner is committed, unaware, hasn't self-corrected,
and has sensory cues available.

IMPORTANT: This module exposes `hazard_risk_map` as an explicit parameter.
  - Oracle teacher passes `true_risk`
  - Particle teacher passes `est_belief.risk_mean` (posterior mean)
  NEVER use true_risk in the particle teacher's online decision path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..agents.planner_astar import bounded_astar, heuristic


# ── Trigger conditions ───────────────────────────────────────────────

@dataclass
class BlockConditions:
    """Diagnostic: which trigger conditions hold."""
    commit: float     # fraction of plan prefix in hazard region
    unaware: bool     # learner risk belief < rho_aware on hazard cells
    backtrack: bool   # agent moving AWAY from hazard
    seen_cue: bool    # hazard cell in recent observation history
    p_fatal: float    # survival probability of current plan on hazard map


def compute_block_decision(
    agent_pos: tuple[int, int],
    agent_plan: list[tuple[int, int]],
    agent_belief_risk: np.ndarray,
    hazard_risk_map: np.ndarray,           # true_risk for oracle, posterior_mean for particle
    observation_positions_recent: list[tuple[int, int]],  # obs positions at t and t-1
    goal: tuple[int, int],
    time_left: int,
    risk_budget_left: float,
    passable_mask: np.ndarray,
    belief_cost_mean: np.ndarray,
    belief_cost_var: np.ndarray,
    # Params
    rho_hazard: float = 0.2,       # threshold for "hazardous cell"
    rho_aware: float = 0.15,       # if learner belief < this, they're unaware
    k: int = 4,                    # plan prefix length to check
    h: int = 3,                    # backtrack horizon (recent positions)
    gamma_r: float = 0.3,          # p_fatal threshold
    search_budget: int = 30,       # budget for deadlock check
) -> tuple[bool, Optional[tuple[int, int]], BlockConditions]:
    """
    Compute whether to BLOCK and which cell.

    Returns:
        (should_block, block_cell, conditions)
    """
    H, W = hazard_risk_map.shape

    # ── Identify hazard region ──
    R_hazard = set()
    for r in range(H):
        for c in range(W):
            if hazard_risk_map[r, c] > rho_hazard:
                R_hazard.add((r, c))

    if not R_hazard:
        return False, None, BlockConditions(0.0, False, False, False, 0.0)

    # ── 1. Commit: fraction of plan prefix in hazard ──
    prefix = agent_plan[1:k+1] if len(agent_plan) > 1 else []
    if not prefix:
        return False, None, BlockConditions(0.0, False, False, False, 0.0)
    n_in_hazard = sum(1 for p in prefix if p in R_hazard)
    commit = n_in_hazard / max(len(prefix), 1)

    # ── 2. Unaware: learner's belief is low on hazard cells in plan ──
    hazard_in_plan = [p for p in prefix if p in R_hazard]
    if hazard_in_plan:
        belief_risk_on_hazard = np.mean([
            agent_belief_risk[r, c] for r, c in hazard_in_plan
        ])
        unaware = belief_risk_on_hazard < rho_aware
    else:
        unaware = False

    # ── 3. Backtrack: distance to hazard NOT increasing (not self-correcting) ──
    # Use observation_positions_recent as proxy for position history
    if len(observation_positions_recent) >= 2:
        nearest_hazard = min(
            (heuristic(p, agent_pos) for p in R_hazard),
            default=999,
        )
        # Check if any recent obs position was closer to hazard
        prev_pos = observation_positions_recent[-2] if len(observation_positions_recent) >= 2 \
            else agent_pos
        prev_nearest = min(
            (heuristic(p, prev_pos) for p in R_hazard),
            default=999,
        )
        backtrack = nearest_hazard > prev_nearest + 0.5  # moving away
    else:
        backtrack = False

    # ── 4. SeenCue: hazard cell in recent observation history ──
    obs_set = set(observation_positions_recent)
    seen_cue = bool(obs_set & R_hazard)

    # ── 5. P_fatal: survival probability of current plan on hazard map ──
    from .cause_scoring import compute_survival_prob
    survival = compute_survival_prob(agent_plan, hazard_risk_map)
    p_fatal = max(0.0, 1.0 - survival)

    conditions = BlockConditions(
        commit=commit,
        unaware=unaware,
        backtrack=backtrack,
        seen_cue=seen_cue,
        p_fatal=p_fatal,
    )

    # ── Decision: all conditions must hold ──
    should = (
        commit >= 0.5
        and unaware
        and not backtrack
        and seen_cue
        and p_fatal > gamma_r
    )

    if not should:
        return False, None, conditions

    # ── Find minimal entry cell: first plan cell entering R_hazard ──
    block_cell = None
    for p in prefix:
        if p in R_hazard:
            block_cell = p
            break

    if block_cell is None:
        return False, None, conditions

    # ── Deadlock check: verify path still exists after block ──
    test_passable = passable_mask.copy()
    test_passable[block_cell] = False
    alt_plan = bounded_astar(
        agent_pos, goal,
        belief_cost_mean, agent_belief_risk, belief_cost_var,
        budget=search_budget, lambda_risk=3.0, passable_mask=test_passable,
    )
    if not alt_plan or alt_plan[-1] != goal:
        # Would create deadlock — abort
        return False, None, conditions

    return True, block_cell, conditions


# ── Timeout estimation ───────────────────────────────────────────────

def compute_timeout_risk(
    wait_plan: list[tuple[int, int]],
    time_left: int,
    replan_penalty: float = 2.0,
    n_expected_replans: float = 2.0,
    tau: float = 3.0,
) -> float:
    """
    Soft sigmoid timeout risk.

    r_timeout = sigma((T_finish - T_left) / tau)

    where T_finish = plan_cost + replan_penalty * n_replans
    """
    if not wait_plan:
        return 1.0
    plan_cost = len(wait_plan) - 1
    est_finish = plan_cost + replan_penalty * n_expected_replans
    x = (est_finish - time_left) / max(tau, 0.1)
    return float(1.0 / (1.0 + np.exp(-x)))
