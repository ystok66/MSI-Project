"""
Oracle Teacher Policy — v0.

Has full oracle access to agent's belief maps and the true map.
Evaluates each candidate intervention by estimating its impact on
success probability and learning gain, then selects the best.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..agents.bounded_agent import BoundedRationalAgent
from ..agents.belief import BeliefMap, apply_warning_to_belief
from .interventions import Intervention, InterventionType, WARNING_VOCAB
from .utilities import (
    pedagogical_utility,
    estimate_success_prob,
    estimate_learning_gain,
)


class OracleTeacherPolicy:
    """
    Oracle teacher that sees the agent's belief maps and true environment.

    Strategy:
    1. First evaluate WAIT: if agent is on track, let it learn.
    2. If risk of failure is high, compare WARN / UNLOCK / SHIELD.
    3. Pick the action with highest pedagogical utility.
    """

    def __init__(
        self,
        w_success: float = 1.0,
        w_learn: float = 0.3,
        w_cost: float = 0.2,
        w_takeover: float = 0.1,
        intervention_costs: Optional[dict[str, float]] = None,
        rollout_budget: int = 30,
    ):
        self.w_success = w_success
        self.w_learn = w_learn
        self.w_cost = w_cost
        self.w_takeover = w_takeover
        self.costs = intervention_costs or {
            "WAIT": 0.0,
            "WARN": 0.1,
            "UNLOCK_DOOR": 0.3,
            "DROP_SHIELD": 0.5,
        }
        self.rollout_budget = rollout_budget

    def select_action(
        self,
        agent: BoundedRationalAgent,
        true_cost: np.ndarray,
        true_risk: np.ndarray,
        goal: tuple[int, int],
        time_left: int,
        risk_budget_left: float,
        passable_mask: np.ndarray,
        door_positions: list[tuple[int, int]] | None = None,
        locked_doors: set[tuple[int, int]] | None = None,
    ) -> tuple[Intervention, dict]:
        """
        Choose the best intervention.

        Returns (Intervention, info_dict) where info_dict has
        per-action utility scores for logging.
        """
        door_positions = door_positions or []
        locked_doors = locked_doors or set()

        candidates: list[tuple[Intervention, float, float, float]] = []

        # --- Score WAIT ---
        wait_success = estimate_success_prob(
            agent.pos, goal,
            agent.belief.cost_mean, agent.belief.risk_mean,
            agent.belief.cost_var,
            time_left, risk_budget_left, passable_mask,
            budget=self.rollout_budget,
        )
        # WAIT has positive learning gain (agent continues to explore)
        wait_learning = self._estimate_exploration_gain(agent, time_left)
        wait_util = pedagogical_utility(
            wait_success, wait_learning, self.costs["WAIT"], 0.0,
            self.w_success, self.w_learn, self.w_cost, self.w_takeover,
        )
        candidates.append((Intervention.wait(), wait_util, wait_success, wait_learning))

        # --- Score WARN variants ---
        best_warn = self._score_best_warning(
            agent, goal, time_left, risk_budget_left, passable_mask,
        )
        if best_warn is not None:
            candidates.append(best_warn)

        # --- Score UNLOCK_DOOR ---
        if locked_doors:
            unlock_intervention, unlock_util, unlock_success, unlock_learning = \
                self._score_unlock(
                    agent, goal, time_left, risk_budget_left,
                    passable_mask, locked_doors,
                )
            candidates.append((unlock_intervention, unlock_util, unlock_success, unlock_learning))

        # --- Score DROP_SHIELD ---
        shield_intervention, shield_util, shield_success, shield_learning = \
            self._score_shield(
                agent, goal, time_left, risk_budget_left, passable_mask,
            )
        candidates.append((shield_intervention, shield_util, shield_success, shield_learning))

        # Pick best
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0]

        info = {
            "predicted_wait_success": wait_success,
            "scores": {c[0].type.value: round(c[1], 4) for c in candidates},
        }

        # Extract per-type success predictions for logging
        for c_intervention, c_util, c_success, c_learning in candidates:
            key = f"predicted_{c_intervention.type.value.lower()}_success"
            info[key] = round(c_success, 4)

        return best[0], info

    def _estimate_exploration_gain(
        self,
        agent: BoundedRationalAgent,
        time_left: int,
    ) -> float:
        """Estimate how much the agent will learn by continuing to explore."""
        unvisited_frac = 1.0 - agent.belief.visited_mask.mean()
        time_frac = min(1.0, time_left / 30.0)
        return unvisited_frac * time_frac * 0.5

    def _score_best_warning(
        self,
        agent: BoundedRationalAgent,
        goal: tuple[int, int],
        time_left: int,
        risk_budget_left: float,
        passable_mask: np.ndarray,
    ) -> tuple[Intervention, float, float, float] | None:
        """Try each warning and pick the best one."""
        best = None
        best_util = -np.inf

        for msg in WARNING_VOCAB:
            belief_copy = agent.belief.copy()
            apply_warning_to_belief(belief_copy, msg)

            warn_success = estimate_success_prob(
                agent.pos, goal,
                belief_copy.cost_mean, belief_copy.risk_mean,
                belief_copy.cost_var,
                time_left, risk_budget_left, passable_mask,
                budget=self.rollout_budget,
            )
            warn_learning = estimate_learning_gain(agent.belief, belief_copy)

            util = pedagogical_utility(
                warn_success, warn_learning, self.costs["WARN"], 0.0,
                self.w_success, self.w_learn, self.w_cost, self.w_takeover,
            )
            if util > best_util:
                best_util = util
                best = (Intervention.warn(msg), util, warn_success, warn_learning)

        return best

    def _score_unlock(
        self,
        agent: BoundedRationalAgent,
        goal: tuple[int, int],
        time_left: int,
        risk_budget_left: float,
        passable_mask: np.ndarray,
        locked_doors: set[tuple[int, int]],
    ) -> tuple[Intervention, float, float, float]:
        """Score unlocking the first locked door."""
        # Simulate unlocking: make door passable in belief
        unlocked_mask = passable_mask.copy()
        belief_copy = agent.belief.copy()
        for dr, dc in locked_doors:
            unlocked_mask[dr, dc] = True
            belief_copy.cost_mean[dr, dc] = 1.0
            belief_copy.cost_var[dr, dc] = 0.1

        unlock_success = estimate_success_prob(
            agent.pos, goal,
            belief_copy.cost_mean, belief_copy.risk_mean,
            belief_copy.cost_var,
            time_left, risk_budget_left, unlocked_mask,
            budget=self.rollout_budget,
        )
        unlock_learning = estimate_learning_gain(agent.belief, belief_copy)
        # Unlocking has moderate takeover cost (robot directly changes world)
        util = pedagogical_utility(
            unlock_success, unlock_learning, self.costs["UNLOCK_DOOR"], 0.3,
            self.w_success, self.w_learn, self.w_cost, self.w_takeover,
        )
        return (Intervention.unlock_door("0"), util, unlock_success, unlock_learning)

    def _score_shield(
        self,
        agent: BoundedRationalAgent,
        goal: tuple[int, int],
        time_left: int,
        risk_budget_left: float,
        passable_mask: np.ndarray,
    ) -> tuple[Intervention, float, float, float]:
        """Score dropping a shield — reduces risk along agent's path."""
        # Simulate: with shield, agent's risk exposure is greatly reduced
        belief_copy = agent.belief.copy()
        belief_copy.risk_mean[:] *= 0.1  # shield reduces perceived risk

        shield_success = estimate_success_prob(
            agent.pos, goal,
            belief_copy.cost_mean, belief_copy.risk_mean,
            belief_copy.cost_var,
            time_left, risk_budget_left + 0.5, passable_mask,
            budget=self.rollout_budget,
        )
        shield_learning = 0.0  # shield doesn't help learning
        # Shield has moderate takeover
        util = pedagogical_utility(
            shield_success, shield_learning, self.costs["DROP_SHIELD"], 0.5,
            self.w_success, self.w_learn, self.w_cost, self.w_takeover,
        )
        return (Intervention.drop_shield(5), util, shield_success, shield_learning)
