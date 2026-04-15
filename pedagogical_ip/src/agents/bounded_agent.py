"""
DEPRECATED — V0 agent class. Not used by the V2 canonical runner path.

BoundedRationalAgent — The internal learner/planner inside the environment.

v1a: Explicit partial plan tracking with online replanning triggers.
Orchestrates: observe → process teacher action → update belief → plan/replan → act.

The V2 runner (lattice_v2_runner.py) inlines equivalent logic directly.
This file is kept for backward compatibility and reference only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .belief import BeliefMap, update_belief_cell, apply_warning_to_belief
# NOTE: generate_observations was removed in Batch D (zero callers on main-line).
# BoundedRationalAgent.observe_and_update() uses it, but this entire class is
# DEPRECATED and unused by the V2 runner. The import is deferred to method body.
from .planner_astar import plan_next_action, bounded_astar, sample_search_budget, MOVES


@dataclass
class AgentState:
    """Observable state of the agent (for teacher / logging)."""
    pos: tuple[int, int]
    has_object: bool
    has_shield: bool
    shield_remaining: int
    replan_count: int
    last_action: str


class BoundedRationalAgent:
    """
    Internal bounded-rational learner.

    v1a: Maintains an explicit partial plan. Replans only when:
      - Plan is exhausted (all steps executed)
      - Agent deviated from planned trajectory
      - Teacher intervention invalidated the plan
    Budget is sampled from a discrete NegBin approximation per replan.
    Action is deterministic from plan with small ε-greedy fallback.
    """

    def __init__(
        self,
        height: int,
        width: int,
        start_pos: tuple[int, int],
        # Belief priors
        prior_cost_mean: float = 1.5,
        prior_cost_var: float = 4.0,
        prior_risk_mean: float = 0.1,
        prior_risk_var: float = 0.25,
        # Observation model
        self_noise_var: float = 0.001,
        neighbor_noise_var: float = 1.0,
        neighbor_radius: int = 1,
        # Planner
        search_budget: int = 30,
        lambda_risk: float = 0.8,
        lambda_uncertainty: float = 0.02,
        # Bounded rationality params (v1a)
        budget_class: int = 8,
        epsilon_greedy: float = 0.05,
        # RNG
        rng: Optional[np.random.Generator] = None,
    ):
        self.height = height
        self.width = width
        self.pos = start_pos
        self.has_object = False
        self.has_shield = False
        self.shield_remaining = 0

        self.belief = BeliefMap.from_prior(
            height, width,
            prior_cost_mean, prior_cost_var,
            prior_risk_mean, prior_risk_var,
        )

        # Observation parameters
        self.self_noise_var = self_noise_var
        self.neighbor_noise_var = neighbor_noise_var
        self.neighbor_radius = neighbor_radius

        # Planner parameters
        self.search_budget = search_budget
        self.lambda_risk = lambda_risk
        self.lambda_uncertainty = lambda_uncertainty

        # v1a: bounded rationality
        self.budget_class = budget_class
        self.epsilon_greedy = epsilon_greedy

        # v1a: explicit partial plan
        self.current_plan: list[tuple[int, int]] = []  # sequence of positions
        self.plan_step_idx: int = 0
        self.plan_invalidated: bool = True  # start needing a plan

        self.rng = rng or np.random.default_rng()
        self.replan_count = 0
        self.last_action = "NONE"
        self.action_history: list[str] = []

    def reset(self, start_pos: tuple[int, int],
              prior_cost_mean: float = 1.5,
              prior_cost_var: float = 4.0,
              prior_risk_mean: float = 0.1,
              prior_risk_var: float = 0.25) -> None:
        """Reset agent state for a new episode."""
        self.pos = start_pos
        self.has_object = False
        self.has_shield = False
        self.shield_remaining = 0
        self.belief = BeliefMap.from_prior(
            self.height, self.width,
            prior_cost_mean, prior_cost_var,
            prior_risk_mean, prior_risk_var,
        )
        self.current_plan = []
        self.plan_step_idx = 0
        self.plan_invalidated = True
        self.replan_count = 0
        self.last_action = "NONE"
        self.action_history = []

    def observe_and_update(
        self,
        true_cost: np.ndarray,
        true_risk: np.ndarray,
    ) -> None:
        """Generate observations from current position and update beliefs.

        DEPRECATED: This method relies on the removed V0 generate_observations.
        BoundedRationalAgent is archival — use lattice_v2_runner instead.
        """
        raise NotImplementedError(
            "BoundedRationalAgent.observe_and_update() is deprecated. "
            "The V0 generate_observations() was removed in Batch D. "
            "Use lattice_v2_runner.py with observe_features() instead."
        )

    def process_teacher_action(
        self,
        action_type: str,
        action_param: str = "",
        shield_duration: int = 5,
        warn_sensitivity: float = 0.4,
    ) -> None:
        """
        Process a teacher/robot intervention.

        - WARN: update belief via warning (v1a: uses apply_rsa_warning if available)
        - DROP_SHIELD: agent gains temporary shield
        - UNLOCK_DOOR: handled by environment (cost map change)
        - WAIT: no-op

        Any intervention except WAIT invalidates the current plan.
        """
        if action_type == "WARN" and action_param:
            # v1a: try RSA-based warning update first
            try:
                from .belief import apply_rsa_warning
                apply_rsa_warning(self.belief, action_param, warn_sensitivity)
            except ImportError:
                apply_warning_to_belief(self.belief, action_param)
            self.plan_invalidated = True
        elif action_type == "DROP_SHIELD":
            self.has_shield = True
            self.shield_remaining = shield_duration
            self.plan_invalidated = True
        elif action_type == "UNLOCK_DOOR":
            self.plan_invalidated = True
        elif action_type == "BLOCK_PATH":
            # Pedagogical blocking: force replan, but don't change beliefs
            self.plan_invalidated = True

    def _needs_replan(self, goal: tuple[int, int]) -> bool:
        """Check if agent needs to replan."""
        # Plan was explicitly invalidated (by intervention)
        if self.plan_invalidated:
            return True
        # No plan exists
        if not self.current_plan:
            return True
        # Plan exhausted
        if self.plan_step_idx >= len(self.current_plan):
            return True
        # Agent deviated from planned trajectory
        if self.plan_step_idx < len(self.current_plan):
            expected_pos = self.current_plan[self.plan_step_idx]
            if self.pos != expected_pos:
                return True
        return False

    def plan_and_act(
        self,
        goal: tuple[int, int],
        passable_mask: np.ndarray,
    ) -> tuple[str, tuple[int, int]]:
        """
        v1a: Explicit plan tracking with online replanning.

        Replans only when:
        1. Plan exhausted
        2. Agent deviated from plan
        3. Teacher intervention invalidated plan

        Budget sampled from discrete NegBin approximation.
        Action is deterministic from plan with ε-greedy fallback.
        """
        # Check if we need to replan
        if self._needs_replan(goal):
            budget = sample_search_budget(self.budget_class, self.rng)
            path = bounded_astar(
                self.pos, goal,
                self.belief.cost_mean, self.belief.risk_mean,
                self.belief.cost_var,
                budget=budget,
                lambda_risk=self.lambda_risk,
                lambda_uncertainty=self.lambda_uncertainty,
                passable_mask=passable_mask,
            )
            self.current_plan = path
            self.plan_step_idx = 0
            self.plan_invalidated = False
            self.replan_count += 1

            # Find our current position in the plan
            if self.current_plan and self.current_plan[0] == self.pos:
                self.plan_step_idx = 1  # skip start pos

        # Extract next position from plan
        if self.plan_step_idx < len(self.current_plan):
            next_pos = self.current_plan[self.plan_step_idx]
            self.plan_step_idx += 1
        else:
            # Fallback: single-step planning
            _, next_pos = plan_next_action(
                self.pos, goal,
                self.belief.cost_mean, self.belief.risk_mean,
                self.belief.cost_var,
                budget=4,
                lambda_risk=self.lambda_risk,
                lambda_uncertainty=self.lambda_uncertainty,
                passable_mask=passable_mask,
            )

        # ε-greedy: with small probability, take a random valid move
        if self.rng.random() < self.epsilon_greedy:
            valid_moves = []
            for dr, dc, name in MOVES:
                nr, nc = self.pos[0] + dr, self.pos[1] + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if passable_mask[nr, nc]:
                        valid_moves.append(((nr, nc), name))
            if valid_moves:
                next_pos, action_name = valid_moves[self.rng.integers(len(valid_moves))]
                self.last_action = action_name
                self.action_history.append(action_name)
                self._tick_shield()
                return action_name, next_pos

        # Determine action name from position delta
        dr = next_pos[0] - self.pos[0]
        dc = next_pos[1] - self.pos[1]
        action_name = "STAY"
        for mdr, mdc, name in MOVES:
            if dr == mdr and dc == mdc:
                action_name = name
                break

        self.last_action = action_name
        self.action_history.append(action_name)
        self._tick_shield()

        return action_name, next_pos

    def _tick_shield(self) -> None:
        """Decrement shield timer."""
        if self.has_shield:
            self.shield_remaining -= 1
            if self.shield_remaining <= 0:
                self.has_shield = False
                self.shield_remaining = 0

    def move_to(self, new_pos: tuple[int, int]) -> None:
        """Update agent position."""
        self.pos = new_pos

    def get_state(self) -> AgentState:
        """Return current observable state."""
        return AgentState(
            pos=self.pos,
            has_object=self.has_object,
            has_shield=self.has_shield,
            shield_remaining=self.shield_remaining,
            replan_count=self.replan_count,
            last_action=self.last_action,
        )

    def get_history(self, k: int = 5) -> list[str]:
        """Return last K actions."""
        return self.action_history[-k:]
