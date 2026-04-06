"""Compositional Goal Bridge — Connects CGC-v2 to POMDP stack.

Adapter for routing CGC-v2 episode data through T3-T5 interfaces:
  - WorldState / AgentBelief / ActionPredictor
  - ConsequenceGroundedRollout
  - OptionInterventionController

Does NOT modify any existing module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np

from ..agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from ..agents.agent_belief_state import AgentBelief
from ..agents.world_state import WorldState
from .compositional_goal_hypotheses import GoalHypothesisSpace, DEFAULT_GOAL_SPACE
from .joint_goal_pref_posterior import JointGoalPrefPosterior, THETA_2
from .consequence_grounded_option_rollout import (
    ConsequenceGroundedRollout, ConsequenceConfig, RolloutResult,
)
from .action_predictor import ActionPredictor


@dataclass
class GoalConditionalOptionScore:
    """Score of an intervention option conditioned on goal posterior."""
    option: str
    expected_success_lift: float   # E_{q(g,θ)} [ΔP(safe)]
    goal_weighted_lifts: Dict[str, float]   # per-goal lift
    best_for_goals: List[str]      # which goals this option is best for


class CompositionalGoalBridge:
    """Bridge between CGC-v2 compositional goals and POMDP intervention stack.

    Usage:
        bridge = CompositionalGoalBridge(action_predictor, consequence_rollout)
        # Update posterior from observation
        bridge.update_posterior(posterior, ws, branches, action)
        # Score interventions conditional on goal posterior
        scores = bridge.score_options(posterior, branches, ws)
    """

    def __init__(self,
                 action_predictor: ActionPredictor,
                 consequence_rollout: Optional[ConsequenceGroundedRollout] = None,
                 goal_space: Optional[GoalHypothesisSpace] = None):
        self._ap = action_predictor
        self._cgr = consequence_rollout or ConsequenceGroundedRollout(action_predictor)
        self._goal_space = goal_space or DEFAULT_GOAL_SPACE

    def update_posterior(self,
                         posterior: JointGoalPrefPosterior,
                         world_state: WorldState,
                         branches: list[BranchAttributes],
                         observed_action: int,
                         agent_belief: Optional[AgentBelief] = None):
        """Update the joint posterior from an observed action."""
        posterior.update(world_state, branches, observed_action, agent_belief)

    def score_options(self,
                      posterior: JointGoalPrefPosterior,
                      branches: list[BranchAttributes],
                      agent_belief: AgentBelief,
                      world_state: Optional[WorldState] = None,
                      safe_branch_idx: int = 0,
                      ) -> Dict[str, GoalConditionalOptionScore]:
        """Score intervention options weighted by goal posterior.

        For each option, computes:
            E_{q(g,θ)} [success_lift(option | g, θ)]

        Returns dict of option → GoalConditionalOptionScore.
        """
        mg = posterior.marginal_goal()
        mp = posterior.marginal_pref()

        options = ["NONE", "WARN", "UNLOCK", "ITEM_DROP"]
        results = {}

        for opt in options:
            # Evaluate option's consequence
            r = self._cgr.evaluate_option(
                opt, branches, agent_belief, world_state, safe_branch_idx)

            # Compute goal-weighted lift
            goal_lifts = {}
            expected_lift = 0.0
            for gl, gw in mg.items():
                # For each goal, how does this option's consequence affect
                # the goal-conditioned action distribution?
                gh = self._goal_space.get(gl)
                # Original probs under this goal
                orig_probs = self._goal_space.compute_choice_probs(
                    branches, gh, posterior.predicted_pref())
                # Modified probs after consequence
                mod_branches = self._cgr.apply_consequence(opt, branches)
                mod_probs = self._goal_space.compute_choice_probs(
                    mod_branches, gh, posterior.predicted_pref())
                lift = float(mod_probs[safe_branch_idx] - orig_probs[safe_branch_idx])
                goal_lifts[gl] = lift
                expected_lift += gw * lift

            # Which goals benefit most from this option
            best_goals = [gl for gl, lf in goal_lifts.items()
                         if lf > 0.01]

            results[opt] = GoalConditionalOptionScore(
                option=opt,
                expected_success_lift=expected_lift,
                goal_weighted_lifts=goal_lifts,
                best_for_goals=best_goals,
            )

        return dict(sorted(results.items(),
                           key=lambda x: x[1].expected_success_lift, reverse=True))

    def best_option(self,
                    posterior: JointGoalPrefPosterior,
                    branches: list[BranchAttributes],
                    agent_belief: AgentBelief,
                    world_state: Optional[WorldState] = None,
                    safe_branch_idx: int = 0,
                    ) -> str:
        """Return option with highest expected goal-weighted lift."""
        scores = self.score_options(
            posterior, branches, agent_belief, world_state, safe_branch_idx)
        return next(iter(scores))
