"""Goal-Conditional Curriculum Hook — Makes macro scoring goal-conditional.

S_macro(ℓ) = E_{q(g,θ,z)} [Q_online(ℓ|g,θ,z) + λ·V_teach - λ_over·R_over]
             + β_κ · g_κ(κ̂)

Uses ConsequenceGroundedRollout under each hypothesis,
weighted by JointGoalPrefPosterior.

κ̂ enters as additive macro state term, NOT as posterior latent.

Does NOT modify any existing module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np

from ..agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from ..agents.agent_belief_state import AgentBelief
from ..agents.world_state import WorldState
from .compositional_goal_hypotheses import GoalHypothesisSpace, DEFAULT_GOAL_SPACE
from .joint_goal_pref_posterior import JointGoalPrefPosterior
from .consequence_grounded_option_rollout import (
    ConsequenceGroundedRollout, ConsequenceConfig,
)
from .action_predictor import ActionPredictor


@dataclass
class CurriculumConfig:
    """Config for goal-conditional curriculum scoring."""
    lambda_teach: float = 0.5      # teaching value weight
    lambda_infl: float = 4.0       # inflation penalty (from T4)
    beta_kappa: float = 0.02       # κ̂ macro bonus (from 5D architecture)
    min_confidence: float = 0.15   # minimum posterior confidence to act


@dataclass
class CurriculumDecision:
    """Result of goal-conditional curriculum decision."""
    chosen_option: str
    scores: Dict[str, float]
    goal_alignment: Dict[str, str]  # goal → best option for that goal
    posterior_entropy: float
    kappa_bonus: float


class GoalConditionalCurriculumHook:
    """Goal-conditional macro curriculum scorer.

    Usage:
        hook = GoalConditionalCurriculumHook(ap, goal_space)
        decision = hook.decide(posterior, branches, agent_belief, ws, kappa_hat)
    """

    def __init__(self,
                 action_predictor: ActionPredictor,
                 goal_space: Optional[GoalHypothesisSpace] = None,
                 config: Optional[CurriculumConfig] = None):
        self._ap = action_predictor
        self._goal_space = goal_space or DEFAULT_GOAL_SPACE
        self._config = config or CurriculumConfig()
        self._cgr = ConsequenceGroundedRollout(action_predictor)

    def decide(self,
               posterior: JointGoalPrefPosterior,
               branches: list[BranchAttributes],
               agent_belief: AgentBelief,
               world_state: Optional[WorldState] = None,
               kappa_hat: float = 0.0,
               nu_hat: float = 0.0,
               safe_branch_idx: int = 0,
               ) -> CurriculumDecision:
        """Make goal-conditional curriculum decision.

        Computes:
            S(option) = E_{q(g,θ)} [success_lift(option|g,θ)]
                       + λ_teach · teaching_value
                       - λ_infl · inflation_cost
                       + β_κ · g_κ(κ̂)

        Args:
            posterior: joint goal-pref posterior
            branches: current branch attributes
            agent_belief: current agent belief
            world_state: current world state
            kappa_hat: estimated κ̂ from 5D observer
            nu_hat: estimated ν̂ from 5D observer
            safe_branch_idx: which branch is "safe"
        """
        cfg = self._config
        mg = posterior.marginal_goal()
        mp = posterior.marginal_pref()

        options = ["NONE", "WARN", "UNLOCK", "ITEM_DROP"]
        scores = {}
        goal_alignment = {}

        for opt in options:
            # ── Q_online: success lift under posterior ──
            expected_lift = 0.0
            per_goal_best = {}

            for gl, gw in mg.items():
                gh = self._goal_space.get(gl)
                # Weighted average over preference types
                lift_for_goal = 0.0
                for tl, tw in mp.items():
                    orig_probs = self._goal_space.compute_choice_probs(
                        branches, gh, tl)
                    mod_branches = self._cgr.apply_consequence(opt, branches)
                    mod_probs = self._goal_space.compute_choice_probs(
                        mod_branches, gh, tl)
                    lift = float(mod_probs[safe_branch_idx] - orig_probs[safe_branch_idx])
                    lift_for_goal += tw * lift
                expected_lift += gw * lift_for_goal
                per_goal_best[gl] = lift_for_goal

            # ── V_teach: teaching value (posterior entropy reduction) ──
            # Higher entropy → more value in intervening
            h = posterior.entropy()
            h_max = posterior.max_entropy()
            teaching_value = h / h_max if h_max > 0 else 0.0

            # ── R_over: inflation cost ──
            inflation_cost = 0.0
            if opt != "NONE":
                # penalty increases with nu_hat (already dependent)
                inflation_cost = max(0.0, nu_hat) * 0.1

            # ── κ̂ bonus ──
            kappa_bonus = cfg.beta_kappa * max(0.0, kappa_hat)

            # ── Composite score ──
            s = (expected_lift
                 + cfg.lambda_teach * teaching_value * (1.0 if opt != "NONE" else 0.0)
                 - cfg.lambda_infl * inflation_cost
                 + kappa_bonus * (1.0 if opt != "NONE" else 0.5))

            scores[opt] = s

        # Find best option per goal
        for gl in mg:
            gh = self._goal_space.get(gl)
            best_opt = "NONE"
            best_lift = -np.inf
            for opt in options:
                mod_branches = self._cgr.apply_consequence(opt, branches)
                for tl, tw in mp.items():
                    probs = self._goal_space.compute_choice_probs(
                        mod_branches, gh, tl)
                    if float(probs[safe_branch_idx]) > best_lift:
                        best_lift = float(probs[safe_branch_idx])
                        best_opt = opt
            goal_alignment[gl] = best_opt

        chosen = max(scores, key=scores.get)

        return CurriculumDecision(
            chosen_option=chosen,
            scores=scores,
            goal_alignment=goal_alignment,
            posterior_entropy=posterior.entropy(),
            kappa_bonus=cfg.beta_kappa * max(0.0, kappa_hat),
        )
