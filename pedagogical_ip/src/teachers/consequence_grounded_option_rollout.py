"""Consequence-Grounded Option Rollout — Makes interventions affect action simulation.

Routes WARN/UNLOCK/ITEM_DROP effects into modified BranchAttributes,
then uses ActionPredictor to compute counterfactual action distributions.

This is the missing link from Task 4: intervention consequences now
actually alter the predicted next-action distribution.

  WARN:      risk_penalty -= α_warn (agent perceives less risk → more likely safe)
  UNLOCK:    shortcut_bonus += α_unlock (new path available → agent may switch)
  ITEM_DROP: risk_penalty *= (1 - γ_shield) (shielded → agent more willing to traverse)

Does NOT modify any existing module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

from ..agents.stochastic_agent_policy import BranchAttributes
from ..agents.agent_belief_state import AgentBelief
from .action_predictor import ActionPredictor, ActionDistribution
from .intervention_semantics import InterventionSemantics


@dataclass
class ConsequenceConfig:
    """Grounding strength for each intervention type."""
    alpha_warn: float = 0.15    # risk_penalty reduction from WARN
    alpha_unlock: float = 0.5   # shortcut_bonus from UNLOCK
    gamma_shield: float = 0.5   # risk multiplier from ITEM_DROP (same as InterventionSemantics)


@dataclass
class RolloutResult:
    """Result of consequence-grounded option evaluation."""
    option: str
    original_dist: ActionDistribution
    counterfactual_dist: ActionDistribution
    p_safe_original: float      # P(safe branch) before intervention
    p_safe_counterfactual: float  # P(safe branch) after intervention
    success_lift: float          # ΔP(safe branch)


class ConsequenceGroundedRollout:
    """Makes intervention consequences affect action prediction.

    For each option, modifies BranchAttributes to reflect the intervention's
    consequence, then computes the counterfactual action distribution.

    Usage:
        cgr = ConsequenceGroundedRollout(action_predictor)
        result = cgr.evaluate_option("WARN", branches, agent_belief, world_state)
        print(result.success_lift)
    """

    def __init__(self,
                 action_predictor: ActionPredictor,
                 config: Optional[ConsequenceConfig] = None,
                 semantics: Optional[InterventionSemantics] = None):
        self._predictor = action_predictor
        self._config = config or ConsequenceConfig()
        self._semantics = semantics or InterventionSemantics()

    def evaluate_option(self,
                        option: str,
                        branches: list[BranchAttributes],
                        agent_belief: AgentBelief,
                        world_state=None,
                        safe_branch_idx: int = 0,
                        ) -> RolloutResult:
        """Evaluate one intervention option's consequence.

        Args:
            option: "NONE", "WARN", "UNLOCK", "ITEM_DROP"
            branches: current BranchAttributes
            agent_belief: current agent belief
            world_state: optional world state
            safe_branch_idx: which branch is "correct" (for success_lift)
        Returns:
            RolloutResult with original and counterfactual distributions
        """
        # Original distribution (no intervention)
        orig = self._predictor.predict(world_state, agent_belief, branches)

        # Modified branches after intervention
        mod_branches = self.apply_consequence(option, branches)

        # Counterfactual distribution
        cf = self._predictor.predict(world_state, agent_belief, mod_branches)

        p_safe_orig = float(orig.probs[safe_branch_idx])
        p_safe_cf = float(cf.probs[safe_branch_idx])

        return RolloutResult(
            option=option,
            original_dist=orig,
            counterfactual_dist=cf,
            p_safe_original=p_safe_orig,
            p_safe_counterfactual=p_safe_cf,
            success_lift=p_safe_cf - p_safe_orig,
        )

    def apply_consequence(self, option: str,
                          branches: list[BranchAttributes],
                          ) -> list[BranchAttributes]:
        """Apply intervention consequence to BranchAttributes.

        Returns modified copy — does NOT modify input.
        """
        if option == "NONE":
            return list(branches)

        cfg = self._config
        mod = []
        for i, b in enumerate(branches):
            new_risk = b.risk_penalty
            new_safety = b.safety_score
            new_shortcut = b.shortcut_bonus
            new_tempt = b.temptation_score

            if option == "WARN":
                # WARN = belief evidence: agent becomes MORE risk-aware
                # Increases perceived risk on risky branches (utility J_risk goes up)
                # Slightly boosts safety perception on safe branches
                # Net effect: agent shifts toward safe branch
                if b.risk_penalty > 0.15:
                    new_risk = b.risk_penalty + cfg.alpha_warn
                if b.safety_score > 0.5:
                    new_safety = min(1.0, b.safety_score + cfg.alpha_warn * 0.5)

            elif option == "UNLOCK":
                # UNLOCK = affordance reveal: new path bonus
                # The previously locked path now has shortcut value
                new_shortcut = b.shortcut_bonus + cfg.alpha_unlock
                # Also reduces time pressure implicitly

            elif option == "ITEM_DROP":
                # ITEM_DROP = traversal mitigation: shield reduces risk cost
                new_risk = b.risk_penalty * (1.0 - cfg.gamma_shield)

            mod.append(BranchAttributes(
                safety_score=new_safety,
                temptation_score=new_tempt,
                texture_novelty=b.texture_novelty,
                shortcut_bonus=new_shortcut,
                risk_penalty=new_risk,
            ))
        return mod

    def rank_options(self,
                     branches: list[BranchAttributes],
                     agent_belief: AgentBelief,
                     world_state=None,
                     safe_branch_idx: int = 0,
                     ) -> Dict[str, RolloutResult]:
        """Evaluate all options and rank by success_lift.

        Returns dict of option → RolloutResult, ordered by success_lift.
        """
        options = ["NONE", "WARN", "UNLOCK", "ITEM_DROP"]
        results = {}
        for opt in options:
            results[opt] = self.evaluate_option(
                opt, branches, agent_belief, world_state, safe_branch_idx)
        return dict(sorted(results.items(),
                           key=lambda x: x[1].success_lift, reverse=True))

    def best_option(self,
                    branches: list[BranchAttributes],
                    agent_belief: AgentBelief,
                    world_state=None,
                    safe_branch_idx: int = 0,
                    ) -> str:
        """Return the option with highest success_lift."""
        ranked = self.rank_options(branches, agent_belief, world_state,
                                  safe_branch_idx)
        return next(iter(ranked))
