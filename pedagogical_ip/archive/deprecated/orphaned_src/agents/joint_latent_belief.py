"""K2 — Joint Latent Belief: factorized q(g,θ) ≈ q(g)·q(θ).

Wraps GoalPosteriorV1 + PreferencePosteriorV2 into a unified
joint latent belief that can be read by any tutor/planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .preference_posterior_v2 import PreferencePosteriorV2
from .goal_posterior_v1 import GoalPosteriorV1
from .stochastic_agent_policy import BranchAttributes, AgentPolicyParams


@dataclass
class JointLatentBelief:
    """Factorized joint belief over preference θ and goal g."""
    pref_posterior: PreferencePosteriorV2 = None
    goal_posterior: GoalPosteriorV1 = None

    def __post_init__(self):
        if self.pref_posterior is None:
            self.pref_posterior = PreferencePosteriorV2()
        if self.goal_posterior is None:
            self.goal_posterior = GoalPosteriorV1()

    @property
    def pref_entropy(self) -> float:
        return self.pref_posterior.entropy

    @property
    def goal_entropy(self) -> float:
        return self.goal_posterior.entropy

    @property
    def total_uncertainty(self) -> float:
        """Sum of normalized entropies."""
        return (self.pref_entropy / max(self.pref_posterior.max_entropy, 1e-6)
                + self.goal_entropy / max(self.goal_posterior.max_entropy, 1e-6)) / 2.0

    def update_from_choice(
        self,
        chosen_idx: int,
        branches: list[BranchAttributes],
        params: AgentPolicyParams,
    ):
        """Update both posteriors from observed choice (factorized)."""
        self.pref_posterior.update_from_choice(chosen_idx, branches, params)
        self.goal_posterior.update_from_choice(chosen_idx, branches, params)

    def observation_value(
        self,
        branches: list[BranchAttributes],
        params: AgentPolicyParams,
    ) -> float:
        """Combined observation value from both latent dimensions."""
        v_pref = self.pref_posterior.posterior_predictive_variance(branches, params)
        v_goal = self.goal_posterior.posterior_predictive_variance(branches, params)
        return v_pref + v_goal

    def to_dict(self) -> dict:
        return {
            "preference": self.pref_posterior.to_dict(),
            "goal": self.goal_posterior.to_dict(),
            "total_uncertainty": round(self.total_uncertainty, 4),
        }
