"""K1 — Goal Posterior v1: discrete goal set with behavior-driven inference.

Maintains q(g) over K goal types. Updates via agent choice likelihoods
from stochastic_agent_policy, similar to PreferencePosteriorV2.

Goal types represent what the agent is trying to achieve:
  - goal_safe_short: reach goal via shortest safe path
  - goal_safe_long:  prefer safe path even if longer
  - goal_collect:    collect high-value items (temptation-adjacent)
  - goal_explore:    visit more cells / maximize info
  - goal_direct:     minimize time regardless of safety
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .stochastic_agent_policy import BranchAttributes, AgentPolicyParams

GOAL_TYPES = [
    "goal_safe_short",
    "goal_safe_long",
    "goal_collect",
    "goal_explore",
    "goal_direct",
]
N_GOALS = len(GOAL_TYPES)

# Goal reward weights: how each goal values branch attributes
# Columns: [safety_bonus, tempt_bonus, texture_novelty, shortcut_bonus]
GOAL_REWARD = {
    "goal_safe_short": np.array([2.0, -0.5,  0.0,  1.5]),
    "goal_safe_long":  np.array([3.0, -1.0,  0.0,  0.0]),
    "goal_collect":    np.array([0.0,  2.5,  0.5,  0.0]),
    "goal_explore":    np.array([0.5,  0.5,  2.0,  0.0]),
    "goal_direct":     np.array([0.0,  0.0,  0.0,  3.0]),
}


def compute_goal_likelihood(
    chosen_idx: int,
    branches: list[BranchAttributes],
    goal: str,
    params: AgentPolicyParams,
) -> float:
    """P(chose branch_idx | goal, params) via softmax."""
    utilities = []
    w = GOAL_REWARD[goal]
    for b in branches:
        u = float(np.dot(w, b.to_array()))
        utilities.append(u)
    utilities = np.array(utilities)

    scaled = params.beta * utilities
    scaled -= np.max(scaled)
    exp_u = np.exp(scaled)
    softmax_probs = exp_u / (exp_u.sum() + 1e-10)

    n = len(branches)
    uniform = np.ones(n) / n
    mixed = (1 - params.epsilon) * softmax_probs + params.epsilon * uniform
    return float(mixed[chosen_idx])


@dataclass
class GoalPosteriorV1:
    """Robot's Bayesian belief over agent's latent goal."""
    log_probs: np.ndarray = field(
        default_factory=lambda: np.zeros(N_GOALS))
    observation_count: int = 0
    forgetting_rate: float = 0.02

    @property
    def probs(self) -> np.ndarray:
        lp = self.log_probs - np.max(self.log_probs)
        p = np.exp(lp)
        return p / (p.sum() + 1e-10)

    @property
    def entropy(self) -> float:
        p = self.probs
        return float(-np.sum(p * np.log(p + 1e-10)))

    @property
    def max_entropy(self) -> float:
        return float(np.log(N_GOALS))

    @property
    def predicted_type(self) -> str:
        return GOAL_TYPES[int(np.argmax(self.probs))]

    @property
    def predicted_prob(self) -> float:
        return float(np.max(self.probs))

    def update_from_choice(
        self,
        chosen_idx: int,
        branches: list[BranchAttributes],
        params: AgentPolicyParams,
    ):
        """Update q(g) from observed agent branch choice."""
        for i, gtype in enumerate(GOAL_TYPES):
            likelihood = compute_goal_likelihood(chosen_idx, branches, gtype, params)
            self.log_probs[i] += np.log(likelihood + 1e-10)

        self.log_probs -= np.mean(self.log_probs)

        if self.forgetting_rate > 0:
            p = self.probs
            p_diffused = (1 - self.forgetting_rate) * p + \
                         self.forgetting_rate * np.ones(N_GOALS) / N_GOALS
            self.log_probs = np.log(p_diffused + 1e-10)
            self.log_probs -= np.mean(self.log_probs)

        self.observation_count += 1

    def posterior_predictive_variance(
        self,
        branches: list[BranchAttributes],
        params: AgentPolicyParams,
    ) -> float:
        """Var_g~q[P(π|s,g)] — observation informativeness."""
        q = self.probs
        p_per_goal = np.array([
            compute_goal_likelihood(0, branches, gtype, params)
            for gtype in GOAL_TYPES
        ])
        mean_p = float(np.dot(q, p_per_goal))
        var_p = float(np.dot(q, (p_per_goal - mean_p) ** 2))
        return var_p

    def to_dict(self) -> dict:
        p = self.probs
        return {
            "probs": {GOAL_TYPES[i]: round(float(p[i]), 4) for i in range(N_GOALS)},
            "predicted": self.predicted_type,
            "predicted_prob": round(self.predicted_prob, 4),
            "entropy": round(self.entropy, 4),
            "n_obs": self.observation_count,
        }
