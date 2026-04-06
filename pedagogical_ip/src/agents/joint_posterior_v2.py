"""L1 — Coupled Joint Posterior v2: full q(g,θ) table.

Replaces factorized q(g)·q(θ) with explicit joint table.
Includes compatibility prior C_{g,θ} for modeling which
goal-preference pairs are naturally aligned vs conflicting.

q_t(g,θ) ∝ q_{t-1}(g,θ) · P_A(a_t | s_t, g, θ)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, PREFERENCE_TYPES,
)
from .goal_posterior_v1 import GOAL_TYPES, GOAL_REWARD
from .stochastic_agent_policy import PREF_REWARD

N_PREF = len(PREFERENCE_TYPES)
N_GOALS = len(GOAL_TYPES)


def compute_joint_utility(
    branch: BranchAttributes,
    goal: str,
    theta: str,
    params: AgentPolicyParams,
) -> float:
    """Agent utility under both goal g and preference θ.

    U(π | g, θ) = R_goal(π; g) + λ_θ · R_pref(π; θ) - J_risk(π)
    """
    x = branch.to_array()
    r_goal = float(np.dot(GOAL_REWARD[goal], x))
    r_pref = float(np.dot(PREF_REWARD[theta], x))
    return r_goal + params.lambda_theta * r_pref - branch.risk_penalty


def compute_joint_likelihood(
    chosen_idx: int,
    branches: list[BranchAttributes],
    goal: str,
    theta: str,
    params: AgentPolicyParams,
) -> float:
    """P(chose branch_idx | g, θ, params) via softmax + lapse."""
    utilities = np.array([
        compute_joint_utility(b, goal, theta, params) for b in branches
    ])
    scaled = params.beta * utilities
    scaled -= np.max(scaled)
    exp_u = np.exp(scaled)
    softmax_probs = exp_u / (exp_u.sum() + 1e-10)
    n = len(branches)
    uniform = np.ones(n) / n
    mixed = (1 - params.epsilon) * softmax_probs + params.epsilon * uniform
    return float(mixed[chosen_idx])


# Compatibility prior: which (goal, pref) pairs are naturally aligned
# Positive = compatible, Negative = conflicting
COMPATIBILITY = np.zeros((N_GOALS, N_PREF))
for gi, g in enumerate(GOAL_TYPES):
    for pi, p in enumerate(PREFERENCE_TYPES):
        # Dot product of reward vectors = natural compatibility
        COMPATIBILITY[gi, pi] = float(np.dot(GOAL_REWARD[g], PREF_REWARD[p])) * 0.1


@dataclass
class JointPosteriorV2:
    """Full joint posterior q(g, θ) with compatibility prior."""
    log_table: np.ndarray = field(
        default_factory=lambda: COMPATIBILITY.copy())
    observation_count: int = 0
    forgetting_rate: float = 0.015

    @property
    def table(self) -> np.ndarray:
        """Normalized joint probability table [N_GOALS, N_PREF]."""
        lt = self.log_table - np.max(self.log_table)
        t = np.exp(lt)
        return t / (t.sum() + 1e-10)

    @property
    def entropy(self) -> float:
        t = self.table.ravel()
        return float(-np.sum(t * np.log(t + 1e-10)))

    @property
    def max_entropy(self) -> float:
        return float(np.log(N_GOALS * N_PREF))

    @property
    def marginal_goal(self) -> np.ndarray:
        return self.table.sum(axis=1)

    @property
    def marginal_pref(self) -> np.ndarray:
        return self.table.sum(axis=0)

    @property
    def predicted_joint(self) -> tuple[str, str]:
        t = self.table
        idx = np.unravel_index(np.argmax(t), t.shape)
        return GOAL_TYPES[idx[0]], PREFERENCE_TYPES[idx[1]]

    @property
    def predicted_goal(self) -> str:
        return GOAL_TYPES[int(np.argmax(self.marginal_goal))]

    @property
    def predicted_pref(self) -> str:
        return PREFERENCE_TYPES[int(np.argmax(self.marginal_pref))]

    @property
    def joint_confidence(self) -> float:
        return float(np.max(self.table))

    def update_from_choice(
        self,
        chosen_idx: int,
        branches: list[BranchAttributes],
        params: AgentPolicyParams,
    ):
        """Update q(g,θ) from observed agent branch choice."""
        for gi, g in enumerate(GOAL_TYPES):
            for pi, p in enumerate(PREFERENCE_TYPES):
                lik = compute_joint_likelihood(
                    chosen_idx, branches, g, p, params)
                self.log_table[gi, pi] += np.log(lik + 1e-10)

        # Numerical stability
        self.log_table -= np.mean(self.log_table)

        # Forgetting / diffusion
        if self.forgetting_rate > 0:
            t = self.table
            uniform = np.ones_like(t) / (N_GOALS * N_PREF)
            t_diffused = (1 - self.forgetting_rate) * t + self.forgetting_rate * uniform
            self.log_table = np.log(t_diffused + 1e-10)
            self.log_table -= np.mean(self.log_table)

        self.observation_count += 1

    def posterior_predictive_variance(
        self,
        branches: list[BranchAttributes],
        params: AgentPolicyParams,
    ) -> float:
        """Var_{(g,θ)~q}[P(π|s,g,θ)] — joint observation informativeness."""
        t = self.table.ravel()
        p_per_cell = np.array([
            compute_joint_likelihood(0, branches, GOAL_TYPES[gi], PREFERENCE_TYPES[pi], params)
            for gi in range(N_GOALS)
            for pi in range(N_PREF)
        ])
        mean_p = float(np.dot(t, p_per_cell))
        var_p = float(np.dot(t, (p_per_cell - mean_p) ** 2))
        return var_p

    def to_dict(self) -> dict:
        t = self.table
        g_pred, p_pred = self.predicted_joint
        return {
            "predicted_goal": g_pred,
            "predicted_pref": p_pred,
            "joint_confidence": round(self.joint_confidence, 4),
            "entropy": round(self.entropy, 4),
            "marginal_goal": {GOAL_TYPES[i]: round(float(self.marginal_goal[i]), 4)
                              for i in range(N_GOALS)},
            "marginal_pref": {PREFERENCE_TYPES[i]: round(float(self.marginal_pref[i]), 4)
                              for i in range(N_PREF)},
            "n_obs": self.observation_count,
        }
