"""J1 — Stochastic Bounded-Rational Agent Policy.

Agent selects branches via softmax + lapse mixture, conditioned
on latent preference type θ.

P(π | s, θ) = (1-ε) · softmax(β · U(π; θ)) + ε · Uniform

U(π; θ) = R_goal(π) + λ_θ · R_pref(π; θ) - J_risk(π)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Preference reward weights: how much each θ values different branch attributes
# Columns: [safety_bonus, tempt_bonus, texture_novelty, shortcut_bonus]
PREF_REWARD = {
    "safe":     np.array([2.0, -1.0,  0.0,  0.0]),
    "risky":    np.array([-0.5, 0.5,  0.0,  0.0]),
    "shiny":    np.array([0.0,  3.0,  0.0,  0.0]),
    "shortcut": np.array([0.0,  0.0,  0.0,  2.0]),
    "neutral":  np.array([0.3,  0.0,  0.0,  0.0]),
}

PREFERENCE_TYPES = list(PREF_REWARD.keys())


@dataclass
class AgentPolicyParams:
    """Parameters for bounded-rational agent."""
    beta: float = 4.0          # softmax temperature (higher = more rational)
    epsilon: float = 0.1       # lapse rate (random exploration)
    lambda_theta: float = 1.0  # preference weight


@dataclass
class BranchAttributes:
    """Observable attributes of a branch for utility computation."""
    safety_score: float = 0.5
    temptation_score: float = 0.0
    texture_novelty: float = 0.0
    shortcut_bonus: float = 0.0
    risk_penalty: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.safety_score,
            self.temptation_score,
            self.texture_novelty,
            self.shortcut_bonus,
        ])


def compute_agent_utility(
    branch_attrs: BranchAttributes,
    theta: str,
    params: AgentPolicyParams,
) -> float:
    """Agent's internal utility for a branch given preference θ.

    U(π|θ) = R_goal(π) + λ_θ · R_pref(π; θ) - J_risk(π)
    """
    r_goal = branch_attrs.safety_score  # safer = better goal completion
    r_pref = float(np.dot(PREF_REWARD[theta], branch_attrs.to_array()))
    j_risk = branch_attrs.risk_penalty

    return r_goal + params.lambda_theta * r_pref - j_risk


def compute_choice_probs(
    branches: list[BranchAttributes],
    theta: str,
    params: AgentPolicyParams,
) -> np.ndarray:
    """Softmax + lapse mixture choice probabilities.

    P_mix(π|s,θ) = (1-ε) · softmax(β·U) + ε · 1/|Π|
    """
    utilities = np.array([
        compute_agent_utility(b, theta, params) for b in branches
    ])

    # Softmax
    scaled = params.beta * utilities
    scaled -= np.max(scaled)  # numerical stability
    exp_u = np.exp(scaled)
    softmax_probs = exp_u / (exp_u.sum() + 1e-10)

    # Lapse mixture
    n = len(branches)
    uniform = np.ones(n) / n
    mixed = (1 - params.epsilon) * softmax_probs + params.epsilon * uniform

    return mixed


def sample_branch_choice(
    branches: list[BranchAttributes],
    theta: str,
    params: AgentPolicyParams,
    rng: np.random.Generator,
) -> int:
    """Sample a branch index from the bounded-rational policy."""
    probs = compute_choice_probs(branches, theta, params)
    return int(rng.choice(len(branches), p=probs))


def compute_likelihood(
    chosen_idx: int,
    branches: list[BranchAttributes],
    theta: str,
    params: AgentPolicyParams,
) -> float:
    """P(chose branch_idx | θ, params) — for posterior update."""
    probs = compute_choice_probs(branches, theta, params)
    return float(probs[chosen_idx])
