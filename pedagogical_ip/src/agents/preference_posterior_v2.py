"""J2 — Preference Posterior v2: behavior-driven Bayesian inference.

Uses actual agent choice likelihoods from stochastic_agent_policy
to update q(θ). Includes forgetting/diffusion to prevent lock-in.

q_t(θ) ∝ q_{t-1}(θ) · P(a_t | s_t, θ)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .stochastic_agent_policy import (
    PREFERENCE_TYPES, compute_likelihood, BranchAttributes, AgentPolicyParams,
)

N_PREF = len(PREFERENCE_TYPES)


@dataclass
class PreferencePosteriorV2:
    """Behavior-driven Bayesian preference posterior."""
    log_probs: np.ndarray = field(
        default_factory=lambda: np.zeros(N_PREF))
    observation_count: int = 0
    forgetting_rate: float = 0.02   # mild diffusion toward uniform

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
        return float(np.log(N_PREF))

    @property
    def predicted_type(self) -> str:
        return PREFERENCE_TYPES[int(np.argmax(self.probs))]

    @property
    def predicted_prob(self) -> float:
        return float(np.max(self.probs))

    def update_from_choice(
        self,
        chosen_idx: int,
        branches: list[BranchAttributes],
        params: AgentPolicyParams,
    ):
        """Update posterior from observed agent branch choice.

        q_t(θ) ∝ q_{t-1}(θ) · P(a_t | s_t, θ)
        Then apply forgetting/diffusion.
        """
        for i, ptype in enumerate(PREFERENCE_TYPES):
            likelihood = compute_likelihood(chosen_idx, branches, ptype, params)
            self.log_probs[i] += np.log(likelihood + 1e-10)

        # Numerical stability
        self.log_probs -= np.mean(self.log_probs)

        # Forgetting: mild diffusion toward uniform
        if self.forgetting_rate > 0:
            p = self.probs
            p_diffused = (1 - self.forgetting_rate) * p + \
                         self.forgetting_rate * np.ones(N_PREF) / N_PREF
            self.log_probs = np.log(p_diffused + 1e-10)
            self.log_probs -= np.mean(self.log_probs)

        self.observation_count += 1

    def posterior_predictive_variance(
        self,
        branches: list[BranchAttributes],
        params: AgentPolicyParams,
    ) -> float:
        """Var_θ~q[P(π|s,θ)] — how much choice distribution varies across θ.

        High variance = observing agent's choice would be very informative.
        """
        q = self.probs
        # For each θ, get choice prob for branch 0
        p_per_theta = np.array([
            compute_likelihood(0, branches, ptype, params)
            for ptype in PREFERENCE_TYPES
        ])
        # Weighted variance under posterior
        mean_p = float(np.dot(q, p_per_theta))
        var_p = float(np.dot(q, (p_per_theta - mean_p) ** 2))
        return var_p

    def expected_info_gain_from_observation(
        self,
        branches: list[BranchAttributes],
        params: AgentPolicyParams,
    ) -> float:
        """Expected H(q_t) - H(q_{t+1}) from observing next agent choice.

        Approximated by posterior predictive variance.
        """
        return self.posterior_predictive_variance(branches, params)

    def to_dict(self) -> dict:
        p = self.probs
        return {
            "probs": {PREFERENCE_TYPES[i]: round(float(p[i]), 4)
                      for i in range(N_PREF)},
            "predicted": self.predicted_type,
            "predicted_prob": round(self.predicted_prob, 4),
            "entropy": round(self.entropy, 4),
            "n_obs": self.observation_count,
        }
