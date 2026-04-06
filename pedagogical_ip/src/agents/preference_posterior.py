"""I2 — Bayesian Preference Posterior for Hidden Preferences.

Maintains robot's belief q(θ) over agent's latent preference type.
Updates via simple likelihood based on observed branch choices.

Preference types: {safe, risky, shiny, shortcut, neutral}
Each defines a bias toward different branch attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


PREFERENCE_TYPES = ["safe", "risky", "shiny", "shortcut", "neutral"]
N_PREF = len(PREFERENCE_TYPES)

# Likelihood: P(choose branch_id | θ, branch attributes)
# Each preference type biases toward a score
PREF_WEIGHTS = {
    "safe":     np.array([1.0, -1.0, 0.0, 0.0]),   # prefers low risk
    "risky":    np.array([-0.5, 0.5, 0.0, 0.0]),    # drawn to risky
    "shiny":    np.array([0.0, 0.0, 1.0, 0.0]),     # drawn to high-salience cue
    "shortcut": np.array([0.0, 0.0, 0.0, 1.0]),     # drawn to shorter paths
    "neutral":  np.array([0.0, 0.0, 0.0, 0.0]),     # no preference
}


@dataclass
class PreferencePosterior:
    """Robot's belief over agent's latent preference type."""
    log_probs: np.ndarray = field(
        default_factory=lambda: np.zeros(N_PREF))  # log q(θ)
    observation_count: int = 0
    temperature: float = 2.0  # softmax temperature for likelihood

    @property
    def probs(self) -> np.ndarray:
        """Normalized probabilities."""
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

    def update(self, branch_attrs: np.ndarray, chose_branch: bool):
        """Update posterior given observed branch choice.

        branch_attrs: [safety_score, risk_score, salience, shortcut_bonus]
        chose_branch: True if agent chose this branch
        """
        for i, ptype in enumerate(PREFERENCE_TYPES):
            w = PREF_WEIGHTS[ptype]
            # Score how much θ prefers this branch
            affinity = float(np.dot(w, branch_attrs))
            if chose_branch:
                self.log_probs[i] += affinity / self.temperature
            else:
                self.log_probs[i] -= affinity / self.temperature

        # Numerical stability
        self.log_probs -= np.mean(self.log_probs)
        self.observation_count += 1

    def entropy_reduction_from_warn(
        self,
        branch_attrs_a: np.ndarray,
        branch_attrs_b: np.ndarray,
    ) -> float:
        """Expected entropy reduction if tutor provides warning.

        Warning reveals risk information → sharpens likelihood for
        preference types that care about risk.
        """
        # Warning makes safe/risky distinction clearer
        # This helps disambiguate safe vs risky preference types
        h_pre = self.entropy

        # Simulate: if agent sees warning and picks safe branch
        pp_safe = PreferencePosterior(log_probs=self.log_probs.copy())
        pp_safe.update(branch_attrs_a, chose_branch=True)

        # Simulate: if agent sees warning and still picks risky branch
        pp_risky = PreferencePosterior(log_probs=self.log_probs.copy())
        pp_risky.update(branch_attrs_b, chose_branch=True)

        # Expected posterior entropy (equal weight since we don't know yet)
        h_post = 0.5 * pp_safe.entropy + 0.5 * pp_risky.entropy
        return max(h_pre - h_post, 0)

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
