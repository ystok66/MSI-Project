"""Mastery Model: Beta-Bernoulli per-mechanism mastery tracker.

u_k = a_k / (a_k + b_k), updated per probe outcome.
Smoothed with forgetting factor λ for curriculum-level planning.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

PROBE_NAMES = ["RC", "TR", "EP", "VA", "IA"]

# Which probes indicate "higher is better" vs "lower is better"
HIGHER_IS_BETTER = {"RC": True, "TR": True, "EP": True, "VA": True, "IA": False}


@dataclass
class MasteryModel:
    """Beta-Bernoulli mastery tracker over 5 mechanism dimensions."""
    alpha: dict = field(default_factory=lambda: {p: 1.0 for p in PROBE_NAMES})
    beta_: dict = field(default_factory=lambda: {p: 1.0 for p in PROBE_NAMES})
    decay: float = 0.95  # forgetting factor
    history: list = field(default_factory=list)

    def mastery(self) -> dict:
        """Current mastery estimates u ∈ [0,1]^5."""
        return {p: round(self.alpha[p] / (self.alpha[p] + self.beta_[p]), 4)
                for p in PROBE_NAMES}

    def uncertainty(self) -> dict:
        """Posterior uncertainty (inverse of total counts)."""
        return {p: round(1.0 / (self.alpha[p] + self.beta_[p]), 4)
                for p in PROBE_NAMES}

    def update(self, probe_outcomes: dict):
        """Update from probe outcomes.

        For "higher is better" probes: outcome > 0.5 → success
        For IA (lower is better): outcome < 0.5 → success
        """
        for p in PROBE_NAMES:
            if p not in probe_outcomes:
                continue
            v = probe_outcomes[p]
            if HIGHER_IS_BETTER[p]:
                success = 1.0 if v > 0.5 else 0.0
            else:
                success = 1.0 if v < 0.5 else 0.0

            self.alpha[p] = self.decay * self.alpha[p] + success
            self.beta_[p] = self.decay * self.beta_[p] + (1.0 - success)

        self.history.append(dict(self.mastery()))

    def entropy(self) -> float:
        """Total uncertainty across all dimensions."""
        u = self.uncertainty()
        return round(sum(u.values()), 4)

    def readiness(self, weights=None) -> float:
        """Overall readiness score."""
        if weights is None:
            weights = {"RC": 1.0, "TR": 1.2, "EP": 2.5, "VA": 1.5, "IA": 2.5}
        m = self.mastery()
        return round(sum(weights[p] * m[p] for p in PROBE_NAMES), 4)

    def progress_since(self, t: int) -> dict:
        """Mastery gain since step t."""
        if t >= len(self.history) or not self.history:
            return {p: 0.0 for p in PROBE_NAMES}
        old = self.history[t]
        cur = self.mastery()
        return {p: round(cur[p] - old.get(p, 0.5), 4) for p in PROBE_NAMES}
