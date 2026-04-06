"""Lesson-Response Model v2: Gain + Harm posteriors.

For each lesson ℓ, bucket b:
  - Gain posterior: p^gain_{ℓ,k,b} ~ Beta(α,β) per mechanism k
  - Harm posteriors: p^ν_{ℓ,b}, p^γg_{ℓ,b}, p^otr_{ℓ,b} ~ Beta(α,β)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

PROBE_NAMES = ["RC", "TR", "EP", "VA", "IA"]
HARM_DIMS = ["nu", "gamma_gen", "otr"]


def mastery_bucket(u: dict) -> str:
    """Discretize mastery into L/H per dimension."""
    return "".join("H" if u.get(p, 0.5) > 0.55 else "L" for p in PROBE_NAMES)


@dataclass
class LessonResponseModelV2:
    """Gain + Harm posteriors per (lesson, bucket)."""
    decay: float = 0.92
    delta_threshold: float = 0.02

    # Gain: {(lesson, probe, bucket): [α, β]}
    gain_posteriors: dict = field(default_factory=lambda: defaultdict(lambda: [1.0, 1.0]))
    # Harm: {(lesson, harm_dim, bucket): [α, β]}
    harm_posteriors: dict = field(default_factory=lambda: defaultdict(lambda: [1.0, 1.0]))

    # ─── Gain ───

    def gain_expected(self, lesson_name: str, probe: str, bucket: str) -> float:
        a, b = self.gain_posteriors[(lesson_name, probe, bucket)]
        return a / (a + b)

    def gain_variance(self, lesson_name: str, probe: str, bucket: str) -> float:
        a, b = self.gain_posteriors[(lesson_name, probe, bucket)]
        t = a + b
        return (a * b) / (t ** 2 * (t + 1))

    def update_gain(self, lesson_name: str, bucket: str,
                    mastery_before: dict, mastery_after: dict):
        for p in PROBE_NAMES:
            delta = mastery_after.get(p, 0.5) - mastery_before.get(p, 0.5)
            success = 1.0 if (delta < -self.delta_threshold if p == "IA" else delta > self.delta_threshold) else 0.0
            k = (lesson_name, p, bucket)
            a, b = self.gain_posteriors[k]
            self.gain_posteriors[k] = [self.decay * a + success, self.decay * b + (1 - success)]

    # ─── Harm ───

    def harm_expected(self, lesson_name: str, harm_dim: str, bucket: str) -> float:
        a, b = self.harm_posteriors[(lesson_name, harm_dim, bucket)]
        return a / (a + b)

    def harm_variance(self, lesson_name: str, harm_dim: str, bucket: str) -> float:
        a, b = self.harm_posteriors[(lesson_name, harm_dim, bucket)]
        t = a + b
        return (a * b) / (t ** 2 * (t + 1))

    def update_harm(self, lesson_name: str, bucket: str,
                    nu_before: float, nu_after: float,
                    gg_before: float, gg_after: float,
                    otr_before: float, otr_after: float,
                    nu_max: float = 0.28, gg_max: float = 0.12, otr_max: float = 0.5):
        events = {
            "nu": 1.0 if nu_after > nu_max or (nu_after - nu_before) > 0.03 else 0.0,
            "gamma_gen": 1.0 if gg_after > gg_max or (gg_after - gg_before) > 0.02 else 0.0,
            "otr": 1.0 if otr_after > otr_max or (otr_after - otr_before) > 0.1 else 0.0,
        }
        for hd in HARM_DIMS:
            k = (lesson_name, hd, bucket)
            a, b = self.harm_posteriors[k]
            self.harm_posteriors[k] = [self.decay * a + events[hd], self.decay * b + (1 - events[hd])]

    # ─── Summary ───

    def total_harm(self, lesson_name: str, bucket: str,
                   weights: dict = None) -> float:
        if weights is None:
            weights = {"nu": 2.0, "gamma_gen": 2.5, "otr": 1.5}
        return sum(weights[hd] * self.harm_expected(lesson_name, hd, bucket) for hd in HARM_DIMS)

    def n_updated(self) -> int:
        count = 0
        for (a, b) in self.gain_posteriors.values():
            if a + b > 2.05: count += 1
        for (a, b) in self.harm_posteriors.values():
            if a + b > 2.05: count += 1
        return count
