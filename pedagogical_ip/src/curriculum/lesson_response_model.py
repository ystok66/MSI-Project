"""Lesson-Response Model: Beta-Bernoulli posterior per (lesson, mechanism, bucket).

For each lesson ℓ and mechanism k, maintains:
  p_{ℓ,k,b} ~ Beta(α, β)
where b = learner bucket based on current mastery level.

Provides:
  - Expected gain: E[p] = α / (α + β)
  - Uncertainty: Var[p] = αβ / ((α+β)²(α+β+1))
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

PROBE_NAMES = ["RC", "TR", "EP", "VA", "IA"]


def mastery_bucket(u: dict) -> str:
    """Discretize mastery into bucket for response model.

    3 levels × 5 dims → 3^5 = 243 possible buckets,
    but we use a coarser 2-level discretization: low/high per dim.
    """
    parts = []
    for p in PROBE_NAMES:
        v = u.get(p, 0.5)
        parts.append("H" if v > 0.55 else "L")
    return "".join(parts)


@dataclass
class LessonResponseModel:
    """Beta-Bernoulli lesson-response posteriors."""
    decay: float = 0.92
    delta_threshold: float = 0.02  # mastery gain > this = "success"

    # {(lesson_name, probe_name, bucket): (alpha, beta)}
    posteriors: dict = field(default_factory=lambda: defaultdict(lambda: [1.0, 1.0]))

    def _key(self, lesson_name: str, probe: str, bucket: str) -> tuple:
        return (lesson_name, probe, bucket)

    def expected_gain(self, lesson_name: str, probe: str, bucket: str) -> float:
        """E[p_{ℓ,k,b}] = α / (α + β)."""
        k = self._key(lesson_name, probe, bucket)
        a, b = self.posteriors[k]
        return a / (a + b)

    def variance(self, lesson_name: str, probe: str, bucket: str) -> float:
        """Var[p_{ℓ,k,b}]."""
        k = self._key(lesson_name, probe, bucket)
        a, b = self.posteriors[k]
        total = a + b
        return (a * b) / (total ** 2 * (total + 1))

    def expected_gains(self, lesson_name: str, bucket: str) -> dict:
        """Expected gain for all probes."""
        return {p: round(self.expected_gain(lesson_name, p, bucket), 4)
                for p in PROBE_NAMES}

    def uncertainties(self, lesson_name: str, bucket: str) -> dict:
        """Variance for all probes."""
        return {p: round(self.variance(lesson_name, p, bucket), 6)
                for p in PROBE_NAMES}

    def update(self, lesson_name: str, bucket: str,
               mastery_before: dict, mastery_after: dict):
        """Update posteriors after a lesson.

        success = mastery improved beyond delta_threshold.
        """
        for p in PROBE_NAMES:
            delta = mastery_after.get(p, 0.5) - mastery_before.get(p, 0.5)
            # For IA, improvement means decrease
            if p == "IA":
                success = 1.0 if delta < -self.delta_threshold else 0.0
            else:
                success = 1.0 if delta > self.delta_threshold else 0.0

            k = self._key(lesson_name, p, bucket)
            a, b = self.posteriors[k]
            self.posteriors[k] = [
                self.decay * a + success,
                self.decay * b + (1.0 - success),
            ]

    def total_uncertainty(self, lesson_name: str, bucket: str) -> float:
        """Sum of variances across probes."""
        return sum(self.variance(lesson_name, p, bucket) for p in PROBE_NAMES)

    def info_value(self, bucket: str) -> float:
        """Total uncertainty across all lessons for this bucket."""
        total = 0.0
        seen = set()
        for (ln, p, b), (a, bt) in self.posteriors.items():
            if b == bucket and (ln, p) not in seen:
                seen.add((ln, p))
                total += self.variance(ln, p, b)
        return total
