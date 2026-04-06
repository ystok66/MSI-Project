"""Lesson-Response Model v3: Hierarchical Empirical Bayes.

Upgrade over v2: local posteriors are shrunk toward pooled estimates,
so sparse buckets borrow strength from richer buckets.

  p̃_gain = (n/(n+λ))·p̂_local + (λ/(n+λ))·p̂_pooled

Same for harm posteriors.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

PROBE_NAMES = ["RC", "TR", "EP", "VA", "IA"]
HARM_DIMS = ["nu", "gamma_gen", "otr"]


def mastery_bucket(u: dict) -> str:
    return "".join("H" if u.get(p, 0.5) > 0.55 else "L" for p in PROBE_NAMES)


@dataclass
class LessonResponseModelV3:
    """Hierarchical empirical Bayes lesson-response posteriors."""
    decay: float = 0.92
    delta_threshold: float = 0.02
    shrinkage_gain: float = 3.0   # λ_g: how much to pull toward pooled
    shrinkage_harm: float = 2.0   # λ_h

    # Local posteriors: {(lesson, probe, bucket): [α, β]}
    gain_local: dict = field(default_factory=lambda: defaultdict(lambda: [1.0, 1.0]))
    # Harm: {(lesson, harm_dim, bucket): [α, β]}
    harm_local: dict = field(default_factory=lambda: defaultdict(lambda: [1.0, 1.0]))

    # ─── Gain: hierarchical ───

    def _pooled_gain(self, lesson_name: str, probe: str) -> float:
        """Pooled estimate across all buckets for this (lesson, probe)."""
        total_a, total_b = 0.0, 0.0
        for (ln, p, b), (a, bt) in self.gain_local.items():
            if ln == lesson_name and p == probe:
                total_a += a; total_b += bt
        if total_a + total_b < 0.01:
            return 0.5  # prior
        return total_a / (total_a + total_b)

    def gain_expected(self, lesson_name: str, probe: str, bucket: str) -> float:
        """Shrunk gain estimate: local → pooled."""
        a, b = self.gain_local[(lesson_name, probe, bucket)]
        n = a + b
        p_local = a / n if n > 0.01 else 0.5
        p_pooled = self._pooled_gain(lesson_name, probe)
        lam = self.shrinkage_gain
        return float((n / (n + lam)) * p_local + (lam / (n + lam)) * p_pooled)

    def gain_variance(self, lesson_name: str, probe: str, bucket: str) -> float:
        """Shrunk variance: scaled by effective sample size."""
        a, b = self.gain_local[(lesson_name, probe, bucket)]
        n_eff = a + b + self.shrinkage_gain
        p = self.gain_expected(lesson_name, probe, bucket)
        return float(p * (1 - p) / (n_eff + 1))

    def update_gain(self, lesson_name: str, bucket: str,
                    mastery_before: dict, mastery_after: dict):
        for p in PROBE_NAMES:
            delta = mastery_after.get(p, 0.5) - mastery_before.get(p, 0.5)
            success = 1.0 if (delta < -self.delta_threshold if p == "IA" else delta > self.delta_threshold) else 0.0
            k = (lesson_name, p, bucket)
            a, b = self.gain_local[k]
            self.gain_local[k] = [self.decay * a + success, self.decay * b + (1 - success)]

    # ─── Harm: hierarchical ───

    def _pooled_harm(self, lesson_name: str, harm_dim: str) -> float:
        total_a, total_b = 0.0, 0.0
        for (ln, hd, b), (a, bt) in self.harm_local.items():
            if ln == lesson_name and hd == harm_dim:
                total_a += a; total_b += bt
        if total_a + total_b < 0.01:
            return 0.5
        return total_a / (total_a + total_b)

    def harm_expected(self, lesson_name: str, harm_dim: str, bucket: str) -> float:
        a, b = self.harm_local[(lesson_name, harm_dim, bucket)]
        n = a + b
        p_local = a / n if n > 0.01 else 0.5
        p_pooled = self._pooled_harm(lesson_name, harm_dim)
        lam = self.shrinkage_harm
        return float((n / (n + lam)) * p_local + (lam / (n + lam)) * p_pooled)

    def harm_variance(self, lesson_name: str, harm_dim: str, bucket: str) -> float:
        a, b = self.harm_local[(lesson_name, harm_dim, bucket)]
        n_eff = a + b + self.shrinkage_harm
        p = self.harm_expected(lesson_name, harm_dim, bucket)
        return float(p * (1 - p) / (n_eff + 1))

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
            a, b = self.harm_local[k]
            self.harm_local[k] = [self.decay * a + events[hd], self.decay * b + (1 - events[hd])]

    def total_harm(self, lesson_name: str, bucket: str,
                   weights: dict = None) -> float:
        if weights is None:
            weights = {"nu": 2.0, "gamma_gen": 2.5, "otr": 1.5}
        return sum(weights[hd] * self.harm_expected(lesson_name, hd, bucket) for hd in HARM_DIMS)

    def n_updated(self) -> int:
        count = 0
        for (a, b) in self.gain_local.values():
            if a + b > 2.05: count += 1
        for (a, b) in self.harm_local.values():
            if a + b > 2.05: count += 1
        return count
