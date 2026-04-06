"""Curriculum Controller v1: macro-level lesson selector.

Q_macro(ℓ) = d·W·ŝ(ℓ) − λ_over·r_over(ℓ) − λ_cost·c(ℓ) + λ_div·r_div(ℓ|h)

d = mechanism deficit vector (how far from target behavior)
ŝ(ℓ) = lesson signature (expected Δ probes)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .lesson_library import Lesson, LESSON_CATALOG
from ..agents.internalization_state_v3 import FactoredInternalizationState
from ..agents.trainable_bridge import TrainableBridge


PROBE_NAMES = ["RC", "TR", "EP", "VA", "IA"]
# Weight matrix for deficit coverage
W_DEFICIT = np.array([1.0, 1.2, 2.5, 1.5, 2.5])

# Target zones (from FICA-validated empirical ranges)
PROBE_TARGETS = {
    "safe":  {"RC": 0.70, "TR": 0.65, "EP": 0.55, "VA": 0.70, "IA": 0.25},
    "shiny": {"RC": 0.70, "TR": 0.70, "EP": 0.50, "VA": 0.70, "IA": 0.25},
}

# For IA (lower is better), deficit is current - target (positive means too high)
IA_LOWER_IS_BETTER = True


@dataclass
class CurriculumControllerV1:
    """One-step myopic curriculum selector."""
    bridge: TrainableBridge = None
    lambda_over: float = 2.0
    lambda_cost: float = 0.5
    lambda_div: float = 1.0
    theta: str = "safe"

    history: List[str] = field(default_factory=list)
    lesson_counts: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.bridge is None:
            self.bridge = TrainableBridge()

    def _deficit(self, m: FactoredInternalizationState) -> np.ndarray:
        """d_t = [d_RC, d_TR, d_EP, d_VA, d_IA]."""
        preds = self.bridge.predict(m)
        targets = PROBE_TARGETS.get(self.theta, PROBE_TARGETS["safe"])
        d = np.zeros(5)
        for i, pn in enumerate(PROBE_NAMES):
            if pn == "IA":
                # Lower is better: deficit = current - target (positive means too high)
                d[i] = max(preds[pn] - targets[pn], 0.0)
            else:
                # Higher is better: deficit = target - current (positive means too low)
                d[i] = max(targets[pn] - preds[pn], 0.0)
        return d

    def _overteach_risk(self, lesson: Lesson, m: FactoredInternalizationState) -> float:
        """Estimate overteaching risk from this lesson."""
        r = 0.0
        # ν push
        if lesson.delta_state[2] > 0:
            r += 2.0 * lesson.delta_state[2] * (1 + m.nu)
        # γ_gen push
        if lesson.delta_state[4] > 0:
            r += 2.5 * lesson.delta_state[4] * (1 + m.gamma_gen)
        return float(r)

    def _diversity_bonus(self, lesson: Lesson) -> float:
        """Anti-collapse: reward lessons not recently used."""
        recent = self.history[-5:] if len(self.history) >= 5 else self.history
        if not recent:
            return 0.0
        n_recent = sum(1 for h in recent if h == lesson.name)
        # Negative if overused, positive if fresh
        return float(0.1 * (1.0 - n_recent / max(len(recent), 1)))

    def select_lesson(self, m: FactoredInternalizationState,
                      candidates: Optional[List[Lesson]] = None) -> tuple:
        """Select best lesson for current learner state.

        Returns (lesson, Q_score, info_dict).
        """
        if candidates is None:
            candidates = LESSON_CATALOG

        d = self._deficit(m)
        best_lesson = candidates[0]
        best_Q = -1e9
        scores = {}

        for lesson in candidates:
            # Deficit coverage: d · W · ŝ(ℓ)
            # ŝ is the probe delta signature
            s = lesson.delta_probe
            # For IA, negative delta is good (reduces IA), so negate
            s_adj = s.copy()
            s_adj[4] = -s_adj[4]  # IA: lower delta = better coverage
            coverage = float(np.sum(d * W_DEFICIT * s_adj))

            # Overteaching risk
            r_over = self._overteach_risk(lesson, m)

            # Diversity
            r_div = self._diversity_bonus(lesson)

            # Cost
            c = lesson.cost

            Q = (coverage
                 - self.lambda_over * r_over
                 - self.lambda_cost * c
                 + self.lambda_div * r_div)

            scores[lesson.name] = round(Q, 4)
            if Q > best_Q:
                best_Q = Q
                best_lesson = lesson

        self.history.append(best_lesson.name)
        self.lesson_counts[best_lesson.name] = self.lesson_counts.get(best_lesson.name, 0) + 1

        return best_lesson, round(best_Q, 4), {
            "deficit": {PROBE_NAMES[i]: round(d[i], 4) for i in range(5)},
            "scores": scores,
        }

    def curriculum_summary(self) -> dict:
        return {
            "total_lessons": len(self.history),
            "unique_lessons": len(set(self.history)),
            "counts": dict(self.lesson_counts),
            "sequence": list(self.history),
        }
