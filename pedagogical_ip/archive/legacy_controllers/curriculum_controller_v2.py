"""CCT-v2: Closed-Loop Curriculum Controller.

macro selects lesson → AEG generates episode → micro acts with dose budget.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .lesson_library import Lesson, LESSON_CATALOG
from .adaptive_episode_generator import EpisodeParams
from ..agents.internalization_state_v3 import FactoredInternalizationState
from ..agents.trainable_bridge import TrainableBridge

PROBE_NAMES = ["RC", "TR", "EP", "VA", "IA"]
W_DEFICIT = np.array([1.0, 1.2, 2.5, 1.5, 2.5])

PROBE_TARGETS = {
    "safe":  {"RC": 0.70, "TR": 0.65, "EP": 0.55, "VA": 0.70, "IA": 0.25},
    "shiny": {"RC": 0.70, "TR": 0.70, "EP": 0.50, "VA": 0.70, "IA": 0.25},
}


@dataclass
class CurriculumControllerV2:
    """Closed-loop curriculum controller with dose budget enforcement."""
    bridge: TrainableBridge = None
    lambda_over: float = 2.0
    lambda_cost: float = 0.5
    lambda_div: float = 1.0
    theta: str = "safe"

    history: List[str] = field(default_factory=list)
    lesson_counts: dict = field(default_factory=dict)
    realized_subtypes: list = field(default_factory=list)

    def __post_init__(self):
        if self.bridge is None:
            self.bridge = TrainableBridge()

    def _deficit(self, m: FactoredInternalizationState) -> np.ndarray:
        preds = self.bridge.predict(m)
        targets = PROBE_TARGETS.get(self.theta, PROBE_TARGETS["safe"])
        d = np.zeros(5)
        for i, pn in enumerate(PROBE_NAMES):
            if pn == "IA":
                d[i] = max(preds[pn] - targets[pn], 0.0)
            else:
                d[i] = max(targets[pn] - preds[pn], 0.0)
        return d

    def _overteach_risk(self, lesson: Lesson, m: FactoredInternalizationState) -> float:
        r = 0.0
        if lesson.delta_state[2] > 0:
            r += 2.0 * lesson.delta_state[2] * (1 + m.nu)
        if lesson.delta_state[4] > 0:
            r += 2.5 * lesson.delta_state[4] * (1 + m.gamma_gen)
        return float(r)

    def _diversity_bonus(self, lesson: Lesson) -> float:
        recent = self.history[-5:] if len(self.history) >= 5 else self.history
        if not recent:
            return 0.0
        n_recent = sum(1 for h in recent if h == lesson.name)
        return float(0.1 * (1.0 - n_recent / max(len(recent), 1)))

    def select_lesson(self, m: FactoredInternalizationState,
                      candidates: Optional[List[Lesson]] = None) -> tuple:
        if candidates is None:
            candidates = LESSON_CATALOG
        d = self._deficit(m)
        best_lesson = candidates[0]
        best_Q = -1e9

        for lesson in candidates:
            s_adj = lesson.delta_probe.copy()
            s_adj[4] = -s_adj[4]
            coverage = float(np.sum(d * W_DEFICIT * s_adj))
            r_over = self._overteach_risk(lesson, m)
            r_div = self._diversity_bonus(lesson)
            Q = coverage - self.lambda_over * r_over - self.lambda_cost * lesson.cost + self.lambda_div * r_div
            if Q > best_Q:
                best_Q = Q; best_lesson = lesson

        self.history.append(best_lesson.name)
        self.lesson_counts[best_lesson.name] = self.lesson_counts.get(best_lesson.name, 0) + 1

        return best_lesson, round(best_Q, 4), {
            "deficit": {PROBE_NAMES[i]: round(d[i], 4) for i in range(5)},
        }

    def record_realization(self, ep_params: EpisodeParams):
        self.realized_subtypes.append(ep_params.subtype)

    def curriculum_summary(self) -> dict:
        return {
            "total": len(self.history),
            "unique": len(set(self.history)),
            "counts": dict(self.lesson_counts),
            "sequence": list(self.history),
            "realized": list(self.realized_subtypes),
        }


@dataclass
class DoseBudgetTracker:
    """Track and enforce dose budget per episode."""
    budget: float = 1.0
    hard_limit: int = 3
    warns_used: int = 0
    dose_spent: float = 0.0

    def reset(self, ep_params: EpisodeParams):
        self.budget = ep_params.dose_budget
        self.hard_limit = ep_params.hard_limit
        self.warns_used = 0
        self.dose_spent = 0.0

    def feasible_doses(self) -> list:
        doses = [0.0]
        if self.budget - self.dose_spent >= 0.5:
            doses.append(0.5)
        if self.budget - self.dose_spent >= 1.0 and self.warns_used < self.hard_limit:
            doses.append(1.0)
        return doses

    def consume(self, dose: float):
        self.dose_spent += dose
        if dose >= 1.0:
            self.warns_used += 1
