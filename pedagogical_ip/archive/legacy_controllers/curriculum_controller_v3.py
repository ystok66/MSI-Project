"""CCT-v3: Mastery-Aware Closed-Loop Curriculum Controller.

Q_macro(ℓ) = d·W·Δu(ℓ) − λ_over·r_over − λ_fid·r_fid − λ_bud·r_bud − λ_rep·r_rep
+ EVAL and STOP as macro actions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .lesson_library_v2 import LessonV2, LESSON_CATALOG_V2, PROBE_NAMES
from .mastery_model import MasteryModel
from .adaptive_episode_generator import EpisodeParams
from ..agents.internalization_state_v3 import FactoredInternalizationState
from ..agents.trainable_bridge import TrainableBridge

W_DEFICIT = np.array([1.0, 1.2, 2.5, 1.5, 2.5])

PROBE_TARGETS = {
    "safe":  {"RC": 0.70, "TR": 0.65, "EP": 0.55, "VA": 0.70, "IA": 0.25},
    "shiny": {"RC": 0.70, "TR": 0.70, "EP": 0.50, "VA": 0.70, "IA": 0.25},
}


@dataclass
class CurriculumControllerV3:
    """Mastery-aware curriculum controller with EVAL/STOP."""
    bridge: TrainableBridge = None
    mastery: MasteryModel = None
    theta: str = "safe"

    lambda_over: float = 2.5
    lambda_fid: float = 1.0
    lambda_bud: float = 1.5
    lambda_rep: float = 0.8
    lambda_eval: float = 0.5
    lambda_stop: float = 0.8

    # Budget
    total_budget: float = 8.0
    spent_budget: float = 0.0

    # History
    history: List[str] = field(default_factory=list)
    lesson_counts: dict = field(default_factory=dict)
    realized_subtypes: list = field(default_factory=list)
    stopped: bool = False
    eval_count: int = 0

    # Thresholds
    nu_max: float = 0.3
    gg_max: float = 0.15
    readiness_threshold: float = 5.5

    def __post_init__(self):
        if self.bridge is None:
            self.bridge = TrainableBridge()
        if self.mastery is None:
            self.mastery = MasteryModel()

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

    def _overteach_risk(self, lesson: LessonV2, m: FactoredInternalizationState) -> float:
        r = 0.0
        nu_next = m.nu + lesson.nu_push
        gg_next = m.gamma_gen + lesson.gg_push
        r += 2.0 * max(nu_next - self.nu_max, 0) ** 2
        r += 2.5 * max(gg_next - self.gg_max, 0) ** 2
        return float(r)

    def _fidelity_penalty(self, lesson: LessonV2) -> float:
        # Empirical: adaptive lessons have ~0.10 fidelity loss
        base_lf = 0.90 if len(self.history) > 3 else 0.95
        return float(1.0 - base_lf)

    def _budget_risk(self, lesson: LessonV2) -> float:
        remaining = self.total_budget - self.spent_budget
        if remaining <= 0.1:
            return 2.0
        ratio = lesson.cost / (remaining + 0.01)
        return float(max(ratio - 0.3, 0) ** 2)

    def _repetition_penalty(self, lesson: LessonV2) -> float:
        recent = self.history[-5:] if len(self.history) >= 5 else self.history
        if not recent:
            return 0.0
        n_recent = sum(1 for h in recent if h == lesson.name)
        return float(n_recent / max(len(recent), 1))

    def _eval_value(self, m: FactoredInternalizationState) -> float:
        """Q(EVAL): value of probing before teaching."""
        ent = self.mastery.entropy()
        eval_cost = 0.8
        return float(self.lambda_eval * ent - eval_cost)

    def _stop_value(self, m: FactoredInternalizationState) -> float:
        """Q(STOP): value of stopping the curriculum."""
        n_teach = sum(1 for h in self.history if h != "EVAL")
        if n_teach < 5:
            return -10.0  # Never stop before 5 teach episodes
        d = self._deficit(m)
        deficit_mag = float(np.sum(d))
        ready = self.mastery.readiness()
        stop_bonus = ready - 1.5 * m.nu - 2.0 * m.gamma_gen
        return float(self.lambda_stop * stop_bonus - 4.0 * deficit_mag)

    def select_action(self, m: FactoredInternalizationState,
                      candidates: Optional[List[LessonV2]] = None) -> tuple:
        """Select best macro action: lesson, EVAL, or STOP.

        Returns (action_type, lesson_or_None, Q_score, info).
        action_type ∈ {"TEACH", "EVAL", "STOP"}
        """
        if self.stopped:
            return "STOP", None, 0.0, {"reason": "already_stopped"}

        if candidates is None:
            candidates = LESSON_CATALOG_V2

        d = self._deficit(m)
        u = self.mastery.mastery()

        # Score all lessons
        best_lesson = None
        best_Q_teach = -1e9
        for lesson in candidates:
            eg = lesson.effective_gain(u)
            coverage = float(np.sum(d * W_DEFICIT * eg))
            r_over = self._overteach_risk(lesson, m)
            r_fid = self._fidelity_penalty(lesson)
            r_bud = self._budget_risk(lesson)
            r_rep = self._repetition_penalty(lesson)

            Q = (coverage
                 - self.lambda_over * r_over
                 - self.lambda_fid * r_fid
                 - self.lambda_bud * r_bud
                 - self.lambda_rep * r_rep)

            if Q > best_Q_teach:
                best_Q_teach = Q
                best_lesson = lesson

        Q_eval = self._eval_value(m)
        Q_stop = self._stop_value(m)

        # Choose best action
        n_teach = sum(1 for h in self.history if h != "EVAL")
        if Q_stop > best_Q_teach and Q_stop > Q_eval and n_teach >= 5:
            self.stopped = True
            return "STOP", None, round(Q_stop, 4), {
                "deficit": {PROBE_NAMES[i]: round(d[i], 4) for i in range(5)},
                "readiness": self.mastery.readiness(),
            }

        if Q_eval > best_Q_teach and self.eval_count < 2 and n_teach >= 1:
            self.eval_count += 1
            self.history.append("EVAL")
            return "EVAL", None, round(Q_eval, 4), {
                "entropy": self.mastery.entropy(),
            }

        # TEACH
        self.history.append(best_lesson.name)
        self.lesson_counts[best_lesson.name] = self.lesson_counts.get(best_lesson.name, 0) + 1
        self.spent_budget += best_lesson.cost

        return "TEACH", best_lesson, round(best_Q_teach, 4), {
            "deficit": {PROBE_NAMES[i]: round(d[i], 4) for i in range(5)},
            "mastery": dict(u),
            "feasibility": round(best_lesson.feasibility(u), 3),
        }

    def record_realization(self, ep_params: EpisodeParams):
        self.realized_subtypes.append(ep_params.subtype)

    def update_mastery(self, probes: dict):
        self.mastery.update(probes)

    def summary(self) -> dict:
        return {
            "total": len(self.history),
            "unique": len(set(self.history)),
            "counts": dict(self.lesson_counts),
            "sequence": list(self.history),
            "stopped": self.stopped,
            "eval_count": self.eval_count,
            "budget_remaining": round(self.total_budget - self.spent_budget, 2),
            "mastery": dict(self.mastery.mastery()),
            "readiness": self.mastery.readiness(),
        }
