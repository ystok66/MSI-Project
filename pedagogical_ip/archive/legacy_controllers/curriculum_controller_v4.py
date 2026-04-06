"""CCT-v4: Bayesian Mastery-Aware Closed-Loop Curriculum Controller.

Key upgrades over v3:
  1. Tighter budget (total=4.0) so budget becomes real scarce resource
  2. Stronger prerequisite gating with explicit feasibility check
  3. Bayesian mastery-informed stop: uses posterior variance, not just mean
  4. Budget risk is continuous and increases as budget depletes
  5. Lesson gain weighted by (1-u)² for faster diminishing returns
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
class CurriculumControllerV4:
    """Bayesian mastery-aware curriculum controller."""
    bridge: TrainableBridge = None
    mastery: MasteryModel = None
    theta: str = "safe"

    lambda_over: float = 3.0
    lambda_fid: float = 1.5
    lambda_bud: float = 2.5
    lambda_rep: float = 1.0
    lambda_eval: float = 0.4
    lambda_stop: float = 0.8

    # Budget — tighter than v3
    total_budget: float = 4.0
    spent_budget: float = 0.0
    dose_spent: float = 0.0

    # History
    history: List[str] = field(default_factory=list)
    lesson_counts: dict = field(default_factory=dict)
    realized_subtypes: list = field(default_factory=list)
    stopped: bool = False
    eval_count: int = 0
    budget_blocked_count: int = 0

    # Thresholds
    nu_max: float = 0.28
    gg_max: float = 0.12

    # Ablation flags
    use_prereq: bool = True
    use_rep_penalty: bool = True
    use_stop: bool = True
    use_fidelity: bool = True
    use_budget: bool = True

    def __post_init__(self):
        if self.bridge is None:
            self.bridge = TrainableBridge()
        if self.mastery is None:
            self.mastery = MasteryModel()

    @property
    def remaining_budget(self):
        return max(self.total_budget - self.spent_budget - self.dose_spent, 0.0)

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

    def _effective_gain(self, lesson: LessonV2, u: dict) -> np.ndarray:
        """feas(ℓ|u) · g · (1-u)²: faster diminishing returns."""
        if self.use_prereq:
            feas = lesson.feasibility(u)
        else:
            feas = 1.0
        u_arr = np.array([u.get(p, 0.5) for p in PROBE_NAMES])
        return feas * lesson.gain * (1.0 - u_arr) ** 2

    def _overteach_risk(self, lesson: LessonV2, m: FactoredInternalizationState) -> float:
        nu_next = m.nu + lesson.nu_push
        gg_next = m.gamma_gen + lesson.gg_push
        r = 2.5 * max(nu_next - self.nu_max, 0) ** 2
        r += 3.0 * max(gg_next - self.gg_max, 0) ** 2
        return float(r)

    def _fidelity_penalty(self, lesson: LessonV2) -> float:
        if not self.use_fidelity:
            return 0.0
        n_teach = sum(1 for h in self.history if h != "EVAL")
        base_lf = 0.93 if n_teach < 3 else 0.88
        return float(1.0 - base_lf)

    def _budget_risk(self, lesson: LessonV2) -> float:
        if not self.use_budget:
            return 0.0
        rem = self.remaining_budget
        if rem <= 0.01:
            return 5.0  # Hard block
        ratio = lesson.cost / rem
        # Steep penalty when budget gets tight
        return float(max(ratio - 0.25, 0) ** 2 + 0.5 * max(ratio - 0.6, 0))

    def _repetition_penalty(self, lesson: LessonV2) -> float:
        if not self.use_rep_penalty:
            return 0.0
        recent = self.history[-5:] if len(self.history) >= 5 else self.history
        if not recent:
            return 0.0
        n_recent = sum(1 for h in recent if h == lesson.name)
        return float(n_recent / max(len(recent), 1))

    def _eval_value(self, m: FactoredInternalizationState) -> float:
        ent = self.mastery.entropy()
        # Also consider posterior variance of mastery
        unc = self.mastery.uncertainty()
        max_unc = max(unc.values())
        return float(self.lambda_eval * (ent + 2.0 * max_unc) - 0.8)

    def _stop_value(self, m: FactoredInternalizationState) -> float:
        if not self.use_stop:
            return -100.0
        n_teach = sum(1 for h in self.history if h != "EVAL")
        if n_teach < 4:
            return -10.0
        d = self._deficit(m)
        deficit_mag = float(np.sum(d))
        ready = self.mastery.readiness()
        # Also penalize stop if mastery variance is still high
        max_unc = max(self.mastery.uncertainty().values())
        stop_bonus = ready - 1.5 * m.nu - 2.0 * m.gamma_gen - 3.0 * max_unc
        return float(self.lambda_stop * stop_bonus - 4.0 * deficit_mag)

    def select_action(self, m: FactoredInternalizationState,
                      candidates: Optional[List[LessonV2]] = None) -> tuple:
        if self.stopped:
            return "STOP", None, 0.0, {"reason": "already_stopped"}
        if candidates is None:
            candidates = LESSON_CATALOG_V2

        d = self._deficit(m)
        u = self.mastery.mastery()

        # Score lessons — filter by budget feasibility first
        best_lesson = None
        best_Q = -1e9
        n_feasible = 0
        for lesson in candidates:
            # Hard budget check
            if self.use_budget and lesson.cost > self.remaining_budget + 0.01:
                self.budget_blocked_count += 1
                continue
            n_feasible += 1

            eg = self._effective_gain(lesson, u)
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

            if Q > best_Q:
                best_Q = Q
                best_lesson = lesson

        Q_eval = self._eval_value(m)
        Q_stop = self._stop_value(m)
        n_teach = sum(1 for h in self.history if h != "EVAL")

        # If no feasible lessons remain
        if n_feasible == 0 or best_lesson is None:
            self.stopped = True
            return "STOP", None, 0.0, {"reason": "budget_exhausted"}

        # STOP check
        if Q_stop > best_Q and Q_stop > Q_eval and n_teach >= 4:
            self.stopped = True
            return "STOP", None, round(Q_stop, 4), {
                "deficit": {PROBE_NAMES[i]: round(d[i], 4) for i in range(5)},
                "readiness": self.mastery.readiness(),
            }

        # EVAL check
        if Q_eval > best_Q and self.eval_count < 2 and n_teach >= 1:
            self.eval_count += 1
            self.history.append("EVAL")
            return "EVAL", None, round(Q_eval, 4), {
                "entropy": self.mastery.entropy(),
            }

        # TEACH
        self.history.append(best_lesson.name)
        self.lesson_counts[best_lesson.name] = self.lesson_counts.get(best_lesson.name, 0) + 1
        self.spent_budget += best_lesson.cost

        return "TEACH", best_lesson, round(best_Q, 4), {
            "deficit": {PROBE_NAMES[i]: round(d[i], 4) for i in range(5)},
            "mastery": dict(u),
            "feasibility": round(best_lesson.feasibility(u), 3) if self.use_prereq else 1.0,
            "remaining_budget": round(self.remaining_budget, 2),
        }

    def consume_dose(self, dose: float):
        self.dose_spent += dose

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
            "budget_remaining": round(self.remaining_budget, 2),
            "budget_blocked": self.budget_blocked_count,
            "mastery": dict(self.mastery.mastery()),
            "readiness": self.mastery.readiness(),
        }
