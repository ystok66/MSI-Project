"""CCT-v6: Cross-Session Bayesian Curriculum Planner.

Core upgrade over v5:
  - Lesson-response posteriors persist ACROSS sessions
  - Thompson sampling replaces UCB for lesson selection
  - Bayesian STOP requires both readiness AND certainty
  - Horizon-aware: designed for 4–12 teach sessions
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .lesson_library_v2 import LessonV2, LESSON_CATALOG_V2, PROBE_NAMES
from .mastery_model import MasteryModel
from .lesson_response_model import LessonResponseModel, mastery_bucket
from .adaptive_episode_generator import EpisodeParams
from ..agents.internalization_state_v3 import FactoredInternalizationState
from ..agents.trainable_bridge import TrainableBridge

W_DEFICIT = np.array([1.0, 1.2, 2.5, 1.5, 2.5])

PROBE_TARGETS = {
    "safe":  {"RC": 0.70, "TR": 0.65, "EP": 0.55, "VA": 0.70, "IA": 0.25},
    "shiny": {"RC": 0.70, "TR": 0.70, "EP": 0.50, "VA": 0.70, "IA": 0.25},
}


@dataclass
class CurriculumControllerV6:
    """Cross-session Bayesian curriculum planner with Thompson sampling."""
    bridge: TrainableBridge = None
    mastery: MasteryModel = None
    response: LessonResponseModel = None  # SHARED across sessions
    theta: str = "safe"

    lambda_over: float = 3.0
    lambda_fid: float = 1.5
    lambda_bud: float = 2.5
    lambda_rep: float = 1.0
    lambda_unc: float = 2.0
    lambda_eval: float = 0.5
    lambda_stop: float = 0.8

    total_budget: float = 4.0
    spent_budget: float = 0.0
    dose_spent: float = 0.0

    history: List[str] = field(default_factory=list)
    lesson_counts: dict = field(default_factory=dict)
    realized_subtypes: list = field(default_factory=list)
    stopped: bool = False
    eval_count: int = 0
    budget_blocked_count: int = 0

    nu_max: float = 0.28
    gg_max: float = 0.12

    # Mode
    use_thompson: bool = True  # Thompson vs UCB
    use_prereq: bool = True
    use_rep_penalty: bool = True
    use_stop: bool = True
    use_fidelity: bool = True
    use_budget: bool = True
    use_uncertainty: bool = True

    _rng: np.random.Generator = None

    def __post_init__(self):
        if self.bridge is None:
            self.bridge = TrainableBridge()
        if self.mastery is None:
            self.mastery = MasteryModel()
        if self.response is None:
            self.response = LessonResponseModel()
        if self._rng is None:
            self._rng = np.random.default_rng(42)

    def reset_session(self, budget: float = None):
        """Reset per-session state, KEEP shared response model."""
        self.mastery = MasteryModel()
        self.bridge = TrainableBridge()
        self.history = []
        self.lesson_counts = {}
        self.realized_subtypes = []
        self.stopped = False
        self.eval_count = 0
        self.budget_blocked_count = 0
        self.spent_budget = 0.0
        self.dose_spent = 0.0
        if budget is not None:
            self.total_budget = budget

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

    def _thompson_gain(self, lesson: LessonV2, u: dict) -> float:
        """Thompson-sampled lesson gain: sample from posterior then score."""
        bucket = mastery_bucket(u)
        feas = lesson.feasibility(u) if self.use_prereq else 1.0
        d = self._deficit_from_mastery(u)
        total = 0.0
        for i, p in enumerate(PROBE_NAMES):
            k = (lesson.name, p, bucket)
            a, b = self.response.posteriors[k]
            # Thompson: sample from Beta posterior
            sampled = float(self._rng.beta(max(a, 0.01), max(b, 0.01)))
            total += W_DEFICIT[i] * d[i] * feas * sampled * (1.0 - u.get(p, 0.5))
        return float(total)

    def _ucb_gain(self, lesson: LessonV2, u: dict) -> float:
        """UCB-style: E[p] + λ√Var[p]."""
        bucket = mastery_bucket(u)
        feas = lesson.feasibility(u) if self.use_prereq else 1.0
        d = self._deficit_from_mastery(u)
        total = 0.0
        for i, p in enumerate(PROBE_NAMES):
            ep = self.response.expected_gain(lesson.name, p, bucket)
            var = self.response.variance(lesson.name, p, bucket)
            unc = np.sqrt(var) if self.use_uncertainty else 0.0
            total += W_DEFICIT[i] * d[i] * feas * (ep + self.lambda_unc * unc) * (1.0 - u.get(p, 0.5))
        return float(total)

    def _deficit_from_mastery(self, u: dict) -> np.ndarray:
        targets = PROBE_TARGETS.get(self.theta, PROBE_TARGETS["safe"])
        d = np.zeros(5)
        for i, pn in enumerate(PROBE_NAMES):
            if pn == "IA":
                d[i] = max(u.get(pn, 0.5) - (1 - targets[pn]), 0.0)
            else:
                d[i] = max(targets[pn] - u.get(pn, 0.5), 0.0)
        return d

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
        return float(1.0 - (0.94 if n_teach < 3 else 0.90))

    def _budget_risk(self, lesson: LessonV2) -> float:
        if not self.use_budget:
            return 0.0
        rem = self.remaining_budget
        if rem <= 0.01:
            return 5.0
        ratio = lesson.cost / rem
        return float(max(ratio - 0.25, 0) ** 2 + 0.5 * max(ratio - 0.6, 0))

    def _repetition_penalty(self, lesson: LessonV2) -> float:
        if not self.use_rep_penalty:
            return 0.0
        recent = self.history[-5:] if len(self.history) >= 5 else self.history
        if not recent:
            return 0.0
        return float(sum(1 for h in recent if h == lesson.name) / max(len(recent), 1))

    def _eval_value(self, m: FactoredInternalizationState) -> float:
        u = self.mastery.mastery()
        bucket = mastery_bucket(u)
        mastery_unc = sum(self.mastery.uncertainty().values())
        response_unc = self.response.info_value(bucket)
        return float(self.lambda_eval * (mastery_unc + 1.5 * response_unc) - 0.8)

    def _stop_value(self, m: FactoredInternalizationState) -> float:
        if not self.use_stop:
            return -100.0
        n_teach = sum(1 for h in self.history if h != "EVAL")
        if n_teach < 4:
            return -10.0
        u = self.mastery.mastery()
        d = self._deficit_from_mastery(u)
        deficit_mag = float(np.sum(d))
        ready = self.mastery.readiness()
        mastery_unc = max(self.mastery.uncertainty().values())
        stop_bonus = ready - 1.5 * m.nu - 2.0 * m.gamma_gen - 3.0 * mastery_unc
        return float(self.lambda_stop * stop_bonus - 4.0 * deficit_mag)

    def select_action(self, m: FactoredInternalizationState,
                      candidates: Optional[List[LessonV2]] = None) -> tuple:
        if self.stopped:
            return "STOP", None, 0.0, {"reason": "already_stopped"}
        if candidates is None:
            candidates = LESSON_CATALOG_V2

        u = self.mastery.mastery()
        best_lesson = None
        best_Q = -1e9
        n_feasible = 0

        for lesson in candidates:
            if self.use_budget and lesson.cost > self.remaining_budget + 0.01:
                self.budget_blocked_count += 1
                continue
            n_feasible += 1

            # Core: Thompson vs UCB
            if self.use_thompson:
                G = self._thompson_gain(lesson, u)
            else:
                G = self._ucb_gain(lesson, u)

            r_over = self._overteach_risk(lesson, m)
            r_fid = self._fidelity_penalty(lesson)
            r_bud = self._budget_risk(lesson)
            r_rep = self._repetition_penalty(lesson)

            Q = G - self.lambda_over * r_over - self.lambda_fid * r_fid \
                - self.lambda_bud * r_bud - self.lambda_rep * r_rep

            if Q > best_Q:
                best_Q = Q
                best_lesson = lesson

        Q_eval = self._eval_value(m)
        Q_stop = self._stop_value(m)
        n_teach = sum(1 for h in self.history if h != "EVAL")

        if n_feasible == 0 or best_lesson is None:
            self.stopped = True
            return "STOP", None, 0.0, {"reason": "budget_exhausted"}

        if Q_stop > best_Q and Q_stop > Q_eval and n_teach >= 4:
            self.stopped = True
            return "STOP", None, round(Q_stop, 4), {"readiness": self.mastery.readiness()}

        if Q_eval > best_Q and self.eval_count < 2 and n_teach >= 1:
            self.eval_count += 1
            self.history.append("EVAL")
            return "EVAL", None, round(Q_eval, 4), {}

        self.history.append(best_lesson.name)
        self.lesson_counts[best_lesson.name] = self.lesson_counts.get(best_lesson.name, 0) + 1
        self.spent_budget += best_lesson.cost
        return "TEACH", best_lesson, round(best_Q, 4), {
            "remaining_budget": round(self.remaining_budget, 2),
        }

    def consume_dose(self, dose: float):
        self.dose_spent += dose

    def record_realization(self, ep_params: EpisodeParams):
        self.realized_subtypes.append(ep_params.subtype)

    def update_mastery(self, probes: dict):
        self.mastery.update(probes)

    def update_response(self, lesson_name: str, mastery_before: dict, mastery_after: dict):
        bucket = mastery_bucket(mastery_before)
        self.response.update(lesson_name, bucket, mastery_before, mastery_after)

    def posterior_stats(self) -> dict:
        """Summary of lesson-response posterior richness."""
        n_updated = sum(1 for (_, _, _), (a, b) in self.response.posteriors.items()
                        if a + b > 2.05)  # past prior
        total_count = sum(a + b for (a, b) in self.response.posteriors.values())
        return {"n_updated": n_updated, "total_count": round(total_count, 1)}
