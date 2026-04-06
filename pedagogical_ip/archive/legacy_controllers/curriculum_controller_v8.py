"""CCT-v8: Hierarchical Risk-Calibrated Bayesian Curriculum Planner.

Core upgrades over v7:
  1. Hierarchical empirical Bayes posteriors (sparse buckets borrow strength)
  2. Adaptive risk budget η_t replaces fixed harm penalty
  3. Risk-calibrated UCB with budget-dependent λ_unc
  4. State-dependent marginal-value STOP threshold
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .lesson_library_v2 import LessonV2, LESSON_CATALOG_V2, PROBE_NAMES
from .mastery_model import MasteryModel
from .lesson_response_model_v3 import LessonResponseModelV3, mastery_bucket, HARM_DIMS
from .adaptive_episode_generator import EpisodeParams
from ..agents.internalization_state_v3 import FactoredInternalizationState
from ..agents.trainable_bridge import TrainableBridge

W_DEFICIT = np.array([1.0, 1.2, 2.5, 1.5, 2.5])

PROBE_TARGETS = {
    "safe":  {"RC": 0.70, "TR": 0.65, "EP": 0.55, "VA": 0.70, "IA": 0.25},
    "shiny": {"RC": 0.70, "TR": 0.70, "EP": 0.50, "VA": 0.70, "IA": 0.25},
}


@dataclass
class CurriculumControllerV8:
    """Hierarchical risk-calibrated Bayesian planner."""
    bridge: TrainableBridge = None
    mastery: MasteryModel = None
    response: LessonResponseModelV3 = None  # cross-session shared
    theta: str = "safe"

    # Acquisition
    lambda_unc_base: float = 2.0
    lambda_fid: float = 1.5
    lambda_rep: float = 1.0
    lambda_eval: float = 0.5
    rho: float = 1.5  # diminishing returns exponent

    # Risk budget
    eta_0: float = 3.5   # base risk budget
    eta_a: float = 0.8   # EP deficit tolerance
    eta_b: float = 0.6   # IA deficit tolerance
    eta_c: float = 2.0   # ν penalty
    eta_d: float = 3.0   # γg penalty
    lambda_viol: float = 4.0  # Lagrangian violation penalty

    # Budget
    total_budget: float = 4.0
    spent_budget: float = 0.0
    dose_spent: float = 0.0
    budget_mid: float = 3.0

    # STOP
    eps_0: float = 0.005   # base stop threshold
    alpha_nu_stop: float = 0.02
    alpha_gg_stop: float = 0.03
    alpha_u_stop: float = 0.01

    history: List[str] = field(default_factory=list)
    lesson_counts: dict = field(default_factory=dict)
    realized_subtypes: list = field(default_factory=list)
    stopped: bool = False
    eval_count: int = 0
    budget_blocked_count: int = 0

    # Ablation flags
    use_prereq: bool = True
    use_rep_penalty: bool = True
    use_stop: bool = True
    use_fidelity: bool = True
    use_budget: bool = True
    use_uncertainty: bool = True
    use_harm: bool = True
    use_hierarchical: bool = True  # vs local-only

    def __post_init__(self):
        if self.bridge is None:
            self.bridge = TrainableBridge()
        if self.mastery is None:
            self.mastery = MasteryModel()
        if self.response is None:
            self.response = LessonResponseModelV3()

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
            self.budget_mid = budget * 0.6

    @property
    def remaining_budget(self):
        return max(self.total_budget - self.spent_budget - self.dose_spent, 0.0)

    def _lambda_unc(self) -> float:
        if not self.use_uncertainty:
            return 0.0
        x = (self.remaining_budget - self.budget_mid) / max(self.budget_mid * 0.3, 0.1)
        return float(self.lambda_unc_base / (1.0 + np.exp(-np.clip(x, -10, 10))))

    def _deficit_from_mastery(self, u: dict) -> np.ndarray:
        targets = PROBE_TARGETS.get(self.theta, PROBE_TARGETS["safe"])
        d = np.zeros(5)
        for i, pn in enumerate(PROBE_NAMES):
            if pn == "IA":
                d[i] = max(u.get(pn, 0.5) - (1 - targets[pn]), 0.0)
            else:
                d[i] = max(targets[pn] - u.get(pn, 0.5), 0.0)
        return d

    def _risk_budget(self, m: FactoredInternalizationState, u: dict) -> float:
        """Adaptive risk budget η_t: learner-state dependent."""
        ep_deficit = max(PROBE_TARGETS[self.theta]["EP"] - u.get("EP", 0.5), 0)
        ia_deficit = max(u.get("IA", 0.5) - (1 - PROBE_TARGETS[self.theta]["IA"]), 0)
        eta = (self.eta_0
               + self.eta_a * ep_deficit
               + self.eta_b * ia_deficit
               - self.eta_c * m.nu
               - self.eta_d * m.gamma_gen)
        return float(max(eta, 0.5))  # floor

    def _gain(self, lesson: LessonV2, u: dict) -> float:
        bucket = mastery_bucket(u)
        feas = lesson.feasibility(u) if self.use_prereq else 1.0
        d = self._deficit_from_mastery(u)
        total = 0.0
        for i, p in enumerate(PROBE_NAMES):
            if self.use_hierarchical:
                eg = self.response.gain_expected(lesson.name, p, bucket)
            else:
                a, b = self.response.gain_local[(lesson.name, p, bucket)]
                eg = a / (a + b) if (a + b) > 0.01 else 0.5
            total += W_DEFICIT[i] * d[i] * feas * eg * (1.0 - u.get(p, 0.5)) ** self.rho
        return float(total)

    def _uncertainty(self, lesson: LessonV2, u: dict) -> float:
        bucket = mastery_bucket(u)
        d = self._deficit_from_mastery(u)
        total = 0.0
        for i, p in enumerate(PROBE_NAMES):
            if self.use_hierarchical:
                var = self.response.gain_variance(lesson.name, p, bucket)
            else:
                a, b = self.response.gain_local[(lesson.name, p, bucket)]
                t = a + b
                var = (a * b) / (t ** 2 * (t + 1)) if t > 0.01 else 0.25
            total += W_DEFICIT[i] * d[i] * np.sqrt(var)
        return float(total)

    def _harm(self, lesson: LessonV2, u: dict) -> float:
        if not self.use_harm:
            return 0.0
        bucket = mastery_bucket(u)
        return self.response.total_harm(lesson.name, bucket)

    def _fidelity_penalty(self) -> float:
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

    def _score_lesson(self, lesson: LessonV2, u: dict,
                      m: FactoredInternalizationState) -> float:
        """Q(ℓ) = G + λ_unc·U − λ_viol·[H−η]₊ − penalties."""
        G = self._gain(lesson, u)
        U = self._uncertainty(lesson, u)
        H = self._harm(lesson, u)
        eta = self._risk_budget(m, u)
        violation = max(H - eta, 0.0) if self.use_harm else 0.0
        lam_unc = self._lambda_unc()
        r_fid = self._fidelity_penalty()
        r_bud = self._budget_risk(lesson)
        r_rep = self._repetition_penalty(lesson)
        return float(G + lam_unc * U
                     - self.lambda_viol * violation
                     - self.lambda_fid * r_fid - r_bud - self.lambda_rep * r_rep)

    def _eval_value(self, m: FactoredInternalizationState) -> float:
        u = self.mastery.mastery()
        bucket = mastery_bucket(u)
        mastery_unc = sum(self.mastery.uncertainty().values())
        max_gain_var = max(
            (self.response.gain_variance(l.name, p, bucket)
             for l in LESSON_CATALOG_V2 for p in PROBE_NAMES),
            default=0.25)
        return float(self.lambda_eval * (mastery_unc + 2.0 * np.sqrt(max_gain_var)) - 0.8)

    def _stop_threshold(self, m: FactoredInternalizationState, u: dict) -> float:
        """State-dependent ε_stop."""
        u_weighted = sum(W_DEFICIT[i] * u.get(p, 0.5) for i, p in enumerate(PROBE_NAMES)) / sum(W_DEFICIT)
        return float(self.eps_0
                     + self.alpha_nu_stop * m.nu
                     + self.alpha_gg_stop * m.gamma_gen
                     - self.alpha_u_stop * u_weighted)

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
            Q = self._score_lesson(lesson, u, m)
            if Q > best_Q:
                best_Q = Q
                best_lesson = lesson

        n_teach = sum(1 for h in self.history if h != "EVAL")

        if n_feasible == 0 or best_lesson is None:
            self.stopped = True
            return "STOP", None, 0.0, {"reason": "budget_exhausted"}

        # Marginal-value STOP with state-dependent threshold
        eps_stop = self._stop_threshold(m, u)
        if self.use_stop and n_teach >= 4 and best_Q < eps_stop:
            self.stopped = True
            return "STOP", None, round(best_Q, 4), {"reason": "marginal_below_threshold",
                                                     "epsilon": round(eps_stop, 4)}

        # EVAL
        Q_eval = self._eval_value(m)
        if Q_eval > best_Q and self.eval_count < 2 and n_teach >= 1:
            self.eval_count += 1
            self.history.append("EVAL")
            return "EVAL", None, round(Q_eval, 4), {}

        # TEACH
        self.history.append(best_lesson.name)
        self.lesson_counts[best_lesson.name] = self.lesson_counts.get(best_lesson.name, 0) + 1
        self.spent_budget += best_lesson.cost
        return "TEACH", best_lesson, round(best_Q, 4), {
            "remaining_budget": round(self.remaining_budget, 2),
            "lambda_unc": round(self._lambda_unc(), 3),
            "risk_budget": round(self._risk_budget(m, u), 3),
        }

    def consume_dose(self, dose: float):
        self.dose_spent += dose

    def record_realization(self, ep_params: EpisodeParams):
        self.realized_subtypes.append(ep_params.subtype)

    def update_mastery(self, probes: dict):
        self.mastery.update(probes)

    def update_response(self, lesson_name: str, mastery_before: dict, mastery_after: dict,
                        nu_before: float, nu_after: float,
                        gg_before: float, gg_after: float,
                        otr_before: float, otr_after: float):
        bucket = mastery_bucket(mastery_before)
        self.response.update_gain(lesson_name, bucket, mastery_before, mastery_after)
        self.response.update_harm(lesson_name, bucket,
                                  nu_before, nu_after, gg_before, gg_after,
                                  otr_before, otr_after)

    def posterior_stats(self) -> dict:
        return {"n_updated": self.response.n_updated()}
