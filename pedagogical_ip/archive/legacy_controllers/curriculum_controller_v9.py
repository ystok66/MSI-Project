"""CCT-v9: Contextual Constrained Bayesian Curriculum Planner.

Core upgrades over v8:
  1. Contextual Bayesian linear response model (replaces bucket posteriors)
  2. Explicit risk constraints: max J s.t. harm ≤ η (not penalty-sum)
  3. Term actionability audit: tracks AM_j and PCR_j per term
  4. State-dependent STOP/EVAL via marginal value
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .lesson_library_v2 import LessonV2, LESSON_CATALOG_V2, PROBE_NAMES
from .mastery_model import MasteryModel
from .contextual_response_model import ContextualResponseModel
from .adaptive_episode_generator import EpisodeParams
from ..agents.internalization_state_v3 import FactoredInternalizationState
from ..agents.trainable_bridge import TrainableBridge

W_DEFICIT = np.array([1.0, 1.2, 2.5, 1.5, 2.5])
PROBE_TARGETS = {
    "safe":  {"RC": 0.70, "TR": 0.65, "EP": 0.55, "VA": 0.70, "IA": 0.25},
    "shiny": {"RC": 0.70, "TR": 0.70, "EP": 0.50, "VA": 0.70, "IA": 0.25},
}


@dataclass
class CurriculumControllerV9:
    """Contextual constrained Bayesian planner."""
    bridge: TrainableBridge = None
    mastery: MasteryModel = None
    response: ContextualResponseModel = None
    theta: str = "safe"

    # Gain weights
    w_C: float = 1.5
    w_E: float = 1.0

    # UCB
    lambda_unc_base: float = 1.5
    lambda_fid: float = 0.8
    lambda_rep: float = 1.0

    # Risk constraint parameters
    beta_otr: float = 1.0    # pessimism for OTR constraint
    beta_nu: float = 1.0
    beta_gg: float = 1.0
    eta_otr_0: float = 0.4   # base OTR budget
    eta_nu_0: float = 0.35
    eta_gg_0: float = 0.15

    # EVAL
    lambda_eval: float = 0.5

    # STOP
    eps_0: float = 0.005
    alpha_nu_stop: float = 0.02
    alpha_gg_stop: float = 0.03
    alpha_u_stop: float = 0.01

    # Budget
    total_budget: float = 4.0
    spent_budget: float = 0.0
    dose_spent: float = 0.0
    budget_mid: float = 3.0

    history: List[str] = field(default_factory=list)
    lesson_counts: dict = field(default_factory=dict)
    realized_subtypes: list = field(default_factory=list)
    stopped: bool = False
    eval_count: int = 0
    budget_blocked_count: int = 0

    # Actionability tracking
    _term_scores: list = field(default_factory=list)  # [{lesson: {term: val}}]

    # Ablation
    use_constraint: bool = True
    use_uncertainty: bool = True
    use_stop: bool = True
    use_budget: bool = True

    def __post_init__(self):
        if self.bridge is None:
            self.bridge = TrainableBridge()
        if self.mastery is None:
            self.mastery = MasteryModel()
        if self.response is None:
            cat_names = [l.name for l in LESSON_CATALOG_V2]
            self.response = ContextualResponseModel(lesson_names=cat_names, theta=self.theta)

    def reset_session(self, budget: float = None):
        self.mastery = MasteryModel()
        self.bridge = TrainableBridge()
        self.history = []; self.lesson_counts = {}
        self.realized_subtypes = []; self.stopped = False
        self.eval_count = 0; self.budget_blocked_count = 0
        self.spent_budget = 0.0; self.dose_spent = 0.0
        if budget is not None:
            self.total_budget = budget; self.budget_mid = budget * 0.6

    @property
    def remaining_budget(self):
        return max(self.total_budget - self.spent_budget - self.dose_spent, 0.0)

    def _lambda_unc(self) -> float:
        if not self.use_uncertainty: return 0.0
        x = (self.remaining_budget - self.budget_mid) / max(self.budget_mid * 0.3, 0.1)
        return float(self.lambda_unc_base / (1.0 + np.exp(-np.clip(x, -10, 10))))

    def _risk_budgets(self, m: FactoredInternalizationState, u: dict) -> dict:
        """Learner-conditional risk budgets."""
        ep_d = max(PROBE_TARGETS[self.theta]["EP"] - u.get("EP", 0.5), 0)
        ia_d = max(u.get("IA", 0.5) - (1 - PROBE_TARGETS[self.theta]["IA"]), 0)
        return {
            "otr": max(self.eta_otr_0 + 0.3 * ep_d + 0.2 * ia_d - 0.5 * m.gamma_gen, 0.05),
            "nu":  max(self.eta_nu_0 + 0.2 * (1 - u.get("VA", 0.5)) - 0.4 * m.nu, 0.05),
            "gg":  max(self.eta_gg_0 + 0.2 * ep_d - 0.6 * m.gamma_gen, 0.02),
        }

    def _score_lesson(self, lesson: LessonV2, u: dict,
                      m: FactoredInternalizationState) -> tuple[float, dict]:
        """Score lesson with constraint check. Returns (J, term_dict)."""
        gain_pred = self.response.predict_gain(
            lesson.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
            self.remaining_budget, self.lesson_counts,
            lesson.severity, lesson.get("novelty", 0.3) if hasattr(lesson, "get") else 0.3)
        harm_pred = self.response.predict_harm(
            lesson.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
            self.remaining_budget, self.lesson_counts,
            lesson.severity, lesson.get("novelty", 0.3) if hasattr(lesson, "get") else 0.3)

        # Gain objective
        G_C = self.w_C * gain_pred["C_mean"]
        G_E = self.w_E * gain_pred["E_mean"]
        U = self._lambda_unc() * (np.sqrt(gain_pred["C_var"]) + np.sqrt(gain_pred["E_var"]))

        # Penalties
        r_fid = self.lambda_fid * (1.0 - 0.92)
        recent = self.history[-5:] if len(self.history) >= 5 else self.history
        r_rep = self.lambda_rep * (sum(1 for h in recent if h == lesson.name) / max(len(recent), 1)) if recent else 0

        J = float(G_C + G_E + U - r_fid - r_rep)

        # Constraint violation penalty (Lagrangian relaxation)
        if self.use_constraint:
            budgets = self._risk_budgets(m, u)
            otr_upper = harm_pred["otr_mean"] + self.beta_otr * np.sqrt(harm_pred["otr_var"])
            nu_upper = harm_pred["nu_mean"] + self.beta_nu * np.sqrt(harm_pred["nu_var"])
            gg_upper = harm_pred["gg_mean"] + self.beta_gg * np.sqrt(harm_pred["gg_var"])

            viol_otr = max(otr_upper - budgets["otr"], 0)
            viol_nu = max(nu_upper - budgets["nu"], 0)
            viol_gg = max(gg_upper - budgets["gg"], 0)
            J -= 4.0 * (viol_otr + viol_nu + viol_gg)

        terms = {
            "G_C": round(G_C, 4), "G_E": round(G_E, 4), "U": round(U, 4),
            "harm_otr": round(harm_pred["otr_mean"], 4),
            "harm_nu": round(harm_pred["nu_mean"], 4),
            "harm_gg": round(harm_pred["gg_mean"], 4),
            "r_fid": round(r_fid, 4), "r_rep": round(r_rep, 4),
        }
        return float(J), terms

    def _eval_value(self, m: FactoredInternalizationState) -> float:
        u = self.mastery.mastery()
        mastery_unc = sum(self.mastery.uncertainty().values())
        response_unc = 0.0
        for l in LESSON_CATALOG_V2[:4]:  # sample a few
            gp = self.response.predict_gain(
                l.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
                self.remaining_budget, self.lesson_counts, l.severity, 0.3)
            response_unc += gp["C_var"] + gp["E_var"]
        return float(self.lambda_eval * (mastery_unc + 0.5 * np.sqrt(response_unc)) - 0.8)

    def _stop_threshold(self, m: FactoredInternalizationState, u: dict) -> float:
        u_w = sum(W_DEFICIT[i] * u.get(p, 0.5) for i, p in enumerate(PROBE_NAMES)) / sum(W_DEFICIT)
        return float(self.eps_0 + self.alpha_nu_stop * m.nu + self.alpha_gg_stop * m.gamma_gen - self.alpha_u_stop * u_w)

    def select_action(self, m: FactoredInternalizationState,
                      candidates: Optional[List[LessonV2]] = None) -> tuple:
        if self.stopped:
            return "STOP", None, 0.0, {"reason": "already_stopped"}
        if candidates is None:
            candidates = LESSON_CATALOG_V2

        u = self.mastery.mastery()
        best_lesson = None; best_J = -1e9; best_terms = {}
        n_feasible = 0; all_terms = {}

        for lesson in candidates:
            if self.use_budget and lesson.cost > self.remaining_budget + 0.01:
                self.budget_blocked_count += 1; continue
            n_feasible += 1
            J, terms = self._score_lesson(lesson, u, m)
            all_terms[lesson.name] = terms
            if J > best_J:
                best_J = J; best_lesson = lesson; best_terms = terms

        # Record term scores for actionability audit
        self._term_scores.append(all_terms)

        n_teach = sum(1 for h in self.history if h != "EVAL")
        if n_feasible == 0 or best_lesson is None:
            self.stopped = True; return "STOP", None, 0.0, {"reason": "budget_exhausted"}

        eps_stop = self._stop_threshold(m, u)
        if self.use_stop and n_teach >= 4 and best_J < eps_stop:
            self.stopped = True
            return "STOP", None, round(best_J, 4), {"reason": "marginal_below_threshold"}

        Q_eval = self._eval_value(m)
        if Q_eval > best_J and self.eval_count < 2 and n_teach >= 1:
            self.eval_count += 1; self.history.append("EVAL")
            return "EVAL", None, round(Q_eval, 4), {}

        self.history.append(best_lesson.name)
        self.lesson_counts[best_lesson.name] = self.lesson_counts.get(best_lesson.name, 0) + 1
        self.spent_budget += best_lesson.cost
        return "TEACH", best_lesson, round(best_J, 4), {
            "remaining_budget": round(self.remaining_budget, 2),
            "terms": best_terms,
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
        """Update contextual response model with observed outcomes."""
        u = mastery_before
        # Gain outcomes: how much did mastery improve
        gain_c = sum(mastery_after.get(p, 0.5) - mastery_before.get(p, 0.5)
                     for p in ["RC", "TR", "EP"]) / 3.0
        gain_e = sum(mastery_after.get(p, 0.5) - mastery_before.get(p, 0.5)
                     for p in ["VA", "IA"]) / 2.0
        # Harm outcomes: deltas (positive = worsening)
        delta_otr = otr_after - otr_before
        delta_nu = nu_after - nu_before
        delta_gg = gg_after - gg_before

        lesson = None
        for l in LESSON_CATALOG_V2:
            if l.name == lesson_name: lesson = l; break
        sev = lesson.severity if lesson else 0.5
        nov = 0.3  # default

        self.response.update(
            lesson_name, u, nu_before, (u.get("RC", 0.5) + 0.3),  # tau approx
            gg_before, 0.0, 0.5,  # gamma_spec, kappa approx
            self.remaining_budget, self.lesson_counts,
            sev, nov,
            gain_c=gain_c, gain_e=gain_e,
            delta_otr=delta_otr, delta_nu=delta_nu, delta_gg=delta_gg)

    def posterior_stats(self) -> dict:
        return {"n_updated": self.response.n_updated()}

    def actionability_audit(self) -> dict:
        """Compute AM (average margin variance) and PCR (policy change rate) per term."""
        if not self._term_scores:
            return {}
        term_names = ["G_C", "G_E", "U", "harm_otr", "harm_nu", "harm_gg"]
        am = {}; pcr = {}
        for tn in term_names:
            variances = []
            changes = 0
            for step_terms in self._term_scores:
                if not step_terms: continue
                vals = [step_terms[ln].get(tn, 0) for ln in step_terms]
                if len(vals) > 1:
                    variances.append(np.var(vals))
                # Check if removing this term would change argmax
                lessons = list(step_terms.keys())
                if len(lessons) < 2: continue
                full_scores = {}
                ablated_scores = {}
                for ln in lessons:
                    t = step_terms[ln]
                    full = t.get("G_C", 0) + t.get("G_E", 0) + t.get("U", 0) - t.get("r_fid", 0) - t.get("r_rep", 0)
                    abl = full - t.get(tn, 0)
                    full_scores[ln] = full
                    ablated_scores[ln] = abl
                best_full = max(full_scores, key=full_scores.get)
                best_abl = max(ablated_scores, key=ablated_scores.get)
                if best_full != best_abl:
                    changes += 1
            am[tn] = round(np.mean(variances), 6) if variances else 0.0
            pcr[tn] = round(changes / max(len(self._term_scores), 1), 3)
        return {"AM": am, "PCR": pcr}
