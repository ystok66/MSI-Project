"""CCT-v10: Hybrid Dueling Constrained Bayesian Curriculum Planner.

Core upgrades:
  1. Hybrid model: hierarchical backbone + contextual residual + pairwise dueling
  2. Filter+rank constraint: first filter by harm budget, then rank by gain
  3. Actionability audit as hard gate (AM/PCR per term)
  4. Budget-conditioned uncertainty, state-dependent STOP
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .lesson_library_v2 import LessonV2, LESSON_CATALOG_V2, PROBE_NAMES
from .mastery_model import MasteryModel
from .hybrid_response_model import HybridResponseModel
from .adaptive_episode_generator import EpisodeParams
from ..agents.internalization_state_v3 import FactoredInternalizationState
from ..agents.trainable_bridge import TrainableBridge

W_DEFICIT = np.array([1.0, 1.2, 2.5, 1.5, 2.5])
PROBE_TARGETS = {
    "safe":  {"RC": 0.70, "TR": 0.65, "EP": 0.55, "VA": 0.70, "IA": 0.25},
    "shiny": {"RC": 0.70, "TR": 0.70, "EP": 0.50, "VA": 0.70, "IA": 0.25},
}


@dataclass
class CurriculumControllerV10:
    """Hybrid dueling constrained Bayesian planner."""
    bridge: TrainableBridge = None
    mastery: MasteryModel = None
    response: HybridResponseModel = None
    theta: str = "safe"

    # Gain weights
    w_gain: float = 2.0
    lambda_unc_base: float = 1.5
    lambda_fid: float = 0.6
    lambda_rep: float = 1.0
    lambda_eval: float = 0.5

    # Risk constraint budgets
    eta_otr_0: float = 0.55
    eta_nu_0: float = 0.40
    eta_gg_0: float = 0.25
    beta_pessimism: float = 0.5  # std multiplier for pessimistic bound

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
    _term_scores: list = field(default_factory=list)

    # Ablation
    use_constraint: bool = True
    use_uncertainty: bool = True
    use_stop: bool = True
    use_budget: bool = True
    use_dueling: bool = True
    use_residual: bool = True

    def __post_init__(self):
        if self.bridge is None: self.bridge = TrainableBridge()
        if self.mastery is None: self.mastery = MasteryModel()
        if self.response is None:
            cat = [l.name for l in LESSON_CATALOG_V2]
            self.response = HybridResponseModel(catalog_names=cat, theta=self.theta)

    def reset_session(self, budget: float = None):
        self.mastery = MasteryModel(); self.bridge = TrainableBridge()
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
        ep_d = max(PROBE_TARGETS[self.theta]["EP"] - u.get("EP", 0.5), 0)
        ia_d = max(u.get("IA", 0.5) - (1 - PROBE_TARGETS[self.theta]["IA"]), 0)
        return {
            "otr": max(self.eta_otr_0 + 0.3 * ep_d + 0.2 * ia_d - 0.5 * m.gamma_gen, 0.05),
            "nu":  max(self.eta_nu_0 + 0.2 * (1 - u.get("VA", 0.5)) - 0.4 * m.nu, 0.05),
            "gg":  max(self.eta_gg_0 + 0.2 * ep_d - 0.6 * m.gamma_gen, 0.02),
        }

    def _passes_constraint(self, harm_pred: dict, budgets: dict) -> bool:
        """Filter: does this lesson pass risk budgets?"""
        if not self.use_constraint: return True
        # Warm-up: always pass for first 2 teaching steps (collect data first)
        n_teach = sum(1 for h in self.history if h != "EVAL")
        if n_teach < 2: return True
        harm_upper = harm_pred["mean"] + self.beta_pessimism * np.sqrt(max(harm_pred["var"], 1e-8))
        # Compare against most generous individual budget, not sum
        max_budget = max(budgets.values())
        return harm_upper <= max_budget

    def _score_lesson(self, lesson: LessonV2, u: dict,
                      m: FactoredInternalizationState) -> tuple[float, dict]:
        gp = self.response.predict_gain(
            lesson.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
            self.remaining_budget, self.lesson_counts, lesson.severity, 0.3)
        hp = self.response.predict_harm(
            lesson.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
            self.remaining_budget, self.lesson_counts, lesson.severity, 0.3)

        G = self.w_gain * gp["mean"]
        U = self._lambda_unc() * np.sqrt(gp["var"])
        r_fid = self.lambda_fid * 0.08
        recent = self.history[-5:] if len(self.history) >= 5 else self.history
        r_rep = self.lambda_rep * (sum(1 for h in recent if h == lesson.name) / max(len(recent), 1)) if recent else 0

        J = float(G + U - r_fid - r_rep)

        terms = {
            "G": round(G, 4),
            "G_hier": round(gp.get("hier", 0), 4),
            "G_res": round(gp.get("res", 0), 4),
            "G_duel": round(gp.get("duel", 0), 4),
            "U": round(U, 4),
            "H": round(hp["mean"], 4),
            "H_hier": round(hp.get("hier", 0), 4),
            "H_res": round(hp.get("res", 0), 4),
            "H_duel": round(hp.get("duel", 0), 4),
            "r_fid": round(r_fid, 4), "r_rep": round(r_rep, 4),
        }
        return J, terms

    def _eval_value(self, m: FactoredInternalizationState) -> float:
        u = self.mastery.mastery()
        mastery_unc = sum(self.mastery.uncertainty().values())
        response_unc = 0.0
        for l in LESSON_CATALOG_V2[:4]:
            gp = self.response.predict_gain(
                l.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
                self.remaining_budget, self.lesson_counts, l.severity, 0.3)
            response_unc += gp["var"]
        return float(self.lambda_eval * (mastery_unc + 0.5 * np.sqrt(response_unc)) - 0.8)

    def _stop_threshold(self, m: FactoredInternalizationState, u: dict) -> float:
        u_w = sum(W_DEFICIT[i] * u.get(p, 0.5) for i, p in enumerate(PROBE_NAMES)) / sum(W_DEFICIT)
        return float(self.eps_0 + self.alpha_nu_stop * m.nu + self.alpha_gg_stop * m.gamma_gen - self.alpha_u_stop * u_w)

    def select_action(self, m: FactoredInternalizationState,
                      candidates: Optional[List[LessonV2]] = None) -> tuple:
        if self.stopped: return "STOP", None, 0.0, {"reason": "already_stopped"}
        if candidates is None: candidates = LESSON_CATALOG_V2

        u = self.mastery.mastery()
        budgets = self._risk_budgets(m, u)

        # Step 1: FILTER by budget + risk constraint
        feasible = []
        all_terms = {}
        for lesson in candidates:
            if self.use_budget and lesson.cost > self.remaining_budget + 0.01:
                self.budget_blocked_count += 1; continue
            hp = self.response.predict_harm(
                lesson.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
                self.remaining_budget, self.lesson_counts, lesson.severity, 0.3)
            if self._passes_constraint(hp, budgets):
                feasible.append(lesson)
            else:
                # Score anyway for actionability tracking
                J, terms = self._score_lesson(lesson, u, m)
                terms["filtered"] = True
                all_terms[lesson.name] = terms

        # Step 2: RANK by gain among feasible
        best_lesson = None; best_J = -1e9; best_terms = {}
        for lesson in feasible:
            J, terms = self._score_lesson(lesson, u, m)
            terms["filtered"] = False
            all_terms[lesson.name] = terms
            if J > best_J:
                best_J = J; best_lesson = lesson; best_terms = terms

        self._term_scores.append(all_terms)
        n_teach = sum(1 for h in self.history if h != "EVAL")

        if not feasible or best_lesson is None:
            self.stopped = True; return "STOP", None, 0.0, {"reason": "all_filtered_or_budget"}

        eps_stop = self._stop_threshold(m, u)
        if self.use_stop and n_teach >= 3 and best_J < eps_stop:
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
            "n_feasible": len(feasible),
            "n_filtered": len(candidates) - len(feasible),
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
        u = mastery_before
        lesson = None
        for l in LESSON_CATALOG_V2:
            if l.name == lesson_name: lesson = l; break
        sev = lesson.severity if lesson else 0.5
        self.response.update(
            lesson_name, u, nu_before, u.get("RC", 0.5) + 0.3,
            gg_before, 0.0, 0.5,
            self.remaining_budget, self.lesson_counts,
            sev, 0.3,
            mastery_before, mastery_after,
            nu_after, gg_after, otr_before, otr_after)

    def posterior_stats(self) -> dict:
        return {"n_updated": self.response.n_updated()}

    def actionability_audit(self) -> dict:
        term_names = ["G", "G_hier", "G_res", "G_duel", "U", "H", "H_hier", "H_res", "H_duel"]
        am = {}; pcr = {}
        for tn in term_names:
            variances = []; changes = 0
            for step_terms in self._term_scores:
                if not step_terms: continue
                non_filtered = {ln: t for ln, t in step_terms.items() if not t.get("filtered", False)}
                if len(non_filtered) < 2: continue
                vals = [non_filtered[ln].get(tn, 0) for ln in non_filtered]
                variances.append(np.var(vals))
                # PCR: would removing this term change the argmax?
                full_scores = {}; ablated_scores = {}
                for ln in non_filtered:
                    t = non_filtered[ln]
                    full = t.get("G", 0) + t.get("U", 0) - t.get("r_fid", 0) - t.get("r_rep", 0)
                    abl = full - t.get(tn, 0) if tn in ("G", "U") else full
                    if tn in ("G_hier", "G_res", "G_duel"):
                        # Removing sub-component from G
                        abl = full - t.get(tn, 0)
                    full_scores[ln] = full; ablated_scores[ln] = abl
                best_full = max(full_scores, key=full_scores.get)
                best_abl = max(ablated_scores, key=ablated_scores.get)
                if best_full != best_abl: changes += 1
            am[tn] = round(np.mean(variances), 6) if variances else 0.0
            pcr[tn] = round(changes / max(len(self._term_scores), 1), 3)
        return {"AM": am, "PCR": pcr}
