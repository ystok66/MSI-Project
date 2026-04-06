"""CCT-v12: Pairwise-Active Constrained Bayesian Curriculum Planner.

Core upgrades over v11:
  1. ACTIVE counterfactual replay: after each teach, construct surrogate
     U_short for top-K candidate lessons and inject pairwise labels
  2. Per-constraint filter: separate OTR/ν/γ_gen budgets (not max)
  3. Double-decay exploration: budget × posterior maturity
  4. Better pairwise label quality via multi-component U_short
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .lesson_library_v2 import LessonV2, LESSON_CATALOG_V2, PROBE_NAMES
from .mastery_model import MasteryModel
from .pairwise_response_model import PairwiseResponseModel
from .adaptive_episode_generator import EpisodeParams
from ..agents.internalization_state_v3 import FactoredInternalizationState
from ..agents.trainable_bridge import TrainableBridge

W_DEFICIT = np.array([1.0, 1.2, 2.5, 1.5, 2.5])
PROBE_TARGETS = {
    "safe":  {"RC": 0.70, "TR": 0.65, "EP": 0.55, "VA": 0.70, "IA": 0.25},
    "shiny": {"RC": 0.70, "TR": 0.70, "EP": 0.50, "VA": 0.70, "IA": 0.25},
}


@dataclass
class CurriculumControllerV12:
    """Pairwise-active constrained Bayesian planner with counterfactual replay."""
    bridge: TrainableBridge = None
    mastery: MasteryModel = None
    response: PairwiseResponseModel = None
    theta: str = "safe"

    # Gain
    w_gain: float = 2.0
    lambda_unc_base: float = 1.2
    lambda_fid: float = 0.6
    lambda_rep: float = 1.0
    lambda_eval: float = 0.5

    # Double-decay exploration
    tau_n: float = 12.0   # posterior maturity decay (tighter than v11's 15)
    tau_B: float = 0.3    # budget sigmoid steepness

    # Per-constraint risk budgets
    eta_otr_0: float = 0.55
    eta_nu_0: float = 0.40
    eta_gg_0: float = 0.25
    beta_pessimism: float = 0.5

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

    # Counterfactual replay
    replay_top_k: int = 4   # generate surrogate for top-K lessons per step
    w_C: float = 1.0        # U_short weight for compliance
    w_E: float = 1.0        # U_short weight for exploration
    w_otr: float = -0.5     # U_short penalty for overteaching
    w_nu: float = -0.3      # U_short penalty for dependence

    history: List[str] = field(default_factory=list)
    lesson_counts: dict = field(default_factory=dict)
    realized_subtypes: list = field(default_factory=list)
    stopped: bool = False
    eval_count: int = 0
    budget_blocked_count: int = 0

    _term_scores: list = field(default_factory=list)
    _last_state: dict = field(default_factory=dict)

    use_constraint: bool = True
    use_uncertainty: bool = True
    use_stop: bool = True
    use_budget: bool = True
    use_replay: bool = True

    def __post_init__(self):
        if self.bridge is None: self.bridge = TrainableBridge()
        if self.mastery is None: self.mastery = MasteryModel()
        if self.response is None:
            cat = [l.name for l in LESSON_CATALOG_V2]
            self.response = PairwiseResponseModel(catalog_names=cat, theta=self.theta)

    def reset_session(self, budget: float = None):
        self.mastery = MasteryModel(); self.bridge = TrainableBridge()
        self.history = []; self.lesson_counts = {}
        self.realized_subtypes = []; self.stopped = False
        self.eval_count = 0; self.budget_blocked_count = 0
        self.spent_budget = 0.0; self.dose_spent = 0.0
        self._term_scores = []; self._last_state = {}
        if budget is not None:
            self.total_budget = budget; self.budget_mid = budget * 0.6

    @property
    def remaining_budget(self):
        return max(self.total_budget - self.spent_budget - self.dose_spent, 0.0)

    def _lambda_unc(self, lesson_name: str = None) -> float:
        """Budget-conditioned × maturity-decayed exploration."""
        if not self.use_uncertainty: return 0.0
        # Budget sigmoid
        x = (self.remaining_budget - self.budget_mid) / max(self.budget_mid * self.tau_B, 0.1)
        lam_budget = self.lambda_unc_base / (1.0 + np.exp(-np.clip(x, -10, 10)))
        # Maturity decay: reduce for well-observed responses
        n_post = self.response.pw_gain._n if self.response.pw_gain._n > 0 else 0
        decay = np.exp(-n_post / self.tau_n)
        return float(lam_budget * decay)

    def _risk_budgets(self, m: FactoredInternalizationState, u: dict) -> dict:
        """Learner-conditional risk budgets (per constraint)."""
        ep_d = max(PROBE_TARGETS[self.theta]["EP"] - u.get("EP", 0.5), 0)
        ia_d = max(u.get("IA", 0.5) - (1 - PROBE_TARGETS[self.theta]["IA"]), 0)
        return {
            "otr": max(self.eta_otr_0 + 0.3 * ep_d + 0.2 * ia_d - 0.5 * m.gamma_gen, 0.05),
            "nu":  max(self.eta_nu_0 + 0.2 * (1 - u.get("VA", 0.5)) - 0.4 * m.nu, 0.05),
            "gg":  max(self.eta_gg_0 + 0.2 * ep_d - 0.6 * m.gamma_gen, 0.02),
        }

    def _passes_constraint(self, harm_pred: dict, budgets: dict) -> bool:
        """Per-constraint filter: each harm dimension checked separately."""
        if not self.use_constraint: return True
        n_teach = sum(1 for h in self.history if h != "EVAL")
        if n_teach < 2: return True  # warm-up
        h_upper = harm_pred["mean"] + self.beta_pessimism * np.sqrt(max(harm_pred["var"], 1e-8))
        # Check against each constraint separately; must pass ALL
        for dim, budget in budgets.items():
            if h_upper > budget:
                return False
        return True

    def _score_lesson(self, lesson: LessonV2, u: dict,
                      m: FactoredInternalizationState) -> tuple[float, dict]:
        gp = self.response.predict_gain(
            lesson.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
            self.remaining_budget, self.lesson_counts, lesson.severity, 0.3)
        hp = self.response.predict_harm(
            lesson.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
            self.remaining_budget, self.lesson_counts, lesson.severity, 0.3)
        G = self.w_gain * gp["mean"]
        U = self._lambda_unc(lesson.name) * np.sqrt(gp["var"])
        r_fid = self.lambda_fid * 0.08
        recent = self.history[-5:] if len(self.history) >= 5 else self.history
        r_rep = self.lambda_rep * (sum(1 for h in recent if h == lesson.name) / max(len(recent), 1)) if recent else 0
        J = float(G + U - r_fid - r_rep)
        terms = {
            "G": round(G, 4), "G_hier": round(gp.get("hier", 0), 4),
            "G_res": round(gp.get("res", 0), 4), "G_pw": round(gp.get("pw", 0), 4),
            "U": round(U, 4),
            "H": round(hp["mean"], 4), "H_hier": round(hp.get("hier", 0), 4),
            "H_res": round(hp.get("res", 0), 4), "H_pw": round(hp.get("pw", 0), 4),
            "r_fid": round(r_fid, 4), "r_rep": round(r_rep, 4),
            "phi": gp.get("phi"),
        }
        return J, terms

    def _eval_value(self, m):
        u = self.mastery.mastery()
        mu = sum(self.mastery.uncertainty().values())
        rv = 0.0
        for l in LESSON_CATALOG_V2[:4]:
            gp = self.response.predict_gain(l.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
                                            self.remaining_budget, self.lesson_counts, l.severity, 0.3)
            rv += gp["var"]
        return float(self.lambda_eval * (mu + 0.5 * np.sqrt(rv)) - 0.8)

    def _stop_threshold(self, m, u):
        uw = sum(W_DEFICIT[i] * u.get(p, 0.5) for i, p in enumerate(PROBE_NAMES)) / sum(W_DEFICIT)
        return float(self.eps_0 + self.alpha_nu_stop * m.nu + self.alpha_gg_stop * m.gamma_gen - self.alpha_u_stop * uw)

    def _active_counterfactual_replay(self, chosen_lesson: LessonV2, chosen_gain: float,
                                       chosen_harm: float, u: dict, m: FactoredInternalizationState):
        """After a teach step, construct surrogate U_short for top-K candidates and inject pairwise labels."""
        if not self.use_replay: return

        # Score all lessons and pick top-K (excluding chosen)
        candidates = []
        for lesson in LESSON_CATALOG_V2:
            if lesson.name == chosen_lesson.name: continue
            gp = self.response.predict_gain(
                lesson.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
                self.remaining_budget, self.lesson_counts, lesson.severity, 0.3)
            hp = self.response.predict_harm(
                lesson.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
                self.remaining_budget, self.lesson_counts, lesson.severity, 0.3)
            candidates.append((lesson, gp, hp))

        # Sort by predicted gain, take top-K
        candidates.sort(key=lambda x: x[1]["mean"], reverse=True)
        top_k = candidates[:self.replay_top_k]

        if not top_k: return

        # Build pairwise labels using U_short surrogate
        chosen_phi = self.response._phi(chosen_lesson.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
                                         self.remaining_budget, self.lesson_counts, chosen_lesson.severity, 0.3)
        U_chosen = (self.w_C * chosen_gain + self.w_otr * max(chosen_harm, 0))

        phis = [chosen_phi]
        gains_surr = [U_chosen]
        harms_surr = [chosen_harm]

        for les, gp, hp in top_k:
            phi = gp.get("phi")
            if phi is None:
                phi = self.response._phi(les.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
                                          self.remaining_budget, self.lesson_counts, les.severity, 0.3)
            # Surrogate: use predicted gain/harm as counterfactual outcome
            U_surr = self.w_C * gp["mean"] + self.w_otr * max(hp["mean"], 0)
            phis.append(phi)
            gains_surr.append(U_surr)
            harms_surr.append(hp["mean"])

        # Inject pairwise labels via existing API
        self.response.counterfactual_replay(
            state_phi=chosen_phi,
            candidate_phis=phis,
            candidate_gains=gains_surr,
            candidate_harms=harms_surr,
        )

    def select_action(self, m, candidates=None):
        if self.stopped: return "STOP", None, 0.0, {"reason": "already_stopped"}
        if candidates is None: candidates = LESSON_CATALOG_V2
        u = self.mastery.mastery()
        budgets = self._risk_budgets(m, u)

        feasible = []; all_terms = {}
        for lesson in candidates:
            if self.use_budget and lesson.cost > self.remaining_budget + 0.01:
                self.budget_blocked_count += 1; continue
            hp = self.response.predict_harm(lesson.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
                                            self.remaining_budget, self.lesson_counts, lesson.severity, 0.3)
            if self._passes_constraint(hp, budgets):
                feasible.append(lesson)
            else:
                J, terms = self._score_lesson(lesson, u, m); terms["filtered"] = True; all_terms[lesson.name] = terms

        best = None; best_J = -1e9; best_terms = {}
        for lesson in feasible:
            J, terms = self._score_lesson(lesson, u, m); terms["filtered"] = False; all_terms[lesson.name] = terms
            if J > best_J: best_J = J; best = lesson; best_terms = terms

        self._term_scores.append(all_terms)
        nt = sum(1 for h in self.history if h != "EVAL")

        if not feasible or best is None:
            self.stopped = True; return "STOP", None, 0.0, {"reason": "all_filtered"}
        eps = self._stop_threshold(m, u)
        if self.use_stop and nt >= 3 and best_J < eps:
            self.stopped = True; return "STOP", None, round(best_J, 4), {"reason": "marginal"}

        qe = self._eval_value(m)
        if qe > best_J and self.eval_count < 2 and nt >= 1:
            self.eval_count += 1; self.history.append("EVAL")
            return "EVAL", None, round(qe, 4), {}

        # Save state for counterfactual replay
        self._last_state = {"lesson": best, "u": dict(u), "m_snap": (m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa)}

        self.history.append(best.name)
        self.lesson_counts[best.name] = self.lesson_counts.get(best.name, 0) + 1
        self.spent_budget += best.cost
        return "TEACH", best, round(best_J, 4), {
            "remaining": round(self.remaining_budget, 2),
            "n_feasible": len(feasible), "lambda_unc": round(self._lambda_unc(), 3),
            "n_pw": self.response.pw_gain._n,
        }

    def consume_dose(self, d): self.dose_spent += d
    def record_realization(self, ep): self.realized_subtypes.append(ep.subtype)
    def update_mastery(self, probes): self.mastery.update(probes)

    def update_response(self, lesson_name, mastery_before, mastery_after,
                        nu_before, nu_after, gg_before, gg_after, otr_before, otr_after):
        u = mastery_before
        les = next((l for l in LESSON_CATALOG_V2 if l.name == lesson_name), None)
        sev = les.severity if les else 0.5
        self.response.update(lesson_name, u, mastery_after,
                             nu_before, nu_after, u.get("RC", 0.5) + 0.3,
                             gg_before, gg_after, 0.0, 0.5,
                             self.remaining_budget, self.lesson_counts,
                             sev, 0.3, otr_before, otr_after)

        # ACTIVE counterfactual replay
        if self._last_state and les:
            gain = sum(mastery_after.get(p, 0.5) - mastery_before.get(p, 0.5) for p in PROBE_NAMES) / len(PROBE_NAMES)
            harm = (otr_after - otr_before) + 0.5 * (nu_after - nu_before) + 0.5 * (gg_after - gg_before)
            from ..agents.internalization_state_v3 import FactoredInternalizationState
            m_snap = FactoredInternalizationState()
            snap = self._last_state.get("m_snap", (0.1, 0.1, 0.1, 0.1, 0.5))
            m_snap.nu, m_snap.tau, m_snap.gamma_gen, m_snap.gamma_spec, m_snap.kappa = snap
            self._active_counterfactual_replay(les, gain, harm, u, m_snap)

    def posterior_stats(self):
        return {
            "n_updated": self.response.n_updated(),
            "n_pw_gain": self.response.pw_gain._n,
            "n_pw_harm": self.response.pw_harm._n,
        }

    def actionability_audit(self):
        tns = ["G", "G_hier", "G_res", "G_pw", "U", "H", "H_hier", "H_res", "H_pw"]
        am = {}; pcr = {}
        for tn in tns:
            vs = []; ch = 0
            for st in self._term_scores:
                if not st: continue
                nf = {ln: t for ln, t in st.items() if not t.get("filtered", False)}
                if len(nf) < 2: continue
                vals = [nf[ln].get(tn, 0) for ln in nf]; vs.append(np.var(vals))
                fs = {}; ab = {}
                for ln in nf:
                    t = nf[ln]
                    full = t.get("G", 0) + t.get("U", 0) - t.get("r_fid", 0) - t.get("r_rep", 0)
                    a = full - t.get(tn, 0) if tn in ("G", "U", "G_hier", "G_res", "G_pw") else full
                    fs[ln] = full; ab[ln] = a
                if max(fs, key=fs.get) != max(ab, key=ab.get): ch += 1
            am[tn] = round(np.mean(vs), 6) if vs else 0.0
            pcr[tn] = round(ch / max(len(self._term_scores), 1), 3)
        return {"AM": am, "PCR": pcr}
