"""Canonical Curriculum Controller v13.3 (Gated STOP).

Final consolidation:
  1. Gated STOP: warm-up (T_min per-θ) + plateau (Δu window) + M_base margin
  2. per-θ STOP coefficients (a_ν, b_γ, c_u, d_B)
  3. Full EVAL, family prior, close-gap removed, G_hier/G_res retained
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .lesson_library_v2 import LessonV2, LESSON_CATALOG_V2, PROBE_NAMES
from .mastery_model import MasteryModel
from .pairwise_response_model import PairwiseResponseModel
from .risk_budget_calibration import AdaptiveRiskBudget
from .family_prior import FamilyPrior
from ..agents.internalization_state_v3 import FactoredInternalizationState
from ..agents.trainable_bridge import TrainableBridge

W_DEFICIT = np.array([1.0, 1.2, 2.5, 1.5, 2.5])
PROBE_TARGETS = {
    "safe":  {"RC": 0.70, "TR": 0.65, "EP": 0.55, "VA": 0.70, "IA": 0.25},
    "shiny": {"RC": 0.70, "TR": 0.70, "EP": 0.50, "VA": 0.70, "IA": 0.25},
}


@dataclass
class ControllerV13Config:
    """Configuration for canonical curriculum controller."""
    # Gain
    w_gain: float = 2.0
    lambda_unc_base: float = 1.2
    lambda_fid: float = 0.6
    lambda_rep: float = 1.0

    # Double-decay exploration (from v12)
    tau_n: float = 12.0
    tau_B: float = 0.3

    # B1: EIG hybrid exploration (ablation, default OFF)
    eig_beta: float = 3.0        # sensitivity to epistemic value
    eig_c_gain: float = 1.0      # weight on gain variance in V_epi
    eig_c_mastery: float = 0.5   # weight on mastery variance in V_epi
    eig_mix: float = 0.5         # ω_eig: blend factor (0=pure count, 1=pure EIG)

    # B3: ZPD feature augmentation (ablation, default OFF)
    alpha_zpd: float = 0.10      # under-mastery penalty weight (reduced from 0.3)
    beta_zpd: float = 0.05       # over-mastery penalty weight (reduced from 0.15)

    # STOP: per-θ learner-conditional coefficients (v13.3)
    # safe: standard penalties (per-θ intercept confirmed +9pp in 6.7)
    eps_0_safe: float = 0.00
    a_s_nu_safe: float = 0.04
    b_s_gg_safe: float = 0.05
    c_s_mastery_safe: float = 0.03
    d_s_budget_safe: float = -0.02
    # shiny: near-zero ν/γ penalties (prevent premature stop at #T=2)
    eps_0_shiny: float = -0.10
    a_s_nu_shiny: float = 0.005   # ν pushes threshold up 8x less for shiny
    b_s_gg_shiny: float = 0.005   # γ pushes threshold up 10x less for shiny
    c_s_mastery_shiny: float = 0.03
    d_s_budget_shiny: float = -0.02
    # Gated STOP: warm-up + plateau
    min_teach_safe: int = 2    # T_min for safe
    min_teach_shiny: int = 3   # T_min for shiny (needs more exposure)
    min_teach_before_stop: int = 3  # legacy fallback
    plateau_window: int = 2    # sliding window for Δu
    plateau_tau_safe: float = 0.02   # mastery change threshold for safe
    plateau_tau_shiny: float = 0.015 # mastery change threshold for shiny
    # Legacy shared coefficients
    eps_0: float = -0.05
    a_s_nu: float = 0.04
    b_s_gg: float = 0.05
    c_s_mastery: float = 0.03
    d_s_budget: float = -0.02

    # EVAL: θ-conditional information action
    lambda_info: float = 0.8     # mastery uncertainty value
    lambda_var_gain: float = 0.4 # response model uncertainty value
    c_eval: float = 0.3          # EVAL cost penalty
    max_eval: int = 3
    probe_only_threshold: float = 0.6  # if q(shiny) > this, use probe-only mode

    # Risk constraints (per-constraint filter)
    eta_otr_0: float = 0.55
    eta_nu_0: float = 0.40
    eta_gg_0: float = 0.25
    beta_pessimism: float = 0.5
    constraint_warmup: int = 2

    # Feasibility
    feas_threshold: float = 0.3  # minimum feasibility to enter ranking
    max_family_recent: int = 3   # max lessons from same family in recent 5

    # Budget
    total_budget: float = 4.0

    # Risk budget mode: "fixed", "theta", "full"
    risk_budget_mode: str = "fixed"

    # Counterfactual replay (from v12)
    replay_top_k: int = 4
    w_C: float = 1.0
    w_otr: float = -0.5


@dataclass
class CurriculumControllerV13:
    """Canonical curriculum controller v13.1 with unified STOP/EVAL/TEACH."""
    cfg: ControllerV13Config = None
    bridge: TrainableBridge = None
    mastery: MasteryModel = None
    response: PairwiseResponseModel = None
    family_prior: FamilyPrior = None
    theta: str = "safe"

    # State
    history: List[str] = field(default_factory=list)
    lesson_counts: dict = field(default_factory=dict)
    realized_subtypes: list = field(default_factory=list)
    stopped: bool = False
    eval_count: int = 0
    budget_blocked_count: int = 0
    spent_budget: float = 0.0
    dose_spent: float = 0.0

    # Mastery history for plateau detection
    _mastery_history: list = field(default_factory=list)

    # Tracing
    _term_scores: list = field(default_factory=list)
    _controller_trace: list = field(default_factory=list)
    _last_state: dict = field(default_factory=dict)

    # Ablation flags
    use_constraint: bool = True
    use_uncertainty: bool = True
    use_stop: bool = True
    use_eval: bool = True
    use_prerequisite: bool = True
    use_replay: bool = True
    use_hier: bool = True
    use_res: bool = True
    use_close_gap: bool = False
    use_family_prior: bool = True
    use_warm_gate: bool = True    # gated STOP: warm-up
    use_plateau_gate: bool = True # gated STOP: plateau
    use_eig_uncertainty: bool = False   # B1: hybrid EIG exploration
    use_zpd_feature: bool = False       # B3: ZPD feature augmentation

    def __post_init__(self):
        if self.cfg is None: self.cfg = ControllerV13Config()
        if self.bridge is None: self.bridge = TrainableBridge()
        if self.mastery is None: self.mastery = MasteryModel()
        if self.family_prior is None: self.family_prior = FamilyPrior()
        if self.response is None:
            cat = [l.name for l in LESSON_CATALOG_V2]
            self.response = PairwiseResponseModel(catalog_names=cat, theta=self.theta)
        self._profile_hook = None  # Optional profile-aware scoring hook

    def reset_session(self, budget: float = None):
        c = self.cfg
        self.mastery = MasteryModel(); self.bridge = TrainableBridge()
        self.history = []; self.lesson_counts = {}
        self.realized_subtypes = []; self.stopped = False
        self.eval_count = 0; self.budget_blocked_count = 0
        self.spent_budget = 0.0; self.dose_spent = 0.0
        self._mastery_history = []
        self._term_scores = []; self._controller_trace = []
        self._last_state = {}
        # NOTE: _profile_hook is intentionally NOT reset — it persists
        # across sessions for the same learner
        if budget is not None:
            self.cfg.total_budget = budget

    def install_profile_hook(self, hook_fn):
        """Install profile-aware scoring hook. Preserved across reset_session.

        Hook signature: hook_fn(lesson: LessonV2, base_J: float, mastery: dict) -> float
        Returns additive score adjustment (positive = boost this lesson).
        """
        self._profile_hook = hook_fn

    def remove_profile_hook(self):
        """Remove profile-aware scoring hook."""
        self._profile_hook = None

    @property
    def remaining_budget(self):
        return max(self.cfg.total_budget - self.spent_budget - self.dose_spent, 0.0)

    @property
    def n_teach(self):
        return sum(1 for h in self.history if h not in ("EVAL", "STOP"))

    # ═══════════════════════════════════════════════════
    # Layer 1: Feasibility filter
    # ═══════════════════════════════════════════════════

    def _feasible_set(self, u: dict) -> list[LessonV2]:
        """Three-layer feasibility: prerequisite + dose + budget."""
        c = self.cfg; feasible = []
        recent_5 = self.history[-5:] if len(self.history) >= 5 else self.history

        for lesson in LESSON_CATALOG_V2:
            # Budget feasibility
            if lesson.cost > self.remaining_budget + 0.01:
                self.budget_blocked_count += 1; continue

            # Prerequisite feasibility
            if self.use_prerequisite:
                feas = lesson.feasibility(u)
                if feas < c.feas_threshold: continue

            # Dose / family feasibility
            family_count = sum(1 for h in recent_5
                if h in LESSON_CATALOG_V2 and
                any(l.name == h and l.family == lesson.family for l in LESSON_CATALOG_V2))
            fam_cnt = sum(1 for h in recent_5
                if any(l.name == h and l.family == lesson.family for l in LESSON_CATALOG_V2))
            if fam_cnt >= c.max_family_recent: continue

            feasible.append(lesson)
        return feasible

    # ═══════════════════════════════════════════════════
    # Layer 2: Risk constraint filter
    # ═══════════════════════════════════════════════════

    def _risk_budgets(self, m: FactoredInternalizationState, u: dict) -> dict:
        arb = AdaptiveRiskBudget(mode=self.cfg.risk_budget_mode)
        return arb.compute(self.theta, m, u)

    def _passes_risk(self, lesson: LessonV2, u: dict, budgets: dict, m) -> bool:
        if not self.use_constraint: return True
        if self.n_teach < self.cfg.constraint_warmup: return True
        hp = self.response.predict_harm(
            lesson.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
            self.remaining_budget, self.lesson_counts, lesson.severity, 0.3)
        h_upper = hp["mean"] + self.cfg.beta_pessimism * np.sqrt(max(hp["var"], 1e-8))
        for dim, budget in budgets.items():
            if h_upper > budget: return False
        return True

    # ═══════════════════════════════════════════════════
    # Layer 3: Scoring
    # ═══════════════════════════════════════════════════

    def _lambda_unc(self) -> float:
        if not self.use_uncertainty: return 0.0
        c = self.cfg
        x = (self.remaining_budget - c.total_budget * 0.6) / max(c.total_budget * 0.6 * c.tau_B, 0.1)
        lam_budget = c.lambda_unc_base / (1.0 + np.exp(-np.clip(x, -10, 10)))
        n_post = self.response.pw_gain._n if self.response.pw_gain._n > 0 else 0
        decay = np.exp(-n_post / c.tau_n)
        return float(lam_budget * decay)

    def _lambda_unc_hybrid(self, gp: dict, lesson: LessonV2 = None) -> tuple[float, dict]:
        """B1: Hybrid EIG exploration gate (lesson-sensitive).

        Blends count-based decay with belief-based V_epi:
          λ_eff = λ_0 · σ(budget) · [(1-ω)·g_count + ω·g_eig]

        V_epi uses lesson.zpd_mask to weight mastery variance by
        which dimensions the lesson actually targets.
        """
        if not self.use_uncertainty:
            return 0.0, {"g_count": 0, "g_eig": 0, "V_epi": 0}
        c = self.cfg
        # Budget sigmoid (unchanged)
        x = (self.remaining_budget - c.total_budget * 0.6) / max(c.total_budget * 0.6 * c.tau_B, 0.1)
        lam_budget = c.lambda_unc_base / (1.0 + np.exp(-np.clip(x, -10, 10)))
        # Count-based decay (canonical)
        n_post = self.response.pw_gain._n if self.response.pw_gain._n > 0 else 0
        g_count = float(np.exp(-n_post / c.tau_n))
        # Belief-based EIG (lesson-sensitive)
        unc = self.mastery.uncertainty()
        if lesson is not None and hasattr(lesson, 'zpd_mask') and lesson.zpd_mask is not None:
            # Weight mastery variance by which dimensions this lesson targets
            mask = lesson.zpd_mask
            mastery_var = sum(
                float(mask[i]) * unc.get(p, 0.1)
                for i, p in enumerate(PROBE_NAMES)
            )
        else:
            mastery_var = sum(unc.values())
        gain_var = max(gp.get("var", 0.01), 1e-8)
        V_epi = c.eig_c_gain * gain_var + c.eig_c_mastery * mastery_var
        g_eig = float(1.0 - np.exp(-c.eig_beta * V_epi))
        # Hybrid blend
        w = c.eig_mix
        decay = (1.0 - w) * g_count + w * g_eig
        lam_eff = float(lam_budget * decay)
        trace = {
            "g_count": round(g_count, 6),
            "g_eig": round(g_eig, 6),
            "V_epi": round(V_epi, 6),
            "eig_mix": w,
            "lam_budget": round(lam_budget, 6),
            "lam_unc_hybrid": round(lam_eff, 6),
        }
        return lam_eff, trace

    def _zpd_adjustment(self, lesson: LessonV2, u: dict) -> tuple[float, dict]:
        """B3: ZPD mismatch penalty (both terms ≤ 0).

        Uses lesson.zpd_target (mastery demand) and lesson.zpd_mask
        (which dimensions matter) instead of lesson.gain.

        ψ_under = -||mask ⊙ ReLU(d_ℓ - u)||²_Ω  (lesson too hard)
        ψ_over  = -||mask ⊙ ReLU(u - d_ℓ)||²_Ω  (lesson too easy)
        """
        if not self.use_zpd_feature:
            return 0.0, {}
        c = self.cfg
        d_ell = lesson.zpd_target  # Semantically correct: mastery demand vector
        mask = lesson.zpd_mask      # Which dimensions this lesson targets
        u_vec = np.array([u.get(p, 0.5) for p in PROBE_NAMES])
        # Ω^{-1} diagonal weights (matches W_DEFICIT)
        omega_inv = W_DEFICIT / W_DEFICIT.sum()
        # Apply mask: only penalize mismatch on targeted dimensions
        diff_under = mask * np.maximum(d_ell - u_vec, 0.0)  # too hard: d > u
        diff_over = mask * np.maximum(u_vec - d_ell, 0.0)    # too easy: u > d
        psi_under = -float(np.sum(omega_inv * diff_under ** 2))
        psi_over = -float(np.sum(omega_inv * diff_over ** 2))
        adj = c.alpha_zpd * psi_under + c.beta_zpd * psi_over
        trace = {
            "zpd_under": round(psi_under, 6),
            "zpd_over": round(psi_over, 6),
            "zpd_adj": round(adj, 6),
        }
        return float(adj), trace

    def _score_lesson(self, lesson: LessonV2, u: dict, m) -> tuple[float, dict]:
        c = self.cfg
        gp = self.response.predict_gain(
            lesson.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
            self.remaining_budget, self.lesson_counts, lesson.severity, 0.3)
        hp = self.response.predict_harm(
            lesson.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
            self.remaining_budget, self.lesson_counts, lesson.severity, 0.3)
        # Ablation: zero out hier/res if disabled
        g_mean = gp["mean"]
        if not self.use_hier:
            g_mean = g_mean - gp.get("hier", 0)
        if not self.use_res:
            g_mean = g_mean - 0.5 * gp.get("res", 0)
        G = c.w_gain * g_mean
        # B1: Use hybrid EIG if enabled, else canonical count-based
        eig_trace = {}
        if self.use_eig_uncertainty:
            lam_unc, eig_trace = self._lambda_unc_hybrid(gp, lesson)
            U = lam_unc * np.sqrt(max(gp["var"], 1e-8))
        else:
            U = self._lambda_unc() * np.sqrt(gp["var"])
        recent = self.history[-5:] if len(self.history) >= 5 else self.history
        r_rep = c.lambda_rep * (sum(1 for h in recent if h == lesson.name) / max(len(recent), 1)) if recent else 0
        r_fid = c.lambda_fid * 0.08
        # B3: ZPD adjustment
        zpd_adj, zpd_trace = self._zpd_adjustment(lesson, u)
        J = float(G + U - r_fid - r_rep + zpd_adj)
        # Profile-aware need hook (Task 2B)
        profile_adj = 0.0
        if self._profile_hook is not None:
            profile_adj = self._profile_hook(lesson, J, u)
            J += profile_adj
        terms = {
            "G": round(G, 4), "G_hier": round(gp.get("hier", 0), 4),
            "G_res": round(gp.get("res", 0), 4), "G_pw": round(gp.get("pw", 0), 4),
            "U": round(U, 4), "H": round(hp["mean"], 4),
            "r_fid": round(r_fid, 4), "r_rep": round(r_rep, 4),
            "profile_need": round(profile_adj, 4),
            "phi": gp.get("phi"),
            **eig_trace, **zpd_trace,
        }
        return J, terms

    # ═══════════════════════════════════════════════════
    # Layer 4: STOP value (learner-conditional)
    # ═══════════════════════════════════════════════════

    def _stop_threshold(self, m, u: dict) -> float:
        c = self.cfg
        mastery_sum = sum(W_DEFICIT[i] * u.get(p, 0.5) for i, p in enumerate(PROBE_NAMES)) / sum(W_DEFICIT)
        # v13.3: per-θ coefficients (not just intercept)
        if self.theta == "shiny":
            return float(
                c.eps_0_shiny
                + c.a_s_nu_shiny * m.nu
                + c.b_s_gg_shiny * m.gamma_gen
                + c.c_s_mastery_shiny * mastery_sum
                + c.d_s_budget_shiny * self.remaining_budget
            )
        else:
            return float(
                c.eps_0_safe
                + c.a_s_nu_safe * m.nu
                + c.b_s_gg_safe * m.gamma_gen
                + c.c_s_mastery_safe * mastery_sum
                + c.d_s_budget_safe * self.remaining_budget
            )

    # ═══════════════════════════════════════════════════
    # Layer 5: EVAL value (information action)
    # ═══════════════════════════════════════════════════

    def _eval_value(self, m, u: dict, best_J: float, second_J: float) -> float:
        c = self.cfg
        if not self.use_eval: return -1e9
        if self.eval_count >= c.max_eval: return -1e9
        if self.n_teach < 1: return -1e9

        # v13.2: full EVAL for all θ (probe-only reverted in 6.6)
        unc = sum(self.mastery.uncertainty().values())
        rv = 0.0
        for l in LESSON_CATALOG_V2[:4]:
            gp = self.response.predict_gain(l.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
                                            self.remaining_budget, self.lesson_counts, l.severity, 0.3)
            rv += gp["var"]
        J_eval = c.lambda_info * (unc + 0.5 * np.sqrt(rv)) - c.c_eval
        return float(J_eval)

    # ═══════════════════════════════════════════════════
    # Main decision
    # ═══════════════════════════════════════════════════

    def select_action(self, m, candidates=None):
        if self.stopped: return "STOP", None, 0.0, {"reason": "already_stopped"}
        c = self.cfg
        u = self.mastery.mastery()

        # Layer 1: Feasibility
        feasible = self._feasible_set(u)

        # Layer 2: Risk constraint
        budgets = self._risk_budgets(m, u)
        risk_ok = []
        all_terms = {}
        for lesson in feasible:
            if self._passes_risk(lesson, u, budgets, m):
                risk_ok.append(lesson)
            else:
                J, terms = self._score_lesson(lesson, u, m)
                terms["filtered"] = "risk"; all_terms[lesson.name] = terms

        # Layer 3: Score remaining
        best = None; best_J = -1e9; best_terms = {}
        second_J = -1e9
        # Compute family-level usage counts for saturation
        fam_counts = {}
        for ln, cnt in self.lesson_counts.items():
            les = next((l for l in LESSON_CATALOG_V2 if l.name == ln), None)
            if les and hasattr(les, 'family'):
                fam_counts[les.family] = fam_counts.get(les.family, 0) + cnt
        for lesson in risk_ok:
            J, terms = self._score_lesson(lesson, u, m)
            # Family prior bonus with saturation
            if self.use_family_prior and self.family_prior is not None:
                fb = self.family_prior.bonus(lesson, self.theta, family_counts=fam_counts)
                J += fb
                terms["fam_bonus"] = round(fb, 4)
            terms["filtered"] = False; all_terms[lesson.name] = terms
            if J > best_J:
                second_J = best_J
                best_J = J; best = lesson; best_terms = terms
            elif J > second_J:
                second_J = J

        self._term_scores.append(all_terms)

        if not risk_ok or best is None:
            self.stopped = True
            self._log_trace("STOP", None, 0, {"reason": "all_filtered"})
            return "STOP", None, 0.0, {"reason": "all_filtered"}

        # Layer 4: STOP decision (gated)
        eps = self._stop_threshold(m, u)
        stop_margin = eps - best_J
        m_base = best_J < eps  # M_base > 0 means STOP candidate

        # Warm-up gate: don't STOP before T_min lessons
        t_min = c.min_teach_shiny if self.theta == "shiny" else c.min_teach_safe
        g_warm = self.n_teach >= t_min if self.use_warm_gate else self.n_teach >= c.min_teach_before_stop

        # Plateau gate: only STOP if mastery has plateaued
        g_plateau = True  # default: allow
        delta_u = None
        if self.use_plateau_gate and len(self._mastery_history) >= 2:
            w = min(c.plateau_window, len(self._mastery_history) - 1)
            deltas = []
            for i in range(-w, 0):
                prev = self._mastery_history[i - 1] if abs(i - 1) <= len(self._mastery_history) else {}
                curr = self._mastery_history[i]
                d = sum(abs(curr.get(p, 0.5) - prev.get(p, 0.5)) for p in PROBE_NAMES)
                deltas.append(d)
            delta_u = np.mean(deltas) if deltas else 0.0
            tau_u = c.plateau_tau_shiny if self.theta == "shiny" else c.plateau_tau_safe
            g_plateau = delta_u <= tau_u

        gate_info = {
            "m_base": m_base, "g_warm": g_warm, "g_plateau": g_plateau,
            "delta_u": round(delta_u, 6) if delta_u is not None else None,
            "t_min": t_min, "n_teach": self.n_teach,
        }

        if self.use_stop and m_base and g_warm and g_plateau:
            self.stopped = True
            self._log_trace("STOP", best, best_J, {
                "reason": "gated", "threshold": round(eps, 4),
                "margin": round(stop_margin, 4), "best_lesson": best.name,
                "counterfactual_J": round(best_J, 4), **gate_info,
            })
            return "STOP", None, round(best_J, 4), {"reason": "gated", "margin": round(stop_margin, 4), **gate_info}

        # Layer 5: EVAL competition
        J_eval = self._eval_value(m, u, best_J, second_J)
        if J_eval > best_J:
            self.eval_count += 1; self.history.append("EVAL")
            self._log_trace("EVAL", None, J_eval, {
                "J_eval": round(J_eval, 4), "J_best_lesson": round(best_J, 4),
                "delta_12": round(abs(best_J - second_J), 4),
            })
            return "EVAL", None, round(J_eval, 4), {"eval_vs_teach": round(J_eval - best_J, 4)}

        # Layer 6: TEACH
        self._last_state = {"lesson": best, "u": dict(u),
                            "m_snap": (m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa)}
        self.history.append(best.name)
        self.lesson_counts[best.name] = self.lesson_counts.get(best.name, 0) + 1
        self.spent_budget += best.cost
        self._log_trace("TEACH", best, best_J, {
            "n_feasible": len(feasible), "n_risk_ok": len(risk_ok),
            "lambda_unc": round(self._lambda_unc(), 3),
            "n_pw": self.response.pw_gain._n,
            "stop_margin": round(stop_margin, 4),
        })
        return "TEACH", best, round(best_J, 4), {
            "remaining": round(self.remaining_budget, 2),
            "n_feasible": len(feasible), "n_risk_ok": len(risk_ok),
            "lambda_unc": round(self._lambda_unc(), 3),
            "n_pw": self.response.pw_gain._n,
        }

    def _log_trace(self, action, lesson, J, info):
        self._controller_trace.append({
            "step": len(self._controller_trace),
            "action": action,
            "lesson": lesson.name if lesson else None,
            "J": round(J, 4) if J else 0,
            "n_teach": self.n_teach,
            **{k: v for k, v in info.items()},
        })

    # ═══════════════════════════════════════════════════
    # Update API
    # ═══════════════════════════════════════════════════

    def consume_dose(self, d): self.dose_spent += d
    def record_realization(self, ep): self.realized_subtypes.append(ep.subtype)
    def update_mastery(self, probes):
        self.mastery.update(probes)
        self._mastery_history.append(dict(self.mastery.mastery()))

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

        # Active counterfactual replay (from v12)
        if self._last_state and les and self.use_replay:
            gain = sum(mastery_after.get(p, 0.5) - mastery_before.get(p, 0.5) for p in PROBE_NAMES) / len(PROBE_NAMES)
            harm = (otr_after - otr_before) + 0.5 * (nu_after - nu_before) + 0.5 * (gg_after - gg_before)
            m_snap = FactoredInternalizationState()
            snap = self._last_state.get("m_snap", (0.1, 0.1, 0.1, 0.1, 0.5))
            m_snap.nu, m_snap.tau, m_snap.gamma_gen, m_snap.gamma_spec, m_snap.kappa = snap
            self._active_replay(les, gain, harm, u, m_snap)

    def _active_replay(self, chosen_lesson, chosen_gain, chosen_harm, u, m):
        c = self.cfg
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
        candidates.sort(key=lambda x: x[1]["mean"], reverse=True)
        top_k = candidates[:c.replay_top_k]
        if not top_k: return
        chosen_phi = self.response._phi(chosen_lesson.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
                                         self.remaining_budget, self.lesson_counts, chosen_lesson.severity, 0.3)
        U_ch = c.w_C * chosen_gain + c.w_otr * max(chosen_harm, 0)
        phis = [chosen_phi]; gains_surr = [U_ch]; harms_surr = [chosen_harm]
        for les, gp, hp in top_k:
            phi = gp.get("phi")
            if phi is None:
                phi = self.response._phi(les.name, u, m.nu, m.tau, m.gamma_gen, m.gamma_spec, m.kappa,
                                          self.remaining_budget, self.lesson_counts, les.severity, 0.3)
            phis.append(phi); gains_surr.append(c.w_C * gp["mean"] + c.w_otr * max(hp["mean"], 0))
            harms_surr.append(hp["mean"])
        self.response.counterfactual_replay(chosen_phi, phis, gains_surr, harms_surr)

    # ═══════════════════════════════════════════════════
    # Diagnostics
    # ═══════════════════════════════════════════════════

    def posterior_stats(self):
        return {
            "n_updated": self.response.n_updated(),
            "n_pw_gain": self.response.pw_gain._n,
            "n_pw_harm": self.response.pw_harm._n,
        }

    def actionability_audit(self):
        tns = ["G", "G_hier", "G_res", "G_pw", "U", "H"]
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

    def controller_summary(self):
        actions = [t["action"] for t in self._controller_trace]
        return {
            "n_teach": actions.count("TEACH"),
            "n_eval": actions.count("EVAL"),
            "n_stop": actions.count("STOP"),
            "stop_reason": self._controller_trace[-1].get("reason") if self._controller_trace and self._controller_trace[-1]["action"] == "STOP" else None,
            "stop_margin": self._controller_trace[-1].get("margin") if self._controller_trace and self._controller_trace[-1]["action"] == "STOP" else None,
            "trace": self._controller_trace,
        }
