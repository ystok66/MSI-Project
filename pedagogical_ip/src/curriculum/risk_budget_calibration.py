"""Risk budget calibration: θ-adaptive and mastery-adaptive budgets.

Replaces v13's fixed η budgets with posterior-weighted adaptive versions:
  η_j(x_t, q_t) = Σ_θ q_t(θ) · η_j^(θ)(x_t)

Three modes:
  fixed:   identical for all θ (v13 default)
  theta:   per-θ base budgets
  full:    per-θ base + mastery-adaptive coefficients
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


PROBE_TARGETS = {
    "safe":  {"RC": 0.70, "TR": 0.65, "EP": 0.55, "VA": 0.70, "IA": 0.25},
    "shiny": {"RC": 0.70, "TR": 0.70, "EP": 0.50, "VA": 0.70, "IA": 0.25},
}

# θ-specific budget parameters
# Key insight: shiny learners need WIDER budgets because their optimal
# lessons are inherently riskier (high temptation → higher predicted harm).
THETA_BUDGETS = {
    "safe": {
        "eta_otr_0": 0.55, "a_otr_ep": 0.3, "a_otr_ia": 0.2, "c_otr_gg": 0.5,
        "eta_nu_0":  0.40, "a_nu_va":  0.2, "c_nu_nu":  0.4,
        "eta_gg_0":  0.25, "a_gg_ep":  0.2, "c_gg_gg":  0.6,
    },
    "shiny": {
        # Wider: shiny needs more room to accept beneficial-risk lessons
        "eta_otr_0": 0.75, "a_otr_ep": 0.2, "a_otr_ia": 0.1, "c_otr_gg": 0.3,
        "eta_nu_0":  0.60, "a_nu_va":  0.15, "c_nu_nu":  0.3,
        "eta_gg_0":  0.40, "a_gg_ep":  0.15, "c_gg_gg":  0.4,
    },
}


@dataclass
class AdaptiveRiskBudget:
    """θ-adaptive risk budget calculator."""
    mode: str = "theta"  # "fixed", "theta", "full"

    def compute(self, theta: str, m, u: dict) -> dict:
        if self.mode == "fixed":
            return self._fixed(theta, m, u)
        elif self.mode == "theta":
            return self._theta_adaptive(theta, m, u)
        elif self.mode == "full":
            return self._full_adaptive(theta, m, u)
        return self._fixed(theta, m, u)

    def _fixed(self, theta, m, u):
        """v13 default: same for all θ."""
        t = PROBE_TARGETS[theta]
        ep_d = max(t["EP"] - u.get("EP", 0.5), 0)
        ia_d = max(u.get("IA", 0.5) - (1 - t["IA"]), 0)
        return {
            "otr": max(0.55 + 0.3 * ep_d + 0.2 * ia_d - 0.5 * m.gamma_gen, 0.05),
            "nu":  max(0.40 + 0.2 * (1 - u.get("VA", 0.5)) - 0.4 * m.nu, 0.05),
            "gg":  max(0.25 + 0.2 * ep_d - 0.6 * m.gamma_gen, 0.02),
        }

    def _theta_adaptive(self, theta, m, u):
        """Per-θ base budgets."""
        p = THETA_BUDGETS.get(theta, THETA_BUDGETS["safe"])
        t = PROBE_TARGETS[theta]
        ep_d = max(t["EP"] - u.get("EP", 0.5), 0)
        ia_d = max(u.get("IA", 0.5) - (1 - t["IA"]), 0)
        return {
            "otr": max(p["eta_otr_0"] + p["a_otr_ep"]*ep_d + p["a_otr_ia"]*ia_d - p["c_otr_gg"]*m.gamma_gen, 0.05),
            "nu":  max(p["eta_nu_0"]  + p["a_nu_va"]*(1 - u.get("VA", 0.5)) - p["c_nu_nu"]*m.nu, 0.05),
            "gg":  max(p["eta_gg_0"]  + p["a_gg_ep"]*ep_d - p["c_gg_gg"]*m.gamma_gen, 0.02),
        }

    def _full_adaptive(self, theta, m, u):
        """Per-θ + mastery-weighted: budgets widen as mastery improves."""
        base = self._theta_adaptive(theta, m, u)
        # Mastery bonus: higher mastery → less risk from teaching → widen budget
        mastery_avg = np.mean([u.get(p, 0.5) for p in ["RC", "TR", "EP", "VA"]])
        bonus = max(0, mastery_avg - 0.5) * 0.3  # up to +0.15 for high mastery
        return {k: v + bonus for k, v in base.items()}
