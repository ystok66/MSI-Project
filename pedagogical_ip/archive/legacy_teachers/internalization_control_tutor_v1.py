"""Internalization-Control Tutor v1 (ICT-v1).

Q_t(a) = Q_online(a) + λ_teach·V_teach(a) − λ_over·R_over(a)

V_teach(a) = L_teach(m_t) - E[L_teach(m_{t+1} | a)]
R_over(a)  = E[overteach_penalty(m_{t+1} | a)]

Extends CAJTv3 with teaching dynamics awareness.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..agents.stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, PREFERENCE_TYPES,
)
from ..agents.internalization_dynamics_v2 import InternalizationStateV2
from ..metrics.teaching_zone import teaching_loss
from ..metrics.overteaching import overteach_penalty
from ..metrics.self_discovery import estimate_self_discovery_prob, estimate_failure_if_wait
from ..envs.observation_mask import make_observation_mask
from ..agents.branch_summary import summarize_branch


def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))


@dataclass
class ICTv1:
    """Internalization-Control Tutor v1."""
    agent_params: AgentPolicyParams = None
    lambda_teach: float = 2.0
    lambda_over: float = 3.0

    # Internal state
    warn_count: int = 0
    wait_count: int = 0

    def __post_init__(self):
        if self.agent_params is None:
            self.agent_params = AgentPolicyParams()

    def _estimate_m_next(self, m: InternalizationStateV2, action: str,
                          tempt_score: float, risk_level: float) -> tuple:
        """Predict m_{t+1} under action."""
        mc = m.copy()
        if action == "WARN":
            mc.update_risk(0.05, 0.15)  # safe path assumed if warned
            mc.update_trust(warn_helpful=True)
            mc.update_suppression(temptation_error=False)
        else:
            # WAIT: agent explores, may hit risk
            mc.update_risk(risk_level, 0.15)
            mc.update_trust(warn_missed=(risk_level > 0.25))
            mc.update_suppression(
                temptation_error=(tempt_score > 0.5 and risk_level > 0.3))
        return mc.kappa, mc.eta, mc.gamma

    def decide(self, sc, fb, lp, lib, scorer, obs, m: InternalizationStateV2,
               theta_posterior=None):
        fv = np.full_like(fb, 0.3)
        dc = getattr(sc, 'commit_depth', obs + 1)
        dr = getattr(sc, 'reveal_depth', 3)
        delta = dc - dr
        p_self = estimate_self_discovery_prob(dc, dr)
        p_fail = estimate_failure_if_wait(dc, dr)

        fork = sc.fork_cell
        mask_a = make_observation_mask(sc.branch_a_cells, fork, obs)
        mask_b = make_observation_mask(sc.branch_b_cells, fork, obs)
        vis_a = [c for c, mm in zip(sc.branch_a_cells, mask_a) if mm > 0.5]
        vis_b = [c for c, mm in zip(sc.branch_b_cells, mask_b) if mm > 0.5]

        sa = summarize_branch(vis_a, fb, fv, lp)
        sb = summarize_branch(vis_b, fb, fv, lp)
        sa2 = summarize_branch(sc.branch_a_cells, fb, fv, lp)
        sb2 = summarize_branch(sc.branch_b_cells, fb, fv, lp)
        delta_s = max(abs(sa2[0] - sb2[0]) - abs(sa[0] - sb[0]), 0)
        dvoi = max(_sigmoid(abs(sa2[0] - sb2[0])) - _sigmoid(abs(sa[0] - sb[0])), 0)

        tempt_str = getattr(sc, 'temptation_strength', 0.0)
        risk_lvl = getattr(sc, 'risk_level', 0.3)

        # Online Q (same as CAJT-v3 structure)
        Q_online_warn = (1.0 * delta_s + 2.0 * dvoi + 1.5 * (1 - p_self)
                         + 1.0 * tempt_str - 0.05)
        Q_online_wait = (2.0 * p_self * delta_s - 1.5 * p_fail + 2.0)

        # Teaching value: L_teach(m_t) - E[L_teach(m_{t+1}|a)]
        theta_str = getattr(sc, 'latent_preference', 'safe')
        L_now = teaching_loss(m.kappa, m.eta, m.gamma, theta_str, theta_posterior)

        kw, ew, gw = self._estimate_m_next(m, "WARN", tempt_str, risk_lvl)
        L_warn = teaching_loss(kw, ew, gw, theta_str, theta_posterior)
        V_teach_warn = L_now - L_warn

        kn, en, gn = self._estimate_m_next(m, "WAIT", tempt_str, risk_lvl)
        L_wait = teaching_loss(kn, en, gn, theta_str, theta_posterior)
        V_teach_wait = L_now - L_wait

        # Overteach penalty
        R_over_warn = overteach_penalty(kw, ew, gw)
        R_over_wait = overteach_penalty(kn, en, gn)

        # Combined Q
        Q_warn = (Q_online_warn
                  + self.lambda_teach * V_teach_warn
                  - self.lambda_over * R_over_warn)
        Q_wait = (Q_online_wait
                  + self.lambda_teach * V_teach_wait
                  - self.lambda_over * R_over_wait)

        action = "WARN" if Q_warn > Q_wait else "WAIT"
        if action == "WARN":
            self.warn_count += 1
        else:
            self.wait_count += 1

        diag = {
            "Q_warn": round(Q_warn, 4), "Q_wait": round(Q_wait, 4),
            "V_teach_warn": round(V_teach_warn, 4),
            "V_teach_wait": round(V_teach_wait, 4),
            "R_over_warn": round(R_over_warn, 4),
            "R_over_wait": round(R_over_wait, 4),
            "L_now": round(L_now, 4),
        }
        return action, diag
