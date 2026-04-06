"""ICT-v2: State-Factored Internalization-Control Tutor.

Q(a) = Q_online(a) + λ_teach·V_teach(a) − λ_over·R_over(a)
V_teach(a) = ΔL_zone(a) + λ_sd·ΔSD(a) − λ_dep·ΔDEP(a)
R_over(a) = factored penalty on ν, γ_spec, γ_gen
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..agents.stochastic_agent_policy import AgentPolicyParams
from ..agents.internalization_state_v3 import FactoredInternalizationState
from ..metrics.teaching_zone_v2 import (
    teaching_loss_v2, path_sensitive_teaching_value, overteach_penalty_v2,
)
from ..metrics.self_discovery import estimate_self_discovery_prob, estimate_failure_if_wait
from ..envs.observation_mask import make_observation_mask
from ..agents.branch_summary import summarize_branch


def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))


@dataclass
class ICTv2:
    """State-Factored Internalization-Control Tutor."""
    agent_params: AgentPolicyParams = None
    lambda_teach: float = 2.5
    lambda_over: float = 3.5
    lambda_sd: float = 1.5
    lambda_dep: float = 2.0

    warn_count: int = 0
    wait_count: int = 0

    def __post_init__(self):
        if self.agent_params is None:
            self.agent_params = AgentPolicyParams()

    def _predict_m(self, m, action, tempt, risk, subtype, has_self_evidence):
        """Predict factored m_{t+1}."""
        mc = m.copy()
        is_verified = (subtype == "verified_warn")
        is_sd_needed = (subtype == "self_discovery_needed")
        is_false_supp = (subtype == "false_suppression_cost")

        if action == "WARN":
            mc.update_risk(0.05, 0.15)
            if is_verified:
                mc.update_trust(warn_helpful=True)
            else:
                mc.update_trust(warn_helpful=(risk > 0.25))
            if not has_self_evidence:
                mc.update_dependence(blind_obey=True)
            if tempt > 0.5:
                mc.update_gamma_spec(tempt_error=False)
            mc.update_gamma_gen(sustained_pressure=True)
        else:  # WAIT
            mc.update_risk(risk, 0.15)
            if is_sd_needed and has_self_evidence:
                mc.update_dependence(self_discovery=True)
            if tempt > 0.5 and risk > 0.3:
                mc.update_gamma_spec(tempt_error=True)
            if is_false_supp:
                mc.update_gamma_gen(successful_exploration=True)
            elif has_self_evidence:
                mc.update_gamma_gen(successful_exploration=True)
        return mc

    def decide(self, sc, fb, lp, lib, scorer, obs, m: FactoredInternalizationState,
               theta_posterior=None):
        fv = np.full_like(fb, 0.3)
        dc = getattr(sc, 'commit_depth', obs + 1)
        dr = getattr(sc, 'reveal_depth', 3)
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

        tempt = getattr(sc, 'temptation_strength', 0.0)
        risk = getattr(sc, 'risk_level', 0.3)
        subtype = getattr(sc, 'episode_subtype', '')
        has_self_ev = (obs >= dc - 1) or p_self > 0.5

        # Online Q
        Q_online_warn = 1.0 * delta_s + 2.0 * dvoi + 1.5 * (1 - p_self) + 1.0 * tempt - 0.05
        Q_online_wait = 2.0 * p_self * delta_s - 1.5 * p_fail + 2.0

        # Teaching value (path-sensitive)
        theta_str = getattr(sc, 'latent_preference', 'safe')
        L_now = teaching_loss_v2(m, theta_str, theta_posterior)

        mc_warn = self._predict_m(m, "WARN", tempt, risk, subtype, has_self_ev)
        mc_wait = self._predict_m(m, "WAIT", tempt, risk, subtype, has_self_ev)

        L_warn = teaching_loss_v2(mc_warn, theta_str, theta_posterior)
        L_wait = teaching_loss_v2(mc_wait, theta_str, theta_posterior)

        p_blind = 0.7 if not has_self_ev else 0.2
        p_sd = p_self * (0.8 if subtype in ("self_discovery_needed", "self_discovery_teach") else 0.4)

        V_warn, V_wait = path_sensitive_teaching_value(
            L_now, L_warn, L_wait, p_sd, p_blind,
            self.lambda_sd, self.lambda_dep)

        # Overteach penalty
        R_warn = overteach_penalty_v2(mc_warn, theta_str)
        R_wait = overteach_penalty_v2(mc_wait, theta_str)

        # Combined
        Q_warn = Q_online_warn + self.lambda_teach * V_warn - self.lambda_over * R_warn
        Q_wait = Q_online_wait + self.lambda_teach * V_wait - self.lambda_over * R_wait

        action = "WARN" if Q_warn > Q_wait else "WAIT"
        if action == "WARN":
            self.warn_count += 1
        else:
            self.wait_count += 1

        return action, {
            "Q_warn": round(Q_warn, 3), "Q_wait": round(Q_wait, 3),
            "V_warn": round(V_warn, 3), "V_wait": round(V_wait, 3),
            "R_warn": round(R_warn, 4), "R_wait": round(R_wait, 4),
            "p_sd": round(p_sd, 3), "p_blind": round(p_blind, 3),
        }
