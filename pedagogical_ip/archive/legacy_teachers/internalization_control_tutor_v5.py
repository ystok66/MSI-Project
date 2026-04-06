"""MC-ICT-v5: Mechanism-Calibrated Internalization Control Tutor.

Q(a) = Q_online + λ_teach·V_teach_beh − λ_over·R_over − λ_acc·R_acc

Key innovations over BC-v4:
  1. Trainable bridge (BCE + Jacobian + ECE) instead of fixed weights
  2. Empirical zones from baseline quantiles
  3. R_acc: accidental correctness penalty (high γ_gen → discount beneficial_novelty)
  4. Dose as formal action: WAIT(0) / SOFT(0.5) / HARD(1.0)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..agents.stochastic_agent_policy import AgentPolicyParams
from ..agents.internalization_state_v3 import FactoredInternalizationState
from ..agents.trainable_bridge import TrainableBridge
from ..metrics.self_discovery import estimate_self_discovery_prob, estimate_failure_if_wait
from ..envs.observation_mask import make_observation_mask
from ..agents.branch_summary import summarize_branch


def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))


@dataclass
class MCICTv5:
    """Mechanism-Calibrated Internalization Control Tutor."""
    agent_params: AgentPolicyParams = None
    lambda_teach: float = 3.5
    lambda_over: float = 4.0
    lambda_sd: float = 1.5
    lambda_dep: float = 2.0
    lambda_acc: float = 2.0
    use_dose: bool = True
    use_trainable: bool = True
    use_mca: bool = True

    bridge: TrainableBridge = None
    warn_count: int = 0
    wait_count: int = 0
    soft_count: int = 0

    def __post_init__(self):
        if self.agent_params is None:
            self.agent_params = AgentPolicyParams()
        if self.bridge is None:
            self.bridge = TrainableBridge()

    def _predict_m(self, m, dose, tempt, risk, subtype, has_self_ev):
        mc = m.copy()
        is_sd = (subtype in ("self_discovery_needed", "self_discovery_teach"))
        is_novel = (subtype in ("false_suppression_cost", "beneficial_novelty"))

        if dose > 0:
            mc.update_risk(0.05, 0.15)
            mc.update_trust(warn_helpful=(risk > 0.25))
            if not has_self_ev:
                old_nu = mc.nu
                mc.update_dependence(blind_obey=True)
                mc.nu = old_nu + dose * (mc.nu - old_nu)
            old_gg = mc.gamma_gen
            mc.update_gamma_gen(sustained_pressure=True)
            mc.gamma_gen = old_gg + dose * (mc.gamma_gen - old_gg)
        else:
            mc.update_risk(risk, 0.15)
            if is_sd and has_self_ev:
                mc.update_dependence(self_discovery=True)
            if tempt > 0.5 and risk > 0.3:
                mc.update_gamma_spec(tempt_error=True)
            if is_novel or has_self_ev:
                mc.update_gamma_gen(successful_exploration=True)
        return mc

    def _accidental_risk(self, m_next, subtype):
        """R_acc: risk that correctness comes from wrong mechanism."""
        if not self.use_mca:
            return 0.0
        r = 0.0
        # High γ_gen on beneficial_novelty → false-suppression rebound
        if subtype in ("beneficial_novelty", "false_suppression_cost"):
            if m_next.gamma_gen > 0.2:
                r += 1.5 * (m_next.gamma_gen - 0.2) ** 2
        # High ν on valid advice → blind obedience, not trust
        if subtype in ("sparse_valid_advice", "verified_warn"):
            if m_next.nu > 0.25:
                r += 1.0 * (m_next.nu - 0.25) ** 2
        return float(r)

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
        novelty = 0.3 if subtype in ("beneficial_novelty",) else 0.0
        has_self_ev = (obs >= dc - 1) or p_self > 0.5
        self_ev = 0.7 if has_self_ev else 0.3

        z = self.bridge.empirical_zones()

        Q_online_warn = 1.0 * delta_s + 2.0 * dvoi + 1.5 * (1 - p_self) + 1.0 * tempt - 0.05
        Q_online_wait = 2.0 * p_self * delta_s - 1.5 * p_fail + 2.0

        doses = [0.0, 1.0]
        if self.use_dose:
            doses = [0.0, 0.5, 1.0]

        best_action = "WAIT"
        best_dose = 0.0
        best_Q = -1e9

        for dose in doses:
            mc = self._predict_m(m, dose, tempt, risk, subtype, has_self_ev)

            L_now = self.bridge.behavior_loss(m, z, risk, tempt, novelty, self_ev)
            L_next = self.bridge.behavior_loss(mc, z, risk, tempt, novelty, self_ev)
            V = L_now - L_next

            p_blind = (0.7 if not has_self_ev else 0.2) * dose
            p_sd = p_self * (0.8 if subtype in ("self_discovery_needed",
                             "self_discovery_teach") else 0.4) * (1.0 - dose)
            V_full = V + self.lambda_sd * p_sd - self.lambda_dep * p_blind

            R = self.bridge.overteach_penalty(mc, z, risk, tempt, novelty, self_ev)
            R_acc = self._accidental_risk(mc, subtype)

            if dose == 0:
                Q = Q_online_wait + self.lambda_teach * V_full - self.lambda_over * R - self.lambda_acc * R_acc
            elif dose == 0.5:
                Q_soft = 0.5 * Q_online_warn + 0.5 * Q_online_wait
                Q = Q_soft + self.lambda_teach * V_full - self.lambda_over * R - self.lambda_acc * R_acc
            else:
                Q = Q_online_warn + self.lambda_teach * V_full - self.lambda_over * R - self.lambda_acc * R_acc

            if Q > best_Q:
                best_Q = Q
                best_dose = dose
                best_action = "WAIT" if dose == 0 else ("SOFT" if dose == 0.5 else "WARN")

        if best_action == "WARN":
            self.warn_count += 1
        elif best_action == "SOFT":
            self.soft_count += 1
        else:
            self.wait_count += 1

        return best_action, best_dose, {"Q": round(best_Q, 3), "dose": best_dose}
