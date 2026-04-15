"""BC-ICT-v4: Behavior-Calibrated Internalization Control Tutor.

ARCHIVAL — This module is superseded by intervention_policy.py for all
main-line experiments. All current callers are in archive/legacy_runners/.
The warn_count double-counting issue (L273-L278 vs L388+) is a known
legacy bug that does NOT affect current main-line results.

Key innovations:
  1. Bridge-predicted probes (m → ẑ) instead of hand-computed probes
  2. Empirically-calibrated behavior zones (from baseline rollout quantiles)
  3. Warning dose control: ω ∈ {0, 0.5, 1.0}
  4. V_teach uses bridge behavior loss
  5. R_over uses bridge overteach penalty

Step 1 extension:
  micro_policy_mode: canonical | old_shadow | micro_bayes_shadow
  p_self_mode: baseline | old_blend | posterior_A | posterior_B | posterior_C
  Default = canonical + baseline (no behavior change).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..agents.stochastic_agent_policy import AgentPolicyParams
from ..agents.internalization_state_v3 import FactoredInternalizationState
from ..agents.behavior_bridge import (
    predict_all_probes, bridge_behavior_loss, bridge_overteach_penalty,
    EmpiricalZoneCalibrator, BRIDGE_WEIGHTS,
)
from ..agents.behavior_probes import BEHAVIOR_ZONES
from ..metrics.self_discovery import estimate_self_discovery_prob, estimate_failure_if_wait
from ..envs.observation_mask import make_observation_mask
from ..agents.branch_summary import summarize_branch


def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))


@dataclass
class BCICTv4:
    """Behavior-Calibrated ICT with dose control."""
    agent_params: AgentPolicyParams = None
    lambda_teach: float = 3.5
    lambda_over: float = 4.0
    lambda_sd: float = 1.5
    lambda_dep: float = 2.0
    use_dose: bool = True
    use_bridge: bool = True
    use_calibrated_zones: bool = True

    # Phase 7 shadow-mode flags (research branch, all default OFF)
    use_epu_shadow: bool = False          # Refactor 1: Expected Pedagogical Utility
    use_belief_horizon_pself: bool = False # Refactor 2: Belief-horizon p_self
    use_eig_observation: bool = False      # Refactor 3: EIG observation value
    eta_belief: float = 0.5               # Mixing weight for belief-horizon p_self
    lambda_eig: float = 1.0               # EIG observation weight
    lambda_deadline: float = 0.8          # Deadline penalty weight

    # Step 1-4: enum-based mode selectors (default = canonical, no behavior change)
    micro_policy_mode: str = "canonical"    # canonical | micro_bayes_shadow | micro_bayes_shadow_v2 | micro_bayes_shadow_v2_1 | micro_bayes_shadow_v3
    p_self_mode: str = "baseline"           # baseline | old_blend | posterior_A | posterior_B | posterior_C

    warn_count: int = 0
    wait_count: int = 0
    soft_count: int = 0
    calibrator: EmpiricalZoneCalibrator = None
    _zones: dict = None

    def __post_init__(self):
        if self.agent_params is None:
            self.agent_params = AgentPolicyParams()
        if self.calibrator is None:
            self.calibrator = EmpiricalZoneCalibrator()

    @property
    def zones(self):
        if self._zones is not None:
            return self._zones
        if self.use_calibrated_zones:
            z = self.calibrator.calibrated_zones()
            if all(len(self.calibrator.probe_data[p]) >= 4 for p in BRIDGE_WEIGHTS):
                return z
        return BEHAVIOR_ZONES.get("safe", {})

    def set_zones(self, zones):
        self._zones = zones

    def _predict_m(self, m, dose, tempt, risk, subtype, has_self_ev):
        mc = m.copy()
        is_sd = (subtype in ("self_discovery_needed", "self_discovery_teach"))
        is_novel = (subtype in ("false_suppression_cost", "beneficial_novelty"))

        if dose > 0:
            mc.update_risk(0.05, 0.15)
            mc.update_trust(warn_helpful=(risk > 0.25))
            if not has_self_ev:
                mc.update_dependence(blind_obey=True)
                # Dose-scaled dependence: soft warn creates less dependence
                if dose < 1.0:
                    mc.nu = m.nu + dose * (mc.nu - m.nu)
            mc.update_gamma_gen(sustained_pressure=True)
            if dose < 1.0:
                mc.gamma_gen = m.gamma_gen + dose * (mc.gamma_gen - m.gamma_gen)
        else:
            mc.update_risk(risk, 0.15)
            if is_sd and has_self_ev:
                mc.update_dependence(self_discovery=True)
            if tempt > 0.5 and risk > 0.3:
                mc.update_gamma_spec(tempt_error=True)
            if is_novel or has_self_ev:
                mc.update_gamma_gen(successful_exploration=True)
        return mc

    # ─── Phase 7 shadow methods ──────────────────────────

    def _epu_shadow(self, m, risk, tempt, subtype, has_self_ev):
        """Refactor 1: One-step rollout EPU for shadow comparison.

        EU(a) = E[V_true(π*(b^a))] - Cost(a)
        V_true = survival_weight·survival + learning_weight·learning - otr_weight·otr
        """
        results = {}
        for dose in [0.0, 0.5, 1.0]:
            mc = self._predict_m(m, dose, tempt, risk, subtype, has_self_ev)
            # V_true: survival + learning balance
            survival = 1.0 - mc.kappa * risk * max(1.0 - dose * 0.5, 0.3)
            # Learning proxy: lower ν and γ_gen are better; higher τ is better
            learning = (1.0 - mc.nu) * 0.4 + mc.tau * 0.4 + (1.0 - mc.gamma_gen) * 0.2
            otr = max(mc.nu + mc.gamma_gen - 1.0, 0.0)
            V_true = 3.0 * survival + 2.0 * learning - 1.5 * otr
            cost = 0.05 * dose
            results[dose] = round(V_true - cost, 4)
        return results

    def _belief_horizon_pself(self, dc, dr, m, risk):
        """Refactor 2: Hybrid geometric + belief-based self-discovery prob.

        p_self_new = (1-η)·p_geom + η·p_belief_horizon
        where p_belief_horizon accounts for learner's epistemic capacity.
        """
        p_geom = estimate_self_discovery_prob(dc, dr)
        if not self.use_belief_horizon_pself:
            return p_geom
        # Belief-based component
        risk_awareness = min(m.kappa * 2.0, 1.0)  # how risk-sensitive
        update_gain = max(1.0 - m.nu, 0.1)         # can still learn independently
        info_window = max(dc - dr, 0) / max(dc, 1)  # fraction with evidence
        p_belief = risk_awareness * update_gain * info_window
        return (1.0 - self.eta_belief) * p_geom + self.eta_belief * min(p_belief, 1.0)

    def _eig_observation_value(self, m, branches, theta_posterior=None):
        """Refactor 3: I(A; θ) — mutual information between action and latent θ.

        Start with z=θ (safe vs shiny), not full joint.
        """
        from ..agents.internalization_state_v3 import compute_factored_utility
        thetas = ["safe", "shiny"]
        p_theta = theta_posterior if theta_posterior else [0.5, 0.5]
        # P(a|θ) for each branch
        pa_given_theta = {}
        for th in thetas:
            u0 = compute_factored_utility(branches[0], th, m, self.agent_params)
            u1 = compute_factored_utility(branches[1], th, m, self.agent_params)
            logit = self.agent_params.beta * (u0 - u1)
            p0 = 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))
            pa_given_theta[th] = [p0, 1.0 - p0]
        # I(A; θ)
        mi = 0.0
        for i, th in enumerate(thetas):
            for a in range(2):
                p_joint = pa_given_theta[th][a] * p_theta[i]
                p_a = sum(pa_given_theta[t][a] * p_theta[j] for j, t in enumerate(thetas))
                if p_joint > 1e-10 and p_a > 1e-10:
                    mi += p_joint * np.log(p_joint / (p_a * p_theta[i] + 1e-10))
        return max(round(mi, 6), 0.0)

    def decide(self, sc, fb, lp, lib, scorer, obs, m: FactoredInternalizationState,
               theta_posterior=None):
        fv = np.full_like(fb, 0.3)
        dc = getattr(sc, 'commit_depth', obs + 1)
        dr = getattr(sc, 'reveal_depth', 3)
        # Refactor 2: use belief-horizon p_self if enabled
        if self.use_belief_horizon_pself:
            risk = getattr(sc, 'risk_level', 0.3)
            p_self = self._belief_horizon_pself(dc, dr, m, risk)
        else:
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

        z = self.zones

        # Record current probes for calibration
        if self.use_bridge:
            preds = predict_all_probes(m, risk, tempt, novelty, self_ev)
            self.calibrator.record(preds)

        # Online Q
        Q_online_warn = 1.0 * delta_s + 2.0 * dvoi + 1.5 * (1 - p_self) + 1.0 * tempt - 0.05
        Q_online_wait = 2.0 * p_self * delta_s - 1.5 * p_fail + 2.0

        # Evaluate doses
        doses = [0.0, 1.0]
        if self.use_dose:
            doses = [0.0, 0.5, 1.0]

        best_action = "WAIT"
        best_dose = 0.0
        best_Q = -1e9
        q_components = {}  # cache per-dose breakdown for q_detail

        for dose in doses:
            mc = self._predict_m(m, dose, tempt, risk, subtype, has_self_ev)

            if self.use_bridge:
                L_now = bridge_behavior_loss(m, z, risk, tempt, novelty, self_ev)
                L_next = bridge_behavior_loss(mc, z, risk, tempt, novelty, self_ev)
                R = bridge_overteach_penalty(mc, z, risk, tempt, novelty, self_ev)
            else:
                from ..agents.behavior_probes import behavior_loss
                L_now = behavior_loss(m, self.agent_params)
                L_next = behavior_loss(mc, self.agent_params)
                R = 0.0

            V = L_now - L_next

            # Path-sensitive
            p_blind = (0.7 if not has_self_ev else 0.2) * dose
            p_sd = p_self * (0.8 if subtype in ("self_discovery_needed",
                             "self_discovery_teach") else 0.4) * (1.0 - dose)

            V_full = V + self.lambda_sd * p_sd - self.lambda_dep * p_blind

            if dose == 0:
                Q_online_this = Q_online_wait
                Q = Q_online_wait + self.lambda_teach * V_full - self.lambda_over * R
            elif dose == 0.5:
                Q_online_this = 0.5 * Q_online_warn + 0.5 * Q_online_wait
                Q = Q_online_this + self.lambda_teach * V_full - self.lambda_over * R
            else:
                Q_online_this = Q_online_warn
                Q = Q_online_warn + self.lambda_teach * V_full - self.lambda_over * R

            # Cache per-dose components (raw + weighted)
            q_components[dose] = {
                "Q": Q, "Q_online": Q_online_this,
                "V_full_raw": V_full, "R_over_raw": R,
                "V_full_weighted": self.lambda_teach * V_full,
                "R_over_weighted": self.lambda_over * R,
            }

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

        info = {"Q": round(best_Q, 3), "dose": best_dose}

        # Expose Q-margin detail (pure read-only output, no decision effect)
        if 0.0 in q_components and 1.0 in q_components:
            cw = q_components[0.0]  # WAIT
            cn = q_components[1.0]  # WARN
            info["q_detail"] = {
                "Q_WAIT": cw["Q"], "Q_WARN": cn["Q"],
                "delta_Q": cn["Q"] - cw["Q"],
                "Q_online_wait": cw["Q_online"], "Q_online_warn": cn["Q_online"],
                "delta_Q_online": cn["Q_online"] - cw["Q_online"],
                "V_full_wait_raw": cw["V_full_raw"], "V_full_warn_raw": cn["V_full_raw"],
                "delta_V_full_raw": cn["V_full_raw"] - cw["V_full_raw"],
                "delta_V_full_weighted": cn["V_full_weighted"] - cw["V_full_weighted"],
                "R_over_wait_raw": cw["R_over_raw"], "R_over_warn_raw": cn["R_over_raw"],
                "delta_R_over_raw": cn["R_over_raw"] - cw["R_over_raw"],
                "delta_R_over_weighted": cn["R_over_weighted"] - cw["R_over_weighted"],
                "p_self": p_self, "p_fail": p_fail,
                "delta_s": delta_s, "dvoi": dvoi,
            }

        # ─── Shadow-mode computations ─────────────────
        if self.use_epu_shadow:
            epu_scores = self._epu_shadow(m, risk, tempt, subtype, has_self_ev)
            epu_best_dose = max(epu_scores, key=epu_scores.get)
            epu_action = "WAIT" if epu_best_dose == 0 else ("SOFT" if epu_best_dose == 0.5 else "WARN")
            info["epu_shadow"] = {
                "scores": epu_scores, "action": epu_action,
                "agrees": epu_action == best_action,
            }

        if self.use_eig_observation:
            from ..agents.stochastic_agent_policy import BranchAttributes
            branches = [
                BranchAttributes(safety_score=float(sa[0]),
                    temptation_score=getattr(sc, 'tempt_score_a', 0.0),
                    risk_penalty=0.1),
                BranchAttributes(safety_score=float(sb[0]),
                    temptation_score=getattr(sc, 'tempt_score_b', 0.0),
                    risk_penalty=risk),
            ]
            eig_val = self._eig_observation_value(m, branches, theta_posterior)
            info["eig_observation"] = {
                "I_A_theta": eig_val,
                "wait_boost": round(self.lambda_eig * eig_val, 4),
            }

        if self.use_belief_horizon_pself:
            p_geom = estimate_self_discovery_prob(dc, dr)
            info["belief_horizon"] = {
                "p_geom": round(p_geom, 4),
                "p_hybrid": round(p_self, 4),
                "delta": round(p_self - p_geom, 4),
            }

        # ─── Step 1: p_self posterior shadow ─────────────────
        if self.p_self_mode != "baseline":
            from .p_self_posterior_shadow import compute_p_self_posterior, PSelfMode
            mode_map = {
                "old_blend": PSelfMode.OLD_BLEND,
                "posterior_A": PSelfMode.POSTERIOR_A,
                "posterior_B": PSelfMode.POSTERIOR_B,
                "posterior_C": PSelfMode.POSTERIOR_C,
            }
            ps_mode = mode_map.get(self.p_self_mode)
            if ps_mode is not None:
                from ..agents.stochastic_agent_policy import BranchAttributes as BA
                br = [
                    BA(safety_score=float(sa[0]),
                       temptation_score=getattr(sc, 'tempt_score_a', 0.0),
                       risk_penalty=0.1),
                    BA(safety_score=float(sb[0]),
                       temptation_score=getattr(sc, 'tempt_score_b', 0.0),
                       risk_penalty=risk),
                ]
                # Read observer estimates (3D only)
                tau_h = getattr(m, 'tau', 0.3)
                nu_h = getattr(m, 'nu', 0.1)
                gg_h = getattr(m, 'gamma_gen', 0.0)
                ps_result = compute_p_self_posterior(
                    ps_mode, dc, dr,
                    tau_hat=tau_h, nu_hat=nu_h, gamma_gen_hat=gg_h,
                    branches=br, agent_params=self.agent_params,
                    obs_depth=obs,
                )
                info["p_self_posterior_shadow"] = ps_result

        # ─── Step 1: micro_bayes_shadow ─────────────────
        if self.micro_policy_mode != "canonical":
            if self.micro_policy_mode == "micro_bayes_shadow":
                from .micro_bayes_shadow import MicroBayesShadow
                mb = MicroBayesShadow(agent_params=self.agent_params)
                # If p_self posterior is also active, use its p_self
                ps_info = info.get("p_self_posterior_shadow", {})
                mb_p_self = ps_info.get("p_self", p_self)
                mb_p_fail = ps_info.get("p_fail", p_fail)
                mb_action, mb_dose, mb_info = mb.score(
                    m, delta_s, dvoi, tempt, risk,
                    mb_p_self, mb_p_fail,
                    subtype, has_self_ev, z,
                    novelty, self_ev,
                    predict_m_fn=self._predict_m,
                )
                info["micro_bayes_shadow"] = mb_info
                # In replace mode: override canonical decision
                best_action = mb_action
                best_dose = mb_dose
                if best_action == "WARN":
                    self.warn_count += 1 - (1 if q_components.get(1.0, {}).get("Q", -1e9) == best_Q else 0)

            # ─── Step 2: micro_bayes_shadow_v2 (conservative-gated) ───
            elif self.micro_policy_mode == "micro_bayes_shadow_v2":
                from .micro_bayes_shadow_v2 import MicroBayesShadowV2
                mb2 = MicroBayesShadowV2(agent_params=self.agent_params)
                # Get three-outcome p_self (prefer posterior C if available)
                ps_info = info.get("p_self_posterior_shadow", {})
                mb2_p_self = ps_info.get("p_self", p_self)
                mb2_p_fail = ps_info.get("p_fail", p_fail)
                mb2_p_undecided = ps_info.get("p_undecided", 0.0)
                mb2_action, mb2_dose, mb2_info = mb2.score(
                    m, delta_s, dvoi, tempt, risk,
                    mb2_p_self, mb2_p_fail, mb2_p_undecided,
                    subtype, has_self_ev, z,
                    novelty, self_ev,
                    predict_m_fn=self._predict_m,
                )
                info["micro_bayes_shadow_v2"] = mb2_info
                # Replace mode: override canonical decision
                best_action = mb2_action
                best_dose = mb2_dose
                if best_action == "WARN":
                    self.warn_count += 1 - (1 if q_components.get(1.0, {}).get("Q", -1e9) == best_Q else 0)

            # ─── Step 3: micro_bayes_shadow_v3 (causal-dependence) ───
            elif self.micro_policy_mode == "micro_bayes_shadow_v3":
                from .micro_bayes_shadow_v3 import MicroBayesShadowV3
                mb3 = MicroBayesShadowV3(agent_params=self.agent_params)
                ps_info = info.get("p_self_posterior_shadow", {})
                mb3_p_self = ps_info.get("p_self", p_self)
                mb3_p_fail = ps_info.get("p_fail", p_fail)
                mb3_p_undecided = ps_info.get("p_undecided", 0.0)
                mb3_action, mb3_dose, mb3_info = mb3.score(
                    m, delta_s, dvoi, tempt, risk,
                    mb3_p_self, mb3_p_fail, mb3_p_undecided,
                    subtype, has_self_ev, z,
                    novelty, self_ev,
                    predict_m_fn=self._predict_m,
                )
                info["micro_bayes_shadow_v3"] = mb3_info
                best_action = mb3_action
                best_dose = mb3_dose
                if best_action == "WARN":
                    self.warn_count += 1 - (1 if q_components.get(1.0, {}).get("Q", -1e9) == best_Q else 0)

            # ─── Step 4: micro_bayes_shadow_v2.1 (converged recommended) ───
            elif self.micro_policy_mode == "micro_bayes_shadow_v2_1":
                from .micro_bayes_shadow_v2_1 import MicroBayesShadowV2_1
                mb21 = MicroBayesShadowV2_1(agent_params=self.agent_params)
                ps_info = info.get("p_self_posterior_shadow", {})
                mb21_p_self = ps_info.get("p_self", p_self)
                mb21_p_fail = ps_info.get("p_fail", p_fail)
                mb21_p_undecided = ps_info.get("p_undecided", 0.0)
                mb21_action, mb21_dose, mb21_info = mb21.score(
                    m, delta_s, dvoi, tempt, risk,
                    mb21_p_self, mb21_p_fail, mb21_p_undecided,
                    subtype, has_self_ev, z,
                    novelty, self_ev,
                    predict_m_fn=self._predict_m,
                )
                info["micro_bayes_shadow_v2_1"] = mb21_info
                best_action = mb21_action
                best_dose = mb21_dose
                if best_action == "WARN":
                    self.warn_count += 1 - (1 if q_components.get(1.0, {}).get("Q", -1e9) == best_Q else 0)

        return best_action, best_dose, info
