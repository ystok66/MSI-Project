"""Micro Bayes Shadow v3 — causal-dependence shadow scorer.

Step 3 evolution. Keeps Step 2's conservative gate + three-outcome input.
Replaces heuristic LearnGain / DepCost with:
  - LearnGain_do: do()-corrected via credit_correction.py
  - EffortLoss: latent effort state via effort_latent_shadow.py

Q_v3(a) = β_T·TaskGain(a) + β_L·LearnGain_do(a) − β_E·EffortLoss(a) − β_C·C(a)

Decision rule (same as v2):
  WARN iff ΔQ > δ AND N > τ_N

Shadow-only. Does NOT modify any frozen module.
3D-consumption only: (τ̂, ν̂, γ̂_gen).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..agents.internalization_state_v3 import FactoredInternalizationState
from ..agents.stochastic_agent_policy import AgentPolicyParams
from ..agents.behavior_bridge import bridge_behavior_loss
from .credit_correction import CreditCorrection
from .effort_latent_shadow import EffortLatentShadow
from .p_self_calibration import PSelfCalibrator, CalibrationMode


@dataclass
class MicroBayesShadowV3:
    """Causal-dependence micro scorer (v3-lite).

    Action space: {WAIT, WARN} only.
    Inputs: 3D (τ̂, ν̂, γ̂_gen) + (p_self, p_fail, p_undecided) + scene context.
    """
    agent_params: AgentPolicyParams = None

    # Component weights — same structure as v2
    beta_task: float = 1.5
    beta_learn: float = 2.5     # now applies to do-corrected gain
    beta_effort: float = 3.0    # replaces beta_dep; higher to respect effort
    beta_cost: float = 1.5

    # TaskGain sub-weights (carried from v2)
    lambda_voi: float = 1.2
    lambda_risk: float = 0.8
    lambda_tempt: float = 0.8
    lambda_sd: float = 2.0
    lambda_fail: float = 1.5
    lambda_undecided: float = 0.5

    # Conservative gate (carried from v2)
    use_conservative_gate: bool = True
    delta_threshold: float = 0.5
    tau_necessity: float = 0.2

    # NecessityScore weights (carried from v2)
    w_fail: float = 2.0
    w_self: float = 1.5
    w_undecided: float = 0.8
    w_delta_s: float = 1.0
    w_dvoi: float = 0.8

    # Submodules
    credit: CreditCorrection = None
    effort: EffortLatentShadow = None
    calibrator: PSelfCalibrator = None

    # Ablation flags
    use_credit_correction: bool = True     # can disable for B/C arm isolation
    use_effort_loss: bool = True           # can disable for B/C arm isolation

    def __post_init__(self):
        if self.agent_params is None:
            self.agent_params = AgentPolicyParams()
        if self.credit is None:
            self.credit = CreditCorrection()
        if self.effort is None:
            self.effort = EffortLatentShadow()
        if self.calibrator is None:
            self.calibrator = PSelfCalibrator(mode=CalibrationMode.FIXED_BETA)

    def score(
        self,
        m: FactoredInternalizationState,
        delta_s: float,
        dvoi: float,
        tempt: float,
        risk: float,
        p_self: float,
        p_fail: float,
        p_undecided: float,
        subtype: str,
        has_self_ev: bool,
        zones: dict,
        novelty: float = 0.0,
        self_ev: float = 0.5,
        predict_m_fn=None,
    ) -> tuple[str, float, dict]:
        """Score WAIT vs WARN with causal credit + effort loss.

        Returns: (action, dose, info_dict)
        """
        # Calibrate
        cal = self.calibrator.calibrate(p_self, p_fail, p_undecided)
        p_s = cal["p_self"]
        p_f = cal["p_fail"]
        p_u = cal["p_undecided"]

        results = {}
        credit_infos = {}
        effort_infos = {}

        for dose in [0.0, 1.0]:
            # Predicted next m-state
            if predict_m_fn is not None:
                mc = predict_m_fn(m, dose, tempt, risk, subtype, has_self_ev)
            else:
                mc = self._default_predict_m(m, dose, tempt, risk, subtype, has_self_ev)

            # TaskGain (same as v2)
            task_gain = self._task_gain(dose, delta_s, dvoi, tempt, p_s, p_f, p_u)

            # Raw learning gain (behavior loss delta)
            L_now = bridge_behavior_loss(m, zones, risk, tempt, novelty, self_ev)
            L_next = bridge_behavior_loss(mc, zones, risk, tempt, novelty, self_ev)
            learn_gain_raw = max(L_now - L_next, 0.0)

            # LearnGain_do: apply credit correction
            if self.use_credit_correction:
                learn_gain_do, credit_info = self.credit.corrected_learn_gain(
                    learn_gain_raw, dose, p_s, p_f, p_u,
                    has_self_ev, risk, tempt)
            else:
                learn_gain_do = learn_gain_raw
                credit_info = {"rho_self": 1.0, "learn_gain_raw": learn_gain_raw,
                               "learn_gain_do": learn_gain_raw, "leakage": 0.0}
            credit_infos[dose] = credit_info

            # EffortLoss
            if self.use_effort_loss:
                effort_loss, effort_info = self.effort.compute_effort_loss(
                    dose, p_s, has_self_ev)
            else:
                # Fallback: simple p_blind (v2 behavior)
                p_blind = (0.7 if not has_self_ev else 0.2) * dose
                effort_loss = p_blind
                effort_info = {"effort_loss": p_blind, "effort_now": 0.5,
                               "effort_next": 0.5}
            effort_infos[dose] = effort_info

            # Intervention cost
            intervention_cost = 1.0 if dose > 0 else 0.0

            # Composite Q
            Q = (self.beta_task * task_gain
                 + self.beta_learn * learn_gain_do
                 - self.beta_effort * effort_loss
                 - self.beta_cost * intervention_cost)

            results[dose] = {
                "Q": round(float(Q), 4),
                "task_gain": round(float(task_gain), 4),
                "learn_gain_raw": round(float(learn_gain_raw), 4),
                "learn_gain_do": round(float(learn_gain_do), 4),
                "effort_loss": round(float(effort_loss), 4),
                "cost": round(float(intervention_cost), 4),
            }

        # Utility margin
        q_wait = results[0.0]["Q"]
        q_warn = results[1.0]["Q"]
        delta_q = q_warn - q_wait

        # NecessityScore
        necessity = self._necessity_score(p_s, p_f, p_u, delta_s, dvoi)

        # Decision with conservative gate
        if self.use_conservative_gate:
            gate_pass = delta_q > self.delta_threshold and necessity > self.tau_necessity
            if gate_pass:
                action, dose = "WARN", 1.0
            else:
                action, dose = "WAIT", 0.0
        else:
            gate_pass = None
            if q_warn > q_wait:
                action, dose = "WARN", 1.0
            else:
                action, dose = "WAIT", 0.0

        # Credit leakage metric
        leakage_warn = credit_infos.get(1.0, {}).get("leakage", 0.0)

        info = {
            "Q_WAIT": q_wait,
            "Q_WARN": q_warn,
            "delta_Q": round(delta_q, 4),
            "necessity": round(necessity, 4),
            "gate_pass": gate_pass,
            "selected": action,
            "components": results,
            "credit": credit_infos,
            "effort": effort_infos,
            "calibrated_p": cal,
            "raw_p": {"p_self": p_self, "p_fail": p_fail, "p_undecided": p_undecided},
            "leakage_warn": round(float(leakage_warn), 4),
            "effort_now": round(float(self.effort.effort), 4),
            "version": "v3",
        }

        return action, dose, info

    def update_effort(self, dose: float, p_self: float,
                      has_self_ev: bool, self_discovery: bool):
        """Update effort state after step outcome. Call after episode step."""
        self.effort.update(dose, p_self, has_self_ev, self_discovery)

    def _task_gain(self, dose, delta_s, dvoi, tempt, p_self, p_fail, p_undecided):
        """TaskGain(a) — same as v2."""
        if dose > 0:
            return (delta_s
                    + self.lambda_voi * dvoi
                    + self.lambda_risk * p_fail
                    + self.lambda_tempt * tempt
                    - self.lambda_undecided * p_undecided)
        else:
            return (self.lambda_sd * p_self * delta_s
                    - self.lambda_fail * p_fail
                    + self.lambda_undecided * p_undecided * 0.5)

    def _necessity_score(self, p_self, p_fail, p_undecided, delta_s, dvoi):
        """NecessityScore — same as v2."""
        return (self.w_fail * p_fail
                - self.w_self * p_self
                - self.w_undecided * p_undecided
                + self.w_delta_s * delta_s
                + self.w_dvoi * dvoi)

    def _default_predict_m(self, m, dose, tempt, risk, subtype, has_self_ev):
        """Default m-prediction (mirrors BCICTv4._predict_m)."""
        mc = m.copy()
        is_sd = (subtype in ("self_discovery_needed", "self_discovery_teach"))
        is_novel = (subtype in ("false_suppression_cost", "beneficial_novelty"))
        if dose > 0:
            mc.update_risk(0.05, 0.15)
            mc.update_trust(warn_helpful=(risk > 0.25))
            if not has_self_ev:
                mc.update_dependence(blind_obey=True)
            mc.update_gamma_gen(sustained_pressure=True)
        else:
            mc.update_risk(risk, 0.15)
            if is_sd and has_self_ev:
                mc.update_dependence(self_discovery=True)
            if tempt > 0.5 and risk > 0.3:
                mc.update_gamma_spec(tempt_error=True)
            if is_novel or has_self_ev:
                mc.update_gamma_gen(successful_exploration=True)
        return mc
