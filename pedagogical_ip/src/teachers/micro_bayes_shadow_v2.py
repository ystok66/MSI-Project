"""Micro Bayes Shadow v2 — conservative-calibrated micro scorer.

Step 2 evolution of micro_bayes_shadow.py. Key changes from v1:

1. MINIMAL NON-REDUNDANT: drops LearnGain entropy variant, drops DepCost ν̂
   increment (both judged redundant in Step 1 ablation).
2. THREE-OUTCOME INPUT: consumes (p_self, p_fail, p_undecided) instead of
   assuming p_fail = 1 - p_self.
3. CONSERVATIVE GATE: WARN requires BOTH utility margin AND necessity score.
   a_t = WARN iff ΔQ > δ_t AND N_t > τ_N

Utility:
  Q_v2(a) = β_T·TaskGain(a) + β_L·LearnGain(a) - β_D·DepCost(a) - β_C·C(a)

Decision rule:
  ΔQ = Q_v2(WARN) - Q_v2(WAIT)
  N_t = w_f·p̃_fail - w_s·p̃_self - w_u·p̃_undecided + w_Δ·Δs + w_v·dVOI
  WARN iff ΔQ > δ AND N > τ_N

Shadow-only. Does NOT modify any frozen module.
3D-consumption only: (τ̂, ν̂, γ̂_gen). No κ̂ / γ̂_spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

from ..agents.internalization_state_v3 import FactoredInternalizationState
from ..agents.stochastic_agent_policy import AgentPolicyParams
from ..agents.behavior_bridge import bridge_behavior_loss
from .p_self_calibration import PSelfCalibrator, CalibrationMode


@dataclass
class MicroBayesShadowV2:
    """Conservative-calibrated micro Bayes shadow scorer.

    Action space: {WAIT, WARN} only.
    Inputs: 3D (τ̂, ν̂, γ̂_gen) + (p_self, p_fail, p_undecided) + scene context.
    """
    agent_params: AgentPolicyParams = None

    # Component weights (β) — REBALANCED from v1
    # v1 had beta_cost=0.5 which was too low, causing over-WARN
    # v2 compound constraint: beta_dep * lambda_blind < beta_task * max(TaskGain)
    beta_task: float = 1.5
    beta_learn: float = 2.5
    beta_dep: float = 2.0     # ↓ from v1's 4.0; compound 2.0*1.0=2.0
    beta_cost: float = 1.5    # ↑ from v1's 0.5: meaningful fixed cost

    # TaskGain sub-weights — LOWERED risk/voi to reduce WARN bias
    lambda_voi: float = 1.2   # ↓ from 2.0
    lambda_risk: float = 0.8  # ↓ from 1.5
    lambda_tempt: float = 0.8 # ↓ from 1.0
    lambda_sd: float = 2.0    # kept
    lambda_fail: float = 1.5  # kept
    lambda_undecided: float = 0.5  # NEW: p_undecided dampens WARN urgency

    # DepCost: simple only (p_blind), no ν̂ increment (Step 1: zero contribution)
    lambda_blind: float = 1.0  # compound: beta_dep * lambda_blind = 2.0

    # Conservative gate parameters
    use_conservative_gate: bool = True
    delta_threshold: float = 0.5   # ΔQ must exceed this to WARN
    tau_necessity: float = 0.2     # NecessityScore must exceed this to WARN

    # NecessityScore weights
    w_fail: float = 2.0      # higher p̃_fail → more necessary to WARN
    w_self: float = 1.5      # higher p̃_self → less necessary
    w_undecided: float = 0.8  # high p̃_undecided → wait and observe
    w_delta_s: float = 1.0    # higher Δs → evidence gap → more necessary
    w_dvoi: float = 0.8       # value of information still available

    # Calibration
    calibrator: PSelfCalibrator = None

    def __post_init__(self):
        if self.agent_params is None:
            self.agent_params = AgentPolicyParams()
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
        """Score WAIT vs WARN with conservative gate.

        Returns: (action, dose, info_dict)
        """
        # ── Step 1: Calibrate p_self inputs ──
        cal = self.calibrator.calibrate(p_self, p_fail, p_undecided)
        p_s_cal = cal["p_self"]
        p_f_cal = cal["p_fail"]
        p_u_cal = cal["p_undecided"]

        # ── Step 2: Compute Q for each action ──
        results = {}
        for dose in [0.0, 1.0]:
            if predict_m_fn is not None:
                mc = predict_m_fn(m, dose, tempt, risk, subtype, has_self_ev)
            else:
                mc = self._default_predict_m(m, dose, tempt, risk, subtype, has_self_ev)

            task_gain = self._task_gain(dose, delta_s, dvoi, tempt,
                                        p_s_cal, p_f_cal, p_u_cal)
            learn_gain = self._learn_gain(m, mc, zones, risk, tempt, novelty, self_ev)
            dep_cost = self._dep_cost(dose, has_self_ev)
            intervention_cost = 1.0 if dose > 0 else 0.0

            Q = (self.beta_task * task_gain
                 + self.beta_learn * learn_gain
                 - self.beta_dep * dep_cost
                 - self.beta_cost * intervention_cost)

            results[dose] = {
                "Q": round(float(Q), 4),
                "task_gain": round(float(task_gain), 4),
                "learn_gain": round(float(learn_gain), 4),
                "dep_cost": round(float(dep_cost), 4),
                "cost": round(float(intervention_cost), 4),
            }

        # ── Step 3: Compute utility margin ──
        q_wait = results[0.0]["Q"]
        q_warn = results[1.0]["Q"]
        delta_q = q_warn - q_wait

        # ── Step 4: Compute NecessityScore ──
        necessity = self._necessity_score(
            p_s_cal, p_f_cal, p_u_cal, delta_s, dvoi)

        # ── Step 5: Decision with conservative gate ──
        if self.use_conservative_gate:
            if delta_q > self.delta_threshold and necessity > self.tau_necessity:
                action, dose = "WARN", 1.0
            else:
                action, dose = "WAIT", 0.0
        else:
            # Fallback: raw argmax (v1 behavior)
            if q_warn > q_wait:
                action, dose = "WARN", 1.0
            else:
                action, dose = "WAIT", 0.0

        info = {
            "Q_WAIT": q_wait,
            "Q_WARN": q_warn,
            "delta_Q": round(delta_q, 4),
            "necessity": round(necessity, 4),
            "delta_threshold": self.delta_threshold,
            "tau_necessity": self.tau_necessity,
            "gate_pass": delta_q > self.delta_threshold and necessity > self.tau_necessity,
            "selected": action,
            "components": results,
            "calibrated_p": cal,
            "raw_p": {"p_self": p_self, "p_fail": p_fail, "p_undecided": p_undecided},
            "version": "v2",
        }

        return action, dose, info

    def _task_gain(self, dose: float, delta_s: float, dvoi: float,
                   tempt: float, p_self: float, p_fail: float,
                   p_undecided: float) -> float:
        """TaskGain(a): immediate payoff.

        Three-outcome aware:
          WARN: benefit from revealing info, reduced by p_undecided
          WAIT: benefit from self-discovery opportunity
        """
        if dose > 0:
            # TaskGain(WARN) = Δs + λ_voi·dVOI + λ_risk·p_fail + λ_tempt·tempt
            #                  - λ_undecided·p_undecided
            # Note: uses p_fail instead of (1-p_self) — three-outcome semantics
            return (delta_s
                    + self.lambda_voi * dvoi
                    + self.lambda_risk * p_fail
                    + self.lambda_tempt * tempt
                    - self.lambda_undecided * p_undecided)
        else:
            # TaskGain(WAIT) = λ_sd · p_self · Δs − λ_fail · p_fail
            #                  + λ_undecided · p_undecided · 0.5
            # Undecided mass slightly favors WAIT (still observing)
            return (self.lambda_sd * p_self * delta_s
                    - self.lambda_fail * p_fail
                    + self.lambda_undecided * p_undecided * 0.5)

    def _learn_gain(self, m: FactoredInternalizationState,
                    mc: FactoredInternalizationState,
                    zones: dict,
                    risk: float, tempt: float,
                    novelty: float, self_ev: float) -> float:
        """LearnGain(a): behavior loss improvement only.

        Step 1 verdict: entropy variant is redundant. Keep behavior_loss only.
        """
        L_now = bridge_behavior_loss(m, zones, risk, tempt, novelty, self_ev)
        L_next = bridge_behavior_loss(mc, zones, risk, tempt, novelty, self_ev)
        return max(L_now - L_next, 0.0)

    def _dep_cost(self, dose: float, has_self_ev: bool) -> float:
        """DepCost(a): simple p_blind only.

        Step 1 verdict: ν̂ increment adds zero discriminative value.
        do(a_tutor) perspective: tutor-caused success ≠ learner capacity.
        """
        p_blind = (0.7 if not has_self_ev else 0.2) * dose
        return self.lambda_blind * p_blind

    def _necessity_score(self, p_self: float, p_fail: float,
                         p_undecided: float,
                         delta_s: float, dvoi: float) -> float:
        """NecessityScore: how necessary is WARN right now?

        N_t = w_f·p̃_fail − w_s·p̃_self − w_u·p̃_undecided + w_Δ·Δs + w_v·dVOI

        High when: failure is likely, self-discovery is unlikely,
                   and there's genuine evidence gap.
        Low when: agent can self-discover, or is still in undecided zone.
        """
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
