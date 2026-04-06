"""Micro Bayes Shadow v2.1 — converged recommended shadow scorer.

This is the FINAL recommended shadow version, consolidating:
  - Step 2: conservative gate + three-outcome p_self + rebalanced weights
  - Step 3: do()-corrected credit assignment (credit_correction.py)

NOT included (demoted to diagnostics-only):
  - effort_latent_shadow: causes WR regression when in utility (Step 3)
  - calibration: marginal policy effect, diagnostics-only (Steps 2-4)
  - LearnGain entropy variant: redundant with behavior_loss (Step 1)
  - DepCost ν̂ increment: zero discriminative value (Step 1)
  - κ̂-aware mode: no signal at micro layer (Step 1)

Utility:
  Q_v2.1(a) = β_T·TaskGain(a) + β_L·LearnGain_cc(a) − β_D·DepCost_lite(a) − β_C·C(a)

where:
  LearnGain_cc(a) = max(L_now − L_next, 0) · ρ_self(a)
  ρ_self(WAIT)  = 1.0
  ρ_self(WARN)  = 1 − λ_cc · p_directed

Decision rule:
  WARN iff ΔQ > δ AND N > τ_N
  N_t = w_f·p_fail − w_s·p_self − w_u·p_undecided + w_Δ·Δs + w_v·dVOI

Shadow-only. Does NOT modify any frozen module.
3D-consumption only: (τ̂, ν̂, γ̂_gen). No κ̂ / γ̂_spec.
Action space: {WAIT, WARN} only (canonical 2-act).

Target definitions for three-outcome p_self:
  p_self:      P(learner discovers correct branch under do(WAIT), before commit)
  p_fail:      P(learner commits to wrong branch under do(WAIT), irreversible)
  p_undecided: P(learner remains uncommitted at window end, neither discovered nor failed)
  Constraint:  p_self + p_fail + p_undecided = 1
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..agents.internalization_state_v3 import FactoredInternalizationState
from ..agents.stochastic_agent_policy import AgentPolicyParams
from ..agents.behavior_bridge import bridge_behavior_loss
from .credit_correction import CreditCorrection


@dataclass
class MicroBayesShadowV2_1:
    """Converged recommended shadow scorer (v2.1).

    v2 conservative gate + three-outcome + credit correction.

    Action space: {WAIT, WARN} only.
    Inputs: 3D (τ̂, ν̂, γ̂_gen) + (p_self, p_fail, p_undecided) + scene context.
    """
    agent_params: AgentPolicyParams = None

    # ═══ Component weights (β) ═══
    # Verified in Step 2 experiments. Compound constraint:
    #   beta_dep * lambda_blind < beta_task * max(TaskGain)
    beta_task: float = 1.5
    beta_learn: float = 2.5     # applies to credit-corrected gain
    beta_dep: float = 2.0       # simple p_blind only
    beta_cost: float = 1.5      # fixed intervention cost

    # ═══ TaskGain sub-weights ═══
    lambda_voi: float = 1.2     # value of information
    lambda_risk: float = 0.8    # risk-driven WARN incentive
    lambda_tempt: float = 0.8   # temptation resistance
    lambda_sd: float = 2.0      # self-discovery opportunity value
    lambda_fail: float = 1.5    # failure penalty
    lambda_undecided: float = 0.5  # undecided dampens WARN urgency

    # ═══ DepCost ═══
    # Simple p_blind only. ν̂ increment adds zero value (Step 1 ablation).
    lambda_blind: float = 1.0

    # ═══ Conservative gate ═══
    delta_threshold: float = 0.5    # ΔQ must exceed this
    tau_necessity: float = 0.2      # N must exceed this

    # ═══ NecessityScore weights ═══
    w_fail: float = 2.0
    w_self: float = 1.5
    w_undecided: float = 0.8
    w_delta_s: float = 1.0
    w_dvoi: float = 0.8

    # ═══ Credit correction ═══
    credit: CreditCorrection = None

    def __post_init__(self):
        if self.agent_params is None:
            self.agent_params = AgentPolicyParams()
        if self.credit is None:
            self.credit = CreditCorrection()

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
        """Score WAIT vs WARN.

        Returns: (action, dose, info_dict)
        """
        results = {}
        credit_infos = {}

        for dose in [0.0, 1.0]:
            # Predicted next m-state
            if predict_m_fn is not None:
                mc = predict_m_fn(m, dose, tempt, risk, subtype, has_self_ev)
            else:
                mc = self._default_predict_m(m, dose, tempt, risk, subtype, has_self_ev)

            # TaskGain
            task_gain = self._task_gain(
                dose, delta_s, dvoi, tempt, p_self, p_fail, p_undecided)

            # Raw LearnGain (behavior loss delta)
            L_now = bridge_behavior_loss(m, zones, risk, tempt, novelty, self_ev)
            L_next = bridge_behavior_loss(mc, zones, risk, tempt, novelty, self_ev)
            learn_gain_raw = max(L_now - L_next, 0.0)

            # Credit-corrected LearnGain
            learn_gain_cc, credit_info = self.credit.corrected_learn_gain(
                learn_gain_raw, dose, p_self, p_fail, p_undecided,
                has_self_ev, risk, tempt)
            credit_infos[dose] = credit_info

            # DepCost (simple p_blind)
            dep_cost = self._dep_cost(dose, has_self_ev)

            # Fixed intervention cost
            intervention_cost = 1.0 if dose > 0 else 0.0

            # Composite Q
            Q = (self.beta_task * task_gain
                 + self.beta_learn * learn_gain_cc
                 - self.beta_dep * dep_cost
                 - self.beta_cost * intervention_cost)

            results[dose] = {
                "Q": round(float(Q), 4),
                "task_gain": round(float(task_gain), 4),
                "learn_gain_raw": round(float(learn_gain_raw), 4),
                "learn_gain_cc": round(float(learn_gain_cc), 4),
                "dep_cost": round(float(dep_cost), 4),
                "cost": round(float(intervention_cost), 4),
            }

        # Utility margin
        q_wait = results[0.0]["Q"]
        q_warn = results[1.0]["Q"]
        delta_q = q_warn - q_wait

        # NecessityScore
        necessity = self._necessity_score(p_self, p_fail, p_undecided, delta_s, dvoi)

        # Conservative gate decision
        gate_pass = delta_q > self.delta_threshold and necessity > self.tau_necessity
        if gate_pass:
            action, dose = "WARN", 1.0
        else:
            action, dose = "WAIT", 0.0

        # Credit leakage diagnostic
        leakage = credit_infos.get(1.0, {}).get("leakage", 0.0)

        info = {
            "Q_WAIT": q_wait,
            "Q_WARN": q_warn,
            "delta_Q": round(delta_q, 4),
            "necessity": round(necessity, 4),
            "gate_pass": gate_pass,
            "selected": action,
            "components": results,
            "credit": credit_infos,
            "p_input": {"p_self": p_self, "p_fail": p_fail, "p_undecided": p_undecided},
            "leakage": round(float(leakage), 4),
            "version": "v2.1",
        }

        return action, dose, info

    # ═══ Component functions ═══

    def _task_gain(self, dose, delta_s, dvoi, tempt, p_self, p_fail, p_undecided):
        """TaskGain(a): three-outcome aware immediate payoff."""
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

    def _dep_cost(self, dose, has_self_ev):
        """DepCost_lite: simple p_blind only."""
        p_blind = (0.7 if not has_self_ev else 0.2) * dose
        return self.lambda_blind * p_blind

    def _necessity_score(self, p_self, p_fail, p_undecided, delta_s, dvoi):
        """NecessityScore: N_t."""
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
