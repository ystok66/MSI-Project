"""Micro Bayes Shadow — unified expected utility shadow scorer.

Rewrites the current Q_online + λ_teach·V_full − λ_over·R_over into
a unified four-component expected utility:

  Q_MB(a | x_t) = β_T·TaskGain(a) + β_L·LearnGain(a) − β_D·DepCost(a) − β_C·C(a)

Semantic decomposition:
  TaskGain  — immediate payoff of action (survivor, cue revelation, temptation)
  LearnGain — improvement in agent's internalization state
  DepCost   — dependence/overteaching cost imposed on agent
  C(a)      — fixed intervention cost

Shadow-only. Does NOT modify any frozen module.
3D-consumption only by default: consumes (τ̂, ν̂, γ̂_gen), NOT κ̂/γ̂_spec.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

from ..agents.internalization_state_v3 import FactoredInternalizationState
from ..agents.stochastic_agent_policy import AgentPolicyParams
from ..agents.behavior_bridge import (
    predict_all_probes, bridge_behavior_loss, bridge_overteach_penalty,
)
from ..agents.behavior_probes import BEHAVIOR_ZONES


class MicroPolicyMode(Enum):
    """Micro tutor policy mode."""
    CANONICAL = "canonical"                # frozen BCICTv4 (default)
    OLD_SHADOW = "old_shadow"              # Phase 7 EPU/belief-horizon
    MICRO_BAYES_SHADOW = "micro_bayes_shadow"  # this module


class LearnGainVariant(Enum):
    """LearnGain computation variant."""
    BEHAVIOR_LOSS = "behavior_loss"    # V1: L_now − L_next (existing)
    ENTROPY_REDUCTION = "entropy_reduction"  # V2: H_t − E[H_{t+1}|a]


class DepCostVariant(Enum):
    """DepCost computation variant."""
    SIMPLE = "simple"              # p_blind only (minimal)
    FULL = "full"                  # p_blind + ν̂ increment


@dataclass
class MicroBayesShadow:
    """Unified expected utility shadow scorer.

    Action space: {WAIT, WARN} only (dose ∈ {0, 1}).
    Inputs: 3D-consumption (τ̂, ν̂, γ̂_gen) + scenario context.
    Does NOT consume κ̂ or γ̂_spec in default mode.
    """
    agent_params: AgentPolicyParams = None

    # Component weights (β)
    beta_task: float = 1.0
    beta_learn: float = 3.5
    beta_dep: float = 4.0
    beta_cost: float = 0.5

    # TaskGain sub-weights
    lambda_voi: float = 2.0
    lambda_risk: float = 1.5
    lambda_tempt: float = 1.0
    lambda_sd: float = 2.0
    lambda_fail: float = 1.5

    # DepCost sub-weights
    lambda_nu: float = 3.0
    lambda_blind: float = 2.0

    # Ablation switches
    learn_gain_variant: LearnGainVariant = LearnGainVariant.BEHAVIOR_LOSS
    dep_cost_variant: DepCostVariant = DepCostVariant.FULL

    # Research-only: κ̂-aware mode (NOT default)
    use_kappa_aware: bool = False
    kappa_hat: float = 1.0  # only read when use_kappa_aware=True

    def __post_init__(self):
        if self.agent_params is None:
            self.agent_params = AgentPolicyParams()

    def score(
        self,
        m: FactoredInternalizationState,
        delta_s: float,
        dvoi: float,
        tempt: float,
        risk: float,
        p_self: float,
        p_fail: float,
        subtype: str,
        has_self_ev: bool,
        zones: dict,
        novelty: float = 0.0,
        self_ev: float = 0.5,
        predict_m_fn=None,
    ) -> tuple[str, float, dict]:
        """Score WAIT vs WARN using unified expected utility.

        Returns: (action, dose, info_dict)
        """
        results = {}

        for dose in [0.0, 1.0]:
            # ── Predicted next state ──
            if predict_m_fn is not None:
                mc = predict_m_fn(m, dose, tempt, risk, subtype, has_self_ev)
            else:
                mc = self._default_predict_m(m, dose, tempt, risk, subtype, has_self_ev)

            # ── TaskGain ──
            task_gain = self._task_gain(dose, delta_s, dvoi, tempt, p_self, p_fail)

            # ── LearnGain ──
            learn_gain = self._learn_gain(m, mc, zones, risk, tempt, novelty, self_ev)

            # ── DepCost ──
            dep_cost = self._dep_cost(m, mc, dose, has_self_ev, p_self)

            # ── C(a) ──
            intervention_cost = 1.0 if dose > 0 else 0.0

            # ── Composite Q ──
            Q = (self.beta_task * task_gain
                 + self.beta_learn * learn_gain
                 - self.beta_dep * dep_cost
                 - self.beta_cost * intervention_cost)

            # κ̂-aware research variant (NOT default)
            if self.use_kappa_aware and dose > 0:
                kappa_penalty = max(self.kappa_hat - 1.0, 0.0) * 0.3 * risk
                Q -= kappa_penalty

            results[dose] = {
                "Q": round(float(Q), 4),
                "task_gain": round(float(task_gain), 4),
                "learn_gain": round(float(learn_gain), 4),
                "dep_cost": round(float(dep_cost), 4),
                "intervention_cost": round(float(intervention_cost), 4),
            }

        # Select best
        q_wait = results[0.0]["Q"]
        q_warn = results[1.0]["Q"]
        if q_warn > q_wait:
            action, dose = "WARN", 1.0
        else:
            action, dose = "WAIT", 0.0

        info = {
            "Q_WAIT": q_wait,
            "Q_WARN": q_warn,
            "delta_Q": round(q_warn - q_wait, 4),
            "selected": action,
            "components": results,
            "learn_gain_variant": self.learn_gain_variant.value,
            "dep_cost_variant": self.dep_cost_variant.value,
            "kappa_aware": self.use_kappa_aware,
        }

        return action, dose, info

    def _task_gain(self, dose: float, delta_s: float, dvoi: float,
                   tempt: float, p_self: float, p_fail: float) -> float:
        """TaskGain(a): immediate payoff of the action.

        WARN: reveal information + prevent error
        WAIT: preserve self-discovery opportunity
        """
        if dose > 0:
            # TaskGain(WARN) = Δs + λ_voi·dVOI + λ_risk·(1−p_self) + λ_tempt·tempt
            return (delta_s
                    + self.lambda_voi * dvoi
                    + self.lambda_risk * (1.0 - p_self)
                    + self.lambda_tempt * tempt)
        else:
            # TaskGain(WAIT) = λ_sd · p_self · Δs − λ_fail · p_fail
            return (self.lambda_sd * p_self * delta_s
                    - self.lambda_fail * p_fail)

    def _learn_gain(self, m: FactoredInternalizationState,
                    mc: FactoredInternalizationState,
                    zones: dict,
                    risk: float, tempt: float,
                    novelty: float, self_ev: float) -> float:
        """LearnGain(a): improvement in agent internalization.

        Two variants:
          V1 (BEHAVIOR_LOSS): max(L_now − L_next, 0)
          V2 (ENTROPY_REDUCTION): conceptual entropy reduction
        """
        if self.learn_gain_variant == LearnGainVariant.BEHAVIOR_LOSS:
            L_now = bridge_behavior_loss(m, zones, risk, tempt, novelty, self_ev)
            L_next = bridge_behavior_loss(mc, zones, risk, tempt, novelty, self_ev)
            return max(L_now - L_next, 0.0)

        elif self.learn_gain_variant == LearnGainVariant.ENTROPY_REDUCTION:
            # V2: use probe predictions as proxy for behavioral entropy
            probes_now = predict_all_probes(m, risk, tempt, novelty, self_ev)
            probes_next = predict_all_probes(mc, risk, tempt, novelty, self_ev)

            def _probe_entropy(probes):
                """Binary entropy of each probe, summed."""
                H = 0.0
                for _, p in probes.items():
                    p_clipped = max(min(p, 0.999), 0.001)
                    H -= p_clipped * np.log(p_clipped) + (1 - p_clipped) * np.log(1 - p_clipped)
                return H

            H_now = _probe_entropy(probes_now)
            H_next = _probe_entropy(probes_next)
            return max(H_now - H_next, 0.0)

        return 0.0

    def _dep_cost(self, m: FactoredInternalizationState,
                  mc: FactoredInternalizationState,
                  dose: float, has_self_ev: bool,
                  p_self: float) -> float:
        """DepCost(a): dependence / overteaching cost.

        Two variants:
          SIMPLE: λ_blind · p_blind(a)
          FULL:   λ_ν · max(ν̂' − ν̂, 0) + λ_blind · p_blind(a)

        do(a_tutor) perspective: tutor-caused success ≠ learner capacity.
        """
        # p_blind: probability of blind obedience (without evidence)
        p_blind = (0.7 if not has_self_ev else 0.2) * dose

        if self.dep_cost_variant == DepCostVariant.SIMPLE:
            return self.lambda_blind * p_blind

        elif self.dep_cost_variant == DepCostVariant.FULL:
            # ν̂ increment: predicted increase in dependence
            delta_nu = max(mc.nu - m.nu, 0.0)
            return self.lambda_nu * delta_nu + self.lambda_blind * p_blind

        return 0.0

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
