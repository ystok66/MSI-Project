"""Option Intervention Controller — Family-selective macro option layer.

Sits above canonical micro {WAIT, WARN} and existing intervention_policy.
Selects from O_macro = {NONE, WARN, UNLOCK, ITEM_DROP} using:

  Q_opt(o) = Q_base(o)           # from existing counterfactual scoring
           + λ_teach · V_teach   # probe-based teaching value
           + λ_time · U_time     # timing urgency (from IRH)
           - λ_infl · R_infl     # inflation penalty (Δν̂, Δγ̂_gen)
           - λ_res · C_res       # resource cost

Does NOT modify canonical micro tutor or intervention_policy.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np

from ..agents.world_state import WorldState
from ..agents.agent_belief_state import AgentBelief
from .intervention_semantics import InterventionSemantics
from .intervention_risk_head import InterventionRiskHead


OPTIONS = ("NONE", "WARN", "UNLOCK", "ITEM_DROP")


@dataclass
class OptionConfig:
    """Weights for option scoring."""
    # Base score weights (applied to existing counterfactual Q)
    lambda_teach: float = 1.5
    lambda_time: float = 1.0
    lambda_infl: float = 4.0
    lambda_res: float = 1.0

    # Inflation coefficients per option type: R_infl = a·Δν + b·Δγ_gen
    infl_warn_a: float = 1.0    # WARN can push ν up
    infl_warn_b: float = 0.5    # WARN can push γ_gen up
    infl_item_a: float = 0.8    # ITEM_DROP creates shield dependency
    infl_item_b: float = 0.0    # ITEM_DROP doesn't directly affect γ_gen
    infl_unlock_a: float = 0.3  # UNLOCK mild ν effect
    infl_unlock_b: float = 0.5  # UNLOCK may suppress exploration

    # Resource costs
    shield_cost: float = 1.5     # cost of deploying shield
    unlock_cost: float = 1.0     # cost of unlocking gate
    warn_cost: float = 0.3       # cost of issuing warning

    # Resource limits
    max_unlocks: int = 2         # soft limit per episode

    # Teaching value weights per option
    teach_warn: float = 1.0      # WARN has highest teaching value
    teach_unlock: float = 0.5    # UNLOCK moderate
    teach_item: float = 0.1      # ITEM_DROP mostly online rescue


@dataclass
class OptionDecision:
    """Result of option controller decision."""
    chosen: str                         # from OPTIONS
    scores: Dict[str, float]            # option → final Q_opt
    base_scores: Dict[str, float]       # raw counterfactual scores
    teaching_value: Dict[str, float]    # per-option V_teach
    timing_urgency: Dict[str, float]    # per-option U_time
    inflation_penalty: Dict[str, float] # per-option R_infl
    resource_cost: Dict[str, float]     # per-option C_res
    scenario_family: str = ""
    primary_lever: str = ""             # expected best option for this family


class OptionInterventionController:
    """Family-selective macro intervention controller.

    Wraps existing counterfactual scoring and adds pedagogical layers:
    - Teaching value (probe-based)
    - Timing urgency (from InterventionRiskHead)
    - Inflation penalty (ν̂/γ̂_gen trajectory)
    - Resource cost (shield depletion, unlock limits)
    """

    def __init__(self, config: Optional[OptionConfig] = None,
                 semantics: Optional[InterventionSemantics] = None):
        self.config = config or OptionConfig()
        self.semantics = semantics or InterventionSemantics()
        self._unlock_count = 0
        self._warn_count = 0
        self._item_count = 0
        self._history = []

    def select_option(self,
                      scenario_family: str,
                      primary_intervention: str,
                      m_hat: dict,
                      base_q: Optional[Dict[str, float]] = None,
                      p_timeout: float = 0.0,
                      p_blind: float = 0.0,
                      has_shield: bool = False,
                      has_locked_doors: bool = False,
                      nu_trajectory: Optional[list] = None,
                      gamma_gen_trajectory: Optional[list] = None,
                      ) -> OptionDecision:
        """Select macro intervention option.

        Args:
            scenario_family: e.g. "fork_trap", "hazard_belt"
            primary_intervention: expected best lever from ScenarioConfig
            m_hat: current learner state estimate {tau, nu, gamma_gen, ...}
            base_q: existing counterfactual Q scores (WAIT/WARN/UNLOCK/ITEM_DROP)
            p_timeout: from InterventionRiskHead
            p_blind: from InterventionRiskHead
            has_shield: agent already has shield
            has_locked_doors: locked cells exist
            nu_trajectory: recent ν̂ values for drift estimation
            gamma_gen_trajectory: recent γ̂_gen values
        """
        cfg = self.config

        # --- Base scores (from existing intervention_policy or defaults) ---
        if base_q is not None:
            bq = {
                "NONE": base_q.get("WAIT", 0.0),
                "WARN": base_q.get("WARN", 0.0),
                "UNLOCK": base_q.get("UNLOCK", -2.0),
                "ITEM_DROP": base_q.get("ITEM_DROP", -2.0),
            }
        else:
            bq = self._heuristic_base_scores(
                scenario_family, primary_intervention,
                has_locked_doors, has_shield)

        # --- Teaching value V_teach(o) ---
        v_teach = {
            "NONE": 0.0,
            "WARN": cfg.teach_warn,
            "UNLOCK": cfg.teach_unlock,
            "ITEM_DROP": cfg.teach_item,
        }

        # --- Timing urgency U_time(o) ---
        # Different options address different timing risks
        u_time = {
            "NONE": 0.0,
            "WARN": p_blind * 1.0,       # WARN addresses blind-commit
            "UNLOCK": p_timeout * 1.0,    # UNLOCK addresses timeout
            "ITEM_DROP": max(p_blind, p_timeout) * 0.5,  # general mitigation
        }

        # --- Inflation penalty R_infl(o) ---
        nu_hat = m_hat.get("nu", 0.0)
        gamma_gen_hat = m_hat.get("gamma_gen", 0.0)

        # Estimate Δν̂ and Δγ̂_gen from trajectory slope
        delta_nu = self._estimate_drift(nu_trajectory) if nu_trajectory else 0.0
        delta_gamma = self._estimate_drift(gamma_gen_trajectory) if gamma_gen_trajectory else 0.0

        # Current level also matters: higher ν already → more penalty
        nu_pressure = max(nu_hat + delta_nu, 0.0)
        gamma_pressure = max(gamma_gen_hat + delta_gamma, 0.0)

        r_infl = {
            "NONE": 0.0,
            "WARN": cfg.infl_warn_a * nu_pressure + cfg.infl_warn_b * gamma_pressure,
            "UNLOCK": cfg.infl_unlock_a * nu_pressure + cfg.infl_unlock_b * gamma_pressure,
            "ITEM_DROP": cfg.infl_item_a * nu_pressure + cfg.infl_item_b * gamma_pressure,
        }

        # --- Resource cost C_res(o) ---
        c_res = {
            "NONE": 0.0,
            "WARN": cfg.warn_cost * (1.0 + self._warn_count * 0.2),
            "UNLOCK": cfg.unlock_cost * (1.0 + max(0, self._unlock_count - cfg.max_unlocks) * 2.0),
            "ITEM_DROP": cfg.shield_cost if not has_shield else cfg.shield_cost * 3.0,
        }

        # Resource unavailability
        if not has_locked_doors:
            c_res["UNLOCK"] += 10.0   # effectively disable
        if has_shield:
            c_res["ITEM_DROP"] += 10.0  # already has shield

        # --- Final Q_opt ---
        scores = {}
        for o in OPTIONS:
            scores[o] = (
                bq[o]
                + cfg.lambda_teach * v_teach[o]
                + cfg.lambda_time * u_time[o]
                - cfg.lambda_infl * r_infl[o]
                - cfg.lambda_res * c_res[o]
            )

        chosen = max(scores, key=scores.get)

        # Track resource usage
        if chosen == "WARN":
            self._warn_count += 1
        elif chosen == "UNLOCK":
            self._unlock_count += 1
        elif chosen == "ITEM_DROP":
            self._item_count += 1

        decision = OptionDecision(
            chosen=chosen,
            scores=scores,
            base_scores=bq,
            teaching_value=v_teach,
            timing_urgency=u_time,
            inflation_penalty=r_infl,
            resource_cost=c_res,
            scenario_family=scenario_family,
            primary_lever=primary_intervention,
        )
        self._history.append(decision)
        return decision

    def get_history(self):
        return list(self._history)

    def reset(self):
        self._unlock_count = 0
        self._warn_count = 0
        self._item_count = 0
        self._history = []

    def _heuristic_base_scores(self, family, primary, has_doors, has_shield):
        """Default base scores when no counterfactual Q available.

        Uses family + primary_intervention metadata.
        """
        bq = {"NONE": 0.5, "WARN": 0.0, "UNLOCK": -1.0, "ITEM_DROP": -1.0}

        # Family-aligned boost
        if primary == "WARN":
            bq["WARN"] = 1.5
        elif primary == "ITEM_DROP":
            bq["ITEM_DROP"] = 1.5
        elif primary == "UNLOCK":
            bq["UNLOCK"] = 1.5

        # Availability
        if has_doors:
            bq["UNLOCK"] = max(bq["UNLOCK"], 0.5)
        if not has_shield:
            bq["ITEM_DROP"] = max(bq["ITEM_DROP"], 0.0)

        return bq

    @staticmethod
    def _estimate_drift(trajectory: list) -> float:
        """Estimate rate of change from recent values."""
        if not trajectory or len(trajectory) < 2:
            return 0.0
        # Simple slope from last few points
        n = min(len(trajectory), 5)
        recent = trajectory[-n:]
        return (recent[-1] - recent[0]) / max(n - 1, 1)
