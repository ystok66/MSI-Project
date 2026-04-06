"""Pedagogical Internalization Agent (PIA).

Agent with updatable teaching internalization state m_t = (κ_t, η_t, γ_t):
  κ_t ∈ R+   : risk sensitivity (amplifies risk penalty)
  η_t ∈ [0,1]: warning trust (how much tutor evidence is absorbed)
  γ_t ∈ [0,1]: temptation suppression (dampens pref-driven lure)

Utility:
  U(π | g,θ,m_t) = R_goal(π;g) + (1-γ_t)·λ_θ·R_pref(π;θ) - κ_t·J_risk(π) + η_t·B_warn(π)

Update rules (non-RL, explicit state transitions):
  κ_{t+1} = clip(κ_t + α_κ·(r_real - r_expected), κ_min, κ_max)
  η_{t+1} = η_t + α_η·(z_t - η_t)         z_t = 1[warn matched truth]
  γ_{t+1} = clip(γ_t + α_γ·1[tempt-caused error], 0, γ_max)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, PREFERENCE_TYPES, PREF_REWARD,
)


@dataclass
class InternalizationState:
    """m_t = (κ, η, γ)."""
    kappa: float = 1.0          # risk sensitivity (starts neutral)
    eta: float = 0.5            # warning trust (starts neutral)
    gamma: float = 0.0          # temptation suppression (starts zero)

    # Update rates — must be large enough to shift β·U in 8 episodes
    alpha_kappa: float = 0.50   # risk learning rate (κ should reach ~2.0 in 8 ep)
    alpha_eta: float = 0.35     # trust learning rate
    alpha_gamma: float = 0.30   # suppression learning rate (γ→0.3+ in 3-4 errors)

    # Bounds
    kappa_min: float = 0.3
    kappa_max: float = 3.0
    gamma_max: float = 0.8

    # History for tracking
    kappa_history: list = field(default_factory=list)
    eta_history: list = field(default_factory=list)
    gamma_history: list = field(default_factory=list)

    def snapshot(self):
        self.kappa_history.append(self.kappa)
        self.eta_history.append(self.eta)
        self.gamma_history.append(self.gamma)

    def update_risk(self, real_risk: float, expected_risk: float):
        """κ_{t+1} = clip(κ_t + α_κ·(r_real - r_expected), κ_min, κ_max)."""
        delta = real_risk - expected_risk
        self.kappa = float(np.clip(
            self.kappa + self.alpha_kappa * delta,
            self.kappa_min, self.kappa_max))

    def update_trust(self, warn_matched_truth: bool):
        """η_{t+1} = η_t + α_η·(z_t - η_t)."""
        z = 1.0 if warn_matched_truth else 0.0
        self.eta = float(self.eta + self.alpha_eta * (z - self.eta))
        self.eta = float(np.clip(self.eta, 0.0, 1.0))

    def update_suppression(self, temptation_caused_error: bool):
        """γ_{t+1} = clip(γ_t + α_γ·1[tempt error], 0, γ_max)."""
        if temptation_caused_error:
            self.gamma = float(np.clip(
                self.gamma + self.alpha_gamma,
                0.0, self.gamma_max))

    def copy(self) -> 'InternalizationState':
        return InternalizationState(
            kappa=self.kappa, eta=self.eta, gamma=self.gamma,
            alpha_kappa=self.alpha_kappa, alpha_eta=self.alpha_eta,
            alpha_gamma=self.alpha_gamma,
            kappa_min=self.kappa_min, kappa_max=self.kappa_max,
            gamma_max=self.gamma_max,
        )


def compute_pia_utility(
    branch: BranchAttributes,
    theta: str,
    m: InternalizationState,
    params: AgentPolicyParams,
    warn_bonus: float = 0.0,
) -> float:
    """U(π | θ, m_t) = R_pref_eff + warn_absorbed - κ²·risk.

    R_pref_eff = (1-γ)·λ_θ·R_pref(π;θ)
    Uses κ² amplification: small κ changes → large risk penalty shifts.
    """
    x = branch.to_array()
    r_pref = float(np.dot(PREF_REWARD[theta], x))
    r_pref_eff = (1.0 - m.gamma) * params.lambda_theta * r_pref
    r_warn = m.eta * warn_bonus
    # κ² amplification: κ=1→1, κ=1.5→2.25, κ=2→4
    r_risk = (m.kappa ** 2) * branch.risk_penalty
    return r_pref_eff + r_warn - r_risk


def sample_pia_choice(
    branches: list[BranchAttributes],
    theta: str,
    m: InternalizationState,
    params: AgentPolicyParams,
    rng: np.random.Generator,
    warn_bonuses: Optional[list[float]] = None,
) -> int:
    """Sample branch choice using PIA utility."""
    if warn_bonuses is None:
        warn_bonuses = [0.0] * len(branches)
    utilities = np.array([
        compute_pia_utility(b, theta, m, params, wb)
        for b, wb in zip(branches, warn_bonuses)
    ])
    scaled = params.beta * utilities
    scaled -= np.max(scaled)
    exp_u = np.exp(scaled)
    sm = exp_u / (exp_u.sum() + 1e-10)
    n = len(branches)
    uniform = np.ones(n) / n
    mixed = (1 - params.epsilon) * sm + params.epsilon * uniform
    return int(rng.choice(n, p=mixed))


def compute_expected_m_change(
    m: InternalizationState,
    was_warned: bool,
    branch_risk: float,
    expected_risk: float,
    tempt_score: float,
    chose_risky: bool,
) -> dict:
    """Estimate Δm components for teaching-aware tutor."""
    # Expected κ change
    if chose_risky:
        delta_kappa = m.alpha_kappa * (branch_risk - expected_risk)
    else:
        delta_kappa = 0.0

    # Expected η change
    if was_warned:
        delta_eta = m.alpha_eta * (1.0 - m.eta)  # assume warn is correct
    else:
        delta_eta = 0.0

    # Expected γ change
    if chose_risky and tempt_score > 0.5:
        delta_gamma = m.alpha_gamma
    else:
        delta_gamma = 0.0

    return {
        "delta_kappa": delta_kappa,
        "delta_eta": delta_eta,
        "delta_gamma": delta_gamma,
        "total_learn_gain": abs(delta_kappa) + abs(delta_eta) + delta_gamma,
    }
