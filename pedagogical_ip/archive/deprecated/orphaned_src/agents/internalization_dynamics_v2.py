"""Internalization Dynamics v2.

Upgrades from v1:
  κ: mean-regression to baseline κ_0, prevents runaway risk aversion
  η: quality-based trust (helpful/unnecessary/missed), not just truth-match
  γ: bidirectional (false suppression cost decays γ)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, PREF_REWARD,
)


@dataclass
class InternalizationStateV2:
    """m_t = (κ, η, γ) with v2 dynamics."""
    kappa: float = 1.0
    eta: float = 0.5
    gamma: float = 0.0

    # v2 params
    kappa_0: float = 1.0       # baseline κ (regression target)
    beta_kappa: float = 0.08   # regression rate toward κ_0
    alpha_kappa: float = 0.40  # risk surprise rate

    alpha_eta_plus: float = 0.30   # warn helpful & verified
    alpha_eta_minus: float = 0.15  # warn unnecessary
    alpha_eta_miss: float = 0.10   # needed but missed

    alpha_gamma_plus: float = 0.25   # temptation error
    alpha_gamma_minus: float = 0.08  # false suppression cost

    kappa_min: float = 0.3
    kappa_max: float = 3.0
    gamma_max: float = 0.8

    kappa_history: list = field(default_factory=list)
    eta_history: list = field(default_factory=list)
    gamma_history: list = field(default_factory=list)

    def snapshot(self):
        self.kappa_history.append(self.kappa)
        self.eta_history.append(self.eta)
        self.gamma_history.append(self.gamma)

    def update_risk(self, real_risk: float, expected_risk: float):
        """κ with mean-regression: κ' = (1-β)κ + β·κ_0 + α·δ."""
        delta = real_risk - expected_risk
        self.kappa = float(np.clip(
            (1 - self.beta_kappa) * self.kappa
            + self.beta_kappa * self.kappa_0
            + self.alpha_kappa * delta,
            self.kappa_min, self.kappa_max))

    def update_trust(self, warn_helpful: bool = False,
                     warn_unnecessary: bool = False,
                     warn_missed: bool = False):
        """η by warning quality category."""
        if warn_helpful:
            self.eta += self.alpha_eta_plus * (1.0 - self.eta)
        elif warn_unnecessary:
            self.eta -= self.alpha_eta_minus * self.eta
        elif warn_missed:
            self.eta -= self.alpha_eta_miss * self.eta
        self.eta = float(np.clip(self.eta, 0.0, 1.0))

    def update_suppression(self, temptation_error: bool = False,
                           false_suppression: bool = False):
        """γ bidirectional."""
        if temptation_error:
            self.gamma += self.alpha_gamma_plus * (1.0 - self.gamma)
        elif false_suppression:
            self.gamma -= self.alpha_gamma_minus * self.gamma
        self.gamma = float(np.clip(self.gamma, 0.0, self.gamma_max))

    def copy(self) -> 'InternalizationStateV2':
        return InternalizationStateV2(
            kappa=self.kappa, eta=self.eta, gamma=self.gamma,
            kappa_0=self.kappa_0, beta_kappa=self.beta_kappa,
            alpha_kappa=self.alpha_kappa,
            alpha_eta_plus=self.alpha_eta_plus,
            alpha_eta_minus=self.alpha_eta_minus,
            alpha_eta_miss=self.alpha_eta_miss,
            alpha_gamma_plus=self.alpha_gamma_plus,
            alpha_gamma_minus=self.alpha_gamma_minus,
            kappa_min=self.kappa_min, kappa_max=self.kappa_max,
            gamma_max=self.gamma_max)


def compute_pia_v2_utility(
    branch: BranchAttributes, theta: str,
    m: InternalizationStateV2, params: AgentPolicyParams,
    warn_bonus: float = 0.0,
) -> float:
    x = branch.to_array()
    r_pref = float(np.dot(PREF_REWARD[theta], x))
    r_pref_eff = (1.0 - m.gamma) * params.lambda_theta * r_pref
    r_warn = m.eta * warn_bonus
    r_risk = (m.kappa ** 2) * branch.risk_penalty
    return r_pref_eff + r_warn - r_risk


def sample_pia_v2_choice(
    branches, theta, m, params, rng, warn_bonuses=None,
) -> int:
    if warn_bonuses is None:
        warn_bonuses = [0.0] * len(branches)
    utilities = np.array([
        compute_pia_v2_utility(b, theta, m, params, wb)
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
