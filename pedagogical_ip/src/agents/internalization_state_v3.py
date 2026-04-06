"""Factorized Internalization State v3.

m_t = (κ, τ, ν, γ_spec, γ_gen)
  κ       : risk sensitivity (regression to baseline)
  τ       : tutor trust (quality-driven, high is OK)
  ν       : tutor dependence (compliance without own evidence, should be LOW)
  γ_spec  : temptation-specific suppression (healthy control)
  γ_gen   : general novelty/exploration suppression (overteaching marker)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, PREF_REWARD,
)


@dataclass
class FactoredInternalizationState:
    """m_t = (κ, τ, ν, γ_spec, γ_gen)."""
    kappa: float = 1.0       # risk sensitivity
    tau: float = 0.3         # trust (starts moderate)
    nu: float = 0.1          # dependence (starts low)
    gamma_spec: float = 0.0  # temptation-specific suppression
    gamma_gen: float = 0.0   # general exploration suppression

    # κ params (regression)
    kappa_0: float = 1.0
    beta_kappa: float = 0.08
    alpha_kappa: float = 0.40

    # τ params
    alpha_tau_plus: float = 0.25    # helpful warning → trust up
    alpha_tau_minus: float = 0.12   # bad warning → trust down

    # ν params
    alpha_nu_plus: float = 0.20     # blind obedience → dependence up
    alpha_nu_minus: float = 0.15    # self-discovery → dependence down

    # γ_spec params
    alpha_gs_plus: float = 0.22     # temptation error → spec up
    alpha_gs_minus: float = 0.10    # false suppression → spec down

    # γ_gen params
    alpha_gg_plus: float = 0.08     # sustained pressure → gen up
    alpha_gg_minus: float = 0.12    # successful exploration → gen down

    # Bounds
    kappa_min: float = 0.3
    kappa_max: float = 3.0
    gamma_spec_max: float = 0.7
    gamma_gen_max: float = 0.5
    nu_max: float = 0.8

    # History
    kappa_history: list = field(default_factory=list)
    tau_history: list = field(default_factory=list)
    nu_history: list = field(default_factory=list)
    gs_history: list = field(default_factory=list)
    gg_history: list = field(default_factory=list)

    def snapshot(self):
        self.kappa_history.append(self.kappa)
        self.tau_history.append(self.tau)
        self.nu_history.append(self.nu)
        self.gs_history.append(self.gamma_spec)
        self.gg_history.append(self.gamma_gen)

    def update_risk(self, real_risk: float, expected_risk: float):
        delta = real_risk - expected_risk
        self.kappa = float(np.clip(
            (1 - self.beta_kappa) * self.kappa
            + self.beta_kappa * self.kappa_0
            + self.alpha_kappa * delta,
            self.kappa_min, self.kappa_max))

    def update_trust(self, warn_helpful: bool = False, warn_bad: bool = False):
        if warn_helpful:
            self.tau += self.alpha_tau_plus * (1.0 - self.tau)
        elif warn_bad:
            self.tau -= self.alpha_tau_minus * self.tau
        self.tau = float(np.clip(self.tau, 0.0, 1.0))

    def update_dependence(self, blind_obey: bool = False,
                          self_discovery: bool = False):
        if blind_obey:
            self.nu += self.alpha_nu_plus * (1.0 - self.nu)
        elif self_discovery:
            self.nu -= self.alpha_nu_minus * self.nu
        self.nu = float(np.clip(self.nu, 0.0, self.nu_max))

    def update_gamma_spec(self, tempt_error: bool = False,
                          false_suppression: bool = False):
        if tempt_error:
            self.gamma_spec += self.alpha_gs_plus * (1.0 - self.gamma_spec)
        elif false_suppression:
            self.gamma_spec -= self.alpha_gs_minus * self.gamma_spec
        self.gamma_spec = float(np.clip(self.gamma_spec, 0.0, self.gamma_spec_max))

    def update_gamma_gen(self, sustained_pressure: bool = False,
                         successful_exploration: bool = False):
        if sustained_pressure:
            self.gamma_gen += self.alpha_gg_plus * (1.0 - self.gamma_gen)
        elif successful_exploration:
            self.gamma_gen -= self.alpha_gg_minus * self.gamma_gen
        self.gamma_gen = float(np.clip(self.gamma_gen, 0.0, self.gamma_gen_max))

    def copy(self) -> 'FactoredInternalizationState':
        return FactoredInternalizationState(
            kappa=self.kappa, tau=self.tau, nu=self.nu,
            gamma_spec=self.gamma_spec, gamma_gen=self.gamma_gen,
            kappa_0=self.kappa_0, beta_kappa=self.beta_kappa,
            alpha_kappa=self.alpha_kappa,
            alpha_tau_plus=self.alpha_tau_plus,
            alpha_tau_minus=self.alpha_tau_minus,
            alpha_nu_plus=self.alpha_nu_plus,
            alpha_nu_minus=self.alpha_nu_minus,
            alpha_gs_plus=self.alpha_gs_plus,
            alpha_gs_minus=self.alpha_gs_minus,
            alpha_gg_plus=self.alpha_gg_plus,
            alpha_gg_minus=self.alpha_gg_minus,
            kappa_min=self.kappa_min, kappa_max=self.kappa_max,
            gamma_spec_max=self.gamma_spec_max,
            gamma_gen_max=self.gamma_gen_max, nu_max=self.nu_max)

    @property
    def as_dict(self) -> dict:
        return {"kappa": self.kappa, "tau": self.tau, "nu": self.nu,
                "gamma_spec": self.gamma_spec, "gamma_gen": self.gamma_gen}


def compute_factored_utility(
    branch: BranchAttributes, theta: str,
    m: FactoredInternalizationState, params: AgentPolicyParams,
    warn_bonus: float = 0.0,
    is_novel: bool = False,
    risk_unc: float = 0.0,
    use_epistemic_risk: bool = False,
    use_epistemic_bonus: bool = False,
) -> float:
    """U = R_pref_eff - risk_term - γ_spec·tempt - γ_gen·novel + τ·warn - ν·obey.

    B2 (when use_epistemic_risk=True):
      risk_term uses α-gated attenuation based on branch risk uncertainty.
      α = α_min + (1-α_min)·exp(-γ·ũ_r)  where ũ_r = clip(risk_unc/u_ref, 0, 1)
      Guarantees risk penalty ≥ κ²·α_min·ρ·risk_penalty (risk floor).

    B2 phase-2 (when use_epistemic_bonus=True):
      Adds small curiosity bonus for uncertain novel branches. Disabled by default
      to avoid double-counting with existing γ_gen/novelty pathway.
    """
    x = branch.to_array()
    r_pref = float(np.dot(PREF_REWARD[theta], x))
    r_pref_eff = params.lambda_theta * r_pref

    # B2: Local epistemic risk shaping
    if use_epistemic_risk and risk_unc > 0:
        alpha_min = 0.25
        rho = 0.35
        gamma_epi = 3.0
        u_ref = 0.5  # reference uncertainty for normalization
        u_tilde_r = float(np.clip(risk_unc / (u_ref + 1e-8), 0.0, 1.0))
        alpha = alpha_min + (1.0 - alpha_min) * np.exp(-gamma_epi * u_tilde_r)
        r_risk = (m.kappa ** 2) * (rho + (1.0 - rho) * alpha) * branch.risk_penalty
    else:
        r_risk = (m.kappa ** 2) * branch.risk_penalty

    r_tempt = m.gamma_spec * branch.temptation_score
    r_novel = m.gamma_gen * (0.3 if is_novel else 0.0)
    r_warn = m.tau * warn_bonus
    r_obey = m.nu * (0.2 if warn_bonus > 0 else 0.0)  # dependence cost

    # B2 phase-2: epistemic curiosity bonus (separate flag, OFF by default)
    r_epi_bonus = 0.0
    if use_epistemic_bonus and use_epistemic_risk and risk_unc > 0:
        lambda_epi = 0.15
        u_tilde_r = float(np.clip(risk_unc / (0.5 + 1e-8), 0.0, 1.0))
        alpha = 0.25 + 0.75 * np.exp(-3.0 * u_tilde_r)
        r_epi_bonus = lambda_epi * (1.0 - alpha) * (0.3 if is_novel else 0.0)

    return r_pref_eff - r_risk - r_tempt - r_novel + r_warn - r_obey + r_epi_bonus


def sample_factored_choice(
    branches, theta, m, params, rng,
    warn_bonuses=None, novel_flags=None,
    risk_uncs=None,
    use_epistemic_risk: bool = False,
    use_epistemic_bonus: bool = False,
) -> int:
    if warn_bonuses is None:
        warn_bonuses = [0.0] * len(branches)
    if novel_flags is None:
        novel_flags = [False] * len(branches)
    if risk_uncs is None:
        risk_uncs = [0.0] * len(branches)
    utilities = np.array([
        compute_factored_utility(
            b, theta, m, params, wb, nf,
            risk_unc=ru,
            use_epistemic_risk=use_epistemic_risk,
            use_epistemic_bonus=use_epistemic_bonus,
        )
        for b, wb, nf, ru in zip(branches, warn_bonuses, novel_flags, risk_uncs)
    ])
    scaled = params.beta * utilities
    scaled -= np.max(scaled)
    exp_u = np.exp(scaled)
    sm = exp_u / (exp_u.sum() + 1e-10)
    n = len(branches)
    uniform = np.ones(n) / n
    mixed = (1 - params.epsilon) * sm + params.epsilon * uniform
    return int(rng.choice(n, p=mixed))

