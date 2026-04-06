"""Behavior Probes: map factored internalization state to observable behavior.

5 probes, each computed as a softmax probability under the factored utility:
  RC(m) = P(choose safe | risk probe)         ← κ
  TR(m) = P(reject lure | temptation probe)   ← γ_spec
  EP(m) = P(choose novel good | novelty probe) ← γ_gen
  VA(m) = P(follow valid advice | low self-ev)  ← τ
  IA(m) = P(follow invalid advice | high self-ev) ← ν (lower is better)
"""

from __future__ import annotations
import numpy as np

from .internalization_state_v3 import (
    FactoredInternalizationState, compute_factored_utility,
)
from .stochastic_agent_policy import BranchAttributes, AgentPolicyParams


def _softmax_prob(u_target, u_other, beta=4.0, epsilon=0.1):
    """P(choose target) under softmax with lapse."""
    delta = beta * (u_target - u_other)
    delta = np.clip(delta, -20, 20)
    p = float(1.0 / (1.0 + np.exp(-delta)))
    return (1 - epsilon) * p + epsilon * 0.5


# ─── Probe definitions ───

def risk_calibration_probe(m: FactoredInternalizationState,
                           params: AgentPolicyParams, theta="safe") -> float:
    """RC: P(choose safe | risk_penalty=0.4 vs 0.05, no temptation)."""
    safe = BranchAttributes(safety_score=0.6, temptation_score=0.1, risk_penalty=0.05)
    risky = BranchAttributes(safety_score=0.4, temptation_score=0.1, risk_penalty=0.4)
    u_safe = compute_factored_utility(safe, theta, m, params)
    u_risky = compute_factored_utility(risky, theta, m, params)
    return _softmax_prob(u_safe, u_risky, params.beta, params.epsilon)


def temptation_resistance_probe(m: FactoredInternalizationState,
                                params: AgentPolicyParams, theta="safe") -> float:
    """TR: P(reject lure | lure_tempt=0.8, safe_tempt=0.1)."""
    safe = BranchAttributes(safety_score=0.6, temptation_score=0.1, risk_penalty=0.05)
    lure = BranchAttributes(safety_score=0.3, temptation_score=0.8, risk_penalty=0.35)
    u_safe = compute_factored_utility(safe, theta, m, params)
    u_lure = compute_factored_utility(lure, theta, m, params)
    return _softmax_prob(u_safe, u_lure, params.beta, params.epsilon)


def exploration_preservation_probe(m: FactoredInternalizationState,
                                   params: AgentPolicyParams, theta="safe") -> float:
    """EP: P(choose beneficial novel | novel_flag=True, low risk)."""
    familiar = BranchAttributes(safety_score=0.5, temptation_score=0.1, risk_penalty=0.05)
    novel_good = BranchAttributes(safety_score=0.55, temptation_score=0.15, risk_penalty=0.08)
    u_fam = compute_factored_utility(familiar, theta, m, params)
    u_nov = compute_factored_utility(novel_good, theta, m, params, is_novel=True)
    return _softmax_prob(u_nov, u_fam, params.beta, params.epsilon)


def valid_advice_adoption_probe(m: FactoredInternalizationState,
                                params: AgentPolicyParams, theta="safe") -> float:
    """VA: P(follow valid advice | low self-evidence)."""
    advised = BranchAttributes(safety_score=0.45, temptation_score=0.1, risk_penalty=0.05)
    other = BranchAttributes(safety_score=0.5, temptation_score=0.2, risk_penalty=0.15)
    # Valid advice: positive warn bonus + low self-evidence
    u_adv = compute_factored_utility(advised, theta, m, params, warn_bonus=0.4)
    u_oth = compute_factored_utility(other, theta, m, params, warn_bonus=-0.2)
    return _softmax_prob(u_adv, u_oth, params.beta, params.epsilon)


def invalid_advice_resistance_probe(m: FactoredInternalizationState,
                                    params: AgentPolicyParams, theta="safe") -> float:
    """IA: P(follow INVALID advice | strong self-evidence says otherwise).
    Lower is better."""
    bad_advised = BranchAttributes(safety_score=0.3, temptation_score=0.1, risk_penalty=0.35)
    self_good = BranchAttributes(safety_score=0.7, temptation_score=0.1, risk_penalty=0.05)
    # Invalid advice: warns toward bad branch, self-evidence clearly says other
    u_bad = compute_factored_utility(bad_advised, theta, m, params, warn_bonus=0.3)
    u_good = compute_factored_utility(self_good, theta, m, params, warn_bonus=-0.1)
    return _softmax_prob(u_bad, u_good, params.beta, params.epsilon)


def all_probes(m, params, theta="safe") -> dict:
    return {
        "RC": round(risk_calibration_probe(m, params, theta), 4),
        "TR": round(temptation_resistance_probe(m, params, theta), 4),
        "EP": round(exploration_preservation_probe(m, params, theta), 4),
        "VA": round(valid_advice_adoption_probe(m, params, theta), 4),
        "IA": round(invalid_advice_resistance_probe(m, params, theta), 4),
    }


# ─── Behavior target zones ───

BEHAVIOR_ZONES = {
    "safe":  {"RC": (0.55, 0.85), "TR": (0.50, 0.80), "EP": (0.45, 0.70),
              "VA": (0.55, 0.80), "IA": (0.20, 0.45)},
    "shiny": {"RC": (0.55, 0.85), "TR": (0.55, 0.85), "EP": (0.40, 0.65),
              "VA": (0.55, 0.80), "IA": (0.20, 0.45)},
}

BEHAVIOR_WEIGHTS = {"RC": 1.0, "TR": 1.2, "EP": 2.0, "VA": 1.5, "IA": 2.5}


def band_loss(x, lo, hi):
    return max(lo - x, 0.0) ** 2 + max(x - hi, 0.0) ** 2


def behavior_loss(m, params, theta="safe", q_theta=None):
    """L_beh(m, q) using probe behaviors."""
    if q_theta is not None:
        from .stochastic_agent_policy import PREFERENCE_TYPES
        loss = 0.0
        for pi, p in enumerate(PREFERENCE_TYPES):
            probes = all_probes(m, params, p)
            bz = BEHAVIOR_ZONES.get(p, BEHAVIOR_ZONES["safe"])
            l = sum(BEHAVIOR_WEIGHTS[k] * band_loss(probes[k], *bz[k]) for k in BEHAVIOR_WEIGHTS)
            loss += q_theta[pi] * l
        return float(loss)
    probes = all_probes(m, params, theta)
    bz = BEHAVIOR_ZONES.get(theta, BEHAVIOR_ZONES["safe"])
    return float(sum(BEHAVIOR_WEIGHTS[k] * band_loss(probes[k], *bz[k]) for k in BEHAVIOR_WEIGHTS))


def behavior_zone_hit(m, params, theta="safe") -> bool:
    probes = all_probes(m, params, theta)
    bz = BEHAVIOR_ZONES.get(theta, BEHAVIOR_ZONES["safe"])
    return all(bz[k][0] <= probes[k] <= bz[k][1] for k in bz)


def behavior_zone_hit_rate(m, params, theta="safe") -> float:
    if not m.kappa_history:
        return 0.0
    n = len(m.kappa_history)
    hits = 0
    for i in range(n):
        mc = m.copy()
        mc.kappa = m.kappa_history[i]
        mc.tau = m.tau_history[i]
        mc.nu = m.nu_history[i]
        mc.gamma_spec = m.gs_history[i]
        mc.gamma_gen = m.gg_history[i]
        if behavior_zone_hit(mc, params, theta):
            hits += 1
    return hits / n
