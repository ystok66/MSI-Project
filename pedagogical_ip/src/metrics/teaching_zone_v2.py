"""Teaching Zone v2: 5-component target intervals + path-sensitive value.

Z_θ = [κ_lo,κ_hi] × [τ_lo,τ_hi] × [ν_lo,ν_hi] × [γs_lo,γs_hi] × [γg_lo,γg_hi]

V_teach(a) = ΔL_zone(a) + λ_sd·ΔSD(a) - λ_dep·ΔDEP(a)
"""

from __future__ import annotations
import numpy as np


TARGET_ZONES_V2 = {
    "safe": {
        "kappa": (0.8, 1.8), "tau": (0.2, 0.8), "nu": (0.0, 0.25),
        "gamma_spec": (0.0, 0.2), "gamma_gen": (0.0, 0.1),
    },
    "shiny": {
        "kappa": (1.0, 2.0), "tau": (0.2, 0.8), "nu": (0.0, 0.25),
        "gamma_spec": (0.15, 0.5), "gamma_gen": (0.0, 0.15),
    },
    "neutral": {
        "kappa": (0.8, 1.5), "tau": (0.2, 0.7), "nu": (0.0, 0.2),
        "gamma_spec": (0.0, 0.15), "gamma_gen": (0.0, 0.1),
    },
    "shortcut": {
        "kappa": (1.0, 2.0), "tau": (0.3, 0.8), "nu": (0.0, 0.2),
        "gamma_spec": (0.1, 0.4), "gamma_gen": (0.0, 0.12),
    },
    "explorer": {
        "kappa": (0.6, 1.5), "tau": (0.2, 0.7), "nu": (0.0, 0.2),
        "gamma_spec": (0.0, 0.15), "gamma_gen": (0.0, 0.08),
    },
}

ZONE_WEIGHTS_V2 = {
    "kappa": 1.0, "tau": 0.8, "nu": 2.0,
    "gamma_spec": 1.2, "gamma_gen": 2.5,
}


def band_loss(x, lo, hi):
    return max(lo - x, 0.0) ** 2 + max(x - hi, 0.0) ** 2


def teaching_loss_v2(m, theta="safe", q_theta=None):
    """L_zone(m, q) = Σ_θ q(θ)·Σ_j w_j·φ(m_j; Z_θ_j)."""
    if q_theta is not None:
        from ..agents.stochastic_agent_policy import PREFERENCE_TYPES
        loss = 0.0
        for pi, p in enumerate(PREFERENCE_TYPES):
            z = TARGET_ZONES_V2.get(p, TARGET_ZONES_V2["safe"])
            l = sum(ZONE_WEIGHTS_V2[k] * band_loss(getattr(m, k, m.get(k, 0.5))
                    if isinstance(m, dict) else getattr(m, k, 0.5), *z[k])
                    for k in ZONE_WEIGHTS_V2)
            loss += q_theta[pi] * l
        return float(loss)
    z = TARGET_ZONES_V2.get(theta, TARGET_ZONES_V2["safe"])
    return float(sum(
        ZONE_WEIGHTS_V2[k] * band_loss(
            getattr(m, k, m.get(k, 0.5)) if isinstance(m, dict) else getattr(m, k, 0.5),
            *z[k])
        for k in ZONE_WEIGHTS_V2))


def in_zone_v2(m, theta="safe"):
    z = TARGET_ZONES_V2.get(theta, TARGET_ZONES_V2["safe"])
    result = {}
    all_in = True
    for k in ZONE_WEIGHTS_V2:
        v = getattr(m, k, 0.5) if not isinstance(m, dict) else m.get(k, 0.5)
        inside = z[k][0] <= v <= z[k][1]
        result[f"{k}_in"] = inside
        if not inside:
            all_in = False
    result["all_in"] = all_in
    return result


def zone_hit_rate_v2(m):
    if not m.kappa_history:
        return 0.0
    n = len(m.kappa_history)
    hits = 0
    for i in range(n):
        d = {"kappa": m.kappa_history[i], "tau": m.tau_history[i],
             "nu": m.nu_history[i],
             "gamma_spec": m.gs_history[i], "gamma_gen": m.gg_history[i]}
        if in_zone_v2(d)["all_in"]:
            hits += 1
    return hits / n


def overteach_rate_v2(m):
    """Per-component overteach rates for factored state."""
    if not m.kappa_history:
        return {"nu_over": 0, "gs_over": 0, "gg_over": 0, "total": 0}
    n = len(m.kappa_history)
    nu_over = sum(1 for v in m.nu_history if v > 0.3) / n
    gs_over = sum(1 for v in m.gs_history if v > 0.55) / n
    gg_over = sum(1 for v in m.gg_history if v > 0.2) / n
    total = sum(1 for i in range(n)
                if m.nu_history[i] > 0.3 or m.gs_history[i] > 0.55
                or m.gg_history[i] > 0.2) / n
    return {"nu_over": round(nu_over, 3), "gs_over": round(gs_over, 3),
            "gg_over": round(gg_over, 3), "total": round(total, 3)}


def path_sensitive_teaching_value(
    L_now, L_next_warn, L_next_wait,
    p_self_discovery: float, p_blind_obey: float,
    lambda_sd: float = 1.5, lambda_dep: float = 2.0,
):
    """V_teach(a) = ΔL_zone + λ_sd·SD - λ_dep·DEP."""
    v_warn = (L_now - L_next_warn) - lambda_dep * p_blind_obey
    v_wait = (L_now - L_next_wait) + lambda_sd * p_self_discovery
    return v_warn, v_wait


def overteach_penalty_v2(m_next, theta="safe"):
    z = TARGET_ZONES_V2.get(theta, TARGET_ZONES_V2["safe"])
    nu_val = m_next.nu if hasattr(m_next, 'nu') else m_next.get("nu", 0)
    gs_val = m_next.gamma_spec if hasattr(m_next, 'gamma_spec') else m_next.get("gamma_spec", 0)
    gg_val = m_next.gamma_gen if hasattr(m_next, 'gamma_gen') else m_next.get("gamma_gen", 0)
    return float(
        2.0 * max(nu_val - z["nu"][1], 0.0) ** 2
        + 1.5 * max(gs_val - z["gamma_spec"][1], 0.0) ** 2
        + 3.0 * max(gg_val - z["gamma_gen"][1], 0.0) ** 2)
