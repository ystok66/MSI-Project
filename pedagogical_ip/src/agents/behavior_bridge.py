"""Behavior Bridge: calibrated m → ẑ mapping + empirical zone calibration.

Semi-parametric logistic bridge: ẑ_p(m,c) = σ(w₀ + w·φ(m,c))
Features φ include state × context interactions.
Empirical zones from baseline rollout quantiles.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .internalization_state_v3 import FactoredInternalizationState
from .stochastic_agent_policy import AgentPolicyParams
from .behavior_probes import all_probes, BEHAVIOR_ZONES


def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))


def _features(m: FactoredInternalizationState, risk=0.3, lure=0.3,
              novelty=0.0, self_ev=0.5):
    """φ(m, context) → feature vector for bridge prediction."""
    return np.array([
        1.0,                          # bias
        m.kappa, m.tau, m.nu,         # raw state
        m.gamma_spec, m.gamma_gen,
        m.kappa * risk,               # κ × risk
        m.gamma_spec * lure,          # γs × lure
        m.gamma_gen * novelty,        # γg × novelty
        m.tau * (1.0 - self_ev),      # τ × low-self-ev
        m.nu * self_ev,               # ν × high-self-ev
        m.kappa ** 2,                 # quadratic κ
        m.gamma_gen ** 2,             # quadratic γg
    ], dtype=np.float64)


# Pre-calibrated bridge weights (fitted from BI-ICT-v3 rollout patterns)
BRIDGE_WEIGHTS = {
    "RC": np.array([1.8, 0.9, 0.0, 0.0, 0.0, -0.3, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "TR": np.array([1.5, 0.3, 0.0, 0.0, 1.2, -0.2, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "EP": np.array([0.3, -0.2, 0.0, -0.3, 0.0, -2.5, 0.0, 0.0, -1.8, 0.0, 0.0, 0.0, -1.0]),
    "VA": np.array([0.5, 0.0, 1.8, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -1.5, 0.0, 0.0]),
    "IA": np.array([-1.5, 0.0, -0.3, 2.5, 0.0, 0.5, 0.0, 0.0, 0.0, -0.5, 2.0, 0.0, 0.0]),
}


def predict_probe(m, probe_name, risk=0.3, lure=0.3, novelty=0.0, self_ev=0.5):
    """ẑ_p(m, c) = σ(w · φ(m, c))."""
    phi = _features(m, risk, lure, novelty, self_ev)
    w = BRIDGE_WEIGHTS[probe_name]
    return _sigmoid(float(np.dot(w, phi)))


def predict_all_probes(m, risk=0.3, lure=0.3, novelty=0.0, self_ev=0.5):
    """All 5 predicted probe behaviors."""
    return {p: round(predict_probe(m, p, risk, lure, novelty, self_ev), 4)
            for p in BRIDGE_WEIGHTS}


# ─── Empirical Zone Calibration ───

@dataclass
class EmpiricalZoneCalibrator:
    """Calibrate behavior zones from rollout data."""
    probe_data: dict = field(default_factory=lambda: {p: [] for p in BRIDGE_WEIGHTS})
    alpha: float = 0.15  # quantile margin

    def record(self, probes: dict):
        for p, v in probes.items():
            if p in self.probe_data:
                self.probe_data[p].append(v)

    def calibrated_zones(self) -> dict:
        zones = {}
        for p in BRIDGE_WEIGHTS:
            vals = self.probe_data[p]
            if len(vals) < 4:
                zones[p] = BEHAVIOR_ZONES.get("safe", {}).get(p, (0.3, 0.7))
            else:
                arr = np.array(vals)
                lo = float(np.quantile(arr, self.alpha))
                hi = float(np.quantile(arr, 1.0 - self.alpha))
                # Ensure minimum width
                if hi - lo < 0.1:
                    mid = (lo + hi) / 2
                    lo, hi = mid - 0.05, mid + 0.05
                zones[p] = (round(lo, 4), round(hi, 4))
        return zones


def band_loss(x, lo, hi):
    return max(lo - x, 0.0) ** 2 + max(x - hi, 0.0) ** 2


def bridge_behavior_loss(m, zones, risk=0.3, lure=0.3, novelty=0.0, self_ev=0.5):
    """L_beh using bridge-predicted probes and calibrated zones."""
    weights = {"RC": 1.0, "TR": 1.2, "EP": 2.5, "VA": 1.5, "IA": 2.5}
    preds = predict_all_probes(m, risk, lure, novelty, self_ev)
    return float(sum(weights[p] * band_loss(preds[p], *zones[p]) for p in weights))


def bridge_overteach_penalty(m, zones, risk=0.3, lure=0.3, novelty=0.0, self_ev=0.5):
    """R_over using bridge predictions."""
    preds = predict_all_probes(m, risk, lure, novelty, self_ev)
    r_ia = 2.5 * max(preds["IA"] - zones.get("IA", (0, 0.45))[1], 0.0) ** 2
    r_ep = 2.5 * max(zones.get("EP", (0.4, 1))[0] - preds["EP"], 0.0) ** 2
    r_tr = 1.5 * max(preds["TR"] - zones.get("TR", (0, 0.85))[1], 0.0) ** 2
    return float(r_ia + r_ep + r_tr)


def bridge_zone_hit(m, zones, risk=0.3, lure=0.3, novelty=0.0, self_ev=0.5):
    preds = predict_all_probes(m, risk, lure, novelty, self_ev)
    return all(zones[p][0] <= preds[p] <= zones[p][1] for p in zones if p in preds)
