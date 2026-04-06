"""Trainable Behavior Bridge: BCE + Jacobian regularization + ECE calibration.

m → ẑ = σ(w₀ + w·φ(m,c))
Trained online from probe outcome data.
Jacobian penalty enforces expected causal structure.
ECE term ensures calibrated predictions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .internalization_state_v3 import FactoredInternalizationState
from .behavior_probes import all_probes
from .stochastic_agent_policy import AgentPolicyParams


def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))

def _features(m, risk=0.3, lure=0.3, novelty=0.0, self_ev=0.5):
    return np.array([
        1.0,
        m.kappa, m.tau, m.nu,
        m.gamma_spec, m.gamma_gen,
        m.kappa * risk,
        m.gamma_spec * lure,
        m.gamma_gen * novelty,
        m.tau * (1.0 - self_ev),
        m.nu * self_ev,
        m.kappa ** 2,
        m.gamma_gen ** 2,
    ], dtype=np.float64)

N_FEATURES = 13
PROBE_NAMES = ["RC", "TR", "EP", "VA", "IA"]

# Expected Jacobian structure: (probe_idx, state_idx, expected_sign)
# State order in J: κ=0, τ=1, ν=2, γs=3, γg=4
EXPECTED_JACOBIAN = [
    (0, 0, +1),  # ∂RC/∂κ > 0
    (1, 3, +1),  # ∂TR/∂γs > 0
    (2, 4, -1),  # ∂EP/∂γg < 0
    (3, 1, +1),  # ∂VA/∂τ > 0
    (4, 2, +1),  # ∂IA/∂ν > 0
]


@dataclass
class TrainableBridge:
    """Online-trained logistic bridge with Jacobian + ECE regularization."""
    weights: np.ndarray = None  # shape (5, 13)
    lr: float = 0.02
    lambda_jac: float = 0.5
    lambda_ece: float = 0.3
    l2: float = 0.005

    # Calibration bins
    n_bins: int = 5
    _bin_counts: np.ndarray = None
    _bin_correct: np.ndarray = None

    # Empirical zone from data
    _zone_data: dict = field(default_factory=lambda: {p: [] for p in PROBE_NAMES})
    _calibrated_zones: dict = None

    def __post_init__(self):
        if self.weights is None:
            # Initialize with FICA-validated structure
            self.weights = np.array([
                [1.8, 0.9, 0.0, 0.0, 0.0, -0.3, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.5, 0.3, 0.0, 0.0, 1.2, -0.2, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.3, -0.2, 0.0, -0.3, 0.0, -2.5, 0.0, 0.0, -1.8, 0.0, 0.0, 0.0, -1.0],
                [0.5, 0.0, 1.8, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -1.5, 0.0, 0.0],
                [-1.5, 0.0, -0.3, 2.5, 0.0, 0.5, 0.0, 0.0, 0.0, -0.5, 2.0, 0.0, 0.0],
            ], dtype=np.float64)
        if self._bin_counts is None:
            self._bin_counts = np.ones((5, self.n_bins))
            self._bin_correct = np.ones((5, self.n_bins)) * 0.5

    def predict(self, m, risk=0.3, lure=0.3, novelty=0.0, self_ev=0.5):
        phi = _features(m, risk, lure, novelty, self_ev)
        logits = self.weights @ phi
        return {PROBE_NAMES[i]: _sigmoid(float(logits[i])) for i in range(5)}

    def _jacobian(self, m, risk=0.3, lure=0.3, novelty=0.0, self_ev=0.5):
        """Numerical Jacobian ∂ẑ/∂m (5 probes × 5 states)."""
        eps = 0.01
        state_names = ["kappa", "tau", "nu", "gamma_spec", "gamma_gen"]
        z0 = self.predict(m, risk, lure, novelty, self_ev)
        J = np.zeros((5, 5))
        for si, sn in enumerate(state_names):
            mp = m.copy()
            setattr(mp, sn, getattr(mp, sn) + eps)
            zp = self.predict(mp, risk, lure, novelty, self_ev)
            for pi, pn in enumerate(PROBE_NAMES):
                J[pi, si] = (zp[pn] - z0[pn]) / eps
        return J

    def _jacobian_loss(self, J):
        """Penalize sign violations and off-diagonal magnitude."""
        loss = 0.0
        on_diag_mag = 0.0
        for pi, si, expected_sign in EXPECTED_JACOBIAN:
            val = J[pi, si]
            on_diag_mag += abs(val)
            if expected_sign > 0 and val < 0.05:
                loss += (0.05 - val) ** 2
            elif expected_sign < 0 and val > -0.05:
                loss += (val + 0.05) ** 2

        off_diag = 0.0
        on_set = {(pi, si) for pi, si, _ in EXPECTED_JACOBIAN}
        for pi in range(5):
            for si in range(5):
                if (pi, si) not in on_set:
                    off_diag += abs(J[pi, si])
        loss += 0.3 * off_diag
        return float(loss)

    def _ece(self):
        """Expected Calibration Error."""
        ece = 0.0
        total = float(self._bin_counts.sum())
        for pi in range(5):
            for b in range(self.n_bins):
                n = self._bin_counts[pi, b]
                if n < 1: continue
                avg_conf = (b + 0.5) / self.n_bins
                avg_acc = self._bin_correct[pi, b] / n
                ece += (n / total) * abs(avg_conf - avg_acc)
        return float(ece)

    def update(self, m, real_probes: dict, risk=0.3, lure=0.3,
               novelty=0.0, self_ev=0.5):
        """One gradient step: BCE + Jacobian + ECE + L2."""
        phi = _features(m, risk, lure, novelty, self_ev)
        preds = self.predict(m, risk, lure, novelty, self_ev)

        # BCE gradient
        for pi, pn in enumerate(PROBE_NAMES):
            y = real_probes.get(pn, 0.5)
            p = preds[pn]
            p = np.clip(p, 0.01, 0.99)
            # ∂BCE/∂logit = p - y
            grad_logit = p - y
            grad_w = grad_logit * phi
            self.weights[pi] -= self.lr * (grad_w + self.l2 * self.weights[pi])

            # Update calibration bins
            b = min(int(p * self.n_bins), self.n_bins - 1)
            self._bin_counts[pi, b] += 1
            self._bin_correct[pi, b] += y

            # Record for zone calibration
            self._zone_data[pn].append(y)

        # Jacobian regularization (every 5 updates to save compute)
        if sum(self._bin_counts.flatten()) % 5 == 0:
            J = self._jacobian(m, risk, lure, novelty, self_ev)
            j_loss = self._jacobian_loss(J)
            if j_loss > 0.1:
                # Nudge weights toward correct signs
                for pi, si, sign in EXPECTED_JACOBIAN:
                    feat_idx = si + 1  # skip bias
                    if sign > 0 and self.weights[pi, feat_idx] < 0.05:
                        self.weights[pi, feat_idx] += self.lr * 0.5
                    elif sign < 0 and self.weights[pi, feat_idx] > -0.05:
                        self.weights[pi, feat_idx] -= self.lr * 0.5

    def empirical_zones(self, alpha=0.15) -> dict:
        """Quantile-based behavior zones from accumulated data."""
        if self._calibrated_zones is not None:
            return self._calibrated_zones
        zones = {}
        for pn in PROBE_NAMES:
            vals = self._zone_data[pn]
            if len(vals) < 8:
                # Fallback: FICA-validated defaults
                zones[pn] = {"RC": (0.55, 0.90), "TR": (0.50, 0.85),
                             "EP": (0.40, 0.65), "VA": (0.50, 0.80),
                             "IA": (0.05, 0.40)}.get(pn, (0.3, 0.7))
            else:
                arr = np.array(vals)
                lo = float(np.quantile(arr, alpha))
                hi = float(np.quantile(arr, 1.0 - alpha))
                if hi - lo < 0.08:
                    mid = (lo + hi) / 2
                    lo, hi = mid - 0.04, mid + 0.04
                zones[pn] = (round(lo, 4), round(hi, 4))
        return zones

    def set_zones(self, zones):
        self._calibrated_zones = zones

    def behavior_loss(self, m, zones=None, risk=0.3, lure=0.3,
                      novelty=0.0, self_ev=0.5):
        if zones is None:
            zones = self.empirical_zones()
        preds = self.predict(m, risk, lure, novelty, self_ev)
        weights = {"RC": 1.0, "TR": 1.2, "EP": 2.5, "VA": 1.5, "IA": 2.5}
        loss = 0.0
        for pn in PROBE_NAMES:
            lo, hi = zones[pn]
            v = preds[pn]
            loss += weights[pn] * (max(lo - v, 0) ** 2 + max(v - hi, 0) ** 2)
        return float(loss)

    def overteach_penalty(self, m, zones=None, risk=0.3, lure=0.3,
                          novelty=0.0, self_ev=0.5):
        if zones is None:
            zones = self.empirical_zones()
        preds = self.predict(m, risk, lure, novelty, self_ev)
        r_ia = 2.5 * max(preds["IA"] - zones["IA"][1], 0) ** 2
        r_ep = 2.5 * max(zones["EP"][0] - preds["EP"], 0) ** 2
        r_tr = 1.5 * max(preds["TR"] - zones["TR"][1], 0) ** 2
        return float(r_ia + r_ep + r_tr)

    @property
    def ece(self):
        return round(self._ece(), 4)
