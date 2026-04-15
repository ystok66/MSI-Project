"""
danger_head.py — V2: Hazard Head + Severity Head.

Two-layer risk learning:
  Head 1 (Hazard): p_h(v) = P(d > 0 | v) — binary safe/risky classifier
  Head 2 (Severity): μ_s(v) = E[d | v, d>0] — regression on severity

Learning signals:
  - Wrong-pick reveal (v, d):  updates both heads
  - RISK_HINT (v, η=0.8):     updates hazard head only (weak label)
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


def _feature_expand(v: np.ndarray) -> np.ndarray:
    """Feature expansion: φ(v) = [v; v²; 1]."""
    return np.concatenate([v, v * v, [1.0]])


class HazardHead:
    """Binary hazard classifier: P(d > 0 | v).

    Online logistic regression with Bayesian-ish updates.
    """

    def __init__(self, m: int, prior_var: float = 1.0, lr: float = 0.1):
        self.m = m
        self.d = 2 * m + 1  # expanded feature dim
        self.lr = lr
        self.w = np.zeros(self.d)  # logistic weights
        # Safe-biased prior: σ(-1) ≈ 0.27 instead of σ(0) = 0.5
        # Untrained learner assumes options are probably safe
        self.w[-1] = -1.0  # bias term toward safe
        self.prior_var = prior_var
        self._n_updates = 0

    def predict(self, v: np.ndarray) -> float:
        """P(risky | v) = σ(w^T φ(v))."""
        phi = _feature_expand(v)
        return float(1.0 / (1.0 + np.exp(-self.w @ phi[:len(self.w)])))

    def update(self, v: np.ndarray, y_h: float) -> None:
        """Online update with binary label y_h ∈ [0, 1].

        For reveal: y_h = 1.0 if d > 0 else 0.0
        For RISK_HINT: y_h = η_hint (e.g. 0.8)
        """
        phi = _feature_expand(v)
        p = self.predict(v)
        # Gradient of cross-entropy: dL/dw = (p - y) * φ
        grad = (p - y_h) * phi[:len(self.w)]
        # L2 regularization
        reg = self.w / self.prior_var
        self.w -= self.lr * (grad + reg / max(self._n_updates + 1, 1))
        self._n_updates += 1

    @property
    def w_mean(self) -> np.ndarray:
        return self.w.copy()


class SeverityHead:
    """Severity regression: E[d | v, d > 0].

    Bayesian linear regression, only updated on risky (d > 0) samples.
    """

    def __init__(self, m: int, prior_var: float = 1.0, lr: float = 0.1):
        self.m = m
        self.d = 2 * m + 1
        self.lr = lr
        self.w = np.zeros(self.d)
        self.prior_var = prior_var
        self._n_updates = 0

    def predict(self, v: np.ndarray) -> Tuple[float, float]:
        """Predict expected severity and uncertainty.

        Returns:
            (mu_s, u_s): predicted damage given risky, uncertainty
        """
        phi = _feature_expand(v)
        mu = float(self.w @ phi[:len(self.w)])
        # Clamp to valid range [1, 4]
        mu = max(1.0, min(4.0, mu))
        # Simple uncertainty: decreases with more updates
        u = 1.0 / (1.0 + 0.1 * self._n_updates)
        return mu, u

    def update(self, v: np.ndarray, damage: int) -> None:
        """Online update with observed (v, d) where d > 0.

        Only called for risky observations.
        """
        phi = _feature_expand(v)
        mu, _ = self.predict(v)
        # Gradient descent on squared error
        grad = (mu - damage) * phi[:len(self.w)]
        reg = self.w / self.prior_var
        self.w -= self.lr * (grad + reg / max(self._n_updates + 1, 1))
        self._n_updates += 1

    @property
    def w_mean(self) -> np.ndarray:
        return self.w.copy()


class DangerHead:
    """V2 composite danger head: hazard + severity.

    Combines HazardHead and SeverityHead for full risk prediction.
    Provides backward-compatible API.
    """

    def __init__(self, m: int, prior_var: float = 1.0, lr: float = 0.1):
        self.m = m
        self._prior_var = prior_var
        self._lr = lr
        self.hazard = HazardHead(m, prior_var, lr)
        self.severity = SeverityHead(m, prior_var, lr)

    def reset(self) -> None:
        """Reset to prior state. [V1 compat]"""
        self.hazard = HazardHead(self.m, self._prior_var, self._lr)
        self.severity = SeverityHead(self.m, self._prior_var, self._lr)

    def predict(self, v: np.ndarray) -> Tuple[float, float]:
        """Predict expected damage and uncertainty.

        μ_d(v) = p_h(v) · μ_s(v)
        u_d(v) = p_h(v) · u_s(v) + (1-p_h) · 0

        Returns:
            (mu_d, u_d): expected damage, uncertainty
        """
        p_h = self.hazard.predict(v)
        mu_s, u_s = self.severity.predict(v)
        mu_d = p_h * mu_s
        u_d = p_h * u_s
        return mu_d, u_d

    def predict_ko_prob(self, v: np.ndarray, hp: int) -> float:
        """P(d >= HP | v) — probability of being KO'd.

        p_ko = p_h(v) · P(d >= HP | risky)
        Simplified: p_ko = p_h(v) · 1[μ_s(v) >= HP]
        """
        p_h = self.hazard.predict(v)
        mu_s, _ = self.severity.predict(v)
        if mu_s >= hp:
            return p_h
        # Graded: linear interpolation between μ_s/HP threshold
        return p_h * max(0.0, mu_s / max(hp, 1))

    def update(self, v: np.ndarray, damage: int) -> None:
        """Full update from observed (v, d) after wrong pick.

        Updates both hazard and severity heads.
        """
        # Hazard: binary label
        y_h = 1.0 if damage > 0 else 0.0
        self.hazard.update(v, y_h)

        # Severity: only update if risky
        if damage > 0:
            self.severity.update(v, damage)

    def update_from_hint(self, v: np.ndarray, eta: float = 0.8) -> None:
        """Update hazard head from RISK_HINT (weak label). [V2]

        Does NOT update severity head (no exact damage info).
        """
        self.hazard.update(v, eta)

    def update_from_ban(self, v: np.ndarray, omega_ban: float = 3.0) -> None:
        """Update hazard head from BAN signal (RSA mode).

        BAN semantics: option is risk-unacceptable.
        This is a stronger signal than RISK_HINT (η=0.8) but weaker
        than a confirmed wrong-pick reveal (y_h=1.0 with full lr).

        Implementation:
            - Uses y_h = 1.0 (definite risk label)
            - Scales learning rate by omega_ban / (1 + omega_ban)
              so the update magnitude matches the RSA logit shift ω_ban
            - Does NOT update severity head (no exact damage info)

        Cross-query effect: makes danger_head avoid similar options
        in future queries when ban_teaches_risk=True.

        Args:
            v: danger vector of the banned option
            omega_ban: RSA BAN strength (from RSAConfig.omega_ban)
        """
        scale = float(omega_ban) / (1.0 + float(omega_ban))
        saved_lr = self.hazard.lr
        self.hazard.lr = saved_lr * scale
        self.hazard.update(v, 1.0)  # y_h = 1.0: definite risk label
        self.hazard.lr = saved_lr

    @property
    def w_mean(self) -> np.ndarray:
        """Legacy: combined weight vector for snapshot."""
        return self.hazard.w_mean


def create_danger_head(m: int = 16, **kwargs) -> DangerHead:
    """Factory function for DangerHead. [V1 compat shim]"""
    return DangerHead(m=m, **kwargs)
