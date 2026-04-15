"""
Structured Basis Cost-Risk Head — Phase 2A.

Replaces the raw 4D linear head with semantically aligned basis functions:

Risk basis φ_r(z):
    [1, z₂, z₃, z₂z₃, |z₂-z₃|, z₁z₂, z₁z₃]   (7D)

Cost basis φ_c(z):
    [1, z₀, z₁, z₂+z₃, z₀z₁, (z₂+z₃)²]           (6D)

Rationale:
- WorldWeights design: z₂,z₃ (texture) are strong risk drivers;
  z₀,z₁ (lane/gate) are weak modulators.
- Cross-terms (z₂z₃, z₁z₂) capture interaction effects that a
  linear head cannot represent.
- The bias is explicitly in the basis, so the linear head has no
  separate bias term.

Implements the full LatentCostRiskHead protocol so it's a drop-in
replacement for planners, robot belief, and intervention policy.
"""

from __future__ import annotations

from typing import Literal
import numpy as np

from .risk_model import BayesianRiskHead, _sigmoid


# ════════════════════════════════════════════════════════════════
# Basis expansion functions
# ════════════════════════════════════════════════════════════════

def risk_basis(z: np.ndarray) -> np.ndarray:
    """Expand 4D feature to 7D risk basis.

    φ_r(z) = [1, z₂, z₃, z₂z₃, |z₂-z₃|, z₁z₂, z₁z₃]
    """
    z0, z1, z2, z3 = z[0], z[1], z[2], z[3]
    return np.array([
        1.0,       # intercept
        z2,        # texture_1 (strong driver)
        z3,        # texture_2 (strong driver)
        z2 * z3,   # texture interaction
        abs(z2 - z3),  # texture dissimilarity
        z1 * z2,   # gate × texture_1
        z1 * z3,   # gate × texture_2
    ], dtype=np.float64)


def cost_basis(z: np.ndarray) -> np.ndarray:
    """Expand 4D feature to 6D cost basis.

    φ_c(z) = [1, z₀, z₁, z₂+z₃, z₀z₁, (z₂+z₃)²]
    """
    z0, z1, z2, z3 = z[0], z[1], z[2], z[3]
    t_sum = z2 + z3
    return np.array([
        1.0,       # intercept
        z0,        # lane_id
        z1,        # gate_flag
        t_sum,     # total texture
        z0 * z1,   # lane × gate
        t_sum ** 2,  # texture quadratic
    ], dtype=np.float64)


def risk_basis_batch(Z: np.ndarray) -> np.ndarray:
    """Batch risk basis expansion: (..., 4) → (..., 7)."""
    shape = Z.shape[:-1]
    Z_flat = Z.reshape(-1, 4)
    out = np.empty((Z_flat.shape[0], 7), dtype=np.float64)
    out[:, 0] = 1.0
    out[:, 1] = Z_flat[:, 2]
    out[:, 2] = Z_flat[:, 3]
    out[:, 3] = Z_flat[:, 2] * Z_flat[:, 3]
    out[:, 4] = np.abs(Z_flat[:, 2] - Z_flat[:, 3])
    out[:, 5] = Z_flat[:, 1] * Z_flat[:, 2]
    out[:, 6] = Z_flat[:, 1] * Z_flat[:, 3]
    return out.reshape(shape + (7,))


def cost_basis_batch(Z: np.ndarray) -> np.ndarray:
    """Batch cost basis expansion: (..., 4) → (..., 6)."""
    shape = Z.shape[:-1]
    Z_flat = Z.reshape(-1, 4)
    t_sum = Z_flat[:, 2] + Z_flat[:, 3]
    out = np.empty((Z_flat.shape[0], 6), dtype=np.float64)
    out[:, 0] = 1.0
    out[:, 1] = Z_flat[:, 0]
    out[:, 2] = Z_flat[:, 1]
    out[:, 3] = t_sum
    out[:, 4] = Z_flat[:, 0] * Z_flat[:, 1]
    out[:, 5] = t_sum ** 2
    return out.reshape(shape + (6,))


# ════════════════════════════════════════════════════════════════
# Structured Basis Heads
# ════════════════════════════════════════════════════════════════

class BasisCostHead:
    """Bayesian linear cost head operating on structured basis features."""

    DIMS = 6  # φ_c output dimension

    def __init__(self, prior_var: float = 1.0, learning_rate: float = 0.1):
        self.d = self.DIMS
        self.w = np.zeros(self.d, dtype=np.float64)
        self.b = 0.0  # NOT used (intercept is in basis); kept for API compat
        self.prior_var = prior_var
        self.lr = learning_rate
        self.n_updates = 0
        self.xx_sum = np.zeros((self.d, self.d), dtype=np.float64)
        self.xy_sum = np.zeros(self.d, dtype=np.float64)

    def predict_cost(self, x: np.ndarray) -> float:
        """Predict cost from RAW 4D feature (applies basis internally)."""
        phi = cost_basis(x)
        return float(max(self.w @ phi + self.b, 0.1))

    def predict_cost_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict cost for (..., 4) feature array."""
        Phi = cost_basis_batch(X)
        shape = Phi.shape[:-1]
        Phi_flat = Phi.reshape(-1, self.d)
        costs = Phi_flat @ self.w + self.b
        return np.clip(costs, 0.1, None).reshape(shape)

    def predict_uncertainty(self, x: np.ndarray) -> float:
        """Approximate predictive uncertainty in basis space."""
        if self.n_updates < 2:
            return 1.0
        phi = cost_basis(x)
        H = self.xx_sum / max(self.n_updates, 1) + np.eye(self.d) / self.prior_var
        try:
            H_inv = np.linalg.inv(H)
            return float(max(phi @ H_inv @ phi, 0.01))
        except np.linalg.LinAlgError:
            return 1.0

    def update_from_label(self, x: np.ndarray, y: float, weight: float = 1.0):
        """Update from (raw_feature, cost_label) pair."""
        phi = cost_basis(x)
        pred = self.w @ phi + self.b
        error = y - pred

        grad_w = -error * phi * weight + self.w / self.prior_var
        grad_b = -error * weight

        grad_norm = float(np.linalg.norm(grad_w))
        max_grad_norm = 5.0
        if not np.isfinite(grad_norm):
            return
        if grad_norm > max_grad_norm:
            grad_w *= max_grad_norm / grad_norm

        self.w -= self.lr * grad_w
        self.b -= self.lr * float(np.clip(grad_b, -max_grad_norm, max_grad_norm))

        max_w_norm = 10.0
        w_norm = float(np.linalg.norm(self.w))
        if w_norm > max_w_norm:
            self.w *= max_w_norm / w_norm

        self.xx_sum += weight * np.outer(phi, phi)
        self.xy_sum += weight * y * phi
        self.n_updates += 1

    def reset(self):
        self.w[:] = 0.0
        self.b = 0.0
        self.xx_sum[:] = 0.0
        self.xy_sum[:] = 0.0
        self.n_updates = 0


class BasisRiskHead:
    """Bayesian linear risk head operating on structured basis features."""

    DIMS = 7  # φ_r output dimension

    def __init__(self, prior_var: float = 1.0, learning_rate: float = 0.3):
        self.d = self.DIMS
        self.w = np.zeros(self.d, dtype=np.float64)
        self.b = 0.0
        self.prior_var = prior_var
        self.lr = learning_rate
        self.n_updates = 0
        self.xx_sum = np.zeros((self.d, self.d), dtype=np.float64)
        self.xy_sum = np.zeros(self.d, dtype=np.float64)

    def predict_risk(self, x: np.ndarray) -> float:
        """Predict risk from RAW 4D feature (applies basis internally)."""
        phi = risk_basis(x)
        logit = self.w @ phi + self.b
        return float(_sigmoid(logit))

    def predict_risk_batch(self, X: np.ndarray) -> np.ndarray:
        Phi = risk_basis_batch(X)
        shape = Phi.shape[:-1]
        Phi_flat = Phi.reshape(-1, self.d)
        logits = Phi_flat @ self.w + self.b
        return _sigmoid(logits).reshape(shape)

    def predict_uncertainty(self, x: np.ndarray) -> float:
        p = self.predict_risk(x)
        if self.n_updates < 2:
            return 0.25
        phi = risk_basis(x)
        H = self.xx_sum / max(self.n_updates, 1) + np.eye(self.d) / self.prior_var
        try:
            H_inv = np.linalg.inv(H)
            logit_var = phi @ H_inv @ phi
            return float(p * (1 - p) * (1 + logit_var))
        except np.linalg.LinAlgError:
            return 0.25

    def update_from_label(self, x: np.ndarray, y: float, weight: float = 1.0):
        """Update from (raw_feature, risk_label) pair."""
        phi = risk_basis(x)
        p = float(_sigmoid(self.w @ phi + self.b))
        error = y - p

        grad_w = -error * phi * weight + self.w / self.prior_var
        grad_b = -error * weight

        grad_norm = float(np.linalg.norm(grad_w))
        max_grad_norm = 5.0
        if not np.isfinite(grad_norm):
            return  # skip update on NaN/Inf gradient
        if grad_norm > max_grad_norm:
            grad_w *= max_grad_norm / grad_norm

        self.w -= self.lr * grad_w
        self.b -= self.lr * float(np.clip(grad_b, -max_grad_norm, max_grad_norm))

        max_w_norm = 10.0
        w_norm = float(np.linalg.norm(self.w))
        if w_norm > max_w_norm:
            self.w *= max_w_norm / w_norm

        self.xx_sum += weight * np.outer(phi, phi)
        self.xy_sum += weight * y * phi
        self.n_updates += 1

    def reset(self):
        self.w[:] = 0.0
        self.b = 0.0
        self.xx_sum[:] = 0.0
        self.xy_sum[:] = 0.0
        self.n_updates = 0


# ════════════════════════════════════════════════════════════════
# Composite Head — LatentCostRiskHead protocol
# ════════════════════════════════════════════════════════════════

class StructuredBasisCostRiskHead:
    """Joint Bayesian predictor with structured basis expansion.

    Drop-in replacement for LatentCostRiskHead. Takes raw 4D features,
    applies basis expansion internally, then predicts via linear heads
    in basis space.

    Parameters: 6 (cost) + 7 (risk) = 13 vs original 4 + 4 = 8.
    """

    def __init__(
        self,
        d: int = 4,  # raw feature dim (must be 4)
        cost_prior_var: float = 1.0,
        risk_prior_var: float = 1.0,
        cost_lr: float = 0.1,
        risk_lr: float = 0.3,
        risk_supervision: str = "oracle_visited",
    ):
        assert d == 4, f"StructuredBasisCostRiskHead requires d=4, got {d}"
        self.d = d  # raw feature dim (for protocol compat)
        self.cost_head = BasisCostHead(
            prior_var=cost_prior_var, learning_rate=cost_lr)
        self.risk_head = BasisRiskHead(
            prior_var=risk_prior_var, learning_rate=risk_lr)
        self.risk_supervision = risk_supervision

    def predict_cost(self, x: np.ndarray) -> float:
        return self.cost_head.predict_cost(x)

    def predict_risk(self, x: np.ndarray) -> float:
        return self.risk_head.predict_risk(x)

    def predict_cost_uncertainty(self, x: np.ndarray) -> float:
        return self.cost_head.predict_uncertainty(x)

    def predict_risk_uncertainty(self, x: np.ndarray) -> float:
        return self.risk_head.predict_uncertainty(x)

    def predict_cost_uncertainty_from_var(self, x_var: np.ndarray) -> float:
        """Jacobian-based cost uncertainty from raw feature variance.

        Propagates Σ_z = diag(x_var) through the cost basis Jacobian:
            g_c(z) = J_φc(z)ᵀ · w_c
            Var[η_c] ≈ g_c(z)ᵀ · Σ_z · g_c(z)
            u_c ≈ softplus'(η_c) · √Var[η_c]
        """
        # Use z_mean = 0.5 as operating point for Jacobian
        z = np.full(4, 0.5)
        z0, z1, z2, z3 = z[0], z[1], z[2], z[3]
        t_sum = z2 + z3
        w = self.cost_head.w  # (6,)

        # Jacobian J_φc: (6, 4) — ∂φ_c[i]/∂z[j]
        # φ_c = [1, z₀, z₁, z₂+z₃, z₀z₁, (z₂+z₃)²]
        J = np.zeros((6, 4), dtype=np.float64)
        # ∂φ/∂z₀: [0, 1, 0, 0, z₁, 0]
        J[1, 0] = 1.0; J[4, 0] = z1
        # ∂φ/∂z₁: [0, 0, 1, 0, z₀, 0]
        J[2, 1] = 1.0; J[4, 1] = z0
        # ∂φ/∂z₂: [0, 0, 0, 1, 0, 2(z₂+z₃)]
        J[3, 2] = 1.0; J[5, 2] = 2 * t_sum
        # ∂φ/∂z₃: [0, 0, 0, 1, 0, 2(z₂+z₃)]
        J[3, 3] = 1.0; J[5, 3] = 2 * t_sum

        # g_c = Jᵀ · w → (4,)
        g = J.T @ w

        # Var[η_c] = gᵀ · diag(x_var) · g
        d_use = min(4, len(x_var))
        var_eta = float(np.sum(g[:d_use] ** 2 * x_var[:d_use]))

        # softplus'(η) = σ(η)
        eta = float(w @ cost_basis(z) + self.cost_head.b)
        sp_deriv = float(1.0 / (1.0 + np.exp(-eta)))

        return float(max(sp_deriv * np.sqrt(max(var_eta, 0)), 0.001))

    def predict_risk_uncertainty_from_var(self, x_var: np.ndarray) -> float:
        """Jacobian-based risk uncertainty from raw feature variance.

        Propagates Σ_z = diag(x_var) through the risk basis Jacobian:
            g_r(z) = J_φr(z)ᵀ · w_r
            Var[η_r] ≈ g_r(z)ᵀ · Σ_z · g_r(z)
            u_r ≈ σ'(η_r) · √Var[η_r]
        """
        z = np.full(4, 0.5)
        z0, z1, z2, z3 = z[0], z[1], z[2], z[3]
        w = self.risk_head.w  # (7,)

        # Jacobian J_φr: (7, 4) — ∂φ_r[i]/∂z[j]
        # φ_r = [1, z₂, z₃, z₂z₃, |z₂-z₃|, z₁z₂, z₁z₃]
        J = np.zeros((7, 4), dtype=np.float64)
        # ∂φ/∂z₁: [0, 0, 0, 0, 0, z₂, z₃]
        J[5, 1] = z2; J[6, 1] = z3
        # ∂φ/∂z₂: [0, 1, 0, z₃, sign(z₂-z₃), z₁, 0]
        J[1, 2] = 1.0; J[3, 2] = z3
        J[4, 2] = np.sign(z2 - z3) if z2 != z3 else 0.0
        J[5, 2] = z1
        # ∂φ/∂z₃: [0, 0, 1, z₂, -sign(z₂-z₃), 0, z₁]
        J[2, 3] = 1.0; J[3, 3] = z2
        J[4, 3] = -np.sign(z2 - z3) if z2 != z3 else 0.0
        J[6, 3] = z1

        # g_r = Jᵀ · w → (4,)
        g = J.T @ w

        # Var[η_r] = gᵀ · diag(x_var) · g
        d_use = min(4, len(x_var))
        var_eta = float(np.sum(g[:d_use] ** 2 * x_var[:d_use]))

        # σ'(η) = σ(η) · (1-σ(η))
        phi = risk_basis(z)
        eta = float(w @ phi + self.risk_head.b)
        p = float(1.0 / (1.0 + np.exp(-eta)))
        sigma_prime = p * (1 - p)

        return float(max(sigma_prime * np.sqrt(max(var_eta, 0)), 0.001))

    def update_from_outcome(self, x, cost_label, risk_label, weight=1.0):
        self.cost_head.update_from_label(x, cost_label, weight=weight)
        self.risk_head.update_from_label(x, risk_label, weight=weight)

    def reset(self):
        self.cost_head.reset()
        self.risk_head.reset()

    @property
    def n_updates(self):
        return self.risk_head.n_updates
