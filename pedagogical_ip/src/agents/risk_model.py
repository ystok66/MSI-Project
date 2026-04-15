"""
Bayesian Risk Head — feature-to-risk predictor using Bayesian linear model.

The agent learns to predict risk from feature vectors, NOT from direct
risk observations. This implements a shared linear model:

  ρ_hat = σ(w · x + b)   where σ = sigmoid

Updated online from outcome labels:
  - y = 1 if agent stepped on trap and died
  - y = risk_value if agent survived but encountered mild risk
  - y = 0 if cell was safe

Uses MAP estimation with L2 prior (Gaussian weight prior).
"""

from __future__ import annotations

import numpy as np


class BayesianRiskHead:
    """Bayesian linear model for feature → risk prediction."""

    def __init__(self, d: int = 4, prior_var: float = 1.0,
                 learning_rate: float = 0.3):
        """
        Args:
            d: feature dimension
            prior_var: variance of Gaussian weight prior
            learning_rate: step size for online SGLD/MAP updates
        """
        self.d = d
        self.w = np.zeros(d, dtype=np.float64)  # weight vector
        self.b = 0.0  # bias
        self.prior_var = prior_var
        self.lr = learning_rate

        # Sufficient statistics for batch update
        self.n_updates = 0
        self.xx_sum = np.zeros((d, d), dtype=np.float64)  # X^T X
        self.xy_sum = np.zeros(d, dtype=np.float64)        # X^T y

    def predict_risk(self, x: np.ndarray) -> float:
        """Predict risk probability from feature vector."""
        logit = self.w @ x + self.b
        return _sigmoid(logit)

    def predict_risk_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict risk for (H, W, d) or (N, d) feature array."""
        shape = X.shape[:-1]
        X_flat = X.reshape(-1, self.d)
        logits = X_flat @ self.w + self.b
        return _sigmoid(logits).reshape(shape)

    def predict_uncertainty(self, x: np.ndarray) -> float:
        """
        Approximate predictive uncertainty using Laplace approximation.
        Returns variance of the predictive distribution.
        """
        p = self.predict_risk(x)
        # Gradient of logistic: p(1-p)
        # Hessian approx: X^T diag(p(1-p)) X + I/prior_var
        # For single point: uncertainty ≈ x^T H^{-1} x * p(1-p)
        if self.n_updates < 2:
            return 0.25  # maximum uncertainty (flat prior)

        # Use empirical Hessian
        H = self.xx_sum / max(self.n_updates, 1) + np.eye(self.d) / self.prior_var
        try:
            H_inv = np.linalg.inv(H)
            logit_var = x @ H_inv @ x
            return float(p * (1 - p) * (1 + logit_var))
        except np.linalg.LinAlgError:
            return 0.25

    def update_from_label(self, x: np.ndarray, y: float, weight: float = 1.0):
        """
        Online MAP update from (feature, label) pair.

        x: feature vector (d,)
        y: risk label (0=safe, 1=fatal, 0.15-0.25=mild risk)
        weight: importance weight
        """
        p = self.predict_risk(x)
        error = y - p

        # SGD on negative log-posterior
        # ∇_w NLL = -(y - p) * x + w / prior_var
        grad_w = -error * x * weight + self.w / self.prior_var
        grad_b = -error * weight

        # Gradient clipping: prevent weight explosions
        grad_norm = float(np.linalg.norm(grad_w))
        max_grad_norm = 5.0
        if not np.isfinite(grad_norm):
            return  # skip update on NaN/Inf gradient
        if grad_norm > max_grad_norm:
            grad_w *= max_grad_norm / grad_norm

        self.w -= self.lr * grad_w
        self.b -= self.lr * float(np.clip(grad_b, -max_grad_norm, max_grad_norm))

        # Weight norm clamping: prevent downstream NaN
        max_w_norm = 10.0
        w_norm = float(np.linalg.norm(self.w))
        if w_norm > max_w_norm:
            self.w *= max_w_norm / w_norm

        # Update sufficient statistics
        self.xx_sum += weight * np.outer(x, x)
        self.xy_sum += weight * y * x
        self.n_updates += 1

    def reset(self):
        """Reset to prior."""
        self.w[:] = 0.0
        self.b = 0.0
        self.xx_sum[:] = 0.0
        self.xy_sum[:] = 0.0
        self.n_updates = 0


def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))
