"""V5.5 — Mixed-Effects Risk Head.

Extends the standard risk head with context-specific residuals and
shrinkage prior to suppress map-/side-specific shortcut bias.

Architecture:
    r̂_i = σ((w_shared + δ_c)ᵀ x_i + b_shared + b_c)

    where c is the context (e.g., mirror_side, map_bucket)
    and δ_c ~ N(0, σ²_δ I) is the shrinkage prior.

MAP objective:
    L = L_risk + λ_δ |δ_c|²

This forces the model to put shared structure into w_shared and
only use residuals for genuinely context-specific adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .risk_model import _sigmoid


@dataclass
class ContextResidual:
    """Per-context weight residual with shrinkage."""
    delta_w: np.ndarray     # (d,) residual weights
    delta_b: float = 0.0    # residual bias
    n_updates: int = 0


class MixedEffectsRiskHead:
    """Risk head with shared weights + context-specific residuals.

    Shrinkage prior on residuals suppresses overfitting to
    position/side/layout shortcuts.
    """

    def __init__(
        self,
        d: int = 4,
        prior_var: float = 1.0,
        learning_rate: float = 0.3,
        lambda_delta: float = 1.0,
        delta_lr_scale: float = 0.5,
    ):
        """
        Parameters
        ----------
        d : feature dimension
        prior_var : shared weight prior variance
        learning_rate : shared weight learning rate
        lambda_delta : shrinkage strength on residuals
        delta_lr_scale : residual learning rate = lr * delta_lr_scale
        """
        self.d = d
        self.w_shared = np.zeros(d, dtype=np.float64)
        self.b_shared = 0.0
        self.prior_var = prior_var
        self.lr = learning_rate
        self.lambda_delta = lambda_delta
        self.delta_lr = learning_rate * delta_lr_scale

        self.contexts: dict[str, ContextResidual] = {}
        self.n_updates = 0

        # Sufficient statistics for shared uncertainty
        self.xx_sum = np.zeros((d, d), dtype=np.float64)

    def _get_context(self, ctx: str) -> ContextResidual:
        """Get or create context residual."""
        if ctx not in self.contexts:
            self.contexts[ctx] = ContextResidual(
                delta_w=np.zeros(self.d, dtype=np.float64),
                delta_b=0.0,
            )
        return self.contexts[ctx]

    def predict_risk(self, x: np.ndarray, ctx: Optional[str] = None
                     ) -> float:
        """Predict risk probability.

        If ctx is None, uses shared weights only.
        """
        w = self.w_shared.copy()
        b = self.b_shared
        if ctx is not None:
            cr = self._get_context(ctx)
            w = w + cr.delta_w
            b = b + cr.delta_b
        logit = w @ x + b
        return float(_sigmoid(logit))

    def predict_risk_batch(self, X: np.ndarray, ctx: Optional[str] = None
                           ) -> np.ndarray:
        """Predict risk for (N, d) or (H, W, d) array."""
        shape = X.shape[:-1]
        X_flat = X.reshape(-1, self.d)
        w = self.w_shared.copy()
        b = self.b_shared
        if ctx is not None:
            cr = self._get_context(ctx)
            w = w + cr.delta_w
            b = b + cr.delta_b
        logits = X_flat @ w + b
        return _sigmoid(logits).reshape(shape)

    def predict_uncertainty(self, x: np.ndarray) -> float:
        """Approximate predictive uncertainty (shared model only)."""
        p = float(_sigmoid(self.w_shared @ x + self.b_shared))
        if self.n_updates < 2:
            return 0.25
        H = self.xx_sum / max(self.n_updates, 1) + np.eye(self.d) / self.prior_var
        try:
            H_inv = np.linalg.inv(H)
            logit_var = x @ H_inv @ x
            return float(p * (1 - p) * (1 + logit_var))
        except np.linalg.LinAlgError:
            return 0.25

    def update_from_label(
        self,
        x: np.ndarray,
        y: float,
        ctx: Optional[str] = None,
        weight: float = 1.0,
    ):
        """Online MAP update with shrinkage on residuals.

        Updates both shared weights and context residual.
        The residual update includes the L2 shrinkage penalty:
            ∇_δ L = -(y-p)·x + δ/σ²_δ  (via lambda_delta)
        """
        # Effective weights
        w = self.w_shared.copy()
        b = self.b_shared
        cr = None
        if ctx is not None:
            cr = self._get_context(ctx)
            w = w + cr.delta_w
            b = b + cr.delta_b

        p = float(_sigmoid(w @ x + b))
        error = y - p

        # ── Shared weight update ──
        grad_w_shared = -error * x * weight + self.w_shared / self.prior_var
        grad_b_shared = -error * weight

        # Clip
        gn = float(np.linalg.norm(grad_w_shared))
        if gn > 5.0:
            grad_w_shared *= 5.0 / gn

        self.w_shared -= self.lr * grad_w_shared
        self.b_shared -= self.lr * float(np.clip(grad_b_shared, -5.0, 5.0))

        # Weight norm clamping
        wn = float(np.linalg.norm(self.w_shared))
        if wn > 10.0:
            self.w_shared *= 10.0 / wn

        # ── Residual update (with shrinkage) ──
        if cr is not None:
            # Gradient includes shrinkage: λ_δ · δ_c
            grad_delta = -error * x * weight + self.lambda_delta * cr.delta_w
            grad_db = -error * weight + self.lambda_delta * cr.delta_b

            gn_d = float(np.linalg.norm(grad_delta))
            if gn_d > 5.0:
                grad_delta *= 5.0 / gn_d

            cr.delta_w -= self.delta_lr * grad_delta
            cr.delta_b -= self.delta_lr * float(np.clip(grad_db, -5.0, 5.0))

            # Residual norm clamping
            dn = float(np.linalg.norm(cr.delta_w))
            if dn > 3.0:
                cr.delta_w *= 3.0 / dn
            cr.n_updates += 1

        self.xx_sum += weight * np.outer(x, x)
        self.n_updates += 1

    def shared_norm(self) -> float:
        """Norm of shared weights."""
        return float(np.linalg.norm(self.w_shared))

    def residual_norm(self, ctx: str) -> float:
        """Norm of context residual."""
        if ctx in self.contexts:
            return float(np.linalg.norm(self.contexts[ctx].delta_w))
        return 0.0

    def side_bias(self) -> float:
        """Measure of side-specific bias: max |δ_c| across contexts."""
        if not self.contexts:
            return 0.0
        return max(self.residual_norm(c) for c in self.contexts)

    def reset(self):
        self.w_shared[:] = 0.0
        self.b_shared = 0.0
        self.contexts.clear()
        self.xx_sum[:] = 0.0
        self.n_updates = 0
