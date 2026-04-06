"""
SlowFast Cost-Risk Head — Task 4: Dual-Timescale Transfer.

Wraps LatentCostRiskHead with a slow-fast weight decomposition:

    θ^(e) = θ_slow + δ_fast^(e)

- θ_slow:  persistent prior that accumulates across episodes
- δ_fast:  per-episode adaptation that starts at 0 each episode

At episode end, the slow weights are updated via EMA:
    θ_slow ← θ_slow + α · δ_fast_end

This addresses the Phase 0 diagnosis: the existing LatentCostRiskHead
is too easily overwritten by within-episode data (~50 cell observations
fully retrain the 4D linear head in ~10 steps), making cross-episode
transfer ineffective.

The slow-fast split ensures that:
1. Fast adaptation gives within-episode optimal performance (unchanged)
2. Slow weights accumulate structural knowledge across episodes
3. At episode start, the prior is warm (not reset to zero)

Shadow mode: SlowFastCostRiskHead can run alongside the canonical
LatentCostRiskHead for comparison, without replacing it.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .cost_risk_model import LatentCostRiskHead


@dataclass
class SlowFastDiagnostics:
    """Per-episode diagnostics for slow-fast analysis."""
    episode_idx: int
    # Weight norms
    slow_cost_w_norm: float
    slow_risk_w_norm: float
    fast_cost_delta_norm: float
    fast_risk_delta_norm: float
    # Prediction quality
    n_updates_this_ep: int
    # Prior advantage: how much did θ_slow predict better than zero at episode start?
    prior_cost_advantage: float  # mean |ĉ_slow - ĉ_zero| on first-seen cells
    prior_risk_advantage: float  # mean |r̂_slow - r̂_zero| on first-seen cells


class SlowFastCostRiskHead:
    """Dual-timescale wrapper around LatentCostRiskHead.

    Decomposes the linear head weights into:
        θ_effective = θ_slow + δ_fast

    The inner LatentCostRiskHead always operates with effective weights.
    This class tracks the decomposition and manages episode boundaries.
    """

    def __init__(
        self,
        d: int = 4,
        alpha: float = 0.1,         # slow EMA rate
        cost_lr_fast: float = 0.1,  # fast learning rate (same as canonical)
        risk_lr_fast: float = 0.3,
        cost_prior_var: float = 1.0,
        risk_prior_var: float = 1.0,
        risk_supervision: str = "oracle_visited",
    ):
        self.d = d
        self.alpha = alpha

        # Slow weights: persistent across episodes
        self._slow_cost_w = np.zeros(d, dtype=np.float64)
        self._slow_cost_b = 1.0  # prior: cost ≈ 1.0
        self._slow_risk_w = np.zeros(d, dtype=np.float64)
        self._slow_risk_b = 0.0

        # Fast delta: reset each episode
        self._fast_cost_w = np.zeros(d, dtype=np.float64)
        self._fast_cost_b = 0.0
        self._fast_risk_w = np.zeros(d, dtype=np.float64)
        self._fast_risk_b = 0.0

        # Inner predictor (operates with effective weights)
        self._inner = LatentCostRiskHead(
            d=d,
            cost_prior_var=cost_prior_var,
            risk_prior_var=risk_prior_var,
            cost_lr=cost_lr_fast,
            risk_lr=risk_lr_fast,
            risk_supervision=risk_supervision,
        )

        # Tracking
        self._episode_count = 0
        self._episode_diagnostics: list[SlowFastDiagnostics] = []

        # Sync effective weights
        self._sync_effective()

    def _sync_effective(self):
        """Set inner predictor weights to θ_slow + δ_fast."""
        self._inner.cost_head.w = self._slow_cost_w + self._fast_cost_w
        self._inner.cost_head.b = self._slow_cost_b + self._fast_cost_b
        self._inner.risk_head.w = self._slow_risk_w + self._fast_risk_w
        self._inner.risk_head.b = self._slow_risk_b + self._fast_risk_b

    def _read_delta(self):
        """Read current δ_fast = θ_effective - θ_slow."""
        self._fast_cost_w = self._inner.cost_head.w - self._slow_cost_w
        self._fast_cost_b = self._inner.cost_head.b - self._slow_cost_b
        self._fast_risk_w = self._inner.risk_head.w - self._slow_risk_w
        self._fast_risk_b = self._inner.risk_head.b - self._slow_risk_b

    # ── Episode lifecycle ──

    def begin_episode(self):
        """Reset fast delta to zero, set effective = slow."""
        self._fast_cost_w[:] = 0.0
        self._fast_cost_b = 0.0
        self._fast_risk_w[:] = 0.0
        self._fast_risk_b = 0.0

        # Reset inner head statistics but keep slow weights
        self._inner.cost_head.n_updates = 0
        self._inner.cost_head.xx_sum[:] = 0.0
        self._inner.cost_head.xy_sum[:] = 0.0
        self._inner.risk_head.n_updates = 0

        # Set effective weights to θ_slow
        self._sync_effective()

    def end_episode(self):
        """Update slow weights: θ_slow += α · δ_fast_end."""
        # Read back the current fast delta
        self._read_delta()

        # Record diagnostics
        diag = SlowFastDiagnostics(
            episode_idx=self._episode_count,
            slow_cost_w_norm=float(np.linalg.norm(self._slow_cost_w)),
            slow_risk_w_norm=float(np.linalg.norm(self._slow_risk_w)),
            fast_cost_delta_norm=float(np.linalg.norm(self._fast_cost_w)),
            fast_risk_delta_norm=float(np.linalg.norm(self._fast_risk_w)),
            n_updates_this_ep=self._inner.n_updates,
            prior_cost_advantage=0.0,  # computed externally if needed
            prior_risk_advantage=0.0,
        )
        self._episode_diagnostics.append(diag)

        # EMA update: θ_slow ← θ_slow + α · δ_fast
        self._slow_cost_w += self.alpha * self._fast_cost_w
        self._slow_cost_b += self.alpha * self._fast_cost_b
        self._slow_risk_w += self.alpha * self._fast_risk_w
        self._slow_risk_b += self.alpha * self._fast_risk_b

        self._episode_count += 1

    # ── LatentCostRiskHead protocol forwarding ──

    @property
    def cost_head(self):
        return self._inner.cost_head

    @property
    def risk_head(self):
        return self._inner.risk_head

    @property
    def risk_supervision(self):
        return self._inner.risk_supervision

    @property
    def n_updates(self):
        return self._inner.n_updates

    def predict_cost(self, x: np.ndarray) -> float:
        return self._inner.predict_cost(x)

    def predict_risk(self, x: np.ndarray) -> float:
        return self._inner.predict_risk(x)

    def predict_cost_uncertainty(self, x: np.ndarray) -> float:
        return self._inner.predict_cost_uncertainty(x)

    def predict_risk_uncertainty(self, x: np.ndarray) -> float:
        return self._inner.predict_risk_uncertainty(x)

    def predict_cost_uncertainty_from_var(self, x_var: np.ndarray) -> float:
        return self._inner.predict_cost_uncertainty_from_var(x_var)

    def predict_risk_uncertainty_from_var(self, x_var: np.ndarray) -> float:
        return self._inner.predict_risk_uncertainty_from_var(x_var)

    def update_from_outcome(self, x, cost_label, risk_label, weight=1.0):
        self._inner.update_from_outcome(x, cost_label, risk_label, weight)

    def reset(self):
        """Full reset (including slow weights)."""
        self._slow_cost_w[:] = 0.0
        self._slow_cost_b = 1.0
        self._slow_risk_w[:] = 0.0
        self._slow_risk_b = 0.0
        self._fast_cost_w[:] = 0.0
        self._fast_cost_b = 0.0
        self._fast_risk_w[:] = 0.0
        self._fast_risk_b = 0.0
        self._inner.reset()
        self._episode_count = 0
        self._episode_diagnostics.clear()

    # ── Diagnostics ──

    @property
    def slow_cost_w(self):
        return self._slow_cost_w.copy()

    @property
    def slow_risk_w(self):
        return self._slow_risk_w.copy()

    def get_diagnostics_summary(self) -> dict:
        """Summary of slow-fast dynamics across episodes."""
        if not self._episode_diagnostics:
            return {}
        diags = self._episode_diagnostics
        return {
            "n_episodes": len(diags),
            "slow_cost_w_norm_final": diags[-1].slow_cost_w_norm,
            "slow_risk_w_norm_final": diags[-1].slow_risk_w_norm,
            "mean_fast_cost_delta": float(np.mean(
                [d.fast_cost_delta_norm for d in diags])),
            "mean_fast_risk_delta": float(np.mean(
                [d.fast_risk_delta_norm for d in diags])),
            "mean_updates_per_ep": float(np.mean(
                [d.n_updates_this_ep for d in diags])),
        }
