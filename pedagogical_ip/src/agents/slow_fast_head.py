"""
GenericSlowFastPredictor — Dual-Timescale Transfer for Arbitrary Predictors.

Wraps any predictor satisfying PredictorProtocol with slow-fast weight
decomposition:

    θ^(e) = θ_slow + δ_fast^(e)

- θ_slow:  persistent prior that accumulates across episodes
- δ_fast:  per-episode adaptation that starts at 0 each episode

Episode lifecycle:
    begin_episode():  P_fast ← copy(P_slow)  [fast starts from slow prior]
    within:          P_fast learns via update_from_outcome()
    end_episode():   θ_slow ← (1-α)·θ_slow + α·θ_fast_end

Uses extract_theta_components() from predictor_protocol for dimension-
agnostic weight manipulation. Works with:
  - LatentCostRiskHead     (4D cost, 4D risk)
  - StructuredBasisCostRiskHead  (6D cost, 7D risk)
  - Any future head implementing PredictorProtocol

Backward compat: SlowFastCostRiskHead = GenericSlowFastPredictor with
LatentCostRiskHead factory.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

from .predictor_protocol import (
    PredictorProtocol,
    snapshot_predictor,
    restore_predictor,
    extract_theta,
    extract_theta_components,
    predictor_summary,
)
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
    # Prior advantage
    prior_cost_advantage: float = 0.0
    prior_risk_advantage: float = 0.0


class GenericSlowFastPredictor:
    """Dual-timescale wrapper around any PredictorProtocol-compatible head.

    Maintains TWO copies of the base predictor:
      - P_slow: accumulated slow weights (persistent across episodes)
      - P_fast: per-episode fast adaptation (reset to P_slow each episode)

    All protocol calls (predict_*, update_*) are forwarded to P_fast.
    At end_episode(), P_slow absorbs α fraction of P_fast's learned weights.

    Weight update formula (component-wise):
        θ_slow ← (1-α)·θ_slow + α·θ_fast_end

    This is equivalent to:
        θ_slow += α·(θ_fast_end - θ_slow)
    """

    def __init__(
        self,
        base_factory: Callable[[], PredictorProtocol],
        alpha: float = 0.1,
        d: int = 4,  # kept for protocol compat, not used internally
    ):
        """
        Args:
            base_factory: Callable that returns a fresh predictor.
                E.g., lambda: StructuredBasisCostRiskHead(d=4)
            alpha: Slow EMA rate. 0 = never accumulate, 1 = full persist.
            d: Raw feature dimension (for protocol compatibility).
        """
        self._base_factory = base_factory
        self.alpha = alpha
        self.d = d

        # Create slow and fast predictors
        self._slow = base_factory()
        self._fast = base_factory()

        # Tracking
        self._episode_count = 0
        self._episode_diagnostics: list[SlowFastDiagnostics] = []

    # ── Episode lifecycle ──

    def begin_episode(self):
        """Reset fast to slow prior: P_fast ← copy(P_slow)."""
        restore_predictor(self._fast, self._slow)
        # Reset sufficient statistics (n_updates, xx_sum, xy_sum)
        # but keep weights from slow prior
        if hasattr(self._fast, 'cost_head'):
            self._fast.cost_head.n_updates = 0
            if hasattr(self._fast.cost_head, 'xx_sum'):
                self._fast.cost_head.xx_sum[:] = 0.0
            if hasattr(self._fast.cost_head, 'xy_sum'):
                self._fast.cost_head.xy_sum[:] = 0.0
        if hasattr(self._fast, 'risk_head'):
            self._fast.risk_head.n_updates = 0
            if hasattr(self._fast.risk_head, 'xx_sum'):
                self._fast.risk_head.xx_sum[:] = 0.0
            if hasattr(self._fast.risk_head, 'xy_sum'):
                self._fast.risk_head.xy_sum[:] = 0.0

    def end_episode(self):
        """Update slow weights: θ_slow ← (1-α)·θ_slow + α·θ_fast_end."""
        # Read slow and fast component weights
        slow_wc, slow_bc, slow_wr, slow_br = extract_theta_components(self._slow)
        fast_wc, fast_bc, fast_wr, fast_br = extract_theta_components(self._fast)

        # Record diagnostics before update
        diag = SlowFastDiagnostics(
            episode_idx=self._episode_count,
            slow_cost_w_norm=float(np.linalg.norm(slow_wc)),
            slow_risk_w_norm=float(np.linalg.norm(slow_wr)),
            fast_cost_delta_norm=float(np.linalg.norm(fast_wc - slow_wc)),
            fast_risk_delta_norm=float(np.linalg.norm(fast_wr - slow_wr)),
            n_updates_this_ep=self._fast.n_updates if hasattr(self._fast, 'n_updates') else 0,
        )
        self._episode_diagnostics.append(diag)

        # EMA update: θ_slow ← (1-α)·θ_slow + α·θ_fast
        α = self.alpha
        new_wc = (1 - α) * slow_wc + α * fast_wc
        new_bc = (1 - α) * slow_bc + α * fast_bc
        new_wr = (1 - α) * slow_wr + α * fast_wr
        new_br = (1 - α) * slow_br + α * fast_br

        # Write back to slow predictor
        self._slow.cost_head.w[:] = new_wc
        self._slow.cost_head.b = float(new_bc)
        self._slow.risk_head.w[:] = new_wr
        self._slow.risk_head.b = float(new_br)

        # Sync sufficient statistics so slow uncertainty improves across episodes.
        # Without this, begin_episode would reset fast n_updates=0 every episode,
        # causing learning_factor = min(1, n_updates/10) to restart from 0.
        for head_name in ('cost_head', 'risk_head'):
            slow_h = getattr(self._slow, head_name)
            fast_h = getattr(self._fast, head_name)
            if hasattr(slow_h, 'xx_sum') and hasattr(fast_h, 'xx_sum'):
                slow_h.xx_sum[:] = (1 - α) * slow_h.xx_sum + α * fast_h.xx_sum
            if hasattr(slow_h, 'xy_sum') and hasattr(fast_h, 'xy_sum'):
                slow_h.xy_sum[:] = (1 - α) * slow_h.xy_sum + α * fast_h.xy_sum
            if hasattr(slow_h, 'n_updates') and hasattr(fast_h, 'n_updates'):
                slow_h.n_updates = max(slow_h.n_updates, fast_h.n_updates)

        self._episode_count += 1

    # ── PredictorProtocol forwarding (all go to P_fast) ──

    @property
    def cost_head(self):
        return self._fast.cost_head

    @property
    def risk_head(self):
        return self._fast.risk_head

    @property
    def risk_supervision(self):
        return getattr(self._fast, 'risk_supervision', 'oracle_visited')

    @property
    def n_updates(self):
        return self._fast.n_updates if hasattr(self._fast, 'n_updates') else 0

    def predict_cost(self, x: np.ndarray) -> float:
        return self._fast.predict_cost(x)

    def predict_risk(self, x: np.ndarray) -> float:
        return self._fast.predict_risk(x)

    def predict_cost_uncertainty(self, x: np.ndarray) -> float:
        return self._fast.predict_cost_uncertainty(x)

    def predict_risk_uncertainty(self, x: np.ndarray) -> float:
        return self._fast.predict_risk_uncertainty(x)

    def predict_cost_uncertainty_from_var(self, x_var: np.ndarray) -> float:
        return self._fast.predict_cost_uncertainty_from_var(x_var)

    def predict_risk_uncertainty_from_var(self, x_var: np.ndarray) -> float:
        return self._fast.predict_risk_uncertainty_from_var(x_var)

    def update_from_outcome(self, x, cost_label, risk_label, weight=1.0):
        self._fast.update_from_outcome(x, cost_label, risk_label, weight)

    def reset(self):
        """Full reset (slow + fast)."""
        self._slow = self._base_factory()
        self._fast = self._base_factory()
        self._episode_count = 0
        self._episode_diagnostics.clear()

    # ── Diagnostics ──

    @property
    def slow_cost_w(self):
        return self._slow.cost_head.w.copy()

    @property
    def slow_risk_w(self):
        return self._slow.risk_head.w.copy()

    def get_diagnostics_summary(self) -> dict:
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


# ════════════════════════════════════════════════════════════════
# Backward-compatible alias
# ════════════════════════════════════════════════════════════════

def SlowFastCostRiskHead(
    d: int = 4,
    alpha: float = 0.1,
    cost_lr_fast: float = 0.1,
    risk_lr_fast: float = 0.3,
    cost_prior_var: float = 1.0,
    risk_prior_var: float = 1.0,
    risk_supervision: str = "oracle_visited",
) -> GenericSlowFastPredictor:
    """Convenience constructor: GenericSlowFastPredictor wrapping LatentCostRiskHead.

    Backward-compatible with the original SlowFastCostRiskHead class.
    """
    def factory():
        return LatentCostRiskHead(
            d=d,
            cost_prior_var=cost_prior_var,
            risk_prior_var=risk_prior_var,
            cost_lr=cost_lr_fast,
            risk_lr=risk_lr_fast,
            risk_supervision=risk_supervision,
        )
    return GenericSlowFastPredictor(base_factory=factory, alpha=alpha, d=d)
