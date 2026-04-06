"""Shadow Observer Types — named-field data structures for Step 3.

All dimensions are accessed by NAME, never by positional index.
Canonical ordering for display: tau, nu, gamma_gen, gamma_spec, kappa.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np


# ── Dimension names (canonical) ──────────────────────────────
DIM_NAMES = ("tau", "nu", "gamma_gen", "gamma_spec", "kappa")

# ── Dimension bounds (from frozen observer / true state) ──────
DIM_BOUNDS = {
    "tau":        (0.0, 1.0),
    "nu":         (0.0, 0.8),
    "gamma_gen":  (0.0, 0.5),
    "gamma_spec": (0.0, 1.0),
    "kappa":      (0.0, 1.0),   # observer range; true state has [0.3, 3.0]
}

# ── Default priors (from frozen A1 observer) ──────────────────
DIM_PRIORS = {
    "tau":        0.3,
    "nu":         0.1,
    "gamma_gen":  0.0,
    "gamma_spec": 0.0,
    "kappa":      0.3,
}


@dataclass
class ShadowDimConfig:
    """Per-dimension configuration."""
    name: str
    lo: float
    hi: float
    prior_mean: float
    n_grid: int = 32
    has_emission: bool = True   # kappa starts with False


@dataclass
class DimPosterior:
    """Posterior distribution for a single dimension (1D grid)."""
    name: str
    grid: np.ndarray          # shape (n_grid,) — grid points
    weights: np.ndarray       # shape (n_grid,) — posterior weights (sum=1)

    @property
    def mean(self) -> float:
        return float(np.dot(self.grid, self.weights))

    @property
    def var(self) -> float:
        mu = self.mean
        return float(np.dot(self.weights, (self.grid - mu) ** 2))

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, 1e-12)))

    @property
    def entropy(self) -> float:
        w = self.weights[self.weights > 1e-15]
        return float(-np.sum(w * np.log(w)))

    def ci(self, level: float = 0.90) -> tuple[float, float]:
        """Credible interval at given level."""
        alpha = (1.0 - level) / 2.0
        cumw = np.cumsum(self.weights)
        lo_idx = np.searchsorted(cumw, alpha)
        hi_idx = np.searchsorted(cumw, 1.0 - alpha)
        lo_idx = max(0, min(lo_idx, len(self.grid) - 1))
        hi_idx = max(0, min(hi_idx, len(self.grid) - 1))
        return (float(self.grid[lo_idx]), float(self.grid[hi_idx]))

    def covers(self, true_val: float, level: float = 0.90) -> bool:
        lo, hi = self.ci(level)
        return lo <= true_val <= hi


@dataclass
class ShadowSnapshot:
    """Full 5D snapshot from one shadow observer step."""
    posteriors: Dict[str, DimPosterior]
    event_loglik: float = 0.0        # log P(y_t | q_t)
    events_used: Dict[str, float] = field(default_factory=dict)

    def mean(self, dim: str) -> float:
        return self.posteriors[dim].mean

    def var(self, dim: str) -> float:
        return self.posteriors[dim].var

    def entropy(self, dim: str) -> float:
        return self.posteriors[dim].entropy

    def as_dict(self) -> dict:
        return {d: round(p.mean, 6) for d, p in self.posteriors.items()}


@dataclass
class ShadowDiagnostics:
    """Aggregate diagnostics comparing shadow vs frozen vs true."""
    n_steps: int = 0

    # Per-dimension metrics (keyed by dim name)
    rmse: Dict[str, float] = field(default_factory=dict)
    mae: Dict[str, float] = field(default_factory=dict)
    coverage_90: Dict[str, float] = field(default_factory=dict)

    # Frozen comparison
    rmse_frozen: Dict[str, float] = field(default_factory=dict)
    mae_frozen: Dict[str, float] = field(default_factory=dict)

    # Event NLL (shadow only)
    mean_event_nll: float = 0.0
    total_event_nll: float = 0.0

    # Directional responses
    directional: Dict[str, float] = field(default_factory=dict)

    def summary_line(self) -> str:
        parts = []
        for d in DIM_NAMES:
            r_s = self.rmse.get(d, 0)
            r_f = self.rmse_frozen.get(d, 0)
            c = self.coverage_90.get(d, 0)
            parts.append(f"{d}: RMSE={r_s:.4f}(vs frozen {r_f:.4f}) Cov90={c:.2f}")
        return " | ".join(parts)
