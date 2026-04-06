"""
Feature Belief Map — agent's noisy beliefs about cell features.

The agent does NOT see true features directly. It maintains a Gaussian
belief (mean, variance) for each cell's feature vector, updated from
noisy observations.

Prior: mean=[0.5, 0.5, 0.5, 0.5], var=[0.25, 0.25, 0.25, 0.25]
  → uninformative starting belief

Observation noise:
  - self cell:   σ²_obs = 0.01 (near-exact)
  - 1-hop nbr:  σ²_obs = 0.08 (blurry but informative)

Phase 10: adds CellMemoryMeta provenance tracking and
intervention-conditioned belief update methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np

from ..envs.lattice_v2 import FEATURE_DIM


# ── Cell Provenance ─────────────────────────────────────────────────

@dataclass
class CellMemoryMeta:
    """Per-cell provenance tracking: HOW the belief was formed.

    This is NOT another belief — it's metadata about observation history
    and intervention context. Distinguishes:
      - patch-seen but not traversed
      - traversed
      - tutor-indicated (warned, unlocked, item-affected)
    """
    ever_seen: bool = False
    seen_count: int = 0
    ever_traversed: bool = False
    traversed_count: int = 0
    last_seen_t: int = -1
    last_traversed_t: int = -1
    best_view_quality: float = 0.0      # 0=unseen, ~0.5=neighbor, 1.0=self
    reachable_since_t: int = -1         # when UNLOCK made this reachable (-1=always)
    intervention_tags: set = field(default_factory=set)
    # Tags: {"warned", "unlocked", "item_affected"}


def _empty_memory_grid(H: int, W: int) -> np.ndarray:
    """Create (H, W) object array of CellMemoryMeta."""
    grid = np.empty((H, W), dtype=object)
    for r in range(H):
        for c in range(W):
            grid[r, c] = CellMemoryMeta()
    return grid


# ── Feature Belief Map ──────────────────────────────────────────────

class FeatureBeliefMap:
    """Per-cell Gaussian belief over d-dimensional feature vectors.

    Maintains both the posterior (mean, var) and provenance metadata
    (CellMemoryMeta) for each cell.
    """

    def __init__(self, H: int, W: int, d: int = FEATURE_DIM,
                 prior_mean: float = 0.5, prior_var: float = 0.25):
        self.H = H
        self.W = W
        self.d = d
        self.mean = np.full((H, W, d), prior_mean, dtype=np.float64)
        self.var = np.full((H, W, d), prior_var, dtype=np.float64)
        self.memory = _empty_memory_grid(H, W)

    # ── Backward-compatible property aliases ──

    @property
    def observed(self) -> np.ndarray:
        """(H, W) bool: ever observed? Alias into CellMemoryMeta."""
        out = np.zeros((self.H, self.W), dtype=bool)
        for r in range(self.H):
            for c in range(self.W):
                out[r, c] = self.memory[r, c].ever_seen
        return out

    @property
    def visit_count(self) -> np.ndarray:
        """(H, W) int: observation count. Alias into CellMemoryMeta."""
        out = np.zeros((self.H, self.W), dtype=int)
        for r in range(self.H):
            for c in range(self.W):
                out[r, c] = self.memory[r, c].seen_count
        return out

    @property
    def last_observed_t(self) -> np.ndarray:
        """(H, W) int: timestep of last observation. Alias into CellMemoryMeta."""
        out = np.full((self.H, self.W), -1, dtype=int)
        for r in range(self.H):
            for c in range(self.W):
                out[r, c] = self.memory[r, c].last_seen_t
        return out

    def update(self, row: int, col: int, obs_mean: np.ndarray, obs_var: float,
               t: int | None = None, view_quality: float = 1.0):
        """
        Kalman update for cell (row, col):
          posterior_mean = (prior_var * obs_mean + obs_var * prior_mean) / (prior_var + obs_var)
          posterior_var  = 1 / (1/prior_var + 1/obs_var)

        Updates both posterior and provenance metadata.
        """
        pv = self.var[row, col]
        pm = self.mean[row, col]
        ov = obs_var
        # Kalman gain for each feature dimension
        K = pv / (pv + ov)
        self.mean[row, col] = pm + K * (obs_mean - pm)
        self.var[row, col] = pv * (1 - K)

        # Update provenance
        mem = self.memory[row, col]
        mem.ever_seen = True
        mem.seen_count += 1
        if t is not None:
            mem.last_seen_t = t
        mem.best_view_quality = max(mem.best_view_quality, view_quality)

    def mark_traversed(self, row: int, col: int, t: int):
        """Record that agent has traversed this cell (separate from observation)."""
        mem = self.memory[row, col]
        mem.ever_traversed = True
        mem.traversed_count += 1
        mem.last_traversed_t = t

    # ── Intervention-conditioned updates ────────────────────────────

    def apply_unlock_update(self, row: int, col: int, beta_unlock: float = 0.5,
                            t: int = 0):
        """UNLOCK: reduce posterior uncertainty without changing mean.

        Interpretation: tutor opened this path → it's worth considering.
        NOT: tutor says this is safe. Just: epistemic uncertainty decreases.

        Only applied to newly unlocked cells themselves (not neighbors).
        """
        self.var[row, col] *= (1 - beta_unlock)
        mem = self.memory[row, col]
        mem.reachable_since_t = t
        mem.intervention_tags.add("unlocked")

    def apply_warn_update(self, row: int, col: int,
                          warn_direction: np.ndarray | None = None,
                          warn_strength: float = 0.15,
                          warn_confidence: float = 2.0):
        """WARN: bias belief toward higher risk via configurable projection.

        Uses a warn_direction vector (aligned with risk predictor) rather
        than hardcoding to specific feature dimensions. If no direction
        is provided, uses a default that shifts all dims equally.

        warn_confidence: multiplier on current variance to set pseudo-obs
          variance. Higher = weaker evidence. 2.0 = moderate.
        """
        if warn_direction is None:
            warn_direction = np.ones(self.d) / np.sqrt(self.d)

        # Pseudo-observation: shift mean in warn_direction
        pseudo_obs = self.mean[row, col] + warn_strength * warn_direction
        pseudo_var = self.var[row, col] * warn_confidence

        # Kalman update with pseudo-observation
        pv = self.var[row, col]
        K = pv / (pv + pseudo_var)
        self.mean[row, col] = self.mean[row, col] + K * (pseudo_obs - self.mean[row, col])
        self.var[row, col] = pv * (1 - K)

        self.memory[row, col].intervention_tags.add("warned")

    # ── Standard interface ──────────────────────────────────────────

    def get_belief(self, row: int, col: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, var) for cell."""
        return self.mean[row, col].copy(), self.var[row, col].copy()

    def get_mean(self, row: int, col: int) -> np.ndarray:
        return self.mean[row, col].copy()

    def copy(self) -> "FeatureBeliefMap":
        return deepcopy(self)

    def reset(self, prior_mean: float = 0.5, prior_var: float = 0.25):
        self.mean[:] = prior_mean
        self.var[:] = prior_var
        self.memory = _empty_memory_grid(self.H, self.W)

