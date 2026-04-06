"""
ca1.py — CA1 Comparator with blockwise Mahalanobis mismatch detection.

Upgraded from Euclidean distance to blockwise diagonal Mahalanobis:
  - Estimates per-block diagonal variance from support data
  - Mixes residual variance with feature variance (adjustment B: a=0.7)
  - Ridge regularization (eps) prevents unstable inv_var (adjustment A)
  - Falls back to Euclidean when insufficient samples

The mismatch signal δ drives a continuous sigmoid gate:
  g(δ) = σ((θ - δ) / T)
  λ_mem = λ_min + (λ_max - λ_min) * g(δ)
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple


def _maha_diag(diff: np.ndarray, inv_var: np.ndarray) -> float:
    """Diagonal Mahalanobis: (diff)^T diag(inv_var) (diff)."""
    return float((diff * diff * inv_var).sum())


class CA1Comparator:
    """
    CA1 mismatch / novelty gate with blockwise Mahalanobis distance.

    Block structure matches EventEncoder.blocks so each feature group
    (BOW, bigram, ...) is whitened independently before combining.
    """

    def __init__(self, lam_min: float = 0.0, lam_max: float = 1.0,
                 default_th: float = 0.5, default_temp: float = 0.1,
                 eps: float = 1e-3, mix_a: float = 0.7):
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.eps = eps
        self.mix_a = mix_a    # residual vs feature var mixing weight

        # Thresholds (overwritten by calibrate)
        self.th = default_th
        self.temp = default_temp
        self.th_low = default_th * 0.6
        self.th_high = default_th * 1.4
        self._calibrated = False

        # Per-block inverse variance (None = not yet estimated → Euclidean)
        self._inv_var_blocks: Optional[List[np.ndarray]] = None
        # Block boundaries [(start, end), ...]
        self._block_ranges: Optional[List[Tuple[int, int]]] = None

    def set_block_ranges(self, blocks: List[Tuple[int, int]]):
        """Set block boundaries from EventEncoder."""
        self._block_ranges = list(blocks)

    def calibrate(self, residuals: List[np.ndarray],
                  features: List[np.ndarray],
                  block_ranges: Optional[List[Tuple[int, int]]] = None):
        """
        Auto-calibrate using blockwise Mahalanobis (adjustments A+B).

        1. Estimate per-block diag var from residuals AND features
        2. Mix: var = a * var_residual + (1-a) * var_feature
        3. inv_var = 1 / (var + eps)
        4. Compute mismatch deltas for all residuals → set thresholds
        """
        if block_ranges is not None:
            self._block_ranges = block_ranges

        if not residuals or len(residuals) < 2:
            return

        res_arr = np.array(residuals)   # (N, d)
        feat_arr = np.array(features)   # (N, d)

        # Default: whole vector as single block
        if self._block_ranges is None:
            d = res_arr.shape[1]
            self._block_ranges = [(0, d)]

        # Estimate per-block inverse variance
        self._inv_var_blocks = []
        for s, t in self._block_ranges:
            res_block = res_arr[:, s:t]
            feat_block = feat_arr[:, s:t]

            var_r = res_block.var(axis=0) if len(res_block) >= 3 else None
            var_f = feat_block.var(axis=0) if len(feat_block) >= 3 else None

            if var_r is None and var_f is None:
                # Not enough data → Euclidean for this block
                self._inv_var_blocks.append(np.ones(t - s))
                continue

            if var_r is None:
                v = var_f
            elif var_f is None:
                v = var_r
            else:
                v = self.mix_a * var_r + (1.0 - self.mix_a) * var_f

            self._inv_var_blocks.append(1.0 / (v + self.eps))

        # Compute mismatch deltas for threshold calibration
        deltas = []
        for r in residuals:
            d = self._compute_maha(r)
            deltas.append(d)

        arr = np.array(deltas)
        self.th_low = float(np.percentile(arr, 30))
        self.th_high = float(np.percentile(arr, 70))
        self.th = (self.th_low + self.th_high) / 2.0

        spread = max(self.th_high - self.th_low, 1e-6)
        self.temp = 0.1 * spread
        self._calibrated = True

    def _compute_maha(self, diff: np.ndarray) -> float:
        """Compute blockwise Mahalanobis distance."""
        if self._inv_var_blocks is None or self._block_ranges is None:
            return float(np.dot(diff, diff))

        delta = 0.0
        for (s, t), inv_var in zip(self._block_ranges, self._inv_var_blocks):
            d_block = diff[s:t]
            block_dim = t - s
            if block_dim > 0:
                # Per-dimension average whitened error
                delta += _maha_diag(d_block, inv_var) / block_dim
        return delta

    def mismatch(self, e_cue: np.ndarray,
                 retrieved_events: List[np.ndarray],
                 weights: List[float]) -> float:
        """
        Compute blockwise Mahalanobis mismatch between cue and
        weighted reconstruction.
        """
        if not retrieved_events:
            return float('inf')

        e_recon = sum(w * e for w, e in zip(weights, retrieved_events))
        diff = e_cue - e_recon
        return self._compute_maha(diff)

    def gate(self, delta: float) -> Tuple[float, str]:
        """
        Continuous sigmoid gating with hysteresis.

        g(δ) = σ((θ - δ) / T)
        λ_mem = λ_min + (λ_max - λ_min) * g(δ)

        Returns (lam_mem, mode).
        """
        if self.temp < 1e-12:
            g = 1.0 if delta < self.th else 0.0
        else:
            z = (self.th - delta) / self.temp
            z = np.clip(z, -20.0, 20.0)
            g = 1.0 / (1.0 + np.exp(-z))

        lam_mem = self.lam_min + (self.lam_max - self.lam_min) * float(g)

        if delta < self.th_low:
            mode = 'retrieve'
        elif delta > self.th_high:
            mode = 'explore'
        else:
            mode = 'mixed'

        return lam_mem, mode
