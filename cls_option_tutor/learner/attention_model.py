"""
attention_model.py — Learner attention weights over target cells.

Implements §10.2 HIGHLIGHT boost:
    w_ℓ^(H) ∝ w_ℓ · exp(ρ_H · 1[ℓ ∈ H])

Provides a uniform baseline and highlight-modified weights.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


class AttentionModel:
    """Learner's attention distribution over target output cells.

    Starts uniform. HIGHLIGHT boosts specified cells.
    Attention weights are used by the semantic scorer for
    weighted mismatch computation.
    """

    def __init__(self, L: int, rho_H: float = 2.0):
        """
        Args:
            L: number of target output cells
            rho_H: highlight attention boost strength (§10.2)
        """
        self.L = L
        self.rho_H = rho_H
        self._weights = np.ones(L) / L  # uniform prior
        self._highlighted: Tuple[int, ...] = ()

    @property
    def weights(self) -> np.ndarray:
        """Current attention weights (sum to 1)."""
        return self._weights.copy()

    def apply_highlight(self, cells: Tuple[int, ...]) -> np.ndarray:
        """Apply HIGHLIGHT boost to specified cells.

        §10.2: w_ℓ^(H) ∝ w_ℓ · exp(ρ_H · 1[ℓ ∈ H])
        Then re-normalize.

        Returns the new weights.
        """
        if not cells:
            return self._weights.copy()

        self._highlighted = cells
        w = self._weights.copy()
        for ell in cells:
            if 0 <= ell < self.L:
                w[ell] *= np.exp(self.rho_H)

        # Re-normalize
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum
        else:
            w = np.ones(self.L) / self.L

        self._weights = w
        return self._weights.copy()

    def reset(self) -> None:
        """Reset to uniform (e.g., after refresh)."""
        self._weights = np.ones(self.L) / self.L
        self._highlighted = ()

    @property
    def highlighted_cells(self) -> Tuple[int, ...]:
        return self._highlighted

    @property
    def is_highlighted(self) -> bool:
        return len(self._highlighted) > 0

    def effective_coverage(self) -> float:
        """Effective number of attended cells (exp of entropy)."""
        p = self._weights[self._weights > 0]
        entropy = -np.sum(p * np.log(p))
        return float(np.exp(entropy))
