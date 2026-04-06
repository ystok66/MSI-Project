"""
dg.py — Dentate Gyrus: pattern separation via random projection + kWTA.

Input:  event vector e ∈ R^{d_in}  (from EventEncoder)
Output: sparse code h ∈ R^m        (only k entries non-zero)
"""
from __future__ import annotations
import numpy as np


def _kwta(u: np.ndarray, k: int) -> np.ndarray:
    """k-Winners-Take-All: keep top-k activations, zero the rest."""
    if k >= len(u):
        return u.copy()
    threshold = np.partition(u, -k)[-k]
    h = np.where(u >= threshold, u, 0.0)
    active = np.nonzero(h)[0]
    if len(active) > k:
        sorted_idx = active[np.argsort(h[active])]
        to_zero = sorted_idx[:len(active) - k]
        h[to_zero] = 0.0
    return h


class DGEncoder:
    """
    Dentate Gyrus: pattern separation via random projection + kWTA.

    The random projection W is fixed per seed (not learned).
    """

    def __init__(self, d_in: int = 128, m: int = 512, k: int = 30,
                 noise_std: float = 0.01, seed: int = 42):
        self.d_in = d_in
        self.m = m
        self.k = k
        self.noise_std = noise_std
        rng = np.random.RandomState(seed)
        self.W = rng.randn(m, d_in) / np.sqrt(d_in)

    def encode(self, e: np.ndarray) -> np.ndarray:
        """Project + noise + kWTA → sparse code."""
        u = self.W @ e
        if self.noise_std > 0:
            u += np.random.randn(self.m) * self.noise_std
        return _kwta(u, self.k)
