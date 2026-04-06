"""
replay.py — Replay sampling strategies for HPC memory consolidation.

Supports uniform sampling and mixed uniform+priority sampling.
Priority is based on mismatch delta (adjustment C: uniform mix to avoid
self-reinforcing loops).
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple
from cls_learner.interfaces import MemoryPayload


class ReplaySampler:
    """
    Sample memories for replay consolidation.

    Mixing: p = (1-ρ) * uniform + ρ * priority
    Priority is based on mismatch delta, clipped to avoid runaway.
    """

    def __init__(self, rho: float = 0.3, priority_clip: float = 5.0):
        self.rho = rho                    # priority mixing weight
        self.priority_clip = priority_clip

    def sample(self, memories: List[Tuple[np.ndarray, np.ndarray, MemoryPayload]],
               deltas: Optional[List[float]] = None,
               batch_size: int = 3) -> List[MemoryPayload]:
        """
        Sample batch_size memories with mixed uniform+priority.

        Args:
            memories: list of (h, e, payload) from CA3
            deltas: per-memory mismatch scores (higher = more novel)
            batch_size: number to sample
        """
        if not memories:
            return []

        n = len(memories)
        k = min(batch_size, n)

        if deltas is None or len(deltas) != n or self.rho <= 0:
            # Pure uniform
            indices = np.random.choice(n, size=k, replace=False)
        else:
            # Mixed uniform + priority
            d = np.array(deltas, dtype=np.float64)
            d = np.clip(d, 0.0, self.priority_clip)

            # softmax-like priority (avoid extreme weights)
            d_max = d.max()
            if d_max > 0:
                priority = d / d_max  # normalize to [0, 1]
            else:
                priority = np.ones(n) / n

            uniform = np.ones(n) / n
            p = (1.0 - self.rho) * uniform + self.rho * priority

            # Normalize
            p = p / p.sum()

            indices = np.random.choice(n, size=k, replace=False, p=p)

        return [memories[i][2] for i in indices]
