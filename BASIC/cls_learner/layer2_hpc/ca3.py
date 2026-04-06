"""
ca3.py — CA3 auto-associative memory with Hopfield matrix + list storage.

Write:    store (h, e, payload) and update M += η·outer(h,h)
Retrieve: iterative completion via M, then find nearest neighbors.
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple
from cls_learner.interfaces import MemoryPayload
from cls_learner.layer2_hpc.dg import _kwta


class CA3Memory:
    """
    CA3 auto-associative memory.

    Stores sparse DG codes, event vectors, and payloads.
    Supports Hopfield pattern completion and similarity retrieval.
    """

    def __init__(self, m: int = 512, k: int = 30,
                 top_r: int = 5, eta: float = 1.0,
                 completion_steps: int = 3, temp: float = 1.0):
        self.m = m
        self.k = k
        self.top_r = top_r
        self.eta = eta
        self.completion_steps = completion_steps
        self.temp = temp

        self.M = np.zeros((m, m), dtype=np.float32)
        self.memories: List[Tuple[np.ndarray, np.ndarray, MemoryPayload]] = []

    def clear(self):
        """Reset all memories."""
        self.M[:] = 0.0
        self.memories.clear()

    def write(self, h: np.ndarray, e: np.ndarray,
              payload: MemoryPayload) -> int:
        """Store one memory. Returns memory index for reconsolidation."""
        idx = len(self.memories)
        self.memories.append((h.copy(), e.copy(), payload))

        outer = np.outer(h, h)
        np.fill_diagonal(outer, 0.0)
        self.M += self.eta * outer
        return idx

    def update_payload(self, idx: int, new_payload: MemoryPayload):
        """Reconsolidation: update payload without changing DG code."""
        if 0 <= idx < len(self.memories):
            h, e, _ = self.memories[idx]
            self.memories[idx] = (h, e, new_payload)

    def retrieve(self, h_q: np.ndarray,
                 top_r: Optional[int] = None
                 ) -> List[Tuple[float, int, MemoryPayload]]:
        """Retrieve top-R most similar memories by sparse dot-product."""
        if not self.memories:
            return []

        top_r = top_r or self.top_r
        sims = []
        for i, (h_i, e_i, payload) in enumerate(self.memories):
            sim = float(np.dot(h_q, h_i))
            sims.append((sim, i, payload))

        sims.sort(key=lambda x: x[0], reverse=True)
        return sims[:top_r]

    def complete(self, h_q: np.ndarray) -> np.ndarray:
        """Pattern completion via Hopfield iteration."""
        h = h_q.copy()
        for _ in range(self.completion_steps):
            h = _kwta(self.M @ h, self.k)
        return h
