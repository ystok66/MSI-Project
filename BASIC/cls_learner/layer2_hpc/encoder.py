"""
encoder.py — EventEncoder for utterance → fixed-size feature vector.

Features:
  φ_bow(x):  hash-BOW over tokens           (d_bow dims)
  ψ_bigr(x): hash over adjacent bigrams     (d_bigr dims)

Output is L2-normalized, providing the input to DG encoding.
Block boundaries are preserved for blockwise Mahalanobis (CA1).
"""
from __future__ import annotations
import numpy as np
import hashlib
from typing import List, Tuple


def _token_hash(token: str, d: int) -> int:
    """Deterministic hash of a token string to bucket index in [0, d)."""
    h = hashlib.md5(token.encode('utf-8')).hexdigest()
    return int(h, 16) % d


def _bigram_hash(t1: str, t2: str, d: int) -> int:
    """Deterministic hash of a bigram to bucket index in [0, d)."""
    h = hashlib.md5(f"{t1}||{t2}".encode('utf-8')).hexdigest()
    return int(h, 16) % d


class EventEncoder:
    """
    Encode utterance (word sequence) into a fixed-size feature vector.

    Only the utterance cue is used for DG indexing (M1: query has no
    output/trace, so including them would cause DG code divergence).

    The encoder preserves block boundaries so CA1 can do blockwise
    Mahalanobis distance (adjustment A).
    """

    def __init__(self, d_bow: int = 64, d_bigr: int = 64):
        self.d_bow = d_bow
        self.d_bigr = d_bigr
        self.d_out = d_bow + d_bigr
        # Block boundaries: [(start, end), ...]
        self.blocks = [
            (0, d_bow),             # φ_bow
            (d_bow, d_bow + d_bigr),  # ψ_bigr
        ]

    def encode_utterance(self, words: List[str]) -> np.ndarray:
        """Encode word sequence → feature vector (L2-normalized)."""
        phi_bow = np.zeros(self.d_bow)
        phi_bigr = np.zeros(self.d_bigr)

        for w in words:
            idx = _token_hash(w, self.d_bow)
            phi_bow[idx] += 1.0

        for i in range(len(words) - 1):
            idx = _bigram_hash(words[i], words[i + 1], self.d_bigr)
            phi_bigr[idx] += 1.0

        e = np.concatenate([phi_bow, phi_bigr])
        norm = np.linalg.norm(e)
        if norm > 1e-12:
            e = e / norm
        return e

    def split_blocks(self, e: np.ndarray) -> List[np.ndarray]:
        """Split a feature vector into per-block sub-vectors."""
        return [e[s:t] for s, t in self.blocks]
