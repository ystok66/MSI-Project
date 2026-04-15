"""
seeds.py — Seed management for reproducible experiments.
"""
from __future__ import annotations
from typing import List


def generate_seeds(base_seed: int, n_seeds: int) -> List[int]:
    """Generate deterministic seed list from a base seed."""
    import numpy as np
    rng = np.random.default_rng(base_seed)
    return [int(rng.integers(0, 2**31)) for _ in range(n_seeds)]
