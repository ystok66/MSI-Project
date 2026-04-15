"""
memory.py — Query-internal short-term memory.

Tracks warned sets, death observations, and retry history
within a single query. Does NOT persist across queries.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import numpy as np


@dataclass
class QueryMemory:
    """Short-term memory for one query.

    Tracks observations that the learner accumulates during a single
    query's SELECT / WARN / PLACE / CONFIRM cycle.
    """
    # Warned sets: list of (ball_indices, ball_observed_vecs) that were warned
    warned_sets: List[Tuple[List[int], np.ndarray]] = field(default_factory=list)

    # Death observations: observed vecs of balls that caused death
    death_observations: List[np.ndarray] = field(default_factory=list)

    # Safe observations: observed vecs of balls that were placed safely
    safe_observations: List[np.ndarray] = field(default_factory=list)

    # Colors that have been fully filled
    completed_colors: Set[str] = field(default_factory=set)

    # Retry count
    n_retries: int = 0

    def record_warning(self, ball_indices: List[int], observed_vecs: np.ndarray):
        """Record a warning event."""
        self.warned_sets.append((list(ball_indices), observed_vecs.copy()))

    def record_death(self, observed_vec: np.ndarray):
        """Record a death event (for immortal baselines)."""
        self.death_observations.append(observed_vec.copy())

    def record_safe_placement(self, observed_vecs: List[np.ndarray]):
        """Record safe ball placements."""
        for v in observed_vecs:
            self.safe_observations.append(v.copy())

    def record_retry(self):
        """Record a retry."""
        self.n_retries += 1

    def reset(self):
        """Clear all memory for a new query."""
        self.warned_sets.clear()
        self.death_observations.clear()
        self.safe_observations.clear()
        self.completed_colors.clear()
        self.n_retries = 0
