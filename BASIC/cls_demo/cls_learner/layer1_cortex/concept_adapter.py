"""
concept_adapter.py — Thin adapter for NeuroConcept API.

Provides a uniform interface for cortex operations without modifying
the original NeuroConcept class. Used by CortexMemory.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from ns_learner.ns_concept import (
    NeuroConcept, NIGParams, ROLES, N_COLORS, COLORS,
    COLOR_VECS, color_to_vec, vec_to_color, REPEAT_RANGE,
)


class ConceptAdapter:
    """
    Adapter wrapping NeuroConcept for CLS integration.

    Encapsulates scoring, updating, and inspection operations
    with a clean interface that doesn't leak NeuroConcept internals.
    """

    def __init__(self, concept: NeuroConcept):
        self._c = concept

    @property
    def name(self) -> str:
        return self._c.name

    def log_role_prob(self, role: str, alpha: Dict[str, float]) -> float:
        """Log probability of a role under Dirichlet posterior."""
        return self._c.log_role_prob(role, alpha)

    def log_emit_prob(self, vec: np.ndarray, nig: NIGParams,
                      eps_obj: float = 1e-3, tau_inc: float = 1.0,
                      delta: Optional[Dict[str, float]] = None,
                      gauss: bool = False) -> float:
        """Log emission probability."""
        return self._c.log_emit_prob(vec, nig, eps_obj, tau_inc, delta, gauss)

    def log_repeat_prob(self, k: int, gamma: Dict[int, float]) -> float:
        """Log probability of repeat count k."""
        return self._c.log_repeat_prob(k, gamma)

    def map_role(self, alpha: Dict[str, float]) -> str:
        """MAP role estimate."""
        return self._c.map_role(alpha)

    def map_color(self, nig: NIGParams, eps_obj: float = 1e-3,
                  tau_inc: float = 1.0,
                  delta: Optional[Dict[str, float]] = None,
                  gauss: bool = False) -> str:
        """MAP color estimate."""
        return self._c.map_color(nig, eps_obj, tau_inc, delta, gauss)

    def soft_update(self, weight: float, role: str,
                    vec: Optional[np.ndarray] = None,
                    k: Optional[int] = None):
        """Accumulate weighted sufficient statistics."""
        self._c.soft_update(weight, role, vec, k)

    def role_probs(self, alpha: Dict[str, float]) -> Dict[str, float]:
        """Posterior role probabilities."""
        return self._c.role_probs(alpha)

    def reset_counts(self):
        """Zero out counts for new E-step."""
        self._c.reset_counts()

    def decay_counts(self, rate: float = 0.5):
        """Decay role/repeat counts (keep emit stats)."""
        for r in ROLES:
            self._c.role_counts[r] *= rate
        for k in REPEAT_RANGE:
            self._c.repeat_counts[k] *= rate
