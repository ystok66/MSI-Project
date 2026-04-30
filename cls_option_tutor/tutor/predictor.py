"""
predictor.py — LearnerPredictor protocol for the inverse tutor scaffold.

Defines the narrow interface through which SparseTutorAgent obtains
learner action predictions, separating:
  - OracleForwardPredictor  (privileged ceiling)
  - InverseShadowPredictor  (clean inverse mainline)

The protocol has two prediction surfaces:
  pick_dist():         backward-compatible (K_active,) pick probs for Q formulas
  full_action_dist():  includes p_refresh for profile inference & richer diagnostics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


# ── Full action distribution (pick + refresh) ─────────────────────────────────

@dataclass
class FullActionDist:
    """Joint action distribution including refresh probability.

    Attributes:
        active_indices: menu indices of pickable options
        pick_probs:     pick distribution over active options, shape (K_active,)
                        sums to (1 - p_refresh)
        p_refresh:      probability that learner refreshes instead of picking
        p_timeout:      optional estimated timeout probability (0 in v1)
    """
    active_indices: List[int]
    pick_probs: np.ndarray          # (K_active,)
    p_refresh: float = 0.0
    p_timeout: float = 0.0


# ── Predictor protocol ────────────────────────────────────────────────────────

@runtime_checkable
class LearnerPredictor(Protocol):
    """Protocol for learner prediction sources.

    All predictive queries in SparseTutorAgent flow through this interface.
    Implementations must NOT hold references to LearnerAgent internals
    unless explicitly documented as privileged (OracleForwardPredictor).
    """

    def observe(self, obs_step) -> None:
        """Process one public observation step.

        Args:
            obs_step: ObservedStep from observation_adapter.
                      Contains only public information.
        """
        ...

    def pick_dist(
        self,
        qs,
        active: list,
        spec: dict,
    ) -> np.ndarray:
        """Backward-compatible pick distribution for existing Q formulas.

        Returns probabilities over active pickable options.
        Shape: (K_active,).  Sums to ~1.0.
        Used by SparseTutorAgent._compute_learner_probs().
        """
        ...

    def full_action_dist(
        self,
        qs,
        active: list,
        spec: dict,
    ) -> FullActionDist:
        """Full action distribution including refresh probability.

        Used by inverse profile update and richer diagnostics.
        pick_probs in the returned FullActionDist sum to (1 - p_refresh).
        """
        ...

    def rollout(
        self,
        qs,
        active: list,
        spec: dict,
        n: int,
    ) -> Tuple[float, float, float]:
        """Simulate n rollouts from current state under spec.

        Returns: (p_death, p_timeout, p_success) empirical means.
        """
        ...

    def clone(self) -> 'LearnerPredictor':
        """Deep-copy this predictor (for rollout branching)."""
        ...
