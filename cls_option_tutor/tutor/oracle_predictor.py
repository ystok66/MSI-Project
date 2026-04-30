"""
oracle_predictor.py — Privileged ceiling predictor.

Holds a live LearnerAgent reference.
May access learner._scorer, learner.policy.danger_head,
learner.policy.attention, and deepcopy learner for rollout.

NOT a clean inverse tutor.

This is the upper-bound reference for the oracle-inverse gap:
    oracle_inverse_gap = J(oracle_forward) - J(inverse_shadow)

Usage:
    tutor = SparseTutorAgent(cfg, predictor=OracleForwardPredictor(tutor_ref))
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .predictor import LearnerPredictor, FullActionDist


class OracleForwardPredictor:
    """Privileged ceiling predictor.

    Holds live LearnerAgent reference.
    May access learner._scorer, learner.policy.danger_head,
    learner.policy.attention, and deepcopy learner for rollout.
    Not a clean inverse tutor.

    Implementation: delegates to SparseTutorAgent._compute_learner_probs_direct()
    and _rollout_estimate_direct() to avoid recursion through the predictor gate.
    """

    def __init__(self, tutor_ref):
        """
        Args:
            tutor_ref: SparseTutorAgent instance.
                       Must have _compute_learner_probs_direct() and
                       _rollout_estimate_direct() methods.
        """
        self._tutor = tutor_ref

    def observe(self, obs_step) -> None:
        """No-op: oracle has direct access to live learner state."""
        pass

    def pick_dist(self, qs, active: list, spec: dict) -> np.ndarray:
        """Delegate to tutor's direct learner-consistent computation.

        This is the SAME code path as the legacy sparse tutor, just
        routed through the predictor interface.
        """
        learner = getattr(self._tutor, '_learner_ref', None)
        if learner is None:
            K = len(active)
            return np.ones(K) / max(K, 1)
        return self._tutor._compute_learner_probs_direct(qs, active, spec, learner)

    def full_action_dist(self, qs, active: list, spec: dict) -> FullActionDist:
        """Full action distribution including refresh probability.

        Refresh probability estimated from learner policy internal state.
        """
        pick_probs = self.pick_dist(qs, active, spec)
        # Oracle can estimate refresh from learner internals
        # For v1, use a simple proxy: p_refresh = 0 if no refreshes left
        p_refresh = 0.0
        if hasattr(qs, 'refreshes_used') and hasattr(qs, 'max_refreshes'):
            if qs.refreshes_used < qs.max_refreshes:
                # Small constant for oracle — in practice the learner
                # rarely refreshes in the current scoring path
                p_refresh = 0.05
            else:
                p_refresh = 0.0

        active_indices = [o.index for o in active]
        return FullActionDist(
            active_indices=active_indices,
            pick_probs=pick_probs,
            p_refresh=p_refresh,
        )

    def rollout(
        self,
        qs,
        active: list,
        spec: dict,
        n: int,
    ) -> Tuple[float, float, float]:
        """Delegate to tutor's direct deepcopy-based rollout."""
        learner = getattr(self._tutor, '_learner_ref', None)
        if learner is None:
            return (0.0, 1.0, 0.0)
        return self._tutor._rollout_estimate_direct(qs, active, spec, learner, n)

    def clone(self) -> 'OracleForwardPredictor':
        """Clone — shares the same tutor/learner reference (stateless)."""
        return OracleForwardPredictor(self._tutor)
