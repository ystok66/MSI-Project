"""
belief_stub.py — Phase 2 tutor belief model interface (stub).

This module defines the interface for tutor's belief about the learner.
Phase 1: not implemented, just interface.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Protocol


class TutorBeliefModel(Protocol):
    """Tutor's belief about the learner (Phase 2 interface)."""

    def update_from_observation(
        self,
        query_words: List[str],
        learner_action: dict,
        outcome: dict,
    ) -> None:
        """Update belief after observing learner behavior."""
        ...

    def estimate_learner_knowledge(self) -> Dict[str, float]:
        """Estimate learner's grammar knowledge state."""
        ...

    def estimate_learner_risk_awareness(self) -> float:
        """Estimate how well learner understands risk."""
        ...

    def predict_success_probability(
        self,
        query_words: List[str],
        remaining_confirms: int,
    ) -> float:
        """Predict P(success before timeout) for the learner."""
        ...
