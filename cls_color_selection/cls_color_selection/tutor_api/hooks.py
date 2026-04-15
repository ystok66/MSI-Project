"""
hooks.py — Tutor intervention hooks (Protocol / interface).

Phase 1: only WARNING and WAIT are implemented.
Phase 2 will add HINT planning.
"""
from __future__ import annotations
from typing import List, Protocol, runtime_checkable

from ..interfaces import TutorAction, CandidateBall
from ..constants import TutorActionType
from ..environment.state import QueryState


@runtime_checkable
class TutorHooks(Protocol):
    """Protocol for tutor intervention at defined hook points."""

    def on_select(
        self,
        state: QueryState,
        selected: List[CandidateBall],
    ) -> TutorAction:
        """Called after learner selects balls, before placement.

        Returns WARNING if selection contains danger, WAIT otherwise.
        Phase 2: may also return COURAGE.
        """
        ...

    def on_confirm_fail(
        self,
        state: QueryState,
        feedback: dict,
    ) -> TutorAction:
        """Called after a failed confirm.

        Phase 1: always returns WAIT.
        Phase 2: may return HINT with placed_balls.
        """
        ...

    def on_courage_check(
        self,
        state: QueryState,
    ) -> TutorAction:
        """Called when learner is stuck (too many retries).

        Phase 1: returns COURAGE if conditions met, WAIT otherwise.
        """
        ...
