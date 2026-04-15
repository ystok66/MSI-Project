"""
dummy_tutor.py — Phase 1 baseline tutor implementations.

Three variants:
  1. NoTutor: always WAIT, death on danger
  2. NoTutorImmortalWarnlike: warns on danger (no death)
  3. NoTutorImmortalNoTimeout: warns + unlimited confirms
"""
from __future__ import annotations
from typing import List

from ..interfaces import TutorAction, CandidateBall
from ..constants import TutorActionType
from ..environment.state import QueryState


class NoTutor:
    """Baseline 1: No tutor intervention.

    Always returns WAIT. If learner selects danger, they die.
    """

    def on_select(self, state: QueryState,
                  selected: List[CandidateBall]) -> TutorAction:
        return TutorAction(action_type=TutorActionType.WAIT)

    def on_confirm_fail(self, state: QueryState,
                        feedback: dict) -> TutorAction:
        return TutorAction(action_type=TutorActionType.WAIT)

    def on_courage_check(self, state: QueryState) -> TutorAction:
        return TutorAction(action_type=TutorActionType.WAIT)


class NoTutorImmortalWarnlike:
    """Baseline 2: Immortal with warning-like feedback.

    Warns on danger (learner doesn't die but learns from warning).
    Has timeout (n_confirm_max enforced).
    """

    def on_select(self, state: QueryState,
                  selected: List[CandidateBall]) -> TutorAction:
        has_danger = any(b.is_danger for b in selected)
        if has_danger:
            return TutorAction(
                action_type=TutorActionType.WARNING,
                message="Your selection contains danger.")
        return TutorAction(action_type=TutorActionType.WAIT)

    def on_confirm_fail(self, state: QueryState,
                        feedback: dict) -> TutorAction:
        return TutorAction(action_type=TutorActionType.WAIT)

    def on_courage_check(self, state: QueryState) -> TutorAction:
        return TutorAction(action_type=TutorActionType.WAIT)


class NoTutorImmortalNoTimeout:
    """Baseline 3: Immortal + no timeout (upper bound).

    Warns on danger AND has unlimited confirms.
    Used to measure the ceiling performance.
    """

    def on_select(self, state: QueryState,
                  selected: List[CandidateBall]) -> TutorAction:
        has_danger = any(b.is_danger for b in selected)
        if has_danger:
            return TutorAction(
                action_type=TutorActionType.WARNING,
                message="Your selection contains danger.")
        return TutorAction(action_type=TutorActionType.WAIT)

    def on_confirm_fail(self, state: QueryState,
                        feedback: dict) -> TutorAction:
        return TutorAction(action_type=TutorActionType.WAIT)

    def on_courage_check(self, state: QueryState) -> TutorAction:
        return TutorAction(action_type=TutorActionType.WAIT)


class OracleWarningTutor:
    """Phase 1 default: warns on danger, normal timeout.

    This is the standard Phase 1 tutor: always warns about danger
    (truthful, no strategic silence). Has normal timeout.
    """

    def on_select(self, state: QueryState,
                  selected: List[CandidateBall]) -> TutorAction:
        has_danger = any(b.is_danger for b in selected)
        if has_danger:
            return TutorAction(
                action_type=TutorActionType.WARNING,
                message="Your selection contains danger.")
        return TutorAction(action_type=TutorActionType.WAIT)

    def on_confirm_fail(self, state: QueryState,
                        feedback: dict) -> TutorAction:
        return TutorAction(action_type=TutorActionType.WAIT)

    def on_courage_check(self, state: QueryState) -> TutorAction:
        # Phase 1: simple courage — check if any safe needed ball exists
        needed = state.needed_colors()
        for ball in state.candidate_pool:
            if not ball.is_danger and ball.color in needed:
                return TutorAction(
                    action_type=TutorActionType.COURAGE,
                    message="A safe needed ball exists.")
        return TutorAction(action_type=TutorActionType.WAIT)
