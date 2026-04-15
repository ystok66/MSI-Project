"""
tutor_rule.py — T0: Rule-based tutor.

Threshold-based decisions:
  - WARNING: always warn if danger present (truthful)
  - COURAGE: if stuck AND safe-needed exists
  - HINT: if P(timeout) > threshold after confirm fail
"""
from __future__ import annotations
from typing import List, Optional

from ..interfaces import TutorAction, CandidateBall
from ..constants import TutorActionType
from ..environment.state import QueryState
from ..config import TutorConfig
from .tutor_state import TutorBelief
from .belief_update import compute_timeout_risk


class RuleTutor:
    """T0: threshold-based tutor with belief-informed decisions.

    Always warns on danger (truthful).
    Hints when timeout risk is high enough.
    Courage when stuck.
    """

    def __init__(self, cfg: TutorConfig, belief: Optional[TutorBelief] = None):
        self.cfg = cfg
        self.belief = belief

    def set_belief(self, belief: TutorBelief):
        self.belief = belief

    def on_select(
        self,
        state: QueryState,
        selected: List[CandidateBall],
    ) -> TutorAction:
        """WARNING if danger present (always truthful)."""
        has_danger = any(b.is_danger for b in selected)
        if has_danger:
            if self.belief:
                self.belief.n_warnings_issued += 1
            return TutorAction(
                action_type=TutorActionType.WARNING,
                message="Your selection contains danger.")
        return TutorAction(action_type=TutorActionType.WAIT)

    def on_confirm_fail(
        self,
        state: QueryState,
        feedback: dict,
    ) -> TutorAction:
        """HINT if timeout risk is high, else WAIT."""
        if not self.cfg.hint_after_confirm_fail:
            return TutorAction(action_type=TutorActionType.WAIT)

        # Compute timeout risk from belief
        if self.belief is not None:
            p_timeout = compute_timeout_risk(self.belief, state)
        else:
            # No belief → use heuristic
            confirms_left = state.n_confirm_max - state.confirm_count
            p_timeout = 1.0 if confirms_left <= 1 else 0.3

        # Rule: hint if timeout risk > 0.5 and confirms nearly exhausted
        confirms_left = state.n_confirm_max - state.confirm_count
        if p_timeout > 0.5 and confirms_left <= 2:
            # Generate hint: place 1 correct ball
            gt = state.ground_truth
            for pos in range(len(gt)):
                if pos < len(state.completion) and (
                    state.completion[pos] is None
                    or state.completion[pos] != gt[pos]
                ):
                    if self.belief:
                        self.belief.n_hints_issued += 1
                    return TutorAction(
                        action_type=TutorActionType.HINT,
                        hint_positions=[(pos, gt[pos])],
                        message="Hint: placing 1 correct ball.",
                    )

        return TutorAction(action_type=TutorActionType.WAIT)

    def on_courage_check(
        self,
        state: QueryState,
    ) -> TutorAction:
        """COURAGE if stuck AND safe-needed ball exists in pool."""
        if state.consecutive_retries < self.cfg.n_retry_courage:
            return TutorAction(action_type=TutorActionType.WAIT)

        needed = state.needed_colors()
        for ball in state.candidate_pool:
            if not ball.is_danger and ball.color in needed:
                if self.belief:
                    self.belief.n_courage_issued += 1
                return TutorAction(
                    action_type=TutorActionType.COURAGE,
                    message="A safe needed ball exists.")

        return TutorAction(action_type=TutorActionType.WAIT)
