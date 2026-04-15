"""
tutor_proxy.py — T1: Proxy utility tutor.

Uses Q(a) = Σ λ_i · G_i(a) - Σ λ_j · C_j(a) to select actions.
Does NOT do full rollout — uses low-dimensional belief proxies.
"""
from __future__ import annotations
from typing import List, Optional

from ..interfaces import TutorAction, CandidateBall
from ..constants import TutorActionType
from ..environment.state import QueryState
from ..config import TutorConfig
from .tutor_state import TutorBelief
from .action_generators import (
    generate_pre_select_actions, generate_post_confirm_actions,
    apply_hint_to_state,
)
from .utility import select_best_action


class ProxyTutor:
    """T1: Proxy utility-based tutor.

    Pre-select: choose among WAIT / WARNING / COURAGE by Q(a).
    Post-confirm: choose among WAIT / HINT_1 / HINT_2 by Q(a).
    """

    def __init__(
        self,
        cfg: TutorConfig,
        belief: Optional[TutorBelief] = None,
        risk_belief=None,
    ):
        self.cfg = cfg
        self.belief = belief
        self.risk_belief = risk_belief  # learner's risk belief (for estimating behavior)

    def set_belief(self, belief: TutorBelief):
        self.belief = belief

    def set_risk_belief(self, risk_belief):
        self.risk_belief = risk_belief

    def on_select(
        self,
        state: QueryState,
        selected: List[CandidateBall],
    ) -> TutorAction:
        """Pre-placement decision via Q(a).

        WARNING always trumps: if danger present, must warn (truthful).
        Otherwise: evaluate WAIT vs COURAGE.
        """
        # Safety first: always warn on danger
        has_danger = any(b.is_danger for b in selected)
        if has_danger:
            if self.belief:
                self.belief.n_warnings_issued += 1
            return TutorAction(
                action_type=TutorActionType.WARNING,
                message="Your selection contains danger.")

        # No danger: evaluate WAIT vs COURAGE
        candidates = generate_pre_select_actions(state, selected, self.cfg)
        # Filter out WARNING (handled above) — keep WAIT and COURAGE
        candidates = [a for a in candidates
                      if a.action_type != TutorActionType.WARNING]

        if len(candidates) <= 1:
            return candidates[0] if candidates else TutorAction(
                action_type=TutorActionType.WAIT)

        best, util = select_best_action(
            candidates, state,
            self.belief or TutorBelief(),
            self.risk_belief,
            self.cfg,
            context='pre_select',
        )

        if best.action_type == TutorActionType.COURAGE and self.belief:
            self.belief.n_courage_issued += 1

        return best

    def on_confirm_fail(
        self,
        state: QueryState,
        feedback: dict,
    ) -> TutorAction:
        """Post-confirm-fail: evaluate WAIT vs HINT_k via Q(a)."""
        candidates = generate_post_confirm_actions(state, feedback, self.cfg)

        if len(candidates) <= 1:
            return candidates[0] if candidates else TutorAction(
                action_type=TutorActionType.WAIT)

        best, util = select_best_action(
            candidates, state,
            self.belief or TutorBelief(),
            self.risk_belief,
            self.cfg,
            context='post_confirm',
        )

        if best.action_type == TutorActionType.HINT and self.belief:
            self.belief.n_hints_issued += 1

        return best

    def on_courage_check(
        self,
        state: QueryState,
    ) -> TutorAction:
        """Courage check: same logic as on_select but for stuck state."""
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


# Import for type annotation only
from .tutor_state import TutorBelief
