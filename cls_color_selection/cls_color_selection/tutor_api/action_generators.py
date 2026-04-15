"""
action_generators.py — Enumerate candidate tutor actions.

Generates WAIT, WARNING, HINT_1, HINT_2, COURAGE candidates
depending on the current query state and phase.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from ..interfaces import TutorAction, CandidateBall
from ..constants import TutorActionType
from ..environment.state import QueryState
from ..config import TutorConfig


class HintSpec:
    """Specification for a hint action (plain struct)."""
    def __init__(self, n_balls: int, target_colors: List[str], target_positions: List[int]):
        self.n_balls = n_balls
        self.target_colors = target_colors
        self.target_positions = target_positions


def generate_pre_select_actions(
    state: QueryState,
    selected: List[CandidateBall],
    cfg: TutorConfig,
) -> List[TutorAction]:
    """Generate candidate actions for the pre-placement hook.

    Only WARNING, COURAGE, and WAIT are valid here.
    """
    actions = [TutorAction(action_type=TutorActionType.WAIT)]

    # WARNING: if any selected ball is danger
    has_danger = any(b.is_danger for b in selected)
    if has_danger:
        actions.append(TutorAction(
            action_type=TutorActionType.WARNING,
            message="Your selection contains danger."))

    # COURAGE: if learner is stuck and safe-needed ball exists
    if state.consecutive_retries >= cfg.n_retry_courage:
        needed = state.needed_colors()
        pool_has_safe_needed = any(
            not b.is_danger and b.color in needed
            for b in state.candidate_pool
        )
        if pool_has_safe_needed:
            actions.append(TutorAction(
                action_type=TutorActionType.COURAGE,
                message="A safe needed ball exists."))

    return actions


def generate_post_confirm_actions(
    state: QueryState,
    feedback: dict,
    cfg: TutorConfig,
) -> List[TutorAction]:
    """Generate candidate actions for the post-confirm-fail hook.

    WAIT and HINT_k are valid here. Hints place safe balls directly.
    """
    actions = [TutorAction(action_type=TutorActionType.WAIT)]

    if not cfg.hint_after_confirm_fail:
        return actions

    # Generate HINT candidates: place 1 or 2 correct balls
    gt = state.ground_truth
    completion = state.completion
    L = min(len(gt), len(completion))

    # Find unfilled or wrong positions
    fixable_positions = []
    for pos in range(L):
        if completion[pos] is None or completion[pos] != gt[pos]:
            fixable_positions.append(pos)

    if not fixable_positions:
        return actions  # nothing to hint

    # HINT_1: fix the first wrong/empty position
    max_k = min(cfg.max_hint_balls, len(fixable_positions))

    for k in range(1, max_k + 1):
        hint_positions = [(fixable_positions[i], gt[fixable_positions[i]])
                          for i in range(k)]
        actions.append(TutorAction(
            action_type=TutorActionType.HINT,
            hint_positions=hint_positions,
            message=f"Hint: placing {k} correct ball(s).",
        ))

    return actions


def apply_hint_to_state(
    state: QueryState,
    hint_action: TutorAction,
) -> QueryState:
    """Apply a HINT action: place balls directly into completion.

    Hint balls are "from thin air" — they go straight into the completion
    without going through the candidate pool.

    Args:
        state: current query state (modified in place)
        hint_action: TutorAction with hint_positions

    Returns:
        Updated state
    """
    if hint_action.hint_positions is None:
        return state

    for pos, color in hint_action.hint_positions:
        if 0 <= pos < len(state.completion):
            state.completion[pos] = color

    state.step_log.append({
        'event': 'hint',
        'positions': hint_action.hint_positions,
        'fill_ratio': state.fill_ratio,
    })

    return state



