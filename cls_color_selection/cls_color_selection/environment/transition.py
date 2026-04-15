"""
transition.py — Pure transition functions for the color-selection environment.

All functions are stateless: take a QueryState, return a new/modified QueryState.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .state import QueryState
from ..interfaces import CandidateBall, TutorAction
from ..constants import Outcome, TutorActionType


def select_balls(state: QueryState, indices: List[int]) -> List[CandidateBall]:
    """Validate and extract selected balls from the candidate pool.

    Args:
        state: current query state
        indices: indices into state.candidate_pool

    Returns:
        List of selected CandidateBall objects.

    Raises:
        ValueError: if any index is out of range.
    """
    pool = state.candidate_pool
    selected = []
    for idx in indices:
        if idx < 0 or idx >= len(pool):
            raise ValueError(f"Invalid ball index {idx}, pool size {len(pool)}")
        selected.append(pool[idx])
    return selected


def check_selection_has_danger(selected: List[CandidateBall]) -> bool:
    """Check if any ball in the selection is dangerous."""
    return any(b.is_danger for b in selected)


def filter_needed_balls(
    selected: List[CandidateBall],
    needed_colors: Set[str],
) -> Tuple[List[CandidateBall], List[CandidateBall]]:
    """Split selected balls into needed and waste.

    Args:
        selected: balls the learner chose
        needed_colors: colors still needed for completion

    Returns:
        (needed_balls, waste_balls)
    """
    needed = []
    waste = []
    # Track remaining gaps to avoid over-selecting same color
    remaining = {}
    for c in needed_colors:
        remaining[c] = remaining.get(c, 0) + 1

    for ball in selected:
        if ball.color in needed_colors and remaining.get(ball.color, 0) > 0:
            needed.append(ball)
            remaining[ball.color] -= 1
        else:
            waste.append(ball)
    return needed, waste


def auto_place(state: QueryState, selected: List[CandidateBall]) -> QueryState:
    """Deterministic placement: Place(z_t, Y*, S_t).

    For each ball's color c in selected, fills the leftmost unfilled
    position in Y* that expects color c.

    Modifies state in-place and returns it.
    """
    target = state.target_output
    completion = state.completion

    for ball in selected:
        c = ball.color
        # Find leftmost unfilled position needing this color
        for pos in range(len(target)):
            if target[pos] == c and completion[pos] is None:
                completion[pos] = c
                break
        # If no position needs this color, the ball is wasted (ignored)

    # Reset consecutive retries since we placed something
    state.consecutive_retries = 0
    return state


def confirm(state: QueryState) -> Tuple[bool, dict]:
    """Check if the current completion matches the ground truth.

    Returns:
        (success, feedback_info)
        feedback_info contains:
          - 'correct': bool
          - 'mode': 'wrong_only' | 'wrong_positions' (from config, passed through)
          - 'mask': List[bool] — per-position correct/wrong (for wrong_positions)
          - 'submitted': the completion that was confirmed
    """
    state.confirm_count += 1
    submitted = list(state.completion)
    gt = state.ground_truth

    # Build per-position mask
    L = len(gt)
    mask = []
    for i in range(L):
        if i < len(submitted) and submitted[i] is not None:
            mask.append(submitted[i] == gt[i])
        else:
            mask.append(False)

    correct = all(mask) and len(mask) == L and all(
        c is not None for c in submitted[:L])

    if correct:
        state.outcome = Outcome.SUCCESS
    elif state.confirm_count >= state.n_confirm_max:
        state.outcome = Outcome.TIMEOUT

    feedback = {
        'correct': correct,
        'mask': mask,
        'submitted': submitted,
    }
    return correct, feedback


def retry_refresh(
    state: QueryState,
    new_pool: List[CandidateBall],
) -> QueryState:
    """Refresh the candidate pool. Completion state is preserved.

    Args:
        state: current query state
        new_pool: freshly generated candidate pool

    Returns:
        Updated state with new pool.
    """
    state.candidate_pool = new_pool
    state.retry_count += 1
    state.consecutive_retries += 1

    # Stuck-retry guard
    if state.consecutive_retries >= state.max_retry_per_confirm_window:
        state.stuck_retry_events += 1
        # Phase 1: log only, do not terminate
        state.step_log.append({
            'event': 'stuck_retry',
            'consecutive_retries': state.consecutive_retries,
        })

    return state


def apply_death(state: QueryState) -> QueryState:
    """Apply death outcome to the query state."""
    state.outcome = Outcome.DEATH
    state.danger_select_count += 1
    return state


def apply_warning(
    state: QueryState,
    selected: List[CandidateBall],
) -> QueryState:
    """Apply warning: discard the selected set, log the event.

    The learner does not place anything; the selection is voided.
    """
    state.danger_select_count += 1
    state.step_log.append({
        'event': 'warning',
        'selected_indices': [b.index for b in selected],
        'danger_count': sum(1 for b in selected if b.is_danger),
    })
    return state
