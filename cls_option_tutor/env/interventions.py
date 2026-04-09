"""
interventions.py — V2: RISK_HINT / HIGHLIGHT / SKIP / WAIT semantics.

V2 changes:
  - BAN archived (kept for legacy but not used in canonical pipeline)
  - RISK_HINT added: coarse hazard teaching signal
  - Refresh: preserves HIGHLIGHT (text unchanged), clears BAN/RISK_HINT only
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from .state import QueryState
from ..interfaces import TutorStep, RiskHintEvent


def apply_wait(qs: QueryState, round_t: int) -> TutorStep:
    """WAIT: no intervention this round."""
    return TutorStep(
        round_t=round_t,
        query_id=qs.query_id,
        action="WAIT",
    )


def apply_risk_hint(
    qs: QueryState,
    hint_index: int,
    round_t: int,
    eta: float = 0.8,
) -> TutorStep:
    """RISK_HINT: mark an option as potentially risky. [V2]

    Does NOT prevent learner from choosing it (unlike BAN).
    Provides a weak hazard label for the learner's hazard head.
    Expires after refresh (risk landscape changes).
    """
    if hint_index < 0 or hint_index >= len(qs.menu):
        raise ValueError(f"RISK_HINT index {hint_index} out of range")
    if hint_index in qs.risk_hints:
        raise ValueError(f"Option {hint_index} already risk-hinted")

    qs.risk_hints.add(hint_index)
    event = RiskHintEvent(round_t=round_t, option_index=hint_index, eta=eta)
    qs.risk_hint_history.append(event)

    return TutorStep(
        round_t=round_t,
        query_id=qs.query_id,
        action="RISK_HINT",
        hint_index=hint_index,
    )


def apply_ban(
    qs: QueryState,
    ban_index: int,
    round_t: int,
) -> TutorStep:
    """BAN: remove one option from active menu. [ARCHIVED in V2]

    Kept for backward compatibility. c_ban=10.0 effectively disables it.
    """
    if ban_index < 0 or ban_index >= len(qs.menu):
        raise ValueError(f"BAN index {ban_index} out of range")
    if ban_index in qs.banned_indices:
        raise ValueError(f"Option {ban_index} already banned")
    qs.banned_indices.add(ban_index)
    return TutorStep(
        round_t=round_t,
        query_id=qs.query_id,
        action="BAN",
        ban_index=ban_index,
    )


def apply_highlight(
    qs: QueryState,
    cells: Tuple[int, ...],
    max_cells: int = 2,
    round_t: int = 0,
) -> TutorStep:
    """HIGHLIGHT: highlight ≤ max_cells target output cells.

    Modifies learner attention weights.
    V2: Persists through refresh (text doesn't change).
    """
    if len(cells) > max_cells:
        raise ValueError(f"HIGHLIGHT max {max_cells} cells, got {len(cells)}")
    L = len(qs.target_output)
    for c in cells:
        if c < 0 or c >= L:
            raise ValueError(f"HIGHLIGHT cell {c} out of range [0, {L})")
    qs.highlighted_cells = cells
    return TutorStep(
        round_t=round_t,
        query_id=qs.query_id,
        action="HIGHLIGHT",
        highlight_cells=cells,
    )


def apply_skip(qs: QueryState, round_t: int) -> TutorStep:
    """SKIP: end the current query immediately.

    Query terminates with skipped=True, no damage.
    """
    qs.done = True
    qs.skipped = True
    return TutorStep(
        round_t=round_t,
        query_id=qs.query_id,
        action="SKIP",
    )


def clear_menu_interventions(qs: QueryState) -> None:
    """Clear risk-sensitive interventions after refresh. [V2]

    V2 semantics:
      - BAN/RISK_HINT: cleared (risk landscape changed)
      - HIGHLIGHT: PRESERVED (text unchanged, semantic hint still valid)
    """
    qs.banned_indices.clear()
    qs.risk_hints.clear()
    # NOTE: qs.highlighted_cells intentionally NOT cleared in V2


def get_active_menu(qs: QueryState) -> list:
    """Return menu options excluding banned ones."""
    return [o for o in qs.menu if o.index not in qs.banned_indices]
