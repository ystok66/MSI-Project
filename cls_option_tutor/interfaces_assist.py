"""
assist_level.py — Assist level ranking constants and helpers.

Shared module used by both env.interventions and tutor.observation_adapter
to avoid circular imports.
"""
from __future__ import annotations

# ── Assist level ranking ──────────────────────────────────────────────────────

ASSIST_RANK = {
    "none": 0,
    "risk_hint": 0,       # semantic update not discounted
    "self_correct": 0,    # Phase 6.4: correct after wrong exploration, no tutor assist
    "highlight": 1,
    "ban": 1,
    "mix": 1,
    "shortlist": 2,
    "direct_answer": 2,
}


def merge_assist_level(old: str, new: str) -> str:
    """Return the higher-rank assist level."""
    return new if ASSIST_RANK.get(new, 0) > ASSIST_RANK.get(old, 0) else old
