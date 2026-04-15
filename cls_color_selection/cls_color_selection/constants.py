"""
constants.py — Shared constants and enums for cls_color_selection.
"""
from __future__ import annotations
from enum import Enum, auto
from typing import List


# ── Default color palette (matches MLC tasks) ──────────────────
DEFAULT_COLORS: List[str] = ['BLUE', 'RED', 'GREEN', 'YELLOW', 'PURPLE', 'PINK']


# ── Query outcome ──────────────────────────────────────────────
class Outcome(Enum):
    """Terminal states for a query."""
    SUCCESS = auto()      # confirm matched Y*
    DEATH = auto()        # selected a danger ball, tutor WAIT
    TIMEOUT = auto()      # exceeded n_confirm_max
    IN_PROGRESS = auto()  # not yet terminal


# ── Learner actions ────────────────────────────────────────────
class LearnerAction(Enum):
    """Atomic learner actions within a query."""
    SELECT = auto()       # pick a subset from candidate pool
    CONFIRM = auto()      # submit current completion for checking
    RETRY = auto()        # refresh candidate pool (auto after select)


# ── Tutor actions (Phase 1: mostly stubs) ──────────────────────
class TutorActionType(Enum):
    """Tutor intervention types."""
    WAIT = auto()         # no intervention
    WARNING = auto()      # "your selection contains danger"
    HINT = auto()         # place balls for learner (Phase 2)
    COURAGE = auto()      # "there exists a safe needed ball" (Phase 1 minimal)


# ── Phase labels ───────────────────────────────────────────────
class Phase(Enum):
    """Experiment phases."""
    SUPPORT = auto()      # pre-training on support examples
    OBSERVATION = auto()  # tutor observes frozen learner (Phase 2)
    TEACH = auto()        # tutor-assisted teaching
    EVAL = auto()         # frozen evaluation


# ── Slot state ─────────────────────────────────────────────────
EMPTY = None  # sentinel for unfilled completion slot
