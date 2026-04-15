"""
interfaces.py — Core dataclasses for cls_color_selection.

All inter-module communication uses these typed structures.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .constants import Outcome, TutorActionType


# ── Candidate ball ─────────────────────────────────────────────

@dataclass
class CandidateBall:
    """One ball in the candidate pool.

    Attributes:
        index: position in current candidate pool (0..n-1)
        color: true color label (e.g. 'RED')
        danger_vec: hidden ground-truth danger vector (m,)
        observed_vec: noisy observation available to learner (m,)
        is_danger: ground truth — True if this ball is dangerous
        danger_type: 0 = safe, 1..K = danger type index
    """
    index: int
    color: str
    danger_vec: np.ndarray
    observed_vec: np.ndarray
    is_danger: bool
    danger_type: int = 0


# ── Tutor action ───────────────────────────────────────────────

@dataclass
class TutorAction:
    """One tutor intervention.

    Phase 1: only WAIT and WARNING are active.
    Phase 2 will add HINT with placed_balls.
    """
    action_type: TutorActionType = TutorActionType.WAIT
    # Phase 2 fields (stubs)
    hint_positions: Optional[List[Tuple[int, str]]] = None  # [(pos, color), ...]
    message: str = ""


# ── Query result ───────────────────────────────────────────────

@dataclass
class QueryResult:
    """Result of one query within an episode."""
    query_id: int
    query_words: List[str]
    target_output: List[str]       # Y* from CLS
    ground_truth: List[str]        # true answer from grammar
    outcome: Outcome = Outcome.IN_PROGRESS
    confirm_count: int = 0
    retry_count: int = 0
    death_count: int = 0           # 0 or 1 (for teach); baselines may count
    danger_select_count: int = 0   # how many times danger was selected (immortal baseline)
    stuck_retry_events: int = 0
    final_completion: Optional[List[Optional[str]]] = None
    # Timing / detail
    steps: List[dict] = field(default_factory=list)  # per-step log entries


# ── Episode result ─────────────────────────────────────────────

@dataclass
class EpisodeResult:
    """Result of one full episode (support → teach → eval)."""
    task_id: str
    seed: int
    teach_results: List[QueryResult] = field(default_factory=list)
    eval_results: List[QueryResult] = field(default_factory=list)
    # Learner state snapshots for diagnostics
    diagnostics: dict = field(default_factory=dict)


# ── Grammar example (reuse-compatible with CLS) ───────────────

@dataclass
class Example:
    """One input→output pair. Compatible with CLS learner's Example."""
    words: List[str]
    output: List[str]
    meta: dict = field(default_factory=dict)
