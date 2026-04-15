"""
state.py — QueryState and EpisodeState for the color-selection environment.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..constants import Outcome, EMPTY
from ..interfaces import CandidateBall


@dataclass
class QueryState:
    """Mutable state for one query within an episode.

    Tracks the target-aligned completion, candidate pool, counters,
    and terminal condition.
    """
    # ── Identity ──
    query_id: int
    query_words: List[str]

    # ── Grammar ──
    target_output: List[str]        # Y* from CLS learner
    ground_truth: List[str]         # true answer from grammar (for scoring)
    grammar_colors: List[str]       # palette of colors in this grammar

    # ── Completion state (target-aligned) ──
    completion: List[Optional[str]]  # z_t: EMPTY or color at each position

    # ── Candidate pool ──
    candidate_pool: List[CandidateBall]

    # ── Counters ──
    confirm_count: int = 0
    retry_count: int = 0
    consecutive_retries: int = 0    # resets after a successful place
    stuck_retry_events: int = 0
    danger_select_count: int = 0    # for immortal baselines

    # ── Config limits ──
    n_confirm_max: int = 5
    max_retry_per_confirm_window: int = 10

    # ── Terminal ──
    outcome: Outcome = Outcome.IN_PROGRESS

    # ── Step log (for diagnostics) ──
    step_log: List[dict] = field(default_factory=list)

    # ── Properties ──
    @property
    def is_terminal(self) -> bool:
        return self.outcome != Outcome.IN_PROGRESS

    @property
    def L(self) -> int:
        """Length of target output."""
        return len(self.target_output)

    @property
    def filled_count(self) -> int:
        """Number of filled positions."""
        return sum(1 for c in self.completion if c is not None)

    @property
    def fill_ratio(self) -> float:
        """Fraction of positions filled."""
        return self.filled_count / max(self.L, 1)

    @property
    def is_complete(self) -> bool:
        """Whether all positions are filled."""
        return self.filled_count == self.L

    def color_gaps(self) -> Dict[str, int]:
        """Remaining color gaps: δ_t(c) = need(c) - have(c)."""
        need: Dict[str, int] = {}
        for c in self.target_output:
            need[c] = need.get(c, 0) + 1
        have: Dict[str, int] = {}
        for c in self.completion:
            if c is not None:
                have[c] = have.get(c, 0) + 1
        gaps = {}
        for c, n in need.items():
            gap = n - have.get(c, 0)
            if gap > 0:
                gaps[c] = gap
        return gaps

    def needed_colors(self) -> set:
        """Set of colors still needed."""
        return set(self.color_gaps().keys())

    def clone(self) -> 'QueryState':
        """Deep copy for rollout / counterfactual."""
        import copy
        return copy.deepcopy(self)


@dataclass
class EpisodeState:
    """State for an entire episode (support → teach → eval)."""
    task_id: str
    seed: int
    grammar_colors: List[str]
    support_examples: List  # List[Example]
    teach_queries: List     # List[Example]
    eval_queries: List      # List[Example]
    current_phase: str = 'support'  # 'support' | 'teach' | 'eval'
    current_query_idx: int = 0
