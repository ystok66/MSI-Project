"""
tutor_policy.py — Tutor action selection from scored interventions.

Implements §11.4:
    a*_T = argmax_a Q(a | s, profile)
    with WAIT as the default (Q(WAIT) = 0 baseline).

Only acts during teaching phase (observation phase → always WAIT).
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

from ..config import TutorConfig
from ..env.state import BlockState, QueryState, ProfileState
from ..interfaces import TutorStep
from .counterfactual import CounterfactualScorer, InterventionScore
from ..learner.semantic_scorer import DeterministicSemanticScorer
from ..learner.danger_head import DangerHead


class TutorPolicy:
    """Tutor action selection policy.

    Lifecycle per block:
        1. Observation phase: WAIT on all queries, collect learner trace
        2. Profile inference (done externally)
        3. Teaching phase: score interventions → pick best action
    """

    def __init__(self, cfg: TutorConfig):
        self.cfg = cfg
        self.cf_scorer = CounterfactualScorer(cfg)

    def select_action(
        self,
        block: BlockState,
        scorer: DeterministicSemanticScorer,
        danger_head: Optional[DangerHead] = None,
    ) -> Tuple[str, dict]:
        """Select the best tutor action for the current query.

        Returns (action_name, kwargs) for env.tutor_act().
        """
        # Observation phase → always WAIT
        if block.in_observation_phase:
            return "WAIT", {}

        qs = block.current_query
        if qs is None or qs.done:
            return "WAIT", {}

        # Score all interventions
        candidates = self.cf_scorer.score_all(
            qs,
            profile=block.profile_state,
            scorer=scorer,
            danger_head=danger_head,
        )

        if not candidates:
            return "WAIT", {}

        # Pick best (already sorted descending)
        best = candidates[0]

        # Only intervene if strictly better than WAIT
        wait_q = next((c.total_q for c in candidates if c.action == "WAIT"), 0.0)
        if best.total_q <= wait_q:
            return "WAIT", {}

        # Convert to env action + kwargs
        if best.action == "RISK_HINT":
            return "RISK_HINT", {"hint_index": best.hint_index}
        elif best.action == "BAN":
            return "BAN", {"ban_index": best.ban_index}
        elif best.action == "HIGHLIGHT":
            return "HIGHLIGHT", {"highlight_cells": best.highlight_cells}
        elif best.action == "SKIP":
            return "SKIP", {}
        else:
            return "WAIT", {}

    def get_diagnostics(
        self,
        block: BlockState,
        scorer: DeterministicSemanticScorer,
        danger_head: Optional[DangerHead] = None,
    ) -> List[InterventionScore]:
        """Get full Q-value breakdown for diagnostics."""
        qs = block.current_query
        if qs is None or qs.done:
            return []

        return self.cf_scorer.score_all(
            qs,
            profile=block.profile_state,
            scorer=scorer,
            danger_head=danger_head,
        )
