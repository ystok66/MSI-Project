"""
state.py — QueryState, BlockState, ProfileState.

Implements §15 of the spec.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from ..interfaces import Option, RevealEvent, RiskHintEvent, LearnerStep, TutorStep, Example


@dataclass
class QueryState:
    """Full mutable state for one query within a block."""
    query_id: int
    target_output: List[str]              # Y* — the target color grid
    true_program: List[str]               # ν* — the hidden true utterance
    hp: int                               # current HP
    rounds_used: int = 0
    max_rounds: int = 5
    menu: List[Option] = field(default_factory=list)
    banned_indices: set = field(default_factory=set)
    risk_hints: set = field(default_factory=set)   # [V2] risk-hinted option indices
    highlighted_cells: Tuple[int, ...] = ()
    reveal_history: List[RevealEvent] = field(default_factory=list)
    risk_hint_history: List[RiskHintEvent] = field(default_factory=list)  # [V2]
    refreshes_used: int = 0
    max_refreshes: int = 2                # V2: was 3
    done: bool = False
    success: bool = False
    skipped: bool = False


@dataclass
class ProfileState:
    """Compact tutor-side learner profile (§9.1).

    Inferred from observation queries; used during teaching queries.
    """
    lambda_risk: float = 0.5              # aversion to predicted damage
    lambda_refresh: float = 0.3           # tendency to redraw menus
    s_order: float = 0.0                  # vulnerability to order confusion
    s_scope: float = 0.0                  # vulnerability to scope/nesting confusion
    g_highlight: float = 1.0              # sensitivity to highlight intervention
    semantic_competence: float = 0.5      # 0=novice, 0.5=intermediate, 1.0=advanced
    tau_trust: float = 0.3                # trust (reserved, inactive)

    def as_dict(self) -> dict:
        return {
            "lambda_risk": self.lambda_risk,
            "lambda_refresh": self.lambda_refresh,
            "s_order": self.s_order,
            "s_scope": self.s_scope,
            "g_highlight": self.g_highlight,
            "semantic_competence": self.semantic_competence,
            "tau_trust": self.tau_trust,
        }


@dataclass
class BlockState:
    """Full mutable state for one block (= one few-shot episode).

    4-phase lifecycle:
      Phase 1 (Pre-train): CLS studies n_sup support examples (before block)
      Phase 2 (Observation): Tutor watches frozen learner (N_obs queries)
      Phase 3 (Teaching): Tutor intervenes, learner CLS learns (N_teach queries)
      Phase 4 (Evaluation): Frozen learner, no tutor (N_eval queries)
    """
    block_id: int = 0
    support_examples: List[Example] = field(default_factory=list)
    queries: List[QueryState] = field(default_factory=list)
    current_query_idx: int = 0
    learner_trace: List[LearnerStep] = field(default_factory=list)
    tutor_trace: List[TutorStep] = field(default_factory=list)
    obs_phase_queries: int = 2            # Phase 2: observe only
    teach_phase_queries: int = 3          # Phase 3: tutor may intervene
    eval_phase_queries: int = 3           # Phase 4: evaluation (frozen)
    profile_state: ProfileState = field(default_factory=ProfileState)
    done: bool = False

    # Aggregate metrics
    total_correct: int = 0
    total_damage: int = 0
    total_rounds: int = 0
    total_skips: int = 0
    total_refreshes: int = 0

    @property
    def current_query(self) -> Optional[QueryState]:
        if self.current_query_idx < len(self.queries):
            return self.queries[self.current_query_idx]
        return None

    @property
    def in_observation_phase(self) -> bool:
        return self.current_query_idx < self.obs_phase_queries

    @property
    def in_teaching_phase(self) -> bool:
        teach_start = self.obs_phase_queries
        teach_end = self.obs_phase_queries + self.teach_phase_queries
        return teach_start <= self.current_query_idx < teach_end

    @property
    def in_evaluation_phase(self) -> bool:
        eval_start = self.obs_phase_queries + self.teach_phase_queries
        return self.current_query_idx >= eval_start
