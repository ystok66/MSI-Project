"""
observation_v2.py — Result-level observation for Phase 4 inverse tutor.

Unlike v1 which exposes per-step process info (selections, retries, warnings),
v2 only records what a REALISTIC tutor would see:
    - query words
    - learner's final submitted output
    - outcome (success / wrong / timeout / death)
    - wrong-position mask (if feedback mode = wrong_positions)

The learner still runs the full query loop internally (it needs step-level
interaction to learn). But the observation summary returned to the tutor
contains ONLY result-level info.

Observation level contract:
    Obs phase  → result-level ONLY (no per-step picks, retries, warnings)
    Teach phase → process-level (tutor can see selected sets, confirms)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import copy
import numpy as np

from ..config import FullConfig
from ..interfaces import QueryResult, Example
from ..constants import Outcome


# ── Result-level observation record ───────────────────────────

@dataclass
class ObservationRecord:
    """One result-level observation from obs phase.

    This is ALL the tutor sees per obs query. No process info.

    Fields:
        query_words: input sentence
        submitted_output: learner's confirmed output (if confirm happened)
        current_completion: learner's partial completion at end of query
            (used when submitted_output is None, e.g. death before confirm)
        outcome: SUCCESS / TIMEOUT / DEATH
        correct: whether submitted matched ground truth
        wrong_mask: per-position correct/wrong (if wrong_positions feedback)
        evidence_type: auto-classified strength of this observation
            'confirm_mask'  — submitted + mask (strong, default for inverse update)
            'confirm_only'  — submitted, no mask (medium)
            'partial'       — only completion available (weak, ablation only)
    """
    query_words: List[str]
    submitted_output: Optional[List[str]]     # from last confirm attempt
    current_completion: Optional[List[str]]    # partial at end of query
    outcome: Outcome
    correct: bool
    wrong_mask: Optional[List[bool]] = None

    @property
    def evidence_type(self) -> str:
        """Auto-classify evidence strength."""
        if self.submitted_output is not None and self.wrong_mask is not None:
            return 'confirm_mask'
        elif self.submitted_output is not None:
            return 'confirm_only'
        return 'partial'

    @property
    def best_output(self) -> Optional[List[str]]:
        """Best available output: submitted if exists, else completion."""
        return self.submitted_output or self.current_completion


# ── Result-level observation summary ──────────────────────────

@dataclass
class ObservationSummaryV2:
    """Statistics collected during observation phase (result-level only).

    Unlike v1's ObservationSummary, this does NOT contain:
        - per-step retry counts
        - per-step danger select counts
        - per-step stuck retry events
        - beam entropies
        - counterfactual death counts
    All stats are derived from result-level records only.
    """
    records: List[ObservationRecord] = field(default_factory=list)

    @property
    def n_queries(self) -> int:
        return len(self.records)

    @property
    def n_correct(self) -> int:
        return sum(1 for r in self.records if r.correct)

    @property
    def n_wrong(self) -> int:
        return sum(1 for r in self.records
                   if not r.correct and r.outcome != Outcome.DEATH)

    @property
    def n_timeout(self) -> int:
        return sum(1 for r in self.records if r.outcome == Outcome.TIMEOUT)

    @property
    def n_death(self) -> int:
        return sum(1 for r in self.records if r.outcome == Outcome.DEATH)

    @property
    def success_rate(self) -> float:
        return self.n_correct / max(self.n_queries, 1)

    def to_dict(self) -> Dict[str, float]:
        """Flat dict for belief initialization (compatible with v1)."""
        return {
            'ObsN': self.n_queries,
            'ObsSuccessRate': self.success_rate,
            'ObsDeathRate': self.n_death / max(self.n_queries, 1),
            'ObsTimeoutRate': self.n_timeout / max(self.n_queries, 1),
            # NOTE: no process-level stats (retries, danger selects, etc.)
        }

    # ── Guard: explicitly NOT present ──
    # These properties exist only to raise clear errors if someone
    # tries to access process-level info from this result-level summary.

    @property
    def total_retries(self):
        raise AttributeError(
            "ObservationSummaryV2 is result-level only. "
            "total_retries is process-level info not available to tutor.")

    @property
    def total_danger_selects(self):
        raise AttributeError(
            "ObservationSummaryV2 is result-level only. "
            "total_danger_selects is process-level info not available to tutor.")


# ── Runner ────────────────────────────────────────────────────

def run_observation_phase_v2(
    env,
    obs_queries: List[Example],
    policy,
    risk_belief,
    feedback_updater,
    predictor,
    target_pred,
    rng: np.random.Generator,
    cfg: FullConfig,
) -> ObservationSummaryV2:
    """Run observation phase, returning ONLY result-level info.

    Internally the learner runs the full query loop (it needs interaction
    to learn risk). But the returned ObservationSummaryV2 only contains
    result-level records — what a realistic tutor would observe.

    Args:
        env: GrammarTaskEnv with loaded task
        obs_queries: queries to observe
        policy: ColorSelectionPolicy
        risk_belief: DangerTypeBelief (deep-copied, not modified)
        feedback_updater: FeedbackUpdater
        predictor: CLSSequencePredictor
        target_pred: TargetPredictor
        rng: random generator
        cfg: full config

    Returns:
        ObservationSummaryV2 with result-level records only
    """
    from ..tutor_api.dummy_tutor import NoTutorImmortalWarnlike
    from ..learner.memory import QueryMemory

    # Deep-copy risk belief so observation doesn't modify teaching state
    frozen_risk = copy.deepcopy(risk_belief)

    # Use immortal warning tutor — learner observes but doesn't die
    obs_tutor = NoTutorImmortalWarnlike()

    summary = ObservationSummaryV2()

    for qi, query in enumerate(obs_queries):
        y_star = target_pred.predict_target(query.words)
        state = env.init_query(query, query_id=qi, target_output=y_star)
        memory = QueryMemory()

        # Run with the main loop (learner needs step-level interaction)
        from ..experiments.run_phase1 import run_single_query

        result = run_single_query(
            env, state, policy, frozen_risk, feedback_updater,
            predictor, target_pred, obs_tutor, memory, rng, cfg,
            immortal=True,         # never die during observation
            enable_feedback=False,  # no grammar updates during obs
        )

        # Extract result-level info ONLY
        # -- Get submitted output from step_log if available
        submitted = None
        current_completion = None
        wrong_mask = None
        correct = (result.outcome == Outcome.SUCCESS)

        # Search step_log for last confirm event
        for step in reversed(state.step_log):
            if step.get('event') == 'confirm':
                submitted = step.get('submitted')
                wrong_mask = step.get('mask')
                break

        # Current completion at end of query
        current_completion = list(state.completion)

        # If no confirm happened but we have completion, use it
        if submitted is None and result.outcome == Outcome.SUCCESS:
            # Success means ground_truth was matched
            submitted = list(state.ground_truth)
            correct = True

        record = ObservationRecord(
            query_words=list(query.words),
            submitted_output=submitted,
            current_completion=current_completion,
            outcome=result.outcome,
            correct=correct,
            wrong_mask=wrong_mask,
        )
        summary.records.append(record)

    return summary
