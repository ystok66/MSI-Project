"""
observation_adapter.py — Public observation records for the inverse tutor.

ObservedStep is a frozen dataclass containing ONLY public information
derivable from BlockState traces.  It must NEVER hold Option objects
(which contain hidden is_correct, risk_class, rendered_output for
unchosen options).

All danger_vecs are defensive copies (.copy()) to prevent shared-reference
leaks back to the environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..env.state import BlockState, QueryState
from ..env.interventions import get_active_menu
from ..interfaces import LearnerStep, TutorStep, RevealEvent, RiskHintEvent
from ..interfaces_assist import ASSIST_RANK, merge_assist_level  # re-export


_ACTION_TO_ASSIST = {
    "WAIT": "none",
    "RISK_HINT": "risk_hint",
    "HIGHLIGHT": "highlight",
    "BAN": "ban",
    "MIX": "mix",
    "SHORTLIST": "direct_answer",
    "SKIP": "none",
    "PASS": "none",
}


def _derive_assist_level(tutor_steps: list, learner_round_t: int) -> str:
    """Derive assist level from causally upstream tutor steps.

    Finds the latest tutor step with round_t <= learner_round_t,
    preferring higher-rank interventions if multiple exist at
    the same round.
    """
    best_level = "none"
    best_round = -1
    for ts in tutor_steps:
        tr = getattr(ts, 'round_t', -1)
        if tr <= learner_round_t:
            level = _ACTION_TO_ASSIST.get(ts.action, "none")
            # Prefer higher round, then higher rank
            if tr > best_round or (
                tr == best_round
                and ASSIST_RANK.get(level, 0) > ASSIST_RANK.get(best_level, 0)
            ):
                best_level = level
                best_round = tr
    return best_level


# ── Public observation record ─────────────────────────────────────────────────

@dataclass(frozen=True)
class ObservedStep:
    """One public observation of the learner acting in the environment.

    This record is the ONLY input to InverseShadowPredictor.observe().

    Invariant: contains NO hidden information about unchosen options.
    Specifically:
      - No is_correct for unchosen options
      - No risk_class for unchosen options
      - No rendered_output for unchosen options
      - No scorer weights / danger head posteriors / attention weights
      - No semantic scores
    """
    # ── Identity ──
    step_id: int                             # global step counter
    phase: str                               # "obs" | "teach" | "eval"
    query_id: int
    round_t: int                             # round within query

    # ── Public menu state (before learner action) ──
    option_texts: Tuple[Tuple[str, ...], ...]  # frozen text per active option
    option_danger_vecs: Tuple[np.ndarray, ...]  # (m,) per active option, defensive copy
    option_indices: Tuple[int, ...]            # menu indices of active options
    target_output: Tuple[str, ...]             # Y*

    # ── Intervention state (applied before learner acts) ──
    active_bans: Tuple[int, ...]
    active_highlights: Tuple[int, ...]
    active_risk_hints: Tuple[int, ...]

    # ── HP / round budget ──
    hp_before: int
    hp_after: int
    rounds_before: int                       # rounds_used before action
    rounds_after: int                        # rounds_used after action

    # ── Learner action (public outcome) ──
    learner_action: str                      # "pick" | "refresh" | "timeout"
    learner_pick_index: Optional[int]        # menu index if pick
    pick_correct: Optional[bool]             # True/False if pick, None if refresh
    pick_damage: Optional[int]               # realized damage if wrong pick

    # ── Reveal (only present if feedback_mode="reveal" AND wrong pick) ──
    revealed_output: Optional[Tuple[str, ...]]
    revealed_danger_vec: Optional[np.ndarray]  # defensive copy

    # ── Outcome classification ──
    outcome: str                             # "pick_correct" | "pick_wrong" | "refresh"
                                             # | "death" | "timeout"

    # ── Assist level (reserved for Phase 6 assist discount) ──
    assist_level: str                        # "none" | "highlight" | "ban" | "mix"
                                             # | "risk_hint" | "direct_answer"


# ── Adapter: BlockState → List[ObservedStep] ─────────────────────────────────

class ObservationAdapter:
    """Converts BlockState traces into public ObservedStep records.

    Usage:
        adapter = ObservationAdapter()
        steps = adapter.extract_steps(block)
        latest = adapter.extract_latest(block)
    """

    def extract_steps(self, block: BlockState) -> List[ObservedStep]:
        """Extract all public steps from a completed or in-progress block."""
        steps: List[ObservedStep] = []
        step_counter = 0

        # Build per-query lookup for learner steps and tutor steps
        learner_by_qid: dict = {}
        for ls in block.learner_trace:
            learner_by_qid.setdefault(ls.query_id, []).append(ls)

        tutor_by_qid: dict = {}
        for ts in block.tutor_trace:
            tutor_by_qid.setdefault(ts.query_id, []).append(ts)

        obs_end = block.obs_phase_queries
        teach_end = obs_end + block.teach_phase_queries

        for qi, qs in enumerate(block.queries):
            phase = self._classify_phase(qi, obs_end, teach_end)
            l_steps = learner_by_qid.get(qs.query_id, [])

            for li, ls in enumerate(l_steps):
                obs = self._build_step(
                    step_id=step_counter,
                    phase=phase,
                    qs=qs,
                    ls=ls,
                    round_t=li,
                    tutor_steps=tutor_by_qid.get(qs.query_id, []),
                )
                if obs is not None:
                    steps.append(obs)
                    step_counter += 1

        return steps

    def extract_latest(self, block: BlockState) -> Optional[ObservedStep]:
        """Extract the most recent public step (for online observe)."""
        if not block.learner_trace:
            return None

        ls = block.learner_trace[-1]
        qs = None
        for q in block.queries:
            if q.query_id == ls.query_id:
                qs = q
                break
        if qs is None:
            return None

        qi = next(
            (i for i, q in enumerate(block.queries) if q.query_id == ls.query_id),
            0,
        )
        obs_end = block.obs_phase_queries
        teach_end = obs_end + block.teach_phase_queries
        phase = self._classify_phase(qi, obs_end, teach_end)

        round_t = sum(
            1 for prev in block.learner_trace
            if prev.query_id == ls.query_id
        ) - 1

        tutor_steps = [ts for ts in block.tutor_trace if ts.query_id == ls.query_id]

        return self._build_step(
            step_id=len(block.learner_trace) - 1,
            phase=phase,
            qs=qs,
            ls=ls,
            round_t=round_t,
            tutor_steps=tutor_steps,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_phase(qi: int, obs_end: int, teach_end: int) -> str:
        if qi < obs_end:
            return "obs"
        elif qi < teach_end:
            return "teach"
        else:
            return "eval"

    @staticmethod
    def _build_step(
        step_id: int,
        phase: str,
        qs: QueryState,
        ls: LearnerStep,
        round_t: int,
        tutor_steps: List[TutorStep],
    ) -> Optional[ObservedStep]:
        """Build one ObservedStep from a LearnerStep + QueryState context."""
        # Public menu projection (NO Option objects — only public fields)
        active = get_active_menu(qs)
        option_texts = tuple(tuple(o.text) for o in active)
        option_danger_vecs = tuple(
            np.asarray(o.danger_vec, dtype=float).copy() for o in active
        )
        option_indices = tuple(o.index for o in active)

        # Intervention state
        active_bans = tuple(sorted(qs.banned_indices))
        active_highlights = tuple(qs.highlighted_cells) if qs.highlighted_cells else ()
        active_risk_hints = tuple(sorted(qs.risk_hints))

        # Classify outcome
        if ls.action == "refresh":
            outcome = "refresh"
            pick_correct = None
            pick_damage = None
        elif ls.action == "pick":
            if ls.correct:
                outcome = "pick_correct"
            elif ls.hp_after <= 0:
                outcome = "death"
            else:
                outcome = "pick_wrong"
            pick_correct = ls.correct
            pick_damage = ls.damage
        else:
            return None  # unknown action type

        # Reveal info (only from wrong picks in reveal mode)
        revealed_output = None
        revealed_danger_vec = None
        if outcome in ("pick_wrong", "death") and qs.reveal_history:
            # Find the matching reveal event for this round
            for rev in reversed(qs.reveal_history):
                if rev.round_t == ls.round_t or rev.option_index == ls.pick_index:
                    revealed_output = tuple(rev.revealed_output)
                    revealed_danger_vec = np.asarray(
                        rev.danger_vec, dtype=float
                    ).copy()
                    break

        # Assist level from tutor trace — use causally upstream intervention.
        # Take the latest tutor step at or before this learner round.
        # Use ASSIST_RANK to pick the highest-rank intervention if multiple.
        assist_level = _derive_assist_level(tutor_steps, round_t)

        return ObservedStep(
            step_id=step_id,
            phase=phase,
            query_id=qs.query_id,
            round_t=round_t,
            option_texts=option_texts,
            option_danger_vecs=option_danger_vecs,
            option_indices=option_indices,
            target_output=tuple(qs.target_output),
            active_bans=active_bans,
            active_highlights=active_highlights,
            active_risk_hints=active_risk_hints,
            hp_before=ls.hp_before,
            hp_after=ls.hp_after,
            rounds_before=round_t,
            rounds_after=round_t + 1,
            learner_action=ls.action,
            learner_pick_index=ls.pick_index,
            pick_correct=pick_correct,
            pick_damage=pick_damage,
            revealed_output=revealed_output,
            revealed_danger_vec=revealed_danger_vec,
            outcome=outcome,
            assist_level=assist_level,
        )
