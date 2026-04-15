"""
task_model.py — TutorTaskModel: tutor's knowledge of the TASK (ground truth).

This is NOT a model of the learner. It represents what the tutor
KNOWS about the correct answers, danger locations, and grammar rules.

Separation principle:
    TutorTaskModel → "What IS correct" (teacher knowledge)
    TutorLearnerModel → "What learner THINKS is correct" (learner modeling)

The tutor uses TaskModel for:
    - Generating correct hints (never wrong)
    - Oracle danger detection (same as T0/T1/T2)
    - Knowing the ground truth output for any query
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from ..interfaces import CandidateBall, Example
from ..environment.grammar_task_env import GrammarTaskEnv


class TutorTaskModel:
    """Tutor's knowledge of the task (ground truth).

    This represents the tutor's TEACHER KNOWLEDGE:
    - Correct grammar output for any query
    - Which balls are dangerous
    - Correct hint positions

    This knowledge is ORACLE-level — the tutor knows the right answer.
    The question is not whether tutor knows the answer, but whether
    tutor knows what the LEARNER knows (that's TutorLearnerModel's job).
    """

    def __init__(self, env: GrammarTaskEnv,
                 queries: Optional[List[Example]] = None):
        self._grammar = env.grammar
        self._danger_model = env.danger_model
        # Build word-sequence → output mapping from queries + support
        self._ground_truth_map: Dict[tuple, List[str]] = {}
        if queries:
            for q in queries:
                self._ground_truth_map[tuple(q.words)] = list(q.output)

    def register_queries(self, queries: List[Example]):
        """Register additional queries for ground truth lookup."""
        for q in queries:
            self._ground_truth_map[tuple(q.words)] = list(q.output)

    def ground_truth_output(self, words: List[str]) -> Optional[List[str]]:
        """Return the correct output for a query.

        Looks up from registered query ground truths.
        Returns None if query not registered.
        """
        key = tuple(words)
        return self._ground_truth_map.get(key)

    def is_danger(self, ball: CandidateBall) -> bool:
        """Oracle danger check.

        Same level of access as T0/T1/T2 — tutor always knows
        which balls are dangerous.
        """
        return ball.is_danger

    def any_danger_in(self, selected: List[CandidateBall]) -> bool:
        """Check if any ball in a selection is dangerous."""
        return any(b.is_danger for b in selected)

    def generate_hint(
        self,
        ground_truth: List[str],
        submitted: List[str],
        max_hints: int = 2,
    ) -> List[Tuple[int, str]]:
        """Generate correct hint positions from ground truth.

        Compares submitted output (wrong) against ground truth,
        returns (position, correct_color) tuples for wrong positions.

        NOTE: Hints are ALWAYS correct because they come from
        ground truth, not from the tutor's guess about the learner.

        Args:
            ground_truth: correct output sequence
            submitted: learner's wrong submission
            max_hints: max number of positions to hint

        Returns:
            [(pos, correct_color), ...] for up to max_hints wrong positions
        """
        hints = []
        for i, (sub, gold) in enumerate(zip(submitted, ground_truth)):
            if sub is not None and sub != gold:
                hints.append((i, gold))
        return hints[:max_hints]

    def generate_hint_from_feedback(
        self,
        ground_truth: List[str],
        feedback: dict,
        max_hints: int = 2,
    ) -> List[Tuple[int, str]]:
        """Generate hints using feedback mask + ground truth.

        Uses the wrong-position mask from feedback to identify
        which positions to correct, then provides the ground truth
        color for those positions.

        Args:
            ground_truth: correct output
            feedback: dict with 'mask' and 'submitted'
            max_hints: max positions to hint

        Returns:
            [(pos, correct_color), ...]
        """
        submitted = feedback.get('submitted', [])
        mask = feedback.get('mask', [])
        hints = []
        for i, (correct_flag, gold) in enumerate(zip(mask, ground_truth)):
            if not correct_flag:
                hints.append((i, gold))
        return hints[:max_hints]
