"""
direct_answer_tutor.py — Benchmark-only oracle baseline.

Always SHORTLIST([j*]) in teach phase.
This is NOT a normal tutor — it's a benchmark ceiling for assist-gap measurement.

Does NOT change risk_class or danger_vec of the correct option.
The tutor only reveals which option is semantically correct.
"""
from __future__ import annotations

from typing import Optional

from ..env.option_env import OptionEnv
from ..env.state import BlockState, QueryState
from ..env.interventions import apply_wait, apply_shortlist, get_active_menu
from ..learner.learner_agent import LearnerAgent
from ..config import FullConfig


class DirectAnswerTutor:
    """Benchmark-only tutor: always reveal j* via SHORTLIST([j*]).

    Not added to sparse tutor candidate search.
    Used only for direct-answer baseline experiments.

    Protocol:
      - obs phase: WAIT (pure observation)
      - teach phase: SHORTLIST([j*]) on first round of each query
      - eval phase: WAIT (frozen)
    """

    def __init__(self, cfg: FullConfig):
        self.cfg = cfg

    def run_block(
        self,
        env: OptionEnv,
        learner: LearnerAgent,
        task_id: str,
        seed: int = 42,
    ) -> BlockState:
        """Run a complete block with direct-answer intervention."""
        block = env.reset_block(task_id, seed=seed)
        support, _, grammar = env.adapter.load_task(task_id)
        learner.init_block(block, grammar, support)

        max_steps = len(block.queries) * 20
        steps = 0

        while not block.done and steps < max_steps:
            steps += 1
            qs = block.current_query
            if qs is None:
                break

            if qs.done:
                block.current_query_idx += 1
                if block.current_query_idx >= len(block.queries):
                    block.done = True
                continue

            # Tutor action
            if block.in_teaching_phase and qs.rounds_used == 0:
                # SHORTLIST the correct option
                j_star_idx = self._find_correct_index(qs)
                if j_star_idx is not None:
                    ts = apply_shortlist(qs, [j_star_idx], round_t=qs.rounds_used)
                    block.tutor_trace.append(ts)
                else:
                    ts = apply_wait(qs, round_t=qs.rounds_used)
                    block.tutor_trace.append(ts)
            else:
                ts = apply_wait(qs, round_t=qs.rounds_used)
                block.tutor_trace.append(ts)

            if qs.done:
                continue

            # Learner acts
            learner.act(block, env)

        if not block.done:
            block.done = True

        return block

    @staticmethod
    def _find_correct_index(qs: QueryState) -> Optional[int]:
        """Find the menu index of j* (correct option)."""
        for opt in qs.menu:
            if opt.is_correct:
                return opt.index
        return None
