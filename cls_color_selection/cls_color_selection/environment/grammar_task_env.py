"""
grammar_task_env.py — Main environment for the color-selection task.

Implements the full query timeline:
  1. Learner selects balls from candidate pool
  2. Tutor hook: WARNING / WAIT
  3. If WAIT and danger: DEATH
  4. If WAIT and safe: AUTO-PLACE
  5. Learner can CONFIRM
  6. If wrong: feedback, tutor hook (HINT stub), retry
  7. If right: SUCCESS
  8. Auto-refresh candidate pool after each select
"""
from __future__ import annotations
from typing import Callable, List, Optional, Tuple
import numpy as np

from .state import QueryState, EpisodeState
from .transition import (
    select_balls, check_selection_has_danger,
    auto_place, confirm, retry_refresh, apply_death, apply_warning,
)
from .generator import (
    DangerModel, Grammar, generate_danger_model,
    generate_candidate_pool, parse_task_file, render_with_grammar,
)
from ..interfaces import CandidateBall, TutorAction, QueryResult, Example
from ..constants import Outcome, TutorActionType
from ..config import FullConfig


class GrammarTaskEnv:
    """Main environment for one episode of the color-selection task.

    Lifecycle:
        env = GrammarTaskEnv(cfg)
        env.load_task(task_id)
        for query in env.get_queries(phase):
            state = env.init_query(query)
            while not state.is_terminal:
                # learner selects, env processes
    """

    def __init__(self, cfg: FullConfig, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.exp.seed)

        # Loaded per-task
        self.grammar: Optional[Grammar] = None
        self.support: List[Example] = []
        self.queries: List[Example] = []
        self.danger_model: Optional[DangerModel] = None

    def load_task(self, task_path: str) -> Tuple[List[Example], List[Example], Grammar]:
        """Load a task file and generate danger model for this episode."""
        self.support, self.queries, self.grammar = parse_task_file(task_path)
        self.danger_model = generate_danger_model(self.cfg.env, self.rng)
        return self.support, self.queries, self.grammar

    def get_grammar_colors(self) -> List[str]:
        """Get the color palette from the loaded grammar."""
        if self.grammar is None:
            raise RuntimeError("No task loaded. Call load_task() first.")
        return self.grammar.colors

    def init_query(
        self,
        query: Example,
        query_id: int,
        target_output: Optional[List[str]] = None,
    ) -> QueryState:
        """Initialize a new query state.

        Args:
            query: the query Example (words + ground-truth output)
            query_id: index of this query
            target_output: Y* from CLS learner. If None, uses ground truth.
        """
        if self.grammar is None or self.danger_model is None:
            raise RuntimeError("No task loaded. Call load_task() first.")

        gt = query.output
        y_star = target_output if target_output is not None else gt
        colors = self.get_grammar_colors()

        # Generate initial candidate pool
        pool = generate_candidate_pool(
            grammar_colors=colors,
            target_output=y_star,
            n_candidates=self.cfg.env.n_candidates,
            danger_model=self.danger_model,
            cfg=self.cfg.env,
            rng=self.rng,
        )

        state = QueryState(
            query_id=query_id,
            query_words=query.words,
            target_output=y_star,
            ground_truth=gt,
            grammar_colors=colors,
            completion=[None] * len(y_star),
            candidate_pool=pool,
            n_confirm_max=self.cfg.env.n_confirm_max,
            max_retry_per_confirm_window=self.cfg.env.max_retry_per_confirm_window,
        )
        return state

    def init_query_with_rng(
        self,
        query: Example,
        query_id: int,
        target_output=None,
        query_rng: np.random.Generator = None,
    ) -> 'QueryState':
        """Init query with a per-query RNG for candidate pool.

        This ensures candidate pool generation is deterministic per query,
        invariant to how many obs queries ran before this one.

        Args:
            query: the query Example
            query_id: index
            target_output: Y* from CLS. If None, uses ground truth.
            query_rng: per-query RNG. If None, falls back to self.rng.
        """
        if self.grammar is None or self.danger_model is None:
            raise RuntimeError("No task loaded. Call load_task() first.")

        gt = query.output
        y_star = target_output if target_output is not None else gt
        colors = self.get_grammar_colors()
        pool_rng = query_rng if query_rng is not None else self.rng

        pool = generate_candidate_pool(
            grammar_colors=colors,
            target_output=y_star,
            n_candidates=self.cfg.env.n_candidates,
            danger_model=self.danger_model,
            cfg=self.cfg.env,
            rng=pool_rng,
        )

        state = QueryState(
            query_id=query_id,
            query_words=query.words,
            target_output=y_star,
            ground_truth=gt,
            grammar_colors=colors,
            completion=[None] * len(y_star),
            candidate_pool=pool,
            n_confirm_max=self.cfg.env.n_confirm_max,
            max_retry_per_confirm_window=self.cfg.env.max_retry_per_confirm_window,
        )
        return state

    def step_select(
        self,
        state: QueryState,
        selected_indices: List[int],
        tutor_action: TutorAction,
        immortal: bool = False,
    ) -> Tuple[QueryState, dict]:
        """Process one SELECT action.

        Timeline:
          1. Validate selection
          2. Check danger
          3. Apply tutor action (WARNING / WAIT)
          4. If WAIT + danger: DEATH (unless immortal)
          5. If safe: AUTO-PLACE
          6. Auto-refresh candidate pool

        Args:
            state: current query state
            selected_indices: ball indices chosen by learner
            tutor_action: tutor's response (WAIT or WARNING)
            immortal: if True, danger does not kill (baseline modes)

        Returns:
            (updated_state, step_info)
        """
        if state.is_terminal:
            return state, {'event': 'already_terminal'}

        selected = select_balls(state, selected_indices)
        has_danger = check_selection_has_danger(selected)

        step_info = {
            'event': 'select',
            'selected_indices': selected_indices,
            'selected_colors': [b.color for b in selected],
            'has_danger': has_danger,
            'tutor_action': tutor_action.action_type.name,
        }

        if has_danger:
            if tutor_action.action_type == TutorActionType.WARNING:
                # Warning: discard selection, learner learns from warning
                state = apply_warning(state, selected)
                step_info['event'] = 'warning'
            elif immortal:
                # Immortal baseline: log danger but don't die
                state.danger_select_count += 1
                step_info['event'] = 'danger_survived_immortal'
                # Still discard selection (same as warning for learning)
                state = apply_warning(state, selected)
            else:
                # Death: tutor chose WAIT but selection had danger
                state = apply_death(state)
                step_info['event'] = 'death'
                state.step_log.append(step_info)
                return state, step_info
        else:
            # Safe selection: auto-place
            state = auto_place(state, selected)
            step_info['event'] = 'placed'
            step_info['fill_ratio'] = state.fill_ratio

        # Auto-refresh candidate pool
        if not state.is_terminal:
            new_pool = generate_candidate_pool(
                grammar_colors=state.grammar_colors,
                target_output=state.target_output,
                n_candidates=self.cfg.env.n_candidates,
                danger_model=self.danger_model,
                cfg=self.cfg.env,
                rng=self.rng,
            )
            state = retry_refresh(state, new_pool)

        state.step_log.append(step_info)
        return state, step_info

    def step_confirm(
        self,
        state: QueryState,
        feedback_mode: Optional[str] = None,
    ) -> Tuple[QueryState, bool, dict]:
        """Process a CONFIRM action.

        Args:
            state: current query state
            feedback_mode: 'wrong_only' or 'wrong_positions' (override config)

        Returns:
            (state, success, feedback_info)
        """
        if state.is_terminal:
            return state, False, {'event': 'already_terminal'}

        success, feedback = confirm(state)
        mode = feedback_mode or self.cfg.learner.feedback_mode
        feedback['mode'] = mode

        step_info = {
            'event': 'confirm',
            'success': success,
            'confirm_count': state.confirm_count,
            'outcome': state.outcome.name,
            'submitted': feedback.get('submitted'),
            'mask': feedback.get('mask'),
        }
        state.step_log.append(step_info)

        return state, success, feedback

    def to_query_result(self, state: QueryState) -> QueryResult:
        """Convert final query state to a QueryResult."""
        return QueryResult(
            query_id=state.query_id,
            query_words=state.query_words,
            target_output=state.target_output,
            ground_truth=state.ground_truth,
            outcome=state.outcome,
            confirm_count=state.confirm_count,
            retry_count=state.retry_count,
            death_count=1 if state.outcome == Outcome.DEATH else 0,
            danger_select_count=state.danger_select_count,
            stuck_retry_events=state.stuck_retry_events,
            final_completion=list(state.completion),
            steps=list(state.step_log),
        )
