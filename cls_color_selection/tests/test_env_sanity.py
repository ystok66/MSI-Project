"""
test_env_sanity.py — Sanity tests for environment lifecycle.

Verifies:
  - All 3 terminal states reachable (SUCCESS, DEATH, TIMEOUT)
  - State transitions are correct
  - Candidate pool generation works
"""
import sys
import os
import numpy as np
import pytest

# Ensure imports work
sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..')))

from cls_color_selection.config import FullConfig, EnvConfig, LearnerConfig
from cls_color_selection.constants import Outcome, TutorActionType
from cls_color_selection.interfaces import TutorAction, Example
from cls_color_selection.environment.grammar_task_env import GrammarTaskEnv
from cls_color_selection.environment.state import QueryState
from cls_color_selection.environment.transition import (
    auto_place, confirm, select_balls, check_selection_has_danger,
)
from cls_color_selection.environment.generator import (
    generate_danger_model, generate_candidate_pool, DangerModel,
)


def _make_simple_cfg():
    cfg = FullConfig()
    cfg.env.n_candidates = 6
    cfg.env.n_confirm_max = 3
    cfg.env.danger_ratio = 0.3
    cfg.env.danger_dim = 5
    cfg.env.n_danger_types = 2
    cfg.env.n_safe_types = 1
    return cfg


def _make_simple_state(target=None, grammar_colors=None, pool=None):
    """Helper to create a minimal QueryState."""
    if target is None:
        target = ['RED', 'BLUE', 'RED']
    if grammar_colors is None:
        grammar_colors = ['RED', 'BLUE', 'GREEN']
    if pool is None:
        pool = []

    return QueryState(
        query_id=0,
        query_words=['dax', 'lug'],
        target_output=list(target),
        ground_truth=list(target),
        grammar_colors=grammar_colors,
        completion=[None] * len(target),
        candidate_pool=pool,
        n_confirm_max=3,
    )


# ── Test: QueryState basic properties ──────────────────────────

class TestQueryState:
    def test_color_gaps(self):
        state = _make_simple_state(target=['RED', 'BLUE', 'RED'])
        gaps = state.color_gaps()
        assert gaps == {'RED': 2, 'BLUE': 1}

    def test_fill_ratio_empty(self):
        state = _make_simple_state()
        assert state.fill_ratio == 0.0
        assert not state.is_complete

    def test_fill_ratio_partial(self):
        state = _make_simple_state(target=['RED', 'BLUE', 'RED'])
        state.completion[0] = 'RED'
        assert abs(state.fill_ratio - 1.0 / 3.0) < 1e-6

    def test_fill_ratio_complete(self):
        state = _make_simple_state(target=['RED', 'BLUE'])
        state.completion = ['RED', 'BLUE']
        assert state.is_complete
        assert state.fill_ratio == 1.0

    def test_needed_colors(self):
        state = _make_simple_state(target=['RED', 'BLUE', 'RED'])
        state.completion[0] = 'RED'
        needed = state.needed_colors()
        assert 'RED' in needed  # still need one more
        assert 'BLUE' in needed


# ── Test: Auto-place ───────────────────────────────────────────

class TestAutoPlace:
    def test_place_single(self):
        from cls_color_selection.interfaces import CandidateBall
        state = _make_simple_state(target=['RED', 'BLUE', 'RED'])
        balls = [CandidateBall(
            index=0, color='RED',
            danger_vec=np.zeros(5), observed_vec=np.zeros(5),
            is_danger=False, danger_type=0)]
        state = auto_place(state, balls)
        assert state.completion[0] == 'RED'
        assert state.completion[1] is None
        assert state.completion[2] is None

    def test_place_multiple(self):
        from cls_color_selection.interfaces import CandidateBall
        state = _make_simple_state(target=['RED', 'BLUE', 'RED'])
        balls = [
            CandidateBall(0, 'RED', np.zeros(5), np.zeros(5), False, 0),
            CandidateBall(1, 'BLUE', np.zeros(5), np.zeros(5), False, 0),
            CandidateBall(2, 'RED', np.zeros(5), np.zeros(5), False, 0),
        ]
        state = auto_place(state, balls)
        assert state.completion == ['RED', 'BLUE', 'RED']
        assert state.is_complete

    def test_place_waste_ignored(self):
        from cls_color_selection.interfaces import CandidateBall
        state = _make_simple_state(target=['RED', 'BLUE'])
        balls = [
            CandidateBall(0, 'GREEN', np.zeros(5), np.zeros(5), False, 0),
        ]
        state = auto_place(state, balls)
        assert state.completion == [None, None]  # GREEN not needed


# ── Test: Confirm ──────────────────────────────────────────────

class TestConfirm:
    def test_confirm_success(self):
        state = _make_simple_state(target=['RED', 'BLUE'])
        state.completion = ['RED', 'BLUE']
        success, feedback = confirm(state)
        assert success
        assert state.outcome == Outcome.SUCCESS

    def test_confirm_wrong(self):
        state = _make_simple_state(target=['RED', 'BLUE'])
        state.completion = ['RED', 'GREEN']
        success, feedback = confirm(state)
        assert not success
        assert feedback['mask'] == [True, False]

    def test_confirm_timeout(self):
        state = _make_simple_state(target=['RED', 'BLUE'])
        state.completion = ['RED', 'GREEN']
        state.n_confirm_max = 1
        success, feedback = confirm(state)
        assert not success
        assert state.outcome == Outcome.TIMEOUT


# ── Test: Candidate pool generation ────────────────────────────

class TestCandidatePool:
    def test_pool_size(self):
        cfg = _make_simple_cfg()
        rng = np.random.default_rng(42)
        model = generate_danger_model(cfg.env, rng)
        pool = generate_candidate_pool(
            grammar_colors=['RED', 'BLUE', 'GREEN'],
            target_output=['RED', 'BLUE'],
            n_candidates=8,
            danger_model=model,
            cfg=cfg.env,
            rng=rng,
        )
        assert len(pool) == 8

    def test_pool_colors_from_grammar(self):
        cfg = _make_simple_cfg()
        rng = np.random.default_rng(42)
        model = generate_danger_model(cfg.env, rng)
        palette = ['RED', 'BLUE']
        pool = generate_candidate_pool(
            grammar_colors=palette,
            target_output=['RED'],
            n_candidates=50,
            danger_model=model,
            cfg=cfg.env,
            rng=rng,
        )
        for ball in pool:
            assert ball.color in palette, f"Color {ball.color} not in palette"

    def test_pool_has_danger(self):
        cfg = _make_simple_cfg()
        cfg.env.danger_ratio = 0.5
        rng = np.random.default_rng(42)
        model = generate_danger_model(cfg.env, rng)
        pool = generate_candidate_pool(
            grammar_colors=['RED', 'BLUE'],
            target_output=['RED'],
            n_candidates=100,
            danger_model=model,
            cfg=cfg.env,
            rng=rng,
        )
        n_danger = sum(1 for b in pool if b.is_danger)
        # With ratio 0.5, expect ~50 danger balls (allow wide margin)
        assert 20 < n_danger < 80


# ── Test: Danger model ─────────────────────────────────────────

class TestDangerModel:
    def test_prototype_shape(self):
        cfg = _make_simple_cfg()
        rng = np.random.default_rng(42)
        model = generate_danger_model(cfg.env, rng)
        assert model.prototypes.shape == (3, 5)  # 1 safe + 2 danger, dim 5
        assert model.is_danger_type(0) == False
        assert model.is_danger_type(1) == True


# ── Test: Environment integration ──────────────────────────────

class TestEnvironment:
    def test_death_reachable(self):
        """Death should occur when tutor WAITs and selection has danger."""
        from cls_color_selection.interfaces import CandidateBall

        cfg = _make_simple_cfg()
        cfg.env.danger_ratio = 1.0  # all danger
        rng = np.random.default_rng(42)
        env = GrammarTaskEnv(cfg, rng)

        # Manual setup
        task_path = os.path.join(
            os.path.dirname(__file__), '..', 'cls_color_selection',
            'environment', '_test_fixture.txt')
        # Create a minimal fixture inline
        env.grammar = type('G', (), {'colors': ['RED', 'BLUE'], 'nouns': {}, 'rules': []})()
        env.danger_model = generate_danger_model(cfg.env, rng)

        query = Example(words=['dax'], output=['RED'])
        state = env.init_query(query, query_id=0, target_output=['RED'])

        # Force all balls to be danger
        for b in state.candidate_pool:
            b.is_danger = True
            b.danger_type = 1

        tutor_wait = TutorAction(action_type=TutorActionType.WAIT)
        state, info = env.step_select(state, [0], tutor_wait)
        assert state.outcome == Outcome.DEATH

    def test_success_reachable(self):
        """Success when correct balls placed and confirmed."""
        from cls_color_selection.interfaces import CandidateBall

        cfg = _make_simple_cfg()
        cfg.env.danger_ratio = 0.0
        rng = np.random.default_rng(42)
        env = GrammarTaskEnv(cfg, rng)

        env.grammar = type('G', (), {'colors': ['RED'], 'nouns': {}, 'rules': []})()
        env.danger_model = generate_danger_model(cfg.env, rng)

        query = Example(words=['dax'], output=['RED'])
        state = env.init_query(query, query_id=0, target_output=['RED'])

        # Make first ball RED and safe
        state.candidate_pool[0].color = 'RED'
        state.candidate_pool[0].is_danger = False

        tutor_wait = TutorAction(action_type=TutorActionType.WAIT)
        state, info = env.step_select(state, [0], tutor_wait)
        assert state.completion[0] == 'RED'

        state, success, feedback = env.step_confirm(state)
        assert success
        assert state.outcome == Outcome.SUCCESS


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
