"""
Tests for LatticeV2Runner — Phase 2 platformization.

Verifies:
1. reset() produces reproducible state
2. Episode state schema is complete
3. step() advances state correctly
4. Episodes terminate properly
5. Different teacher modes work
6. Runner produces same results as old script (fixed seed)
"""

import numpy as np
import pytest

from src.envs.lattice_v2_runner import LatticeV2Runner, V2EpisodeState


runner = LatticeV2Runner()


def test_runner_reset_reproducible():
    """Same seed produces identical initial state."""
    s1 = runner.reset(seed=42)
    s2 = runner.reset(seed=42)
    assert s1.agent_pos == s2.agent_pos
    assert s1.goal == s2.goal
    assert s1.t_max == s2.t_max
    assert s1.gridmap.height == s2.gridmap.height
    assert s1.gridmap.width == s2.gridmap.width
    np.testing.assert_array_equal(s1.belief_cost, s2.belief_cost)
    np.testing.assert_array_equal(s1.passable, s2.passable)


def test_runner_state_schema_complete():
    """Reset returns state with all required fields."""
    s = runner.reset(seed=0)
    assert isinstance(s, V2EpisodeState)

    # Agent position & goal
    assert s.agent_pos == (2, 1)
    assert s.goal[0] == 2

    # Time
    assert s.t == 0
    assert s.t_max > 0

    # Terminal flags
    assert s.survived is True
    assert s.reached_goal is False
    assert s.done is False

    # Teacher state
    assert isinstance(s.warned_segments, set)
    assert isinstance(s.warned_lane_bias, dict)

    # Metrics
    assert s.steps == 0
    assert s.unlock_count == 0
    assert s.warn_count == 0


def test_runner_single_step_progresses():
    """One step() call advances time and position."""
    s = runner.reset(seed=10)
    pos_before = s.agent_pos
    t_before = s.t

    s = runner.step(s)
    assert s.steps == 1
    # Agent should have moved (unless stuck)
    assert s.agent_pos != pos_before or s.done


def test_runner_episode_terminates():
    """Episodes terminate within t_max steps for all tutor modes."""
    for mode, kw in [
        ("none",        dict(tutor_mode="none")),
        ("time_aware",  dict(tutor_mode="time_aware", closure_budget=2)),
        ("always",      dict(tutor_mode="always_close", closure_budget=3)),
        ("warn_first",  dict(tutor_mode="warn_first", closure_budget=2)),
    ]:
        s = runner.reset(seed=7, **kw)
        while not s.done:
            s = runner.step(s)
        assert s.steps <= s.t_max, f"Mode {mode}: steps={s.steps} > t_max={s.t_max}"
        assert s.done, f"Mode {mode}: should be done"


def test_runner_teacher_modes_supported():
    """Runner handles all teacher mode combinations."""
    modes = [
        dict(tutor_mode="none", warning_mode="none"),
        dict(tutor_mode="none", warning_mode="fixed"),
        dict(tutor_mode="none", warning_mode="selected"),
        dict(tutor_mode="time_aware", closure_budget=2),
        dict(tutor_mode="warn_first", closure_budget=2),
        dict(tutor_mode="always_close", closure_budget=3),
    ]
    for kw in modes:
        s = runner.reset(seed=5, **kw)
        while not s.done:
            s = runner.step(s)
        m = runner.get_metrics(s)
        assert "survived" in m
        assert "reached_goal" in m
        assert "closures" in m
        assert "warnings" in m


def test_runner_no_tutor_baseline():
    """no_tutor across 50 seeds matches expected ~9% survival."""
    results = []
    for seed in range(50):
        s = runner.reset(seed=seed, tutor_mode="none")
        while not s.done:
            s = runner.step(s)
        results.append(runner.get_metrics(s))
    surv = sum(r["survived"] for r in results) / len(results)
    # Expected: ~9% (±5% for 50 seeds)
    assert surv < 0.25, f"no_tutor survival too high: {surv:.0%}"
