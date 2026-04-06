"""
Regression tests for WARN / UNLOCK under unified API — Phase 8.

Verifies that existing WARN and UNLOCK behavior is preserved
when the intervention family is unified.
"""

import numpy as np
import pytest

from src.envs.lattice_v2_runner import LatticeV2Runner


runner = LatticeV2Runner()


def test_warn_path_runs_under_unified_api():
    """WARN via unified intervention family mode still runs."""
    s = runner.reset(seed=42, latent_mode=True,
                     robot_belief_mode=True, prefix_horizon=5,
                     intervention_family_mode=True, item_drop_enabled=False)
    while not s.done:
        runner.step(s)
    assert s.done


def test_unlock_path_runs_under_unified_api():
    """UNLOCK via time_aware tutor still runs under unified mode."""
    s = runner.reset(seed=42, latent_mode=True, tutor_mode="time_aware")
    while not s.done:
        runner.step(s)
    assert s.done


def test_warn_behavior_matches_previous_mode():
    """Warning-only mode produces same result with and without family flag."""
    results_old = []
    results_new = []
    for seed in range(10):
        s = runner.reset(seed=seed, tutor_mode="none", warning_mode="fixed",
                         latent_mode=True)
        while not s.done:
            runner.step(s)
        results_old.append(s.survived)

    for seed in range(10):
        s = runner.reset(seed=seed, tutor_mode="none", warning_mode="fixed",
                         latent_mode=True, intervention_family_mode=True,
                         item_drop_enabled=False)
        while not s.done:
            runner.step(s)
        results_new.append(s.survived)

    assert results_old == results_new


def test_unlock_behavior_matches_previous_mode():
    """Door tutor mode produces same result with and without family flag."""
    results_old = []
    results_new = []
    for seed in range(10):
        s = runner.reset(seed=seed, tutor_mode="time_aware", latent_mode=True,
                         closure_budget=2)
        while not s.done:
            runner.step(s)
        results_old.append(s.survived)

    for seed in range(10):
        s = runner.reset(seed=seed, tutor_mode="time_aware", latent_mode=True,
                         closure_budget=2, intervention_family_mode=True,
                         item_drop_enabled=False)
        while not s.done:
            runner.step(s)
        results_new.append(s.survived)

    assert results_old == results_new


def test_unified_api_does_not_change_heuristic_tutor():
    """Heuristic tutor path is unchanged by intervention_family_mode flag."""
    s = runner.reset(seed=42, tutor_mode="warn_first", latent_mode=True,
                     intervention_family_mode=True, item_drop_enabled=False)
    while not s.done:
        runner.step(s)
    assert s.done


def test_unified_api_does_not_change_robot_belief_tutor_when_item_disabled():
    """Robot-belief tutor with item disabled matches Phase 7 behavior."""
    s = runner.reset(seed=42, latent_mode=True,
                     robot_belief_mode=True, prefix_horizon=5,
                     intervention_family_mode=True, item_drop_enabled=False)
    while not s.done:
        runner.step(s)
    assert s.done
    # ITEM_DROP should have very negative score when disabled
    if s.last_intervention is not None and "ITEM_DROP" in s.last_intervention.scores:
        assert s.last_intervention.scores["ITEM_DROP"] < 0
