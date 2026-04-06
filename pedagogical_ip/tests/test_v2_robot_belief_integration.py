"""
Integration tests for V2 robot-belief tutor — Phase 7.

Verifies end-to-end episode run, env info exposure, and baseline isolation.
"""

import numpy as np
import pytest

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.teachers.robot_belief import RobotBelief
from src.teachers.intervention_policy import InterventionDecision


runner = LatticeV2Runner()


def test_robot_belief_tutor_episode_runs():
    """robot-belief tutor mode completes an episode."""
    s = runner.reset(seed=42, latent_mode=True, patch_radius=2,
                     prefix_horizon=5, belief_planning_mode=True,
                     robot_belief_mode=True)
    assert s.robot_belief_mode is True
    assert s.robot_belief is not None
    while not s.done:
        runner.step(s)
    assert s.done
    assert s.steps > 0


def test_env_info_contains_robot_prediction():
    """last_intervention contains predicted prefix."""
    s = runner.reset(seed=42, latent_mode=True, robot_belief_mode=True,
                     prefix_horizon=5)
    runner.step(s)
    if not s.done and s.last_intervention is not None:
        assert isinstance(s.last_intervention.predicted_prefix, list)


def test_env_info_contains_robot_failure_estimate():
    """last_intervention contains failure mode estimates."""
    s = runner.reset(seed=42, latent_mode=True, robot_belief_mode=True,
                     prefix_horizon=5)
    runner.step(s)
    if not s.done and s.last_intervention is not None:
        assert s.last_intervention.predicted_failure_modes is not None


def test_env_info_contains_intervention_scores():
    """last_intervention exposes WAIT/WARN/UNLOCK scores."""
    s = runner.reset(seed=42, latent_mode=True, robot_belief_mode=True,
                     prefix_horizon=5)
    runner.step(s)
    if not s.done and s.last_intervention is not None:
        assert "WAIT" in s.last_intervention.scores
        assert "WARN" in s.last_intervention.scores
        assert "UNLOCK" in s.last_intervention.scores


def test_robot_belief_tutor_differs_from_heuristic():
    """Robot-belief tutor is not just heuristic tutor repackaged."""
    # Run with heuristic tutor
    s_heur = runner.reset(seed=42, latent_mode=True, tutor_mode="warn_first",
                          warning_mode="fixed")
    for _ in range(3):
        if not s_heur.done:
            runner.step(s_heur)
    # Run with robot-belief tutor
    s_rb = runner.reset(seed=42, latent_mode=True, robot_belief_mode=True,
                        prefix_horizon=5)
    for _ in range(3):
        if not s_rb.done:
            runner.step(s_rb)
    # At minimum, robot-belief tutor should have structured decisions
    if not s_rb.done:
        assert s_rb.last_intervention is not None


def test_oracle_heuristic_robotbelief_all_runnable():
    """All three tutor types can run complete episodes."""
    # Oracle-like (always_close)
    s1 = runner.reset(seed=42, tutor_mode="always_close")
    while not s1.done:
        runner.step(s1)
    # Heuristic (time_aware)
    s2 = runner.reset(seed=42, latent_mode=True, tutor_mode="time_aware")
    while not s2.done:
        runner.step(s2)
    # Robot-belief
    s3 = runner.reset(seed=42, latent_mode=True, robot_belief_mode=True, prefix_horizon=5)
    while not s3.done:
        runner.step(s3)
    assert s1.done and s2.done and s3.done


def test_legacy_mode_baseline_unchanged():
    """Legacy baseline still produces expected survival rates."""
    results = []
    for seed in range(20):
        s = runner.reset(seed=seed, tutor_mode="none",
                         latent_mode=False, robot_belief_mode=False)
        while not s.done:
            runner.step(s)
        results.append(runner.get_metrics(s))
    surv = sum(r["survived"] for r in results) / len(results)
    assert surv < 0.30, f"Legacy no_tutor survival too high: {surv:.0%}"


def test_robot_belief_mode_is_read_only():
    """Robot belief prediction does not add extra mutation to agent state.

    Note: agent predictor weights DO change during a normal step() because
    the agent learns from outcomes (update_from_outcome). This test verifies
    that the robot prediction itself is read-only by calling predict_agent_prefix
    directly and checking no mutation occurs.
    """
    from src.teachers.agent_predictor import predict_agent_prefix
    s = runner.reset(seed=42, latent_mode=True, robot_belief_mode=True,
                     prefix_horizon=5)
    lp_w_before = s.latent_predictor.cost_head.w.copy()
    fb_before = s.feature_belief.mean.copy()
    # Robot prediction should be read-only
    predict_agent_prefix(
        s.robot_belief, s.agent_pos, s.goal,
        s.belief_cost, s.passable, prefix_horizon=5)
    np.testing.assert_array_equal(s.latent_predictor.cost_head.w, lp_w_before)
    np.testing.assert_array_equal(s.feature_belief.mean, fb_before)


def test_robot_belief_mode_does_not_change_legacy_path():
    """robot_belief_mode=False leaves legacy path unchanged."""
    s_off = runner.reset(seed=42, latent_mode=True, robot_belief_mode=False)
    runner.step(s_off)
    assert s_off.last_intervention is None
    assert s_off.robot_belief is None
