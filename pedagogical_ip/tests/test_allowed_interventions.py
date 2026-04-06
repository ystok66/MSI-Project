"""
Tests for allowed_interventions enforcement — cleanup round 1.

Verifies that teacher conditions actually restrict which actions the robot
can select. Without enforcement, experiment matrix results are invalid.
"""

import pytest
from src.envs.lattice_v2_runner import LatticeV2Runner


runner = LatticeV2Runner()


def _run_episode_with_constraint(allowed, seed=42):
    """Run one full episode with a specific allowed_interventions set."""
    s = runner.reset(
        seed=seed,
        latent_mode=True,
        robot_belief_mode=True,
        intervention_family_mode=True,
        item_drop_enabled=True,
        allowed_interventions=frozenset(allowed),
    )
    actions_chosen = []
    while not s.done:
        runner.step(s)
        if s.last_intervention is not None:
            actions_chosen.append(s.last_intervention.action)
    return actions_chosen


def test_warning_only_never_selects_unlock():
    """warning_only condition: only WAIT and WARN allowed."""
    actions = _run_episode_with_constraint({"WAIT", "WARN"})
    for a in actions:
        assert a in {"WAIT", "WARN"}, f"warning_only selected {a}"


def test_unlock_only_never_selects_warn():
    """unlock_only condition: only WAIT and UNLOCK allowed."""
    actions = _run_episode_with_constraint({"WAIT", "UNLOCK"})
    for a in actions:
        assert a in {"WAIT", "UNLOCK"}, f"unlock_only selected {a}"


def test_item_only_never_selects_warn_or_unlock():
    """item_only condition: only WAIT and ITEM_DROP allowed."""
    actions = _run_episode_with_constraint({"WAIT", "ITEM_DROP"})
    for a in actions:
        assert a in {"WAIT", "ITEM_DROP"}, f"item_only selected {a}"


def test_no_tutor_always_waits():
    """no_tutor condition: only WAIT allowed."""
    actions = _run_episode_with_constraint({"WAIT"})
    for a in actions:
        assert a == "WAIT", f"no_tutor selected {a}"


def test_full_intervention_allows_all():
    """robot_belief condition: all actions allowed."""
    actions = _run_episode_with_constraint({"WAIT", "WARN", "UNLOCK", "ITEM_DROP"})
    # Just verify it runs — any action is valid
    assert isinstance(actions, list)


def test_none_allowed_means_all_allowed():
    """allowed_interventions=None means no restriction (legacy compat)."""
    s = runner.reset(
        seed=42,
        latent_mode=True,
        robot_belief_mode=True,
        intervention_family_mode=True,
        item_drop_enabled=True,
        allowed_interventions=None,
    )
    while not s.done:
        runner.step(s)
    # Should complete without error; any action could have been chosen
    assert s.done


def test_wait_always_available_even_if_not_specified():
    """WAIT is always included as fallback, even if not in the set."""
    actions = _run_episode_with_constraint({"WARN"})  # only WARN explicitly
    for a in actions:
        assert a in {"WAIT", "WARN"}, f"selected {a} when only WARN allowed"


def test_enforcement_consistent_across_seeds():
    """warning_only is enforced regardless of seed."""
    for seed in [1, 10, 42, 77, 123]:
        actions = _run_episode_with_constraint({"WAIT", "WARN"}, seed=seed)
        for a in actions:
            assert a in {"WAIT", "WARN"}, f"seed={seed}: {a}"


def test_allowed_interventions_stored_on_state():
    """V2EpisodeState stores allowed_interventions."""
    s = runner.reset(
        seed=42,
        latent_mode=True,
        robot_belief_mode=True,
        allowed_interventions=frozenset({"WAIT", "WARN"}),
    )
    assert s.allowed_interventions == frozenset({"WAIT", "WARN"})


def test_score_interventions_filters_scores():
    """score_interventions only returns allowed actions in scores dict."""
    from src.teachers.intervention_policy import score_interventions
    from src.teachers.robot_belief import init_robot_belief

    s = runner.reset(
        seed=42,
        latent_mode=True,
        robot_belief_mode=True,
        intervention_family_mode=True,
        item_drop_enabled=True,
    )
    # Run a step to get observation
    runner.observe(s)

    decision = score_interventions(
        s.robot_belief, s.agent_pos, s.goal,
        s.belief_cost, s.passable, s.meta,
        warned_segments=s.warned_segments,
        prefix_horizon=5,
        t=s.t, t_max=s.t_max,
        inventory_state=s.inventory,
        allowed_actions=frozenset({"WAIT", "WARN"}),
    )
    assert set(decision.scores.keys()).issubset({"WAIT", "WARN"})
    assert decision.action in {"WAIT", "WARN"}
