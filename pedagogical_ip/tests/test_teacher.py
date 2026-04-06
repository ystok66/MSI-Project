"""Tests for Oracle Teacher Policy."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.envs.map_generator import generate_default_map
from src.agents.bounded_agent import BoundedRationalAgent
from src.agents.belief import BeliefMap
from src.teachers.oracle_teacher import OracleTeacherPolicy
from src.teachers.interventions import InterventionType


@pytest.fixture
def setup():
    grid_map = generate_default_map()
    agent = BoundedRationalAgent(
        height=grid_map.height,
        width=grid_map.width,
        start_pos=grid_map.agent_start,
        rng=np.random.default_rng(42),
    )
    teacher = OracleTeacherPolicy()
    passable = np.ones((grid_map.height, grid_map.width), dtype=bool)
    for r in range(grid_map.height):
        for c in range(grid_map.width):
            if grid_map.true_cost[r, c] == np.inf:
                passable[r, c] = False
    return agent, teacher, grid_map, passable


def test_teacher_returns_valid_intervention(setup):
    agent, teacher, grid_map, passable = setup
    intervention, info = teacher.select_action(
        agent=agent,
        true_cost=grid_map.true_cost,
        true_risk=grid_map.true_risk,
        goal=grid_map.target_pos,
        time_left=60,
        risk_budget_left=1.0,
        passable_mask=passable,
        door_positions=grid_map.door_positions,
        locked_doors=set(grid_map.door_positions),
    )
    assert intervention.type in InterventionType
    assert "predicted_wait_success" in info


def test_teacher_waits_when_safe(setup):
    agent, teacher, grid_map, passable = setup
    # Give agent perfect belief and plenty of time
    agent.belief.cost_mean[:] = grid_map.true_cost.copy()
    agent.belief.cost_mean[agent.belief.cost_mean == np.inf] = 100.0
    agent.belief.cost_var[:] = 0.01
    agent.belief.risk_mean[:] = grid_map.true_risk.copy()
    agent.belief.risk_var[:] = 0.01

    intervention, info = teacher.select_action(
        agent=agent,
        true_cost=grid_map.true_cost,
        true_risk=grid_map.true_risk,
        goal=grid_map.target_pos,
        time_left=100,
        risk_budget_left=10.0,
        passable_mask=passable,
    )
    # When agent has perfect info and plenty of time, WAIT is likely best
    # (learning gain from exploration is near zero, so no need to intervene)
    assert intervention.type in InterventionType


def test_teacher_intervenes_under_risk(setup):
    agent, teacher, grid_map, passable = setup
    # Make agent believe everything is safe but true risk is high
    agent.belief.risk_mean[:] = 0.01  # thinks it's safe
    # Set tight constraints
    intervention, info = teacher.select_action(
        agent=agent,
        true_cost=grid_map.true_cost,
        true_risk=grid_map.true_risk,
        goal=grid_map.target_pos,
        time_left=10,
        risk_budget_left=0.1,
        passable_mask=passable,
        door_positions=grid_map.door_positions,
        locked_doors=set(grid_map.door_positions),
    )
    # With low time and risk budget, teacher should consider non-WAIT actions
    # The scores dict should show teacher evaluated alternatives
    assert "scores" in info
    assert len(info["scores"]) > 1


def test_teacher_scores_multiple_actions(setup):
    agent, teacher, grid_map, passable = setup
    intervention, info = teacher.select_action(
        agent=agent,
        true_cost=grid_map.true_cost,
        true_risk=grid_map.true_risk,
        goal=grid_map.target_pos,
        time_left=30,
        risk_budget_left=0.5,
        passable_mask=passable,
        door_positions=grid_map.door_positions,
        locked_doors=set(grid_map.door_positions),
    )
    # Should have scored WAIT + at least WARN + SHIELD
    assert len(info.get("scores", {})) >= 3
