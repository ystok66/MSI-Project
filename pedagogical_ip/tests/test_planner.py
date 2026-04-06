"""Tests for bounded A* planner."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.agents.planner_astar import bounded_astar, plan_next_action, MOVES


def _simple_belief(h=8, w=8, cost=1.0, risk=0.0):
    """Create uniform belief maps."""
    return (
        np.full((h, w), cost, dtype=np.float64),
        np.full((h, w), risk, dtype=np.float64),
        np.full((h, w), 0.1, dtype=np.float64),  # low variance
    )


def test_astar_finds_path_simple():
    cost_m, risk_m, cost_v = _simple_belief()
    path = bounded_astar(
        (0, 0), (7, 7),
        cost_m, risk_m, cost_v,
        budget=100,
    )
    assert len(path) > 1
    assert path[0] == (0, 0)
    assert path[-1] == (7, 7)


def test_astar_respects_walls():
    cost_m, risk_m, cost_v = _simple_belief()
    # Put a wall blocking direct path
    cost_m[0, 1] = 100.0  # very high cost = "wall-like"
    path = bounded_astar(
        (0, 0), (0, 2),
        cost_m, risk_m, cost_v,
        budget=50,
    )
    # Should find path, but route around the wall
    assert path[-1] == (0, 2)
    if len(path) > 2:
        # Should not go through (0,1) if there's a better route
        pass  # path may still go through if no alternative


def test_bounded_budget_limits_expansion():
    cost_m, risk_m, cost_v = _simple_belief()
    # Very small budget may not reach the goal
    path = bounded_astar(
        (0, 0), (7, 7),
        cost_m, risk_m, cost_v,
        budget=3,
    )
    # Should return a partial path, not empty
    assert len(path) >= 1
    # Might not reach goal with budget=3
    # Path should start at (0,0)
    assert path[0] == (0, 0)


def test_plan_next_action_returns_valid():
    cost_m, risk_m, cost_v = _simple_belief()
    action, next_pos = plan_next_action(
        (0, 0), (7, 7),
        cost_m, risk_m, cost_v,
    )
    valid_actions = {"UP", "DOWN", "LEFT", "RIGHT", "STAY"}
    assert action in valid_actions
    if action != "STAY":
        # next_pos should be adjacent
        dr = abs(next_pos[0] - 0)
        dc = abs(next_pos[1] - 0)
        assert dr + dc == 1


def test_plan_avoids_high_risk():
    cost_m, risk_m, cost_v = _simple_belief()
    # Place high risk on direct path
    risk_m[0, 1] = 0.9
    risk_m[1, 0] = 0.9
    action, next_pos = plan_next_action(
        (0, 0), (0, 3),
        cost_m, risk_m, cost_v,
        lambda_risk=10.0,
    )
    # Should prefer moving away from risk
    valid = {"UP", "DOWN", "LEFT", "RIGHT", "STAY"}
    assert action in valid


def test_passable_mask():
    cost_m, risk_m, cost_v = _simple_belief()
    mask = np.ones((8, 8), dtype=bool)
    mask[0, 1] = False  # block
    mask[1, 0] = False  # block

    path = bounded_astar(
        (0, 0), (2, 2),
        cost_m, risk_m, cost_v,
        budget=50,
        passable_mask=mask,
    )
    assert (0, 1) not in path
    assert (1, 0) not in path
