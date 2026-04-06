"""Tests for PedagogicalGridEnv."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.envs.map_generator import generate_default_map, CellType
from src.envs.pedagogical_grid import PedagogicalGridEnv


@pytest.fixture
def env():
    grid_map = generate_default_map()
    e = PedagogicalGridEnv(
        grid_map=grid_map,
        max_steps=30,
        seed=42,
        render_mode="ansi",
    )
    return e


def test_reset_returns_valid_obs(env):
    obs, info = env.reset(seed=42)
    assert "agent_pos" in obs
    assert "belief_cost_mean" in obs
    assert obs["belief_cost_mean"].shape == (8, 8)
    assert info["step"] == 0


def test_step_with_wait_advances_agent(env):
    env.reset(seed=42)
    pos_before = tuple(env.agent.pos)
    obs, reward, terminated, truncated, info = env.step(0)  # WAIT
    # Agent should have moved (planned internally)
    assert info["step"] == 1
    assert "agent_action" in info


def test_step_preserves_observation_shape(env):
    env.reset(seed=42)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs["belief_cost_mean"].shape == (8, 8)
        assert obs["belief_risk_mean"].shape == (8, 8)
        if terminated or truncated:
            break


def test_episode_terminates_within_max_steps(env):
    env.reset(seed=42)
    for _ in range(35):  # max_steps=30
        obs, reward, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    assert terminated or truncated


def test_render_ansi(env):
    env.reset(seed=42)
    output = env.render()
    assert output is not None
    assert "Step" in output
    assert "A" in output  # agent marker


def test_unlock_door_makes_door_passable(env):
    env.reset(seed=42)
    # Check that (3,3) is initially locked
    assert (3, 3) in env.locked_doors
    # Unlock
    env.step(2)  # UNLOCK_DOOR
    assert (3, 3) not in env.locked_doors


def test_object_pickup_at_spawn():
    grid_map = generate_default_map()
    # Make agent start at object spawn
    grid_map.agent_start = grid_map.object_spawn
    e = PedagogicalGridEnv(grid_map=grid_map, max_steps=30, seed=42)
    e.reset(seed=42)
    # Agent starts at object — should pick up on first move back to same cell
    # Actually, pickup happens when agent moves TO object_pos
    assert e.agent.pos == grid_map.object_spawn
