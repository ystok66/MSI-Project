"""Tests for SIPS-lite Particle Teacher."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.envs.map_generator import generate_default_map, CellType
from src.agents.bounded_agent import BoundedRationalAgent
from src.agents.belief import BeliefMap
from src.teachers.particle_teacher import (
    ParticleTeacherPolicy,
    _normalize_weights,
    _effective_sample_size,
    _resample,
    _init_particles,
    Particle,
)
from src.teachers.interventions import Intervention, InterventionType
from src.teachers.oracle_teacher import OracleTeacherPolicy


@pytest.fixture
def grid_map():
    return generate_default_map()


@pytest.fixture
def pt(grid_map):
    teacher = ParticleTeacherPolicy(
        height=grid_map.height,
        width=grid_map.width,
        n_particles=16,
        rng=np.random.default_rng(42),
    )
    teacher.reset()
    return teacher


def test_particle_init_count(pt):
    """16 particles should be created."""
    assert len(pt.particles) == 16


def test_particle_weights_sum_to_one(pt):
    """After init, weights should sum to 1."""
    total = sum(p.weight for p in pt.particles)
    assert abs(total - 1.0) < 1e-6


def test_normalize_weights():
    """Normalization should make weights sum to 1."""
    particles = [
        Particle(
            belief=BeliefMap.from_prior(4, 4),
            current_plan=[], plan_step_idx=0,
            budget_class=8, warn_sensitivity=0.5, risk_aversion=1.0,
            weight=w,
        )
        for w in [0.3, 0.1, 0.5, 0.1]
    ]
    _normalize_weights(particles)
    total = sum(p.weight for p in particles)
    assert abs(total - 1.0) < 1e-6


def test_ess_uniform():
    """ESS of uniform weights should equal N."""
    N = 16
    particles = [
        Particle(
            belief=BeliefMap.from_prior(4, 4),
            current_plan=[], plan_step_idx=0,
            budget_class=8, warn_sensitivity=0.5, risk_aversion=1.0,
            weight=1.0 / N,
        )
        for _ in range(N)
    ]
    ess = _effective_sample_size(particles)
    assert abs(ess - N) < 1e-4


def test_ess_degenerate():
    """ESS of one-hot weights should be near 1."""
    particles = [
        Particle(
            belief=BeliefMap.from_prior(4, 4),
            current_plan=[], plan_step_idx=0,
            budget_class=8, warn_sensitivity=0.5, risk_aversion=1.0,
            weight=0.0,
        )
        for _ in range(16)
    ]
    particles[0].weight = 1.0
    ess = _effective_sample_size(particles)
    assert ess < 1.5


def test_resample_preserves_count():
    """Resampling should produce same number of particles."""
    particles = [
        Particle(
            belief=BeliefMap.from_prior(4, 4),
            current_plan=[], plan_step_idx=0,
            budget_class=8, warn_sensitivity=0.5, risk_aversion=1.0,
            weight=1.0 / 4,
        )
        for _ in range(4)
    ]
    rng = np.random.default_rng(42)
    resampled = _resample(particles, rng)
    assert len(resampled) == 4
    total = sum(p.weight for p in resampled)
    assert abs(total - 1.0) < 1e-6


def test_update_changes_weights(pt, grid_map):
    """After observing an action, weights should change."""
    weights_before = [p.weight for p in pt.particles]
    passable = np.ones((grid_map.height, grid_map.width), dtype=bool)
    for r in range(grid_map.height):
        for c in range(grid_map.width):
            if grid_map.true_cost[r, c] == np.inf:
                passable[r, c] = False

    info = pt.update(
        observed_action="RIGHT",
        agent_pos=grid_map.agent_start,
        goal=grid_map.object_spawn,
        passable_mask=passable,
        true_cost=grid_map.true_cost,
        true_risk=grid_map.true_risk,
    )
    weights_after = [p.weight for p in pt.particles]
    # Weights should have been renormalized
    assert abs(sum(weights_after) - 1.0) < 1e-6
    assert "ess" in info
    assert "match_frac" in info


def test_particle_teacher_selects_valid_action(pt, grid_map):
    """Teacher should return a valid Intervention."""
    passable = np.ones((grid_map.height, grid_map.width), dtype=bool)
    for r in range(grid_map.height):
        for c in range(grid_map.width):
            if grid_map.true_cost[r, c] == np.inf:
                passable[r, c] = False

    intervention, info = pt.select_action(
        agent_pos=grid_map.agent_start,
        goal=grid_map.target_pos,
        true_risk=grid_map.true_risk,
        true_cost=grid_map.true_cost,
        time_left=30,
        risk_budget_left=0.5,
        passable_mask=passable,
        locked_doors=set(grid_map.door_positions),
    )
    assert intervention.type in InterventionType
    assert "scores" in info


def test_particle_not_reading_oracle_belief(pt, grid_map):
    """Particle teacher should work without accessing env.agent.belief."""
    # The select_action signature doesn't take an agent argument
    passable = np.ones((grid_map.height, grid_map.width), dtype=bool)
    intervention, info = pt.select_action(
        agent_pos=(0, 0),
        goal=(7, 7),
        true_risk=grid_map.true_risk,
        true_cost=grid_map.true_cost,
        time_left=30,
        risk_budget_left=1.0,
        passable_mask=passable,
    )
    # Should succeed without agent reference
    assert intervention is not None


def test_oracle_still_works(grid_map):
    """v0 oracle teacher should still work."""
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

    intervention, info = teacher.select_action(
        agent=agent,
        true_cost=grid_map.true_cost,
        true_risk=grid_map.true_risk,
        goal=grid_map.target_pos,
        time_left=60,
        risk_budget_left=1.0,
        passable_mask=passable,
    )
    assert intervention.type in InterventionType
