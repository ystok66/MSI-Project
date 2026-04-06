"""Tests for v1b benchmark suite: map families + generator."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.envs.map_families import (
    FAMILY_NAMES, FAMILY_GENERATORS,
    generate_semantic_trap,
    generate_planning_trap,
    generate_exploration_useful,
    generate_mixed,
    FamilyConfig,
)
from src.envs.benchmark_generator import (
    generate_benchmark_map,
    generate_transfer_map,
    DIFFICULTIES,
)
from src.envs.map_generator import CellType
from src.envs.pedagogical_grid import PedagogicalGridEnv


# ── Deterministic generation ─────────────────────────────────────────

def test_deterministic_same_seed():
    """Same (family, seed, difficulty) → same map."""
    gm1, cfg1 = generate_benchmark_map("semantic_trap", 42, "medium")
    gm2, cfg2 = generate_benchmark_map("semantic_trap", 42, "medium")
    assert np.array_equal(gm1.true_cost, gm2.true_cost)
    assert np.array_equal(gm1.true_risk, gm2.true_risk)
    assert np.array_equal(gm1.cell_types, gm2.cell_types)
    assert cfg1.max_steps == cfg2.max_steps


def test_different_seeds_differ():
    """Different seeds → different layouts."""
    gm1, _ = generate_benchmark_map("semantic_trap", 42, "medium")
    gm2, _ = generate_benchmark_map("semantic_trap", 99, "medium")
    # At least one cell should differ (risky or wall placement varies)
    assert not np.array_equal(gm1.cell_types, gm2.cell_types) or \
           not np.array_equal(gm1.true_risk, gm2.true_risk)


def test_transfer_map_differs():
    """Transfer map (offset seed) should differ from interaction map."""
    gm1, _ = generate_benchmark_map("semantic_trap", 42, "medium")
    gm2, _ = generate_transfer_map("semantic_trap", 42, "medium")
    assert not np.array_equal(gm1.cell_types, gm2.cell_types) or \
           not np.array_equal(gm1.true_risk, gm2.true_risk)


# ── All families generate valid maps ────────────────────────────────

@pytest.mark.parametrize("family", FAMILY_NAMES)
@pytest.mark.parametrize("diff", DIFFICULTIES)
def test_family_generates_valid_map(family, diff):
    """Each family × difficulty produces a valid GridMap."""
    gm, cfg = generate_benchmark_map(family, 0, diff)
    expected_dims = {
        "door_lattice_sanity": (9, 17),
        "deceptive_fork": (6, 8),
    }
    if family in expected_dims:
        eh, ew = expected_dims[family]
        assert gm.height == eh
        assert gm.width == ew
    else:
        assert gm.height == 10
        assert gm.width == 10
    assert gm.agent_start is not None
    assert gm.target_pos is not None
    assert gm.object_spawn is not None
    assert cfg.max_steps > 0
    assert cfg.risk_budget > 0


# ── Structural property tests ────────────────────────────────────────

def test_semantic_trap_has_risky_region():
    """Family A should have risky cells in right-side corridor."""
    gm, _ = generate_semantic_trap(seed=0, difficulty="hard")
    # Risky cells should exist in cols 5-7
    right_risk = gm.true_risk[:, 5:8]
    assert right_risk.max() >= 0.3, "No risky cells in right corridor"


def test_planning_trap_has_door():
    """Family B should have at least one locked door."""
    gm, _ = generate_planning_trap(seed=0, difficulty="medium")
    assert len(gm.door_positions) >= 1, "No door in planning trap"
    # Door cell should have inf cost
    dr, dc = gm.door_positions[0]
    assert gm.true_cost[dr, dc] == np.inf


def test_exploration_useful_low_risk():
    """Family C should have low actual risk (safe exploration)."""
    gm, _ = generate_exploration_useful(seed=0, difficulty="medium")
    max_risk = gm.true_risk.max()
    assert max_risk <= 0.3, f"Risk too high for exploration family: {max_risk}"


def test_mixed_has_three_zones():
    """Family D should have risky cells (phase 2) and a door (phase 3)."""
    gm, _ = generate_mixed(seed=0, difficulty="medium")
    # Phase 2: risky corridor in rows 4-5
    mid_risk = gm.true_risk[4:6, :].sum()
    assert mid_risk > 0, "No risky corridor in mixed map phase 2"
    # Phase 3: door
    assert len(gm.door_positions) >= 1, "No door in mixed map phase 3"


# ── Difficulty scaling ───────────────────────────────────────────────

def test_difficulty_scales_budget():
    """Harder difficulty should give tighter constraints."""
    _, cfg_easy = generate_semantic_trap(seed=0, difficulty="easy")
    _, cfg_hard = generate_semantic_trap(seed=0, difficulty="hard")
    assert cfg_easy.max_steps >= cfg_hard.max_steps
    assert cfg_easy.risk_budget >= cfg_hard.risk_budget


def test_difficulty_scales_risk():
    """Harder difficulty should have higher risk values."""
    gm_easy, _ = generate_semantic_trap(seed=0, difficulty="easy")
    gm_hard, _ = generate_semantic_trap(seed=0, difficulty="hard")
    easy_max = gm_easy.true_risk.max()
    hard_max = gm_hard.true_risk.max()
    assert hard_max >= easy_max


# ── Env compatibility ────────────────────────────────────────────────

@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_env_runs_with_family_map(family):
    """Each family map should work inside PedagogicalGridEnv."""
    gm, cfg = generate_benchmark_map(family, 42, "medium")
    env = PedagogicalGridEnv(
        grid_map=gm,
        max_steps=cfg.max_steps,
        initial_risk_budget=cfg.risk_budget,
        prior_risk_mean=cfg.prior_risk_mean,
        prior_risk_var=cfg.prior_risk_var,
        search_budget=cfg.search_budget,
        seed=0,
    )
    obs, info = env.reset()
    assert obs is not None
    # Run a few steps
    for _ in range(5):
        obs, rew, term, trunc, info = env.step(0)  # WAIT
        if term or trunc:
            break


# ── Invalid family ───────────────────────────────────────────────────

def test_invalid_family_raises():
    with pytest.raises(ValueError):
        generate_benchmark_map("nonexistent_family", 0, "easy")
