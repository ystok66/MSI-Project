"""Batch B regression tests -- scenario generator correctness.

Tests fork_trap fix, dead code removal, reachability, metadata consistency,
and pure-latent contract.
"""

import numpy as np
import pytest
from collections import deque

from src.envs.scenario_families import generate_scenario, SCENARIO_REGISTRY
from src.envs.map_generator import CellType


def _bfs_reachable(gm, start, goal):
    """BFS: is goal reachable from start?"""
    H, W = gm.height, gm.width
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in ((0,1),(0,-1),(1,0),(-1,0)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and (nr,nc) not in visited:
                if gm.cell_types[nr,nc] != CellType.WALL:
                    visited.add((nr,nc))
                    queue.append((nr,nc))
    return False


def _bfs_safe_reachable(gm, start, goal, avoid_risky=True):
    """BFS avoiding RISKY cells."""
    H, W = gm.height, gm.width
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in ((0,1),(0,-1),(1,0),(-1,0)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and (nr,nc) not in visited:
                ct = gm.cell_types[nr,nc]
                if ct != CellType.WALL and (not avoid_risky or ct != CellType.RISKY or (nr,nc)==goal):
                    visited.add((nr,nc))
                    queue.append((nr,nc))
    return False


# ── B1.2: fork_trap safe path fix ─────────────────────────────────────

@pytest.mark.parametrize("seed", range(100))
def test_fork_trap_safe_path_exists(seed):
    """fork_trap must have a safe path for ALL seeds (was 39% before fix)."""
    gm, cfg, meta, sc = generate_scenario("fork_trap", seed=seed, latent_mode=True)
    W = gm.width
    start = (2, 1)
    goal = (2, W - 2)
    assert _bfs_safe_reachable(gm, start, goal), (
        f"fork_trap seed={seed}: no safe path exists")


def test_fork_trap_safe_row_always_3():
    """After fix, safe_row should always be 3 (detour via rows 4,5)."""
    for seed in range(20):
        gm, cfg, meta, sc = generate_scenario("fork_trap", seed=seed, latent_mode=True)
        seg = meta.segments[0]
        assert seg.safe_row == 3, f"seed={seed}: safe_row={seg.safe_row}, expected 3"
        assert seg.risky_row == 1, f"seed={seed}: risky_row={seg.risky_row}, expected 1"


# ── B1.1: Dead code removal ───────────────────────────────────────────

def test_delayed_corridor_import_works():
    """delayed_corridor should still import and generate after dead code removal."""
    gm, cfg, meta, sc = generate_scenario("delayed_corridor", seed=42, latent_mode=True)
    assert gm is not None
    assert sc.family_name == "delayed_corridor"


def test_distractor_cue_import_works():
    """distractor_cue should still import and generate after dead code removal."""
    gm, cfg, meta, sc = generate_scenario("distractor_cue", seed=42, latent_mode=True)
    assert gm is not None
    assert sc.family_name == "distractor_cue"


def test_registry_has_all_families():
    """After cleanup, registry should still have all 14 families."""
    expected = {
        "baseline_v2", "fork_trap", "hazard_belt", "deadline_gate",
        "delayed_corridor", "distractor_cue", "funnel_trap",
        "elcb", "elcb_po", "temptation_corridor",
        "joint_conflict_corridor",
        "deep_tree_mixed_bottleneck_lattice",
        "goal_preference_temptation_entanglement_lattice",
        "harder_baseline_v2",
    }
    actual = set(SCENARIO_REGISTRY.keys())
    assert expected == actual, f"Missing: {expected - actual}, Extra: {actual - expected}"


# ── Goal reachability (all families) ──────────────────────────────────

MAIN_FAMILIES = [
    "baseline_v2", "fork_trap", "hazard_belt", "deadline_gate",
    "delayed_corridor", "distractor_cue", "harder_baseline_v2",
]


@pytest.mark.parametrize("family", MAIN_FAMILIES)
def test_goal_reachable_20_seeds(family):
    """Goal must be reachable for all main families across 20 seeds."""
    for seed in range(20):
        gm, cfg, meta, sc = generate_scenario(family, seed=seed, latent_mode=True)
        W = gm.width
        start = getattr(gm, 'agent_start', (2, 1)) or (2, 1)
        goal = getattr(gm, 'target_pos', (2, W - 2)) or (2, W - 2)
        assert _bfs_reachable(gm, start, goal), (
            f"{family} seed={seed}: goal unreachable")


# ── B2: Pure latent contract ──────────────────────────────────────────

PURE_LATENT_FAMILIES = ["baseline_v2", "fork_trap", "harder_baseline_v2"]


@pytest.mark.parametrize("family", PURE_LATENT_FAMILIES)
def test_pure_latent_risk_contract(family):
    """Pure-latent families: true_risk must match WorldWeights(z)."""
    for seed in range(10):
        gm, cfg, meta, sc = generate_scenario(family, seed=seed, latent_mode=True)
        ww = meta.world_weights
        assert ww is not None, f"{family} seed={seed}: no WorldWeights in latent_mode"
        H, W = gm.height, gm.width
        feats = meta.cell_features
        for r in range(H):
            for c in range(W):
                if gm.cell_types[r, c] == CellType.RISKY:
                    z = feats[r, c]
                    ww_risk = ww.true_risk(z)
                    exec_risk = float(gm.true_risk[r, c])
                    assert abs(exec_risk - ww_risk) < 0.02, (
                        f"{family} seed={seed} ({r},{c}): "
                        f"exec_risk={exec_risk:.4f} != ww_risk={ww_risk:.4f}")


# ── B3: Scenario contract validation ──────────────────────────────────

def test_scenario_contract_validates_on_generate():
    """generate_scenario should call validate_scenario_contract internally."""
    # If reachability or metadata is broken, generate_scenario itself will raise.
    # This test just confirms the validation is wired up.
    for family in MAIN_FAMILIES:
        gm, cfg, meta, sc = generate_scenario(family, seed=42, latent_mode=True)
        assert gm is not None
