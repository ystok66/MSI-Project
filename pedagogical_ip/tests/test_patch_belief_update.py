"""
Tests for patch-based belief updates — Phase 5.

Verifies:
1. Patch update changes multiple cells
2. Unobserved cells unchanged
3. visit_count increments for observed cells
4. last_observed_t updates
5. Variance decreases with repeated observation
6. CellBelief protocol satisfied
"""

import numpy as np
import pytest

from src.agents.feature_belief import FeatureBeliefMap
from src.agents.observation_model import observe_features_patch
from src.agents.belief_protocol import CellBelief
from src.envs.lattice_v2 import generate_lattice_v2, FEATURE_DIM


def _make_belief_and_grid():
    gm, cfg, meta = generate_lattice_v2(seed=42)
    H, W = gm.height, gm.width
    fb = FeatureBeliefMap(H, W, d=FEATURE_DIM)
    return fb, gm, meta


def test_patch_update_changes_multiple_cells():
    """A patch observation updates belief for multiple cells."""
    fb, gm, meta = _make_belief_and_grid()
    rng = np.random.default_rng(42)
    fobs = observe_features_patch(
        (2, 1), meta.cell_features, gm.cell_types,
        patch_radius=2, rng=rng)
    for pos, f_obs, f_var in zip(fobs.positions, fobs.feature_obs, fobs.feature_var):
        fb.update(pos[0], pos[1], f_obs, f_var, t=0)
    observed_count = np.sum(fb.observed)
    assert observed_count > 1, f"Should observe >1 cell, got {observed_count}"
    assert observed_count == len(fobs.positions)


def test_unobserved_cells_unchanged():
    """Cells outside the patch remain at prior."""
    fb, gm, meta = _make_belief_and_grid()
    rng = np.random.default_rng(42)
    fobs = observe_features_patch(
        (2, 1), meta.cell_features, gm.cell_types,
        patch_radius=1, rng=rng)
    observed_set = set(fobs.positions)
    for pos, f_obs, f_var in zip(fobs.positions, fobs.feature_obs, fobs.feature_var):
        fb.update(pos[0], pos[1], f_obs, f_var, t=0)
    # Check a cell far from (2,3) that shouldn't be observed
    H, W = fb.H, fb.W
    for r in range(H):
        for c in range(W):
            if (r, c) not in observed_set:
                mean, var = fb.get_belief(r, c)
                assert np.allclose(mean, 0.5), f"Cell ({r},{c}) mean changed but shouldn't"
                assert np.allclose(var, 0.25), f"Cell ({r},{c}) var changed but shouldn't"


def test_visit_count_updated_for_observed_cells():
    """visit_count increments only for observed (not traversed) cells."""
    fb, gm, meta = _make_belief_and_grid()
    rng = np.random.default_rng(42)
    fobs = observe_features_patch(
        (2, 1), meta.cell_features, gm.cell_types,
        patch_radius=2, rng=rng)
    for pos, f_obs, f_var in zip(fobs.positions, fobs.feature_obs, fobs.feature_var):
        fb.update(pos[0], pos[1], f_obs, f_var, t=5)
    for pos in fobs.positions:
        assert fb.visit_count[pos[0], pos[1]] == 1
    # Unobserved cells should have visit_count=0
    observed_set = set(fobs.positions)
    assert np.all(fb.visit_count[~np.isin(
        np.arange(fb.H * fb.W).reshape(fb.H, fb.W),
        [r * fb.W + c for r, c in observed_set]
    )] == 0) or True  # simplified check below
    for r in range(fb.H):
        for c in range(fb.W):
            if (r, c) not in observed_set:
                assert fb.visit_count[r, c] == 0


def test_last_observed_time_updated():
    """last_observed_t records the latest observation timestep."""
    fb, gm, meta = _make_belief_and_grid()
    rng = np.random.default_rng(42)
    fobs = observe_features_patch(
        (2, 1), meta.cell_features, gm.cell_types,
        patch_radius=1, rng=rng)
    for pos, f_obs, f_var in zip(fobs.positions, fobs.feature_obs, fobs.feature_var):
        fb.update(pos[0], pos[1], f_obs, f_var, t=7)
    for pos in fobs.positions:
        assert fb.last_observed_t[pos[0], pos[1]] == 7
    # Update again at t=12
    rng2 = np.random.default_rng(99)
    fobs2 = observe_features_patch(
        (2, 1), meta.cell_features, gm.cell_types,
        patch_radius=1, rng=rng2)
    for pos, f_obs, f_var in zip(fobs2.positions, fobs2.feature_obs, fobs2.feature_var):
        fb.update(pos[0], pos[1], f_obs, f_var, t=12)
    for pos in fobs2.positions:
        assert fb.last_observed_t[pos[0], pos[1]] == 12


def test_diagonal_variance_decreases_with_repeated_observation():
    """Multiple observations reduce variance monotonically."""
    fb, gm, meta = _make_belief_and_grid()
    _, var_before = fb.get_belief(2, 1)
    for i in range(5):
        rng = np.random.default_rng(i)
        fobs = observe_features_patch(
            (2, 1), meta.cell_features, gm.cell_types,
            patch_radius=1, rng=rng)
        for pos, f_obs, f_var in zip(fobs.positions, fobs.feature_obs, fobs.feature_var):
            fb.update(pos[0], pos[1], f_obs, f_var, t=i)
    _, var_after = fb.get_belief(2, 1)
    assert np.all(var_after < var_before), "Variance should decrease with observations"


def test_patch_update_still_satisfies_belief_protocol():
    """Extended FeatureBeliefMap still satisfies CellBelief protocol."""
    fb, _, _ = _make_belief_and_grid()
    assert isinstance(fb, CellBelief)
