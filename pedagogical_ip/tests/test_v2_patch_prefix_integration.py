"""
Integration tests for V2 patch + prefix — Phase 5.

Verifies:
1. latent+patch mode episode runs
2. latent+patch+prefix mode episode runs
3. env info contains prefix predictions
4. legacy mode baseline unchanged
"""

import numpy as np
import pytest

from src.envs.lattice_v2_runner import LatticeV2Runner


runner = LatticeV2Runner()


def test_latent_patch_mode_episode_runs():
    """latent_mode + patch_radius=2 completes an episode."""
    s = runner.reset(seed=42, latent_mode=True, patch_radius=2)
    assert s.patch_radius == 2
    while not s.done:
        runner.step(s)
    assert s.done
    assert s.steps > 0


def test_latent_patch_prefix_mode_episode_runs():
    """latent_mode + patch + prefix completes an episode."""
    s = runner.reset(seed=42, latent_mode=True, patch_radius=2, prefix_horizon=5)
    assert s.prefix_horizon == 5
    while not s.done:
        runner.step(s)
    assert s.done
    assert s.steps > 0


def test_env_info_contains_prefix_predictions():
    """When prefix_horizon > 0, last_prefix is populated."""
    s = runner.reset(seed=42, latent_mode=True, patch_radius=2, prefix_horizon=5)
    runner.step(s)
    if not s.done:
        # After at least one step, prefix should be computed
        assert s.last_prefix is not None, "last_prefix should be populated"
        assert len(s.last_prefix.prefix_cells) > 0
        assert s.last_prefix.cumulative_cost > 0
        assert 0 <= s.last_prefix.cumulative_risk < 1.0


def test_legacy_mode_baseline_unchanged():
    """Legacy mode (no latent, no patch, no prefix) produces expected results."""
    results = []
    for seed in range(20):
        s = runner.reset(seed=seed, tutor_mode="none",
                         latent_mode=False, patch_radius=1, prefix_horizon=0)
        while not s.done:
            runner.step(s)
        results.append(runner.get_metrics(s))
    surv = sum(r["survived"] for r in results) / len(results)
    assert surv < 0.30, f"Legacy no_tutor survival too high: {surv:.0%}"
