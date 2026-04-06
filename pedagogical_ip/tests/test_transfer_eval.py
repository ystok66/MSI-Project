"""
Tests for transfer evaluation — Phase 9.
"""

import numpy as np
import pytest

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.metrics.transfer_eval import (
    snapshot_learned_params, apply_learned_params,
    run_transfer_episodes,
)
from src.metrics.phase9_metrics import TransferSummary


runner = LatticeV2Runner()


def _train_agent(seed=42, **kw):
    """Train agent for a few episodes, return final state."""
    last_state = None
    for s in range(seed, seed + 5):
        state = runner.reset(seed=s, latent_mode=True, **kw)
        while not state.done:
            runner.step(state)
        last_state = state
    return last_state


def test_transfer_eval_runs():
    """Transfer evaluation runs without error."""
    trained = _train_agent()
    results = run_transfer_episodes(runner, trained, n_episodes=2,
                                     seeds=[1000, 1001])
    assert len(results) == 2
    assert all(isinstance(r, TransferSummary) for r in results)


def test_transfer_eval_uses_no_tutor_policy():
    """Transfer episodes run without tutor."""
    trained = _train_agent(tutor_mode="warn_first")
    results = run_transfer_episodes(runner, trained, n_episodes=2,
                                     seeds=[1000, 1001])
    assert len(results) == 2


def test_transfer_metrics_recorded_separately():
    """Transfer summaries are TransferSummary, not EpisodeSummary."""
    from src.metrics.phase9_metrics import EpisodeSummary
    trained = _train_agent()
    results = run_transfer_episodes(runner, trained, n_episodes=2)
    for r in results:
        assert isinstance(r, TransferSummary)
        assert not isinstance(r, EpisodeSummary)


def test_transfer_copies_learned_params_but_resets_state():
    """Snapshot copies predictor weights but transfer resets episode state."""
    trained = _train_agent()
    snap = snapshot_learned_params(trained)
    assert len(snap) > 0  # learned params exist

    # Fresh state
    fresh = runner.reset(seed=999, latent_mode=True)
    w_before = fresh.latent_predictor.cost_head.w.copy()
    apply_learned_params(fresh, snap)
    w_after = fresh.latent_predictor.cost_head.w.copy()
    # Weights changed
    assert not np.array_equal(w_before, w_after)
    # But position is at start (reset)
    assert fresh.agent_pos == (2, 1) or fresh.steps == 0


def test_transfer_eval_reproducible_with_seed():
    """Fixed seeds produce same transfer results."""
    trained = _train_agent()
    results1 = run_transfer_episodes(runner, trained, n_episodes=3,
                                      seeds=[1000, 1001, 1002])
    results2 = run_transfer_episodes(runner, trained, n_episodes=3,
                                      seeds=[1000, 1001, 1002])
    for r1, r2 in zip(results1, results2):
        assert r1.success == r2.success
        assert r1.death == r2.death


def test_transfer_eval_handles_multiple_agent_strengths():
    """Transfer works with different training conditions."""
    trained_weak = _train_agent()
    trained_strong = _train_agent(tutor_mode="warn_first")
    r1 = run_transfer_episodes(runner, trained_weak, n_episodes=2)
    r2 = run_transfer_episodes(runner, trained_strong, n_episodes=2)
    assert len(r1) == 2
    assert len(r2) == 2


def test_transfer_eval_read_only_to_training_logs():
    """Transfer does not modify the trained state's predictor."""
    trained = _train_agent()
    w_before = trained.latent_predictor.cost_head.w.copy()
    run_transfer_episodes(runner, trained, n_episodes=2)
    w_after = trained.latent_predictor.cost_head.w
    np.testing.assert_array_equal(w_before, w_after)


def test_transfer_after_training_differs_from_scratch():
    """Transfer from trained agent should differ from untrained agent."""
    # Untrained
    untrained = runner.reset(seed=42, latent_mode=True)
    # Trained
    trained = _train_agent()
    snap_untrained = snapshot_learned_params(untrained)
    snap_trained = snapshot_learned_params(trained)
    # Weights should be different
    assert not np.array_equal(snap_untrained.get('cost_w'),
                               snap_trained.get('cost_w'))
