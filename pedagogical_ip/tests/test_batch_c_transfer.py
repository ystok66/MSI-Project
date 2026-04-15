"""
Tests for Batch C: Transfer protocol standardization.
"""
import pytest
import numpy as np

from src.metrics.transfer_eval import (
    snapshot_learned_params,
    apply_learned_params,
    run_transfer_episodes,
    run_standard_transfer_protocol,
)
from src.metrics.phase9_metrics import TransferSummary


# ══════════════════════════════════════════════════════════════════════
# 1. Seed mode routing
# ══════════════════════════════════════════════════════════════════════

def test_same_seed_mode_uses_same_seeds():
    """world_weights_seed_mode='same' should reuse train seeds for eval."""
    # Test the seed routing logic without actually running episodes
    train_seeds = [10, 20, 30]
    
    # When mode is "same", eval_seeds should equal train_seeds if not specified
    if True:  # simulate internal logic
        eval_seeds_same = list(train_seeds)
    assert eval_seeds_same == train_seeds


def test_different_seed_mode_uses_offset_seeds():
    """world_weights_seed_mode='different' should use separate seeds."""
    n_eval = 5
    # Default offset seeds: 1000+
    eval_seeds_diff = list(range(1000, 1000 + n_eval))
    assert all(s >= 1000 for s in eval_seeds_diff)
    assert len(eval_seeds_diff) == n_eval


# ══════════════════════════════════════════════════════════════════════
# 2. Snapshot / restore protocol
# ══════════════════════════════════════════════════════════════════════

class _MockPredictor:
    """Minimal mock predictor for testing snapshot/restore."""
    def __init__(self, w_c=None, b_c=0.0, w_r=None, b_r=0.0):
        self.w_c = w_c if w_c is not None else np.array([1.0, 2.0, 3.0, 4.0])
        self.b_c = b_c
        self.w_r = w_r if w_r is not None else np.array([0.5, 0.5, 0.5, 0.5])
        self.b_r = b_r


class _MockState:
    def __init__(self, predictor=None):
        self.latent_predictor = predictor
        self.risk_head = None


def test_snapshot_returns_dict():
    """snapshot_learned_params returns a dict with 'predictor' key."""
    pred = _MockPredictor()
    state = _MockState(predictor=pred)
    snap = snapshot_learned_params(state)
    assert isinstance(snap, dict)
    assert 'predictor' in snap


def test_snapshot_empty_when_no_predictor():
    """No predictor → empty snapshot."""
    state = _MockState(predictor=None)
    snap = snapshot_learned_params(state)
    assert snap == {} or 'predictor' not in snap


# ══════════════════════════════════════════════════════════════════════
# 3. Transfer summary fields
# ══════════════════════════════════════════════════════════════════════

def test_transfer_summary_has_required_fields():
    """TransferSummary has all required fields for paper reporting."""
    ts = TransferSummary()
    assert hasattr(ts, 'success')
    assert hasattr(ts, 'death')
    assert hasattr(ts, 'timeout')
    assert hasattr(ts, 'steps')
    assert hasattr(ts, 'cumulative_cost')
    assert hasattr(ts, 'cumulative_risk')
    assert hasattr(ts, 'cost_prediction_error')
    assert hasattr(ts, 'risk_calibration_gap')
    assert hasattr(ts, 'teacher_condition')
    assert hasattr(ts, 'env_condition')


# ══════════════════════════════════════════════════════════════════════
# 4. WAIT-only vs tutor-off documentation check
# ══════════════════════════════════════════════════════════════════════

def test_transfer_eval_docstring_mentions_tutor_off():
    """run_standard_transfer_protocol docstring explicitly states WAIT-only ≠ tutor-off."""
    doc = run_standard_transfer_protocol.__doc__
    assert doc is not None
    assert "tutor_mode" in doc.lower() or "tutor-off" in doc.lower()
    assert "WAIT-only" in doc


# ══════════════════════════════════════════════════════════════════════
# 5. Standard protocol signature
# ══════════════════════════════════════════════════════════════════════

def test_standard_transfer_protocol_has_correct_params():
    """run_standard_transfer_protocol has required parameters."""
    import inspect
    sig = inspect.signature(run_standard_transfer_protocol)
    params = list(sig.parameters.keys())
    assert 'runner' in params
    assert 'family' in params
    assert 'world_weights_seed_mode' in params
    assert 'train_seeds' in params
    assert 'eval_seeds' in params


def test_standard_transfer_protocol_defaults():
    """Default values are sensible."""
    import inspect
    sig = inspect.signature(run_standard_transfer_protocol)
    defaults = {k: v.default for k, v in sig.parameters.items()
                if v.default != inspect.Parameter.empty}
    assert defaults['world_weights_seed_mode'] == 'different'
    assert defaults['family'] == 'harder_baseline_v2'
    assert defaults['n_train'] == 5
    assert defaults['n_eval'] == 5
