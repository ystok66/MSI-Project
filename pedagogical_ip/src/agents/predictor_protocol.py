"""
PredictorProtocol — Shape-agnostic predictor interface.

All teacher-side, evaluation, and logging code should depend ONLY
on this protocol. No module outside the predictor itself should
access .cost_head.w, .risk_head.w, or assume fixed dimensions.

Provides:
  - PredictorProtocol: typed interface for cost/risk prediction
  - snapshot_predictor: deep-copy a predictor (replaces manual w/b extraction)
  - restore_predictor: restore weights from snapshot (replaces w[:] = ...)
  - extract_theta: dynamic-dim parameter vector for logging
  - predictor_summary: weight norms, update counts, type info
"""

from __future__ import annotations

from copy import deepcopy
from typing import Protocol, runtime_checkable

import numpy as np


# ════════════════════════════════════════════════════════════════
# Protocol
# ════════════════════════════════════════════════════════════════

@runtime_checkable
class PredictorProtocol(Protocol):
    """Shape-agnostic predictor interface.

    Teacher-side code should depend ONLY on these methods,
    never on internal .cost_head.w / .risk_head.w / fixed dim.

    All three head types implement this by duck typing:
      - LatentCostRiskHead (4D linear)
      - StructuredBasisCostRiskHead (6D cost / 7D risk basis)
      - SlowFastCostRiskHead (wraps any inner head)
    """

    def predict_cost(self, x: np.ndarray) -> float: ...
    def predict_risk(self, x: np.ndarray) -> float: ...
    def predict_cost_uncertainty(self, x: np.ndarray) -> float: ...
    def predict_risk_uncertainty(self, x: np.ndarray) -> float: ...
    def predict_cost_uncertainty_from_var(self, x_var: np.ndarray) -> float: ...
    def predict_risk_uncertainty_from_var(self, x_var: np.ndarray) -> float: ...

    @property
    def n_updates(self) -> int: ...


# ════════════════════════════════════════════════════════════════
# Utilities
# ════════════════════════════════════════════════════════════════

def snapshot_predictor(predictor):
    """Deep-copy a predictor. Shape-agnostic.

    Replaces manual (w, b) extraction. Works with any head type.
    Returns an independent copy safe for read-only surrogate use.
    """
    if predictor is None:
        return None
    return deepcopy(predictor)


def restore_predictor(target, source) -> None:
    """Restore target's full internal state from source. Shape-agnostic.

    Replaces manual w[:] = snapshot['cost_w'] assignment.
    Uses deepcopy of source's __dict__ to avoid shared references.

    Args:
        target: predictor to overwrite (must be same type as source)
        source: predictor to copy from
    """
    if source is None or target is None:
        return
    # Copy all internal state, preserving target's identity
    for key, value in source.__dict__.items():
        try:
            setattr(target, key, deepcopy(value))
        except (TypeError, AttributeError):
            pass  # skip non-copyable attributes


def extract_theta(predictor) -> list:
    """Extract flat parameter vector from any predictor type.

    Dynamic-dim: reads cost_head.w/b and risk_head.w/b regardless
    of their actual dimensions. No hardcoded d_f split.

    Returns:
        [*cost_w, cost_b, *risk_w, risk_b]
        Length varies by head type (e.g., 10 for 4D linear, 15 for basis).
    """
    if predictor is None:
        return []

    parts = []

    # Cost head
    if hasattr(predictor, 'cost_head'):
        ch = predictor.cost_head
        if hasattr(ch, 'w'):
            parts.extend(ch.w.tolist())
        if hasattr(ch, 'b'):
            parts.append(float(ch.b))

    # Risk head
    if hasattr(predictor, 'risk_head'):
        rh = predictor.risk_head
        if hasattr(rh, 'w'):
            parts.extend(rh.w.tolist())
        if hasattr(rh, 'b'):
            parts.append(float(rh.b))

    return parts


def extract_theta_components(predictor):
    """Extract (cost_w, cost_b, risk_w, risk_b) as numpy arrays.

    Dynamic-dim: returns arrays of whatever dimension the heads use.
    Callers must NOT assume a fixed dimension.

    Returns:
        (cost_w: ndarray, cost_b: float, risk_w: ndarray, risk_b: float)
    """
    if predictor is None:
        return np.zeros(0), 0.0, np.zeros(0), 0.0

    cost_w = np.zeros(0)
    cost_b = 0.0
    risk_w = np.zeros(0)
    risk_b = 0.0

    if hasattr(predictor, 'cost_head'):
        ch = predictor.cost_head
        cost_w = ch.w.copy() if hasattr(ch, 'w') else np.zeros(0)
        cost_b = float(ch.b) if hasattr(ch, 'b') else 0.0

    if hasattr(predictor, 'risk_head'):
        rh = predictor.risk_head
        risk_w = rh.w.copy() if hasattr(rh, 'w') else np.zeros(0)
        risk_b = float(rh.b) if hasattr(rh, 'b') else 0.0

    return cost_w, cost_b, risk_w, risk_b


def predictor_summary(predictor) -> dict:
    """Return diagnostic summary of a predictor. Shape-agnostic.

    Replaces ad-hoc `risk_w_norm` and `n_updates` reads scattered
    across logging and eval scripts.

    Returns dict with:
      - predictor_type: class name
      - cost_w_norm, risk_w_norm: L2 norms
      - cost_w_dim, risk_w_dim: actual parameter dimensions
      - cost_b, risk_b: bias values
      - n_updates: total update count
    """
    if predictor is None:
        return {"predictor_type": "None", "n_updates": 0}

    cost_w, cost_b, risk_w, risk_b = extract_theta_components(predictor)

    return {
        "predictor_type": type(predictor).__name__,
        "cost_w_norm": float(np.linalg.norm(cost_w)) if len(cost_w) > 0 else 0.0,
        "risk_w_norm": float(np.linalg.norm(risk_w)) if len(risk_w) > 0 else 0.0,
        "cost_w_dim": len(cost_w),
        "risk_w_dim": len(risk_w),
        "cost_b": cost_b,
        "risk_b": risk_b,
        "n_updates": predictor.n_updates if hasattr(predictor, 'n_updates') else 0,
    }
