"""Semantic Subspace — principled separation of semantic vs identity features.

Feature dimensions are partitioned into:
  - SEMANTIC dims (2, 3): texture features that SHOULD drive risk/cost
  - IDENTITY dims (0):    lane_id / row identity — MUST NOT drive risk
  - NUISANCE dims (1):    gate_flag — weak/irrelevant to risk

This module provides:
  1. generate_world_weights_orthogonal(): risk weights zeroed on identity dims
  2. identity_leakage_probe(): test if identity alone can predict safe/risky
  3. Feature neutralization utilities
"""

from __future__ import annotations

import numpy as np

# ── Feature partition ────────────────────────────────────────────
SEMANTIC_DIMS = [2, 3]    # texture → risk/cost signal
IDENTITY_DIMS = [0]        # lane_id → position identity (nuisance for risk)
NUISANCE_DIMS = [1]        # gate_flag → structural marker

ALL_DIMS = SEMANTIC_DIMS + IDENTITY_DIMS + NUISANCE_DIMS


def generate_world_weights_orthogonal(
    rng: np.random.Generator,
    d: int = 4,
    identity_dims: list[int] | None = None,
) -> "WorldWeights":
    """Generate world weights with risk orthogonal to identity dims.

    w_risk[i] = 0 for all i in identity_dims.
    This ensures risk predictions depend only on semantic features.
    """
    from ..agents.cost_risk_model import WorldWeights

    if identity_dims is None:
        identity_dims = IDENTITY_DIMS

    # Cost: all dims contribute (cost is layout-dependent, that's OK)
    w_cost = rng.uniform(-0.3, 0.3, size=d).astype(np.float64)
    b_cost = 1.0

    # Risk: ONLY semantic dims contribute
    w_risk = np.zeros(d, dtype=np.float64)
    for i in range(d):
        if i in identity_dims:
            w_risk[i] = 0.0  # explicitly zeroed
        elif i in NUISANCE_DIMS:
            w_risk[i] = rng.uniform(-0.1, 0.1)  # very weak
        else:
            # Semantic dims: strong signal
            w_risk[i] = rng.uniform(2.0, 4.0)

    b_risk = rng.uniform(-3.0, -1.5)

    return WorldWeights(w_cost=w_cost, b_cost=b_cost,
                        w_risk=w_risk, b_risk=b_risk)


def neutralize_identity_features(
    features: np.ndarray,
    cells: list[tuple[int, int]],
    neutral_value: float = 0.5,
    identity_dims: list[int] | None = None,
) -> np.ndarray:
    """Set identity dims to neutral value for specified cells.

    Returns modified feature array (copy).
    """
    if identity_dims is None:
        identity_dims = IDENTITY_DIMS
    out = features.copy()
    for r, c in cells:
        for d in identity_dims:
            out[r, c, d] = neutral_value
    return out


def identity_leakage_probe(
    features: np.ndarray,
    safe_cells: list[tuple[int, int]],
    risky_cells: list[tuple[int, int]],
    identity_dims: list[int] | None = None,
    n_trials: int = 100,
) -> float:
    """Test if identity dims alone can predict safe/risky.

    Returns accuracy of a simple linear probe using only identity features.
    Ideal: ≈ 0.50 (chance). Leakage: > 0.70.
    """
    if identity_dims is None:
        identity_dims = IDENTITY_DIMS

    X, y = [], []
    for r, c in safe_cells:
        X.append(features[r, c][identity_dims])
        y.append(0)
    for r, c in risky_cells:
        X.append(features[r, c][identity_dims])
        y.append(1)

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        return 0.5

    # Simple threshold classifier on each identity dim
    best_acc = 0.5
    for d in range(X.shape[1]):
        thresh = np.median(X[:, d])
        pred = (X[:, d] > thresh).astype(int)
        acc = np.mean(pred == y)
        best_acc = max(best_acc, acc, 1. - acc)

    return round(best_acc, 3)
