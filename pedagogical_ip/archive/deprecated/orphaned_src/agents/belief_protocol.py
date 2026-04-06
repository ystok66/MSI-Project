"""
CellBelief — shared protocol for per-cell belief maps.

Inspired by pomdp-py's Belief abstraction: minimal interface for
update/query/copy/reset without coupling to internal representation.

Both BeliefMap (v0-v1d: scalar cost/risk) and FeatureBeliefMap (V2: 4D feature)
satisfy this protocol despite having different internal semantics.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Any

import numpy as np


@runtime_checkable
class CellBelief(Protocol):
    """Minimal shared protocol for per-cell Gaussian belief maps.

    Common operations across both scalar-belief (v0-v1d) and
    feature-belief (V2) systems, without forcing semantic unification.
    """

    H: int
    W: int

    def get_belief(self, row: int, col: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, variance) for a cell.

        For BeliefMap:      returns (array([cost, risk]), array([c_var, r_var]))
        For FeatureBeliefMap: returns (array([f0,..,f3]),  array([v0,..,v3]))
        """
        ...

    def copy(self) -> "CellBelief":
        """Deep copy of the belief map."""
        ...

    def reset(self, **kwargs) -> None:
        """Reset all beliefs to prior."""
        ...
