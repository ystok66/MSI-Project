"""G1 — Self-Discovery Probability Estimation.

Replaces binary V_self = 1[d_commit >= d_reveal] with smooth
sigmoid probability p_self(Δ, τ_v).

p_self = σ((d_commit - d_reveal - m) / τ_v)

Where:
  d_commit: steps before agent is committed to a branch
  d_reveal: depth at which strong cues begin
  m: margin offset (default 0)
  τ_v: temperature controlling transition sharpness
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))


def estimate_self_discovery_prob(
    d_commit: int,
    d_reveal: int,
    margin: float = 0.0,
    tau_v: float = 1.0,
) -> float:
    """Smooth probability that agent can self-discover strong cues.

    p_self = σ((d_commit - d_reveal - margin) / τ_v)

    When d_commit >> d_reveal: p_self → 1 (agent sees cues first)
    When d_commit << d_reveal: p_self → 0 (agent commits blind)
    Boundary zone: smooth transition.
    """
    delta = d_commit - d_reveal - margin
    return _sigmoid(delta / tau_v)


def estimate_self_discovery_obs(
    branch_cells: list,
    obs_radius: int,
    reveal_depth: int,
    n_strong_needed: int = 1,
) -> float:
    """Observation-model-based p_self.

    Counts how many strong-cue cells (idx >= reveal_depth) are
    within obs_radius of fork. If at least n_strong_needed are
    visible, agent can plausibly self-discover.

    Returns: fraction of strong cells visible (soft probability).
    """
    n_strong_total = max(len(branch_cells) - reveal_depth, 1)
    n_visible_strong = sum(1 for i in range(len(branch_cells))
                           if i >= reveal_depth and i < obs_radius)
    return min(n_visible_strong / n_strong_total, 1.0)


def estimate_failure_if_wait(
    d_commit: int,
    d_reveal: int,
    tau_f: float = 1.5,
) -> float:
    """P(fail if wait): probability of committing to wrong branch
    if tutor doesn't warn.

    Higher when d_commit < d_reveal (agent commits blind).
    """
    return 1.0 - estimate_self_discovery_prob(d_commit, d_reveal, tau_v=tau_f)
