"""
Boredom Proxy — Phase 1B.

Provides a boredom/frustration penalty for Q_WAIT scoring.

The boredom proxy captures a state where the agent's expected information gain
is near zero but movement cost continues to accumulate. This corresponds to
the proposal's definition of boredom: "expected information gain ≈ 0 while
cost continues".

Formula:
    B_wait = avg_prefix_cost / (ε + learning_gain)

Where:
    avg_prefix_cost = expected_cost / max(1, prefix_len)
    learning_gain   = mean uncertainty along predicted prefix
    ε               = 1e-6 (anti-divzero)

The new Q_WAIT becomes:
    Q_WAIT_new = Q_WAIT_old - β_bore · B_wait

Only one new hyperparameter: β_bore (boredom_weight).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WaitUtilityTerms:
    """Decomposition of WAIT utility for diagnostics and Phase 1B boredom."""
    # Original terms
    learning_gain: float
    risk_wait: float
    deadline_miss: float
    q_wait_old: float

    # Phase 1B: boredom
    avg_prefix_cost: float
    boredom_penalty: float     # B_wait = avg_prefix_cost / (ε + LG)
    q_wait_new: float

    # Context
    prefix_len: int
    expected_cost: float


def compute_boredom_penalty(
    learning_gain: float,
    expected_cost: float,
    prefix_len: int,
    eps: float = 1e-6,
) -> float:
    """Compute boredom penalty B_wait.

    B_wait = avg_prefix_cost / (ε + learning_gain)

    High when:
      - learning_gain is near zero (nothing to learn)
      - cost is non-trivial (agent is still spending effort)

    Low when:
      - learning_gain is high (agent is learning, so WAIT is justified)
      - cost is near zero (agent isn't spending much anyway)
    """
    avg_cost = expected_cost / max(1, prefix_len)
    return avg_cost / (eps + max(0.0, learning_gain))


def compute_wait_utility_with_boredom(
    learning_gain: float,
    risk_wait: float,
    deadline_miss: float,
    expected_cost: float,
    prefix_len: int,
    learning_gain_weight: float = 1.0,
    catastrophe_weight: float = 5.0,
    deadline_weight: float = 2.0,
    boredom_weight: float = 0.5,
    eps: float = 1e-6,
) -> WaitUtilityTerms:
    """Compute Q_WAIT with Phase 1B boredom penalty.

    Q_WAIT_new = β_learn · LG - β_cat · R - β_ddl · D - β_bore · B

    Args:
        learning_gain: mean uncertainty along prefix
        risk_wait: expected risk of WAIT prefix
        deadline_miss: deadline miss probability
        expected_cost: total expected cost of prefix
        prefix_len: number of prefix cells
        learning_gain_weight: β_learn
        catastrophe_weight: β_cat
        deadline_weight: β_ddl
        boredom_weight: β_bore (the ONE new hyperparameter)
        eps: anti-divzero

    Returns:
        WaitUtilityTerms with old & new scores and decomposition
    """
    q_old = (
        learning_gain_weight * learning_gain
        - catastrophe_weight * risk_wait
        - deadline_weight * deadline_miss
    )

    avg_prefix_cost = expected_cost / max(1, prefix_len)
    boredom = compute_boredom_penalty(learning_gain, expected_cost, prefix_len, eps)

    q_new = q_old - boredom_weight * boredom

    return WaitUtilityTerms(
        learning_gain=learning_gain,
        risk_wait=risk_wait,
        deadline_miss=deadline_miss,
        q_wait_old=q_old,
        avg_prefix_cost=avg_prefix_cost,
        boredom_penalty=boredom,
        q_wait_new=q_new,
        prefix_len=prefix_len,
        expected_cost=expected_cost,
    )
