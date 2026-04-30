from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from ..env.state import QueryState
from ..interfaces import Option


def route_pick_distribution(
    predictor,
    qs: QueryState,
    active: List[Option],
    spec,
    learner,
    *,
    direct_fn: Callable[[QueryState, List[Option], dict, object], np.ndarray],
) -> np.ndarray:
    """Route pick-distribution queries through predictor or direct learner access."""
    if predictor is not None:
        return predictor.pick_dist(qs, active, spec)
    return direct_fn(qs, active, spec, learner)


def route_rollout_estimate(
    predictor,
    qs: QueryState,
    active: List[Option],
    spec,
    learner,
    n_rollout: int,
    *,
    direct_fn: Callable[[QueryState, List[Option], dict, object, Optional[int]], Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    """Route rollout queries through predictor or direct rollout implementation."""
    if predictor is not None:
        return predictor.rollout(qs, active, spec, n_rollout)
    return direct_fn(qs, active, spec, learner, n_rollout)


def compute_p_death_proxy(
    qs: QueryState,
    active: List[Option],
    probs: np.ndarray,
) -> float:
    """Single-step P(death) proxy over non-correct lethal options."""
    if len(probs) == 0 or len(probs) != len(active):
        return 0.0
    hp = qs.hp
    p_d = 0.0
    for i, opt in enumerate(active):
        if opt.is_correct:
            continue
        if opt.risk_class >= hp:
            p_d += float(probs[i])
    return max(0.0, p_d)


def compute_p_timeout_proxy(
    qs: QueryState,
    active: List[Option],
    probs: np.ndarray,
) -> float:
    """Geometric P(timeout) proxy from one-step correct-pick probability."""
    tau_t = max(0, qs.max_rounds - qs.rounds_used)
    if tau_t <= 0:
        return 1.0

    if len(probs) == 0 or len(probs) != len(active):
        return 1.0

    p_j_star = 0.0
    for i, opt in enumerate(active):
        if opt.is_correct:
            p_j_star = float(probs[i])
            break

    p_success = 1.0 - (1.0 - p_j_star) ** tau_t
    return float(max(0.0, 1.0 - p_success))


def should_use_rollout(
    rollout_mode: str,
    mode: str,
    action: str,
    *,
    q_wait: Optional[float] = None,
    p_timeout_proxy: Optional[float] = None,
) -> bool:
    """Gate whether rollout should be used for this candidate."""
    if rollout_mode == "proxy":
        return False
    if action == "WAIT":
        return False
    if rollout_mode == "full":
        return True
    if mode == "rescue":
        return True
    if q_wait is not None:
        pass
    high_timeout = p_timeout_proxy is not None and p_timeout_proxy > 0.5
    return high_timeout
