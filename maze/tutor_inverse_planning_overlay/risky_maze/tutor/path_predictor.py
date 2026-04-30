from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .compat import Coord, action_between, neighbors4, safe_float
from .profiles import LearnerProfile
from .shadow import ShadowLearnerState
from .world_model import (
    current_pos,
    feature_at,
    known_walkable,
    objective_coord,
    observed_vector,
    shadow_traversable,
    shortest_path,
    visited_count,
)


@dataclass
class PredictedPath:
    cells: list[Coord]
    actions: list[str]
    probability: float
    predicted_cost: float
    predicted_risk: float
    predicted_info_gain: float
    predicted_revisit_count: int


def _danger_probability(belief: Any, feature: Any, default: float = 0.25) -> float:
    if belief is None or feature is None:
        return default
    method = getattr(belief, "danger_probability", None)
    if callable(method):
        try:
            return max(0.0, min(1.0, float(method(feature))))
        except Exception:
            return default
    posterior = getattr(belief, "posterior", None)
    if callable(posterior):
        try:
            probs = posterior(feature)
            if isinstance(probs, dict):
                return max(0.0, min(1.0, 1.0 - float(probs.get("safe", probs.get(0, 0.0)))))
        except Exception:
            return default
    return default


class LearnerPathPredictor:
    """Profile-conditioned short-horizon path predictor.

    The predictor is intentionally lightweight: it enumerates feasible first
    moves, completes each with a profile-specific A* route, and softmaxes over
    the resulting first-step values.  This is enough for the tutor to reason
    counterfactually without turning the prototype into a full POMDP solver.
    """

    def __init__(self, unknown_risk_prior: float = 0.25):
        self.unknown_risk_prior = unknown_risk_prior

    def cell_risk(self, shadow: ShadowLearnerState, layout: Any, coord: Coord) -> float:
        feat = observed_vector(shadow.memory_hat, layout, coord, allow_oracle=False)
        if feat is None:
            feat = feature_at(layout, coord) if coord in known_walkable(shadow.memory_hat) else None
        return _danger_probability(shadow.risk_belief_hat, feat, self.unknown_risk_prior)

    def cell_cost(self, shadow: ShadowLearnerState, layout: Any, coord: Coord) -> float:
        profile = shadow.profile
        known = coord in known_walkable(shadow.memory_hat)
        unknown = 0.0 if known else 1.0
        risk = self.cell_risk(shadow, layout, coord)
        revisit = visited_count(shadow.memory_hat, coord)
        # Keep the final edge cost positive even for curiosity-heavy profiles.
        return max(
            0.05,
            1.0
            + profile.risk_weight * risk
            + profile.unknown_penalty * unknown
            + profile.revisit_penalty * revisit
            - profile.info_bonus * unknown,
        )

    def _score_path(self, shadow: ShadowLearnerState, layout: Any, cells: list[Coord]) -> tuple[float, float, float, int]:
        if not cells:
            return float("inf"), 0.0, 0.0, 0
        total = 0.0
        risk = 0.0
        info = 0.0
        revisits = 0
        kw = known_walkable(shadow.memory_hat)
        for c in cells:
            total += self.cell_cost(shadow, layout, c)
            r = self.cell_risk(shadow, layout, c)
            risk += r
            if c not in kw:
                info += 1.0
            if visited_count(shadow.memory_hat, c) > 0:
                revisits += 1
        return total, risk, info, revisits

    def predict_topk(
        self,
        shadow: ShadowLearnerState,
        env_state: Any,
        layout: Any | None = None,
        k: int = 4,
        horizon: int = 8,
        waypoint: Coord | None = None,
    ) -> list[PredictedPath]:
        # Backwards-compatible call shape: predict_topk(shadow, env_state, k, horizon)
        if layout is not None and isinstance(layout, int):  # type: ignore[unreachable]
            k = int(layout)
            layout = None
        if layout is None:
            layout = getattr(env_state, "layout", None)
        if layout is None:
            return []

        try:
            start = current_pos(env_state)
        except Exception:
            return []
        target = waypoint or shadow.objective_state_hat.target or objective_coord(layout, env_state)
        if target is None:
            return []

        def trav(c: Coord) -> bool:
            return shadow_traversable(layout, shadow.memory_hat, c)

        candidates: list[PredictedPath] = []
        first_steps = [n for n in neighbors4(start) if trav(n)]
        if not first_steps and trav(start):
            first_steps = [start]
        for first in first_steps:
            if first == start:
                raw = [start]
            else:
                suffix = shortest_path(
                    layout,
                    first,
                    target,
                    traversable=trav,
                    extra_cost=lambda c: max(0.0, self.cell_cost(shadow, layout, c) - 1.0),
                )
                raw = [start] + suffix if suffix else []
            if not raw:
                continue
            cells = raw[1 : horizon + 1]
            if not cells:
                continue
            cost, risk, info, revisits = self._score_path(shadow, layout, cells)
            actions = [action_between(a, b) for a, b in zip(raw[:-1], raw[1:])][: len(cells)]
            candidates.append(
                PredictedPath(
                    cells=list(cells),
                    actions=actions,
                    probability=0.0,
                    predicted_cost=cost,
                    predicted_risk=risk,
                    predicted_info_gain=info,
                    predicted_revisit_count=revisits,
                )
            )

        # Deduplicate paths with the same first step and prefix, keeping the cheaper one.
        by_key: dict[tuple[Coord, ...], PredictedPath] = {}
        for p in candidates:
            key = tuple(p.cells[: min(3, len(p.cells))])
            if key not in by_key or p.predicted_cost < by_key[key].predicted_cost:
                by_key[key] = p
        candidates = sorted(by_key.values(), key=lambda p: p.predicted_cost)
        candidates = candidates[: max(1, k)]
        if not candidates:
            return []

        beta = max(1e-6, safe_float(shadow.profile.softmax_beta, 1.0))
        m = min(p.predicted_cost for p in candidates)
        weights = [math.exp(-beta * (p.predicted_cost - m)) for p in candidates]
        z = sum(weights) or 1.0
        for p, w in zip(candidates, weights):
            p.probability = w / z
        return candidates


def first_action_probability(paths: list[PredictedPath], observed_action: Any) -> float:
    from .compat import normalize_action_name

    name = normalize_action_name(observed_action)
    prob = 0.0
    for path in paths:
        if path.actions and normalize_action_name(path.actions[0]) == name:
            prob += path.probability
    return prob
