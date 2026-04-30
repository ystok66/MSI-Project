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
    known_walls,
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

    The predictor enumerates feasible first moves, completes each with a
    profile-specific A* route, and softmaxes over the resulting first-step
    values.  This is enough for the tutor to reason counterfactually without
    turning the prototype into a full POMDP solver.

    Performance Design (2026-04 optimisation)
    -----------------------------------------
    ``predict_topk`` is the single largest contributor to inverse-tutor
    latency (~96 s of the original 108 s profile).  Three optimisations
    reduce this to ~37 s:

    1. **Pre-computed traversable set**: Instead of calling
       ``shadow_traversable(layout, memory, coord)`` per A* expansion
       (~17 M calls), we compute ``walkable_not_walled = layout._walkable_coords
       - known_wall_cells`` once per ``predict_topk`` call and use a
       ``coord in frozenset`` check.  Falls back to the generic call chain
       when layout lacks ``_walkable_coords``.

    2. **k=1 fast path**: 97 %% of calls come from rollout simulation with
       ``k=1``.  The original code ran 3–4 separate A* calls (one per
       first-step neighbour) then picked the cheapest.  For k=1, a single
       A* from start→goal suffices, giving ~3× fewer A* calls.

    3. **k>1 multi-path mode** (original interface, fully preserved): When
       ``k>1`` (used by ``_build_profile_path_cache`` for profile-belief
       likelihood evaluation), the code enumerates all first-step neighbours,
       runs A* for each, deduplicates, and softmaxes.  This provides diverse
       path candidates needed for Bayesian profile inference.
    """

    def __init__(self, unknown_risk_prior: float = 0.25):
        self.unknown_risk_prior = unknown_risk_prior

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

        known_walkable_cells = known_walkable(shadow.memory_hat)
        known_wall_cells = known_walls(shadow.memory_hat)
        visited_map = getattr(shadow.memory_hat, "visited_count", None)
        risk_cache: dict[Coord, float] = {}
        cost_cache: dict[Coord, float] = {}

        def cell_risk_cached(coord: Coord) -> float:
            if coord in risk_cache:
                return risk_cache[coord]
            feat = observed_vector(shadow.memory_hat, layout, coord, allow_oracle=False)
            if feat is None and coord in known_walkable_cells:
                feat = feature_at(layout, coord)
            risk = _danger_probability(shadow.risk_belief_hat, feat, self.unknown_risk_prior)
            risk_cache[coord] = risk
            return risk

        def visit_count_cached(coord: Coord) -> int:
            if isinstance(visited_map, dict):
                return int(visited_map.get(coord, 0) or 0)
            return visited_count(shadow.memory_hat, coord)

        def cell_cost_cached(coord: Coord) -> float:
            if coord in cost_cache:
                return cost_cache[coord]
            profile = shadow.profile
            known = coord in known_walkable_cells
            unknown = 0.0 if known else 1.0
            risk = cell_risk_cached(coord)
            revisit = visit_count_cached(coord)
            value = max(
                0.05,
                1.0
                + profile.risk_weight * risk
                + profile.unknown_penalty * unknown
                + profile.revisit_penalty * revisit
                - profile.info_bonus * unknown,
            )
            cost_cache[coord] = value
            return value

        def score_path_cached(cells: list[Coord]) -> tuple[float, float, float, int]:
            if not cells:
                return float("inf"), 0.0, 0.0, 0
            total = 0.0
            risk = 0.0
            info = 0.0
            revisits = 0
            for c in cells:
                total += cell_cost_cached(c)
                r = cell_risk_cached(c)
                risk += r
                if c not in known_walkable_cells:
                    info += 1.0
                if visit_count_cached(c) > 0:
                    revisits += 1
            return total, risk, info, revisits

        def trav(c: Coord) -> bool:
            return c in walkable_not_walled

        # Pre-compute the traversable set: layout-walkable minus learner-known
        # walls.  This turns every ``trav(coord)`` call in the A* inner loop
        # into a single ``coord in frozenset`` check, replacing the original
        # shadow_traversable → in_bounds → is_wall → char_at chain that
        # consumed ~40 % of total profile time.
        #
        # Fast path: FixedRuntimeLayout provides _walkable_coords frozenset.
        # Generic fallback: enumerate all walkable cells via shadow_traversable.
        _layout_walkable = getattr(layout, '_walkable_coords', None)
        if _layout_walkable is not None:
            walkable_not_walled = _layout_walkable - known_wall_cells
        else:
            walkable_not_walled = frozenset(
                c for c in known_walkable_cells
                if shadow_traversable(layout, shadow.memory_hat, c, known_wall_cells=known_wall_cells)
            )
            # Also include unknown cells that are layout-walkable
            h_bound, w_bound = getattr(layout, '_h', 0), getattr(layout, '_w', 0)
            if h_bound and w_bound:
                for r in range(h_bound):
                    for c_col in range(w_bound):
                        cc = (r, c_col)
                        if cc not in walkable_not_walled and shadow_traversable(layout, shadow.memory_hat, cc, known_wall_cells=known_wall_cells):
                            walkable_not_walled = walkable_not_walled | {cc}

        # ---- k≤1 fast path: single A* from start→goal -------------------
        # Rollout calls predict_topk with k=1 (~97 % of all calls).  The
        # original code enumerates 3–4 first-step neighbours and runs a
        # separate A* for each, then picks the cheapest.  For k=1 the optimal
        # path is unique, so a single A* from start directly to goal produces
        # the same result at ~3–4× lower cost (29 926 A* → 10 323 A*).
        #
        # The k>1 multi-path mode below is fully preserved for callers that
        # need diverse paths (e.g. _build_profile_path_cache with top_k=3).
        if k <= 1:
            raw = shortest_path(
                layout,
                start,
                target,
                traversable=trav,
                extra_cost=cell_cost_cached,
                extra_cost_offset=1.0,
            )
            if not raw or len(raw) < 2:
                return []
            cells = raw[1 : horizon + 1]
            if not cells:
                return []
            cost, risk, info, revisits = score_path_cached(cells)
            actions = [action_between(a, b) for a, b in zip(raw[:-1], raw[1:])][: len(cells)]
            return [
                PredictedPath(
                    cells=list(cells),
                    actions=actions,
                    probability=1.0,
                    predicted_cost=cost,
                    predicted_risk=risk,
                    predicted_info_gain=info,
                    predicted_revisit_count=revisits,
                )
            ]

        # ---- k>1: enumerate first-step neighbours for diverse paths ------
        # Used by _build_profile_path_cache (top_k=3) which needs multiple
        # candidate paths per profile for Bayesian likelihood evaluation.
        # This is the original A* multi-path interface, fully preserved.
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
                    extra_cost=cell_cost_cached,
                    extra_cost_offset=1.0,
                )
                raw = [start] + suffix if suffix else []
            if not raw:
                continue
            cells = raw[1 : horizon + 1]
            if not cells:
                continue
            cost, risk, info, revisits = score_path_cached(cells)
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
