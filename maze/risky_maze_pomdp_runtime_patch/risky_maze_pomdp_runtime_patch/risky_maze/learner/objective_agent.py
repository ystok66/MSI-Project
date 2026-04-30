"""Objective-aware learner used by the fixed-map POMDP smoke runner.

It is intentionally simple: map memory + online Gaussian risk belief + A* over
known/unknown cells.  It does not inspect full layout or true r/m/q labels.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

from risky_maze.env.objectives import Coord
from risky_maze.env.pomdp_episode import Observation, RuntimeAction, StepOutcome


@dataclass(slots=True)
class SimpleMapMemory:
    known_walls: set[Coord] = field(default_factory=set)
    known_walkable: set[Coord] = field(default_factory=set)
    seen_kind: dict[Coord, str] = field(default_factory=dict)
    observed_vectors: dict[Coord, list[np.ndarray]] = field(default_factory=dict)
    confirmed_traps: set[Coord] = field(default_factory=set)
    confirmed_safe: set[Coord] = field(default_factory=set)
    visited_count: dict[Coord, int] = field(default_factory=dict)
    warning_suspicion: dict[Coord, float] = field(default_factory=dict)

    def update_from_observation(self, obs: Observation) -> int:
        new_walkable = 0
        for coord, cell in obs.visible_cells.items():
            self.seen_kind[coord] = cell.visible_kind
            if cell.is_walkable_observed:
                if coord not in self.known_walkable:
                    new_walkable += 1
                self.known_walkable.add(coord)
                self.known_walls.discard(coord)
                if cell.risk_vector is not None:
                    self.observed_vectors.setdefault(coord, []).append(np.array(cell.risk_vector, copy=True))
            else:
                self.known_walls.add(coord)
        self.visited_count[obs.pos] = self.visited_count.get(obs.pos, 0) + 1
        return new_walkable

    def mean_vector(self, coord: Coord) -> Optional[np.ndarray]:
        values = self.observed_vectors.get(coord)
        if not values:
            return None
        return np.mean(np.stack(values, axis=0), axis=0)


@dataclass(slots=True)
class OnlineGaussianRiskBelief:
    """Tiny binary Gaussian concept model for danger probability diagnostics."""

    risk_dim: int = 6
    prior_danger: float = 0.25
    var: float = 1.0
    safe_count: float = 1.0
    danger_count: float = 1.0
    safe_mean: np.ndarray | None = None
    danger_mean: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.safe_mean is None:
            self.safe_mean = np.zeros(self.risk_dim, dtype=float)
        if self.danger_mean is None:
            self.danger_mean = np.ones(self.risk_dim, dtype=float) * 2.0

    def danger_probability(self, x: np.ndarray | None) -> float:
        if x is None:
            return float(self.prior_danger)
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.safe_mean.shape[0]:
            # Resize robustly for mismatched configs; better than crashing in a
            # smoke runner, and still keeps learner from seeing hidden labels.
            x = np.resize(x, self.safe_mean.shape)
        logp_safe = np.log(max(1e-6, 1.0 - self.prior_danger)) + self._log_gauss(x, self.safe_mean)
        logp_danger = np.log(max(1e-6, self.prior_danger)) + self._log_gauss(x, self.danger_mean)
        m = max(logp_safe, logp_danger)
        p_d = np.exp(logp_danger - m)
        p_s = np.exp(logp_safe - m)
        return float(p_d / max(1e-12, p_d + p_s))

    def update_labeled(self, x: np.ndarray, danger: bool, weight: float = 1.0) -> None:
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.safe_mean.shape[0]:
            x = np.resize(x, self.safe_mean.shape)
        if danger:
            self.danger_mean = (self.danger_mean * self.danger_count + x * weight) / (self.danger_count + weight)
            self.danger_count += weight
        else:
            self.safe_mean = (self.safe_mean * self.safe_count + x * weight) / (self.safe_count + weight)
            self.safe_count += weight
        total = self.safe_count + self.danger_count
        self.prior_danger = float(np.clip(self.danger_count / total, 0.01, 0.99))

    def update_soft(self, x: np.ndarray, p_danger: float, weight: float = 1.0) -> None:
        self.update_labeled(x, True, weight=max(0.0, p_danger) * weight)
        self.update_labeled(x, False, weight=max(0.0, 1.0 - p_danger) * weight)

    def warning_update(self, features: list[np.ndarray]) -> dict[str, float]:
        """Set-level warning update: condition on at least one danger cell.

        Returns before/after aggregate diagnostics used by warning IG metrics.
        """
        if not features:
            return {"mean_abs_delta": 0.0, "sum_delta": 0.0}
        before = np.array([self.danger_probability(x) for x in features], dtype=float)
        p_all_safe = float(np.prod(1.0 - before))
        denom = max(1e-9, 1.0 - p_all_safe)
        after = before / denom
        after = np.clip(after, 0.0, 1.0)
        for x, p in zip(features, after):
            self.update_soft(x, float(p), weight=0.35)
        delta = after - before
        return {"mean_abs_delta": float(np.mean(np.abs(delta))), "sum_delta": float(np.sum(delta))}

    def _log_gauss(self, x: np.ndarray, mean: np.ndarray) -> float:
        diff = x - mean
        return float(-0.5 * np.dot(diff, diff) / max(1e-6, self.var))


class ObjectiveAwareLearner:
    """A minimal belief-state policy for fixed task sequences."""

    def __init__(
        self,
        *,
        risk_dim: int = 6,
        risk_weight: float = 4.0,
        revisit_penalty: float = 0.15,
        unknown_penalty: float = 0.20,
        warning_suspicion_weight: float = 2.0,
    ) -> None:
        self.memory = SimpleMapMemory()
        self.risk_belief = OnlineGaussianRiskBelief(risk_dim=risk_dim)
        self.risk_weight = risk_weight
        self.revisit_penalty = revisit_penalty
        self.unknown_penalty = unknown_penalty
        self.warning_suspicion_weight = warning_suspicion_weight
        self.last_plan: list[Coord] = []
        self.warning_information_gains: list[float] = []
        self.warning_delta_sums: list[float] = []

    def observe(self, obs: Observation) -> None:
        self.memory.update_from_observation(obs)

    def act(self, obs: Observation) -> RuntimeAction:
        self.observe(obs)
        obj = obs.current_objective
        if obj is None:
            self.last_plan = [obs.pos]
            return RuntimeAction.STAY
        path = self._astar(obs.pos, obj.coord, obs.map_shape)
        self.last_plan = path or [obs.pos]
        if not path or len(path) < 2:
            return RuntimeAction.STAY
        return _action_from_to(path[0], path[1])

    def apply_outcome(self, outcome: StepOutcome) -> None:
        if outcome.trap_coord is not None:
            self.memory.confirmed_traps.add(outcome.trap_coord)
            if outcome.danger_feature is not None:
                self.risk_belief.update_labeled(outcome.danger_feature, True, weight=1.0)
        if outcome.moved and outcome.damage == 0 and outcome.trap_coord is None:
            vec = self.memory.mean_vector(outcome.to_pos)
            if vec is not None:
                self.memory.confirmed_safe.add(outcome.to_pos)
                self.risk_belief.update_labeled(vec, False, weight=0.15)

    def apply_warning(self, coords: list[Coord], features: list[np.ndarray]) -> dict[str, float]:
        diagnostics = self.risk_belief.warning_update(features)
        for coord in coords:
            self.memory.warning_suspicion[coord] = self.memory.warning_suspicion.get(coord, 0.0) + 1.0
        self.warning_information_gains.append(diagnostics["mean_abs_delta"])
        self.warning_delta_sums.append(diagnostics["sum_delta"])
        return diagnostics

    def danger_probability(self, x: np.ndarray | None) -> float:
        return self.risk_belief.danger_probability(x)

    def _astar(self, start: Coord, goal: Coord, shape: tuple[int, int]) -> list[Coord] | None:
        h, w = shape
        frontier: list[tuple[float, int, Coord]] = []
        heapq.heappush(frontier, (0.0, 0, start))
        came: dict[Coord, Coord | None] = {start: None}
        cost_so_far: dict[Coord, float] = {start: 0.0}
        tie = 0
        while frontier:
            _, _, cur = heapq.heappop(frontier)
            if cur == goal:
                return _reconstruct(came, cur)
            for nb in _neighbors(cur, h, w):
                if nb in self.memory.known_walls:
                    continue
                step_cost = self._cell_cost(nb)
                new_cost = cost_so_far[cur] + step_cost
                if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                    cost_so_far[nb] = new_cost
                    tie += 1
                    priority = new_cost + abs(nb[0] - goal[0]) + abs(nb[1] - goal[1])
                    heapq.heappush(frontier, (priority, tie, nb))
                    came[nb] = cur
        return None

    def _cell_cost(self, coord: Coord) -> float:
        vec = self.memory.mean_vector(coord)
        p_danger = self.risk_belief.danger_probability(vec)
        known = coord in self.memory.known_walkable
        visits = self.memory.visited_count.get(coord, 0)
        suspicion = self.memory.warning_suspicion.get(coord, 0.0)
        return (
            1.0
            + self.risk_weight * p_danger
            + self.revisit_penalty * visits
            + (0.0 if known else self.unknown_penalty)
            + self.warning_suspicion_weight * suspicion
        )


def _neighbors(coord: Coord, h: int, w: int) -> Iterable[Coord]:
    r, c = coord
    for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nb[0] < h and 0 <= nb[1] < w:
            yield nb


def _reconstruct(came: dict[Coord, Coord | None], cur: Coord) -> list[Coord]:
    out = [cur]
    while came[cur] is not None:
        cur = came[cur]  # type: ignore[assignment]
        out.append(cur)
    out.reverse()
    return out


def _action_from_to(a: Coord, b: Coord) -> RuntimeAction:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if dr == -1 and dc == 0:
        return RuntimeAction.UP
    if dr == 1 and dc == 0:
        return RuntimeAction.DOWN
    if dr == 0 and dc == -1:
        return RuntimeAction.LEFT
    if dr == 0 and dc == 1:
        return RuntimeAction.RIGHT
    return RuntimeAction.STAY
