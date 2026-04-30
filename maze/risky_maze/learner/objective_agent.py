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
class TransferGraphMemory:
    route_node_confidence: dict[Coord, float] = field(default_factory=dict)
    route_edge_confidence: dict[tuple[Coord, Coord], float] = field(default_factory=dict)
    landmark_confidence: dict[Coord, float] = field(default_factory=dict)
    landmark_kind: dict[Coord, str] = field(default_factory=dict)
    landmark_graph_confidence: dict[tuple[Coord, Coord], float] = field(default_factory=dict)
    autonomy_credit_history: list[float] = field(default_factory=list)
    successful_commits: int = 0
    total_commits: int = 0
    objective_learning_event_count: int = 0

    def clone(self) -> "TransferGraphMemory":
        return TransferGraphMemory(
            route_node_confidence=dict(self.route_node_confidence),
            route_edge_confidence=dict(self.route_edge_confidence),
            landmark_confidence=dict(self.landmark_confidence),
            landmark_kind=dict(self.landmark_kind),
            landmark_graph_confidence=dict(self.landmark_graph_confidence),
            autonomy_credit_history=list(self.autonomy_credit_history),
            successful_commits=self.successful_commits,
            total_commits=self.total_commits,
            objective_learning_event_count=self.objective_learning_event_count,
        )

    def clear(self) -> None:
        self.route_node_confidence.clear()
        self.route_edge_confidence.clear()
        self.landmark_confidence.clear()
        self.landmark_kind.clear()
        self.landmark_graph_confidence.clear()
        self.autonomy_credit_history.clear()
        self.successful_commits = 0
        self.total_commits = 0
        self.objective_learning_event_count = 0

    def mean_autonomy_credit(self) -> float:
        if not self.autonomy_credit_history:
            return 0.0
        return float(sum(self.autonomy_credit_history) / len(self.autonomy_credit_history))

    def route_graph_confidence_score(self) -> float:
        vals = list(self.route_edge_confidence.values())
        if not vals:
            vals = list(self.route_node_confidence.values())
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    def landmark_graph_confidence_score(self) -> float:
        vals = list(self.landmark_graph_confidence.values())
        if not vals:
            vals = list(self.landmark_confidence.values())
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))


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
    transfer_graph: TransferGraphMemory = field(default_factory=TransferGraphMemory)
    _mean_vector_cache: dict[Coord, np.ndarray] = field(default_factory=dict)

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
                    self._mean_vector_cache.pop(coord, None)
                    self.observed_vectors.setdefault(coord, []).append(np.array(cell.risk_vector, copy=True))
            else:
                self.known_walls.add(coord)
        self.visited_count[obs.pos] = self.visited_count.get(obs.pos, 0) + 1
        return new_walkable

    def mean_vector(self, coord: Coord) -> Optional[np.ndarray]:
        cached = self._mean_vector_cache.get(coord)
        if cached is not None:
            return cached
        values = self.observed_vectors.get(coord)
        if not values:
            return None
        mean = np.mean(np.stack(values, axis=0), axis=0)
        self._mean_vector_cache[coord] = mean
        return mean

    def clone(self) -> "SimpleMapMemory":
        """Lightweight copy for tutor shadow rollouts.

        The rollout only mutates the container structure, visit counts, and
        warning/trap annotations.  The stored feature vectors are treated as
        immutable observations, so we can safely reuse the ndarray objects while
        copying the surrounding containers.
        """
        return SimpleMapMemory(
            known_walls=set(self.known_walls),
            known_walkable=set(self.known_walkable),
            seen_kind=dict(self.seen_kind),
            observed_vectors={coord: list(values) for coord, values in self.observed_vectors.items()},
            confirmed_traps=set(self.confirmed_traps),
            confirmed_safe=set(self.confirmed_safe),
            visited_count=dict(self.visited_count),
            warning_suspicion=dict(self.warning_suspicion),
            transfer_graph=self.transfer_graph.clone(),
            # Cached means are treated as immutable feature summaries in shadow
            # rollouts, so we only need a shallow dict copy here.
            _mean_vector_cache=dict(self._mean_vector_cache),
        )

    def clone_empty(self, *, keep_long_term: bool = True) -> "SimpleMapMemory":
        return SimpleMapMemory(
            transfer_graph=self.transfer_graph.clone() if keep_long_term else TransferGraphMemory()
        )


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
    warning_update_mode: str = "effective_sample"
    warning_eta0: float = 0.35
    warning_kl_epsilon: float = 1e-6

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

    def danger_probabilities_batch(self, xs: Iterable[np.ndarray | None]) -> np.ndarray:
        values = list(xs)
        if not values:
            return np.zeros(0, dtype=float)
        out = np.full(len(values), float(self.prior_danger), dtype=float)
        valid_idx: list[int] = []
        valid_x: list[np.ndarray] = []
        for idx, x in enumerate(values):
            if x is None:
                continue
            arr = np.asarray(x, dtype=float)
            if arr.shape[0] != self.safe_mean.shape[0]:
                arr = np.resize(arr, self.safe_mean.shape)
            valid_idx.append(idx)
            valid_x.append(arr)
        if not valid_x:
            return out
        xmat = np.stack(valid_x, axis=0)
        diff_safe = xmat - self.safe_mean[None, :]
        diff_danger = xmat - self.danger_mean[None, :]
        scale = max(1e-6, self.var)
        log_safe_prior = np.log(max(1e-6, 1.0 - self.prior_danger))
        log_danger_prior = np.log(max(1e-6, self.prior_danger))
        logp_safe = log_safe_prior - 0.5 * np.sum(diff_safe * diff_safe, axis=1) / scale
        logp_danger = log_danger_prior - 0.5 * np.sum(diff_danger * diff_danger, axis=1) / scale
        m = np.maximum(logp_safe, logp_danger)
        p_d = np.exp(logp_danger - m)
        p_s = np.exp(logp_safe - m)
        probs = p_d / np.maximum(1e-12, p_d + p_s)
        out[np.array(valid_idx, dtype=int)] = probs
        return out

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

    def _warning_posterior(self, features: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if not features:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
        before = self.danger_probabilities_batch(features)
        p_all_safe = float(np.prod(1.0 - before))
        denom = max(1e-9, 1.0 - p_all_safe)
        after = before / denom
        after = np.clip(after, 0.0, 1.0)
        return before, after

    def warning_update_literal(self, features: list[np.ndarray]) -> dict[str, float]:
        before, after = self._warning_posterior(features)
        if before.size == 0:
            return {
                "mean_abs_delta": 0.0,
                "sum_delta": 0.0,
                "warning_ess": 0.0,
                "warning_kl": 0.0,
                "warning_set_size": 0.0,
            }
        weight = max(0.0, float(self.warning_eta0))
        for x, p in zip(features, after):
            self.update_soft(x, float(p), weight=weight)
        delta = after - before
        return {
            "mean_abs_delta": float(np.mean(np.abs(delta))),
            "sum_delta": float(np.sum(delta)),
            "warning_ess": float(weight),
            "warning_kl": float(self._bernoulli_kl(after, before).sum()),
            "warning_set_size": float(len(features)),
        }

    def warning_update_effective_sample(self, features: list[np.ndarray]) -> dict[str, float]:
        before, after = self._warning_posterior(features)
        if before.size == 0:
            return {
                "mean_abs_delta": 0.0,
                "sum_delta": 0.0,
                "warning_ess": 0.0,
                "warning_kl": 0.0,
                "warning_set_size": 0.0,
            }
        delta = after - before
        kl_total = float(self._bernoulli_kl(after, before).sum())
        weight = max(
            0.0,
            float(self.warning_eta0) * kl_total / max(self.warning_kl_epsilon, float(np.log1p(len(features)))),
        )
        weight = min(weight, 2.0)
        for x, p in zip(features, after):
            self.update_soft(x, float(p), weight=weight)
        return {
            "mean_abs_delta": float(np.mean(np.abs(delta))),
            "sum_delta": float(np.sum(delta)),
            "warning_ess": float(weight),
            "warning_kl": kl_total,
            "warning_set_size": float(len(features)),
        }

    def warning_update(self, features: list[np.ndarray]) -> dict[str, float]:
        """Set-level warning update: condition on at least one danger cell."""
        mode = str(self.warning_update_mode or "effective_sample").lower()
        if mode in {"literal", "literal_bayes"}:
            return self.warning_update_literal(features)
        return self.warning_update_effective_sample(features)

    def _bernoulli_kl(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        eps = max(1e-9, float(self.warning_kl_epsilon))
        p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
        q = np.clip(np.asarray(q, dtype=float), eps, 1.0 - eps)
        return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))

    def _log_gauss(self, x: np.ndarray, mean: np.ndarray) -> float:
        diff = x - mean
        return float(-0.5 * np.dot(diff, diff) / max(1e-6, self.var))

    def clone(self) -> "OnlineGaussianRiskBelief":
        return OnlineGaussianRiskBelief(
            risk_dim=self.risk_dim,
            prior_danger=self.prior_danger,
            var=self.var,
            safe_count=self.safe_count,
            danger_count=self.danger_count,
            safe_mean=np.array(self.safe_mean, copy=True),
            danger_mean=np.array(self.danger_mean, copy=True),
            warning_update_mode=self.warning_update_mode,
            warning_eta0=self.warning_eta0,
            warning_kl_epsilon=self.warning_kl_epsilon,
        )


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
        warning_suspicion_mode: str = "persistent",
        warning_suspicion_decay: float = 1.0,
        consolidation_mode: str = "none",
        long_term_memory_weight: float = 0.35,
        autonomy_assist_discount: float = 0.05,
        enable_objective_learning_events: bool = True,
        use_long_term_route_graph: bool = True,
        use_landmark_graph: bool = True,
        warning_update_mode: str = "effective_sample",
        warning_eta0: float = 0.35,
        warning_kl_epsilon: float = 1e-6,
        enable_warning_update: bool = True,
        enable_trap_risk_update: bool = True,
        enable_safe_risk_update: bool = True,
    ) -> None:
        self.memory = SimpleMapMemory()
        self.risk_belief = OnlineGaussianRiskBelief(
            risk_dim=risk_dim,
            warning_update_mode=warning_update_mode,
            warning_eta0=warning_eta0,
            warning_kl_epsilon=warning_kl_epsilon,
        )
        self.risk_weight = risk_weight
        self.revisit_penalty = revisit_penalty
        self.unknown_penalty = unknown_penalty
        self.warning_suspicion_weight = warning_suspicion_weight
        self.warning_suspicion_mode = str(warning_suspicion_mode or "persistent").lower()
        self.warning_suspicion_decay = float(np.clip(warning_suspicion_decay, 0.0, 1.0))
        self.consolidation_mode = str(consolidation_mode or "none").lower()
        self.long_term_memory_weight = float(max(0.0, long_term_memory_weight))
        self.autonomy_assist_discount = float(max(0.0, autonomy_assist_discount))
        self.enable_objective_learning_events = bool(enable_objective_learning_events)
        self.use_long_term_route_graph = bool(use_long_term_route_graph)
        self.use_landmark_graph = bool(use_landmark_graph)
        self.enable_warning_update = bool(enable_warning_update)
        self.enable_trap_risk_update = bool(enable_trap_risk_update)
        self.enable_safe_risk_update = bool(enable_safe_risk_update)
        self.last_plan: list[Coord] = []
        self.warning_information_gains: list[float] = []
        self.warning_delta_sums: list[float] = []
        self.warning_effective_sample_sizes: list[float] = []
        self.warning_kls: list[float] = []
        self.current_waypoint: Coord | None = None
        self._last_observation_token: int | None = None
        self._transient_warning_suspicion: dict[Coord, float] = {}
        self._active_objective_kind: str | None = None

    def observe(self, obs: Observation) -> None:
        token = id(obs)
        if self._last_observation_token == token:
            return
        self.memory.update_from_observation(obs)
        self._last_observation_token = token

    def act(self, obs: Observation) -> RuntimeAction:
        self.observe(obs)
        self._active_objective_kind = None if obs.current_objective is None else str(obs.current_objective.kind)
        if self.current_waypoint is not None and obs.pos == self.current_waypoint:
            self.current_waypoint = None
        obj = obs.current_objective
        if obj is None:
            self.last_plan = [obs.pos]
            return RuntimeAction.STAY
        target = self.current_waypoint or obj.coord
        path = self._astar(obs.pos, target, obs.map_shape)
        self.last_plan = path or [obs.pos]
        if self.warning_suspicion_mode in {"replan_only", "query_only"}:
            self._transient_warning_suspicion.clear()
        if not path or len(path) < 2:
            return RuntimeAction.STAY
        return _action_from_to(path[0], path[1])

    def set_waypoint(self, coord: Coord | None) -> None:
        self.current_waypoint = coord

    def apply_outcome(self, outcome: StepOutcome) -> None:
        if outcome.trap_coord is not None:
            self.memory.confirmed_traps.add(outcome.trap_coord)
            if self.enable_trap_risk_update and outcome.danger_feature is not None:
                self.risk_belief.update_labeled(outcome.danger_feature, True, weight=1.0)
        if self.enable_safe_risk_update and outcome.moved and outcome.damage == 0 and outcome.trap_coord is None:
            vec = self.memory.mean_vector(outcome.to_pos)
            if vec is not None:
                self.memory.confirmed_safe.add(outcome.to_pos)
                self.risk_belief.update_labeled(vec, False, weight=0.15)
        if self.warning_suspicion_mode == "episode_decay" and self.warning_suspicion_decay < 1.0:
            self._decay_warning_suspicion()

    def apply_warning(self, coords: list[Coord], features: list[np.ndarray]) -> dict[str, float]:
        diagnostics = {
            "mean_abs_delta": 0.0,
            "sum_delta": 0.0,
            "warning_ess": 0.0,
            "warning_kl": 0.0,
            "warning_set_size": float(len(features)),
        }
        if self.enable_warning_update:
            diagnostics = self.risk_belief.warning_update(features)
        self._apply_warning_suspicion(coords)
        self.warning_information_gains.append(diagnostics["mean_abs_delta"])
        self.warning_delta_sums.append(diagnostics["sum_delta"])
        self.warning_effective_sample_sizes.append(diagnostics.get("warning_ess", 0.0))
        self.warning_kls.append(diagnostics.get("warning_kl", 0.0))
        return diagnostics

    def finalize_episode(
        self,
        *,
        phase: str,
        success: bool,
        path: list[Coord],
        learning_events: list[dict[str, Any]] | None = None,
        objective_events: list[dict[str, Any]] | None = None,
        assist_leakage: float = 0.0,
        waypoint_count: int = 0,
        warning_count: int = 0,
        waypoint_progress_gift: float = 0.0,
        waypoint_novelty_leak: float = 0.0,
    ) -> None:
        if self.warning_suspicion_mode == "query_only":
            self.clear_warning_suspicion()
        self.current_waypoint = None
        if not str(phase).startswith("teach"):
            return

        mode = self.consolidation_mode
        if mode in {"none", "disabled", "off"}:
            return
        if mode in {"success_gated", "success_gated_assist_discounted"} and not success:
            return

        autonomy_credit = 1.0
        if mode == "success_gated_assist_discounted":
            autonomy_credit = float(np.exp(-self.autonomy_assist_discount * max(0.0, float(assist_leakage))))
        events = (
            list(learning_events)
            if learning_events is not None
            else list(objective_events or [])
        )
        self._commit_transfer_memory(
            success=success,
            path=path,
            learning_events=events,
            autonomy_credit=autonomy_credit,
        )

    def danger_probability(self, x: np.ndarray | None) -> float:
        return self.risk_belief.danger_probability(x)

    def danger_probabilities_batch(self, xs: Iterable[np.ndarray | None]) -> np.ndarray:
        return self.risk_belief.danger_probabilities_batch(xs)

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
                if self.use_long_term_route_graph and self.consolidation_mode not in {"none", "disabled", "off"}:
                    step_cost = max(0.05, step_cost - 0.5 * self.long_term_memory_weight * self._route_edge_bonus(cur, nb))
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
        suspicion = self.warning_suspicion_value(coord)
        route_bonus = self._route_node_bonus(coord)
        landmark_bonus = self._landmark_bonus(coord)
        base = (
            1.0
            + self.risk_weight * p_danger
            + self.revisit_penalty * visits
            + (0.0 if known else self.unknown_penalty)
            + self.warning_suspicion_weight * suspicion
        )
        return max(0.05, base - self.long_term_memory_weight * (route_bonus + landmark_bonus))

    def clone_for_eval(
        self,
        *,
        clear_memory: bool = False,
        clear_risk_belief: bool = False,
        clear_warning_suspicion: bool = False,
        clear_long_term_memory: bool = False,
    ) -> "ObjectiveAwareLearner":
        cloned = ObjectiveAwareLearner(
            risk_dim=self.risk_belief.risk_dim,
            risk_weight=self.risk_weight,
            revisit_penalty=self.revisit_penalty,
            unknown_penalty=self.unknown_penalty,
            warning_suspicion_weight=self.warning_suspicion_weight,
            warning_suspicion_mode=self.warning_suspicion_mode,
            warning_suspicion_decay=self.warning_suspicion_decay,
            consolidation_mode=self.consolidation_mode,
            long_term_memory_weight=self.long_term_memory_weight,
            autonomy_assist_discount=self.autonomy_assist_discount,
            enable_objective_learning_events=self.enable_objective_learning_events,
            use_long_term_route_graph=self.use_long_term_route_graph,
            use_landmark_graph=self.use_landmark_graph,
            warning_update_mode=self.risk_belief.warning_update_mode,
            warning_eta0=self.risk_belief.warning_eta0,
            warning_kl_epsilon=self.risk_belief.warning_kl_epsilon,
            enable_warning_update=self.enable_warning_update,
            enable_trap_risk_update=self.enable_trap_risk_update,
            enable_safe_risk_update=self.enable_safe_risk_update,
        )
        cloned.memory = (
            self.memory.clone_empty(keep_long_term=not clear_long_term_memory)
            if clear_memory
            else self.memory.clone()
        )
        if clear_warning_suspicion:
            cloned.memory.warning_suspicion.clear()
        if clear_long_term_memory and not clear_memory:
            cloned.memory.transfer_graph.clear()
        cloned.risk_belief = (
            OnlineGaussianRiskBelief(
                risk_dim=self.risk_belief.risk_dim,
                warning_update_mode=self.risk_belief.warning_update_mode,
                warning_eta0=self.risk_belief.warning_eta0,
                warning_kl_epsilon=self.risk_belief.warning_kl_epsilon,
            )
            if clear_risk_belief
            else self.risk_belief.clone()
        )
        cloned.warning_information_gains = []
        cloned.warning_delta_sums = []
        cloned.warning_effective_sample_sizes = []
        cloned.warning_kls = []
        cloned.last_plan = list(self.last_plan)
        cloned.current_waypoint = None
        cloned._transient_warning_suspicion = {}
        cloned._active_objective_kind = None
        return cloned

    def warning_suspicion_value(self, coord: Coord) -> float:
        return float(self.memory.warning_suspicion.get(coord, 0.0) + self._transient_warning_suspicion.get(coord, 0.0))

    def warning_suspicion_mass(self, coords: Iterable[Coord] | None = None) -> float:
        if coords is None:
            keys = set(self.memory.warning_suspicion) | set(self._transient_warning_suspicion)
            return float(sum(self.warning_suspicion_value(coord) for coord in keys))
        return float(sum(self.warning_suspicion_value(coord) for coord in coords))

    def clear_warning_suspicion(self) -> None:
        self.memory.warning_suspicion.clear()
        self._transient_warning_suspicion.clear()

    def _apply_warning_suspicion(self, coords: Iterable[Coord]) -> None:
        mode = self.warning_suspicion_mode
        if mode in {"none", "disabled", "off"}:
            return
        target = self._transient_warning_suspicion if mode in {"replan_only", "query_only"} else self.memory.warning_suspicion
        for coord in coords:
            target[coord] = float(target.get(coord, 0.0) + 1.0)

    def _decay_warning_suspicion(self) -> None:
        if not self.memory.warning_suspicion:
            return
        decayed: dict[Coord, float] = {}
        for coord, value in self.memory.warning_suspicion.items():
            new_value = float(value) * self.warning_suspicion_decay
            if new_value > 1e-6:
                decayed[coord] = new_value
        self.memory.warning_suspicion = decayed

    def _commit_transfer_memory(
        self,
        *,
        success: bool,
        path: list[Coord],
        learning_events: list[dict[str, Any]],
        autonomy_credit: float,
    ) -> None:
        tg = self.memory.transfer_graph
        tg.total_commits += 1
        if success:
            tg.successful_commits += 1
        tg.autonomy_credit_history.append(float(autonomy_credit))
        if not path:
            return

        unique_nodes = list(dict.fromkeys(path))
        node_inc = float(autonomy_credit) / max(1.0, float(np.sqrt(len(unique_nodes))))
        edge_inc = float(autonomy_credit) / max(1.0, float(np.sqrt(max(1, len(path) - 1))))
        for coord in unique_nodes:
            tg.route_node_confidence[coord] = float(tg.route_node_confidence.get(coord, 0.0) + node_inc)
        for a, b in zip(path, path[1:]):
            key = _edge_key(a, b)
            tg.route_edge_confidence[key] = float(tg.route_edge_confidence.get(key, 0.0) + edge_inc)

        if not self.enable_objective_learning_events:
            return
        prev_landmark: Coord | None = None
        for event in learning_events:
            kind = str(event.get("kind", ""))
            coord = tuple(event.get("coord", ()))  # type: ignore[assignment]
            if len(coord) != 2:
                continue
            tg.objective_learning_event_count += 1
            tg.landmark_kind[coord] = kind
            tg.landmark_confidence[coord] = float(tg.landmark_confidence.get(coord, 0.0) + autonomy_credit)
            if prev_landmark is not None:
                edge = _edge_key(prev_landmark, coord)
                tg.landmark_graph_confidence[edge] = float(tg.landmark_graph_confidence.get(edge, 0.0) + autonomy_credit)
            prev_landmark = coord

    def _route_node_bonus(self, coord: Coord) -> float:
        if not self.use_long_term_route_graph or self.consolidation_mode in {"none", "disabled", "off"}:
            return 0.0
        return min(1.0, float(self.memory.transfer_graph.route_node_confidence.get(coord, 0.0)))

    def _route_edge_bonus(self, a: Coord, b: Coord) -> float:
        if not self.use_long_term_route_graph or self.consolidation_mode in {"none", "disabled", "off"}:
            return 0.0
        return min(1.0, float(self.memory.transfer_graph.route_edge_confidence.get(_edge_key(a, b), 0.0)))

    def _landmark_bonus(self, coord: Coord) -> float:
        if not self.use_landmark_graph or self.consolidation_mode in {"none", "disabled", "off"}:
            return 0.0
        conf = float(self.memory.transfer_graph.landmark_confidence.get(coord, 0.0))
        if conf <= 0.0:
            return 0.0
        kind = self.memory.transfer_graph.landmark_kind.get(coord)
        if kind is not None and self._active_objective_kind is not None and kind == self._active_objective_kind:
            return min(1.25, conf)
        return min(0.5, 0.35 * conf)


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


def _edge_key(a: Coord, b: Coord) -> tuple[Coord, Coord]:
    return (a, b) if a <= b else (b, a)


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
