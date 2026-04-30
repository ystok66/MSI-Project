"""Metrics and oracle comparators for fixed risky-maze blocks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from risky_maze.env.fixed_loader import FixedRuntimeLayout, MazeTask
from risky_maze.env.objectives import Coord, Objective


@dataclass(slots=True)
class EpisodeMetrics:
    task_id: str
    phase: str
    success: bool = False
    died: bool = False
    timeout: bool = False
    steps: int = 0
    damage: int = 0
    warnings: int = 0
    waypoints: int = 0
    assist_leakage: float = 0.0
    discovered_cells: int = 0
    repeated_steps: int = 0
    blocked_steps: int = 0
    trap_entries: int = 0
    path: list[Coord] = field(default_factory=list)
    discovered_walkable: set[Coord] = field(default_factory=set)
    warning_information_gain: float = 0.0
    posterior_shift_after_warning: float = 0.0
    warning_effective_sample_size: float = 0.0
    warning_kl: float = 0.0
    immortal_danger_events: int = 0
    objective_completed_count: int = 0
    no_info_steps: int = 0
    frontier_progress_steps: int = 0
    warning_action_total: int = 0
    warning_actionable_count: int = 0
    warning_path_changed_count: int = 0
    useful_wait_count: int = 0
    bad_wait_count: int = 0
    preventable_death_count: int = 0
    safety_shield_triggered: int = 0
    waypoint_progress_gift: float = 0.0
    waypoint_novelty_leak: float = 0.0
    map_gain_after_waypoint: float = 0.0
    risk_ig_after_waypoint: float = 0.0
    predicted_risk_ig_after_waypoint: float = 0.0
    oracle_safe_steps: Optional[int] = None
    eval_regret_to_oracle_safe_path: Optional[int] = None
    elapsed_seconds: float = 0.0
    step_records: list[dict[str, Any]] = field(default_factory=list)
    tutor_decisions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def loop_rate(self) -> float:
        return self.repeated_steps / max(1, self.steps)

    @property
    def no_info_step_rate(self) -> float:
        return self.no_info_steps / max(1, self.steps)

    @property
    def repeated_known_cell_rate(self) -> float:
        return self.repeated_steps / max(1, self.steps)

    @property
    def damage_per_100_steps(self) -> float:
        return 100.0 * float(self.damage) / max(1, self.steps)

    @property
    def frontier_progress_rate(self) -> float:
        return self.frontier_progress_steps / max(1, self.steps)

    @property
    def warning_actionability(self) -> float:
        return self.warning_actionable_count / max(1, self.warning_action_total)

    @property
    def useful_wait_rate(self) -> float:
        return self.useful_wait_count / max(1, self.steps)

    @property
    def bad_wait_rate(self) -> float:
        return self.bad_wait_count / max(1, self.steps)

    @property
    def mean_waypoint_progress_gift(self) -> float:
        return self.waypoint_progress_gift / max(1, self.waypoints)

    @property
    def mean_waypoint_novelty_leak(self) -> float:
        return self.waypoint_novelty_leak / max(1, self.waypoints)

    @property
    def seconds_per_step(self) -> float:
        return float(self.elapsed_seconds) / max(1, self.steps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "phase": self.phase,
            "success": self.success,
            "died": self.died,
            "timeout": self.timeout,
            "steps": self.steps,
            "damage": self.damage,
            "warnings": self.warnings,
            "waypoints": self.waypoints,
            "assist_leakage": self.assist_leakage,
            "discovered_cells": self.discovered_cells,
            "repeated_steps": self.repeated_steps,
            "blocked_steps": self.blocked_steps,
            "trap_entries": self.trap_entries,
            "warning_information_gain": self.warning_information_gain,
            "posterior_shift_after_warning": self.posterior_shift_after_warning,
            "warning_effective_sample_size": self.warning_effective_sample_size,
            "warning_kl": self.warning_kl,
            "immortal_danger_events": self.immortal_danger_events,
            "objective_completed_count": self.objective_completed_count,
            "no_info_step_rate": self.no_info_step_rate,
            "loop_rate": self.loop_rate,
            "repeated_known_cell_rate": self.repeated_known_cell_rate,
            "damage_per_100_steps": self.damage_per_100_steps,
            "frontier_progress_rate": self.frontier_progress_rate,
            "warning_actionability": self.warning_actionability,
            "warning_action_total": self.warning_action_total,
            "warning_path_changed_count": self.warning_path_changed_count,
            "useful_wait_rate": self.useful_wait_rate,
            "bad_wait_rate": self.bad_wait_rate,
            "preventable_death_count": self.preventable_death_count,
            "safety_shield_triggered": self.safety_shield_triggered,
            "mean_waypoint_progress_gift": self.mean_waypoint_progress_gift,
            "mean_waypoint_novelty_leak": self.mean_waypoint_novelty_leak,
            "map_gain_after_waypoint": self.map_gain_after_waypoint,
            "risk_ig_after_waypoint": self.risk_ig_after_waypoint,
            "predicted_risk_ig_after_waypoint": self.predicted_risk_ig_after_waypoint,
            "oracle_safe_steps": self.oracle_safe_steps,
            "eval_regret_to_oracle_safe_path": self.eval_regret_to_oracle_safe_path,
            "elapsed_seconds": self.elapsed_seconds,
            "seconds_per_step": self.seconds_per_step,
            "episode_cost": episode_cost(self),
        }


@dataclass(slots=True)
class BlockMetrics:
    teach: list[dict[str, Any]]
    eval_same_map: list[dict[str, Any]]
    aggregate: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"teach": self.teach, "eval_same_map": self.eval_same_map, "aggregate": self.aggregate}

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def __contains__(self, key: str) -> bool:
        return key in self.as_dict()


def aggregate_block_metrics(
    teach_eps: list[EpisodeMetrics],
    eval_eps: list[EpisodeMetrics],
    *,
    learner: Any | None,
    teach_learner: Any | None = None,
    eval_probe_learner: Any | None = None,
    layout: FixedRuntimeLayout,
    risk_dataset: tuple[np.ndarray, np.ndarray] | None = None,
    risk_coords: list[Coord] | None = None,
) -> BlockMetrics:
    teach_new_cells: set[Coord] = set().union(*(ep.discovered_walkable for ep in teach_eps)) if teach_eps else set()
    eval_used_cells: set[Coord] = set().union(*(set(ep.path) for ep in eval_eps)) if eval_eps else set()
    useful_exploration_rate = len(teach_new_cells & eval_used_cells) / max(1, len(teach_new_cells))
    map_reuse_eval = len(eval_used_cells & teach_new_cells) / max(1, len(eval_used_cells))

    risk_auc = None
    risk_nll = None
    risk_calibration_ece = None
    risk_auc_seen = None
    risk_auc_unseen_same_map = None
    risk_learner = eval_probe_learner if eval_probe_learner is not None else learner
    if risk_dataset is not None and risk_learner is not None and hasattr(risk_learner, "danger_probability"):
        xs, ys = risk_dataset
        batch_fn = getattr(risk_learner, "danger_probabilities_batch", None)
        if callable(batch_fn):
            probs = np.asarray(batch_fn(xs), dtype=float)
        else:
            probs = np.array([float(risk_learner.danger_probability(x)) for x in xs], dtype=float)
        risk_auc = binary_auc(ys, probs)
        risk_nll = binary_nll(ys, probs)
        risk_calibration_ece = binary_ece(ys, probs)
        if risk_coords is not None:
            seen_cells = set(getattr(getattr(risk_learner, "memory", None), "known_walkable", set()) or set())
            seen_mask = np.array([coord in seen_cells for coord in risk_coords], dtype=bool)
            unseen_mask = ~seen_mask
            if seen_mask.any():
                risk_auc_seen = binary_auc(ys[seen_mask], probs[seen_mask])
            if unseen_mask.any():
                risk_auc_unseen_same_map = binary_auc(ys[unseen_mask], probs[unseen_mask])

    teach_memory = getattr(teach_learner, "memory", None)
    eval_memory = getattr(eval_probe_learner, "memory", None)
    teach_suspicion = dict(getattr(teach_memory, "warning_suspicion", {}) or {})
    eval_suspicion = dict(getattr(eval_memory, "warning_suspicion", {}) or {})
    warning_suspicion_mass_end_teach = float(sum(teach_suspicion.values()))
    warning_suspicion_cells_end_teach = float(len([v for v in teach_suspicion.values() if float(v) > 0.0]))
    warning_suspicion_mass_eval_path = float(sum(eval_suspicion.get(coord, 0.0) for coord in eval_used_cells))
    warning_suspicion_mass_eval_start = float(sum(eval_suspicion.values()))
    transfer_graph = getattr(teach_memory, "transfer_graph", None)
    autonomy_credit = float(getattr(transfer_graph, "mean_autonomy_credit", lambda: 0.0)()) if transfer_graph is not None else 0.0
    route_graph_confidence = (
        float(getattr(transfer_graph, "route_graph_confidence_score", lambda: 0.0)()) if transfer_graph is not None else 0.0
    )
    landmark_graph_confidence = (
        float(getattr(transfer_graph, "landmark_graph_confidence_score", lambda: 0.0)()) if transfer_graph is not None else 0.0
    )
    successful_teach_commits = int(getattr(transfer_graph, "successful_commits", 0)) if transfer_graph is not None else 0
    total_teach_commits = int(getattr(transfer_graph, "total_commits", 0)) if transfer_graph is not None else 0
    objective_learning_event_count = int(getattr(transfer_graph, "objective_learning_event_count", 0)) if transfer_graph is not None else 0
    route_graph_node_count = int(len(getattr(transfer_graph, "route_node_confidence", {}) or {})) if transfer_graph is not None else 0
    landmark_node_count = int(len(getattr(transfer_graph, "landmark_confidence", {}) or {})) if transfer_graph is not None else 0

    all_eps = teach_eps + eval_eps
    aggregate = {
        "success_rate": mean([ep.success for ep in all_eps]),
        "teach_success_rate": mean([ep.success for ep in teach_eps]),
        "eval_success_rate": mean([ep.success for ep in eval_eps]),
        "teach_safe_success_rate": mean([ep.success and ep.damage == 0 and not ep.died and not ep.timeout for ep in teach_eps]),
        "eval_safe_success_rate": mean([ep.success and ep.damage == 0 and not ep.died and not ep.timeout for ep in eval_eps]),
        "death_rate": mean([ep.died for ep in all_eps]),
        "teach_death_rate": mean([ep.died for ep in teach_eps]),
        "eval_death_rate": mean([ep.died for ep in eval_eps]),
        "timeout_rate": mean([ep.timeout for ep in all_eps]),
        "teach_timeout_rate": mean([ep.timeout for ep in teach_eps]),
        "eval_timeout_rate": mean([ep.timeout for ep in eval_eps]),
        "mean_steps": mean([ep.steps for ep in all_eps]),
        "teach_mean_steps": mean([ep.steps for ep in teach_eps]),
        "eval_mean_steps": mean([ep.steps for ep in eval_eps]),
        "mean_elapsed_seconds": mean([ep.elapsed_seconds for ep in all_eps]),
        "teach_mean_elapsed_seconds": mean([ep.elapsed_seconds for ep in teach_eps]),
        "eval_mean_elapsed_seconds": mean([ep.elapsed_seconds for ep in eval_eps]),
        "mean_seconds_per_step": mean([ep.seconds_per_step for ep in all_eps]),
        "teach_mean_seconds_per_step": mean([ep.seconds_per_step for ep in teach_eps]),
        "eval_mean_seconds_per_step": mean([ep.seconds_per_step for ep in eval_eps]),
        "mean_damage": mean([ep.damage for ep in all_eps]),
        "teach_mean_damage": mean([ep.damage for ep in teach_eps]),
        "eval_mean_damage": mean([ep.damage for ep in eval_eps]),
        "mean_damage_per_100_steps": mean([ep.damage_per_100_steps for ep in all_eps]),
        "teach_mean_damage_per_100_steps": mean([ep.damage_per_100_steps for ep in teach_eps]),
        "eval_mean_damage_per_100_steps": mean([ep.damage_per_100_steps for ep in eval_eps]),
        "mean_trap_entries": mean([ep.trap_entries for ep in all_eps]),
        "teach_mean_trap_entries": mean([ep.trap_entries for ep in teach_eps]),
        "eval_mean_trap_entries": mean([ep.trap_entries for ep in eval_eps]),
        "warnings": int(sum(ep.warnings for ep in all_eps)),
        "teach_mean_warnings": mean([ep.warnings for ep in teach_eps]),
        "eval_mean_warnings": mean([ep.warnings for ep in eval_eps]),
        "waypoints": int(sum(ep.waypoints for ep in all_eps)),
        "teach_mean_waypoints": mean([ep.waypoints for ep in teach_eps]),
        "eval_mean_waypoints": mean([ep.waypoints for ep in eval_eps]),
        "assist_leakage": mean([ep.assist_leakage for ep in all_eps]),
        "teach_assist_leakage": mean([ep.assist_leakage for ep in teach_eps]),
        "eval_assist_leakage": mean([ep.assist_leakage for ep in eval_eps]),
        "map_coverage_teach": len(teach_new_cells) / max(1, len(layout.walkable_cells())),
        "map_reuse_eval": map_reuse_eval,
        "useful_exploration_rate": useful_exploration_rate,
        "warning_information_gain": mean([ep.warning_information_gain for ep in all_eps]),
        "posterior_shift_after_warning": mean([ep.posterior_shift_after_warning for ep in all_eps]),
        "warning_effective_sample_size": mean([ep.warning_effective_sample_size for ep in all_eps]),
        "warning_kl": mean([ep.warning_kl for ep in all_eps]),
        "immortal_danger_events": int(sum(ep.immortal_danger_events for ep in all_eps)),
        "objective_completed_count": mean([ep.objective_completed_count for ep in all_eps]),
        "risk_auc": risk_auc,
        "risk_auc_seen": risk_auc_seen,
        "risk_auc_unseen_same_map": risk_auc_unseen_same_map,
        "risk_nll": risk_nll,
        "risk_calibration_ece": risk_calibration_ece,
        "eval_regret_to_oracle_safe_path": mean([
            ep.eval_regret_to_oracle_safe_path for ep in eval_eps if ep.eval_regret_to_oracle_safe_path is not None
        ]),
        "teach_cost": mean([episode_cost(ep) for ep in teach_eps]),
        "eval_cost": mean([episode_cost(ep) for ep in eval_eps]),
        "loop_rate": mean([ep.loop_rate for ep in all_eps]),
        "repeated_known_cell_rate": mean([ep.repeated_known_cell_rate for ep in all_eps]),
        "no_info_step_rate": mean([ep.no_info_step_rate for ep in all_eps]),
        "frontier_progress_rate": mean([ep.frontier_progress_rate for ep in all_eps]),
        "warning_actionability": mean([ep.warning_actionability for ep in all_eps]),
        "warning_action_total": mean([ep.warning_action_total for ep in all_eps]),
        "warning_path_changed_count": mean([ep.warning_path_changed_count for ep in all_eps]),
        "useful_wait_rate": mean([ep.useful_wait_rate for ep in all_eps]),
        "bad_wait_rate": mean([ep.bad_wait_rate for ep in all_eps]),
        "preventable_death_rate": mean([ep.preventable_death_count > 0 for ep in all_eps]),
        "predicted_preventable_death_wait_rate": mean([ep.preventable_death_count > 0 for ep in all_eps]),
        "safety_shield_trigger_rate": mean([ep.safety_shield_triggered > 0 for ep in all_eps]),
        "waypoint_progress_gift": mean([ep.mean_waypoint_progress_gift for ep in all_eps]),
        "waypoint_novelty_leak": mean([ep.mean_waypoint_novelty_leak for ep in all_eps]),
        "map_gain_after_waypoint": mean([ep.map_gain_after_waypoint for ep in all_eps]),
        "risk_ig_after_waypoint": mean([ep.risk_ig_after_waypoint for ep in all_eps]),
        "predicted_risk_ig_after_waypoint": mean([ep.predicted_risk_ig_after_waypoint for ep in all_eps]),
        "warning_suspicion_mass_end_teach": warning_suspicion_mass_end_teach,
        "warning_suspicion_cells_end_teach": warning_suspicion_cells_end_teach,
        "warning_suspicion_mass_eval_start": warning_suspicion_mass_eval_start,
        "warning_suspicion_mass_on_eval_path": warning_suspicion_mass_eval_path,
        "autonomy_credit": autonomy_credit,
        "route_graph_confidence": route_graph_confidence,
        "landmark_graph_confidence": landmark_graph_confidence,
        "successful_teach_commits": successful_teach_commits,
        "total_teach_commits": total_teach_commits,
        "objective_learning_event_count": objective_learning_event_count,
        "route_graph_node_count": route_graph_node_count,
        "landmark_node_count": landmark_node_count,
    }
    return BlockMetrics(
        teach=[ep.as_dict() for ep in teach_eps],
        eval_same_map=[ep.as_dict() for ep in eval_eps],
        aggregate=aggregate,
    )


def oracle_safe_shortest_path(layout: FixedRuntimeLayout, start: Coord, objectives: list[Objective]) -> Optional[int]:
    """Sum shortest safe path length over an ordered objective sequence.

    Safe means walls and true trap cells are excluded.  If a route segment is
    impossible without traps, ``None`` is returned so regret is not fabricated.
    """
    total = 0
    cur = start
    for obj in objectives:
        dist = _bfs_distance(layout, cur, obj.coord, allow_traps=False)
        if dist is None:
            return None
        total += dist
        cur = obj.coord
    return total


def _bfs_distance(layout: FixedRuntimeLayout, start: Coord, goal: Coord, *, allow_traps: bool) -> Optional[int]:
    if start == goal:
        return 0
    q: deque[tuple[Coord, int]] = deque([(start, 0)])
    seen = {start}
    while q:
        cur, d = q.popleft()
        for nb in layout.neighbors4(cur, allow_traps=allow_traps):
            if nb in seen:
                continue
            if nb == goal:
                return d + 1
            seen.add(nb)
            q.append((nb, d + 1))
    return None


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # Average tied ranks.
    for score in np.unique(y_score):
        mask = y_score == score
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    sum_pos_ranks = float(ranks[pos].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / max(1.0, n_pos * n_neg)


def binary_nll(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true).astype(float)
    p = np.clip(np.asarray(p).astype(float), 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def binary_ece(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true).astype(float)
    p = np.clip(np.asarray(p).astype(float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not mask.any():
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += float(mask.mean()) * abs(conf - acc)
    return float(ece)


def mean(values: list[Any]) -> Optional[float]:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return float(sum(cleaned) / len(cleaned))


def episode_cost(
    ep: EpisodeMetrics,
    *,
    damage_weight: float = 8.0,
    timeout_penalty: float = 25.0,
    failure_penalty: float = 25.0,
) -> float:
    return (
        float(ep.steps)
        + damage_weight * float(ep.damage)
        + (timeout_penalty if ep.timeout else 0.0)
        + (failure_penalty if not ep.success else 0.0)
    )
