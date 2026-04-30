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
    discovered_cells: int = 0
    repeated_steps: int = 0
    blocked_steps: int = 0
    path: list[Coord] = field(default_factory=list)
    discovered_walkable: set[Coord] = field(default_factory=set)
    warning_information_gain: float = 0.0
    posterior_shift_after_warning: float = 0.0
    no_info_steps: int = 0
    frontier_progress_steps: int = 0
    oracle_safe_steps: Optional[int] = None
    eval_regret_to_oracle_safe_path: Optional[int] = None

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
    def frontier_progress_rate(self) -> float:
        return self.frontier_progress_steps / max(1, self.steps)

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
            "discovered_cells": self.discovered_cells,
            "repeated_steps": self.repeated_steps,
            "blocked_steps": self.blocked_steps,
            "warning_information_gain": self.warning_information_gain,
            "posterior_shift_after_warning": self.posterior_shift_after_warning,
            "no_info_step_rate": self.no_info_step_rate,
            "loop_rate": self.loop_rate,
            "repeated_known_cell_rate": self.repeated_known_cell_rate,
            "frontier_progress_rate": self.frontier_progress_rate,
            "oracle_safe_steps": self.oracle_safe_steps,
            "eval_regret_to_oracle_safe_path": self.eval_regret_to_oracle_safe_path,
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
    layout: FixedRuntimeLayout,
    risk_dataset: tuple[np.ndarray, np.ndarray] | None = None,
) -> BlockMetrics:
    teach_new_cells: set[Coord] = set().union(*(ep.discovered_walkable for ep in teach_eps)) if teach_eps else set()
    eval_used_cells: set[Coord] = set().union(*(set(ep.path) for ep in eval_eps)) if eval_eps else set()
    useful_exploration_rate = len(teach_new_cells & eval_used_cells) / max(1, len(teach_new_cells))
    map_reuse_eval = len(eval_used_cells & teach_new_cells) / max(1, len(eval_used_cells))

    risk_auc = None
    risk_nll = None
    risk_calibration_ece = None
    if risk_dataset is not None and learner is not None and hasattr(learner, "danger_probability"):
        xs, ys = risk_dataset
        probs = np.array([float(learner.danger_probability(x)) for x in xs], dtype=float)
        risk_auc = binary_auc(ys, probs)
        risk_nll = binary_nll(ys, probs)
        risk_calibration_ece = binary_ece(ys, probs)

    all_eps = teach_eps + eval_eps
    aggregate = {
        "success_rate": mean([ep.success for ep in all_eps]),
        "teach_success_rate": mean([ep.success for ep in teach_eps]),
        "eval_success_rate": mean([ep.success for ep in eval_eps]),
        "death_rate": mean([ep.died for ep in all_eps]),
        "timeout_rate": mean([ep.timeout for ep in all_eps]),
        "mean_steps": mean([ep.steps for ep in all_eps]),
        "mean_damage": mean([ep.damage for ep in all_eps]),
        "warnings": int(sum(ep.warnings for ep in all_eps)),
        "map_coverage_teach": len(teach_new_cells) / max(1, len(layout.walkable_cells())),
        "map_reuse_eval": map_reuse_eval,
        "useful_exploration_rate": useful_exploration_rate,
        "warning_information_gain": mean([ep.warning_information_gain for ep in all_eps]),
        "posterior_shift_after_warning": mean([ep.posterior_shift_after_warning for ep in all_eps]),
        "risk_auc": risk_auc,
        "risk_nll": risk_nll,
        "risk_calibration_ece": risk_calibration_ece,
        "eval_regret_to_oracle_safe_path": mean([
            ep.eval_regret_to_oracle_safe_path for ep in eval_eps if ep.eval_regret_to_oracle_safe_path is not None
        ]),
        "loop_rate": mean([ep.loop_rate for ep in all_eps]),
        "repeated_known_cell_rate": mean([ep.repeated_known_cell_rate for ep in all_eps]),
        "no_info_step_rate": mean([ep.no_info_step_rate for ep in all_eps]),
        "frontier_progress_rate": mean([ep.frontier_progress_rate for ep in all_eps]),
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
