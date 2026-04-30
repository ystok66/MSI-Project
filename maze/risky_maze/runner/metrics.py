from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeMetrics:
    success: float = 0.0
    died: float = 0.0
    timeout: float = 0.0
    steps: float = 0.0
    damage: float = 0.0
    warnings: float = 0.0
    discovered_cells: float = 0.0
    repeated_steps: float = 0.0


def merge_episode_metrics(metrics: list[EpisodeMetrics]) -> dict[str, float]:
    if not metrics:
        return {}
    n = float(len(metrics))
    return {
        "success_rate": sum(m.success for m in metrics) / n,
        "death_rate": sum(m.died for m in metrics) / n,
        "timeout_rate": sum(m.timeout for m in metrics) / n,
        "mean_steps": sum(m.steps for m in metrics) / n,
        "mean_damage": sum(m.damage for m in metrics) / n,
        "mean_warnings": sum(m.warnings for m in metrics) / n,
        "mean_discovered_cells": sum(m.discovered_cells for m in metrics) / n,
        "mean_repeated_steps": sum(m.repeated_steps for m in metrics) / n,
    }
