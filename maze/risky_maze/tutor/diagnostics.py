from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class TutorDecisionLog:
    step: int
    selected_action: str
    candidate_count: int
    q_wait: float = 0.0
    q_best_warning: float = float("-inf")
    q_best_waypoint: float = float("-inf")
    predicted_p_death_wait: float = 0.0
    predicted_p_timeout_wait: float = 0.0
    predicted_map_gain_wait: float = 0.0
    predicted_risk_ig_warning: float = 0.0
    predicted_assist_leakage: float = 0.0
    actual_next_damage: float = 0.0
    actual_next_new_cells: int = 0
    actual_warning_ig: float = 0.0
    actual_progress: float = 0.0
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass
class TutorEpisodeDiagnostics:
    decisions: list[TutorDecisionLog] = field(default_factory=list)

    def append(self, log: TutorDecisionLog) -> None:
        self.decisions.append(log)

    def update_last_actual(
        self,
        actual_next_damage: float = 0.0,
        actual_next_new_cells: int = 0,
        actual_warning_ig: float = 0.0,
        actual_progress: float = 0.0,
    ) -> None:
        if not self.decisions:
            return
        last = self.decisions[-1]
        last.actual_next_damage = actual_next_damage
        last.actual_next_new_cells = actual_next_new_cells
        last.actual_warning_ig = actual_warning_ig
        last.actual_progress = actual_progress

    def episode_summary(self) -> dict[str, float]:
        n = len(self.decisions)
        warnings = sum(1 for d in self.decisions if d.selected_action == "WARNING")
        waypoints = sum(1 for d in self.decisions if d.selected_action == "WAYPOINT")
        leakage = sum(d.predicted_assist_leakage for d in self.decisions)
        warning_ig = sum(d.actual_warning_ig for d in self.decisions)
        map_gain_after_wait = sum(d.actual_next_new_cells for d in self.decisions if d.selected_action == "WAIT")
        actual_damage = sum(d.actual_next_damage for d in self.decisions)
        predicted_death_wait = sum(d.predicted_p_death_wait for d in self.decisions)
        warning_actionability = sum(float(d.diagnostics.get("warning_actionability", 0.0)) for d in self.decisions)
        safety_shield = sum(float(d.diagnostics.get("safety_shield_triggered", 0.0)) for d in self.decisions)
        return {
            "tutor_decisions": float(n),
            "warnings_per_episode": float(warnings),
            "waypoints_per_episode": float(waypoints),
            "assist_leakage": float(leakage),
            "warning_information_gain": float(warning_ig),
            "warning_actionability": float(warning_actionability) / max(1, warnings),
            "safety_shield_triggers": float(safety_shield),
            "map_gain_after_wait": float(map_gain_after_wait),
            "actual_damage_after_decisions": float(actual_damage),
            "mean_predicted_p_death_wait": predicted_death_wait / max(1, n),
        }
