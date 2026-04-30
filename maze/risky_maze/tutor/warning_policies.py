from __future__ import annotations

from dataclasses import dataclass

from ..config import MazeScenarioConfig
from ..core import Observation, PolicySnapshot, Pos, TutorAction
from ..env import MazeEpisode
from ..learner import LearnerAgent
from .base import TutorPolicy
from .factory import build_inverse_tutor


def _warning_prefix(
    snapshot: PolicySnapshot,
    obs: Observation,
    horizon: int,
) -> tuple[Pos, ...]:
    prefix: list[Pos] = []
    visible = {cell.pos for cell in obs.visible_cells if cell.walkable}
    for pos in snapshot.planned_path[1 : horizon + 1]:
        if pos in visible:
            prefix.append(pos)
    return tuple(prefix)


class NoTutor(TutorPolicy):
    name = "no_tutor"

    def maybe_intervene(
        self,
        episode: MazeEpisode,
        learner: LearnerAgent,
        obs: Observation,
        snapshot: PolicySnapshot,
    ) -> TutorAction:
        return TutorAction(kind="WAIT")


class AlwaysWarnTutor(TutorPolicy):
    name = "always_warn"

    def __init__(self, cfg: MazeScenarioConfig) -> None:
        self.cfg = cfg

    def maybe_intervene(
        self,
        episode: MazeEpisode,
        learner: LearnerAgent,
        obs: Observation,
        snapshot: PolicySnapshot,
    ) -> TutorAction:
        prefix = _warning_prefix(snapshot, obs, self.cfg.warning_horizon)
        if not prefix:
            return TutorAction(kind="WAIT")
        if any(episode.layout.trap_type_at(pos) > 0 for pos in prefix):
            return TutorAction(kind="WARNING", cells=prefix)
        return TutorAction(kind="WAIT")


@dataclass
class InverseWarnTutor(TutorPolicy):
    cfg: MazeScenarioConfig
    name: str = "inverse_warn"

    def maybe_intervene(
        self,
        episode: MazeEpisode,
        learner: LearnerAgent,
        obs: Observation,
        snapshot: PolicySnapshot,
    ) -> TutorAction:
        prefix = _warning_prefix(snapshot, obs, self.cfg.warning_horizon)
        if not prefix:
            return TutorAction(kind="WAIT")

        true_risk = float(sum(episode.layout.trap_type_at(pos) for pos in prefix))
        info_gain = float(sum(1 for pos in prefix if learner.memory.visited(pos) == 0))
        lethal_soon = any(episode.layout.trap_type_at(pos) >= obs.hp for pos in prefix)
        warn_score = true_risk - self.cfg.tutor_info_credit * info_gain - self.cfg.tutor_warning_cost

        if lethal_soon or warn_score > self.cfg.tutor_risk_budget:
            return TutorAction(kind="WARNING", cells=prefix)
        return TutorAction(kind="WAIT")


def build_tutor(name: str, cfg: MazeScenarioConfig) -> TutorPolicy:
    if name == "no_tutor":
        return NoTutor()
    if name == "always_warn":
        return AlwaysWarnTutor(cfg)
    if name == "inverse_warn":
        return InverseWarnTutor(cfg)
    inverse_names = {
        "risk_threshold_warn",
        "risk_threshold_warning",
        "threshold_warn",
        "always_waypoint",
        "always_oracle_waypoint",
        "waypoint_oracle",
        "warning_only_inverse",
        "inverse_plan_warn_only",
        "inverse_warning_rollout",
        "inverse_wait_warning",
        "warning_only_safety_shield",
        "inverse_plan_warn_only_safety_shield",
        "safety_shield_only",
        "shield_only",
        "warning_shield",
        "shield_plus_minimal_waypoint",
        "safety_scaffold",
        "inverse_safety_scaffold",
        "shield_plus_random_frontier_waypoint",
        "safety_scaffold_random_frontier",
        "shield_plus_frontier_waypoint",
        "safety_scaffold_frontier_only",
        "shield_plus_oracle_when_needed",
        "shield_plus_oracle_waypoint",
        "safety_scaffold_oracle",
        "full_inverse",
        "inverse_plan_full",
        "inverse_planning",
        "inverse_wait_warning_waypoint",
    }
    if name in inverse_names:
        return build_inverse_tutor(name, cfg)
    raise ValueError(f"unknown tutor policy: {name}")
