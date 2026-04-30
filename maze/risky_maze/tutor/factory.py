from __future__ import annotations

from typing import Any

from .baselines import AlwaysWaypointTutor, RiskThresholdWarnTutor
from .inverse_planner import (
    FullInverseTutor,
    SafetyScaffoldTutor,
    SafetyShieldOnlyTutor,
    TutorConfig,
    WarningOnlyInverseTutor,
)
from .profiles import default_profiles
from .rollout import RolloutConfig


def build_inverse_tutor(name: str, config: Any = None) -> Any:
    """Factory extension for risky_maze/tutor/warning_policies.py.

    Usage inside the existing build_tutor():

        from risky_maze.tutor.factory import build_inverse_tutor
        if name in {...}: return build_inverse_tutor(name, config)
    """
    key = str(name).lower()
    if key in {"risk_threshold", "risk_threshold_warn", "risk_threshold_warning", "threshold_warn"}:
        return RiskThresholdWarnTutor()
    if key in {"always_waypoint", "always_oracle_waypoint", "waypoint_oracle"}:
        return AlwaysWaypointTutor()
    rollout_horizon = int(getattr(config, "tutor_rollout_horizon", 6) or 6) if config is not None else 6
    top_k_paths = int(getattr(config, "tutor_top_k_paths", 2) or 2) if config is not None else 2
    max_candidates = int(getattr(config, "tutor_max_candidates", 8) or 8) if config is not None else 8
    profile_count = int(getattr(config, "tutor_profile_count", 3) or 3) if config is not None else 3
    waypoint_cooldown_steps = (
        int(getattr(config, "tutor_waypoint_cooldown_steps", 6) or 6) if config is not None else 6
    )
    max_waypoints_per_episode = (
        int(getattr(config, "tutor_max_waypoints_per_episode", 3) or 3) if config is not None else 3
    )
    profiles = default_profiles()[: max(1, profile_count)]
    tutor_cfg = TutorConfig(
        top_k_paths=top_k_paths,
        rollout_horizon=rollout_horizon,
        max_candidates=max_candidates,
        waypoint_cooldown_steps=waypoint_cooldown_steps,
        max_waypoints_per_episode=max_waypoints_per_episode,
        frontier_only_waypoint=bool(getattr(config, "tutor_frontier_only_waypoint", False)) if config is not None else False,
        warning_actionability_threshold=float(getattr(config, "tutor_warning_actionability_threshold", 0.0)) if config is not None else 0.0,
        waypoint_damage_veto_margin=float(getattr(config, "tutor_waypoint_damage_veto_margin", float("inf"))) if config is not None else float("inf"),
        safety_shield_enabled=bool(getattr(config, "tutor_safety_shield_enabled", False)) if config is not None else False,
        catastrophe_threshold=float(getattr(config, "tutor_catastrophe_threshold", 0.35)) if config is not None else 0.35,
        rollout=RolloutConfig(
            horizon=rollout_horizon,
            catastrophe_damage_threshold=float(getattr(config, "tutor_catastrophe_damage_threshold", 2.0)) if config is not None else 2.0,
        ),
    )
    if "safety_shield" in key:
        tutor_cfg.safety_shield_enabled = True
    if "frontier_only" in key:
        tutor_cfg.frontier_only_waypoint = True
        tutor_cfg.scaffold_waypoint_types = ("frontier",)
    if key in {"warning_only_inverse", "inverse_plan_warn_only", "inverse_warning_rollout", "inverse_wait_warning"}:
        tutor_cfg.mode = "warning_only"
        return WarningOnlyInverseTutor(tutor_cfg, profiles=profiles)
    if key in {"warning_only_safety_shield", "inverse_plan_warn_only_safety_shield"}:
        tutor_cfg.mode = "warning_only"
        tutor_cfg.safety_shield_enabled = True
        return WarningOnlyInverseTutor(tutor_cfg, profiles=profiles)
    if key in {"safety_shield_only", "shield_only", "warning_shield"}:
        tutor_cfg.mode = "safety_shield_only"
        tutor_cfg.safety_shield_enabled = True
        return SafetyShieldOnlyTutor(tutor_cfg, profiles=profiles)
    if key in {"shield_plus_minimal_waypoint", "safety_scaffold", "inverse_safety_scaffold"}:
        tutor_cfg.mode = "shield_plus_minimal_waypoint"
        tutor_cfg.safety_shield_enabled = True
        tutor_cfg.scaffold_waypoint_types = ("frontier", "landmark", "bottleneck")
        return SafetyScaffoldTutor(tutor_cfg, profiles=profiles)
    if key in {"shield_plus_frontier_waypoint", "safety_scaffold_frontier_only"}:
        tutor_cfg.mode = "shield_plus_minimal_waypoint"
        tutor_cfg.safety_shield_enabled = True
        tutor_cfg.frontier_only_waypoint = True
        tutor_cfg.scaffold_waypoint_types = ("frontier",)
        return SafetyScaffoldTutor(tutor_cfg, profiles=profiles)
    if key in {"shield_plus_random_frontier_waypoint", "safety_scaffold_random_frontier"}:
        tutor_cfg.mode = "shield_plus_minimal_waypoint"
        tutor_cfg.safety_shield_enabled = True
        tutor_cfg.frontier_only_waypoint = True
        tutor_cfg.scaffold_waypoint_types = ("frontier",)
        tutor_cfg.randomize_scaffold_choice = True
        return SafetyScaffoldTutor(tutor_cfg, profiles=profiles)
    if key in {"shield_plus_oracle_when_needed", "shield_plus_oracle_waypoint", "safety_scaffold_oracle"}:
        tutor_cfg.mode = "shield_plus_minimal_waypoint"
        tutor_cfg.safety_shield_enabled = True
        tutor_cfg.scaffold_waypoint_types = ("oracle",)
        return SafetyScaffoldTutor(tutor_cfg, profiles=profiles)
    if key in {"full_inverse", "inverse_plan_full", "inverse_planning", "inverse_wait_warning_waypoint"}:
        tutor_cfg.mode = "full"
        return FullInverseTutor(tutor_cfg, profiles=profiles)
    if key in {"inverse_plan_full_frontier_only"}:
        tutor_cfg.mode = "full"
        tutor_cfg.frontier_only_waypoint = True
        return FullInverseTutor(tutor_cfg, profiles=profiles)
    raise ValueError(f"Unknown inverse tutor policy name: {name}")
