from __future__ import annotations

from typing import Any

from .baselines import AlwaysWaypointTutor, RiskThresholdWarnTutor
from .inverse_planner import FullInverseTutor, TutorConfig, WarningOnlyInverseTutor


def build_inverse_tutor(name: str, config: Any = None) -> Any:
    """Factory extension for risky_maze/tutor/warning_policies.py.

    Usage inside the existing build_tutor():

        from risky_maze.tutor.factory import build_inverse_tutor
        if name in {...}: return build_inverse_tutor(name, config)
    """
    key = str(name).lower()
    if key in {"risk_threshold", "risk_threshold_warning", "threshold_warn"}:
        return RiskThresholdWarnTutor()
    if key in {"always_waypoint", "waypoint_oracle"}:
        return AlwaysWaypointTutor()
    if key in {"warning_only_inverse", "inverse_warning_rollout", "inverse_wait_warning"}:
        return WarningOnlyInverseTutor(TutorConfig(mode="warning_only"))
    if key in {"full_inverse", "inverse_planning", "inverse_wait_warning_waypoint"}:
        return FullInverseTutor(TutorConfig(mode="full"))
    raise ValueError(f"Unknown inverse tutor policy name: {name}")
