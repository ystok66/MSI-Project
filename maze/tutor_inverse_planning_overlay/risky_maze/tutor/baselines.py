from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .compat import get_any, make_tutor_action
from .context import coerce_context
from .path_predictor import LearnerPathPredictor
from .profiles import LearnerProfile
from .shadow import clone_from_snapshots
from .world_model import current_pos, is_walkable, objective_coord, shortest_path, true_damage


@dataclass
class RiskThresholdWarnConfig:
    threshold: float = 0.5
    prefix_len: int = 5
    risk_aggregation: str = "sum"  # "sum" or "max"


class RiskThresholdWarnTutor:
    """Baseline: warn when true risk on predicted prefix exceeds a threshold."""

    def __init__(self, config: RiskThresholdWarnConfig | None = None):
        self.config = config or RiskThresholdWarnConfig()
        self.predictor = LearnerPathPredictor()
        self.profile = LearnerProfile("threshold_balanced", 4.0, 0.45, 0.2, 0.15, 2.0)

    def act(self, *args: Any, **kwargs: Any) -> Any:
        ctx = coerce_context(*args, **kwargs)
        if str(ctx.phase).lower() == "eval":
            return make_tutor_action("WAIT", reason="eval_phase_disabled")
        shadow = clone_from_snapshots(ctx.learner_memory_snapshot, ctx.learner_risk_belief_snapshot, self.profile, env_state=ctx.true_env_state, layout=ctx.true_layout)
        paths = self.predictor.predict_topk(shadow, ctx.true_env_state, layout=ctx.true_layout, k=1, horizon=self.config.prefix_len)
        if not paths:
            return make_tutor_action("WAIT", reason="no_predicted_path")
        prefix = tuple(paths[0].cells[: self.config.prefix_len])
        risks = [true_damage(ctx.true_layout, c) for c in prefix]
        score = max(risks) if self.config.risk_aggregation == "max" else sum(risks)
        if score > self.config.threshold:
            return make_tutor_action("WARNING", cells=prefix, reason="risk_threshold_prefix", diagnostics={"true_prefix_risk": float(score)})
        return make_tutor_action("WAIT", reason="risk_below_threshold", diagnostics={"true_prefix_risk": float(score)})


@dataclass
class AlwaysWaypointConfig:
    stride: int = 5
    safe_route: bool = True


class AlwaysWaypointTutor:
    """Over-help ceiling: points to the next waypoint on an oracle route.

    This intentionally uses oracle layout and should not be treated as a fair
    tutoring policy; it is a ceiling / leakage stress-test baseline.
    """

    def __init__(self, config: AlwaysWaypointConfig | None = None):
        self.config = config or AlwaysWaypointConfig()

    def act(self, *args: Any, **kwargs: Any) -> Any:
        ctx = coerce_context(*args, **kwargs)
        if str(ctx.phase).lower() == "eval":
            return make_tutor_action("WAIT", reason="eval_phase_disabled")
        try:
            start = current_pos(ctx.true_env_state)
        except Exception:
            return make_tutor_action("WAIT", reason="no_position")
        goal = objective_coord(ctx.true_layout, ctx.true_env_state)
        if goal is None:
            return make_tutor_action("WAIT", reason="no_objective")
        route = shortest_path(
            ctx.true_layout,
            start,
            goal,
            traversable=lambda c: is_walkable(ctx.true_layout, c) and (true_damage(ctx.true_layout, c) <= 0.0 if self.config.safe_route else True),
        )
        if len(route) <= 1:
            return make_tutor_action("WAIT", reason="on_oracle_goal_or_no_route")
        idx = min(max(1, self.config.stride), len(route) - 1)
        wp = route[idx]
        return make_tutor_action("WAYPOINT", waypoint=wp, reason="oracle_route_waypoint", diagnostics={"assist_leakage": 1.0})
