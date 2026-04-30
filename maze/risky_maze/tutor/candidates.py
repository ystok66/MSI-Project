from __future__ import annotations

from typing import Any, Iterable

from .compat import Coord, action_key, make_tutor_action
from .path_predictor import PredictedPath, _danger_probability
from .world_model import (
    degree,
    feature_at,
    is_known_to_learner,
    is_walkable,
    known_walkable,
    current_pos,
    objective_coord,
    observed_vector,
    reachable_known_frontiers,
    shortest_path,
    true_damage,
    visible_cells,
)


def _unique_actions(actions: Iterable[Any], limit: int) -> list[Any]:
    seen: set[tuple[str, tuple[Coord, ...], Coord | None]] = set()
    out: list[Any] = []
    for a in actions:
        key = action_key(a)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
        if len(out) >= limit:
            break
    return out


def generate_warning_candidates(
    predicted_paths: list[PredictedPath],
    true_layout: Any,
    max_set_size: int = 5,
    prefix_lengths: tuple[int, ...] = (3, 5),
) -> list[Any]:
    actions: list[Any] = []
    for rank, path in enumerate(predicted_paths):
        if not path.cells:
            continue
        for k in prefix_lengths:
            prefix = tuple(path.cells[: min(k, max_set_size, len(path.cells))])
            if prefix:
                actions.append(
                    make_tutor_action(
                        "WARNING",
                        cells=prefix,
                        reason=f"predicted_path_prefix_k{k}_rank{rank}",
                        diagnostics={"path_probability": path.probability},
                    )
                )

        # Prefix until first junction, capped.  This remains set-level evidence;
        # it does not label a specific trap cell.
        prefix_cells: list[Coord] = []
        for c in path.cells[:max_set_size]:
            prefix_cells.append(c)
            if degree(true_layout, c) != 2:
                break
        if prefix_cells:
            actions.append(
                make_tutor_action(
                    "WARNING",
                    cells=tuple(prefix_cells),
                    reason=f"prefix_until_junction_rank{rank}",
                    diagnostics={"path_probability": path.probability},
                )
            )
    return _unique_actions(actions, limit=8)


def generate_visible_risky_cluster_candidate(context: Any, max_set_size: int = 5, risk_threshold: float = 0.55) -> list[Any]:
    obs = context.learner_observation
    memory = context.learner_memory_snapshot
    belief = context.learner_risk_belief_snapshot
    layout = context.true_layout
    scored: list[tuple[float, Coord]] = []
    for c in visible_cells(obs):
        feat = observed_vector(memory, layout, c, allow_oracle=False)
        if feat is None:
            feat = feature_at(layout, c) if is_known_to_learner(memory, obs, c) else None
        p = _danger_probability(belief, feat, default=0.0)
        if p >= risk_threshold:
            scored.append((p, c))
    if not scored:
        return []
    scored.sort(reverse=True)
    cells = tuple(c for _, c in scored[:max_set_size])
    return [
        make_tutor_action(
            "WARNING",
            cells=cells,
            reason="visible_risky_cluster",
            diagnostics={"max_belief_risk": scored[0][0]},
        )
    ]


def _progress_gift(start: Coord | None, current_objective: Coord | None, coord: Coord | None) -> float:
    if coord is None or current_objective is None or start is None:
        return 0.0
    base = abs(start[0] - current_objective[0]) + abs(start[1] - current_objective[1])
    if base <= 0:
        return 0.0
    after = abs(coord[0] - current_objective[0]) + abs(coord[1] - current_objective[1])
    return max(0.0, float(base - after) / float(base))


def _make_waypoint_action(
    *,
    context: Any,
    coord: Coord | None,
    reason: str,
    leakage: float,
    novelty_leak: float,
    waypoint_type: str,
) -> Any | None:
    if coord is None:
        return None
    memory = context.learner_memory_snapshot
    obs = context.learner_observation
    if not is_known_to_learner(memory, obs, coord):
        return None
    start = current_pos(context.true_env_state)
    cur_obj = objective_coord(context.true_layout, context.true_env_state)
    gift = _progress_gift(start, cur_obj, coord)
    return make_tutor_action(
        "WAYPOINT",
        waypoint=coord,
        reason=reason,
        diagnostics={
            "base_assist_leakage": leakage,
            "waypoint_progress_gift": gift,
            "waypoint_novelty_leak": novelty_leak,
            "waypoint_type": waypoint_type,
            "waypoint_visible_to_learner": 1.0,
            "waypoint_is_frontier": 1.0 if waypoint_type == "frontier" else 0.0,
            "waypoint_is_landmark": 1.0 if waypoint_type == "landmark" else 0.0,
            "waypoint_is_bottleneck": 1.0 if waypoint_type == "bottleneck" else 0.0,
            "waypoint_is_oracle": 1.0 if waypoint_type == "oracle" else 0.0,
        },
    )


def generate_frontier_waypoints(context: Any, max_candidates: int = 4) -> list[Any]:
    layout = context.true_layout
    state = context.true_env_state
    memory = context.learner_memory_snapshot
    cur_obj = objective_coord(layout, state)
    frontiers = reachable_known_frontiers(layout, memory)
    if cur_obj is not None:
        frontiers.sort(key=lambda c: abs(c[0] - cur_obj[0]) + abs(c[1] - cur_obj[1]))
    else:
        frontiers.sort()
    actions: list[Any] = []
    for coord in frontiers[:max_candidates]:
        action = _make_waypoint_action(
            context=context,
            coord=coord,
            reason="nearest_known_frontier",
            leakage=0.25,
            novelty_leak=0.25,
            waypoint_type="frontier",
        )
        if action is not None:
            actions.append(action)
    return actions


def generate_landmark_waypoints(context: Any, max_candidates: int = 2) -> list[Any]:
    layout = context.true_layout
    state = context.true_env_state
    cur_obj = objective_coord(layout, state)
    actions: list[Any] = []
    action = _make_waypoint_action(
        context=context,
        coord=cur_obj,
        reason="current_objective_if_known",
        leakage=0.50,
        novelty_leak=0.50,
        waypoint_type="landmark",
    )
    if action is not None:
        actions.append(action)
    return actions[:max_candidates]


def generate_bottleneck_waypoints(context: Any, max_candidates: int = 6) -> list[Any]:
    layout = context.true_layout
    state = context.true_env_state
    memory = context.learner_memory_snapshot
    actions: list[Any] = []
    cur_obj = objective_coord(layout, state)
    kw = list(known_walkable(memory))
    if cur_obj is not None:
        kw.sort(key=lambda c: abs(c[0] - cur_obj[0]) + abs(c[1] - cur_obj[1]))
    for c in kw:
        d = degree(layout, c)
        if d >= 3 or d == 1:
            if true_damage(layout, c) <= 0.0:
                action = _make_waypoint_action(
                    context=context,
                    coord=c,
                    reason="known_junction_or_bottleneck",
                    leakage=0.50,
                    novelty_leak=0.50,
                    waypoint_type="bottleneck",
                )
                if action is not None:
                    actions.append(action)
        if len(actions) >= max_candidates:
            break
    return actions[:max_candidates]


def generate_oracle_waypoints(context: Any, max_candidates: int = 2) -> list[Any]:
    layout = context.true_layout
    state = context.true_env_state
    cur_obj = objective_coord(layout, state)
    try:
        start = current_pos(state)
    except Exception:
        start = None
    actions: list[Any] = []

    # Over-help ablation: choose a known cell that lies on an oracle-safe route.
    # The waypoint itself is still not hidden, but its selection uses oracle route
    # structure and therefore carries higher assist leakage.
    if cur_obj is None or start is None:
        return actions
    try:
        walkable = getattr(layout, "_walkable_coords", None)
        if walkable is not None:
            safe_set = frozenset(c for c in walkable if true_damage(layout, c) <= 0.0)
            safe_trav = lambda c: c in safe_set
        else:
            safe_trav = lambda c: is_walkable(layout, c) and true_damage(layout, c) <= 0.0
        route = shortest_path(layout, start, cur_obj, traversable=safe_trav)
        for c in route[1:8]:
            action = _make_waypoint_action(
                context=context,
                coord=c,
                reason="oracle_safe_detour_entrance",
                leakage=1.00,
                novelty_leak=0.75,
                waypoint_type="oracle",
            )
            if action is not None:
                actions.append(action)
                break
    except Exception:
        return actions
    return actions[:max_candidates]


def generate_waypoint_candidates(
    context: Any,
    predicted_paths: list[PredictedPath],
    max_candidates: int = 10,
    frontier_only: bool = False,
    allowed_types: tuple[str, ...] | None = None,
) -> list[Any]:
    del predicted_paths
    actions: list[Any] = []
    waypoint_types = tuple(allowed_types or ())
    if frontier_only:
        waypoint_types = ("frontier",)
    if not waypoint_types:
        waypoint_types = ("landmark", "frontier", "bottleneck", "oracle")
    for waypoint_type in waypoint_types:
        if waypoint_type == "frontier":
            actions.extend(generate_frontier_waypoints(context))
        elif waypoint_type == "landmark":
            actions.extend(generate_landmark_waypoints(context))
        elif waypoint_type == "bottleneck":
            actions.extend(generate_bottleneck_waypoints(context))
        elif waypoint_type == "oracle":
            actions.extend(generate_oracle_waypoints(context))

    return _unique_actions(actions, limit=max_candidates)


def generate_tutor_candidates(
    context: Any,
    predicted_paths: list[PredictedPath],
    allow_waypoint: bool = True,
    max_candidates: int = 20,
    frontier_only_waypoint: bool = False,
) -> list[Any]:
    actions: list[Any] = [make_tutor_action("WAIT", reason="default_wait")]
    actions.extend(generate_warning_candidates(predicted_paths, context.true_layout))
    actions.extend(generate_visible_risky_cluster_candidate(context))
    if allow_waypoint:
        actions.extend(
            generate_waypoint_candidates(
                context,
                predicted_paths,
                max_candidates=10,
                frontier_only=frontier_only_waypoint,
            )
        )
    return _unique_actions(actions, limit=max_candidates)
