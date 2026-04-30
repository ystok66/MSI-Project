from __future__ import annotations

from typing import Any, Iterable

from .compat import Coord, action_key, make_tutor_action
from .path_predictor import PredictedPath, _danger_probability
from .world_model import (
    degree,
    feature_at,
    is_known_to_learner,
    is_true_danger,
    known_walkable,
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


def generate_waypoint_candidates(context: Any, predicted_paths: list[PredictedPath], max_candidates: int = 10) -> list[Any]:
    layout = context.true_layout
    state = context.true_env_state
    memory = context.learner_memory_snapshot
    obs = context.learner_observation
    actions: list[Any] = []
    cur_obj = objective_coord(layout, state)

    def add_waypoint(c: Coord | None, reason: str, leakage: float) -> None:
        if c is None:
            return
        # Do not reveal hidden shortcuts or hidden arbitrary cells.  A waypoint is
        # allowed only if the learner has seen/known it, or it is already the
        # public/current objective visible in the learner state.
        if not is_known_to_learner(memory, obs, c):
            return
        actions.append(make_tutor_action("WAYPOINT", waypoint=c, reason=reason, diagnostics={"base_assist_leakage": leakage}))

    add_waypoint(cur_obj, "current_objective_if_known", 1.0)

    # Nearest visible/known frontier: useful when the learner is looping or making
    # no progress, but less oracle-like than pointing at the hidden route.
    frontiers = reachable_known_frontiers(layout, memory)
    if frontiers and cur_obj is not None:
        frontiers.sort(key=lambda c: abs(c[0] - cur_obj[0]) + abs(c[1] - cur_obj[1]))
    elif frontiers:
        frontiers.sort()
    for c in frontiers[:4]:
        add_waypoint(c, "nearest_known_frontier", 0.55)

    # Known junctions or bottlenecks close to the predicted route can break loops
    # without giving a full oracle path.
    kw = list(known_walkable(memory))
    if cur_obj is not None:
        kw.sort(key=lambda c: abs(c[0] - cur_obj[0]) + abs(c[1] - cur_obj[1]))
    for c in kw:
        d = degree(layout, c)
        if d >= 3 or d == 1:
            if true_damage(layout, c) <= 0.0:
                add_waypoint(c, "known_junction_or_bottleneck", 0.65)
        if len(actions) >= max_candidates:
            break

    # Safe detour entrance: first known non-danger cell on an oracle-safe path to objective.
    if cur_obj is not None:
        try:
            from .world_model import current_pos, is_walkable

            start = current_pos(state)
            route = shortest_path(
                layout,
                start,
                cur_obj,
                traversable=lambda c: is_walkable(layout, c) and true_damage(layout, c) <= 0.0,
            )
            for c in route[1:8]:
                if is_known_to_learner(memory, obs, c):
                    add_waypoint(c, "known_safe_detour_entrance", 0.75)
                    break
        except Exception:
            pass

    return _unique_actions(actions, limit=max_candidates)


def generate_tutor_candidates(
    context: Any,
    predicted_paths: list[PredictedPath],
    allow_waypoint: bool = True,
    max_candidates: int = 20,
) -> list[Any]:
    actions: list[Any] = [make_tutor_action("WAIT", reason="default_wait")]
    actions.extend(generate_warning_candidates(predicted_paths, context.true_layout))
    actions.extend(generate_visible_risky_cluster_candidate(context))
    if allow_waypoint:
        actions.extend(generate_waypoint_candidates(context, predicted_paths, max_candidates=10))
    return _unique_actions(actions, limit=max_candidates)
