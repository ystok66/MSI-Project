"""Episode loop for the fixed POMDP runtime."""

from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
from typing import Any

from risky_maze.env.pomdp_episode import RiskyMazePOMDPEnv
from risky_maze.learner.objective_agent import ObjectiveAwareLearner
from risky_maze.runner.fixed_metrics import EpisodeMetrics, oracle_safe_shortest_path
from risky_maze.tutor.context import EpisodeHistory, TutorDecisionContext
from risky_maze.tutor.diagnostics import TutorDecisionLog
from risky_maze.tutor.factory import build_inverse_tutor


_ORACLE_SAFE_PATH_CACHE: dict[tuple[tuple[str, ...], tuple[int, int], tuple[tuple[str, tuple[int, int]], ...]], int | None] = {}


def run_fixed_episode(
    env: RiskyMazePOMDPEnv,
    learner: ObjectiveAwareLearner,
    *,
    config: Any | None = None,
    tutor_name: str = "no_tutor",
    tutor_off: bool = False,
    seed: int | None = None,
    warning_prefix_len: int = 5,
    inverse_info_credit: float = 0.30,
    inverse_warning_cost: float = 0.10,
    inverse_tutor: Any | None = None,
) -> EpisodeMetrics:
    """Run one fixed task using learner-only observations.

    Tutors may inspect the hidden env, as in the old prototype, but eval calls
    should pass ``tutor_off=True`` to force no intervention.
    """

    obs = env.reset(seed=seed)
    metrics = EpisodeMetrics(task_id=env.task.task_id, phase=env.phase, path=[obs.pos])
    started = time.perf_counter()
    history = EpisodeHistory()
    learning_events: list[dict[str, Any]] = []
    seen_learning_events: set[tuple[str, tuple[int, int]]] = set()
    inverse_tutor = inverse_tutor if inverse_tutor is not None else _maybe_build_overlay_tutor(tutor_name, config)
    if inverse_tutor is not None and hasattr(inverse_tutor, "reset_episode"):
        try:
            inverse_tutor.reset_episode()
        except Exception:
            pass
    elif inverse_tutor is not None and hasattr(inverse_tutor, "reset"):
        try:
            inverse_tutor.reset()
        except Exception:
            pass
    record_step_details = bool(getattr(config, "record_step_details", True)) if config is not None else True
    metrics.discovered_walkable.update(
        coord for coord, cell in obs.visible_cells.items() if cell.is_walkable_observed
    )
    oracle = _cached_oracle_safe_shortest_path(env.layout, env.task.start, env.task.objectives)
    metrics.oracle_safe_steps = oracle

    terminated = truncated = False
    while not (terminated or truncated):
        pos_before = obs.pos
        action = learner.act(obs)
        action_before_assist = getattr(action, "name", str(action))
        plan_before_assist = list(getattr(learner, "last_plan", []))
        warning_ig = 0.0
        warning_ess = 0.0
        warning_kl = 0.0
        assist = None
        warning_before_has_trap = 0.0
        warning_after_has_trap = 0.0
        warned_next_cell_is_trap = 0.0
        post_warning_action_changed = 0.0
        warning_actionable = 0.0

        if not tutor_off:
            assist = _select_tutor_action(
                env=env,
                learner=learner,
                obs=obs,
                tutor_name=tutor_name,
                inverse_tutor=inverse_tutor,
                history=history,
                action=action,
                warning_prefix_len=warning_prefix_len,
                inverse_info_credit=inverse_info_credit,
                inverse_warning_cost=inverse_warning_cost,
            )
            if assist.kind == "WARNING":
                warning_coords = list(assist.cells)
                warning_before_has_trap = 1.0 if any(env.true_is_trap(c) for c in warning_coords) else 0.0
                warned_next_cell_is_trap = (
                    1.0 if plan_before_assist and len(plan_before_assist) > 1 and env.true_is_trap(plan_before_assist[1]) else 0.0
                )
                features = env.features_for(warning_coords, observed_noise=True)
                diagnostics = learner.apply_warning(warning_coords, features)
                metrics.warnings += 1
                metrics.warning_information_gain += diagnostics.get("mean_abs_delta", 0.0)
                metrics.posterior_shift_after_warning += diagnostics.get("sum_delta", 0.0)
                warning_ess = diagnostics.get("warning_ess", 0.0)
                warning_kl = diagnostics.get("warning_kl", 0.0)
                metrics.warning_effective_sample_size += warning_ess
                metrics.warning_kl += warning_kl
                warning_ig = diagnostics.get("mean_abs_delta", 0.0)
                action = learner.act(obs)
                plan_after_warning = list(getattr(learner, "last_plan", []))
                warning_after_has_trap = (
                    1.0 if plan_after_warning and any(env.true_is_trap(c) for c in plan_after_warning[1 : 1 + warning_prefix_len]) else 0.0
                )
                post_warning_action_changed = 1.0 if getattr(action, "name", str(action)) != action_before_assist else 0.0
                warning_actionable = 1.0 if warning_before_has_trap > 0.0 and warning_after_has_trap <= 0.0 else 0.0
                metrics.warning_action_total += 1
                metrics.warning_actionable_count += int(warning_actionable)
            elif assist.kind == "WAYPOINT" and assist.waypoint is not None:
                learner.set_waypoint(tuple(assist.waypoint))
                metrics.waypoints += 1
                waypoint_diag = dict(getattr(assist, "diagnostics", {}) or {})
                metrics.waypoint_progress_gift += float(waypoint_diag.get("waypoint_progress_gift", 0.0) or 0.0)
                metrics.waypoint_novelty_leak += float(waypoint_diag.get("waypoint_novelty_leak", 0.0) or 0.0)
                action = learner.act(obs)
        else:
            from risky_maze.tutor.compat import make_tutor_action

            assist = make_tutor_action("WAIT", reason="tutor_off")

        obs, outcome, terminated, truncated, info = env.step(action)
        del info
        learner.observe(obs)
        risk_belief_before_outcome = learner.risk_belief.clone()
        learner.apply_outcome(outcome)
        actual_risk_update = _risk_belief_delta(risk_belief_before_outcome, learner.risk_belief)

        metrics.steps = obs.step_count
        metrics.damage += outcome.damage
        metrics.died = metrics.died or outcome.died
        metrics.timeout = metrics.timeout or outcome.timeout
        metrics.success = outcome.success
        metrics.trap_entries += int(outcome.trap_coord is not None)
        metrics.immortal_danger_events += int(outcome.immortal_danger_event)
        metrics.objective_completed_count = max(metrics.objective_completed_count, env.objective_state.index)
        metrics.repeated_steps += int(outcome.repeated_step)
        metrics.blocked_steps += int(outcome.blocked)
        metrics.no_info_steps += int(outcome.no_info_step)
        metrics.frontier_progress_steps += int(outcome.discovered_new_cells > 0)
        metrics.discovered_cells += outcome.discovered_new_cells
        metrics.path.append(obs.pos)
        for event in tuple(getattr(outcome, "learning_events", ()) or ()):
            event_key = (str(event.kind), tuple(event.coord))
            if event_key in seen_learning_events:
                continue
            seen_learning_events.add(event_key)
            learning_events.append(
                {
                    "kind": str(event.kind),
                    "coord": tuple(event.coord),
                    "objective_advanced": bool(getattr(event, "objective_advanced", False)),
                    "picked_key": bool(outcome.objective_event.picked_key),
                    "passed_door": bool(getattr(outcome.objective_event, "passed_door", False)),
                    "collected_gem": tuple(outcome.objective_event.collected_gem) if outcome.objective_event.collected_gem is not None else None,
                    "reached_exit": bool(getattr(outcome.objective_event, "reached_exit", False)),
                }
            )
        metrics.discovered_walkable.update(
            coord for coord, cell in obs.visible_cells.items() if cell.is_walkable_observed
        )
        decision_row = _decision_row(
            assist=assist,
            inverse_tutor=inverse_tutor,
            step_index=int(obs.step_count),
            phase=str(env.phase),
            pos_before=pos_before,
        )
        decision_row.update(
            {
                "warning_before_has_trap": warning_before_has_trap,
                "warning_after_has_trap": warning_after_has_trap,
                "warned_next_cell_is_trap": warned_next_cell_is_trap,
                "post_warning_action_changed": post_warning_action_changed,
                "warning_actionable": warning_actionable,
            }
        )
        metrics.assist_leakage += float(decision_row.get("assist_leakage", 0.0) or 0.0)
        if decision_row["selected_action"] == "WAIT":
            if outcome.discovered_new_cells > 0 and not outcome.died and not outcome.timeout:
                metrics.useful_wait_count += 1
            if outcome.died or outcome.no_info_step or outcome.repeated_step:
                metrics.bad_wait_count += 1
        if decision_row["selected_action"] == "WAYPOINT":
            metrics.map_gain_after_waypoint += int(outcome.discovered_new_cells)
            metrics.risk_ig_after_waypoint += float(actual_risk_update)
            metrics.predicted_risk_ig_after_waypoint += float(
                decision_row.get("diag_expected_risk_info_gain", 0.0) or 0.0
            )
        if decision_row["selected_action"] == "WARNING":
            metrics.warning_path_changed_count += int(post_warning_action_changed)
        metrics.safety_shield_triggered += int(float(decision_row.get("diag_safety_shield_triggered", 0.0) or 0.0) > 0.0)
        if (
            decision_row["selected_action"] == "WAIT"
            and outcome.died
            and float(decision_row.get("diag_safe_alternative_exists", 0.0) or 0.0) > 0.0
        ):
            metrics.preventable_death_count += 1
        if record_step_details:
            metrics.tutor_decisions.append(decision_row)
            metrics.step_records.append(
                {
                    "phase": str(env.phase),
                    "step": int(obs.step_count),
                    "task_id": env.task.task_id,
                    "pos_before": pos_before,
                    "pos_after": obs.pos,
                    "action": getattr(action, "name", str(action)),
                    "assist_kind": decision_row["selected_action"],
                    "assist_reason": decision_row.get("reason", ""),
                    "warning_cells": tuple(decision_row.get("warning_cells", ()) or ()),
                    "waypoint": decision_row.get("waypoint"),
                    "damage": int(outcome.damage),
                    "died": bool(outcome.died),
                    "timeout": bool(outcome.timeout),
                    "success": bool(outcome.success),
                    "immortal_danger_event": bool(outcome.immortal_danger_event),
                    "discovered_new_cells": int(outcome.discovered_new_cells),
                    "repeated_step": bool(outcome.repeated_step),
                    "no_info_step": bool(outcome.no_info_step),
                    "objective_advanced": bool(outcome.objective_event.advanced),
                    "objective_completed_count": int(env.objective_state.index),
                    "warning_information_gain": float(warning_ig),
                    "warning_effective_sample_size": float(warning_ess),
                    "warning_kl": float(warning_kl),
                    "warning_actionable": float(warning_actionable),
                    "post_warning_action_changed": float(post_warning_action_changed),
                    "warning_before_has_trap": float(warning_before_has_trap),
                    "warning_after_has_trap": float(warning_after_has_trap),
                    "hp": int(obs.hp),
                    "remaining_time": int(max(0, obs.time_limit - obs.step_count)),
                }
            )
        history.append(
            learner_action=getattr(action, "name", str(action)),
            pos=obs.pos,
            damage=outcome.damage,
            new_cells=outcome.discovered_new_cells,
            warning_information_gain=warning_ig,
        )
        if inverse_tutor is not None and hasattr(inverse_tutor, "diagnostics"):
            try:
                inverse_tutor.diagnostics.update_last_actual(
                    actual_next_damage=float(outcome.damage),
                    actual_next_new_cells=int(outcome.discovered_new_cells),
                    actual_warning_ig=float(warning_ig),
                    actual_progress=float(outcome.objective_event.advanced),
                )
            except Exception:
                pass

        # Hard guard against bad custom configs when immortal_no_timeout is used.
        if env.baseline_mode == "immortal_no_timeout" and env.phase == "teach":
            max_guard = max(env.task.time_limit * 5, env.layout.width * env.layout.height * 4)
            if metrics.steps >= max_guard:
                break

    if env.phase.startswith("eval") and oracle is not None:
        metrics.eval_regret_to_oracle_safe_path = metrics.steps - oracle
    metrics.elapsed_seconds = max(0.0, time.perf_counter() - started)
    finalize = getattr(learner, "finalize_episode", None)
    if callable(finalize):
        finalize(
            phase=str(env.phase),
            success=bool(metrics.success),
            path=list(metrics.path),
            learning_events=learning_events,
            assist_leakage=float(metrics.assist_leakage),
            waypoint_count=int(metrics.waypoints),
            warning_count=int(metrics.warnings),
            waypoint_progress_gift=float(metrics.waypoint_progress_gift),
            waypoint_novelty_leak=float(metrics.waypoint_novelty_leak),
        )
    return metrics


def _select_tutor_action(
    *,
    env: RiskyMazePOMDPEnv,
    learner: ObjectiveAwareLearner,
    obs: Any,
    tutor_name: str,
    inverse_tutor: Any,
    history: EpisodeHistory,
    action: Any,
    warning_prefix_len: int,
    inverse_info_credit: float,
    inverse_warning_cost: float,
) -> Any:
    if inverse_tutor is not None:
        phase = "eval" if str(env.phase).startswith("eval") else "teach"
        context = TutorDecisionContext(
            true_env_state=SimpleNamespace(
                pos=obs.pos,
                hp=obs.hp,
                step_count=obs.step_count,
                time_limit=obs.time_limit,
                has_key=obs.has_key,
                current_objective=obs.current_objective,
                objective_index=obs.objective_index,
                objective_sequence=obs.objective_sequence,
                has_gem=bool(obs.has_gem_or_collected),
            ),
            true_layout=env.layout,
            learner_observation=obs,
            learner_memory_snapshot=learner.memory.clone(),
            learner_risk_belief_snapshot=learner.risk_belief.clone(),
            learner_policy_snapshot=SimpleNamespace(
                action=getattr(action, "name", str(action)),
                planned_path=list(getattr(learner, "last_plan", [])),
                first_action=getattr(action, "name", str(action)),
            ),
            history=history,
            phase=phase,
            remaining_time=max(0, obs.time_limit - obs.step_count),
        )
        return inverse_tutor.act(context)

    warning_coords = _warning_coords(
        env,
        learner,
        tutor_name,
        warning_prefix_len,
        inverse_info_credit,
        inverse_warning_cost,
    )
    if warning_coords:
        from risky_maze.tutor.compat import make_tutor_action

        return make_tutor_action("WARNING", cells=warning_coords, reason="legacy_warning_policy")
    from risky_maze.tutor.compat import make_tutor_action

    return make_tutor_action("WAIT", reason="legacy_wait")


def _maybe_build_overlay_tutor(tutor_name: str, config: Any | None = None) -> Any | None:
    key = str(tutor_name).lower()
    overlay_names = {
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
        "inverse_plan_full_frontier_only",
        "inverse_planning",
        "inverse_wait_warning_waypoint",
    }
    if key not in overlay_names:
        return None
    if key == "inverse_plan_warn_only":
        key = "warning_only_inverse"
    if key == "inverse_plan_full":
        key = "full_inverse"
    return build_inverse_tutor(key, config)


def _decision_row(
    *,
    assist: Any,
    inverse_tutor: Any,
    step_index: int,
    phase: str,
    pos_before: tuple[int, int],
) -> dict[str, Any]:
    row = {
        "phase": phase,
        "step": step_index,
        "pos": pos_before,
        "selected_action": getattr(assist, "kind", "WAIT"),
        "warning_cells": tuple(getattr(assist, "cells", ()) or ()),
        "waypoint": getattr(assist, "waypoint", None),
        "reason": getattr(assist, "reason", ""),
        "q_wait": None,
        "q_best_warning": None,
        "q_best_waypoint": None,
        "predicted_p_death_wait": None,
        "predicted_p_timeout_wait": None,
        "predicted_map_gain_wait": None,
        "predicted_risk_ig_warning": None,
        "assist_leakage": float((getattr(assist, "diagnostics", {}) or {}).get("assist_leakage", 0.0)),
        "selected_action_reason": getattr(assist, "reason", ""),
        "decision_layer": "baseline" if inverse_tutor is None else "unknown",
        "waypoint_type": (getattr(assist, "diagnostics", {}) or {}).get("waypoint_type"),
        "waypoint_progress_gift": float((getattr(assist, "diagnostics", {}) or {}).get("waypoint_progress_gift", 0.0) or 0.0),
        "waypoint_novelty_leak": float((getattr(assist, "diagnostics", {}) or {}).get("waypoint_novelty_leak", 0.0) or 0.0),
    }
    if inverse_tutor is None or not hasattr(inverse_tutor, "diagnostics"):
        return row
    try:
        decisions = getattr(inverse_tutor.diagnostics, "decisions", [])
        if not decisions:
            return row
        last = decisions[-1]
        log = _decision_log_to_dict(last)
        row.update(
            {
                "selected_action": log["selected_action"],
                "q_wait": log["q_wait"],
                "q_best_warning": log["q_best_warning"],
                "q_best_waypoint": log["q_best_waypoint"],
                "predicted_p_death_wait": log["predicted_p_death_wait"],
                "predicted_p_timeout_wait": log["predicted_p_timeout_wait"],
                "predicted_map_gain_wait": log["predicted_map_gain_wait"],
                "predicted_risk_ig_warning": log["predicted_risk_ig_warning"],
                "assist_leakage": log["predicted_assist_leakage"],
            }
        )
        row.update({k: v for k, v in log.items() if k.startswith("diag_")})
        if float(row.get("diag_decision_layer_safety", 0.0) or 0.0) > 0.0:
            row["decision_layer"] = "safety"
        elif float(row.get("diag_decision_layer_scaffold", 0.0) or 0.0) > 0.0:
            row["decision_layer"] = "scaffold"
        if row.get("diag_waypoint_type") and not row.get("waypoint_type"):
            row["waypoint_type"] = row.get("diag_waypoint_type")
    except Exception:
        return row
    return row


def _decision_log_to_dict(log: TutorDecisionLog | Any) -> dict[str, Any]:
    if is_dataclass(log):
        base = asdict(log)
    else:
        base = dict(getattr(log, "__dict__", {}))
    diagnostics = dict(base.pop("diagnostics", {}) or {})
    for key, value in diagnostics.items():
        base[f"diag_{key}"] = value
    return base


def _cached_oracle_safe_shortest_path(
    layout: Any,
    start: tuple[int, int],
    objectives: list[Any],
) -> int | None:
    key = (
        tuple(getattr(layout, "rows", ()) or ()),
        tuple(start),
        tuple(
            (
                str(getattr(obj, "kind", "")),
                tuple(getattr(obj, "coord", ())),
            )
            for obj in objectives
        ),
    )
    if key not in _ORACLE_SAFE_PATH_CACHE:
        _ORACLE_SAFE_PATH_CACHE[key] = oracle_safe_shortest_path(layout, start, objectives)
    return _ORACLE_SAFE_PATH_CACHE[key]


def _risk_belief_delta(before: Any, after: Any) -> float:
    try:
        delta = 0.0
        delta += abs(float(getattr(after, "prior_danger", 0.0)) - float(getattr(before, "prior_danger", 0.0)))
        for name in ("safe_mean", "danger_mean"):
            a = getattr(after, name, None)
            b = getattr(before, name, None)
            if a is not None and b is not None:
                delta += float(abs((a - b)).sum())
        delta += abs(float(getattr(after, "safe_count", 0.0)) - float(getattr(before, "safe_count", 0.0)))
        delta += abs(float(getattr(after, "danger_count", 0.0)) - float(getattr(before, "danger_count", 0.0)))
        return float(delta)
    except Exception:
        return 0.0


def _warning_coords(
    env: RiskyMazePOMDPEnv,
    learner: ObjectiveAwareLearner,
    tutor_name: str,
    warning_prefix_len: int,
    inverse_info_credit: float,
    inverse_warning_cost: float,
) -> list[tuple[int, int]]:
    if tutor_name in {"no_tutor", "none", "wait"}:
        return []
    path = list(getattr(learner, "last_plan", []))
    if len(path) <= 1:
        return []
    prefix = path[1 : 1 + warning_prefix_len]
    true_traps = [coord for coord in prefix if env.true_is_trap(coord)]
    if not true_traps:
        return []
    if tutor_name in {"always_warn", "always_warning"}:
        return prefix
    if tutor_name in {"inverse_warn", "inverse_warning"}:
        # Lightweight version matching the old heuristic: warn when hidden risk
        # dominates expected info value and message cost, or the next trap could
        # be catastrophic.
        true_risk = sum(env.layout.trap_damage(c) for c in true_traps)
        info_gain = sum(1.0 for c in prefix if c not in learner.memory.known_walkable)
        hp = env.state.hp if env.state is not None else 1
        catastrophic = any(env.layout.trap_damage(c) >= hp for c in true_traps)
        warn_score = true_risk - inverse_info_credit * info_gain - inverse_warning_cost
        if catastrophic or warn_score > 0.0:
            return prefix
        return []
    raise ValueError(f"Unsupported tutor_name={tutor_name!r}")
