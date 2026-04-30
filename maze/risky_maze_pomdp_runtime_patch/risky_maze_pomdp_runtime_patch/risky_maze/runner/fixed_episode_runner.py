"""Episode loop for the fixed POMDP runtime."""

from __future__ import annotations

from typing import Any

from risky_maze.env.pomdp_episode import RiskyMazePOMDPEnv
from risky_maze.learner.objective_agent import ObjectiveAwareLearner
from risky_maze.runner.fixed_metrics import EpisodeMetrics, oracle_safe_shortest_path


def run_fixed_episode(
    env: RiskyMazePOMDPEnv,
    learner: ObjectiveAwareLearner,
    *,
    tutor_name: str = "no_tutor",
    tutor_off: bool = False,
    seed: int | None = None,
    warning_prefix_len: int = 5,
    inverse_info_credit: float = 0.30,
    inverse_warning_cost: float = 0.10,
) -> EpisodeMetrics:
    """Run one fixed task using learner-only observations.

    Tutors may inspect the hidden env, as in the old prototype, but eval calls
    should pass ``tutor_off=True`` to force no intervention.
    """

    obs = env.reset(seed=seed)
    metrics = EpisodeMetrics(task_id=env.task.task_id, phase=env.phase, path=[obs.pos])
    metrics.discovered_walkable.update(
        coord for coord, cell in obs.visible_cells.items() if cell.is_walkable_observed
    )
    oracle = oracle_safe_shortest_path(env.layout, env.task.start, env.task.objectives)
    metrics.oracle_safe_steps = oracle

    terminated = truncated = False
    while not (terminated or truncated):
        action = learner.act(obs)

        if not tutor_off:
            warning_coords = _warning_coords(env, learner, tutor_name, warning_prefix_len, inverse_info_credit, inverse_warning_cost)
            if warning_coords:
                features = env.features_for(warning_coords, observed_noise=True)
                diagnostics = learner.apply_warning(warning_coords, features)
                metrics.warnings += 1
                metrics.warning_information_gain += diagnostics.get("mean_abs_delta", 0.0)
                metrics.posterior_shift_after_warning += diagnostics.get("sum_delta", 0.0)

        obs, outcome, terminated, truncated, info = env.step(action)
        del info
        learner.observe(obs)
        learner.apply_outcome(outcome)

        if outcome.immortal_danger_event and outcome.danger_feature is not None:
            # Upper-bound no-tutor learning signal for immortal baselines.
            learner.risk_belief.update_labeled(outcome.danger_feature, True, weight=1.0)

        metrics.steps = obs.step_count
        metrics.damage += outcome.damage
        metrics.died = metrics.died or outcome.died
        metrics.timeout = metrics.timeout or outcome.timeout
        metrics.success = outcome.success
        metrics.repeated_steps += int(outcome.repeated_step)
        metrics.blocked_steps += int(outcome.blocked)
        metrics.no_info_steps += int(outcome.no_info_step)
        metrics.frontier_progress_steps += int(outcome.discovered_new_cells > 0)
        metrics.discovered_cells += outcome.discovered_new_cells
        metrics.path.append(obs.pos)
        metrics.discovered_walkable.update(
            coord for coord, cell in obs.visible_cells.items() if cell.is_walkable_observed
        )

        # Hard guard against bad custom configs when immortal_no_timeout is used.
        if env.baseline_mode == "immortal_no_timeout" and env.phase == "teach":
            max_guard = max(env.task.time_limit * 5, env.layout.width * env.layout.height * 4)
            if metrics.steps >= max_guard:
                break

    if env.phase.startswith("eval") and oracle is not None:
        metrics.eval_regret_to_oracle_safe_path = metrics.steps - oracle
    return metrics


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
