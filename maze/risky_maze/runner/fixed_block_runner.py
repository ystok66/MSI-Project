"""Fixed-map block runner for HugeRiskyGemMaze_v0."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from risky_maze.env.fixed_loader import (
    build_layout_from_spec,
    build_task_from_spec,
    list_task_ids,
    load_fixed_spec,
)
from risky_maze.env.pomdp_episode import RiskyMazePOMDPEnv
from risky_maze.learner.objective_agent import ObjectiveAwareLearner
from risky_maze.runner.fixed_episode_runner import _maybe_build_overlay_tutor, run_fixed_episode
from risky_maze.runner.fixed_metrics import BlockMetrics, EpisodeMetrics, aggregate_block_metrics


def _scaled_task_time_limit(task: Any, scale: float) -> Any:
    factor = float(scale)
    if abs(factor - 1.0) <= 1e-9:
        return task
    scaled = max(1, int(round(float(task.time_limit) * factor)))
    return replace(task, time_limit=scaled)


@dataclass(slots=True)
class ExperimentMode:
    layout_mode: Literal["random", "fixed"] = "fixed"
    fixed_spec_name: str | None = "HugeRiskyGemMaze_v0"
    teach_task_ids: list[str] | None = None
    eval_task_ids: list[str] | None = None
    baseline_mode: str = "mortal"
    tutor_name: str = "no_tutor"
    eval_tutor_off: bool = True


@dataclass(slots=True)
class FixedBlockRun:
    spec_name: str
    tutor_name: str
    baseline_mode: str
    seed: int
    teach: list[EpisodeMetrics]
    eval_same_map: list[EpisodeMetrics]
    aggregate: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec_name": self.spec_name,
            "tutor_name": self.tutor_name,
            "baseline_mode": self.baseline_mode,
            "seed": self.seed,
            "teach": [ep.as_dict() for ep in self.teach],
            "eval_same_map": [ep.as_dict() for ep in self.eval_same_map],
            "aggregate": dict(self.aggregate),
        }


def run_fixed_block(
    config: Any | None = None,
    *,
    spec_name: str = "HugeRiskyGemMaze_v0",
    teach_task_ids: list[str] | None = None,
    eval_task_ids: list[str] | None = None,
    tutor_name: str = "no_tutor",
    baseline_mode: str = "mortal",
    seed: int = 0,
) -> dict[str, Any]:
    """Run teach tasks then same-map eval tasks on a fixed spec.

    Returns a plain dict for compatibility with existing smoke tests:

    ``{"teach": [...], "eval_same_map": [...], "aggregate": {...}}``.
    """
    return run_fixed_block_detailed(
        config=config,
        spec_name=spec_name,
        teach_task_ids=teach_task_ids,
        eval_task_ids=eval_task_ids,
        tutor_name=tutor_name,
        baseline_mode=baseline_mode,
        seed=seed,
    ).as_dict()


def run_fixed_block_detailed(
    config: Any | None = None,
    *,
    spec_name: str = "HugeRiskyGemMaze_v0",
    teach_task_ids: list[str] | None = None,
    eval_task_ids: list[str] | None = None,
    tutor_name: str = "no_tutor",
    baseline_mode: str = "mortal",
    seed: int = 0,
) -> FixedBlockRun:
    """Return rich per-episode results for experiment scripts."""

    spec = load_fixed_spec(spec_name)
    layout = build_layout_from_spec(spec, config)
    teach_ids = teach_task_ids or list_task_ids(spec, "teach")
    eval_ids = eval_task_ids or list_task_ids(spec, "eval_same_map")

    # One learner persists across teach and eval_same_map so map memory and risk
    # concept learning can be reused.  Eval disables tutor interventions below.
    risk_dim = int(getattr(config, "risk_dim", 6)) if config is not None else 6
    learner = ObjectiveAwareLearner(
        risk_dim=risk_dim,
        risk_weight=float(getattr(config, "learner_risk_weight", 4.0)) if config is not None else 4.0,
        revisit_penalty=float(getattr(config, "learner_revisit_penalty", 0.15)) if config is not None else 0.15,
        unknown_penalty=float(getattr(config, "learner_unknown_penalty", 0.20)) if config is not None else 0.20,
        warning_suspicion_weight=float(getattr(config, "learner_warning_suspicion_weight", 2.0)) if config is not None else 2.0,
        warning_suspicion_mode=str(getattr(config, "learner_warning_suspicion_mode", "persistent")) if config is not None else "persistent",
        warning_suspicion_decay=float(getattr(config, "learner_warning_suspicion_decay", 1.0)) if config is not None else 1.0,
        consolidation_mode=str(getattr(config, "learner_consolidation_mode", "none")) if config is not None else "none",
        long_term_memory_weight=float(getattr(config, "learner_long_term_memory_weight", 0.35)) if config is not None else 0.35,
        autonomy_assist_discount=float(getattr(config, "learner_autonomy_assist_discount", 0.05)) if config is not None else 0.05,
        enable_objective_learning_events=bool(getattr(config, "learner_enable_objective_learning_events", True)) if config is not None else True,
        use_long_term_route_graph=bool(getattr(config, "learner_use_long_term_route_graph", True)) if config is not None else True,
        use_landmark_graph=bool(getattr(config, "learner_use_landmark_graph", True)) if config is not None else True,
        warning_update_mode=str(getattr(config, "warning_update_mode", "effective_sample")) if config is not None else "effective_sample",
        warning_eta0=float(getattr(config, "warning_eta0", 0.35)) if config is not None else 0.35,
        warning_kl_epsilon=float(getattr(config, "warning_kl_epsilon", 1e-6)) if config is not None else 1e-6,
        enable_warning_update=not bool(getattr(config, "ablate_warning_update", False)) if config is not None else True,
        enable_trap_risk_update=not bool(getattr(config, "ablate_trap_risk_update", False)) if config is not None else True,
        enable_safe_risk_update=not bool(getattr(config, "ablate_safe_risk_update", False)) if config is not None else True,
    )
    shared_inverse_tutor = _maybe_build_overlay_tutor(tutor_name, config)
    dataset_source_env: RiskyMazePOMDPEnv | None = None

    teach_metrics = []
    for i, task_id in enumerate(teach_ids):
        task = build_task_from_spec(spec, "teach", task_id)
        task = _scaled_task_time_limit(
            task,
            float(getattr(config, "teach_time_limit_scale", 1.0)) if config is not None else 1.0,
        )
        env = RiskyMazePOMDPEnv(
            layout=layout,
            task=task,
            config=config,
            seed=seed + 1000 + i,
            prototype_seed=seed,
            baseline_mode=baseline_mode,
            phase="teach",
        )
        dataset_source_env = env
        teach_metrics.append(
            run_fixed_episode(
                env,
                learner,
                config=config,
                tutor_name=tutor_name,
                tutor_off=(tutor_name in {"no_tutor", "none", "wait"}),
                seed=seed + 1000 + i,
                inverse_tutor=shared_inverse_tutor,
            )
        )

    eval_metrics = []
    eval_learner = learner.clone_for_eval(
        clear_memory=bool(getattr(config, "ablate_eval_clear_map_memory", False)) if config is not None else False,
        clear_risk_belief=bool(getattr(config, "ablate_eval_clear_risk_belief", False)) if config is not None else False,
        clear_warning_suspicion=bool(getattr(config, "ablate_eval_clear_warning_suspicion", False)) if config is not None else False,
        clear_long_term_memory=bool(getattr(config, "ablate_eval_clear_long_term_memory", False)) if config is not None else False,
    )
    # Keep a frozen eval-start clone for risk probes so ablations affect the
    # reported belief, while eval-phase observations do not shift the probe.
    risk_probe_learner = eval_learner.clone_for_eval(clear_memory=False, clear_risk_belief=False)
    for i, task_id in enumerate(eval_ids):
        task = build_task_from_spec(spec, "eval_same_map", task_id)
        task = _scaled_task_time_limit(
            task,
            float(getattr(config, "eval_time_limit_scale", 1.0)) if config is not None else 1.0,
        )
        env = RiskyMazePOMDPEnv(
            layout=layout,
            task=task,
            config=config,
            seed=seed + 2000 + i,
            prototype_seed=seed,
            baseline_mode=baseline_mode,
            phase="eval_same_map",
        )
        dataset_source_env = env
        eval_metrics.append(
            run_fixed_episode(
                env,
                eval_learner,
                config=config,
                tutor_name="no_tutor",
                tutor_off=True,
                seed=seed + 2000 + i,
            )
        )

    risk_dataset = dataset_source_env.risk_eval_dataset(observed_noise=False) if dataset_source_env is not None else None
    risk_coords = list(layout.walkable_cells())

    block: BlockMetrics = aggregate_block_metrics(
        teach_metrics,
        eval_metrics,
        learner=risk_probe_learner,
        teach_learner=learner,
        eval_probe_learner=risk_probe_learner,
        layout=layout,
        risk_dataset=risk_dataset,
        risk_coords=risk_coords,
    )
    return FixedBlockRun(
        spec_name=spec_name,
        tutor_name=tutor_name,
        baseline_mode=baseline_mode,
        seed=seed,
        teach=teach_metrics,
        eval_same_map=eval_metrics,
        aggregate=block.aggregate,
    )
