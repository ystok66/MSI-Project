"""Fixed-map block runner for HugeRiskyGemMaze_v0."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from risky_maze.env.fixed_loader import (
    build_layout_from_spec,
    build_task_from_spec,
    list_task_ids,
    load_fixed_spec,
)
from risky_maze.env.pomdp_episode import RiskyMazePOMDPEnv
from risky_maze.learner.objective_agent import ObjectiveAwareLearner
from risky_maze.runner.fixed_episode_runner import run_fixed_episode
from risky_maze.runner.fixed_metrics import BlockMetrics, aggregate_block_metrics


@dataclass(slots=True)
class ExperimentMode:
    layout_mode: Literal["random", "fixed"] = "fixed"
    fixed_spec_name: str | None = "HugeRiskyGemMaze_v0"
    teach_task_ids: list[str] | None = None
    eval_task_ids: list[str] | None = None
    baseline_mode: str = "mortal"
    tutor_name: str = "no_tutor"
    eval_tutor_off: bool = True


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

    spec = load_fixed_spec(spec_name)
    layout = build_layout_from_spec(spec, config)
    teach_ids = teach_task_ids or list_task_ids(spec, "teach")
    eval_ids = eval_task_ids or list_task_ids(spec, "eval_same_map")

    # One learner persists across teach and eval_same_map so map memory and risk
    # concept learning can be reused.  Eval disables tutor interventions below.
    risk_dim = int(getattr(config, "risk_dim", 6)) if config is not None else 6
    learner = ObjectiveAwareLearner(risk_dim=risk_dim)

    teach_metrics = []
    for i, task_id in enumerate(teach_ids):
        task = build_task_from_spec(spec, "teach", task_id)
        env = RiskyMazePOMDPEnv(
            layout=layout,
            task=task,
            config=config,
            seed=seed + 1000 + i,
            prototype_seed=seed,
            baseline_mode=baseline_mode,
            phase="teach",
        )
        teach_metrics.append(
            run_fixed_episode(
                env,
                learner,
                tutor_name=tutor_name,
                tutor_off=(tutor_name in {"no_tutor", "none", "wait"}),
                seed=seed + 1000 + i,
            )
        )

    eval_metrics = []
    for i, task_id in enumerate(eval_ids):
        task = build_task_from_spec(spec, "eval_same_map", task_id)
        env = RiskyMazePOMDPEnv(
            layout=layout,
            task=task,
            config=config,
            seed=seed + 2000 + i,
            prototype_seed=seed,
            baseline_mode=baseline_mode,
            phase="eval_same_map",
        )
        eval_metrics.append(
            run_fixed_episode(
                env,
                learner,
                tutor_name="no_tutor",
                tutor_off=True,
                seed=seed + 2000 + i,
            )
        )

    # Risk dataset from any fixed env with the same layout/prototypes.
    if teach_ids:
        probe_task = build_task_from_spec(spec, "teach", teach_ids[0])
    else:
        probe_task = build_task_from_spec(spec, "eval_same_map", eval_ids[0])
    probe_env = RiskyMazePOMDPEnv(
        layout=layout,
        task=probe_task,
        config=config,
        seed=seed + 9999,
        prototype_seed=seed,
        baseline_mode=baseline_mode,
        phase="probe",
    )
    risk_dataset = probe_env.risk_eval_dataset(observed_noise=False)

    block: BlockMetrics = aggregate_block_metrics(
        teach_metrics,
        eval_metrics,
        learner=learner,
        layout=layout,
        risk_dataset=risk_dataset,
    )
    return block.as_dict()
