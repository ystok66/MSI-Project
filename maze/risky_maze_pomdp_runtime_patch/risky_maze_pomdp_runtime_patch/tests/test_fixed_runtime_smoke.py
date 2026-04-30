"""Smoke tests for the fixed-map POMDP runtime.

These tests assume the existing HugeRiskyGemMaze_v0 scenario asset is present in
``risky_maze/scenarios``.
"""

from risky_maze.runner.fixed_block_runner import run_fixed_block
from risky_maze.env.fixed_loader import load_fixed_spec, build_layout_from_spec, build_task_from_spec
from risky_maze.env.pomdp_episode import RiskyMazePOMDPEnv, RuntimeAction


def test_fixed_map_all_tasks_run_no_tutor():
    result = run_fixed_block(
        spec_name="HugeRiskyGemMaze_v0",
        tutor_name="no_tutor",
        baseline_mode="mortal",
        seed=0,
    )
    assert "teach" in result
    assert "eval_same_map" in result
    assert "aggregate" in result
    assert "useful_exploration_rate" in result["aggregate"]
    assert "map_reuse_eval" in result["aggregate"]


def test_fixed_map_all_tasks_run_immortal_warnlike():
    result = run_fixed_block(
        spec_name="HugeRiskyGemMaze_v0",
        tutor_name="no_tutor",
        baseline_mode="immortal_warnlike",
        seed=1,
    )
    assert "teach" in result
    assert "eval_same_map" in result


def test_fixed_map_all_tasks_run_immortal_no_timeout():
    result = run_fixed_block(
        spec_name="HugeRiskyGemMaze_v0",
        tutor_name="no_tutor",
        baseline_mode="immortal_no_timeout",
        seed=2,
    )
    assert "teach" in result
    assert "eval_same_map" in result


def test_objective_sequence_advances_on_coordinates():
    spec = load_fixed_spec("HugeRiskyGemMaze_v0")
    teach_ids = spec.task_ids("teach")
    assert teach_ids
    task = build_task_from_spec(spec, "teach", teach_ids[0])
    layout = build_layout_from_spec(spec)
    env = RiskyMazePOMDPEnv(layout=layout, task=task, seed=0, prototype_seed=0)
    obs = env.reset(seed=0)
    assert obs.current_objective == task.objectives[0]
    # Directly placing state is acceptable in a unit test of the objective
    # machine; learner observations still never expose true trap labels.
    first = task.objectives[0]
    assert env.state is not None
    env.state.pos = first.coord
    env.objective_state.update(first.coord, layout, env.inventory)
    assert env.objective_state.index >= 1


def test_observation_does_not_expose_trap_symbols():
    spec = load_fixed_spec("HugeRiskyGemMaze_v0")
    task = build_task_from_spec(spec, "teach", spec.task_ids("teach")[0])
    env = RiskyMazePOMDPEnv(layout=build_layout_from_spec(spec), task=task, seed=0, prototype_seed=0)
    obs = env.reset(seed=0)
    hidden_labels = {"r", "m", "q"}
    visible_kinds = {cell.visible_kind for cell in obs.visible_cells.values()}
    assert not (visible_kinds & hidden_labels)
