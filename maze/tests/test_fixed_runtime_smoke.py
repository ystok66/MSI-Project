from __future__ import annotations

import unittest

from risky_maze.env.fixed_loader import build_layout_from_spec, build_task_from_spec, load_fixed_spec
from risky_maze.env.pomdp_episode import RiskyMazePOMDPEnv
from risky_maze.runner.fixed_block_runner import run_fixed_block


class FixedRuntimeSmokeTests(unittest.TestCase):
    def _smoke_ids(self) -> tuple[list[str], list[str]]:
        return (["T01_NW_key_gem_NE_exit"], ["E01_WestGarden_to_NE_exit"])

    def _mini_smoke_ids(self) -> tuple[list[str], list[str]]:
        return (["A_T01_key_danger_gate_NE_exit"], ["A_E01_NE_exit_from_left"])

    def _didactic_smoke_cases(self) -> list[tuple[str, list[str], list[str]]]:
        return [
            (
                "TutorSafetyScaffoldGate_v1",
                ["S1_T01_short_risky_gate_vs_safe_detour"],
                ["S1_E01_reuse_safe_detour_to_gem"],
            ),
            (
                "TutorAutonomyLoop_v1",
                ["S2_T01_west_gem_to_ne_exit"],
                ["S2_E01_reverse_east_to_west_gem"],
            ),
            (
                "TutorPrincipleDoorTransfer_v1",
                ["S3_T01_full_key_door_gem_exit"],
                ["S3_E01_reverse_full_transfer"],
            ),
        ]

    def test_fixed_map_all_tasks_run_no_tutor(self) -> None:
        teach_ids, eval_ids = self._smoke_ids()
        result = run_fixed_block(
            spec_name="HugeRiskyGemMaze_v0",
            teach_task_ids=teach_ids,
            eval_task_ids=eval_ids,
            tutor_name="no_tutor",
            baseline_mode="mortal",
            seed=0,
        )
        self.assertIn("teach", result)
        self.assertIn("eval_same_map", result)
        self.assertIn("aggregate", result)
        self.assertIn("useful_exploration_rate", result["aggregate"])
        self.assertIn("map_reuse_eval", result["aggregate"])
        self.assertIn("teach_mean_elapsed_seconds", result["aggregate"])
        self.assertIn("eval_mean_elapsed_seconds", result["aggregate"])
        self.assertIn("teach_cost", result["aggregate"])
        self.assertIn("eval_cost", result["aggregate"])

    def test_fixed_map_all_tasks_run_immortal_warnlike(self) -> None:
        teach_ids, eval_ids = self._smoke_ids()
        result = run_fixed_block(
            spec_name="HugeRiskyGemMaze_v0",
            teach_task_ids=teach_ids,
            eval_task_ids=eval_ids,
            tutor_name="no_tutor",
            baseline_mode="immortal_warnlike",
            seed=1,
        )
        self.assertIn("teach", result)
        self.assertIn("eval_same_map", result)

    def test_fixed_map_all_tasks_run_immortal_no_timeout(self) -> None:
        teach_ids, eval_ids = self._smoke_ids()
        result = run_fixed_block(
            spec_name="HugeRiskyGemMaze_v0",
            teach_task_ids=teach_ids,
            eval_task_ids=eval_ids,
            tutor_name="no_tutor",
            baseline_mode="immortal_no_timeout",
            seed=2,
        )
        self.assertIn("teach", result)
        self.assertIn("eval_same_map", result)

    def test_mini_risk_gate_runs_no_tutor_smoke(self) -> None:
        teach_ids, eval_ids = self._mini_smoke_ids()
        result = run_fixed_block(
            spec_name="MiniRiskGate_v0",
            teach_task_ids=teach_ids,
            eval_task_ids=eval_ids,
            tutor_name="no_tutor",
            baseline_mode="mortal",
            seed=0,
        )
        self.assertIn("teach", result)
        self.assertIn("eval_same_map", result)
        self.assertIn("aggregate", result)

    def test_didactic_suite_specs_run_no_tutor_smoke(self) -> None:
        for spec_name, teach_ids, eval_ids in self._didactic_smoke_cases():
            with self.subTest(spec=spec_name):
                result = run_fixed_block(
                    spec_name=spec_name,
                    teach_task_ids=teach_ids,
                    eval_task_ids=eval_ids,
                    tutor_name="no_tutor",
                    baseline_mode="mortal",
                    seed=0,
                )
                self.assertIn("teach", result)
                self.assertIn("eval_same_map", result)
                self.assertIn("aggregate", result)

    def test_all_fixed_tasks_can_reset(self) -> None:
        spec = load_fixed_spec("HugeRiskyGemMaze_v0")
        layout = build_layout_from_spec(spec)
        for split in ("teach", "eval_same_map"):
            for task_id in spec.task_ids(split):
                task = build_task_from_spec(spec, split, task_id)
                env = RiskyMazePOMDPEnv(layout=layout, task=task, seed=0, prototype_seed=0)
                obs = env.reset(seed=0)
                self.assertEqual(obs.pos, task.start)
                self.assertIsNotNone(obs.current_objective)

    def test_objective_sequence_advances_on_coordinates(self) -> None:
        spec = load_fixed_spec("HugeRiskyGemMaze_v0")
        teach_ids = spec.task_ids("teach")
        self.assertTrue(teach_ids)
        task = build_task_from_spec(spec, "teach", teach_ids[0])
        layout = build_layout_from_spec(spec)
        env = RiskyMazePOMDPEnv(layout=layout, task=task, seed=0, prototype_seed=0)
        obs = env.reset(seed=0)
        self.assertEqual(obs.current_objective, task.objectives[0])
        first = task.objectives[0]
        self.assertIsNotNone(env.state)
        env.state.pos = first.coord
        env.objective_state.update(first.coord, layout, env.inventory)
        self.assertGreaterEqual(env.objective_state.index, 1)

    def test_observation_does_not_expose_trap_symbols(self) -> None:
        spec = load_fixed_spec("HugeRiskyGemMaze_v0")
        task = build_task_from_spec(spec, "teach", spec.task_ids("teach")[0])
        env = RiskyMazePOMDPEnv(layout=build_layout_from_spec(spec), task=task, seed=0, prototype_seed=0)
        obs = env.reset(seed=0)
        hidden_labels = {"r", "m", "q"}
        visible_kinds = {cell.visible_kind for cell in obs.visible_cells.values()}
        self.assertFalse(bool(visible_kinds & hidden_labels))


if __name__ == "__main__":
    unittest.main()
