from __future__ import annotations

import unittest

from risky_maze.scenarios import (
    HUGE_RISKY_GEM_MAZE_V0_SPEC,
    MINI_EXPLORE_LOOP_V0_SPEC,
    MINI_RISK_GATE_V0_SPEC,
    MINI_WAYPOINT_BOTTLENECK_V0_SPEC,
    TUTOR_AUTONOMY_LOOP_V1_SPEC,
    TUTOR_PRINCIPLE_DOOR_TRANSFER_V1_SPEC,
    TUTOR_SAFETY_SCAFFOLD_GATE_V1_SPEC,
    validate_fixed_map_spec,
)


class FixedMapSpecTests(unittest.TestCase):
    def test_huge_fixed_map_spec_has_no_validation_errors(self) -> None:
        report = validate_fixed_map_spec(HUGE_RISKY_GEM_MAZE_V0_SPEC)
        self.assertEqual(report["errors"], [])

    def test_huge_fixed_map_summary_matches_expected_counts(self) -> None:
        report = validate_fixed_map_spec(HUGE_RISKY_GEM_MAZE_V0_SPEC)
        summary = report["summary"]
        self.assertEqual(summary["width"], 61)
        self.assertEqual(summary["height"], 39)
        self.assertEqual(summary["passable_cells"], 1367)
        self.assertEqual(summary["symbol_counts"]["D"], 5)
        self.assertEqual(summary["symbol_counts"]["K"], 4)
        self.assertEqual(summary["symbol_counts"]["g"], 7)
        self.assertEqual(summary["symbol_counts"]["E"], 2)
        self.assertEqual(summary["symbol_counts"]["m"], 3)

    def test_huge_fixed_map_reports_expected_start_warning(self) -> None:
        report = validate_fixed_map_spec(HUGE_RISKY_GEM_MAZE_V0_SPEC)
        self.assertTrue(
            any("E04_SW_to_EastGem_SE_exit" in warning for warning in report["warnings"])
        )

    def test_all_mini_specs_have_no_validation_errors(self) -> None:
        for spec in (
            MINI_RISK_GATE_V0_SPEC,
            MINI_EXPLORE_LOOP_V0_SPEC,
            MINI_WAYPOINT_BOTTLENECK_V0_SPEC,
            TUTOR_SAFETY_SCAFFOLD_GATE_V1_SPEC,
            TUTOR_AUTONOMY_LOOP_V1_SPEC,
            TUTOR_PRINCIPLE_DOOR_TRANSFER_V1_SPEC,
        ):
            with self.subTest(spec=spec["name"]):
                report = validate_fixed_map_spec(spec)
                self.assertEqual(report["errors"], [])

    def test_mini_specs_have_expected_size_and_task_counts(self) -> None:
        for spec in (
            MINI_RISK_GATE_V0_SPEC,
            MINI_EXPLORE_LOOP_V0_SPEC,
            MINI_WAYPOINT_BOTTLENECK_V0_SPEC,
            TUTOR_SAFETY_SCAFFOLD_GATE_V1_SPEC,
            TUTOR_AUTONOMY_LOOP_V1_SPEC,
            TUTOR_PRINCIPLE_DOOR_TRANSFER_V1_SPEC,
        ):
            with self.subTest(spec=spec["name"]):
                report = validate_fixed_map_spec(spec)
                summary = report["summary"]
                self.assertEqual(summary["width"], 31)
                self.assertEqual(summary["height"], 19)
                self.assertEqual(summary["n_teach_tasks"], 3)
                self.assertEqual(summary["n_eval_same_map_tasks"], 3)


if __name__ == "__main__":
    unittest.main()
