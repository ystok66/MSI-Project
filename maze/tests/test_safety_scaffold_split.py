from __future__ import annotations

import unittest

from risky_maze.tutor.compat import make_tutor_action
from risky_maze.tutor.factory import build_inverse_tutor
from risky_maze.tutor.inverse_planner import InversePlanningTutor, TutorConfig
from risky_maze.tutor.rollout import TutorActionValue


class SafetyScaffoldSplitTests(unittest.TestCase):
    def test_safety_layer_prefers_warning_when_wait_is_catastrophic(self) -> None:
        tutor = InversePlanningTutor(TutorConfig(mode="safety_shield_only", catastrophe_threshold=0.2))
        wait = make_tutor_action("WAIT", reason="wait")
        warning = make_tutor_action("WARNING", cells=((1, 1),), reason="warn")
        selected, value, diag = tutor._select_safety_action(
            [
                (wait, TutorActionValue(q_total=5.0, p_catastrophe=0.8)),
                (warning, TutorActionValue(q_total=1.0, p_catastrophe=0.0, diagnostics={"warning_actionability": 1.0})),
            ]
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected.kind, "WARNING")
        self.assertIsNotNone(value)
        self.assertEqual(diag["safety_shield_triggered"], 1.0)

    def test_safety_layer_returns_none_when_wait_is_safe(self) -> None:
        tutor = InversePlanningTutor(TutorConfig(mode="safety_shield_only", catastrophe_threshold=0.2))
        wait = make_tutor_action("WAIT", reason="wait")
        warning = make_tutor_action("WARNING", cells=((1, 1),), reason="warn")
        selected, value, diag = tutor._select_safety_action(
            [
                (wait, TutorActionValue(q_total=1.0, p_catastrophe=0.0)),
                (warning, TutorActionValue(q_total=2.0, p_catastrophe=0.0)),
            ]
        )
        self.assertIsNone(selected)
        self.assertIsNone(value)
        self.assertEqual(diag["safety_shield_triggered"], 0.0)

    def test_scaffold_prefers_lower_leakage_waypoint(self) -> None:
        tutor = InversePlanningTutor(
            TutorConfig(
                mode="shield_plus_minimal_waypoint",
                waypoint_min_advantage_over_wait=0.0,
                waypoint_damage_veto_margin=0.0,
            )
        )
        wait = make_tutor_action("WAIT", reason="wait")
        high_leak = make_tutor_action(
            "WAYPOINT",
            waypoint=(1, 2),
            reason="oracle",
            diagnostics={"waypoint_type": "oracle"},
        )
        low_leak = make_tutor_action(
            "WAYPOINT",
            waypoint=(1, 1),
            reason="frontier",
            diagnostics={"waypoint_type": "frontier"},
        )
        selected, _value, diag = tutor._select_scaffold_action(
            [
                (wait, TutorActionValue(q_total=0.0, p_timeout=0.4, boredom_cost=2.0)),
                (high_leak, TutorActionValue(q_total=0.8, p_timeout=0.1, boredom_cost=0.0, assist_leakage=1.0)),
                (low_leak, TutorActionValue(q_total=0.6, p_timeout=0.1, boredom_cost=0.0, assist_leakage=0.25)),
            ],
            step=0,
        )
        self.assertEqual(selected.kind, "WAYPOINT")
        self.assertEqual(selected.waypoint, (1, 1))
        self.assertGreaterEqual(diag["scaffold_improving_waypoint_count"], 2.0)

    def test_scaffold_respects_damage_veto(self) -> None:
        tutor = InversePlanningTutor(
            TutorConfig(
                mode="shield_plus_minimal_waypoint",
                waypoint_min_advantage_over_wait=0.0,
                waypoint_damage_veto_margin=0.0,
            )
        )
        wait = make_tutor_action("WAIT", reason="wait")
        waypoint = make_tutor_action("WAYPOINT", waypoint=(1, 1), reason="frontier")
        selected, _value, diag = tutor._select_scaffold_action(
            [
                (wait, TutorActionValue(q_total=0.0, expected_damage=0.2, p_timeout=0.4, boredom_cost=2.0)),
                (waypoint, TutorActionValue(q_total=1.0, expected_damage=1.0, p_timeout=0.1, boredom_cost=0.0, assist_leakage=0.25)),
            ],
            step=0,
        )
        self.assertEqual(selected.kind, "WAIT")
        self.assertGreater(diag["waypoint_damage_veto_blocked"], 0.0)

    def test_factory_builds_frontier_only_scaffold_mode(self) -> None:
        tutor = build_inverse_tutor("shield_plus_frontier_waypoint")
        self.assertEqual(str(tutor.config.mode), "shield_plus_minimal_waypoint")
        self.assertEqual(tuple(tutor.config.scaffold_waypoint_types), ("frontier",))
        self.assertTrue(tutor.config.safety_shield_enabled)

    def test_factory_builds_random_frontier_scaffold_mode(self) -> None:
        tutor = build_inverse_tutor("shield_plus_random_frontier_waypoint")
        self.assertEqual(str(tutor.config.mode), "shield_plus_minimal_waypoint")
        self.assertEqual(tuple(tutor.config.scaffold_waypoint_types), ("frontier",))
        self.assertTrue(tutor.config.randomize_scaffold_choice)
        self.assertTrue(tutor.config.safety_shield_enabled)

    def test_factory_builds_oracle_when_needed_alias(self) -> None:
        tutor = build_inverse_tutor("shield_plus_oracle_when_needed")
        self.assertEqual(str(tutor.config.mode), "shield_plus_minimal_waypoint")
        self.assertEqual(tuple(tutor.config.scaffold_waypoint_types), ("oracle",))
        self.assertTrue(tutor.config.safety_shield_enabled)


if __name__ == "__main__":
    unittest.main()
