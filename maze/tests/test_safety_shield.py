from __future__ import annotations

import unittest

from risky_maze.tutor.compat import make_tutor_action
from risky_maze.tutor.inverse_planner import InversePlanningTutor, TutorConfig
from risky_maze.tutor.rollout import TutorActionValue


class SafetyShieldTests(unittest.TestCase):
    def test_no_shield_when_disabled(self) -> None:
        tutor = InversePlanningTutor(TutorConfig(safety_shield_enabled=False, catastrophe_threshold=0.2))
        wait = make_tutor_action("WAIT", reason="wait")
        warning = make_tutor_action("WARNING", cells=((1, 1),), reason="warn")
        selected, _value, diag = tutor._select_with_guardrails(
            [
                (wait, TutorActionValue(q_total=1.0, p_catastrophe=0.8)),
                (warning, TutorActionValue(q_total=0.1, p_catastrophe=0.0)),
            ],
            step=0,
        )
        self.assertEqual(selected.kind, "WAIT")
        self.assertEqual(diag["safety_shield_triggered"], 0.0)

    def test_warning_selected_when_wait_is_catastrophic(self) -> None:
        tutor = InversePlanningTutor(TutorConfig(safety_shield_enabled=True, catastrophe_threshold=0.2))
        wait = make_tutor_action("WAIT", reason="wait")
        warning = make_tutor_action("WARNING", cells=((1, 1),), reason="warn")
        selected, _value, diag = tutor._select_with_guardrails(
            [
                (wait, TutorActionValue(q_total=1.0, p_catastrophe=0.8)),
                (warning, TutorActionValue(q_total=0.1, p_catastrophe=0.0)),
            ],
            step=0,
        )
        self.assertEqual(selected.kind, "WARNING")
        self.assertEqual(diag["safety_shield_triggered"], 1.0)

    def test_waypoint_selected_when_warning_is_not_safe(self) -> None:
        tutor = InversePlanningTutor(TutorConfig(safety_shield_enabled=True, catastrophe_threshold=0.2))
        wait = make_tutor_action("WAIT", reason="wait")
        warning = make_tutor_action("WARNING", cells=((1, 1),), reason="warn")
        waypoint = make_tutor_action("WAYPOINT", waypoint=(1, 2), reason="wp")
        selected, _value, diag = tutor._select_with_guardrails(
            [
                (wait, TutorActionValue(q_total=1.0, p_catastrophe=0.8)),
                (warning, TutorActionValue(q_total=0.3, p_catastrophe=0.4)),
                (waypoint, TutorActionValue(q_total=0.2, p_catastrophe=0.0)),
            ],
            step=0,
        )
        self.assertEqual(selected.kind, "WAYPOINT")
        self.assertEqual(diag["safety_shield_triggered"], 1.0)

    def test_warning_actionability_threshold_blocks_low_actionability_warning(self) -> None:
        tutor = InversePlanningTutor(TutorConfig(warning_actionability_threshold=0.5))
        wait = make_tutor_action("WAIT", reason="wait")
        warning = make_tutor_action("WARNING", cells=((1, 1),), reason="warn")
        selected, _value, diag = tutor._select_with_guardrails(
            [
                (wait, TutorActionValue(q_total=1.0, p_catastrophe=0.0)),
                (warning, TutorActionValue(q_total=2.0, p_catastrophe=0.0, diagnostics={"warning_actionability": 0.1})),
            ],
            step=0,
        )
        self.assertEqual(selected.kind, "WAIT")
        self.assertGreater(diag["warning_actionability_blocked"], 0.0)


if __name__ == "__main__":
    unittest.main()
