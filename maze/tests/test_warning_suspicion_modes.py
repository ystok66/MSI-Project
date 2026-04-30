from __future__ import annotations

import unittest

import numpy as np

from risky_maze.env.objectives import Objective
from risky_maze.env.pomdp_episode import Observation, ObservedCell
from risky_maze.learner.objective_agent import ObjectiveAwareLearner


def _obs() -> Observation:
    objective = Objective(kind="exit", coord=(1, 2))
    visible = {
        (1, 1): ObservedCell(coord=(1, 1), visible_kind="floor", risk_vector=np.zeros(2), is_walkable_observed=True),
        (1, 2): ObservedCell(coord=(1, 2), visible_kind="floor", risk_vector=np.zeros(2), is_walkable_observed=True),
    }
    return Observation(
        pos=(1, 1),
        hp=3,
        step_count=0,
        time_limit=10,
        view_radius=1,
        visible_cells=visible,
        has_key=False,
        objective_index=0,
        current_objective=objective,
        objective_sequence=[objective],
        has_gem_or_collected=frozenset(),
        map_shape=(3, 4),
    )


class WarningSuspicionModeTests(unittest.TestCase):
    def test_replan_only_suspicion_clears_after_act(self) -> None:
        learner = ObjectiveAwareLearner(risk_dim=2, warning_suspicion_mode="replan_only")
        learner.apply_warning([(1, 2)], [np.array([0.5, 0.5])])
        self.assertGreater(learner.warning_suspicion_value((1, 2)), 0.0)
        learner.act(_obs())
        self.assertEqual(learner.warning_suspicion_value((1, 2)), 0.0)

    def test_none_suspicion_mode_does_not_store_warning_bias(self) -> None:
        learner = ObjectiveAwareLearner(risk_dim=2, warning_suspicion_mode="none")
        learner.apply_warning([(1, 2)], [np.array([0.5, 0.5])])
        self.assertEqual(learner.warning_suspicion_value((1, 2)), 0.0)


if __name__ == "__main__":
    unittest.main()
