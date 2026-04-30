from __future__ import annotations

import unittest

import numpy as np

from risky_maze.learner.objective_agent import ObjectiveAwareLearner


class EvalAblationCloneTests(unittest.TestCase):
    def test_clone_for_eval_can_clear_memory_but_keep_risk_belief(self) -> None:
        learner = ObjectiveAwareLearner(risk_dim=2)
        learner.memory.known_walkable.add((1, 1))
        learner.memory.observed_vectors[(1, 1)] = [np.array([0.5, 0.5])]
        learner.risk_belief.update_labeled(np.array([2.0, 2.0]), True, weight=1.0)
        cloned = learner.clone_for_eval(clear_memory=True, clear_risk_belief=False)
        self.assertFalse(cloned.memory.known_walkable)
        self.assertGreater(cloned.risk_belief.prior_danger, 0.25)

    def test_clone_for_eval_can_clear_risk_belief(self) -> None:
        learner = ObjectiveAwareLearner(risk_dim=2)
        learner.risk_belief.update_labeled(np.array([2.0, 2.0]), True, weight=2.0)
        cloned = learner.clone_for_eval(clear_memory=False, clear_risk_belief=True)
        self.assertAlmostEqual(cloned.risk_belief.prior_danger, 0.25, places=6)

    def test_clone_for_eval_can_clear_warning_suspicion(self) -> None:
        learner = ObjectiveAwareLearner(risk_dim=2)
        learner.memory.warning_suspicion[(1, 1)] = 2.0
        cloned = learner.clone_for_eval(clear_warning_suspicion=True)
        self.assertFalse(cloned.memory.warning_suspicion)
        self.assertEqual(learner.memory.warning_suspicion[(1, 1)], 2.0)


if __name__ == "__main__":
    unittest.main()
