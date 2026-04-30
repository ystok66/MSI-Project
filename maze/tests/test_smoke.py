from __future__ import annotations

import unittest

import numpy as np

from risky_maze import MazeScenarioConfig, run_block
from risky_maze.learner import GaussianRiskBelief



class WarningUpdateTests(unittest.TestCase):
    def test_warning_increases_total_danger_mass(self) -> None:
        belief = GaussianRiskBelief(risk_dim=4, n_trap_types=2, seed=3)
        features = [
            np.array([1.0, 0.2, -0.1, 0.4], dtype=float),
            np.array([-0.2, 0.1, 0.5, 0.6], dtype=float),
        ]
        before = [belief.danger_probability(x) for x in features]
        belief.warning_update(features)
        after = [belief.danger_probability(x) for x in features]
        self.assertGreater(sum(after), sum(before))


class BlockSmokeTests(unittest.TestCase):
    def test_run_block_returns_main_metrics(self) -> None:
        cfg = MazeScenarioConfig(
            width=11,
            height=11,
            teach_episodes=2,
            eval_same_map_episodes=2,
            eval_new_map_episodes=2,
            seed=5,
        )
        results = run_block(cfg, tutor_name="inverse_warn")
        required = {
            "teach_success_rate",
            "teach_mean_warnings",
            "eval_same_map_success_rate",
            "eval_new_map_success_rate",
        }
        self.assertTrue(required.issubset(results.keys()))
        for key, value in results.items():
            self.assertTrue(np.isfinite(value), msg=key)
            if key.endswith("_rate"):
                self.assertGreaterEqual(value, 0.0, msg=key)
                self.assertLessEqual(value, 1.0, msg=key)

    def test_warning_tutor_emits_nonzero_warnings_in_shortcut_regime(self) -> None:
        cfg = MazeScenarioConfig(
            width=13,
            height=13,
            teach_episodes=4,
            eval_same_map_episodes=2,
            eval_new_map_episodes=2,
            trap_density=0.18,
            extra_loop_prob=0.25,
            learner_unknown_penalty=0.1,
            learner_info_bonus=0.45,
            seed=22,
        )
        results = run_block(cfg, tutor_name="inverse_warn")
        self.assertGreater(results["teach_mean_warnings"], 0.0)


if __name__ == "__main__":
    unittest.main()
