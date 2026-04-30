from __future__ import annotations

import unittest

import numpy as np

from risky_maze.learner.objective_agent import OnlineGaussianRiskBelief


class WarningUpdateModeTests(unittest.TestCase):
    def test_literal_warning_update_uses_fixed_effective_sample(self) -> None:
        belief = OnlineGaussianRiskBelief(
            risk_dim=2,
            warning_update_mode="literal",
            warning_eta0=0.35,
        )
        features = [np.array([1.0, 1.0]), np.array([1.2, 1.2])]
        diag = belief.warning_update(features)
        self.assertAlmostEqual(diag["warning_ess"], 0.35, places=6)
        self.assertGreater(diag["warning_kl"], 0.0)
        self.assertGreater(diag["mean_abs_delta"], 0.0)

    def test_effective_sample_warning_update_is_kl_scaled(self) -> None:
        belief = OnlineGaussianRiskBelief(
            risk_dim=2,
            warning_update_mode="effective_sample",
            warning_eta0=0.35,
        )
        features = [np.array([1.0, 1.0]), np.array([1.2, 1.2]), np.array([1.4, 1.4])]
        diag = belief.warning_update(features)
        self.assertGreater(diag["warning_ess"], 0.0)
        self.assertGreater(diag["warning_kl"], 0.0)
        self.assertLessEqual(diag["warning_ess"], 2.0)
        self.assertEqual(diag["warning_set_size"], 3.0)


if __name__ == "__main__":
    unittest.main()
