from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from risky_maze.experiments import run_fixed_experiment
from risky_maze.experiments.run_d4_fix_comparison import run_d4_fix_comparison


class ExperimentRunnerTests(unittest.TestCase):
    def test_run_fixed_experiment_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            out_dir = Path(tmp) / "run"
            result = run_fixed_experiment(
                spec_name="HugeRiskyGemMaze_v0",
                tutor_name="no_tutor",
                baseline_mode="mortal",
                seeds=[0],
                out_dir=out_dir,
                workers=1,
                teach_task_ids=["T01_NW_key_gem_NE_exit"],
                eval_task_ids=["E01_WestGarden_to_NE_exit"],
                config_overrides={
                    "risk_dim": 8,
                    "obs_noise": 0.25,
                    "cluster_std": 0.35,
                    "view_radius": 2,
                    "hp": 3,
                    "n_safe_types": 3,
                    "n_trap_types": 3,
                    "tutor_rollout_horizon": 3,
                    "tutor_top_k_paths": 1,
                    "tutor_max_candidates": 4,
                    "tutor_profile_count": 2,
                },
                show_progress=False,
            )
            self.assertIn("summary", result)
            expected = [
                "config.json",
                "summary.csv",
                "seed_summary.csv",
                "episodes.csv",
                "steps.csv",
                "tutor_decisions.csv",
                "risk_eval.csv",
                "map_reuse.csv",
                "objective_progress.csv",
                "README.md",
            ]
            for name in expected:
                self.assertTrue((out_dir / name).exists(), msg=name)
            trajectory_dir = out_dir / "trajectories"
            self.assertTrue(trajectory_dir.exists())
            self.assertTrue(any(trajectory_dir.iterdir()))
            risk_eval_text = (out_dir / "risk_eval.csv").read_text(encoding="utf-8")
            self.assertIn("risk_auc_seen", risk_eval_text)
            self.assertIn("risk_auc_unseen_same_map", risk_eval_text)

    def test_run_fixed_experiment_can_disable_step_detail(self) -> None:
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            out_dir = Path(tmp) / "run_no_detail"
            result = run_fixed_experiment(
                spec_name="HugeRiskyGemMaze_v0",
                tutor_name="no_tutor",
                baseline_mode="mortal",
                seeds=[0],
                out_dir=out_dir,
                workers=1,
                teach_task_ids=["T01_NW_key_gem_NE_exit"],
                eval_task_ids=["E01_WestGarden_to_NE_exit"],
                config_overrides={
                    "risk_dim": 8,
                    "obs_noise": 0.25,
                    "cluster_std": 0.35,
                    "view_radius": 2,
                    "hp": 3,
                    "n_safe_types": 3,
                    "n_trap_types": 3,
                    "tutor_rollout_horizon": 3,
                    "tutor_top_k_paths": 1,
                    "tutor_max_candidates": 4,
                    "tutor_profile_count": 2,
                    "record_step_details": False,
                },
                show_progress=False,
            )
            self.assertIn("summary", result)
            self.assertEqual((out_dir / "steps.csv").read_text(encoding="utf-8").strip(), "")
            self.assertEqual((out_dir / "tutor_decisions.csv").read_text(encoding="utf-8").strip(), "")

    def test_run_d4_fix_comparison_writes_comparison_outputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            out_dir = Path(tmp) / "d4_compare"
            result = run_d4_fix_comparison(
                spec_name="HugeRiskyGemMaze_v0",
                out_dir=out_dir,
                workers=1,
                seeds=[0],
                teach_task_ids=["T01_NW_key_gem_NE_exit"],
                eval_task_ids=["E01_WestGarden_to_NE_exit"],
                conditions=[
                    {
                        "condition_name": "no_tutor_mortal",
                        "tutor_name": "no_tutor",
                        "baseline_mode": "mortal",
                        "config_overrides": {
                            "risk_dim": 8,
                            "obs_noise": 0.25,
                            "cluster_std": 0.35,
                            "view_radius": 2,
                            "hp": 3,
                            "n_safe_types": 3,
                            "n_trap_types": 3,
                            "record_step_details": False,
                        },
                    }
                ],
                show_progress=False,
            )
            self.assertIn("comparison", result)
            for name in [
                "comparison_config.json",
                "variant_condition_summaries.csv",
                "variant_condition_tutor_behavior.csv",
                "comparison_summary.csv",
                "comparison_tutor_behavior.csv",
                "comparison_index.json",
                "D4_FIX_COMPARISON_REPORT.md",
            ]:
                self.assertTrue((out_dir / name).exists(), msg=name)


if __name__ == "__main__":
    unittest.main()
