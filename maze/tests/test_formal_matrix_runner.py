from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from risky_maze.experiments.run_formal_tutor_matrix import run_formal_tutor_matrix


class FormalMatrixRunnerTests(unittest.TestCase):
    def test_run_formal_tutor_matrix_writes_summary_outputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            out_dir = Path(tmp) / "formal_matrix"
            result = run_formal_tutor_matrix(
                spec_name="HugeRiskyGemMaze_v0",
                out_dir=out_dir,
                workers=1,
                seeds=[0],
                slices=[
                    {
                        "slice_name": "smoke_slice",
                        "teach_task_ids": ["T01_NW_key_gem_NE_exit"],
                        "eval_task_ids": ["E01_WestGarden_to_NE_exit"],
                        "description": "test slice",
                    }
                ],
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
            self.assertIn("summary", result)
            self.assertTrue((out_dir / "matrix_config.json").exists())
            self.assertTrue((out_dir / "matrix_summary.csv").exists())
            self.assertTrue((out_dir / "matrix_tutor_behavior.csv").exists())
            self.assertTrue((out_dir / "FORMAL_TUTOR_MATRIX_REPORT.md").exists())
            self.assertTrue((out_dir / "progress.log").exists())
            self.assertTrue((out_dir / "completed_blocks.csv").exists())
            self.assertTrue((out_dir / "smoke_slice" / "no_tutor_mortal" / "partial_summary.csv").exists())
            self.assertTrue((out_dir / "smoke_slice" / "no_tutor_mortal" / "partial_seed_summary.csv").exists())
            self.assertTrue((out_dir / "smoke_slice" / "no_tutor_mortal" / "partial_status.json").exists())
            self.assertTrue((out_dir / "smoke_slice" / "no_tutor_mortal" / "checkpoints" / "seed_000.json").exists())


if __name__ == "__main__":
    unittest.main()
