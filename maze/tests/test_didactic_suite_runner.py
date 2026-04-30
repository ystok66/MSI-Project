from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from risky_maze.experiments.run_didactic_tutor_suite import didactic_conditions, didactic_maps, run_didactic_tutor_suite


class DidacticSuiteRunnerTests(unittest.TestCase):
    def test_run_didactic_tutor_suite_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            out_dir = Path(tmp) / "didactic_suite"
            result = run_didactic_tutor_suite(
                out_dir=out_dir,
                workers=1,
                seeds=[0],
                maps=didactic_maps()[:1],
                conditions=didactic_conditions()[:1],
                show_progress=False,
            )
            self.assertIn("summary", result)
            self.assertTrue((out_dir / "suite_config.json").exists())
            self.assertTrue((out_dir / "matrix_summary.csv").exists())
            self.assertTrue((out_dir / "matrix_tutor_behavior.csv").exists())
            self.assertTrue((out_dir / "DIDACTIC_TUTOR_SUITE_REPORT.md").exists())
            self.assertTrue((out_dir / "progress.log").exists())
            self.assertTrue((out_dir / "completed_blocks.csv").exists())
            map_name = didactic_maps()[0]["map_name"]
            self.assertTrue((out_dir / map_name / "no_tutor_mortal" / "partial_summary.csv").exists())
            self.assertTrue((out_dir / map_name / "no_tutor_mortal" / "partial_seed_summary.csv").exists())
            self.assertTrue((out_dir / map_name / "no_tutor_mortal" / "partial_status.json").exists())
            self.assertTrue((out_dir / map_name / "no_tutor_mortal" / "checkpoints" / "seed_000.json").exists())

    def test_didactic_maps_support_hard_variants(self) -> None:
        maps = didactic_maps(variant="didactic_hard")
        names = {m["map_name"] for m in maps}
        self.assertIn("TutorSafetyScaffoldGate_v1", names)
        self.assertIn("TutorAutonomyLoop_v1__autonomy_hard", names)
        self.assertIn("TutorPrincipleDoorTransfer_v1__transfer_hard", names)


if __name__ == "__main__":
    unittest.main()
