"""Standalone prototype for a risky maze tutoring scenario."""

from .config import MazeScenarioConfig
from .runner import run_block, run_fixed_block

__all__ = ["MazeScenarioConfig", "run_block", "run_fixed_block"]
