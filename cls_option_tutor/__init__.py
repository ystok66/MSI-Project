"""Active package surface for the CLS option-world tutoring benchmark.

The current frozen sparse-tutor mainline is exposed here so scripts and docs
can refer to a stable short alias instead of duplicating long condition names.
"""

from .experiments.mainline_registry import (
    ACTIVE_BASELINE_CONDITIONS,
    ACTIVE_MAINLINE_ALIAS,
    ACTIVE_MAINLINE_CANONICAL,
    ACTIVE_MAINLINE_NATIVEALLOW_ALIAS,
    ACTIVE_MAINLINE_NATIVEALLOW_CANONICAL,
    NO_TUTOR_BUDGETED_CONDITION,
    SCRIPTED_SAFE_GOLD_CONDITION,
)

__all__ = [
    "ACTIVE_BASELINE_CONDITIONS",
    "ACTIVE_MAINLINE_ALIAS",
    "ACTIVE_MAINLINE_CANONICAL",
    "ACTIVE_MAINLINE_NATIVEALLOW_ALIAS",
    "ACTIVE_MAINLINE_NATIVEALLOW_CANONICAL",
    "NO_TUTOR_BUDGETED_CONDITION",
    "SCRIPTED_SAFE_GOLD_CONDITION",
]
