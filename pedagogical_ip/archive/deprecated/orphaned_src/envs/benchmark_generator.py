"""
Benchmark generator — unified API for map family generation.

generate_benchmark_map(family, seed, difficulty) → (GridMap, FamilyConfig)
"""

from __future__ import annotations

from typing import Literal

from .map_families import (
    FAMILY_GENERATORS,
    FAMILY_NAMES,
    FamilyConfig,
    DifficultyLevel,
)
from .map_generator import GridMap


DIFFICULTIES: list[DifficultyLevel] = ["easy", "medium", "hard"]


def generate_benchmark_map(
    family: str,
    seed: int,
    difficulty: DifficultyLevel = "medium",
) -> tuple[GridMap, FamilyConfig]:
    """
    Generate a benchmark map from (family, seed, difficulty).

    Fully deterministic: same args → same map.
    """
    if family not in FAMILY_GENERATORS:
        raise ValueError(
            f"Unknown family '{family}'. Available: {FAMILY_NAMES}"
        )
    return FAMILY_GENERATORS[family](seed=seed, difficulty=difficulty)


def generate_transfer_map(
    family: str,
    seed: int,
    difficulty: DifficultyLevel = "medium",
    offset: int = 10000,
) -> tuple[GridMap, FamilyConfig]:
    """
    Generate a transfer-phase map: same family/difficulty, different layout.

    Uses seed + offset to ensure unseen layout.
    """
    return generate_benchmark_map(family, seed + offset, difficulty)
