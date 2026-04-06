"""V6.1 — Branch Candidate Extractor.

Extracts competing branch candidates at fork points.
Only activates when multiple passable, similar-length branches exist.

For ELCB: directly reads branch metadata from ScenarioConfig.
For general maps: BFS-based fork detection (future extension).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BranchCandidate:
    """A candidate branch for reranking."""
    branch_id: int
    cells: list[tuple[int, int]]
    length: int
    fork_cell: tuple[int, int]
    merge_cell: tuple[int, int]
    entry_gate: tuple[int, int]
    exit_gate: tuple[int, int]


def extract_elcb_branches(sc) -> list[BranchCandidate]:
    """Extract branch candidates from ELCB ScenarioConfig.

    ELCB has exactly 2 branches with known cells, fork, and merge points.
    """
    fork = sc.fork_cell
    merge = sc.merge_cell

    return [
        BranchCandidate(
            branch_id=0,
            cells=list(sc.branch_a_cells),
            length=len(sc.branch_a_cells),
            fork_cell=fork,
            merge_cell=merge,
            entry_gate=(1, fork[1]),
            exit_gate=(1, merge[1]),
        ),
        BranchCandidate(
            branch_id=1,
            cells=list(sc.branch_b_cells),
            length=len(sc.branch_b_cells),
            fork_cell=fork,
            merge_cell=merge,
            entry_gate=(3, fork[1]),
            exit_gate=(3, merge[1]),
        ),
    ]


def should_activate_branch_reranker(
    candidates: list[BranchCandidate],
    max_length_diff: int = 2,
    min_branches: int = 2,
) -> bool:
    """Gating: should the branch reranker be invoked?

    Criteria:
    1. At least min_branches candidates
    2. Length difference within threshold
    3. All candidates have cells

    Returns True if branch-level reasoning is warranted.
    """
    if len(candidates) < min_branches:
        return False

    lengths = [c.length for c in candidates]
    if max(lengths) - min(lengths) > max_length_diff:
        return False

    if any(c.length == 0 for c in candidates):
        return False

    return True
