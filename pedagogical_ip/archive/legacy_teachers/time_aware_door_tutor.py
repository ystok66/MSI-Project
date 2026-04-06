"""
Time-Aware Door Tutor — strategic door closing based on time budget.

The tutor decides when to close risky-lane entry gates,
balancing safety, learning opportunities, and time pressure.

Trigger: agent in merge zone (corridor between segments, row 2).
Action: close risky_entry_gate of the NEXT segment.

Strategies based on time slack:
  tight  (slack < 0.2): close ALL risky gates ahead
  medium (0.2 ≤ slack < 0.5): close gates where cue features are strongest
  loose  (slack ≥ 0.5): leave doors open (let agent explore & learn)

Slack = (T_left - L_safe_remaining) / T_left
  where L_safe_remaining = BFS from agent_pos to goal through safe lanes
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Optional

import numpy as np

from ..envs.lattice_v2 import LatticeV2Meta, SegmentMeta, _bfs_len
from ..envs.map_generator import CellType, GridMap


@dataclass
class TutorAction:
    """Record of tutor's decision."""
    step: int
    segment_index: int
    action: str             # "close_risky_gate" or "keep_open"
    slack: float
    mode: str               # "tight", "medium", "loose"
    gate_cell: tuple[int, int]


class TimeAwareDoorTutor:
    """Strategic door-closing tutor for lattice_v2."""

    def __init__(
        self,
        gridmap: GridMap,
        meta: LatticeV2Meta,
        tight_threshold: float = 0.3,
        medium_threshold: float = 0.7,
    ):
        self.gm = gridmap
        self.meta = meta
        self.tight_th = tight_threshold
        self.medium_th = medium_threshold

        # Track which gates have been closed
        self.closed_gates: set[tuple[int, int]] = set()
        self.warned_segments: set[int] = set()
        self.actions_log: list[TutorAction] = []

        # Precompute: safe path length from each segment's entry to goal
        self._precompute_safe_paths()

    def _precompute_safe_paths(self):
        """Compute BFS distances using only safe lanes."""
        H, W = self.gm.height, self.gm.width
        goal = (2, W - 2)
        
        # Avoid all risky lane cells
        all_risky = set()
        for seg in self.meta.segments:
            all_risky.update(seg.risky_cells)

        self.safe_dist_to_goal = {}
        for seg in self.meta.segments:
            # Distance from segment entry (row 2) to goal through safe lanes
            entry = (2, seg.col_start)
            d = _bfs_len(self.gm, entry, goal, all_risky)
            self.safe_dist_to_goal[seg.index] = d

    def step(
        self,
        agent_pos: tuple[int, int],
        t_left: int,
        step_num: int,
    ) -> list[TutorAction]:
        """
        Decide tutor actions.

        Three truly distinct modes:
          tight:  door-first (always close if budget allows)
          medium: close door for traps, warn for non-trap risky lanes
          loose:  WARNING-FIRST — try warning, only close if needed
        """
        actions = []
        r, c = agent_pos

        if r != 2:
            return actions

        for seg in self.meta.segments:
            gate = seg.risky_entry_gate
            if gate in self.closed_gates:
                continue
            if seg.index in self.warned_segments:
                continue

            if abs(c - seg.col_start) > 1:
                continue

            safe_remaining = self.safe_dist_to_goal.get(seg.index, 999)
            if safe_remaining >= 999:
                slack = 0.0
            else:
                slack = (t_left - safe_remaining) / max(t_left, 1)

            if slack < self.tight_th:
                # TIGHT: always close door
                mode = "tight"
                action = self._close_gate(gate, seg.index, step_num, slack, mode)
                actions.append(action)

            elif slack < self.medium_th:
                # MEDIUM: close door for traps, warn for non-traps
                mode = "medium"
                if seg.trap_cell is not None:
                    action = self._close_gate(gate, seg.index, step_num, slack, mode)
                    actions.append(action)
                else:
                    # Non-trap risky lane: warn instead of close
                    actions.append(TutorAction(
                        step=step_num, segment_index=seg.index,
                        action="warn_only", slack=slack,
                        mode=mode, gate_cell=gate))
                    self.warned_segments.add(seg.index)

            else:
                # LOOSE: WARNING-FIRST — do not close door, issue warning
                mode = "loose"
                actions.append(TutorAction(
                    step=step_num, segment_index=seg.index,
                    action="warn_only", slack=slack,
                    mode=mode, gate_cell=gate))
                self.warned_segments.add(seg.index)

        self.actions_log.extend(actions)
        return actions

    def _close_gate(self, gate, seg_idx, step, slack, mode):
        """Mark gate as closed and return action record."""
        self.closed_gates.add(gate)
        return TutorAction(
            step=step, segment_index=seg_idx,
            action="close_risky_gate", slack=slack,
            mode=mode, gate_cell=gate)

    def is_gate_closed(self, cell: tuple[int, int]) -> bool:
        return cell in self.closed_gates

    def get_passable_mask(self, base_mask: np.ndarray) -> np.ndarray:
        """Apply closed gates to passability mask."""
        mask = base_mask.copy()
        for gate in self.closed_gates:
            mask[gate] = False
        return mask

    def reset(self):
        self.closed_gates.clear()
        self.warned_segments.clear()
        self.actions_log.clear()
