"""WorldState — True environment state snapshot.

Immutable read-only snapshot of the simulator ground truth.
Teacher-side code may read WorldState, but must never directly read
"what the agent knows" from here.

This is a POMDP-interface shell (Task 3 Phase A).
Does not change any existing behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class WorldState:
    """True environment state at time t.

    s_t^world = (x_agent, G, D, I, T, Φ_cost, Φ_risk, g)

    This is the GROUND TRUTH — not what the agent believes.
    Teacher/robot may read this for intervention planning.
    Agent should NEVER consume this directly.
    """
    # Agent position / orientation
    agent_pos: Tuple[int, int] = (0, 0)

    # Map topology
    height: int = 8
    width: int = 8
    cell_types: Optional[np.ndarray] = None      # (H, W) int, CellType
    passable: Optional[np.ndarray] = None         # (H, W) bool

    # Door / unlock state
    door_positions: Tuple[Tuple[int, int], ...] = ()
    doors_unlocked: Tuple[bool, ...] = ()

    # Item / shield state
    shield_available: bool = False
    shield_active: bool = False

    # Time budget
    t: int = 0
    t_max: int = 100

    # True latent fields (oracle only)
    true_cost: Optional[np.ndarray] = None        # (H, W)
    true_risk: Optional[np.ndarray] = None        # (H, W)

    # True latent field parameters (if analytic form available)
    phi_cost: Optional[np.ndarray] = None         # latent cost params
    phi_risk: Optional[np.ndarray] = None         # latent risk params

    # Goal
    goal_pos: Tuple[int, int] = (7, 7)

    @property
    def remaining_budget(self) -> int:
        return max(self.t_max - self.t, 0)


def world_state_from_grid_map(gm, t: int = 0, t_max: int = 100,
                               agent_pos=None) -> WorldState:
    """Adapter: build WorldState from existing GridMap + runtime state."""
    from ..envs.map_generator import CellType

    passable = np.array(gm.cell_types != CellType.WALL, dtype=bool)
    # Also mark locked doors as impassable
    for dp in gm.door_positions:
        if gm.cell_types[dp] == CellType.LOCKED_DOOR:
            passable[dp] = False

    return WorldState(
        agent_pos=agent_pos or gm.agent_start,
        height=gm.height, width=gm.width,
        cell_types=gm.cell_types.copy(),
        passable=passable,
        door_positions=tuple(gm.door_positions),
        doors_unlocked=tuple(False for _ in gm.door_positions),
        t=t, t_max=t_max,
        true_cost=gm.true_cost.copy(),
        true_risk=gm.true_risk.copy(),
        goal_pos=gm.target_pos,
    )
