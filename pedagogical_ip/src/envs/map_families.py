"""
Parameterized map families for v1b benchmark suite.

Four families, each testing a specific teacher intervention advantage:
  A. SemanticTrap   — WARN is best (learner misbelieves risk)
  B. PlanningTrap   — UNLOCK is best (bounded planner can't find safe detour)
  C. ExplorationUseful — WAIT is best (exploration improves transfer)
  D. Mixed          — optimal intervention varies by phase

All generation is deterministic from (family, seed, difficulty).
All maps are 10×10 for richer structure than 8×8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .map_generator import CellType, GridMap

DifficultyLevel = Literal["easy", "medium", "hard"]

# ── Family config dataclass ──────────────────────────────────────────

@dataclass
class FamilyConfig:
    """Per-episode overrides returned alongside the map."""
    max_steps: int = 60
    risk_budget: float = 1.0
    prior_risk_mean: float = 0.1
    prior_risk_var: float = 0.25
    search_budget: int = 30
    budget_class: int = 8

# ── Difficulty parameter tables ──────────────────────────────────────

DIFFICULTY_PARAMS: dict[str, dict[str, dict]] = {
    "semantic_trap": {
        "easy":   {"risk_val": 0.35, "max_steps": 50, "risk_budget": 0.8, "safe_detour": 4},
        "medium": {"risk_val": 0.50, "max_steps": 40, "risk_budget": 0.5, "safe_detour": 6},
        "hard":   {"risk_val": 0.70, "max_steps": 35, "risk_budget": 0.3, "safe_detour": 8},
    },
    "planning_trap": {
        "easy":   {"detour_len": 4, "max_steps": 50, "risk_budget": 0.6, "budget_class": 8},
        "medium": {"detour_len": 6, "max_steps": 40, "risk_budget": 0.4, "budget_class": 4},
        "hard":   {"detour_len": 8, "max_steps": 35, "risk_budget": 0.3, "budget_class": 4},
    },
    "exploration_useful": {
        "easy":   {"unk_var": 0.5,  "max_steps": 55, "risk_budget": 1.0, "risk_val": 0.15},
        "medium": {"unk_var": 1.0,  "max_steps": 50, "risk_budget": 0.8, "risk_val": 0.20},
        "hard":   {"unk_var": 2.0,  "max_steps": 45, "risk_budget": 0.6, "risk_val": 0.25},
    },
    "mixed": {
        "easy":   {"max_steps": 55, "risk_budget": 0.7, "risk_val": 0.35, "budget_class": 8},
        "medium": {"max_steps": 45, "risk_budget": 0.5, "risk_val": 0.50, "budget_class": 8},
        "hard":   {"max_steps": 38, "risk_budget": 0.3, "risk_val": 0.65, "budget_class": 4},
    },
}

# ── Helper: vectorized grid builders ─────────────────────────────────

def _empty_grid(H: int, W: int):
    """Return blank cell_types, cost, risk arrays."""
    ct = np.full((H, W), CellType.NORMAL, dtype=np.int32)
    cost = np.ones((H, W), dtype=np.float64)
    risk = np.zeros((H, W), dtype=np.float64)
    return ct, cost, risk


def _set_wall(ct, cost, positions):
    """Set wall cells (vectorized-friendly)."""
    for r, c in positions:
        ct[r, c] = CellType.WALL
        cost[r, c] = np.inf


def _set_risky(ct, cost, risk, positions, risk_val: float):
    """Set risky cells."""
    for r, c in positions:
        ct[r, c] = CellType.RISKY
        cost[r, c] = 1.0
        risk[r, c] = risk_val


def _set_highcost(ct, cost, positions, cost_val: float = 5.0):
    """Set high-cost cells."""
    for r, c in positions:
        ct[r, c] = CellType.HIGH_COST
        cost[r, c] = cost_val


def _set_door(ct, cost, positions):
    """Set locked door cells."""
    for r, c in positions:
        ct[r, c] = CellType.LOCKED_DOOR
        cost[r, c] = np.inf


def _build_gridmap(
    H, W, ct, cost, risk,
    agent_start, object_spawn, target_pos, door_positions,
) -> GridMap:
    ct[object_spawn] = CellType.OBJECT_SPAWN
    ct[target_pos] = CellType.TARGET
    return GridMap(
        height=H, width=W,
        cell_types=ct, true_cost=cost, true_risk=risk,
        object_spawn=object_spawn,
        target_pos=target_pos,
        agent_start=agent_start,
        door_positions=door_positions,
    )


# ══════════════════════════════════════════════════════════════════════
# Family A: Semantic Trap
# ══════════════════════════════════════════════════════════════════════

def generate_semantic_trap(
    seed: int = 0,
    difficulty: DifficultyLevel = "medium",
) -> tuple[GridMap, FamilyConfig]:
    """
    Two routes from agent→object→target:
      - Right corridor: SHORT but RISKY (learner doesn't know risk is high)
      - Left corridor:  LONGER but SAFE

    Structural guarantee: without WARN, learner's prior (risk_mean=0.1)
    makes right route look cheaper → learner takes it → risk death.
    With WARN("RIGHT_RISKY"), learner updates belief → takes safe route.

    Layout (10×10):
        A . . . . . . . . .    row 0: agent start (0,0)
        . # # . . . . . . .    rows 1-2: wall blocks force route choice
        . # # . . . . . . .
        . . . . # ! ! ! . .    row 3: wall + risky corridor (right path)
        . . . . # ! ! ! . .    row 4: wall + risky corridor
        . . . . # . . . . O    row 5: object (5,9)
        . . . . # . . . . .    row 6: wall divider continues
        . . . . . . . . . .    row 7: merge point
        . . . . . . . . . .    row 8:
        . . . . . . . . . T    row 9: target (9,9)
    """
    rng = np.random.default_rng(seed)
    params = DIFFICULTY_PARAMS["semantic_trap"][difficulty]
    H, W = 10, 10
    ct, cost, risk = _empty_grid(H, W)

    # Walls: force two-route structure
    walls = [(1, 1), (1, 2), (2, 1), (2, 2)]
    # Central divider: rows 3-6, col 4
    walls += [(r, 4) for r in range(3, 7)]
    # Slight randomization: add 1-2 extra walls from seed
    extra_wall_candidates = [(0, 3), (1, 5), (7, 2), (8, 1)]
    n_extra = rng.integers(0, min(3, len(extra_wall_candidates)) + 1)
    walls += list(rng.choice(extra_wall_candidates, size=n_extra, replace=False))
    _set_wall(ct, cost, walls)

    # Risky corridor: right side of divider (rows 3-4, cols 5-7)
    risk_val = params["risk_val"]
    risky_cells = [(3, 5), (3, 6), (3, 7), (4, 5), (4, 6), (4, 7)]
    # Add some variance from seed
    extra_risky = [(2, 7), (2, 8), (5, 7), (5, 8)]
    n_risky = rng.integers(1, len(extra_risky) + 1)
    risky_cells += list(rng.choice(extra_risky, size=n_risky, replace=False))
    _set_risky(ct, cost, risk, risky_cells, risk_val)

    # High-cost on safe detour (left side) to make it slower but safe
    detour = params["safe_detour"]
    hc_cells = [(r, c) for r in range(3, 3 + min(detour, 4))
                for c in range(0, min(2, detour // 2 + 1))]
    _set_highcost(ct, cost, hc_cells, cost_val=3.0 + rng.uniform(0, 2))

    agent_start = (0, 0)
    object_spawn = (5, 9)
    target_pos = (9, 9)

    gm = _build_gridmap(H, W, ct, cost, risk,
                         agent_start, object_spawn, target_pos, [])

    cfg = FamilyConfig(
        max_steps=params["max_steps"],
        risk_budget=params["risk_budget"],
        prior_risk_mean=0.1,     # learner underestimates right-side risk
        prior_risk_var=0.25,
        search_budget=30,
        budget_class=8,
    )
    return gm, cfg


# ══════════════════════════════════════════════════════════════════════
# Family B: Planning Trap
# ══════════════════════════════════════════════════════════════════════

def generate_planning_trap(
    seed: int = 0,
    difficulty: DifficultyLevel = "medium",
) -> tuple[GridMap, FamilyConfig]:
    """
    v1f: Construct-valid planning trap.

    TrapValid condition:
      - WITHOUT door: bounded A* finds a LONG path around the wall (risky)
      - WITH door: bounded A* finds a SHORT path through the door (safe)

    Layout (10×10):
        A . . . . . . . . .    row 0: agent at (0,0)
        . . . . . . . . . .    row 1: open
        . . . . . . . . . .    row 2: open
        # # # # D # # # # #    row 3: HORIZONTAL WALL + DOOR at (3,4)
        . . . . . . . . . .    row 4: open (below wall)
        . . . . . . . . . O    row 5: object at (5,9)
        . . . . . . . . . .    row 6: open
        . . . . . . . . . .    row 7: open
        . . . . . . . . . .    row 8: open
        . . . . . . . . . T    row 9: target at (9,9)

    Without door: the only way around is through risky open gaps at wall edges.
    Risky cells guard the wall-end openings.

    With door: straight path (0,0)→(3,4)→(5,9) is short and safe.
    """
    rng = np.random.default_rng(seed)
    params = DIFFICULTY_PARAMS["planning_trap"][difficulty]
    H, W = 10, 10
    ct, cost, risk = _empty_grid(H, W)

    # Horizontal wall at row 3 — leave gap for door at col 4
    wall_row = 3
    door_col = 4
    for c in range(W):
        if c != door_col:
            ct[wall_row, c] = CellType.WALL
            cost[wall_row, c] = np.inf

    # Door at (3, 4)
    door_pos = (wall_row, door_col)
    _set_door(ct, cost, [door_pos])

    # Without the door, agent must detour around one wall end.
    # The wall spans all of row 3, so there's NO gap (door is locked).
    # We need to create an alternative path: open a gap in the wall
    # at the far-right end, but guard it with risky cells.
    gap_col = 9  # far-right gap
    ct[wall_row, gap_col] = CellType.NORMAL
    cost[wall_row, gap_col] = 1.0

    # Risky cells guard the detour gap
    detour_len = params["detour_len"]
    risk_val = 0.3 + 0.1 * (detour_len / 4)
    risky_cells = [(2, 9), (2, 8)]  # above gap
    if detour_len >= 6:
        risky_cells += [(2, 7), (4, 9)]
    if detour_len >= 8:
        risky_cells += [(4, 8), (1, 9)]
    _set_risky(ct, cost, risk, risky_cells, risk_val)

    # High-cost cells to make the detour even less attractive
    hc_cells = [(1, 8), (4, 7)]
    n_hc = min(rng.integers(0, 3), len(hc_cells))
    for i in range(n_hc):
        r, c = hc_cells[i]
        ct[r, c] = CellType.HIGH_COST
        cost[r, c] = 3.0 + rng.uniform(0, 1)

    agent_start = (0, 0)
    object_spawn = (5, 9)
    target_pos = (9, 9)

    gm = _build_gridmap(H, W, ct, cost, risk,
                         agent_start, object_spawn, target_pos, [door_pos])

    cfg = FamilyConfig(
        max_steps=params["max_steps"],
        risk_budget=params["risk_budget"],
        prior_risk_mean=0.1,    # learner underestimates risky cells
        prior_risk_var=0.15,    # moderate uncertainty
        search_budget=40,       # enough to find both paths
        budget_class=params["budget_class"],
    )
    return gm, cfg


# ══════════════════════════════════════════════════════════════════════
# Family C: Exploration Useful
# ══════════════════════════════════════════════════════════════════════

def generate_exploration_useful(
    seed: int = 0,
    difficulty: DifficultyLevel = "medium",
) -> tuple[GridMap, FamilyConfig]:
    """
    Low-risk grid where exploration is safe and builds transferable knowledge.
    WAIT is best: lets learner explore and learn risk structure.
    always_help prevents exploration → worse transfer.

    Layout (10×10):
        A . . . . . . . . .    row 0: agent start; low risk throughout
        . . . . . . . . . .    rows 0-4: safe exploration zone
        . . . ? ? . . . . .    ? = uncertain region (low actual risk)
        . . . ? ? . . . . .    learner prior has HIGH variance here
        . . . . . . . . . O    row 4: object
        . . . . . . . . . .    rows 5-9: relatively clear path to target
        . . . . . . . . . .
        . . . . . . . . . .
        . . . . . . . . . .
        . . . . . . . . . T    row 9: target
    """
    rng = np.random.default_rng(seed)
    params = DIFFICULTY_PARAMS["exploration_useful"][difficulty]
    H, W = 10, 10
    ct, cost, risk = _empty_grid(H, W)

    # Uncertain region: low actual risk but learner doesn't know
    risk_val = params["risk_val"]
    uncertain_cells = [(2, 3), (2, 4), (3, 3), (3, 4),
                       (2, 5), (3, 5), (4, 3), (4, 4)]
    _set_risky(ct, cost, risk, uncertain_cells, risk_val)

    # A few walls for structure variety
    wall_candidates = [(1, 2), (5, 5), (6, 5), (7, 1), (8, 6)]
    n_walls = rng.integers(2, 4)
    chosen = rng.choice(len(wall_candidates), size=n_walls, replace=False)
    _set_wall(ct, cost, [wall_candidates[i] for i in chosen])

    # Some light high-cost terrain for realism
    hc_candidates = [(5, 2), (6, 3), (7, 4), (1, 7)]
    n_hc = rng.integers(1, 3)
    chosen_hc = rng.choice(len(hc_candidates), size=n_hc, replace=False)
    _set_highcost(ct, cost, [hc_candidates[i] for i in chosen_hc], 3.0)

    agent_start = (0, 0)
    object_spawn = (4, 9)
    target_pos = (9, 9)

    gm = _build_gridmap(H, W, ct, cost, risk,
                         agent_start, object_spawn, target_pos, [])

    cfg = FamilyConfig(
        max_steps=params["max_steps"],
        risk_budget=params["risk_budget"],
        prior_risk_mean=0.1,
        prior_risk_var=params["unk_var"],   # HIGH variance → high uncertainty
        search_budget=30,
        budget_class=8,
    )
    return gm, cfg


# ══════════════════════════════════════════════════════════════════════
# Family D: Mixed (multi-phase)
# ══════════════════════════════════════════════════════════════════════

def generate_mixed(
    seed: int = 0,
    difficulty: DifficultyLevel = "medium",
) -> tuple[GridMap, FamilyConfig]:
    """
    Three-phase map requiring different interventions at each stage:
      Phase 1 (rows 0-3): safe exploration zone → WAIT is best
      Phase 2 (rows 4-6): risky corridor → WARN is best
      Phase 3 (rows 7-9): door bottleneck → UNLOCK is best

    Layout (10×10):
        A . . . . . . . . .    Phase 1: safe, low-risk
        . . . . . . . . . .
        . . . . . . . . O .    object at (2,8)
        . . . . . . . . . .
        . . ! ! ! ! ! . . .    Phase 2: risky corridor
        . . ! ! ! ! ! . . .
        . . . . . . . . . .
        . # # D . . . . . .    Phase 3: door bottleneck
        . # # # . . . . . .
        . . . . . . . . . T    target at (9,9)
    """
    rng = np.random.default_rng(seed)
    params = DIFFICULTY_PARAMS["mixed"][difficulty]
    H, W = 10, 10
    ct, cost, risk = _empty_grid(H, W)

    # Phase 2: risky corridor (rows 4-5, cols 2-6)
    risk_val = params["risk_val"]
    risky_cells = [(r, c) for r in range(4, 6) for c in range(2, 7)]
    _set_risky(ct, cost, risk, risky_cells, risk_val)

    # Phase 3: wall block + door
    walls = [(7, 1), (7, 2), (8, 1), (8, 2), (8, 3)]
    _set_wall(ct, cost, walls)
    door_pos = (7, 3)
    _set_door(ct, cost, [door_pos])

    # Extra walls for variety
    extra = [(3, 5), (6, 8)]
    n_extra = rng.integers(0, len(extra) + 1)
    if n_extra > 0:
        chosen = rng.choice(len(extra), size=n_extra, replace=False)
        _set_wall(ct, cost, [extra[i] for i in chosen])

    # High-cost to discourage trivial bypass of phase 2
    hc_cells = [(4, 0), (4, 1), (5, 0), (5, 1), (4, 8), (4, 9), (5, 8), (5, 9)]
    _set_highcost(ct, cost, hc_cells, 4.0 + rng.uniform(0, 2))

    agent_start = (0, 0)
    object_spawn = (2, 8)
    target_pos = (9, 9)

    gm = _build_gridmap(H, W, ct, cost, risk,
                         agent_start, object_spawn, target_pos, [door_pos])

    cfg = FamilyConfig(
        max_steps=params["max_steps"],
        risk_budget=params["risk_budget"],
        prior_risk_mean=0.1,
        prior_risk_var=0.25,
        search_budget=20 if params["budget_class"] == 4 else 30,
        budget_class=params["budget_class"],
    )
    return gm, cfg


# ══════════════════════════════════════════════════════════════════════
# Family E: Door Lattice Sanity Check
# NOT for main benchmark — only for verifying UNLOCK/BLOCK ↔ planner
# ══════════════════════════════════════════════════════════════════════

DIFFICULTY_PARAMS["door_lattice_sanity"] = {
    # time_ratio: max_steps = time_ratio * shortest_path_length
    "easy":   {"risk_budget": 0.8, "risk_val": 0.70, "door_prob": 0.5,
               "risk_behind_door_prob": 0.8, "time_ratio": 1.6},
    "medium": {"risk_budget": 0.5, "risk_val": 0.80, "door_prob": 0.6,
               "risk_behind_door_prob": 0.8, "time_ratio": 1.4},
    "hard":   {"risk_budget": 0.3, "risk_val": 0.90, "door_prob": 0.7,
               "risk_behind_door_prob": 0.8, "time_ratio": 1.2},
}


def generate_door_lattice_sanity(
    seed: int = 0,
    difficulty: DifficultyLevel = "medium",
    # Override params for sweep experiments
    door_prob_override: float | None = None,
    time_ratio_override: float | None = None,
) -> tuple[GridMap, FamilyConfig]:
    """
    Door Lattice Sanity Check — 9×17 grid (5 wall rows × 5 wall cols).

    Template:
        1 ddd 1 ddd 1 ddd 1 ddd 1    row 0,2,4,6,8: wall rows
        1 000 d 000 d 000 d 000 1    row 1,3,5,7: room rows

    Rules:
      - ddd: at most 1 becomes locked door, rest FREE
      - single d: locked door or FREE
      - 50% chance of risky cell behind each door
      - Start col 0, goal col 16
      - max_steps = time_ratio * shortest_path_length
    """
    rng = np.random.default_rng(seed)
    params = DIFFICULTY_PARAMS["door_lattice_sanity"][difficulty]
    door_prob = door_prob_override if door_prob_override is not None else params["door_prob"]
    time_ratio = time_ratio_override if time_ratio_override is not None else params["time_ratio"]
    risk_val = params["risk_val"]
    risk_behind_prob = params["risk_behind_door_prob"]

    H, W = 9, 17
    ct, cost, risk = _empty_grid(H, W)

    TEMPLATE = [
        "1ddd1ddd1ddd1ddd1",
        "1000d000d000d0001",
        "1ddd1ddd1ddd1ddd1",
        "1000d000d000d0001",
        "1ddd1ddd1ddd1ddd1",
        "1000d000d000d0001",
        "1ddd1ddd1ddd1ddd1",
        "1000d000d000d0001",
        "1ddd1ddd1ddd1ddd1",
    ]

    door_positions: list[tuple[int, int]] = []

    # First pass: walls and free
    for r in range(H):
        for c in range(W):
            ch = TEMPLATE[r][c]
            if ch == '1':
                ct[r, c] = CellType.WALL
                cost[r, c] = np.inf

    # Second pass: identify door-slot groups
    ddd_segments: list[list[tuple[int, int]]] = []
    single_d: list[tuple[int, int]] = []

    for r in range(H):
        c = 0
        while c < W:
            if TEMPLATE[r][c] == 'd':
                if c + 2 < W and TEMPLATE[r][c+1] == 'd' and TEMPLATE[r][c+2] == 'd':
                    ddd_segments.append([(r, c), (r, c+1), (r, c+2)])
                    c += 3
                else:
                    single_d.append((r, c))
                    c += 1
            else:
                c += 1

    # ── Gate positions: identify where doors COULD be, but all start OPEN ──
    # gate_positions tracks choke points for risk placement;
    # NO doors are locked — all passages start free.
    # Teacher uses BLOCK to close gates when agent approaches risk.
    gate_positions: list[tuple[int, int]] = []

    # Process ddd segments: pick one gate per segment
    for seg in ddd_segments:
        if rng.random() < door_prob:
            door_idx = rng.integers(0, len(seg))
            gate_positions.append(seg[door_idx])
            # Cell stays NORMAL (free) — no _set_door()

    # Process single d: gate or free
    for r, c in single_d:
        if rng.random() < door_prob:
            gate_positions.append((r, c))
            # Cell stays NORMAL (free) — no _set_door()

    # ── Risk behind gates: risky cell adjacent to each gate ──
    for gr, gc in gate_positions:
        candidates = []
        for nr, nc in [(gr-1, gc), (gr+1, gc), (gr, gc-1), (gr, gc+1)]:
            if 0 <= nr < H and 0 <= nc < W:
                if ct[nr, nc] == CellType.NORMAL and (nr, nc) not in gate_positions:
                    candidates.append((nr, nc))
        if candidates and rng.random() < risk_behind_prob:
            rc = candidates[rng.integers(0, len(candidates))]
            _set_risky(ct, cost, risk, [rc], risk_val)

    # ── Start: column 0 (break wall at a room row) ──
    room_rows = [1, 3, 5, 7]
    start_row = rng.choice(room_rows)
    agent_start = (start_row, 0)
    ct[start_row, 0] = CellType.NORMAL
    cost[start_row, 0] = 1.0

    # ── Goal: column 16 (break wall at a room row) ──
    goal_row = rng.choice(room_rows)
    target_pos = (goal_row, W - 1)
    ct[goal_row, W - 1] = CellType.NORMAL
    cost[goal_row, W - 1] = 1.0

    # ── Object: center of grid ──
    center_open = [(r, c) for r in room_rows for c in [7, 8, 9]
                   if ct[r, c] in (CellType.NORMAL, CellType.RISKY)]
    object_spawn = (center_open[rng.integers(0, len(center_open))]
                    if center_open else (3, 8))

    gm = _build_gridmap(H, W, ct, cost, risk,
                         agent_start, object_spawn, target_pos, door_positions)

    # ── Compute shortest path for time_ratio ──
    from ..agents.planner_astar import bounded_astar
    full_passable = np.ones((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if ct[r, c] == CellType.WALL:
                full_passable[r, c] = False
    # Unlock all doors for shortest-path calc
    for dr, dc in door_positions:
        full_passable[dr, dc] = True
    zero_arr = np.zeros((H, W))
    sp1 = bounded_astar(
        agent_start, object_spawn,
        cost, zero_arr, zero_arr,
        budget=500, lambda_risk=0.0, passable_mask=full_passable,
    )
    sp2 = bounded_astar(
        object_spawn, target_pos,
        cost, zero_arr, zero_arr,
        budget=500, lambda_risk=0.0, passable_mask=full_passable,
    )
    shortest = 30  # fallback
    if sp1 and sp2:
        shortest = (len(sp1) - 1) + (len(sp2) - 1)
    max_steps = max(int(time_ratio * shortest), shortest + 3)

    cfg = FamilyConfig(
        max_steps=max_steps,
        risk_budget=params["risk_budget"],
        prior_risk_mean=0.1,
        prior_risk_var=0.25,
        search_budget=40,
        budget_class=8,
    )
    return gm, cfg


# ══════════════════════════════════════════════════════════════════════
# Family F: Deceptive Fork — MVP for teacher intervention validation
# ══════════════════════════════════════════════════════════════════════
#
# Fixed 6×8 grid with two paths from S to G:
#
#   0 1 2 3 4 5 6 7
# 0 W W W W W W W W
# 1 W . . a T . . W     Path A (bait): trap at (1,4), ρ*=1.0
# 2 S . F . . G . W     S(2,0), Fork(2,2), Goal(2,5)
# 3 W . . . . . . W
# 4 W . . D b . . W     Path B (safe): door at (4,3)
# 5 W W W W W W W W
#
# Path A: S(2,0)→(2,1)→(2,2=fork)→(1,2)→(1,3)→(1,4=TRAP)→(1,5)→(2,5=G)
#   L_A = 6 total, fork→G_A = 4 steps
#   fork→trap Manhattan = |1-2| + |4-2| = 3 ✓
#
# Path B: fork(2,2)→(3,2)→(4,2)→(4,3=DOOR)→(4,4)→(3,4)→(3,5)→(2,5=G)
#   L_B = 8 total, fork→G_B = 6 steps
#
# T_max ≈ 9: B reachable (8<9), backtrack from A impossible.
#

DIFFICULTY_PARAMS["deceptive_fork"] = {
    "easy":   {"trap_risk": 1.0, "time_ratio": 1.15, "risk_budget": 1.0},
    "medium": {"trap_risk": 1.0, "time_ratio": 1.05, "risk_budget": 1.0},
    "hard":   {"trap_risk": 0.85, "time_ratio": 1.0,  "risk_budget": 0.5},
}


def generate_deceptive_fork(
    seed: int = 0,
    difficulty: DifficultyLevel = "medium",
    with_door: bool = True,
) -> tuple[GridMap, FamilyConfig]:
    """
    Deceptive Fork — fixed-geometry MVP for teacher intervention validation.

    6x8 grid, two paths from S(2,0) to G(2,5):
      Path A (bait): 6 steps, trap at (1,4)
      Path B (safe): 8 steps, door at (4,3) if with_door=True

    Fork at (2,2), fork->trap Manhattan=3.

    Env-A (with_door=False): tests WARN effectiveness in isolation
    Env-B (with_door=True):  tests WARN+UNLOCK complementarity
    """
    rng = np.random.default_rng(seed)
    params = DIFFICULTY_PARAMS["deceptive_fork"][difficulty]
    trap_risk = params["trap_risk"]

    H, W = 6, 8
    ct, cost, risk = _empty_grid(H, W)

    # Start with all walls
    ct[:] = CellType.WALL
    cost[:] = np.inf

    # ── Free cells ──
    free_cells = [
        # Start and approach to fork
        (2, 0), (2, 1), (2, 2),  # S -> fork
        # Path A (bait): fork -> (1,2)->(1,3)->(1,4=trap)->(1,5)-> G
        (1, 2), (1, 3), (1, 4), (1, 5),
        # Goal
        (2, 5),
        # Path B (safe): fork -> (3,2)->(4,2)->(4,3)->(4,4)->(3,4)->(3,5)-> G
        (3, 2), (4, 2), (4, 3), (4, 4), (3, 4), (3, 5),
    ]
    for r, c in free_cells:
        ct[r, c] = CellType.NORMAL
        cost[r, c] = 1.0

    # ── Trap cell on Path A ──
    trap_cell = (1, 4)
    _set_risky(ct, cost, risk, [trap_cell], trap_risk)

    # ── Door on Path B (optional) ──
    door_cell = (4, 3)
    door_positions = []
    if with_door:
        _set_door(ct, cost, [door_cell])
        door_positions = [door_cell]

    # ── Positions ──
    agent_start = (2, 0)
    target_pos = (2, 5)
    object_spawn = target_pos  # single-goal mode

    # ── Geometry assertions ──
    fork = (2, 2)
    d_fork_trap = abs(trap_cell[0] - fork[0]) + abs(trap_cell[1] - fork[1])
    assert d_fork_trap >= 3, f"fork->trap Manhattan must be >=3, got {d_fork_trap}"

    gm = _build_gridmap(H, W, ct, cost, risk,
                         agent_start, object_spawn, target_pos, door_positions)

    # Time budget: L_B = 8 (no door) / L_B = 8 + unlock step (with door)
    shortest_safe = 8
    t_max = max(int(params["time_ratio"] * shortest_safe), shortest_safe + 1)

    cfg = FamilyConfig(
        max_steps=t_max,
        risk_budget=params["risk_budget"],
        prior_risk_mean=0.02,
        prior_risk_var=0.20,
        search_budget=30,
        budget_class=8,
    )
    return gm, cfg


# ── Registry ─────────────────────────────────────────────────────────

FAMILY_GENERATORS = {
    "semantic_trap": generate_semantic_trap,
    "planning_trap": generate_planning_trap,
    "exploration_useful": generate_exploration_useful,
    "mixed": generate_mixed,
    "door_lattice_sanity": generate_door_lattice_sanity,
    "deceptive_fork": generate_deceptive_fork,
}

FAMILY_NAMES = list(FAMILY_GENERATORS.keys())


