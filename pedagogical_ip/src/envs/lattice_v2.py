"""
Lattice Trap V2 — forced-choice segments with feature vectors.

All doors start OPEN. Tutor CLOSES risky-entry doors to force safe lane.

KEY DESIGN: Both lanes span same columns. Safe lane has internal
zigzag that makes its path LONGER than the straight risky lane.

Per segment (5 rows × Ls columns):
  Row 0: wall
  Row 1: risky lane — straight path, Ls cells
  Row 2: wall inside segment (lane separator)
  Row 3: safe lane — zigzag path, Ls + 2*detour cells

The safe lane zigzag:
  Normal:  →→→→→→     (Ls cells straight)
  Zigzag:  →→→↓→→→    (has to go down to row 4, across, back up)
           ··· ↑     row 4 detour cells

We use a 7-row grid to accommodate the detour:
  Row 0: wall
  Row 1: risky lane
  Row 2: wall (separator)  
  Row 3: safe lane main
  Row 4: wall except at detour columns
  Row 5: safe lane detour
  Row 6: wall
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np
from collections import deque

from .map_generator import CellType, GridMap
from .map_families import _empty_grid, _build_gridmap, FamilyConfig

FEATURE_DIM = 4
F_LANE_ID = 0
F_GATE_FLAG = 1
F_TEXTURE_1 = 2
F_TEXTURE_2 = 3


@dataclass
class SegmentMeta:
    index: int
    col_start: int
    col_end: int
    risky_row: int        # always 1
    safe_row: int         # always 3
    L_risky: int          # path length = Ls
    L_safe: int           # path length = Ls + 2*detour
    detour_len: int       # number of detour cols
    risky_cells: list[tuple[int, int]]
    safe_cells: list[tuple[int, int]]
    risky_entry_gate: tuple[int, int]
    safe_entry_gate: tuple[int, int]
    trap_cell: tuple[int, int] | None
    weak_cue_cells: list[tuple[int, int]]


@dataclass
class LatticeV2Meta:
    segments: list[SegmentMeta]
    all_gate_cells: list[tuple[int, int]]
    all_door_positions: list[tuple[int, int]]
    shortest_any: int
    shortest_safe: int
    cell_features: np.ndarray
    world_weights: Optional[object] = None   # WorldWeights when latent_mode=True
    latent_mode: bool = False
    # ── DTMB extension (Option A: optional fields, backward-compatible) ──
    decision_stages: int = 1
    decision_points_by_stage: Optional[list[list]] = None
    merge_points_by_stage: Optional[list[list]] = None
    commitment_points_by_stage: Optional[list[list]] = None
    reveal_events_by_stage: Optional[list[list]] = None
    goal_cue_cells: Optional[list] = None
    temptation_cue_cells: Optional[list] = None
    door_positions_by_stage: Optional[list[list]] = None
    belt_cells_by_stage: Optional[list[list]] = None
    safe_routes: Optional[list] = None
    risky_routes: Optional[list] = None
    route_count: int = 0
    route_depths: Optional[list[int]] = None
    slack_profile: Optional[dict] = None
    dominant_bottleneck_gt_by_stage: Optional[list[str]] = None
    recommended_primary_lever_by_stage: Optional[list[str]] = None


def generate_lattice_v2(
    seed: int = 0,
    difficulty: Literal["easy", "medium", "hard"] = "medium",
    n_segments: int = 3,
    latent_mode: bool = False,
) -> tuple[GridMap, FamilyConfig, LatticeV2Meta]:
    """
    7-row grid. Each segment: same column span, different path lengths.

    Risky lane (row 1): straight, Ls cells.
    Safe lane (rows 3+5): zigzag with detour through row 5.
    Path length difference = 2 * detour per segment.

    Entry: both lanes accessible from row 2 at seg_start.
    Exit: both lanes exit to row 2 at seg_end.
    """
    rng = np.random.default_rng(seed)

    # Segment widths (number of columns)
    seg_widths = rng.integers(5, 8, size=n_segments)  # 5, 6, or 7 cols
    # Detour length per segment: 1-2 cols
    detour_lens = np.ones(n_segments, dtype=int)  # fixed dt=1 → delta ≈ 4-6

    CORR_W = 1
    W = 1 + CORR_W
    for i in range(n_segments):
        W += int(seg_widths[i])
        W += CORR_W
    W += 1

    H = 7
    ct, cost, risk = _empty_grid(H, W)
    features = np.full((H, W, FEATURE_DIM), 0.5, dtype=np.float64)

    # Default: all wall
    ct[:, :] = CellType.WALL
    cost[:, :] = np.inf

    # Row 3 as corridor between segments (merge row)
    # Actually: use row 2 as corridor for S/G and between-segment movement
    # Let me keep row 2 area for the corridor approach
    # Rows: 0=wall, 1=risky, 2=corridor/wall, 3=safe_main, 4=wall, 5=safe_detour, 6=wall
    
    # Corridor: row 2 passable between segments only
    # Inside segments: row 2 is wall (forces lane choice)
    # Between segments: row 2 is passable (corridor)
    for c in range(1, W - 1):
        ct[2, c] = CellType.NORMAL
        cost[2, c] = 1.0
        features[2, c] = np.array([0.5, 1.0, 0.0, 0.0])

    segments_meta = []
    all_gates = []
    all_doors = []

    col_cursor = 1 + CORR_W

    for seg_i in range(n_segments):
        sw = int(seg_widths[seg_i])
        dt = int(detour_lens[seg_i])
        
        seg_start = col_cursor
        seg_end = col_cursor + sw - 1

        # Wall row 2 inside segment (force onto lanes)
        for c in range(seg_start, seg_end + 1):
            ct[2, c] = CellType.WALL
            cost[2, c] = np.inf

        # Punch entry and exit holes in row 2
        ct[2, seg_start] = CellType.NORMAL
        cost[2, seg_start] = 1.0
        ct[2, seg_end] = CellType.NORMAL
        cost[2, seg_end] = 1.0

        # ── Entry gates ──
        for r in [1, 3]:
            ct[r, seg_start] = CellType.NORMAL
            cost[r, seg_start] = 1.0
            features[r, seg_start] = np.array([
                0.0 if r == 1 else 1.0, 1.0, 0.0, 0.0])

        risky_entry = (1, seg_start)
        safe_entry = (3, seg_start)
        all_gates.extend([risky_entry, safe_entry])
        all_doors.append(risky_entry)

        # ── Exit gates ──
        for r in [1, 3]:
            ct[r, seg_end] = CellType.NORMAL
            cost[r, seg_end] = 1.0
            features[r, seg_end] = np.array([
                0.0 if r == 1 else 1.0, 1.0, 0.0, 0.0])

        # ── Risky lane (row 1): straight ──
        risky_cells = []
        for c in range(seg_start + 1, seg_end):
            ct[1, c] = CellType.NORMAL
            cost[1, c] = 1.0
            risky_cells.append((1, c))

        # ── Safe lane (row 3): main with zigzag ──
        safe_cells = []
        # Detour position: at midpoint of segment
        detour_start = seg_start + (sw // 2) - dt
        detour_end = detour_start + dt

        for c in range(seg_start + 1, seg_end):
            if detour_start <= c < detour_end:
                # Gap in row 3: force detour through rows 4,5
                # Row 3 is WALL at detour columns
                ct[3, c] = CellType.WALL
                cost[3, c] = np.inf
            else:
                ct[3, c] = CellType.NORMAL
                cost[3, c] = 1.0
                safe_cells.append((3, c))

        # Detour vertical entries: row 3→4→5 at detour_start column
        # and row 5→4→3 at detour_end column
        ct[4, detour_start] = CellType.NORMAL
        cost[4, detour_start] = 1.0
        safe_cells.append((4, detour_start))

        ct[4, detour_end] = CellType.NORMAL
        cost[4, detour_end] = 1.0
        safe_cells.append((4, detour_end))

        # Detour horizontal: row 5 from detour_start to detour_end
        for c in range(detour_start, detour_end + 1):
            ct[5, c] = CellType.NORMAL
            cost[5, c] = 1.0
            safe_cells.append((5, c))

        # Detour entry/exit on row 3: cells just before/after the gap
        # Already handled: row 3 cells adjacent to detour are passable
        # from the main loop. The agent goes:
        #   (3, detour_start-1) → (4, detour_start) ↓
        #   → (5, detour_start) → ... → (5, detour_end) ↓
        #   → (4, detour_end) → (3, detour_end) ↑
        # Total extra cost: 2 (verticals) + dt (horizontal row 5) extra

        # But wait: I want row 3 at detour_start-1 to connect to row 4 at
        # detour_start. That requires (3, detour_start-1) → (3, detour_start)
        # → but (3, detour_start) is walled! So agent can't reach (4, detour_start).
        # Fix: agent goes (3, detour_start-1) → (4, detour_start-1) 
        # but (4, detour_start-1) is wall...
        
        # Better: use detour_start-1 as the DOWN column
        dc_down = detour_start  # the first gap column
        dc_up = detour_end      # the last gap column
        
        # Row 3 at dc_down is WALL (the gap). 
        # Row 3 at dc_down-1 is passable (just before gap).
        # We need vertical from row 3 to row 4 at the gap boundary.
        # Solution: at dc_down-1, open row 4 for vertical movement
        if dc_down - 1 >= seg_start:
            ct[4, dc_down - 1] = CellType.NORMAL
            cost[4, dc_down - 1] = 1.0
            safe_cells.append((4, dc_down - 1))
            # Connect to row 5
            ct[5, dc_down - 1] = CellType.NORMAL
            cost[5, dc_down - 1] = 1.0
            safe_cells.append((5, dc_down - 1))

        # Similarly at dc_up+1 (or dc_up if it's the last gap col)
        if dc_up < seg_end:
            ct[4, dc_up + 1] = CellType.NORMAL
            cost[4, dc_up + 1] = 1.0
            safe_cells.append((4, dc_up + 1))
            ct[5, dc_up + 1] = CellType.NORMAL
            cost[5, dc_up + 1] = 1.0
            safe_cells.append((5, dc_up + 1))

        # Risky lane path length = sw - 1 (straight from start to end)
        L_risky = sw - 1 + 2  # +2 for entry/exit verticals (row 2→1→2)
        # Safe lane path = (sw - 1 - dt) + (2 + dt + 2) + 2 verticals
        # Main: (sw - 1 - dt) cells on row 3
        # Detour: 2 verticals + dt+2 cells on row 5 + 2 verticals back
        # Actually just let BFS compute the true path lengths

        # ── Features and risk ──
        trap_cell = None
        weak_cue_cells = []
        has_trap = True
        if difficulty == "easy":
            has_trap = rng.random() < 0.50
        elif difficulty == "medium":
            has_trap = rng.random() < 0.70
        else:  # hard
            has_trap = rng.random() < 0.90

        if has_trap and len(risky_cells) >= 2:
            trap_idx = rng.integers(0, len(risky_cells))
            trap_cell = risky_cells[trap_idx]
            for i, (r, c) in enumerate(risky_cells):
                ct[r, c] = CellType.RISKY
                if i == trap_idx:
                    features[r, c] = _trap_feature(rng, 0.0)
                    risk[r, c] = rng.uniform(0.30, 0.50)  # stochastic, survivable
                elif abs(i - trap_idx) <= 1:
                    features[r, c] = _weak_cue_feature(rng, 0.0)
                    risk[r, c] = rng.uniform(0.15, 0.25)
                    weak_cue_cells.append((r, c))
                else:
                    features[r, c] = _lane_feature(rng, 0.0, True)
                    risk[r, c] = rng.uniform(0.08, 0.15)
        else:
            for r, c in risky_cells:
                ct[r, c] = CellType.RISKY
                features[r, c] = _lane_feature(rng, 0.0, False)
                risk[r, c] = rng.uniform(0.05, 0.10)

        for r, c in safe_cells:
            lid = 0.0 if r == 1 else 1.0
            features[r, c] = _safe_feature(rng, lid)

        segments_meta.append(SegmentMeta(
            index=seg_i, col_start=seg_start, col_end=seg_end,
            risky_row=1, safe_row=3,
            L_risky=0, L_safe=0,  # computed by BFS below
            detour_len=dt,
            risky_cells=risky_cells, safe_cells=safe_cells,
            risky_entry_gate=risky_entry,
            safe_entry_gate=safe_entry,
            trap_cell=trap_cell,
            weak_cue_cells=weak_cue_cells,
        ))

        col_cursor = seg_end + 1 + CORR_W

    agent_start = (2, 1)
    target_pos = (2, W - 2)
    object_spawn = target_pos

    gm = _build_gridmap(H, W, ct, cost, risk,
                         agent_start, object_spawn, target_pos, [])

    shortest_any = _bfs_len(gm, agent_start, target_pos, set())
    risky_gates = set(seg.risky_entry_gate for seg in segments_meta)
    shortest_safe = _bfs_len(gm, agent_start, target_pos, risky_gates)

    base = shortest_safe if shortest_safe < 999 else 40
    if difficulty == "easy":
        t_max = int(1.5 * base)
    elif difficulty == "medium":
        t_max = int(1.4 * base)
    else:
        t_max = int(1.3 * base)
    t_max = max(t_max, base + 2)

    cfg = FamilyConfig(
        max_steps=t_max, risk_budget=1.0,
        prior_risk_mean=0.02, prior_risk_var=0.20,
        search_budget=30, budget_class=8,
    )

    # ── Latent mode: derive cost and risk from features ──
    ww = None
    if latent_mode:
        from ..agents.cost_risk_model import generate_world_weights, WorldWeights
        ww = generate_world_weights(rng, d=FEATURE_DIM)
        for r in range(H):
            for c in range(W):
                if ct[r, c] == CellType.WALL:
                    continue
                z = features[r, c]
                cost[r, c] = ww.true_cost(z)
                risk[r, c] = ww.true_risk(z)
        # Rebuild gridmap with latent-derived cost/risk
        gm = _build_gridmap(H, W, ct, cost, risk,
                            agent_start, object_spawn, target_pos, [])

    meta = LatticeV2Meta(
        segments=segments_meta,
        all_gate_cells=all_gates,
        all_door_positions=all_doors,
        shortest_any=shortest_any,
        shortest_safe=shortest_safe,
        cell_features=features,
        world_weights=ww,
        latent_mode=latent_mode,
    )
    return gm, cfg, meta


def _safe_feature(rng, lid):
    return np.array([lid, 0.0, rng.uniform(0.0, 0.1), rng.uniform(0.0, 0.1)])
def _trap_feature(rng, lid):
    return np.array([lid, 0.0, rng.uniform(0.80, 0.95), rng.uniform(0.70, 0.90)])
def _weak_cue_feature(rng, lid):
    return np.array([lid, 0.0, rng.uniform(0.30, 0.50), rng.uniform(0.20, 0.40)])
def _lane_feature(rng, lid, mild_cue):
    if mild_cue:
        return np.array([lid, 0.0, rng.uniform(0.10, 0.20), rng.uniform(0.05, 0.15)])
    return np.array([lid, 0.0, rng.uniform(0.0, 0.10), rng.uniform(0.0, 0.10)])


def _bfs_len(gm, start, goal, avoid):
    H, W = gm.height, gm.width
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        pos, d = queue.popleft()
        if pos == goal:
            return d
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) in visited:
                continue
            if gm.cell_types[nr, nc] == CellType.WALL:
                continue
            if (nr, nc) in avoid:
                continue
            visited.add((nr, nc))
            queue.append(((nr, nc), d + 1))
    return 999
