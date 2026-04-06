"""GTET-L: Goal–Preference–Temptation Entanglement Lattice.

Family 2 of the complex scenario hierarchy. Generates lattice scenarios where
the same prefix behavior can be explained by multiple latent hypotheses
(g, θ, z), forcing the joint posterior q(g,θ,z) to maintain ambiguity until
staged reveal events disambiguate.

Key design principle: NOT "bigger maps", but "deeper entanglement". The
generator creates 2-merge / 3-stage lattices where early-stage branch choices
are consistent with multiple (goal, preference, temptation) combinations.

Reuses existing GridMap, CellType, 4D features, LatticeV2Meta.
Does NOT modify feature dimensionality; cue information is stored in sidecar
metadata arrays (goal_cue_tags, temptation_cue_tags, preference_cue_tags).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .map_generator import CellType, GridMap
from .map_families import _empty_grid, _build_gridmap, FamilyConfig
from .lattice_v2 import (
    LatticeV2Meta, SegmentMeta,
    FEATURE_DIM, _bfs_len,
    _safe_feature, _trap_feature, _weak_cue_feature, _lane_feature,
)


# ══════════════════════════════════════════════════════════════════════
# Default parameters per difficulty
# ══════════════════════════════════════════════════════════════════════

GTET_PARAMS = {
    "easy": {
        "H": 13, "W": 40,
        "stage1_branch_count": 3,
        "stage2_branch_count": 2,    # per parent
        "stage3_belt_width": 3,
        # Entanglement controls
        "goal_cue_reliability": 0.70,   # how clearly goal cues identify g
        "goal_cue_leadlag": 2,          # goal cue appears 2 cols before tempt cue
        "lure_strength": 0.40,
        "tempt_offset_z": 0.30,
        "goal_ambiguity": 0.35,         # overlap of goal cues across branches
        "preference_cue_strength": 0.60,
        # Resolution
        "deadline_slack_final": 1.25,
        "has_locked_fast_lane": False,
        "belt_risk": 0.25,
        "search_budget": 30,
    },
    "medium": {
        "H": 15, "W": 50,
        "stage1_branch_count": 3,
        "stage2_branch_count": 2,
        "stage3_belt_width": 4,
        "goal_cue_reliability": 0.50,
        "goal_cue_leadlag": 0,          # simultaneous — higher ambiguity
        "lure_strength": 0.65,
        "tempt_offset_z": 0.50,
        "goal_ambiguity": 0.55,
        "preference_cue_strength": 0.45,
        "deadline_slack_final": 1.15,
        "has_locked_fast_lane": True,
        "belt_risk": 0.35,
        "search_budget": 35,
    },
    "hard": {
        "H": 17, "W": 60,
        "stage1_branch_count": 3,
        "stage2_branch_count": 3,
        "stage3_belt_width": 5,
        "goal_cue_reliability": 0.35,
        "goal_cue_leadlag": -2,         # tempt cue BEFORE goal cue
        "lure_strength": 0.85,
        "tempt_offset_z": 0.70,
        "goal_ambiguity": 0.75,
        "preference_cue_strength": 0.30,
        "deadline_slack_final": 1.08,
        "has_locked_fast_lane": True,
        "belt_risk": 0.45,
        "search_budget": 40,
    },
}


def _resolve_gtet_config(difficulty: str, user_cfg: dict | None = None) -> dict:
    cfg = dict(GTET_PARAMS[difficulty])
    if user_cfg:
        for k, v in user_cfg.items():
            if k in cfg:
                cfg[k] = v
    return cfg


# ══════════════════════════════════════════════════════════════════════
# BFS / passability utilities
# ══════════════════════════════════════════════════════════════════════

def _bfs_gtet(ct: np.ndarray, start: tuple, goal: tuple,
              avoid: set | None = None) -> int:
    """BFS shortest path on cell_types grid. Returns 999 if unreachable."""
    H, W = ct.shape
    avoid = avoid or set()
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        (r, c), d = queue.popleft()
        if (r, c) == goal:
            return d
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                if (nr, nc) in avoid:
                    continue
                if ct[nr, nc] in (CellType.WALL, CellType.LOCKED_DOOR):
                    continue
                visited.add((nr, nc))
                queue.append(((nr, nc), d + 1))
    return 999


def _allocate_rows(H: int, n: int, center: int, gap: int = 2) -> list[int]:
    """Allocate vertically spaced rows centered around center."""
    if n == 1:
        return [center]
    total = (n - 1) * gap
    top = max(1, center - total // 2)
    if top + total >= H - 1:
        top = max(1, H - 2 - total)
    rows = [top + i * gap for i in range(n)]
    return sorted(max(1, min(H - 2, r)) for r in rows)


# ══════════════════════════════════════════════════════════════════════
# Feature helpers for GTET cue cells
# ══════════════════════════════════════════════════════════════════════

def _goal_cue_feature(rng: np.random.Generator, lid: float,
                      reliability: float) -> np.ndarray:
    """Goal cue cell: moderately distinctive textures."""
    noise = 1.0 - reliability
    t1 = np.clip(0.2 + reliability * 0.4 + rng.uniform(-0.1, 0.1) * noise,
                 0.05, 0.85)
    t2 = np.clip(0.15 + reliability * 0.3 + rng.uniform(-0.1, 0.1) * noise,
                 0.05, 0.80)
    return np.array([lid, rng.uniform(0.3, 0.5), t1, t2])


def _tempt_cue_feature(rng: np.random.Generator, lid: float,
                       strength: float) -> np.ndarray:
    """Temptation cue cell: attractive textures (high gate_flag analog)."""
    return np.array([lid, rng.uniform(0.6, 0.9) * strength,
                     rng.uniform(0.15, 0.35), rng.uniform(0.10, 0.25)])


def _pref_cue_feature(rng: np.random.Generator, lid: float,
                      pref_type: str, strength: float) -> np.ndarray:
    """Preference cue cell: θ-indicating textures."""
    if pref_type == "safe":
        t1 = rng.uniform(0.02, 0.10) * strength
        t2 = rng.uniform(0.02, 0.10) * strength
    else:  # "shiny"
        t1 = rng.uniform(0.40, 0.70) * strength
        t2 = rng.uniform(0.50, 0.80) * strength
    return np.array([lid, rng.uniform(0.1, 0.3), t1, t2])


def _belt_feature(rng: np.random.Generator, lid: float,
                  risk_level: float) -> np.ndarray:
    """Terminal hazard belt cell."""
    t1 = np.clip(rng.uniform(0.7, 0.95) * risk_level / 0.6, 0.0, 0.99)
    t2 = np.clip(rng.uniform(0.6, 0.85) * risk_level / 0.6, 0.0, 0.99)
    return np.array([lid, 0.0, t1, t2])


# ══════════════════════════════════════════════════════════════════════
# Stage band computation
# ══════════════════════════════════════════════════════════════════════

def _compute_gtet_bands(W: int) -> list[tuple[int, int]]:
    """Split width into 3 stage bands with merge zones.

    Layout: [trunk=3] [stage1=30%] [merge1=3] [stage2=35%] [merge2=3] [stage3=rest]
    """
    trunk = 3
    merge_w = 3
    usable = W - trunk - 1 - 2 * merge_w  # -1 for border
    s1_w = max(6, int(usable * 0.30))
    s2_w = max(6, int(usable * 0.35))
    s3_w = max(6, usable - s1_w - s2_w)

    c = trunk
    b1 = (c, c + s1_w)
    c = b1[1] + merge_w
    b2 = (c, c + s2_w)
    c = b2[1] + merge_w
    b3 = (c, c + s3_w)
    return [b1, b2, b3]


# ══════════════════════════════════════════════════════════════════════
# Main generator
# ══════════════════════════════════════════════════════════════════════

@dataclass
class GTETMeta:
    """GTET-specific sidecar metadata (lives alongside LatticeV2Meta)."""
    # Cue sidecar arrays (H×W)
    goal_cue_tags: np.ndarray = None        # int: -1=none, 0,1,2=subgoal
    temptation_cue_tags: np.ndarray = None  # float: 0.0=none, >0=lure intensity
    preference_cue_tags: np.ndarray = None  # int: -1=none, 0=safe, 1=shiny

    # Route-level entanglement
    goal_consistent_routes: dict = field(default_factory=dict)  # subgoal_id -> [route_idx]
    temptation_preferred_routes: dict = field(default_factory=dict)  # z_bucket -> [route_idx]
    latent_explanation_overlap: list = field(default_factory=list)  # per-stage ambiguity records

    # Structure
    subgoal_reveal_order: list = field(default_factory=list)  # [(col, subgoal_id), ...]
    decision_points_by_stage: list = field(default_factory=list)
    merge_points: list = field(default_factory=list)  # [(row, col), ...]


def generate_gtet_lattice(
    seed: int,
    difficulty: str = "medium",
    latent_mode: bool = True,
    user_cfg: dict | None = None,
    **kwargs,
) -> tuple[GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig]:
    """Generate a GTET-L scenario.

    Returns (GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig).
    GTET-specific metadata is stored in meta.gtet_meta (GTETMeta).
    """
    rng = np.random.default_rng(seed)
    cfg = _resolve_gtet_config(difficulty, user_cfg)

    H, W = cfg["H"], cfg["W"]
    ct, cost, risk = _empty_grid(H, W)
    features = np.full((H, W, FEATURE_DIM), 0.5, dtype=np.float64)
    ct[:, :] = CellType.WALL
    cost[:, :] = np.inf

    # Sidecar cue tags
    goal_tags = np.full((H, W), -1, dtype=np.int8)
    tempt_tags = np.zeros((H, W), dtype=np.float64)
    pref_tags = np.full((H, W), -1, dtype=np.int8)

    # ── Trunk corridor (leftmost cols) ──
    center = H // 2
    agent_start = (center, 1)
    for c in range(1, 3):
        ct[center, c] = CellType.NORMAL
        cost[center, c] = 1.0
        features[center, c] = np.array([0.5, 1.0, 0.0, 0.0])

    # ── Compute bands ──
    bands = _compute_gtet_bands(W)
    n_s1 = cfg["stage1_branch_count"]
    n_s2_per = cfg["stage2_branch_count"]

    # ═══ Stage 1: Weak goal-cue fork ═══
    s1_start, s1_end = bands[0]
    s1_rows = _allocate_rows(H, n_s1, center)

    # Fork point at trunk end
    fork_col = s1_start
    for r in s1_rows:
        # Vertical connector from center to each branch row
        r_lo, r_hi = sorted([center, r])
        for rr in range(r_lo, r_hi + 1):
            ct[rr, fork_col] = CellType.NORMAL
            cost[rr, fork_col] = 1.0
            features[rr, fork_col] = np.array([0.5, 1.0, 0.0, 0.0])

    # Subgoal assignment: each branch gets a primary subgoal
    n_subgoals = min(n_s1, 3)  # up to 3 subgoals
    subgoal_labels = [f"sg_{i}" for i in range(n_subgoals)]
    branch_subgoals = []
    for br_i in range(n_s1):
        branch_subgoals.append(br_i % n_subgoals)

    # Randomize subgoal-to-branch mapping
    rng.shuffle(branch_subgoals)

    s1_exits = []
    s1_route_cells = []
    decision_pts_s1 = []
    tempt_cells_s1 = []
    goal_cue_cells_by_sg = {i: [] for i in range(n_subgoals)}

    goal_rel = cfg["goal_cue_reliability"]
    leadlag = cfg["goal_cue_leadlag"]
    ambiguity = cfg["goal_ambiguity"]
    lure_str = cfg["lure_strength"]

    for br_i, row in enumerate(s1_rows):
        sg_id = branch_subgoals[br_i]
        route_cells = []

        for c in range(s1_start, s1_end):
            ct[row, c] = CellType.NORMAL
            cost[row, c] = 1.0
            lid = row / (H - 1)

            # Position within stage
            depth = c - s1_start
            stage_len = s1_end - s1_start

            # Goal cue placement: appears at goal_cue_leadlag offset
            goal_cue_col = s1_start + max(0, stage_len // 3 + leadlag)
            tempt_cue_col = s1_start + max(0, stage_len // 3 - leadlag)

            if c == s1_start:
                decision_pts_s1.append((row, c))

            if c >= goal_cue_col and c < goal_cue_col + 3:
                # Goal cue zone
                features[row, c] = _goal_cue_feature(rng, lid, goal_rel)
                goal_tags[row, c] = sg_id
                goal_cue_cells_by_sg[sg_id].append((row, c))

                # Ambiguity: also place weak cues for OTHER subgoals
                if rng.random() < ambiguity:
                    other_sg = (sg_id + 1) % n_subgoals
                    goal_tags[row, c] = other_sg  # ambiguous!
            elif c >= tempt_cue_col and c < tempt_cue_col + 2:
                # Temptation cue zone (only on some branches)
                if br_i >= n_s1 - 1 or rng.random() < 0.5:
                    features[row, c] = _tempt_cue_feature(rng, lid, lure_str)
                    tempt_tags[row, c] = lure_str
                    tempt_cells_s1.append((row, c))
                else:
                    features[row, c] = _safe_feature(rng, lid)
            else:
                features[row, c] = _safe_feature(rng, lid)

            route_cells.append((row, c))

        s1_exits.append((row, s1_end - 1))
        s1_route_cells.append(route_cells)

    # ═══ Merge 1: converge branches before Stage 2 ═══
    merge1_col = s1_end
    merge1_end = bands[1][0]
    merge1_center = center

    # Vertical merge column
    for r in range(min(s1_rows), max(s1_rows) + 1):
        for c in range(merge1_col, merge1_end):
            ct[r, c] = CellType.NORMAL
            cost[r, c] = 1.0
            features[r, c] = np.array([r / (H - 1), 0.8, 0.1, 0.1])

    # Horizontal corridor at center through merge
    for c in range(merge1_col, merge1_end):
        ct[center, c] = CellType.NORMAL
        cost[center, c] = 1.0
        features[center, c] = np.array([0.5, 1.0, 0.0, 0.0])

    merge_pts_1 = [(center, merge1_col)]

    # ═══ Stage 2: Temptation fork ═══
    s2_start, s2_end = bands[1]
    total_s2 = n_s2_per * 2  # at least 2 groups
    total_s2 = min(total_s2, (H - 2) // 2)  # fit vertically
    s2_rows = _allocate_rows(H, total_s2, center)

    # Fork at start of stage 2
    for r in s2_rows:
        r_lo, r_hi = sorted([center, r])
        for rr in range(r_lo, r_hi + 1):
            ct[rr, s2_start] = CellType.NORMAL
            cost[rr, s2_start] = 1.0
            features[rr, s2_start] = np.array([rr / (H - 1), 1.0, 0.0, 0.0])

    s2_exits = []
    s2_route_cells = []
    decision_pts_s2 = []
    tempt_cells_s2 = []
    pref_cue_cells = []
    pref_strength = cfg["preference_cue_strength"]
    tempt_z = cfg["tempt_offset_z"]

    # Assign each branch a (temptation level, preference type)
    # Key entanglement: some branches are BOTH goal-consistent AND temptation-consistent
    for br_i, row in enumerate(s2_rows):
        route_cells = []
        is_temptation_route = (br_i % 2 == 1)
        pref_type = "shiny" if is_temptation_route else "safe"

        for c in range(s2_start, s2_end):
            ct[row, c] = CellType.NORMAL
            cost[row, c] = 1.0
            lid = row / (H - 1)
            depth = c - s2_start
            stage_len = s2_end - s2_start

            if c == s2_start:
                decision_pts_s2.append((row, c))

            # Temptation cues in middle third
            if stage_len // 3 <= depth < 2 * stage_len // 3:
                if is_temptation_route:
                    features[row, c] = _tempt_cue_feature(rng, lid, lure_str)
                    tempt_tags[row, c] = tempt_z + rng.uniform(-0.1, 0.1)
                    tempt_cells_s2.append((row, c))

                    # ENTANGLEMENT: temptation route also has weak goal cue
                    # (agent could be following goal OR temptation)
                    if rng.random() < ambiguity:
                        sg_alias = rng.integers(0, n_subgoals)
                        goal_tags[row, c] = sg_alias
                else:
                    features[row, c] = _safe_feature(rng, lid)
            # Preference cues in last third
            elif depth >= 2 * stage_len // 3:
                features[row, c] = _pref_cue_feature(rng, lid, pref_type,
                                                      pref_strength)
                pref_tags[row, c] = 0 if pref_type == "safe" else 1
                pref_cue_cells.append((row, c))
            else:
                features[row, c] = _safe_feature(rng, lid)

            route_cells.append((row, c))

        s2_exits.append((row, s2_end - 1))
        s2_route_cells.append(route_cells)

    # ═══ Merge 2: converge before Stage 3 ═══
    merge2_col = s2_end
    merge2_end = bands[2][0]

    for r in range(min(s2_rows), max(s2_rows) + 1):
        for c in range(merge2_col, merge2_end):
            ct[r, c] = CellType.NORMAL
            cost[r, c] = 1.0
            features[r, c] = np.array([r / (H - 1), 0.8, 0.1, 0.1])

    for c in range(merge2_col, merge2_end):
        ct[center, c] = CellType.NORMAL
        cost[center, c] = 1.0

    merge_pts_2 = [(center, merge2_col)]

    # ═══ Stage 3: Resolution (belt + optional fast lane) ═══
    s3_start, s3_end = bands[2]
    belt_width = cfg["stage3_belt_width"]
    belt_risk_val = cfg["belt_risk"]
    has_fast = cfg["has_locked_fast_lane"]

    # Two paths: upper (risky belt) and lower (safe detour)
    upper_row = max(1, center - 2)
    lower_row = min(H - 2, center + 2)
    fast_lane_row = max(1, center - 4) if has_fast else None

    # Upper path (through belt)
    belt_cells = []
    s3_route_cells_upper = []
    for c in range(s3_start, s3_end):
        ct[upper_row, c] = CellType.NORMAL
        cost[upper_row, c] = 1.0
        lid = upper_row / (H - 1)
        depth = c - s3_start

        if depth >= 2 and depth < 2 + belt_width:
            # Belt zone
            ct[upper_row, c] = CellType.RISKY
            risk[upper_row, c] = belt_risk_val + rng.uniform(-0.05, 0.05)
            features[upper_row, c] = _belt_feature(rng, lid, belt_risk_val)
            belt_cells.append((upper_row, c))
        else:
            features[upper_row, c] = _safe_feature(rng, lid)
        s3_route_cells_upper.append((upper_row, c))

    # Lower path (detour, longer, partial belt at different offset)
    s3_route_cells_lower = []
    lower_belt_offset = belt_width + 3  # belt at different column than upper
    lower_belt_width = max(2, belt_width - 1)  # slightly narrower
    lower_belt_risk = belt_risk_val * 0.7  # slightly lower risk
    for c in range(s3_start, s3_end):
        ct[lower_row, c] = CellType.NORMAL
        cost[lower_row, c] = 1.0
        lid = lower_row / (H - 1)
        depth = c - s3_start

        if depth >= lower_belt_offset and depth < lower_belt_offset + lower_belt_width:
            # Lower-path belt zone (different position from upper)
            ct[lower_row, c] = CellType.RISKY
            risk[lower_row, c] = lower_belt_risk + rng.uniform(-0.03, 0.03)
            features[lower_row, c] = _belt_feature(rng, lid, lower_belt_risk)
            belt_cells.append((lower_row, c))
        else:
            features[lower_row, c] = _safe_feature(rng, lid)
        s3_route_cells_lower.append((lower_row, c))

    # Add extra detour length to lower path (zigzag through extra rows)
    detour_mid = s3_start + (s3_end - s3_start) // 2
    extra_row = min(H - 2, lower_row + 2)
    for c in [detour_mid - 1, detour_mid, detour_mid + 1]:
        if s3_start <= c < s3_end:
            ct[extra_row, c] = CellType.NORMAL
            cost[extra_row, c] = 1.0
            features[extra_row, c] = _safe_feature(rng, extra_row / (H - 1))
    # Vertical connectors
    for rr in range(lower_row, extra_row + 1):
        for c in [detour_mid - 1, detour_mid + 1]:
            if s3_start <= c < s3_end:
                ct[rr, c] = CellType.NORMAL
                cost[rr, c] = 1.0

    # Fast lane (locked, very short, safe)
    door_positions = []
    if has_fast and fast_lane_row is not None and fast_lane_row >= 1:
        fast_start = s3_start + 1
        # Gate at start
        ct[fast_lane_row, fast_start] = CellType.LOCKED_DOOR
        cost[fast_lane_row, fast_start] = np.inf
        features[fast_lane_row, fast_start] = np.array([
            fast_lane_row / (H - 1), 1.0, 0.0, 0.0])
        door_positions.append((fast_lane_row, fast_start))

        # Lane cells
        for c in range(fast_start + 1, s3_end):
            ct[fast_lane_row, c] = CellType.NORMAL
            cost[fast_lane_row, c] = 1.0
            features[fast_lane_row, c] = _safe_feature(
                rng, fast_lane_row / (H - 1))

        # Vertical connectors to upper path
        for rr in range(fast_lane_row, upper_row + 1):
            ct[rr, fast_start] = (CellType.LOCKED_DOOR
                                  if rr == fast_lane_row else CellType.NORMAL)
            if rr != fast_lane_row:
                cost[rr, fast_start] = 1.0
            ct[rr, s3_end - 1] = CellType.NORMAL
            cost[rr, s3_end - 1] = 1.0

    # Fork into Stage 3 from merge
    for r in range(min(upper_row, center), max(lower_row, center) + 1):
        ct[r, s3_start] = CellType.NORMAL
        cost[r, s3_start] = 1.0
        features[r, s3_start] = np.array([r / (H - 1), 1.0, 0.0, 0.0])

    # Goal cell
    goal_col = min(s3_end, W - 2)
    target_pos = (center, goal_col)
    # Converge paths at goal
    for r in range(min(upper_row, center), max(lower_row, center) + 1):
        ct[r, goal_col] = CellType.NORMAL
        cost[r, goal_col] = 1.0
        features[r, goal_col] = np.array([r / (H - 1), 1.0, 0.0, 0.0])

    # ═══ Build GridMap ═══
    gm = _build_gridmap(H, W, ct, cost, risk,
                         agent_start, target_pos, target_pos, door_positions)

    # ═══ Path lengths & deadline ═══
    shortest_any = _bfs_gtet(ct, agent_start, target_pos)
    avoid_risky = {(r, c) for r in range(H) for c in range(W)
                   if ct[r, c] == CellType.RISKY}
    avoid_doors = {(r, c) for r in range(H) for c in range(W)
                   if ct[r, c] == CellType.LOCKED_DOOR}
    shortest_safe = _bfs_gtet(ct, agent_start, target_pos,
                               avoid_risky | avoid_doors)
    if shortest_safe >= 999:
        shortest_safe = _bfs_gtet(ct, agent_start, target_pos, avoid_risky)
    if shortest_safe >= 999:
        shortest_safe = shortest_any

    base = max(shortest_safe, shortest_any + 5)
    t_max = max(int(cfg["deadline_slack_final"] * base), base + 3)

    # ═══ Enumerate routes ═══
    s3_path_rows = [upper_row, lower_row]
    if has_fast and fast_lane_row is not None:
        s3_path_rows.append(fast_lane_row)
    stage_rows = [s1_rows, s2_rows, s3_path_rows]
    routes = _enumerate_routes(ct, agent_start, target_pos, H, W,
                               stage_rows=stage_rows)
    route_count = len(routes)

    # ═══ Compute latent explanation overlap ═══
    goal_consistent = _compute_goal_route_consistency(
        routes, goal_tags, n_subgoals)
    tempt_preferred = _compute_tempt_route_preference(
        routes, tempt_tags)
    overlap = _compute_overlap(goal_consistent, tempt_preferred, n_subgoals)

    # ═══ Subgoal reveal order ═══
    reveal_order = []
    for sg_id in range(n_subgoals):
        cells = goal_cue_cells_by_sg[sg_id]
        if cells:
            avg_col = int(np.mean([c for _, c in cells]))
            reveal_order.append((avg_col, f"sg_{sg_id}"))
    reveal_order.sort(key=lambda x: x[0])

    # ═══ GTET sidecar metadata ═══
    gtet_meta = GTETMeta(
        goal_cue_tags=goal_tags,
        temptation_cue_tags=tempt_tags,
        preference_cue_tags=pref_tags,
        goal_consistent_routes=goal_consistent,
        temptation_preferred_routes=tempt_preferred,
        latent_explanation_overlap=overlap,
        subgoal_reveal_order=reveal_order,
        decision_points_by_stage=[decision_pts_s1, decision_pts_s2, []],
        merge_points=merge_pts_1 + merge_pts_2,
    )

    # ═══ Build LatticeV2Meta ═══
    # Create pseudo-segments for compatibility
    all_gate_cells = ([(r, s1_start) for r in s1_rows] +
                      [(r, s2_start) for r in s2_rows] +
                      [(upper_row, s3_start), (lower_row, s3_start)])
    all_door_pos = door_positions

    # Combine temptation cells from both stages
    all_tempt_cells = tempt_cells_s1 + tempt_cells_s2

    meta = LatticeV2Meta(
        segments=[],  # GTET uses different topology than segments
        all_gate_cells=all_gate_cells,
        all_door_positions=all_door_pos,
        shortest_any=shortest_any,
        shortest_safe=shortest_safe,
        cell_features=features,
        world_weights=None,
        latent_mode=latent_mode,
        # DTMB-compatible extensions
        decision_stages=3,
        decision_points_by_stage=gtet_meta.decision_points_by_stage,
        merge_points_by_stage=[merge_pts_1, merge_pts_2, []],
        goal_cue_cells=list(
            set(c for cells in goal_cue_cells_by_sg.values() for c in cells)),
        temptation_cue_cells=all_tempt_cells,
        door_positions_by_stage=[[], [], door_positions],
        belt_cells_by_stage=[[], [], belt_cells],
        route_count=route_count,
        dominant_bottleneck_gt_by_stage=[
            "epistemic", "temptation", "outcome"],
    )

    # Attach GTET-specific sidecar
    meta.gtet_meta = gtet_meta

    # ═══ FamilyConfig ═══
    family_cfg = FamilyConfig(
        max_steps=t_max, risk_budget=1.0,
        prior_risk_mean=0.02, prior_risk_var=0.20,
        search_budget=cfg["search_budget"], budget_class=8,
    )

    # ═══ ScenarioConfig (lazy import to avoid circular dependency) ═══
    from .scenario_families import ScenarioConfig
    sc = ScenarioConfig(
        family_name="goal_preference_temptation_entanglement_lattice",
        difficulty=difficulty,
        primary_intervention="warn_mixed",
        cue_reliability=cfg["goal_cue_reliability"],
        expected_failure_mode="latent_disambiguation_failure",
        requires_gate=cfg["has_locked_fast_lane"],
        gate_mode="unlock_shortcut" if cfg["has_locked_fast_lane"] else "block_risky",
    )

    return gm, family_cfg, meta, sc


# ══════════════════════════════════════════════════════════════════════
# Route enumeration and overlap analysis
# ══════════════════════════════════════════════════════════════════════

def _enumerate_routes(ct: np.ndarray, start: tuple, goal: tuple,
                      H: int, W: int, max_routes: int = 20,
                      stage_rows: list[list[int]] | None = None) -> list[list]:
    """Enumerate structurally distinct routes.

    Instead of full DFS (which explodes on merge zones), we enumerate
    routes as sequences of branch-row choices at each stage fork.
    Each route is represented by the list of cells on its unique path.

    Falls back to a bounded BFS if explicit stage structure is unavailable.
    """
    if stage_rows:
        # Fast path: enumerate combinations of branch choices
        return _enumerate_via_branch_combos(ct, start, goal, H, W,
                                            stage_rows, max_routes)
    # Fallback: bounded forward-only BFS
    return _enumerate_forward_bfs(ct, start, goal, H, W, max_routes)


def _enumerate_via_branch_combos(ct, start, goal, H, W,
                                  stage_rows, max_routes):
    """Enumerate routes by trying all combinations of branch rows per stage."""
    import itertools
    routes = []
    combos = list(itertools.product(*stage_rows))
    if len(combos) > max_routes:
        combos = combos[:max_routes]

    for combo in combos:
        path = _trace_route(ct, start, goal, H, W, list(combo))
        if path and path[-1] == goal:
            routes.append(path)
        if len(routes) >= max_routes:
            break
    return routes


def _trace_route(ct, start, goal, H, W, branch_rows):
    """Trace a single route through the grid following specified branch rows."""
    path = [start]
    pos = start
    visited = {start}
    max_steps = W * 4

    for step in range(max_steps):
        if pos == goal:
            return path
        r, c = pos
        # Determine target row: use the branch row for the current stage
        target_row = r
        for br in branch_rows:
            if br != r:
                # Check if we're at a fork column
                if ct[br, c] not in (CellType.WALL, CellType.LOCKED_DOOR):
                    target_row = br
                    break

        # Priority: move right, then toward target row
        candidates = []
        for dr, dc in [(0, 1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited
                    and ct[nr, nc] not in (CellType.WALL, CellType.LOCKED_DOOR)):
                candidates.append((nr, nc))

        if not candidates:
            break

        # Score: prefer rightward movement
        best = min(candidates,
                   key=lambda p: (-p[1], abs(p[0] - target_row)))
        path.append(best)
        visited.add(best)
        pos = best

    return path


def _enumerate_forward_bfs(ct, start, goal, H, W, max_routes):
    """Bounded forward BFS: enumerate distinct paths moving primarily rightward.

    Uses column-based forward sweep to avoid exponential blowup.
    A 'route' is identified by its row at each stage's fork column.
    """
    routes = []
    # Find fork columns (columns with multiple passable rows)
    fork_cols = []
    for c in range(W):
        passable_rows = [r for r in range(H)
                         if ct[r, c] not in (CellType.WALL, CellType.LOCKED_DOOR)]
        if len(passable_rows) > 1:
            fork_cols.append(c)

    # Simple: trace routes using BFS through the fork structure
    # Each "route signature" is the row at each fork column
    seen_sigs = set()

    # Try different starting row combinations at early forks
    start_r, start_c = start
    goal_r, goal_c = goal

    # Enumerate by trying different rows at each fork
    passable_by_col = {}
    for c in range(W):
        rows = [r for r in range(H)
                if ct[r, c] not in (CellType.WALL, CellType.LOCKED_DOOR)]
        if rows:
            passable_by_col[c] = rows

    # Simple greedy enumeration: for each passable row at each fork,
    # trace a route forward
    first_fork = None
    for c in sorted(passable_by_col.keys()):
        if len(passable_by_col[c]) > 1 and c > start_c:
            first_fork = c
            break

    if first_fork is None:
        # No forks — just trace one route
        path = _trace_single_bfs(ct, start, goal, H, W)
        return [path] if path else []

    for try_row in passable_by_col.get(first_fork, []):
        path = _trace_single_bfs(ct, start, goal, H, W,
                                  preferred_row=try_row)
        if path and path[-1] == goal:
            sig = tuple(r for r, c in path if c in (first_fork,))
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                routes.append(path)
        if len(routes) >= max_routes:
            break

    return routes


def _trace_single_bfs(ct, start, goal, H, W, preferred_row=None):
    """BFS a single route preferring rightward movement."""
    path = [start]
    pos = start
    visited = {start}
    target_r = preferred_row or start[0]

    for _ in range(W * 4):
        if pos == goal:
            return path
        r, c = pos
        candidates = []
        for dr, dc in [(0, 1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited
                    and ct[nr, nc] not in (CellType.WALL, CellType.LOCKED_DOOR)):
                candidates.append((nr, nc))
        if not candidates:
            break
        # Score: rightward > toward preferred row > anything
        best = min(candidates,
                   key=lambda p: (-p[1], abs(p[0] - target_r)))
        path.append(best)
        visited.add(best)
        pos = best
    return path


def _compute_goal_route_consistency(
    routes: list[list], goal_tags: np.ndarray, n_subgoals: int
) -> dict[int, list[int]]:
    """For each subgoal, which routes pass through its cue cells?"""
    result = {sg: [] for sg in range(n_subgoals)}
    for ri, route in enumerate(routes):
        for r, c in route:
            sg = goal_tags[r, c]
            if sg >= 0:
                if ri not in result[sg]:
                    result[sg].append(ri)
    return result


def _compute_tempt_route_preference(
    routes: list[list], tempt_tags: np.ndarray
) -> dict[str, list[int]]:
    """Which routes pass through high-temptation zones?"""
    result = {"low": [], "high": []}
    for ri, route in enumerate(routes):
        max_tempt = max(tempt_tags[r, c] for r, c in route)
        if max_tempt > 0.3:
            result["high"].append(ri)
        else:
            result["low"].append(ri)
    return result


def _compute_overlap(goal_consistent, tempt_preferred, n_subgoals) -> list:
    """Compute how many routes are consistent with BOTH a goal AND temptation."""
    overlap = []
    tempt_high = set(tempt_preferred.get("high", []))
    for sg in range(n_subgoals):
        sg_routes = set(goal_consistent.get(sg, []))
        both = sg_routes & tempt_high
        if both:
            overlap.append({
                "subgoal": sg,
                "shared_routes": list(both),
                "n_shared": len(both),
                "n_goal": len(sg_routes),
                "n_tempt": len(tempt_high),
            })
    return overlap
