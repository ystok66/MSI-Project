"""
Deep Tree Mixed-Bottleneck Lattice (DTMB-L) — scenario generator.

Creates a 3-stage tree-lattice where the dominant bottleneck shifts
across stages within a single episode:

    Stage 1: epistemic ambiguity  → WARN / WAIT lever
    Stage 2: structural pressure  → UNLOCK lever
    Stage 3: outcome bottleneck   → ITEM_DROP lever

Grid layout uses variable height (13–17 rows) on current V2 cell/feature
infrastructure. Returns (GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig)
for direct integration with generate_scenario() registry.

Does NOT modify any frozen module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional
from collections import deque
import math

import numpy as np

from .map_generator import CellType, GridMap
from .map_families import _empty_grid, _build_gridmap, FamilyConfig
from .lattice_v2 import (
    LatticeV2Meta, SegmentMeta,
    FEATURE_DIM, F_LANE_ID, F_GATE_FLAG, F_TEXTURE_1, F_TEXTURE_2,
    _bfs_len, _safe_feature, _trap_feature, _weak_cue_feature, _lane_feature,
)
# ScenarioConfig imported lazily inside generate_dtmb_lattice (circular import guard)

DifficultyLevel = Literal["easy", "medium", "hard"]

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

DTMB_PARAMS = {
    "easy": {
        "H": 13, "W": 35,
        "tree_depth": 3,
        "branching_schedule": [3, 2],
        "stage1_cue_reliability": 0.85,
        "stage1_reveal_depth": 2,
        "stage1_commit_depth": 5,
        "mid_door_fraction": 0.15,
        "door_gain": 6,
        "terminal_belt_fraction": 0.20,
        "belt_risk": 0.30,
        "lure_subtree_fraction": 0.15,
        "lure_strength": 0.35,
        "deadline_ratio": 1.25,
        "misleading_fraction": 0.05,
        "search_budget": 30,
    },
    "medium": {
        "H": 15, "W": 45,
        "tree_depth": 3,
        "branching_schedule": [3, 2],
        "stage1_cue_reliability": 0.65,
        "stage1_reveal_depth": 3,
        "stage1_commit_depth": 4,
        "mid_door_fraction": 0.25,
        "door_gain": 4,
        "terminal_belt_fraction": 0.35,
        "belt_risk": 0.45,
        "lure_subtree_fraction": 0.25,
        "lure_strength": 0.55,
        "deadline_ratio": 1.15,
        "misleading_fraction": 0.15,
        "search_budget": 35,
    },
    # HARD_v2 — calibrated via sweep_dtmb_hard_calibration.py (Exp B)
    # J_hard=0.250, Surv_can=0.20, Surv_nt=0.00, Surv_ni=0.10
    "hard": {
        "H": 17, "W": 60,
        "tree_depth": 3,
        "branching_schedule": [4, 2],
        "stage1_cue_reliability": 0.45,
        "stage1_reveal_depth": 4,
        "stage1_commit_depth": 3,
        "mid_door_fraction": 0.35,
        "door_gain": 3,
        "terminal_belt_fraction": 0.50,
        "belt_risk": 0.40,
        "lure_subtree_fraction": 0.35,
        "lure_strength": 0.80,
        "deadline_ratio": 1.16,
        "misleading_fraction": 0.25,
        "search_budget": 40,
    },
}


def _resolve_config(difficulty: str, user_cfg: dict | None = None) -> dict:
    """Merge user overrides into difficulty defaults."""
    cfg = dict(DTMB_PARAMS[difficulty])
    if user_cfg:
        for k, v in user_cfg.items():
            if k in cfg:
                cfg[k] = v
    return cfg


# ══════════════════════════════════════════════════════════════════════
# Stage band helpers
# ══════════════════════════════════════════════════════════════════════

def _compute_bands(W: int) -> list[tuple[int, int]]:
    """Split column space into 3 stage bands [trunk+s1, s2, s3].

    Returns [(c_start, c_end), ...] for stages 1, 2, 3.
    Band boundaries are approximate 30% / 40% / 30% split after trunk.
    """
    trunk_end = 3  # first 3 cols are trunk
    usable = W - trunk_end - 1  # last col is border
    s1_w = max(8, int(usable * 0.30))
    s2_w = max(8, int(usable * 0.40))
    s3_w = max(6, usable - s1_w - s2_w)
    b1 = (trunk_end, trunk_end + s1_w)
    b2 = (b1[1], b1[1] + s2_w)
    b3 = (b2[1], b2[1] + s3_w)
    return [b1, b2, b3]


def _allocate_branch_rows(H: int, n_branches: int, center_row: int,
                          min_gap: int = 2) -> list[int]:
    """Allocate vertically spaced rows for branches, centered around center_row.

    Returns sorted list of row indices with at least min_gap between them.
    All rows must be within [1, H-2] (leaving top/bottom wall rows).
    """
    if n_branches == 1:
        return [center_row]

    total_span = (n_branches - 1) * min_gap
    top = max(1, center_row - total_span // 2)
    # Adjust if we'd go past bottom
    if top + total_span >= H - 1:
        top = max(1, H - 2 - total_span)
    rows = [top + i * min_gap for i in range(n_branches)]
    # Clamp
    rows = [max(1, min(H - 2, r)) for r in rows]
    return sorted(rows)


# ══════════════════════════════════════════════════════════════════════
# Feature helpers (extending existing V2 helpers)
# ══════════════════════════════════════════════════════════════════════

def _misleading_feature(rng: np.random.Generator, lid: float) -> np.ndarray:
    """Misleading cue: looks safe (low texture_1) but is actually risky."""
    return np.array([lid, rng.uniform(0.3, 0.6),
                     rng.uniform(0.05, 0.15),   # low texture_1 = appears safe
                     rng.uniform(0.05, 0.15)])   # low texture_2 = appears safe


def _temptation_feature(rng: np.random.Generator, lid: float,
                        strength: float) -> np.ndarray:
    """Temptation lure: high gate_flag, moderate textures."""
    return np.array([lid, rng.uniform(0.6, 0.9) * strength,
                     rng.uniform(0.15, 0.35),
                     rng.uniform(0.10, 0.25)])


def _belt_feature(rng: np.random.Generator, lid: float,
                  risk_level: float) -> np.ndarray:
    """Terminal hazard belt: high texture_1/2 for high risk."""
    t1 = rng.uniform(0.7, 0.95) * risk_level / 0.6
    t2 = rng.uniform(0.6, 0.85) * risk_level / 0.6
    return np.array([lid, 0.0, min(t1, 0.99), min(t2, 0.99)])


def _door_feature(lid: float) -> np.ndarray:
    """Locked door cell feature."""
    return np.array([lid, 1.0, 0.0, 0.0])


# ══════════════════════════════════════════════════════════════════════
# BFS utilities
# ══════════════════════════════════════════════════════════════════════

def _bfs_shortest(ct: np.ndarray, start: tuple[int, int],
                  goal: tuple[int, int],
                  avoid: set | None = None) -> int:
    """BFS shortest path length on cell_types array. WALL/LOCKED_DOOR blocked."""
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
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) in visited:
                continue
            if ct[nr, nc] in (CellType.WALL, CellType.LOCKED_DOOR):
                continue
            if (nr, nc) in avoid:
                continue
            visited.add((nr, nc))
            queue.append(((nr, nc), d + 1))
    return 999


def _bfs_reachable(ct: np.ndarray, start: tuple[int, int]) -> set:
    """Return all reachable cells from start (ignoring WALL/LOCKED_DOOR)."""
    H, W = ct.shape
    visited = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) in visited:
                continue
            if ct[nr, nc] in (CellType.WALL, CellType.LOCKED_DOOR):
                continue
            visited.add((nr, nc))
            queue.append((nr, nc))
    return visited


def _count_routes_structural(
    s1_meta: "_StageMeta",
    s2_metas: "list[_StageMeta]",
) -> int:
    """Count distinct terminal-to-goal routes from tree structure.

    For a tree-lattice, routes = n_s1_branches × n_s2_per_branch.
    This is O(1) — no BFS needed.
    """
    n_s1 = len(s1_meta.exits)
    total = 0
    for s2m in s2_metas:
        total += len(s2m.exits)
    return max(total, n_s1)


def _build_route_summaries(
    s1_meta: "_StageMeta",
    s2_metas: "list[_StageMeta]",
    s3_meta: "_StageMeta",
) -> list[list[tuple[int, int]]]:
    """Build lightweight route summaries as decision-point sequences.

    Each 'route' is a list of key waypoints: [s1_fork, s1_exit, s2_fork, s2_exit, s3_entry].
    No expensive BFS — purely structural traversal of the tree skeleton.
    """
    routes = []
    s1_fork = s1_meta.decision_points[0] if s1_meta.decision_points else None

    for s1_idx, s1_exit in enumerate(s1_meta.exits):
        if s1_idx >= len(s2_metas):
            # Direct route (no Stage 2 split)
            route = []
            if s1_fork:
                route.append(s1_fork)
            route.append(s1_exit)
            if s3_meta.exits:
                route.append(s3_meta.exits[0])
            routes.append(route)
            continue

        s2m = s2_metas[s1_idx]
        s2_fork = s2m.decision_points[0] if s2m.decision_points else None

        for s2_exit in s2m.exits:
            route = []
            if s1_fork:
                route.append(s1_fork)
            route.append(s1_exit)
            if s2_fork:
                route.append(s2_fork)
            route.append(s2_exit)
            if s3_meta.exits:
                route.append(s3_meta.exits[0])
            routes.append(route)

    return routes


# ══════════════════════════════════════════════════════════════════════
# Core generator
# ══════════════════════════════════════════════════════════════════════

@dataclass
class _StageMeta:
    """Internal per-stage metadata collected during generation."""
    stage_id: int
    branch_rows: list[int]
    col_range: tuple[int, int]
    exits: list[tuple[int, int]]  # (row, col) where each branch exits
    decision_points: list[tuple[int, int]] = field(default_factory=list)
    commitment_points: list[tuple[int, int]] = field(default_factory=list)
    reveal_events: list[tuple[int, int]] = field(default_factory=list)
    merge_points: list[tuple[int, int]] = field(default_factory=list)
    weak_cue_cells: list[tuple[int, int]] = field(default_factory=list)
    misleading_cells: list[tuple[int, int]] = field(default_factory=list)
    temptation_cells: list[tuple[int, int]] = field(default_factory=list)
    door_positions: list[tuple[int, int]] = field(default_factory=list)
    belt_cells: list[tuple[int, int]] = field(default_factory=list)


def generate_dtmb_lattice(
    seed: int,
    difficulty: DifficultyLevel = "medium",
    latent_mode: bool = True,
    **kwargs,
) -> tuple[GridMap, FamilyConfig, LatticeV2Meta, "ScenarioConfig"]:
    """Generate a Deep Tree Mixed-Bottleneck Lattice scenario.

    Three stages:
      Stage 1 — epistemic root-split: 3 branches with weak/misleading cues
      Stage 2 — structural split: sub-branches with LOCKED_DOOR shortcuts
      Stage 3 — terminal zone: hazard belts with near-unavoidable risk

    Returns (GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig).
    """
    from .scenario_families import ScenarioConfig  # lazy import (circular guard)
    rng = np.random.default_rng(seed)
    cfg = _resolve_config(difficulty, kwargs.get("user_cfg"))

    H, W = cfg["H"], cfg["W"]
    ct, cost, risk = _empty_grid(H, W)
    features = np.full((H, W, FEATURE_DIM), 0.5, dtype=np.float64)

    # Start from all-wall canvas
    ct[:, :] = CellType.WALL
    cost[:, :] = np.inf

    bands = _compute_bands(W)
    trunk_row = H // 2
    agent_start = (trunk_row, 1)

    # ── Carve trunk (cols 1 to band1 start) ─────────────────────────
    for c in range(1, bands[0][0] + 1):
        ct[trunk_row, c] = CellType.NORMAL
        cost[trunk_row, c] = 1.0
        features[trunk_row, c] = _safe_feature(rng, 0.5)

    # ── Stage 1: root split into 3 branches ─────────────────────────
    n_s1 = cfg["branching_schedule"][0]  # 3
    s1_rows = _allocate_branch_rows(H, n_s1, trunk_row, min_gap=3)
    s1_meta = _carve_stage1(
        ct, cost, risk, features, rng, cfg,
        H, W, bands[0], trunk_row, s1_rows,
    )

    # ── Stage 2: each S1 branch splits into sub-branches ─────────────
    n_s2_per = cfg["branching_schedule"][1] if len(cfg["branching_schedule"]) > 1 else 2
    s2_metas: list[_StageMeta] = []
    all_s2_exits: list[tuple[int, int]] = []

    for s1_idx, s1_exit in enumerate(s1_meta.exits):
        s1_row = s1_exit[0]
        s2_rows = _allocate_branch_rows(H, n_s2_per, s1_row, min_gap=2)
        # Ensure we don't overlap with other S1 branch ranges
        s2_sub = _carve_stage2(
            ct, cost, risk, features, rng, cfg,
            H, W, bands[1], s1_exit, s2_rows, s1_idx,
        )
        s2_metas.append(s2_sub)
        all_s2_exits.extend(s2_sub.exits)

    # ── Stage 3: terminal zone with belts ────────────────────────────
    s3_meta = _carve_stage3(
        ct, cost, risk, features, rng, cfg,
        H, W, bands[2], all_s2_exits,
    )

    # ── Place target ─────────────────────────────────────────────────
    target_pos = s3_meta.exits[0] if s3_meta.exits else (trunk_row, W - 2)
    ct[target_pos] = CellType.TARGET
    cost[target_pos] = 1.0

    object_spawn = target_pos

    # Collect all door positions
    all_doors = []
    for s2m in s2_metas:
        all_doors.extend(s2m.door_positions)

    # ── Build GridMap ────────────────────────────────────────────────
    gm = _build_gridmap(H, W, ct, cost, risk,
                        agent_start, object_spawn, target_pos, all_doors)

    # ── BFS shortest paths ──────────────────────────────────────────
    shortest_any = _bfs_shortest(ct, agent_start, target_pos)
    # Shortest safe: avoid belt cells with very high risk
    high_risk_cells = set(s3_meta.belt_cells)
    shortest_safe = _bfs_shortest(ct, agent_start, target_pos, high_risk_cells)
    if shortest_safe >= 999:
        # Belt may be near-unavoidable but should still have some path
        shortest_safe = shortest_any

    # ── Route enumeration (structural, O(branches)) ────────────────────
    route_count = _count_routes_structural(s1_meta, s2_metas)
    routes = _build_route_summaries(s1_meta, s2_metas, s3_meta)

    # Classify routes by whether they pass through belt-row exits
    safe_routes = []
    risky_routes = []
    belt_rows = set(r for r, c in s3_meta.belt_cells)
    for route in routes:
        # A route is risky if any of its S2 exits are on a belt row
        if any(r in belt_rows for r, c in route):
            risky_routes.append(route)
        else:
            safe_routes.append(route)

    # ── Deadline ─────────────────────────────────────────────────────
    base_len = shortest_safe if shortest_safe < 999 else shortest_any
    t_max = max(int(cfg["deadline_ratio"] * base_len), base_len + 3)

    # ── Slack profile ────────────────────────────────────────────────
    slack_profile = {
        "total_slack": t_max - shortest_any,
        "safe_slack": t_max - shortest_safe if shortest_safe < 999 else 0,
        "shortest_any": shortest_any,
        "shortest_safe": shortest_safe,
        "t_max": t_max,
    }

    # ── Ground-truth bottleneck by stage ──────────────────────────────
    gt_bottlenecks = _infer_gt_bottlenecks(
        s1_meta, s2_metas, s3_meta,
        shortest_any, shortest_safe, t_max, cfg,
    )

    # ── Latent mode: derive cost/risk from features ──────────────────
    ww = None
    if latent_mode:
        from ..agents.cost_risk_model import generate_world_weights
        ww = generate_world_weights(rng, d=FEATURE_DIM)
        for r in range(H):
            for c in range(W):
                if ct[r, c] in (CellType.WALL, CellType.LOCKED_DOOR):
                    continue
                z = features[r, c]
                cost[r, c] = ww.true_cost(z)
                risk[r, c] = ww.true_risk(z)
        # Enforce explicit risk for belt cells (outcome bottleneck must be real)
        for br, bc in s3_meta.belt_cells:
            risk[br, bc] = max(risk[br, bc], cfg["belt_risk"] * 0.8)
        # Rebuild gridmap with latent-derived values
        gm = _build_gridmap(H, W, ct, cost, risk,
                            agent_start, object_spawn, target_pos, all_doors)

    # ── Package metadata ─────────────────────────────────────────────
    # Build pseudo-SegmentMeta for compatibility with existing pipelines
    seg_meta = SegmentMeta(
        index=0, col_start=bands[0][0], col_end=bands[2][1],
        risky_row=s1_rows[0], safe_row=s1_rows[-1],
        L_risky=0, L_safe=0, detour_len=0,
        risky_cells=s3_meta.belt_cells,
        safe_cells=[],
        risky_entry_gate=(trunk_row, bands[0][0]),
        safe_entry_gate=(trunk_row, bands[0][0]),
        trap_cell=None,
        weak_cue_cells=s1_meta.weak_cue_cells,
    )

    meta = LatticeV2Meta(
        segments=[seg_meta],
        all_gate_cells=s1_meta.decision_points,
        all_door_positions=all_doors,
        shortest_any=shortest_any,
        shortest_safe=shortest_safe,
        cell_features=features,
        world_weights=ww,
        latent_mode=latent_mode,
        # DTMB extension fields
        decision_stages=3,
        decision_points_by_stage=[
            s1_meta.decision_points,
            [dp for s2m in s2_metas for dp in s2m.decision_points],
            s3_meta.decision_points,
        ],
        merge_points_by_stage=[
            s1_meta.merge_points,
            [mp for s2m in s2_metas for mp in s2m.merge_points],
            s3_meta.merge_points,
        ],
        commitment_points_by_stage=[
            s1_meta.commitment_points,
            [cp for s2m in s2_metas for cp in s2m.commitment_points],
            s3_meta.commitment_points,
        ],
        reveal_events_by_stage=[
            s1_meta.reveal_events,
            [re for s2m in s2_metas for re in s2m.reveal_events],
            s3_meta.reveal_events,
        ],
        goal_cue_cells=s3_meta.belt_cells[:2] if s3_meta.belt_cells else [],
        temptation_cue_cells=s1_meta.temptation_cells,
        door_positions_by_stage=[
            [],
            [dp for s2m in s2_metas for dp in s2m.door_positions],
            [],
        ],
        belt_cells_by_stage=[
            [],
            [],
            s3_meta.belt_cells,
        ],
        safe_routes=[list(r) for r in safe_routes[:5]],
        risky_routes=[list(r) for r in risky_routes[:5]],
        route_count=route_count,
        route_depths=[len(r) for r in routes],
        slack_profile=slack_profile,
        dominant_bottleneck_gt_by_stage=gt_bottlenecks,
        recommended_primary_lever_by_stage=["WAIT/WARN", "UNLOCK", "ITEM_DROP"],
    )

    sc = ScenarioConfig(
        family_name="deep_tree_mixed_bottleneck_lattice",
        difficulty=difficulty,
        primary_intervention="mixed",
        cue_reliability=cfg["stage1_cue_reliability"],
        expected_failure_mode="mixed",
        requires_gate=len(all_doors) > 0,
        requires_item=len(s3_meta.belt_cells) > 0,
        gate_mode="unlock_shortcut",
        commitment_cells=(s1_meta.commitment_points +
                          [cp for s2m in s2_metas for cp in s2m.commitment_points]),
        belt_regime="near_unavoidable",
    )

    family_cfg = FamilyConfig(
        max_steps=t_max,
        risk_budget=1.0,
        prior_risk_mean=0.02,
        prior_risk_var=0.20,
        search_budget=cfg["search_budget"],
        budget_class=10,
    )

    return gm, family_cfg, meta, sc


# ══════════════════════════════════════════════════════════════════════
# Stage 1: Epistemic root-split
# ══════════════════════════════════════════════════════════════════════

def _carve_stage1(
    ct, cost, risk, features, rng, cfg,
    H, W, band, trunk_row, branch_rows,
) -> _StageMeta:
    """Carve Stage 1: root splits into 3 branches with weak/misleading cues.

    Each branch is a horizontal corridor on its assigned row, connected
    to the trunk via a vertical connector at band[0].
    """
    c_start, c_end = band
    meta = _StageMeta(stage_id=1, branch_rows=branch_rows,
                      col_range=band, exits=[])

    # Vertical connector from trunk to each branch row
    fork_col = c_start
    meta.decision_points.append((trunk_row, fork_col))

    # Carve vertical spine at fork_col
    all_rows = sorted(set(branch_rows + [trunk_row]))
    r_min, r_max = min(all_rows), max(all_rows)
    for r in range(r_min, r_max + 1):
        ct[r, fork_col] = CellType.NORMAL
        cost[r, fork_col] = 1.0
        features[r, fork_col] = _safe_feature(rng, 0.5)

    # Determine which branch is the "lure" (temptation trap)
    n_branches = len(branch_rows)
    lure_idx = rng.integers(0, n_branches) if rng.random() < cfg["lure_subtree_fraction"] else -1
    # Determine which branch has misleading cues
    misleading_idx = rng.integers(0, n_branches) if rng.random() < cfg["misleading_fraction"] else -1

    reveal_depth = cfg["stage1_reveal_depth"]
    commit_depth = cfg["stage1_commit_depth"]
    cue_reliability = cfg["stage1_cue_reliability"]

    for b_idx, b_row in enumerate(branch_rows):
        lid = float(b_idx) / max(n_branches - 1, 1)
        is_lure = (b_idx == lure_idx)
        is_misleading = (b_idx == misleading_idx)

        # Carve horizontal corridor
        branch_len = c_end - c_start
        for c in range(c_start + 1, c_end):
            ct[b_row, c] = CellType.NORMAL
            cost[b_row, c] = 1.0

            # Feature assignment based on position within branch
            depth_in_branch = c - c_start
            if depth_in_branch < reveal_depth:
                # Early cells: weak or misleading cues
                if is_misleading:
                    features[b_row, c] = _misleading_feature(rng, lid)
                    meta.misleading_cells.append((b_row, c))
                elif rng.random() > cue_reliability:
                    features[b_row, c] = _weak_cue_feature(rng, lid)
                    meta.weak_cue_cells.append((b_row, c))
                else:
                    features[b_row, c] = _safe_feature(rng, lid)
            elif is_lure:
                # Lure branch: temptation features
                features[b_row, c] = _temptation_feature(
                    rng, lid, cfg["lure_strength"])
                meta.temptation_cells.append((b_row, c))
            else:
                # Honest features past reveal depth
                features[b_row, c] = _lane_feature(rng, lid, depth_in_branch > reveal_depth + 1)

        # Commitment point: point past which branch cannot be switched
        commit_col = min(c_start + commit_depth, c_end - 2)
        meta.commitment_points.append((b_row, commit_col))

        # Reveal event: point where strong cues begin
        reveal_col = c_start + reveal_depth
        if reveal_col < c_end:
            meta.reveal_events.append((b_row, reveal_col))

        # Exit at the end of the band
        meta.exits.append((b_row, c_end - 1))

    return meta


# ══════════════════════════════════════════════════════════════════════
# Stage 2: Structural split with locked shortcuts
# ══════════════════════════════════════════════════════════════════════

def _carve_stage2(
    ct, cost, risk, features, rng, cfg,
    H, W, band, parent_exit, sub_rows, parent_idx,
) -> _StageMeta:
    """Carve Stage 2: parent branch splits into sub-branches.

    One sub-branch may contain a LOCKED_DOOR with a shortcut behind it.
    The other sub-branch has a longer detour (always passable).
    """
    c_start, c_end = band
    parent_row = parent_exit[0]
    parent_col = parent_exit[1]
    meta = _StageMeta(stage_id=2, branch_rows=sub_rows,
                      col_range=band, exits=[])

    # Connect parent exit to stage 2 start
    # Horizontal connector from parent_col to c_start
    for c in range(parent_col, c_start + 1):
        ct[parent_row, c] = CellType.NORMAL
        cost[parent_row, c] = 1.0
        features[parent_row, c] = _safe_feature(rng, 0.5)

    # Vertical fork at c_start
    fork_col = c_start
    all_rows = sorted(set(sub_rows + [parent_row]))
    r_min, r_max = min(all_rows), max(all_rows)
    for r in range(r_min, r_max + 1):
        ct[r, fork_col] = CellType.NORMAL
        cost[r, fork_col] = 1.0
        features[r, fork_col] = _safe_feature(rng, 0.5)

    meta.decision_points.append((parent_row, fork_col))

    # Determine which sub-branch gets a door
    n_subs = len(sub_rows)
    has_door = rng.random() < cfg["mid_door_fraction"]
    door_idx = rng.integers(0, n_subs) if has_door else -1
    door_gain = cfg["door_gain"]

    for s_idx, s_row in enumerate(sub_rows):
        lid = float(parent_idx * n_subs + s_idx) / 6.0
        is_door_branch = (s_idx == door_idx)

        # Carve the sub-branch corridor
        for c in range(c_start + 1, c_end):
            ct[s_row, c] = CellType.NORMAL
            cost[s_row, c] = 1.0
            features[s_row, c] = _lane_feature(rng, lid, False)

        if is_door_branch:
            # Place locked door partway through
            door_col = c_start + (c_end - c_start) // 3
            ct[s_row, door_col] = CellType.LOCKED_DOOR
            cost[s_row, door_col] = np.inf
            features[s_row, door_col] = _door_feature(lid)
            meta.door_positions.append((s_row, door_col))

            # Shortcut behind door: carve a faster direct path
            # (the cells after the door are already passable from the loop above)
            # The gain comes from the other branch being forced into a detour
            # by NOT having the shortcut alignment

            meta.commitment_points.append((s_row, door_col))
        else:
            # Non-door branch: add a detour (zigzag) to make it longer
            detour_col = c_start + (c_end - c_start) // 2
            detour_len = min(door_gain // 2, 3)

            # Create detour by walling part of direct path
            # and routing through adjacent row
            detour_row = s_row + (1 if s_row < H - 3 else -1)
            for dc in range(detour_len):
                dc_col = detour_col + dc
                if dc_col >= c_end:
                    break
                # Wall the direct path at this col
                ct[s_row, dc_col] = CellType.WALL
                cost[s_row, dc_col] = np.inf
                # Open detour row
                ct[detour_row, dc_col] = CellType.NORMAL
                cost[detour_row, dc_col] = 1.0
                features[detour_row, dc_col] = _safe_feature(rng, lid)

            # Vertical connectors for detour entry/exit
            if detour_col > c_start + 1:
                entry_col = detour_col - 1
                ct[detour_row, entry_col] = CellType.NORMAL
                cost[detour_row, entry_col] = 1.0
                features[detour_row, entry_col] = _safe_feature(rng, lid)
            exit_col = min(detour_col + detour_len, c_end - 1)
            ct[detour_row, exit_col] = CellType.NORMAL
            cost[detour_row, exit_col] = 1.0
            features[detour_row, exit_col] = _safe_feature(rng, lid)

        # Exit at end of band
        meta.exits.append((s_row, c_end - 1))

    # Merge point at band exit
    for s_row in sub_rows:
        meta.merge_points.append((s_row, c_end - 1))

    return meta


# ══════════════════════════════════════════════════════════════════════
# Stage 3: Terminal outcome bottleneck
# ══════════════════════════════════════════════════════════════════════

def _carve_stage3(
    ct, cost, risk, features, rng, cfg,
    H, W, band, parent_exits,
) -> _StageMeta:
    """Carve Stage 3: terminal zone with hazard belt.

    Converges all sub-branches toward a single target column.
    Places near-unavoidable belt of risky cells before the target.
    """
    c_start, c_end = band
    belt_fraction = cfg["terminal_belt_fraction"]
    belt_risk_val = cfg["belt_risk"]
    meta = _StageMeta(stage_id=3, branch_rows=[],
                      col_range=band, exits=[])

    # Collect unique entry rows
    entry_rows = sorted(set(ex[0] for ex in parent_exits))
    meta.branch_rows = entry_rows

    # Target position: center of right edge
    target_row = entry_rows[len(entry_rows) // 2] if entry_rows else H // 2
    target_col = c_end - 1
    if target_col >= W:
        target_col = W - 2

    # Connect each entry to a convergence corridor
    for e_row in entry_rows:
        e_col = parent_exits[entry_rows.index(e_row)][1] if entry_rows.index(e_row) < len(parent_exits) else c_start

        # Horizontal corridor to target col
        for c in range(max(e_col, c_start), target_col + 1):
            if ct[e_row, c] != CellType.NORMAL:
                ct[e_row, c] = CellType.NORMAL
                cost[e_row, c] = 1.0
                features[e_row, c] = _safe_feature(rng, 0.5)

        # If not on target row, add diagonal/vertical convergence
        if e_row != target_row:
            # Vertical connector at a merge column
            merge_col = target_col - 2
            if merge_col < c_start:
                merge_col = c_start + 1
            r_min, r_max = min(e_row, target_row), max(e_row, target_row)
            for r in range(r_min, r_max + 1):
                if ct[r, merge_col] != CellType.NORMAL:
                    ct[r, merge_col] = CellType.NORMAL
                    cost[r, merge_col] = 1.0
                    features[r, merge_col] = _safe_feature(rng, 0.5)

    # Ensure target row corridor is passable
    for c in range(c_start, target_col + 1):
        if ct[target_row, c] != CellType.NORMAL:
            ct[target_row, c] = CellType.NORMAL
            cost[target_row, c] = 1.0
            features[target_row, c] = _safe_feature(rng, 0.5)

    # ── Place hazard belt ────────────────────────────────────────────
    belt_width = max(2, int((target_col - c_start) * belt_fraction))
    belt_start_col = target_col - belt_width - 1

    # Belt spans multiple rows to be near-unavoidable
    belt_rows = entry_rows if len(entry_rows) >= 2 else [target_row]
    # Also include target_row
    belt_rows = sorted(set(belt_rows + [target_row]))

    for br in belt_rows:
        for bc in range(belt_start_col, belt_start_col + belt_width):
            if bc < c_start or bc >= target_col or bc >= W:
                continue
            if ct[br, bc] == CellType.NORMAL:
                ct[br, bc] = CellType.RISKY
                risk[br, bc] = rng.uniform(belt_risk_val * 0.8, belt_risk_val)
                features[br, bc] = _belt_feature(rng, 0.5, belt_risk_val)
                meta.belt_cells.append((br, bc))

    # Decision point: just before belt
    if belt_start_col > c_start:
        meta.decision_points.append((target_row, belt_start_col - 1))
        meta.commitment_points.append((target_row, belt_start_col))

    # Reveal event at belt start
    meta.reveal_events.append((target_row, belt_start_col))

    meta.exits.append((target_row, target_col))
    return meta


# ══════════════════════════════════════════════════════════════════════
# Ground-truth bottleneck inference
# ══════════════════════════════════════════════════════════════════════

def _infer_gt_bottlenecks(
    s1: _StageMeta,
    s2_list: list[_StageMeta],
    s3: _StageMeta,
    shortest_any: int,
    shortest_safe: int,
    t_max: int,
    cfg: dict,
) -> list[str]:
    """Infer ground-truth dominant bottleneck per stage.

    Stage 1: epistemic if there are weak/misleading cues
    Stage 2: structural if there are locked doors
    Stage 3: outcome if there is a hazard belt
    """
    bottlenecks = []

    # Stage 1
    has_epistemic = (len(s1.weak_cue_cells) > 0 or
                     len(s1.misleading_cells) > 0 or
                     cfg["stage1_cue_reliability"] < 0.9)
    bottlenecks.append("epistemic" if has_epistemic else "none")

    # Stage 2
    has_structural = any(len(s2m.door_positions) > 0 for s2m in s2_list)
    safe_slack = t_max - shortest_safe
    if has_structural and safe_slack < t_max * 0.15:
        bottlenecks.append("structural")
    elif has_structural:
        bottlenecks.append("structural")
    else:
        bottlenecks.append("epistemic")  # falls back to epistemic if no doors

    # Stage 3
    has_outcome = len(s3.belt_cells) > 0
    bottlenecks.append("outcome" if has_outcome else "none")

    return bottlenecks


# ══════════════════════════════════════════════════════════════════════
# ASCII visualization (debug helper)
# ══════════════════════════════════════════════════════════════════════

def print_dtmb_ascii(gm: GridMap, meta: LatticeV2Meta) -> str:
    """Print ASCII representation of DTMB grid for debugging."""
    H, W = gm.height, gm.width
    lines = []
    door_set = set(meta.all_door_positions)
    belt_set = set(meta.belt_cells_by_stage[2]) if meta.belt_cells_by_stage else set()

    for r in range(H):
        row_chars = []
        for c in range(W):
            pos = (r, c)
            if pos == gm.agent_start:
                row_chars.append("S")
            elif pos == gm.target_pos:
                row_chars.append("G")
            elif gm.cell_types[r, c] == CellType.WALL:
                row_chars.append("█")
            elif gm.cell_types[r, c] == CellType.LOCKED_DOOR:
                row_chars.append("D")
            elif pos in belt_set:
                row_chars.append("!")
            elif gm.cell_types[r, c] == CellType.RISKY:
                row_chars.append("~")
            elif pos in door_set:
                row_chars.append("D")
            else:
                row_chars.append("·")
        lines.append("".join(row_chars))

    result = "\n".join(lines)
    return result
