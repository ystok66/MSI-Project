"""
Phase 2B — Harder baseline_v2 generator.

Design goals (from user specification):
  A1. Increase texture ambiguity: safe z₂,z₃ raised to [0.25, 0.50],
      trap texture lowered to [0.45, 0.65]. The gap is ~0.15 instead of ~0.7.
  A2. Tighter deadline: time_ratio = 1.25 (was 1.4 for medium)
  A3. Higher trap probability: 90% (was 70% for medium)
  A4. Fewer risky cells per segment (3-5 cols), tighter grids
  A5. Trap risk raised to 0.40-0.65 (was 0.30-0.50) so even
      partial texture confusion → death more often

CRITICAL: Does NOT change WorldWeights generation, learner math,
          planner formula, or any canonical defaults.
          Only modifies feature distributions and scenario parameters.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Literal

from .lattice_v2 import (
    _empty_grid, _build_gridmap, _bfs_len,
    CellType, LatticeV2Meta, SegmentMeta, FamilyConfig,
    FEATURE_DIM,
)


# ════════════════════════════════════════════════════════════════
# Feature generators — tighter texture distributions
# ════════════════════════════════════════════════════════════════

def _hb_safe_feature(rng, lid):
    """Safe cells: raised texture floor → harder to distinguish from risky."""
    return np.array([lid, 0.0,
                     rng.uniform(0.25, 0.50),
                     rng.uniform(0.20, 0.45)])

def _hb_trap_feature(rng, lid):
    """Trap cells: lowered texture ceiling → harder to distinguish from safe."""
    return np.array([lid, 0.0,
                     rng.uniform(0.45, 0.65),
                     rng.uniform(0.40, 0.60)])

def _hb_weak_cue_feature(rng, lid):
    """Weak cue: between safe and trap, very ambiguous."""
    return np.array([lid, 0.0,
                     rng.uniform(0.30, 0.55),
                     rng.uniform(0.28, 0.48)])

def _hb_lane_feature(rng, lid, mild_cue):
    """Regular risky lane cells."""
    if mild_cue:
        return np.array([lid, 0.0,
                         rng.uniform(0.20, 0.35),
                         rng.uniform(0.15, 0.30)])
    return np.array([lid, 0.0,
                     rng.uniform(0.10, 0.25),
                     rng.uniform(0.10, 0.20)])


# ════════════════════════════════════════════════════════════════
# Generator
# ════════════════════════════════════════════════════════════════

def generate_harder_baseline_v2(
    seed: int = 0,
    difficulty: str = "medium",
    latent_mode: bool = True,
    n_segments: int = 3,       # fewer segments, shorter
    time_ratio: float = 1.25,  # tighter than medium(1.4) but survivable
    trap_prob: float = 0.90,   # higher trap probability
    **kwargs,
) -> tuple:
    """Generate a harder baseline_v2 lattice.

    Same topology as baseline_v2 (7-row, risky/safe lane per segment)
    but with:
      - Tighter texture gap (A1)
      - Tighter deadline (A2)
      - Higher trap probability (A3)
      - More segments (A4) for earlier risky contact

    Returns (GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig).
    """
    rng = np.random.default_rng(seed)

    # Build multi-segment grid — shorter segments for fewer obs per episode
    seg_widths = rng.integers(3, 5, size=n_segments)  # 3-4 cols (was 4-6)
    detour_lens = np.ones(n_segments, dtype=int)

    CORR_W = 1
    W = 1 + CORR_W
    for i in range(n_segments):
        W += int(seg_widths[i])
        W += CORR_W
    W += 1

    H = 7
    ct, cost, risk = _empty_grid(H, W)
    features = np.full((H, W, FEATURE_DIM), 0.5, dtype=np.float64)

    ct[:, :] = CellType.WALL
    cost[:, :] = np.inf

    # Corridor (row 2)
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

        # Wall row 2 inside segment
        for c in range(seg_start, seg_end + 1):
            ct[2, c] = CellType.WALL
            cost[2, c] = np.inf

        # Entry/exit holes
        ct[2, seg_start] = CellType.NORMAL
        cost[2, seg_start] = 1.0
        ct[2, seg_end] = CellType.NORMAL
        cost[2, seg_end] = 1.0

        # Entry gates
        for r in [1, 3]:
            ct[r, seg_start] = CellType.NORMAL
            cost[r, seg_start] = 1.0
            features[r, seg_start] = np.array([
                0.0 if r == 1 else 1.0, 1.0, 0.0, 0.0])

        risky_entry = (1, seg_start)
        safe_entry = (3, seg_start)
        all_gates.extend([risky_entry, safe_entry])
        all_doors.append(risky_entry)

        # Exit gates
        for r in [1, 3]:
            ct[r, seg_end] = CellType.NORMAL
            cost[r, seg_end] = 1.0
            features[r, seg_end] = np.array([
                0.0 if r == 1 else 1.0, 1.0, 0.0, 0.0])

        # Risky lane (row 1)
        risky_cells = []
        for c in range(seg_start + 1, seg_end):
            ct[1, c] = CellType.NORMAL
            cost[1, c] = 1.0
            risky_cells.append((1, c))

        # Safe lane (row 3) with zigzag
        safe_cells = []
        detour_start = seg_start + (sw // 2) - dt
        detour_end = detour_start + dt

        for c in range(seg_start + 1, seg_end):
            if detour_start <= c < detour_end:
                ct[3, c] = CellType.WALL
                cost[3, c] = np.inf
            else:
                ct[3, c] = CellType.NORMAL
                cost[3, c] = 1.0
                safe_cells.append((3, c))

        # Detour verticals
        ct[4, detour_start] = CellType.NORMAL
        cost[4, detour_start] = 1.0
        safe_cells.append((4, detour_start))
        ct[4, detour_end] = CellType.NORMAL
        cost[4, detour_end] = 1.0
        safe_cells.append((4, detour_end))

        for c in range(detour_start, detour_end + 1):
            ct[5, c] = CellType.NORMAL
            cost[5, c] = 1.0
            safe_cells.append((5, c))

        dc_down = detour_start
        dc_up = detour_end

        if dc_down - 1 >= seg_start:
            ct[4, dc_down - 1] = CellType.NORMAL
            cost[4, dc_down - 1] = 1.0
            safe_cells.append((4, dc_down - 1))
            ct[5, dc_down - 1] = CellType.NORMAL
            cost[5, dc_down - 1] = 1.0
            safe_cells.append((5, dc_down - 1))

        if dc_up < seg_end:
            ct[4, dc_up + 1] = CellType.NORMAL
            cost[4, dc_up + 1] = 1.0
            safe_cells.append((4, dc_up + 1))
            ct[5, dc_up + 1] = CellType.NORMAL
            cost[5, dc_up + 1] = 1.0
            safe_cells.append((5, dc_up + 1))

        # ── Features and risk ──
        trap_cell = None
        weak_cue_cells = []
        has_trap = rng.random() < trap_prob

        if has_trap and len(risky_cells) >= 2:
            trap_idx = rng.integers(0, len(risky_cells))
            trap_cell = risky_cells[trap_idx]
            for i, (r, c) in enumerate(risky_cells):
                ct[r, c] = CellType.RISKY
                if i == trap_idx:
                    features[r, c] = _hb_trap_feature(rng, 0.0)
                    risk[r, c] = rng.uniform(0.40, 0.65)  # higher than baseline_v2
                elif abs(i - trap_idx) <= 1:
                    features[r, c] = _hb_weak_cue_feature(rng, 0.0)
                    risk[r, c] = rng.uniform(0.15, 0.30)
                    weak_cue_cells.append((r, c))
                else:
                    features[r, c] = _hb_lane_feature(rng, 0.0, True)
                    risk[r, c] = rng.uniform(0.10, 0.20)
        else:
            for r, c in risky_cells:
                ct[r, c] = CellType.RISKY
                features[r, c] = _hb_lane_feature(rng, 0.0, False)
                risk[r, c] = rng.uniform(0.05, 0.10)

        for r, c in safe_cells:
            lid = 0.0 if r == 1 else 1.0
            features[r, c] = _hb_safe_feature(rng, lid)

        segments_meta.append(SegmentMeta(
            index=seg_i, col_start=seg_start, col_end=seg_end,
            risky_row=1, safe_row=3,
            L_risky=0, L_safe=0,
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
    t_max = max(int(time_ratio * base), base + 1)

    cfg = FamilyConfig(
        max_steps=t_max, risk_budget=1.0,
        prior_risk_mean=0.02, prior_risk_var=0.20,
        search_budget=30, budget_class=8,
    )

    # Latent mode: derive cost and risk from features via WorldWeights
    ww = None
    if latent_mode:
        from ..agents.cost_risk_model import generate_world_weights
        ww = generate_world_weights(rng, d=FEATURE_DIM)
        for r in range(H):
            for c in range(W):
                if ct[r, c] == CellType.WALL:
                    continue
                z = features[r, c]
                cost[r, c] = ww.true_cost(z)
                risk[r, c] = ww.true_risk(z)
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
    sc_kwargs = dict(
        family_name="harder_baseline_v2",
        difficulty=difficulty,
        primary_intervention="WARN",
        expected_failure_mode="risk",
    )
    # Lazy import to avoid circular dependency
    from .scenario_families import ScenarioConfig
    sc = ScenarioConfig(**sc_kwargs)
    return gm, cfg, meta, sc
