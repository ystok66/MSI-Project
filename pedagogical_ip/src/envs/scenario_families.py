"""
Scenario Families — 5 parameterized V2 lattice variants + baseline.

Each family produces a different failure mechanism / intervention leverage,
but all share the same 7-row segment grid, 4D features, latent mode,
runner, tutor, and metrics pipeline.

Families
--------
- baseline_v2      : default V2 lattice (regression anchor)
- fork_trap        : ambiguous lane fork — WARN lever
- hazard_belt      : unavoidable risk zone — ITEM_DROP lever
- deadline_gate    : tight deadline + gated shortcut — UNLOCK lever
- delayed_corridor : late-revealing risk — prefix-aware WARN lever
- distractor_cue   : misleading local cues — WARN + transfer lever
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from .map_generator import CellType, GridMap
from .map_families import _empty_grid, _build_gridmap, FamilyConfig
from .lattice_v2 import (
    generate_lattice_v2,
    SegmentMeta, LatticeV2Meta,
    FEATURE_DIM, F_LANE_ID, F_GATE_FLAG, F_TEXTURE_1, F_TEXTURE_2,
    _bfs_len, _safe_feature, _trap_feature, _weak_cue_feature, _lane_feature,
)
from .dtmb_lattice import generate_dtmb_lattice
from .gtet_lattice import generate_gtet_lattice

DifficultyLevel = Literal["easy", "medium", "hard"]


# ── Extended config ──────────────────────────────────────────────────

@dataclass
class ScenarioConfig:
    """Scenario-level metadata carried alongside FamilyConfig."""
    family_name: str = "baseline_v2"
    difficulty: str = "medium"
    primary_intervention: str = "WARN"        # most natural lever
    cue_reliability: float = 1.0              # 1.0 = honest, 0.0 = uncorrelated
    hazard_density: float = 0.0               # fraction of cells in unavoidable belt
    requires_gate: bool = False               # needs unlock_shortcut gate
    requires_item: bool = False               # needs ITEM_DROP to be meaningful
    expected_failure_mode: str = "risk"        # risk / timeout / cue_error / commitment
    gate_mode: str = "block_risky"            # "block_risky" | "unlock_shortcut"
    # Delayed corridor specifics
    commitment_cells: list = field(default_factory=list)  # cells past which backtrack > deadline
    # Hazard belt specifics
    belt_regime: str = "unavoidable"           # "unavoidable" | "near_unavoidable"


# ── Entry point ──────────────────────────────────────────────────────

def generate_scenario(
    family: str,
    seed: int,
    difficulty: DifficultyLevel = "medium",
    latent_mode: bool = True,
    **kwargs,
) -> tuple[GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig]:
    """Unified entry point for all scenario families.

    Returns (GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig).
    ScenarioConfig carries family-specific metadata for metrics/reporting.
    """
    if family not in SCENARIO_REGISTRY:
        available = list(SCENARIO_REGISTRY.keys())
        raise ValueError(f"Unknown scenario family '{family}'. Available: {available}")
    return SCENARIO_REGISTRY[family](seed, difficulty, latent_mode, **kwargs)


# ══════════════════════════════════════════════════════════════════════
# baseline_v2 — default V2 lattice, regression anchor
# ══════════════════════════════════════════════════════════════════════

def generate_baseline_v2(
    seed: int,
    difficulty: DifficultyLevel = "medium",
    latent_mode: bool = True,
    **kwargs,
) -> tuple[GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig]:
    """Default V2 lattice. Wraps generate_lattice_v2 for registry compat."""
    gm, cfg, meta = generate_lattice_v2(
        seed=seed, difficulty=difficulty, latent_mode=latent_mode)
    sc = ScenarioConfig(
        family_name="baseline_v2",
        difficulty=difficulty,
        primary_intervention="WARN",
        expected_failure_mode="risk",
    )
    return gm, cfg, meta, sc


# ══════════════════════════════════════════════════════════════════════
# fork_trap — Ambiguous Lane Fork
# ══════════════════════════════════════════════════════════════════════

FORK_TRAP_PARAMS = {
    "easy":   {"cue_ambiguity": 0.3, "trap_depth": 1, "trap_risk": 0.30,
               "time_ratio": 1.50, "branch_len": 6},
    "medium": {"cue_ambiguity": 0.6, "trap_depth": 2, "trap_risk": 0.45,
               "time_ratio": 1.35, "branch_len": 6},
    "hard":   {"cue_ambiguity": 0.9, "trap_depth": 3, "trap_risk": 0.60,
               "time_ratio": 1.20, "branch_len": 7},
}


def generate_fork_trap(
    seed: int,
    difficulty: DifficultyLevel = "medium",
    latent_mode: bool = True,
    **kwargs,
) -> tuple[GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig]:
    """Ambiguous Lane Fork — 7-row pseudo-fork.

    Two branches with near-symmetric local cues. One branch has a trap
    deeper in (trap_depth cells from entry), only distinguishable after
    partial commitment. WARN is the primary intervention lever.

    Topology:
        Row 0: wall
        Row 1: branch A (risky in one direction)
        Row 2: corridor → fork → merge → goal
        Row 3: branch B (safe, but with zigzag detour)
        Row 4-6: detour space
    """
    rng = np.random.default_rng(seed)
    params = FORK_TRAP_PARAMS[difficulty]
    ambiguity = params["cue_ambiguity"]
    trap_depth = params["trap_depth"]
    trap_risk = params["trap_risk"]
    time_ratio = params["time_ratio"]
    branch_len = params["branch_len"]

    # ── Grid dimensions ──
    # Single fork-merge unit: corridor + fork segment + corridor
    CORR_W = 1
    seg_width = branch_len
    W = 1 + CORR_W + seg_width + CORR_W + 1
    H = 7

    ct, cost, risk = _empty_grid(H, W)
    features = np.full((H, W, FEATURE_DIM), 0.5, dtype=np.float64)
    ct[:, :] = CellType.WALL
    cost[:, :] = np.inf

    # ── Row 2 corridor ──
    for c in range(1, W - 1):
        ct[2, c] = CellType.NORMAL
        cost[2, c] = 1.0
        features[2, c] = np.array([0.5, 1.0, 0.0, 0.0])

    seg_start = 1 + CORR_W
    seg_end = seg_start + seg_width - 1

    # Wall row 2 inside segment (force lane choice)
    for c in range(seg_start, seg_end + 1):
        ct[2, c] = CellType.WALL
        cost[2, c] = np.inf

    # Punch entry/exit holes
    ct[2, seg_start] = CellType.NORMAL
    cost[2, seg_start] = 1.0
    ct[2, seg_end] = CellType.NORMAL
    cost[2, seg_end] = 1.0

    # ── Randomize which branch is risky ──
    risky_row = rng.choice([1, 3])
    safe_row = 3 if risky_row == 1 else 1

    # ── Entry/exit gates ──
    for r in [1, 3]:
        ct[r, seg_start] = CellType.NORMAL
        cost[r, seg_start] = 1.0
        features[r, seg_start] = np.array([
            0.0 if r == 1 else 1.0, 1.0, 0.0, 0.0])
        ct[r, seg_end] = CellType.NORMAL
        cost[r, seg_end] = 1.0
        features[r, seg_end] = np.array([
            0.0 if r == 1 else 1.0, 1.0, 0.0, 0.0])

    risky_entry = (risky_row, seg_start)
    safe_entry = (safe_row, seg_start)

    # ── Risky branch (straight) ──
    risky_cells = []
    for c in range(seg_start + 1, seg_end):
        ct[risky_row, c] = CellType.RISKY
        cost[risky_row, c] = 1.0
        risky_cells.append((risky_row, c))

    # ── Safe branch with zigzag detour ──
    safe_cells = []
    dt = 1  # detour length
    detour_start = seg_start + (seg_width // 2) - dt
    detour_end = detour_start + dt

    for c in range(seg_start + 1, seg_end):
        if detour_start <= c < detour_end:
            ct[safe_row, c] = CellType.WALL
            cost[safe_row, c] = np.inf
        else:
            ct[safe_row, c] = CellType.NORMAL
            cost[safe_row, c] = 1.0
            safe_cells.append((safe_row, c))

    # Detour vertical connections
    dc_down = detour_start
    dc_up = detour_end

    # Open detour path through rows 4, 5 (or 4, 0 if safe_row == 1)
    if safe_row == 3:
        detour_rows = [4, 5]
    else:
        # safe_row == 1, detour not needed in current architecture
        # but we still use rows 4, 5 for consistency
        detour_rows = [4, 5]

    for c in [dc_down, dc_up]:
        for dr in detour_rows:
            ct[dr, c] = CellType.NORMAL
            cost[dr, c] = 1.0
            safe_cells.append((dr, c))

    for c in range(dc_down, dc_up + 1):
        ct[detour_rows[-1], c] = CellType.NORMAL
        cost[detour_rows[-1], c] = 1.0
        safe_cells.append((detour_rows[-1], c))

    # Extra vertical connectors
    if dc_down - 1 >= seg_start:
        for dr in detour_rows:
            ct[dr, dc_down - 1] = CellType.NORMAL
            cost[dr, dc_down - 1] = 1.0
            safe_cells.append((dr, dc_down - 1))
    if dc_up + 1 <= seg_end:
        for dr in detour_rows:
            ct[dr, dc_up + 1] = CellType.NORMAL
            cost[dr, dc_up + 1] = 1.0
            safe_cells.append((dr, dc_up + 1))

    # ── Features: ambiguous cues ──
    # Key: front cells of both branches look similar
    # Risky branch: escalating risk deeper in
    trap_cell = None
    weak_cue_cells = []

    for i, (r, c) in enumerate(risky_cells):
        depth = i  # 0-indexed depth from entry
        if depth >= trap_depth and trap_cell is None:
            # Trap cell
            trap_cell = (r, c)
            risk[r, c] = trap_risk
            # Feature: high texture but blended with ambiguity
            t1 = 0.80 - ambiguity * 0.3 + rng.uniform(-0.05, 0.05)
            t2 = 0.70 - ambiguity * 0.2 + rng.uniform(-0.05, 0.05)
            features[r, c] = np.array([0.0 if r == 1 else 1.0, 0.0,
                                        np.clip(t1, 0.1, 0.95),
                                        np.clip(t2, 0.1, 0.90)])
        elif depth < trap_depth:
            # Pre-trap: looks AMBIGUOUS — similar to safe branch
            risk[r, c] = rng.uniform(0.03, 0.08)
            # Texture blended toward safe side by ambiguity
            base_t1 = rng.uniform(0.10, 0.25)
            base_t2 = rng.uniform(0.05, 0.15)
            features[r, c] = np.array([0.0 if r == 1 else 1.0, 0.0,
                                        base_t1, base_t2])
            weak_cue_cells.append((r, c))
        else:
            # Post-trap: moderate risk, clearer cue
            risk[r, c] = rng.uniform(0.15, 0.25)
            features[r, c] = np.array([0.0 if r == 1 else 1.0, 0.0,
                                        rng.uniform(0.30, 0.50),
                                        rng.uniform(0.20, 0.40)])
            weak_cue_cells.append((r, c))

    # Safe branch: low risk, features similar to pre-trap risky cells
    for r, c in safe_cells:
        lid = 0.0 if r == 1 else 1.0
        # With high ambiguity, safe cells look similar to risky pre-trap
        if ambiguity > 0.5:
            t1 = rng.uniform(0.08, 0.20)
            t2 = rng.uniform(0.05, 0.15)
        else:
            t1 = rng.uniform(0.0, 0.10)
            t2 = rng.uniform(0.0, 0.08)
        features[r, c] = np.array([lid, 0.0, t1, t2])

    # ── Build map ──
    agent_start = (2, 1)
    target_pos = (2, W - 2)
    gm = _build_gridmap(H, W, ct, cost, risk,
                         agent_start, target_pos, target_pos, [])

    # ── Path lengths ──
    shortest_any = _bfs_len(gm, agent_start, target_pos, set())
    risky_gates = {risky_entry}
    shortest_safe = _bfs_len(gm, agent_start, target_pos, risky_gates)
    base = shortest_safe if shortest_safe < 999 else 20
    t_max = max(int(time_ratio * base), base + 2)

    # ── Latent mode ──
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
                             agent_start, target_pos, target_pos, [])

    seg_meta = SegmentMeta(
        index=0, col_start=seg_start, col_end=seg_end,
        risky_row=risky_row, safe_row=safe_row,
        L_risky=seg_width - 1, L_safe=0,
        detour_len=dt,
        risky_cells=risky_cells, safe_cells=safe_cells,
        risky_entry_gate=risky_entry,
        safe_entry_gate=safe_entry,
        trap_cell=trap_cell,
        weak_cue_cells=weak_cue_cells,
    )

    meta = LatticeV2Meta(
        segments=[seg_meta],
        all_gate_cells=[risky_entry, safe_entry],
        all_door_positions=[risky_entry],
        shortest_any=shortest_any,
        shortest_safe=shortest_safe,
        cell_features=features,
        world_weights=ww,
        latent_mode=latent_mode,
    )

    cfg = FamilyConfig(
        max_steps=t_max, risk_budget=1.0,
        prior_risk_mean=0.02, prior_risk_var=0.20,
        search_budget=30, budget_class=8,
    )

    sc = ScenarioConfig(
        family_name="fork_trap",
        difficulty=difficulty,
        primary_intervention="WARN",
        cue_reliability=1.0 - ambiguity,
        expected_failure_mode="risk",
    )

    return gm, cfg, meta, sc


# ══════════════════════════════════════════════════════════════════════
# hazard_belt — Unavoidable Risk Zone + Shield
# ══════════════════════════════════════════════════════════════════════

HAZARD_BELT_PARAMS = {
    "easy":   {"belt_width": 2, "belt_risk": 0.25, "bypass_extra": 6,
               "time_ratio": 1.50},
    "medium": {"belt_width": 2, "belt_risk": 0.30, "bypass_extra": 8,
               "time_ratio": 1.35},
    "hard":   {"belt_width": 3, "belt_risk": 0.35, "bypass_extra": 10,
               "time_ratio": 1.20},
}


def generate_hazard_belt(
    seed: int,
    difficulty: DifficultyLevel = "medium",
    latent_mode: bool = True,
    belt_regime: str = "unavoidable",
    **kwargs,
) -> tuple[GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig]:
    """Hazard Belt — unavoidable high-risk zone.

    Topology: safe_seg → BELT (both lanes risky) → safe_seg → goal.
    Shield halves belt risk → primary intervention lever.

    belt_regime:
        "unavoidable" — no safe lane in belt segment, direct path through risk
        "near_unavoidable" — safe bypass exists but costs bypass_extra steps
    """
    rng = np.random.default_rng(seed)
    params = HAZARD_BELT_PARAMS[difficulty]
    belt_w = params["belt_width"]
    belt_risk_val = params["belt_risk"]
    bypass_extra = params["bypass_extra"]
    time_ratio = params["time_ratio"]

    # 3 segments: safe + belt + safe
    seg_widths = [5, belt_w + 2, 5]  # belt segment slightly wider
    CORR_W = 1
    W = 1 + CORR_W
    for sw in seg_widths:
        W += sw + CORR_W
    W += 1

    H = 7
    ct, cost, risk = _empty_grid(H, W)
    features = np.full((H, W, FEATURE_DIM), 0.5, dtype=np.float64)
    ct[:, :] = CellType.WALL
    cost[:, :] = np.inf

    # Row 2 corridor
    for c in range(1, W - 1):
        ct[2, c] = CellType.NORMAL
        cost[2, c] = 1.0
        features[2, c] = np.array([0.5, 1.0, 0.0, 0.0])

    segments_meta = []
    all_gates = []
    all_doors = []
    col_cursor = 1 + CORR_W

    for seg_i, sw in enumerate(seg_widths):
        seg_start = col_cursor
        seg_end = col_cursor + sw - 1
        is_belt = (seg_i == 1)

        # Wall row 2 inside segment
        for c in range(seg_start, seg_end + 1):
            ct[2, c] = CellType.WALL
            cost[2, c] = np.inf
        ct[2, seg_start] = CellType.NORMAL
        cost[2, seg_start] = 1.0
        ct[2, seg_end] = CellType.NORMAL
        cost[2, seg_end] = 1.0

        # Entry/exit gates
        for r in [1, 3]:
            ct[r, seg_start] = CellType.NORMAL
            cost[r, seg_start] = 1.0
            features[r, seg_start] = np.array([
                0.0 if r == 1 else 1.0, 1.0, 0.0, 0.0])
            ct[r, seg_end] = CellType.NORMAL
            cost[r, seg_end] = 1.0
            features[r, seg_end] = np.array([
                0.0 if r == 1 else 1.0, 1.0, 0.0, 0.0])

        risky_entry = (1, seg_start)
        safe_entry = (3, seg_start)
        all_gates.extend([risky_entry, safe_entry])
        all_doors.append(risky_entry)

        risky_cells = []
        safe_cells = []
        trap_cell = None
        weak_cue_cells = []

        if is_belt:
            # BELT SEGMENT: both lanes are risky
            for r in [1, 3]:
                for c in range(seg_start + 1, seg_end):
                    ct[r, c] = CellType.RISKY
                    cost[r, c] = 1.0
                    risk[r, c] = belt_risk_val
                    features[r, c] = np.array([
                        0.0 if r == 1 else 1.0, 0.0,
                        rng.uniform(0.70, 0.90),
                        rng.uniform(0.65, 0.85)])
                    risky_cells.append((r, c))

            if belt_regime == "near_unavoidable":
                # Add bypass: expensive detour through rows 4-5
                for c in range(seg_start, seg_end + 1):
                    for dr in [4, 5]:
                        ct[dr, c] = CellType.NORMAL
                        cost[dr, c] = 3.0  # high cost
                        features[dr, c] = np.array([0.5, 0.0, 0.1, 0.1])
                        safe_cells.append((dr, c))
        else:
            # NON-BELT SEGMENT: mild risk (agent should survive to reach belt)
            for c in range(seg_start + 1, seg_end):
                ct[1, c] = CellType.RISKY
                cost[1, c] = 1.0
                # Very low risk in non-belt segments — belt is the real challenge
                risk[1, c] = rng.uniform(0.02, 0.08)
                # Use safe-leaning features so latent ww.true_risk also stays low
                features[1, c] = _safe_feature(rng, 0.0)
                risky_cells.append((1, c))

            # Safe lane with detour
            dt = 1
            detour_start = seg_start + (sw // 2)
            detour_end = detour_start + dt
            for c in range(seg_start + 1, seg_end):
                if detour_start <= c < detour_end:
                    ct[3, c] = CellType.WALL
                    cost[3, c] = np.inf
                else:
                    ct[3, c] = CellType.NORMAL
                    cost[3, c] = 1.0
                    safe_cells.append((3, c))
                    features[3, c] = _safe_feature(rng, 1.0)

            dc_down = detour_start
            dc_up = detour_end
            for c in [dc_down, dc_up]:
                for dr in [4, 5]:
                    ct[dr, c] = CellType.NORMAL
                    cost[dr, c] = 1.0
                    safe_cells.append((dr, c))
            for c in range(dc_down, dc_up + 1):
                ct[5, c] = CellType.NORMAL
                cost[5, c] = 1.0
                safe_cells.append((5, c))
            if dc_down - 1 >= seg_start:
                for dr in [4, 5]:
                    ct[dr, dc_down - 1] = CellType.NORMAL
                    cost[dr, dc_down - 1] = 1.0
                    safe_cells.append((dr, dc_down - 1))
            if dc_up + 1 <= seg_end:
                for dr in [4, 5]:
                    ct[dr, dc_up + 1] = CellType.NORMAL
                    cost[dr, dc_up + 1] = 1.0
                    safe_cells.append((dr, dc_up + 1))

        segments_meta.append(SegmentMeta(
            index=seg_i, col_start=seg_start, col_end=seg_end,
            risky_row=1, safe_row=3,
            L_risky=0, L_safe=0, detour_len=0,
            risky_cells=risky_cells, safe_cells=safe_cells,
            risky_entry_gate=risky_entry,
            safe_entry_gate=safe_entry,
            trap_cell=trap_cell,
            weak_cue_cells=weak_cue_cells,
        ))
        col_cursor = seg_end + 1 + CORR_W

    agent_start = (2, 1)
    target_pos = (2, W - 2)
    gm = _build_gridmap(H, W, ct, cost, risk,
                         agent_start, target_pos, target_pos, [])

    shortest_any = _bfs_len(gm, agent_start, target_pos, set())
    risky_gates = {seg.risky_entry_gate for seg in segments_meta}
    shortest_safe = _bfs_len(gm, agent_start, target_pos, risky_gates)
    base = shortest_safe if shortest_safe < 999 else shortest_any
    t_max = max(int(time_ratio * base), base + 2)

    # Latent mode
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

        # Post-process: enforce family risk structure despite latent mapping.
        # Belt cells stay dangerous; everything else gets clamped safe.
        belt_cells_set = set()
        for seg in segments_meta:
            if seg.index == 1:  # belt segment
                belt_cells_set.update(seg.risky_cells)
        for r in range(H):
            for c in range(W):
                if ct[r, c] == CellType.WALL:
                    continue
                if (r, c) in belt_cells_set:
                    # Belt: scale latent risk into intended belt range
                    risk[r, c] = belt_risk_val + rng.uniform(-0.05, 0.05)
                elif ct[r, c] == CellType.RISKY:
                    # Non-belt risky: cap to mild risk
                    risk[r, c] = min(risk[r, c], 0.05)
                else:
                    # Corridor / safe / normal cells: very low risk
                    risk[r, c] = min(risk[r, c], 0.03)

        gm = _build_gridmap(H, W, ct, cost, risk,
                             agent_start, target_pos, target_pos, [])

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

    cfg = FamilyConfig(
        max_steps=t_max, risk_budget=1.0,
        prior_risk_mean=0.02, prior_risk_var=0.20,
        search_budget=30, budget_class=8,
    )

    sc = ScenarioConfig(
        family_name="hazard_belt",
        difficulty=difficulty,
        primary_intervention="ITEM_DROP",
        hazard_density=belt_w / (W - 2),
        requires_item=True,
        expected_failure_mode="risk",
        belt_regime=belt_regime,
    )

    return gm, cfg, meta, sc


# ══════════════════════════════════════════════════════════════════════
# deadline_gate — Tight Deadline + Gated Shortcut
# ══════════════════════════════════════════════════════════════════════

DEADLINE_GATE_PARAMS = {
    # safe_risk: per-cell risk on safe-path RISKY cells
    # n_risky_per_seg: RISKY cells per safe-path segment
    "easy":   {"n_long_segments": 3, "shortcut_len": 5, "shortcut_risk": 0.0,
               "safe_risk": 0.15, "n_risky_per_seg": 1, "time_ratio": 1.15},
    "medium": {"n_long_segments": 4, "shortcut_len": 5, "shortcut_risk": 0.0,
               "safe_risk": 0.20, "n_risky_per_seg": 2, "time_ratio": 1.10},
    "hard":   {"n_long_segments": 4, "shortcut_len": 5, "shortcut_risk": 0.0,
               "safe_risk": 0.25, "n_risky_per_seg": 2, "time_ratio": 1.05},
}


def generate_deadline_gate(
    seed: int,
    difficulty: DifficultyLevel = "medium",
    latent_mode: bool = True,
    **kwargs,
) -> tuple[GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig]:
    """Deadline Shortcut Gate — tight deadline + gated shortcut.

    Safe long path is always theoretically possible (even in hard mode),
    but tight deadline makes shortcut highly valuable.
    Shortcut is GATED (closed by default). UNLOCK opens it.

    gate_mode = "unlock_shortcut": shortcut gate starts closed,
    tutor can UNLOCK to give agent access.
    """
    rng = np.random.default_rng(seed)
    params = DEADLINE_GATE_PARAMS[difficulty]
    n_long = params["n_long_segments"]
    shortcut_len = params["shortcut_len"]
    shortcut_risk_val = params["shortcut_risk"]
    safe_risk_val = params.get("safe_risk", 0.15)
    n_risky_per_seg = params.get("n_risky_per_seg", 1)
    time_ratio = params["time_ratio"]

    # Long safe path: n_long segments of width 5
    long_seg_widths = [5] * n_long
    # Shortcut: one narrow segment, gated
    # Layout: shortcut on row 1, long path on rows 3-5

    CORR_W = 1
    total_long_width = sum(long_seg_widths) + (n_long - 1) * CORR_W
    # Shortcut spans same total column range as long path
    W = 1 + CORR_W + total_long_width + CORR_W + 1
    H = 7

    ct, cost, risk = _empty_grid(H, W)
    features = np.full((H, W, FEATURE_DIM), 0.5, dtype=np.float64)
    ct[:, :] = CellType.WALL
    cost[:, :] = np.inf

    # Row 2 corridor
    for c in range(1, W - 1):
        ct[2, c] = CellType.NORMAL
        cost[2, c] = 1.0
        features[2, c] = np.array([0.5, 1.0, 0.0, 0.0])

    segments_meta = []
    all_gates = []
    all_doors = []

    # Combined span for all long segments
    long_start = 1 + CORR_W
    long_end = long_start + total_long_width - 1

    # ── Shortcut (row 1): straight, gated at entry ──
    # Shortcut spans from long_start to long_end on row 1
    shortcut_gate = (1, long_start)  # GATE: closed by default
    ct[1, long_start] = CellType.LOCKED_DOOR  # closed gate
    cost[1, long_start] = np.inf

    # Shortcut: ALL NORMAL cells (zero risk once unlocked).
    # UNLOCK is pure topology assistance — the shortcut is genuinely safe.
    shortcut_risky_cells = []
    for c in range(long_start + 1, long_end + 1):
        ct[1, c] = CellType.NORMAL
        cost[1, c] = 1.0
        risk[1, c] = 0.0
        features[1, c] = _safe_feature(rng, 0.0)

    # Shortcut entry/exit connectors on row 2
    ct[2, long_start] = CellType.NORMAL
    cost[2, long_start] = 1.0
    ct[2, long_end] = CellType.NORMAL
    cost[2, long_end] = 1.0

    # Shortcut exit needs connector from row 1 to row 2 at long_end
    ct[1, long_end] = CellType.NORMAL
    cost[1, long_end] = 1.0

    # Shortcut segment meta
    segments_meta.append(SegmentMeta(
        index=0, col_start=long_start, col_end=long_end,
        risky_row=1, safe_row=3,
        L_risky=long_end - long_start, L_safe=0, detour_len=0,
        risky_cells=shortcut_risky_cells, safe_cells=[],
        risky_entry_gate=shortcut_gate,
        safe_entry_gate=(3, long_start),
        trap_cell=None, weak_cue_cells=[],
    ))
    all_gates.append(shortcut_gate)
    all_doors.append(shortcut_gate)

    # ── Long safe path (rows 3-5): multiple segments ──
    col_cursor = long_start
    for seg_i, sw in enumerate(long_seg_widths):
        seg_start = col_cursor
        seg_end = col_cursor + sw - 1

        # Wall row 2 inside segment
        for c in range(seg_start, seg_end + 1):
            ct[2, c] = CellType.WALL
            cost[2, c] = np.inf
        ct[2, seg_start] = CellType.NORMAL
        cost[2, seg_start] = 1.0
        ct[2, seg_end] = CellType.NORMAL
        cost[2, seg_end] = 1.0

        # Row 3 safe path — includes genuine RISKY hazard cells
        safe_cells = []
        seg_cols = list(range(seg_start, seg_end + 1))
        # Place n_risky_per_seg RISKY cells on the safe path
        n_rsk = min(n_risky_per_seg, len(seg_cols))
        risky_safe_idx = set(rng.choice(len(seg_cols), size=n_rsk, replace=False))
        risky_cells_in_seg = []
        for i_col, c in enumerate(seg_cols):
            if i_col in risky_safe_idx:
                ct[3, c] = CellType.RISKY
                cost[3, c] = 1.0
                risk[3, c] = safe_risk_val
                features[3, c] = _lane_feature(rng, 0.0, True)
                risky_cells_in_seg.append((3, c))
            else:
                ct[3, c] = CellType.NORMAL
                cost[3, c] = 1.0
                features[3, c] = _safe_feature(rng, 1.0)
            safe_cells.append((3, c))

        # Zigzag detour through rows 4-5
        dt = 1
        mid = seg_start + sw // 2
        if seg_start + 1 < seg_end:
            ct[3, mid] = CellType.WALL
            cost[3, mid] = np.inf
            safe_cells = [(r, c) for r, c in safe_cells if (r, c) != (3, mid)]
            for dr in [4, 5]:
                for dc in [mid - 1, mid, mid + 1]:
                    if seg_start <= dc <= seg_end:
                        ct[dr, dc] = CellType.NORMAL
                        cost[dr, dc] = 1.0
                        safe_cells.append((dr, dc))
                        features[dr, dc] = _safe_feature(rng, 1.0)

        safe_entry = (3, seg_start)
        all_gates.append(safe_entry)

        segments_meta.append(SegmentMeta(
            index=seg_i + 1, col_start=seg_start, col_end=seg_end,
            risky_row=1, safe_row=3,
            L_risky=0, L_safe=0, detour_len=dt,
            risky_cells=[], safe_cells=safe_cells,
            risky_entry_gate=(1, seg_start),
            safe_entry_gate=safe_entry,
            trap_cell=None, weak_cue_cells=[],
        ))

        col_cursor = seg_end + 1 + CORR_W

    # Ensure corridor between segments is passable on row 2
    for c in range(1, W - 1):
        if ct[2, c] == CellType.WALL:
            # Check if it's between segments
            in_any_seg = False
            for seg in segments_meta:
                if seg.col_start < c < seg.col_end:
                    in_any_seg = True
                    break
            if not in_any_seg:
                ct[2, c] = CellType.NORMAL
                cost[2, c] = 1.0

    agent_start = (2, 1)
    target_pos = (2, W - 2)
    gm = _build_gridmap(H, W, ct, cost, risk,
                         agent_start, target_pos, target_pos,
                         [shortcut_gate])

    shortest_any = _bfs_len(gm, agent_start, target_pos, set())
    # Safe path: avoid shortcut (gated)
    shortest_safe = _bfs_len(gm, agent_start, target_pos, {shortcut_gate})
    # Shortcut path: if gate were open
    gm_open = _build_gridmap(H, W, ct, cost, risk,
                              agent_start, target_pos, target_pos, [])
    # Temporarily make gate passable for BFS
    ct_save = ct[shortcut_gate[0], shortcut_gate[1]]
    cost_save = cost[shortcut_gate[0], shortcut_gate[1]]
    ct[shortcut_gate[0], shortcut_gate[1]] = CellType.NORMAL
    cost[shortcut_gate[0], shortcut_gate[1]] = 1.0
    gm_open = _build_gridmap(H, W, ct, cost, risk,
                              agent_start, target_pos, target_pos, [])
    shortest_with_shortcut = _bfs_len(gm_open, agent_start, target_pos, set())
    ct[shortcut_gate[0], shortcut_gate[1]] = ct_save
    cost[shortcut_gate[0], shortcut_gate[1]] = cost_save

    # Time budget: tight at higher difficulties.
    # Necessity-aware planner handles unvisited cell planning.
    base = shortest_safe if shortest_safe < 999 else 30
    t_max = max(int(time_ratio * base), base + 1)

     # Latent mode
    ww = None
    if latent_mode:
        from ..agents.cost_risk_model import generate_world_weights
        ww = generate_world_weights(rng, d=FEATURE_DIM)
        for r in range(H):
            for c in range(W):
                if ct[r, c] == CellType.WALL or ct[r, c] == CellType.LOCKED_DOOR:
                    continue
                z = features[r, c]
                cost[r, c] = ww.true_cost(z)
                risk[r, c] = ww.true_risk(z)

        # Post-process: enforce family risk structure despite latent mapping.
        # Safe-path RISKY cells KEEP intended risk (this is Fix C).
        # Shortcut cells (all NORMAL) forced to zero risk.
        shortcut_cols_set = {(1, c) for c in range(long_start + 1, long_end + 1)}
        for r in range(H):
            for c in range(W):
                if ct[r, c] in (CellType.WALL, CellType.LOCKED_DOOR):
                    continue
                if (r, c) in shortcut_cols_set:
                    # Shortcut: genuinely safe (zero risk)
                    risk[r, c] = 0.0
                elif ct[r, c] == CellType.RISKY:
                    # Safe-path RISKY cells: keep intended family risk
                    risk[r, c] = safe_risk_val + rng.uniform(-0.03, 0.03)
                else:
                    # Corridor / normal: low risk
                    risk[r, c] = min(risk[r, c], 0.03)

        gm = _build_gridmap(H, W, ct, cost, risk,
                             agent_start, target_pos, target_pos,
                             [shortcut_gate])

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

    cfg = FamilyConfig(
        max_steps=t_max, risk_budget=1.0,
        prior_risk_mean=0.02, prior_risk_var=0.20,
        search_budget=30, budget_class=8,
    )

    sc = ScenarioConfig(
        family_name="deadline_gate",
        difficulty=difficulty,
        primary_intervention="UNLOCK",
        requires_gate=True,
        expected_failure_mode="timeout",
        gate_mode="unlock_shortcut",
    )

    return gm, cfg, meta, sc


# ══════════════════════════════════════════════════════════════════════
# delayed_corridor — Late-Revealing Risk
# ══════════════════════════════════════════════════════════════════════

DELAYED_CORRIDOR_PARAMS = {
    "easy":   {"corridor_len": 7, "safe_prefix": 2, "deep_risk": 0.35,
               "time_ratio": 1.30},
    "medium": {"corridor_len": 8, "safe_prefix": 3, "deep_risk": 0.45,
               "time_ratio": 1.20},
    "hard":   {"corridor_len": 9, "safe_prefix": 4, "deep_risk": 0.60,
               "time_ratio": 1.10},
}


def generate_delayed_corridor(
    seed: int,
    difficulty: DifficultyLevel = "medium",
    latent_mode: bool = True,
    **kwargs,
) -> tuple[GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig]:
    """Delayed Commitment Corridor — risk hidden behind safe-looking prefix.

    A single long segment. Corridor A looks safe at entry (first
    safe_prefix cells), but risk escalates past the commitment point.
    Once the agent is deep enough, backtracking + taking corridor B
    exceeds the deadline.

    Prefix-aware WARN is the primary lever: a tutor that can predict
    the agent's path prefix can warn BEFORE the commitment point.
    A myopic tutor that only sees current-cell risk will warn too late.
    """
    rng = np.random.default_rng(seed)
    params = DELAYED_CORRIDOR_PARAMS[difficulty]
    corridor_len = params["corridor_len"]
    safe_prefix = params["safe_prefix"]
    deep_risk = params["deep_risk"]
    time_ratio = params["time_ratio"]

    # Single long segment
    CORR_W = 1
    seg_width = corridor_len
    W = 1 + CORR_W + seg_width + CORR_W + 1
    H = 7

    ct, cost, risk = _empty_grid(H, W)
    features = np.full((H, W, FEATURE_DIM), 0.5, dtype=np.float64)
    ct[:, :] = CellType.WALL
    cost[:, :] = np.inf

    # Row 2 corridor
    for c in range(1, W - 1):
        ct[2, c] = CellType.NORMAL
        cost[2, c] = 1.0
        features[2, c] = np.array([0.5, 1.0, 0.0, 0.0])

    seg_start = 1 + CORR_W
    seg_end = seg_start + seg_width - 1

    # Wall row 2 inside segment
    for c in range(seg_start, seg_end + 1):
        ct[2, c] = CellType.WALL
        cost[2, c] = np.inf
    ct[2, seg_start] = CellType.NORMAL
    cost[2, seg_start] = 1.0
    ct[2, seg_end] = CellType.NORMAL
    cost[2, seg_end] = 1.0

    # Entry/exit gates
    for r in [1, 3]:
        ct[r, seg_start] = CellType.NORMAL
        cost[r, seg_start] = 1.0
        features[r, seg_start] = np.array([
            0.0 if r == 1 else 1.0, 1.0, 0.0, 0.0])
        ct[r, seg_end] = CellType.NORMAL
        cost[r, seg_end] = 1.0
        features[r, seg_end] = np.array([
            0.0 if r == 1 else 1.0, 1.0, 0.0, 0.0])

    risky_entry = (1, seg_start)
    safe_entry = (3, seg_start)

    # ── Corridor A (row 1): deceptive — safe prefix, then deep risk ──
    risky_cells = []
    trap_cell = None
    weak_cue_cells = []
    commitment_cells = []

    for c in range(seg_start + 1, seg_end):
        depth = c - seg_start - 1  # 0-indexed
        ct[1, c] = CellType.RISKY
        cost[1, c] = 1.0
        risky_cells.append((1, c))

        if depth < safe_prefix:
            # Safe-looking prefix: LOW features, LOW risk
            risk[1, c] = rng.uniform(0.02, 0.06)
            features[1, c] = np.array([0.0, 0.0,
                                        rng.uniform(0.03, 0.12),
                                        rng.uniform(0.02, 0.10)])
            weak_cue_cells.append((1, c))
        else:
            # Deep zone: HIGH risk, features become diagnostic
            risk[1, c] = deep_risk + rng.uniform(-0.05, 0.05)
            features[1, c] = np.array([0.0, 0.0,
                                        rng.uniform(0.65, 0.85),
                                        rng.uniform(0.55, 0.80)])
            if trap_cell is None:
                trap_cell = (1, c)
            # Mark commitment point = first deep-risk cell
            if depth == safe_prefix:
                commitment_cells.append((1, c))

    # ── Corridor B (row 3): safe but longer (zigzag detour) ──
    safe_cells = []
    dt = 2  # larger detour → corridor B is significantly longer
    detour_start = seg_start + (seg_width // 2) - dt
    detour_end = detour_start + dt

    for c in range(seg_start + 1, seg_end):
        if detour_start <= c < detour_end:
            ct[3, c] = CellType.WALL
            cost[3, c] = np.inf
        else:
            ct[3, c] = CellType.NORMAL
            cost[3, c] = 1.0
            safe_cells.append((3, c))
            features[3, c] = _safe_feature(rng, 1.0)

    # Detour through rows 4, 5
    dc_down = detour_start
    dc_up = detour_end
    for c in range(dc_down - 1, dc_up + 2):
        if seg_start <= c <= seg_end:
            for dr in [4, 5]:
                ct[dr, c] = CellType.NORMAL
                cost[dr, c] = 1.0
                safe_cells.append((dr, c))
                features[dr, c] = _safe_feature(rng, 1.0)

    # Build map
    agent_start = (2, 1)
    target_pos = (2, W - 2)
    gm = _build_gridmap(H, W, ct, cost, risk,
                         agent_start, target_pos, target_pos, [])

    shortest_any = _bfs_len(gm, agent_start, target_pos, set())
    risky_gates = {risky_entry}
    shortest_safe = _bfs_len(gm, agent_start, target_pos, risky_gates)
    base = shortest_safe if shortest_safe < 999 else 25
    t_max = max(int(time_ratio * base), base + 2)

    # Latent mode
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
                             agent_start, target_pos, target_pos, [])

    seg_meta = SegmentMeta(
        index=0, col_start=seg_start, col_end=seg_end,
        risky_row=1, safe_row=3,
        L_risky=seg_width - 1, L_safe=0,
        detour_len=dt,
        risky_cells=risky_cells, safe_cells=safe_cells,
        risky_entry_gate=risky_entry,
        safe_entry_gate=safe_entry,
        trap_cell=trap_cell,
        weak_cue_cells=weak_cue_cells,
    )

    meta = LatticeV2Meta(
        segments=[seg_meta],
        all_gate_cells=[risky_entry, safe_entry],
        all_door_positions=[risky_entry],
        shortest_any=shortest_any,
        shortest_safe=shortest_safe,
        cell_features=features,
        world_weights=ww,
        latent_mode=latent_mode,
    )

    cfg = FamilyConfig(
        max_steps=t_max, risk_budget=1.0,
        prior_risk_mean=0.02, prior_risk_var=0.20,
        search_budget=30, budget_class=8,
    )

    sc = ScenarioConfig(
        family_name="delayed_corridor",
        difficulty=difficulty,
        primary_intervention="WARN",
        expected_failure_mode="commitment",
        commitment_cells=commitment_cells,
    )

    return gm, cfg, meta, sc


# ══════════════════════════════════════════════════════════════════════
# distractor_cue — Misleading Local Cues
# ══════════════════════════════════════════════════════════════════════

DISTRACTOR_CUE_PARAMS = {
    "easy":   {"cue_reliability": 0.6, "distractor_frac": 0.10,
               "time_ratio": 1.50, "n_segments": 3},
    "medium": {"cue_reliability": 0.3, "distractor_frac": 0.25,
               "time_ratio": 1.35, "n_segments": 3},
    "hard":   {"cue_reliability": 0.0, "distractor_frac": 0.40,
               "time_ratio": 1.20, "n_segments": 3},
}


def generate_distractor_cue(
    seed: int,
    difficulty: DifficultyLevel = "medium",
    latent_mode: bool = True,
    cue_mode: str = "weak",
    **kwargs,
) -> tuple[GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig]:
    """Distractor Cue Corridor — misleading or unreliable local features.

    Standard V2 segment topology, but feature-risk correlation is
    controlled by cue_reliability:
        1.0 = features perfectly predict risk (standard V2)
        0.5 = noisy
        0.0 = uncorrelated
        negative = inverted (misleading)

    cue_mode:
        "weak"      — low correlation, agent gets noisy signals
        "misleading" — some cells have inverted features (safe looks risky
                       and risky looks safe), controlled by distractor_frac

    WARN is the primary lever because it provides GROUND TRUTH about
    risk, overriding the misleading features.
    """
    rng = np.random.default_rng(seed)
    params = DISTRACTOR_CUE_PARAMS[difficulty]
    reliability = params["cue_reliability"]
    distractor_frac = params["distractor_frac"]
    time_ratio = params["time_ratio"]
    n_segments = params["n_segments"]

    if cue_mode == "misleading":
        # Increase distractor fraction and allow negative reliability
        distractor_frac = min(distractor_frac + 0.15, 0.50)
        reliability = max(reliability - 0.3, -0.3)

    # Use standard V2 generation, then CORRUPT features
    gm, cfg, meta = generate_lattice_v2(
        seed=seed, difficulty=difficulty, n_segments=n_segments,
        latent_mode=False)  # Don't apply latent yet — we modify features first

    features = meta.cell_features.copy()
    H, W = gm.height, gm.width

    # ── Corrupt features based on cue_reliability ──
    weak_cue_cells = []
    for seg in meta.segments:
        all_lane_cells = seg.risky_cells + seg.safe_cells

        n_distractor = max(1, int(len(all_lane_cells) * distractor_frac))
        distractor_indices = rng.choice(
            len(all_lane_cells),
            size=min(n_distractor, len(all_lane_cells)),
            replace=False)

        for idx in distractor_indices:
            r, c = all_lane_cells[idx]
            actual_risk = gm.true_risk[r, c]
            is_risky = actual_risk > 0.15

            if cue_mode == "misleading":
                # INVERT: risky cells get safe-looking features, vice versa
                if is_risky:
                    features[r, c, F_TEXTURE_1] = rng.uniform(0.02, 0.12)
                    features[r, c, F_TEXTURE_2] = rng.uniform(0.01, 0.08)
                else:
                    features[r, c, F_TEXTURE_1] = rng.uniform(0.50, 0.80)
                    features[r, c, F_TEXTURE_2] = rng.uniform(0.40, 0.70)
            else:
                # WEAK: add noise proportional to (1 - reliability)
                noise_scale = 1.0 - reliability
                noise1 = rng.uniform(-0.3, 0.3) * noise_scale
                noise2 = rng.uniform(-0.3, 0.3) * noise_scale
                features[r, c, F_TEXTURE_1] = np.clip(
                    features[r, c, F_TEXTURE_1] + noise1, 0.0, 1.0)
                features[r, c, F_TEXTURE_2] = np.clip(
                    features[r, c, F_TEXTURE_2] + noise2, 0.0, 1.0)

            weak_cue_cells.append((r, c))

    # Update meta with corrupted features
    meta_new = LatticeV2Meta(
        segments=meta.segments,
        all_gate_cells=meta.all_gate_cells,
        all_door_positions=meta.all_door_positions,
        shortest_any=meta.shortest_any,
        shortest_safe=meta.shortest_safe,
        cell_features=features,
        world_weights=meta.world_weights,
        latent_mode=latent_mode,
    )

    # Record weak_cue_cells in each segment
    for seg in meta_new.segments:
        seg.weak_cue_cells = [
            (r, c) for r, c in weak_cue_cells
            if seg.col_start <= c <= seg.col_end]

    # ── Latent mode: derive cost/risk from (corrupted) features ──
    ww = None
    cost_arr = gm.true_cost.copy()
    risk_arr = gm.true_risk.copy()
    if latent_mode:
        from ..agents.cost_risk_model import generate_world_weights
        ww = generate_world_weights(rng, d=FEATURE_DIM)
        for r in range(H):
            for c in range(W):
                if gm.cell_types[r, c] == CellType.WALL:
                    continue
                z = features[r, c]
                cost_arr[r, c] = ww.true_cost(z)
                risk_arr[r, c] = ww.true_risk(z)
        gm = _build_gridmap(H, W, gm.cell_types.copy(), cost_arr, risk_arr,
                             gm.agent_start, gm.target_pos, gm.target_pos,
                             gm.door_positions)
        meta_new = LatticeV2Meta(
            segments=meta_new.segments,
            all_gate_cells=meta_new.all_gate_cells,
            all_door_positions=meta_new.all_door_positions,
            shortest_any=meta_new.shortest_any,
            shortest_safe=meta_new.shortest_safe,
            cell_features=features,
            world_weights=ww,
            latent_mode=True,
        )

    # Time budget
    base = meta_new.shortest_safe if meta_new.shortest_safe < 999 else 30
    t_max = max(int(time_ratio * base), base + 2)
    cfg = FamilyConfig(
        max_steps=t_max, risk_budget=1.0,
        prior_risk_mean=0.02, prior_risk_var=0.20,
        search_budget=30, budget_class=8,
    )

    sc = ScenarioConfig(
        family_name="distractor_cue",
        difficulty=difficulty,
        primary_intervention="WARN",
        cue_reliability=reliability,
        expected_failure_mode="cue_error",
    )

    return gm, cfg, meta_new, sc


# ══════════════════════════════════════════════════════════════════════
# funnel_trap — Multi-Stage Funnel Trap (complex family)
# ══════════════════════════════════════════════════════════════════════
#
# Two-stage topology:
#   Stage 1 — Wide fork: 3 entry branches → merge at M1
#   Stage 2 — Re-fork: 2 commitment corridors, one safe, one traps
#
# Topology (7-row grid, wider than basic families):
#   Row 0: wall
#   Row 1: branch A / late corridor α
#   Row 2: main corridor (S, M1, fork2, G)
#   Row 3: branch B / late corridor β
#   Row 4: branch C vertical connector
#   Row 5: branch C / detour
#   Row 6: wall
#
# Auto-labels:
#   decision_points: cells where branches diverge
#   commitment_points: cells past which backtracking exceeds deadline
#   cue_cells: cells with weak/ambiguous features
#   safe_branches: list of branch cell lists that are safe
#   trap_branches: list of branch cell lists that are traps
#   merge_cells: cells where branches rejoin
# ══════════════════════════════════════════════════════════════════════

FUNNEL_TRAP_PARAMS = {
    "easy":   {"n_entry": 3, "stage1_len": 5, "corridor_len": 3,
               "cue_ambiguity": 0.3, "trap_risk": 0.30,
               "time_ratio": 1.50},
    "medium": {"n_entry": 3, "stage1_len": 6, "corridor_len": 4,
               "cue_ambiguity": 0.6, "trap_risk": 0.45,
               "time_ratio": 1.35},
    "hard":   {"n_entry": 3, "stage1_len": 7, "corridor_len": 5,
               "cue_ambiguity": 0.9, "trap_risk": 0.60,
               "time_ratio": 1.20},
}


def generate_funnel_trap(
    seed: int,
    difficulty: DifficultyLevel = "medium",
    latent_mode: bool = True,
    **kwargs,
) -> tuple[GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig]:
    """Multi-Stage Funnel Trap — delayed commitment via 2-stage fork-merge.

    Stage 1: 3 branches with ambiguous cues → merge at M1
      - One branch has a weak danger cue (pre-trap zone)
      - The other two are safe but vary in length

    Stage 2: Re-fork into 2 commitment corridors
      - One corridor is safe (direct to goal)
      - One corridor has escalating risk (the funnel trap)
      - Once past the commitment point, backtracking exceeds deadline

    Primary intervention lever: WARN (timing-sensitive)
    Key experiment: does TPM warn BEFORE commitment point, not after?

    Structure labels stored in ScenarioConfig for metric computation:
      decision_points, commitment_points, cue_cells,
      safe_branches, trap_branches, merge_cells
    """
    rng = np.random.default_rng(seed)
    params = FUNNEL_TRAP_PARAMS[difficulty]
    s1_len = params["stage1_len"]       # Stage 1 branch length
    corr_len = params["corridor_len"]   # Stage 2 corridor length
    ambiguity = params["cue_ambiguity"]
    trap_risk = params["trap_risk"]
    time_ratio = params["time_ratio"]

    # ── Grid dimensions ──
    # Layout: [wall][entry corr][Stage1 fork][merge corr][Stage2 fork][exit corr][wall]
    CORR = 1   # corridor cells between stages
    W = 1 + CORR + s1_len + CORR + corr_len + CORR + 1
    H = 7

    ct, cost, risk = _empty_grid(H, W)
    features = np.full((H, W, FEATURE_DIM), 0.5, dtype=np.float64)
    ct[:, :] = CellType.WALL
    cost[:, :] = np.inf

    # ── Row 2: main corridor (passable everywhere except inside forks) ──
    for c in range(1, W - 1):
        ct[2, c] = CellType.NORMAL
        cost[2, c] = 1.0
        features[2, c] = np.array([0.5, 1.0, 0.0, 0.0])

    # ═══ Stage 1: 3-branch fork ═══════════════════════════════════
    s1_start = 1 + CORR      # first column of Stage 1
    s1_end = s1_start + s1_len - 1

    # Wall row 2 inside Stage 1 (force branch choice)
    for c in range(s1_start, s1_end + 1):
        ct[2, c] = CellType.WALL
        cost[2, c] = np.inf

    # Re-open entry and exit of Stage 1 on row 2
    ct[2, s1_start] = CellType.NORMAL
    cost[2, s1_start] = 1.0
    ct[2, s1_end] = CellType.NORMAL
    cost[2, s1_end] = 1.0

    # Randomize which branch carries the pre-trap cue
    danger_branch = rng.integers(0, 3)  # 0=row1, 1=row3, 2=row5
    branch_rows = [1, 3, 5]

    # Decision point: the fork entry on row 2
    decision_points = [(2, s1_start)]

    # Open entry/exit gates for branches
    for r in branch_rows[:2]:  # rows 1, 3
        ct[r, s1_start] = CellType.NORMAL
        cost[r, s1_start] = 1.0
        features[r, s1_start] = np.array([
            0.0 if r == 1 else 1.0, 1.0, 0.0, 0.0])
        ct[r, s1_end] = CellType.NORMAL
        cost[r, s1_end] = 1.0
        features[r, s1_end] = np.array([
            0.0 if r == 1 else 1.0, 1.0, 0.0, 0.0])

    # Row 5 branch uses row 4 as vertical connector
    ct[4, s1_start] = CellType.NORMAL
    cost[4, s1_start] = 1.0
    ct[5, s1_start] = CellType.NORMAL
    cost[5, s1_start] = 1.0
    features[5, s1_start] = np.array([0.0, 1.0, 0.0, 0.0])

    ct[4, s1_end] = CellType.NORMAL
    cost[4, s1_end] = 1.0
    ct[5, s1_end] = CellType.NORMAL
    cost[5, s1_end] = 1.0
    features[5, s1_end] = np.array([0.0, 1.0, 0.0, 0.0])

    # Build 3 branches
    branch_cells = {0: [], 1: [], 2: []}   # branch_idx -> list of cells
    cue_cells_all = []
    trap_cells_all = []
    safe_branch_cells = []
    trap_branch_cells = []

    for bi, brow in enumerate(branch_rows):
        cells = []
        for c in range(s1_start + 1, s1_end):
            if brow == 5:
                # Row 5 branch: open row 5
                ct[5, c] = CellType.NORMAL
                cost[5, c] = 1.0
                cells.append((5, c))
            else:
                ct[brow, c] = CellType.NORMAL
                cost[brow, c] = 1.0
                cells.append((brow, c))

        # Features: ambiguous cues
        if bi == danger_branch:
            # This branch has escalating weak cues → pre-trap
            ct_type = CellType.RISKY
            for i, (r, c) in enumerate(cells):
                ct[r, c] = ct_type
                depth_ratio = i / max(len(cells) - 1, 1)
                # Early cells: very ambiguous (look safe)
                # Deep cells: slightly riskier cue
                base_t1 = 0.10 + depth_ratio * (0.40 - ambiguity * 0.15)
                base_t2 = 0.05 + depth_ratio * (0.30 - ambiguity * 0.10)
                base_t1 += rng.uniform(-0.05, 0.05)
                base_t2 += rng.uniform(-0.05, 0.05)
                lid = 0.0 if r <= 2 else 1.0
                features[r, c] = np.array([
                    lid, 0.0,
                    np.clip(base_t1, 0.05, 0.90),
                    np.clip(base_t2, 0.05, 0.85)])
                risk[r, c] = rng.uniform(0.05, 0.15)  # modest risk in Stage 1
                cue_cells_all.append((r, c))
            trap_branch_cells.append(cells)
        else:
            # Safe branch: low-risk features
            for r, c in cells:
                lid = 0.0 if r <= 2 else 1.0
                features[r, c] = _safe_feature(rng, lid)
                risk[r, c] = rng.uniform(0.01, 0.05)
            safe_branch_cells.append(cells)

        branch_cells[bi] = cells

    # Merge point M1: row 2 just after Stage 1
    merge_col = s1_end + 1
    if merge_col < W - 1:
        ct[2, merge_col] = CellType.NORMAL
        cost[2, merge_col] = 1.0
    merge_cells = [(2, merge_col)]

    # ═══ Stage 2: Commitment Fork ═════════════════════════════════
    s2_start = merge_col + 1
    s2_end = s2_start + corr_len - 1

    # Check bounds
    s2_end = min(s2_end, W - 2)

    # Wall row 2 inside Stage 2 (force corridor choice)
    for c in range(s2_start, s2_end + 1):
        ct[2, c] = CellType.WALL
        cost[2, c] = np.inf

    # Re-open entry and exit on row 2
    ct[2, s2_start] = CellType.NORMAL
    cost[2, s2_start] = 1.0
    ct[2, s2_end] = CellType.NORMAL
    cost[2, s2_end] = 1.0

    # Decision point 2
    decision_points.append((2, s2_start))

    # Corridor assignment: safe must be row 3 (connects to detour rows 4-5).
    # Trap corridor is always row 1 (straight, tempting).
    trap_corridor = 1
    safe_corridor = 3

    corridor_α_cells = []
    corridor_β_cells = []

    for r in [1, 3]:
        # Open entry/exit
        ct[r, s2_start] = CellType.NORMAL
        cost[r, s2_start] = 1.0
        ct[r, s2_end] = CellType.NORMAL
        cost[r, s2_end] = 1.0

    # Trap corridor: straight (fast, tempting)
    trap_corr_cells = []
    for c in range(s2_start + 1, s2_end):
        ct[trap_corridor, c] = CellType.NORMAL
        cost[trap_corridor, c] = 1.0
        trap_corr_cells.append((trap_corridor, c))

    # Safe corridor: zigzag detour (slower but safe)
    # Put a wall gap at mid-point, force through rows 4-5
    safe_corr_cells = []
    s2_mid = s2_start + max(1, (s2_end - s2_start) // 2)
    for c in range(s2_start + 1, s2_end):
        if c == s2_mid:
            # Gap in safe corridor — wall here, force detour
            ct[safe_corridor, c] = CellType.WALL
            cost[safe_corridor, c] = np.inf
        else:
            ct[safe_corridor, c] = CellType.NORMAL
            cost[safe_corridor, c] = 1.0
            safe_corr_cells.append((safe_corridor, c))

    # Detour path: safe_corridor → row 4 → row 5 → row 4 → safe_corridor
    # Vertical connectors at gap boundaries
    dc_down = s2_mid - 1 if s2_mid - 1 >= s2_start else s2_mid
    dc_up = s2_mid + 1 if s2_mid + 1 <= s2_end else s2_mid

    detour_rows = [4, 5] if safe_corridor == 3 else [4, 5]
    for dc_col in [dc_down, dc_up]:
        for dr in detour_rows:
            if ct[dr, dc_col] == CellType.WALL:
                ct[dr, dc_col] = CellType.NORMAL
                cost[dr, dc_col] = 1.0
                safe_corr_cells.append((dr, dc_col))
    # Horizontal on row 5
    for c in range(dc_down, dc_up + 1):
        if ct[5, c] == CellType.WALL:
            ct[5, c] = CellType.NORMAL
            cost[5, c] = 1.0
            safe_corr_cells.append((5, c))

    # Commitment point: 2 cells into the trap corridor is irreversible
    commitment_depth = min(2, len(trap_corr_cells))
    commitment_points = []
    if commitment_depth > 0 and len(trap_corr_cells) >= commitment_depth:
        commitment_points.append(trap_corr_cells[commitment_depth - 1])

    # Trap corridor: escalating risk, ambiguous early cues
    main_trap_cell = None
    for i, (r, c) in enumerate(trap_corr_cells):
        ct[r, c] = CellType.RISKY
        depth_ratio = i / max(len(trap_corr_cells) - 1, 1)

        if depth_ratio < 0.4:
            # Early: looks similar to safe (ambiguous)
            t1 = rng.uniform(0.10, 0.25) + ambiguity * rng.uniform(-0.05, 0.05)
            t2 = rng.uniform(0.05, 0.15) + ambiguity * rng.uniform(-0.05, 0.05)
            risk[r, c] = rng.uniform(0.05, 0.15)
            cue_cells_all.append((r, c))
        else:
            # Late: real trap zone
            t1 = rng.uniform(0.50, 0.85) - ambiguity * 0.1
            t2 = rng.uniform(0.40, 0.70) - ambiguity * 0.1
            risk[r, c] = trap_risk * (0.5 + 0.5 * depth_ratio)
            if main_trap_cell is None:
                main_trap_cell = (r, c)
            trap_cells_all.append((r, c))

        lid = 0.0 if r == 1 else 1.0
        features[r, c] = np.array([lid, 0.0,
                                    np.clip(t1, 0.05, 0.95),
                                    np.clip(t2, 0.05, 0.90)])

    trap_branch_cells.append(trap_corr_cells)

    # Safe corridor: clean safe features
    for r, c in safe_corr_cells:
        lid = 0.0 if r == 1 else 1.0
        features[r, c] = _safe_feature(rng, lid)
        risk[r, c] = rng.uniform(0.01, 0.04)
    safe_branch_cells.append(safe_corr_cells)

    # ── Build gridmap ──
    agent_start = (2, 1)
    target_pos = (2, W - 2)
    gm = _build_gridmap(H, W, ct, cost, risk,
                         agent_start, target_pos, target_pos, [])

    # ── Path lengths ──
    shortest_any = _bfs_len(gm, agent_start, target_pos, set())
    # Shortest safe: avoid all trap branch cells
    trap_gates = set()
    for cells in trap_branch_cells:
        for cell in cells:
            trap_gates.add(cell)
    shortest_safe = _bfs_len(gm, agent_start, target_pos, trap_gates)
    base = shortest_safe if shortest_safe < 999 else 30
    t_max = max(int(time_ratio * base), base + 2)

    # ── Latent mode ──
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
                             agent_start, target_pos, target_pos, [])

    # ── Segments: describe both stages as SegmentMeta ──
    # Stage 1 segment (the fork)
    all_risky_s1 = branch_cells[danger_branch]
    all_safe_s1 = []
    for bi in range(3):
        if bi != danger_branch:
            all_safe_s1.extend(branch_cells[bi])

    risky_entry_s1 = (branch_rows[danger_branch], s1_start)
    safe_entry_s1 = (branch_rows[(danger_branch + 1) % 3], s1_start)

    seg1 = SegmentMeta(
        index=0, col_start=s1_start, col_end=s1_end,
        risky_row=branch_rows[danger_branch],
        safe_row=branch_rows[(danger_branch + 1) % 3],
        L_risky=s1_len - 1, L_safe=0,
        detour_len=0,
        risky_cells=all_risky_s1,
        safe_cells=all_safe_s1,
        risky_entry_gate=risky_entry_s1,
        safe_entry_gate=safe_entry_s1,
        trap_cell=None,  # Stage 1 has cues, not the main trap
        weak_cue_cells=[c for c in cue_cells_all if c[1] <= s1_end],
    )

    # Stage 2 segment (commitment corridor)
    seg2 = SegmentMeta(
        index=1, col_start=s2_start, col_end=s2_end,
        risky_row=trap_corridor,
        safe_row=safe_corridor,
        L_risky=corr_len - 1, L_safe=corr_len - 1,
        detour_len=0,
        risky_cells=trap_corr_cells,
        safe_cells=safe_corr_cells,
        risky_entry_gate=(trap_corridor, s2_start),
        safe_entry_gate=(safe_corridor, s2_start),
        trap_cell=main_trap_cell,
        weak_cue_cells=[c for c in cue_cells_all if c[1] >= s2_start],
    )

    all_gates = [
        risky_entry_s1, safe_entry_s1,
        (trap_corridor, s2_start), (safe_corridor, s2_start),
    ]
    all_doors = [risky_entry_s1, (trap_corridor, s2_start)]

    meta = LatticeV2Meta(
        segments=[seg1, seg2],
        all_gate_cells=all_gates,
        all_door_positions=all_doors,
        shortest_any=shortest_any,
        shortest_safe=shortest_safe,
        cell_features=features,
        world_weights=ww,
        latent_mode=latent_mode,
    )

    cfg = FamilyConfig(
        max_steps=t_max, risk_budget=1.0,
        prior_risk_mean=0.02, prior_risk_var=0.20,
        search_budget=40, budget_class=10,
    )

    # Structure labels for WBCR / TQ / PRCR metrics
    sc = ScenarioConfig(
        family_name="funnel_trap",
        difficulty=difficulty,
        primary_intervention="WARN",
        cue_reliability=1.0 - ambiguity,
        expected_failure_mode="commitment",
        commitment_cells=commitment_points,
    )
    # Attach extended structure labels
    sc.decision_points = decision_points
    sc.merge_cells = merge_cells
    sc.cue_cells = cue_cells_all
    sc.safe_branches = safe_branch_cells
    sc.trap_branches = trap_branch_cells
    sc.trap_cells = trap_cells_all
    sc.commitment_points = commitment_points
    sc.danger_branch_stage1 = danger_branch
    sc.trap_corridor_stage2 = trap_corridor

    return gm, cfg, meta, sc
# ══════════════════════════════════════════════════════════════════════
# elcb — Equal-Length Competing Branches (diagnostic family)
# ══════════════════════════════════════════════════════════════════════
#
# PURPOSE: Diagnostic, NOT a new difficulty family.
# Tests whether learned predictions can flip branch choice when
# topology confound is completely removed.
#
# Topology:
#              U1—U2—U3—U4
#            /              \
#   S — E —+                +— M — G
#            \              /
#              L1—L2—L3—L4
#
# Invariants (HARD CONSTRAINTS):
#   │πA│ = │πB│  (equal length)
#   passable(i) = 1  ∀ i ∈ πA ∪ πB  (both fully passable)
#   visibility symmetric  (same patch geometry)
#   deadline symmetric  (loose, no time pressure)
#
# Semantic mode (first version): Pure Risk Competition
#   ĉA ≈ ĉB,  r̂A < r̂B
#
# Structure labels (no legacy safe_cells/risky_cells names):
#   branch_a_cells, branch_b_cells
#   oracle_safe_branch_id (0=a, 1=b)
#   oracle_risky_branch_id
#   fork_cell, merge_cell
# ══════════════════════════════════════════════════════════════════════

ELCB_PARAMS = {
    "easy":   {"branch_len": 4, "risk_gap": 0.30, "cue_strength": 0.8,
               "time_ratio": 2.0},
    "medium": {"branch_len": 5, "risk_gap": 0.20, "cue_strength": 0.6,
               "time_ratio": 2.0},
    "hard":   {"branch_len": 6, "risk_gap": 0.12, "cue_strength": 0.4,
               "time_ratio": 2.0},
}


def generate_elcb(
    seed: int,
    difficulty: DifficultyLevel = "medium",
    latent_mode: bool = True,
    **kwargs,
) -> tuple[GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig]:
    """Equal-Length Competing Branches — diagnostic for prediction→planning coupling.

    Topology is neutral: two equal-length, fully passable branches.
    Semantic difference (risk) is the ONLY factor that should determine
    which branch the planner prefers.

    oracle_safe_branch_id: randomized per seed (0=a on row 1, 1=b on row 3)
    """
    rng = np.random.default_rng(seed)
    params = ELCB_PARAMS[difficulty]
    branch_len = params["branch_len"]
    risk_gap = params["risk_gap"]
    cue_strength = params["cue_strength"]
    time_ratio = params["time_ratio"]

    # ── Grid dimensions ──
    # Layout: [wall] [S] [fork_col] [branch×branch_len] [merge_col] [G] [wall]
    W = 1 + 1 + 1 + branch_len + 1 + 1 + 1  # wall+S+fork+branch+merge+G+wall
    H = 7

    ct, cost, risk = _empty_grid(H, W)
    features = np.full((H, W, FEATURE_DIM), 0.5, dtype=np.float64)
    ct[:, :] = CellType.WALL
    cost[:, :] = np.inf

    # ── Shared corridor: row 2 ──
    # S at col 1, fork at col 2, merge at col 2+1+branch_len, G at merge+1
    s_col = 1
    fork_col = 2
    merge_col = fork_col + 1 + branch_len
    g_col = merge_col + 1

    # Open S, fork, merge, G on row 2
    for c in [s_col, fork_col, merge_col, g_col]:
        ct[2, c] = CellType.NORMAL
        cost[2, c] = 1.0
        features[2, c] = np.array([0.5, 1.0, 0.0, 0.0])

    # ── Fork entry: row 2, fork_col connects to rows 1 and 3 ──
    ct[1, fork_col] = CellType.NORMAL
    cost[1, fork_col] = 1.0
    ct[3, fork_col] = CellType.NORMAL
    cost[3, fork_col] = 1.0

    # ── Merge exit: rows 1 and 3 connect back at merge_col ──
    ct[1, merge_col] = CellType.NORMAL
    cost[1, merge_col] = 1.0
    ct[3, merge_col] = CellType.NORMAL
    cost[3, merge_col] = 1.0

    # ── Branch A (row 1) and Branch B (row 3): equal length, both passable ──
    branch_a_cells = []
    branch_b_cells = []

    for i in range(branch_len):
        c = fork_col + 1 + i

        # Tie-breaking noise: break perfect cost equality between branches.
        # A* MOVES tries UP before DOWN → without this, branch A (row 1) is
        # systematically preferred when costs are equal.
        # Noise magnitude 0.005 << smallest risk_gap (0.12 on hard).
        noise_a = rng.uniform(-0.005, 0.005)
        noise_b = rng.uniform(-0.005, 0.005)

        # Branch A (row 1): fully passable
        ct[1, c] = CellType.NORMAL
        cost[1, c] = 1.0 + noise_a
        branch_a_cells.append((1, c))

        # Branch B (row 3): fully passable
        ct[3, c] = CellType.NORMAL
        cost[3, c] = 1.0 + noise_b
        branch_b_cells.append((3, c))

    # ── Semantic assignment: randomize which branch is safe ──
    oracle_safe_branch_id = int(rng.integers(0, 2))  # 0=a is safe, 1=b is safe
    oracle_risky_branch_id = 1 - oracle_safe_branch_id

    safe_cells = branch_a_cells if oracle_safe_branch_id == 0 else branch_b_cells
    risky_cells = branch_a_cells if oracle_risky_branch_id == 0 else branch_b_cells

    # ── Features: Pure Risk Competition ──
    # Safe branch: low texture values (low risk), moderate cost
    # Risky branch: high texture values (high risk), similar cost
    # Base risk level: moderate
    base_risk_level = 0.15
    risky_risk_level = base_risk_level + risk_gap

    for r, c in safe_cells:
        # Safe: low texture → low risk
        t1 = rng.uniform(0.05, 0.15) * cue_strength + (1 - cue_strength) * 0.3
        t2 = rng.uniform(0.05, 0.10) * cue_strength + (1 - cue_strength) * 0.25
        lid = 0.0 if r == 1 else 1.0
        features[r, c] = np.array([lid, 0.0,
                                    np.clip(t1, 0.02, 0.95),
                                    np.clip(t2, 0.02, 0.90)])
        risk[r, c] = base_risk_level + rng.uniform(-0.03, 0.03)

    for r, c in risky_cells:
        # Risky: high texture → high risk
        t1 = rng.uniform(0.45, 0.75) * cue_strength + (1 - cue_strength) * 0.3
        t2 = rng.uniform(0.35, 0.65) * cue_strength + (1 - cue_strength) * 0.25
        lid = 0.0 if r == 1 else 1.0
        features[r, c] = np.array([lid, 0.0,
                                    np.clip(t1, 0.02, 0.95),
                                    np.clip(t2, 0.02, 0.90)])
        risk[r, c] = risky_risk_level + rng.uniform(-0.03, 0.03)

    # Entry/exit gate features: symmetric (lane-neutral)
    for c in [fork_col, merge_col]:
        features[1, c] = np.array([0.0, 1.0, 0.0, 0.0])
        features[3, c] = np.array([1.0, 1.0, 0.0, 0.0])
        risk[1, c] = 0.01
        risk[3, c] = 0.01

    # ── Build gridmap ──
    agent_start = (2, s_col)
    target_pos = (2, g_col)
    gm = _build_gridmap(H, W, ct, cost, risk,
                         agent_start, target_pos, target_pos, [])

    # ── Path lengths (both should be equal) ──
    path_len_a = 1 + 1 + branch_len + 1 + 1  # S→fork + up + branch + down + merge→G
    path_len_b = path_len_a   # identical by construction
    shortest = path_len_a
    t_max = max(int(time_ratio * shortest), shortest + 5)

    # ── Latent mode ──
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
                             agent_start, target_pos, target_pos, [])

    # ── BFS for verification ──
    shortest_any = _bfs_len(gm, agent_start, target_pos, set())
    shortest_safe = _bfs_len(gm, agent_start, target_pos,
                              set(risky_cells))

    # ── Segments ──
    safe_row = 1 if oracle_safe_branch_id == 0 else 3
    risky_row = 3 if oracle_safe_branch_id == 0 else 1

    seg = SegmentMeta(
        index=0,
        col_start=fork_col,
        col_end=merge_col,
        risky_row=risky_row,
        safe_row=safe_row,
        L_risky=branch_len,
        L_safe=branch_len,
        detour_len=0,
        risky_cells=risky_cells,
        safe_cells=safe_cells,
        risky_entry_gate=(risky_row, fork_col),
        safe_entry_gate=(safe_row, fork_col),
        trap_cell=None,
        weak_cue_cells=[],
    )

    meta = LatticeV2Meta(
        segments=[seg],
        all_gate_cells=[(1, fork_col), (3, fork_col),
                        (1, merge_col), (3, merge_col)],
        all_door_positions=[],
        shortest_any=shortest_any,
        shortest_safe=shortest_safe,
        cell_features=features,
        world_weights=ww,
        latent_mode=latent_mode,
    )

    cfg = FamilyConfig(
        max_steps=t_max, risk_budget=1.0,
        prior_risk_mean=0.02, prior_risk_var=0.20,
        search_budget=40, budget_class=10,
    )

    sc = ScenarioConfig(
        family_name="elcb",
        difficulty=difficulty,
        primary_intervention="WARN",
        cue_reliability=cue_strength,
        expected_failure_mode="risk",
    )
    # ELCB-specific structure labels (not inheriting safe_cells/risky_cells naming)
    sc.branch_a_cells = branch_a_cells
    sc.branch_b_cells = branch_b_cells
    sc.oracle_safe_branch_id = oracle_safe_branch_id      # 0=a, 1=b
    sc.oracle_risky_branch_id = oracle_risky_branch_id
    sc.fork_cell = (2, fork_col)
    sc.merge_cell = (2, merge_col)
    sc.safe_cells = safe_cells      # alias for convenience
    sc.risky_cells = risky_cells    # alias for convenience
    sc.safe_row = safe_row
    sc.risky_row = risky_row
    sc.branch_len = branch_len
    sc.risk_gap = risk_gap

    return gm, cfg, meta, sc


# ══════════════════════════════════════════════════════════════════════
# ELCB-PO (Partial Observability) — Tutor-Sensitive Branch Diagnostic
#
# Same topology as ELCB: two equal-length, fully passable branches.
# Key difference: early branch cells have WEAK cues (ambiguous texture)
# and STRONG discriminative cues only appear deeper in the branch.
# At the fork point, branch-aware planner sees near-equal summaries,
# making tutor warning genuinely informative.
# ══════════════════════════════════════════════════════════════════════

ELCB_PO_PARAMS = {
    "easy":   {"branch_len": 8,  "risk_gap": 0.30, "cue_strength": 0.9,
               "reveal_depth": 2, "weak_contrast": 0.02, "time_ratio": 2.5},
    "medium": {"branch_len": 10, "risk_gap": 0.25, "cue_strength": 0.8,
               "reveal_depth": 3, "weak_contrast": 0.02, "time_ratio": 2.5},
    "hard":   {"branch_len": 12, "risk_gap": 0.18, "cue_strength": 0.6,
               "reveal_depth": 4, "weak_contrast": 0.02, "time_ratio": 2.5},
}


def generate_elcb_po(
    seed: int,
    difficulty: DifficultyLevel = "medium",
    latent_mode: bool = True,
    **kwargs,
) -> tuple[GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig]:
    """ELCB-PO: Equal-Length Competing Branches with Partial Observability.

    Same topology as ELCB but with staged cue visibility:
    - Early cells (depth < reveal_depth): WEAK cues — similar texture for
      both safe and risky branches (ambiguous to learner)
    - Late cells (depth >= reveal_depth): STRONG cues — clear texture
      separation (discriminative)

    This ensures branch-aware planner can't trivially distinguish branches
    from fork-point observations, making tutor warning genuinely informative.
    """
    rng = np.random.default_rng(seed)
    params = ELCB_PO_PARAMS[difficulty]
    branch_len = params["branch_len"]
    risk_gap = params["risk_gap"]
    cue_strength = params["cue_strength"]
    reveal_depth = params["reveal_depth"]
    weak_contrast = params["weak_contrast"]
    time_ratio = params["time_ratio"]

    # ── Grid dimensions ──
    W = 1 + 1 + 1 + branch_len + 1 + 1 + 1
    H = 7

    ct, cost, risk = _empty_grid(H, W)
    features = np.full((H, W, FEATURE_DIM), 0.5, dtype=np.float64)
    ct[:, :] = CellType.WALL
    cost[:, :] = np.inf

    # ── Shared corridor: row 2 ──
    s_col = 1
    fork_col = 2
    merge_col = fork_col + 1 + branch_len
    g_col = merge_col + 1

    for c in [s_col, fork_col, merge_col, g_col]:
        ct[2, c] = CellType.NORMAL
        cost[2, c] = 1.0
        features[2, c] = np.array([0.5, 1.0, 0.0, 0.0])

    # Fork/merge gates
    for row in [1, 3]:
        ct[row, fork_col] = CellType.NORMAL
        cost[row, fork_col] = 1.0
        ct[row, merge_col] = CellType.NORMAL
        cost[row, merge_col] = 1.0

    # ── Branches ──
    branch_a_cells = []
    branch_b_cells = []
    for i in range(branch_len):
        c = fork_col + 1 + i
        noise_a = rng.uniform(-0.005, 0.005)
        noise_b = rng.uniform(-0.005, 0.005)
        ct[1, c] = CellType.NORMAL
        cost[1, c] = 1.0 + noise_a
        branch_a_cells.append((1, c))
        ct[3, c] = CellType.NORMAL
        cost[3, c] = 1.0 + noise_b
        branch_b_cells.append((3, c))

    # ── Semantic assignment ──
    oracle_safe_branch_id = int(rng.integers(0, 2))
    oracle_risky_branch_id = 1 - oracle_safe_branch_id
    safe_cells = branch_a_cells if oracle_safe_branch_id == 0 else branch_b_cells
    risky_cells = branch_a_cells if oracle_risky_branch_id == 0 else branch_b_cells

    # ── Staged cue visibility ──
    # Base ambiguous texture: both branches look similar
    base_t1 = 0.25
    base_t2 = 0.20

    base_risk_level = 0.15
    risky_risk_level = base_risk_level + risk_gap

    for idx, (r, c) in enumerate(safe_cells):
        if idx < reveal_depth:
            # WEAK cue: nearly identical texture for safe/risky
            t1 = base_t1 + rng.uniform(-weak_contrast, weak_contrast)
            t2 = base_t2 + rng.uniform(-weak_contrast, weak_contrast)
        else:
            # STRONG cue: clearly low texture → safe
            t1 = rng.uniform(0.05, 0.12) * cue_strength + (1 - cue_strength) * base_t1
            t2 = rng.uniform(0.04, 0.09) * cue_strength + (1 - cue_strength) * base_t2
        lid = 0.0 if r == 1 else 1.0
        features[r, c] = np.array([lid, 0.0,
                                    np.clip(t1, 0.02, 0.95),
                                    np.clip(t2, 0.02, 0.90)])
        if idx < reveal_depth:
            risk[r, c] = base_risk_level + rng.uniform(-0.02, 0.02)
        else:
            risk[r, c] = base_risk_level + rng.uniform(-0.03, 0.03)

    for idx, (r, c) in enumerate(risky_cells):
        if idx < reveal_depth:
            # WEAK cue: nearly identical texture
            t1 = base_t1 + rng.uniform(-weak_contrast, weak_contrast)
            t2 = base_t2 + rng.uniform(-weak_contrast, weak_contrast)
        else:
            # STRONG cue: clearly high texture → risky
            t1 = rng.uniform(0.45, 0.75) * cue_strength + (1 - cue_strength) * base_t1
            t2 = rng.uniform(0.35, 0.65) * cue_strength + (1 - cue_strength) * base_t2
        lid = 0.0 if r == 1 else 1.0
        features[r, c] = np.array([lid, 0.0,
                                    np.clip(t1, 0.02, 0.95),
                                    np.clip(t2, 0.02, 0.90)])
        if idx < reveal_depth:
            risk[r, c] = base_risk_level + rng.uniform(-0.02, 0.02)
        else:
            risk[r, c] = risky_risk_level + rng.uniform(-0.03, 0.03)

    # Entry/exit gate features: symmetric
    for c in [fork_col, merge_col]:
        features[1, c] = np.array([0.0, 1.0, 0.0, 0.0])
        features[3, c] = np.array([1.0, 1.0, 0.0, 0.0])
        risk[1, c] = 0.01
        risk[3, c] = 0.01

    # ── Build gridmap ──
    agent_start = (2, s_col)
    target_pos = (2, g_col)
    gm = _build_gridmap(H, W, ct, cost, risk,
                         agent_start, target_pos, target_pos, [])

    # ── Latent mode ──
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
                             agent_start, target_pos, target_pos, [])

    # ── Paths ──
    shortest_any = _bfs_len(gm, agent_start, target_pos, set())
    shortest_safe = _bfs_len(gm, agent_start, target_pos, set(risky_cells))

    path_len = 1 + 1 + branch_len + 1 + 1
    t_max = max(int(time_ratio * path_len), path_len + 5)

    safe_row = 1 if oracle_safe_branch_id == 0 else 3
    risky_row = 3 if oracle_safe_branch_id == 0 else 1

    seg = SegmentMeta(
        index=0,
        col_start=fork_col, col_end=merge_col,
        risky_row=risky_row, safe_row=safe_row,
        L_risky=branch_len, L_safe=branch_len,
        detour_len=0,
        risky_cells=risky_cells, safe_cells=safe_cells,
        risky_entry_gate=(risky_row, fork_col),
        safe_entry_gate=(safe_row, fork_col),
        trap_cell=None, weak_cue_cells=[],
    )

    meta = LatticeV2Meta(
        segments=[seg],
        all_gate_cells=[(1, fork_col), (3, fork_col),
                        (1, merge_col), (3, merge_col)],
        all_door_positions=[],
        shortest_any=shortest_any,
        shortest_safe=shortest_safe,
        cell_features=features,
        world_weights=ww,
        latent_mode=latent_mode,
    )

    cfg = FamilyConfig(
        max_steps=t_max, risk_budget=1.0,
        prior_risk_mean=0.02, prior_risk_var=0.20,
        search_budget=40, budget_class=10,
    )

    sc = ScenarioConfig(
        family_name="elcb_po",
        difficulty=difficulty,
        primary_intervention="WARN",
        cue_reliability=cue_strength,
        expected_failure_mode="risk",
    )
    sc.branch_a_cells = branch_a_cells
    sc.branch_b_cells = branch_b_cells
    sc.oracle_safe_branch_id = oracle_safe_branch_id
    sc.oracle_risky_branch_id = oracle_risky_branch_id
    sc.fork_cell = (2, fork_col)
    sc.merge_cell = (2, merge_col)
    sc.safe_cells = safe_cells
    sc.risky_cells = risky_cells
    sc.safe_row = safe_row
    sc.risky_row = risky_row
    sc.branch_len = branch_len
    sc.risk_gap = risk_gap
    sc.reveal_depth = reveal_depth
    # Track which cells are weak vs strong for analysis
    sc.weak_cue_indices = list(range(reveal_depth))
    sc.strong_cue_indices = list(range(reveal_depth, branch_len))

    return gm, cfg, meta, sc


# ══════════════════════════════════════════════════════════════════════
# Delayed Commitment Corridor
#
# Key difference from ELCB-PO: commit_depth is parameterized
# independently from reveal_depth, creating 3 regimes:
#   Δ = commit_depth - reveal_depth
#   Δ < 0: must warn (agent commits before seeing strong cues)
#   Δ ≈ 0: boundary (interesting selective zone)
#   Δ > 0: should wait (agent can self-discover)
# ══════════════════════════════════════════════════════════════════════

DELAYED_COMMITMENT_PARAMS = {
    "easy":   {"branch_len": 10, "risk_gap": 0.30, "cue_strength": 0.9,
               "weak_contrast": 0.02, "time_ratio": 2.5},
    "medium": {"branch_len": 10, "risk_gap": 0.25, "cue_strength": 0.8,
               "weak_contrast": 0.02, "time_ratio": 2.5},
    "hard":   {"branch_len": 12, "risk_gap": 0.18, "cue_strength": 0.6,
               "weak_contrast": 0.02, "time_ratio": 2.5},
}


def generate_delayed_corridor(
    seed: int,
    difficulty: DifficultyLevel = "medium",
    latent_mode: bool = True,
    commit_depth: int = 3,
    reveal_depth: int = 3,
    **kwargs,
) -> tuple[GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig]:
    """Delayed Commitment Corridor: parameterized commit vs reveal timing.

    commit_depth: how many cells agent traverses before commitment
                  (after this, switching costs are prohibitive)
    reveal_depth: how many cells are weak cue (strong cues start after)

    Δ = commit_depth - reveal_depth:
      Δ < 0 → must warn (commits before seeing strong cues)
      Δ ≈ 0 → boundary
      Δ > 0 → can self-discover (sees strong cues before committing)
    """
    rng = np.random.default_rng(seed)
    params = DELAYED_COMMITMENT_PARAMS[difficulty]
    branch_len = params["branch_len"]
    risk_gap = params["risk_gap"]
    cue_strength = params["cue_strength"]
    weak_contrast = params["weak_contrast"]
    time_ratio = params["time_ratio"]

    # Override from kwargs if provided
    commit_depth = kwargs.get("commit_depth", commit_depth)
    reveal_depth = kwargs.get("reveal_depth", reveal_depth)

    # Clamp to valid range
    commit_depth = max(1, min(commit_depth, branch_len - 1))
    reveal_depth = max(1, min(reveal_depth, branch_len - 1))

    # ── Grid ──
    W = 1 + 1 + 1 + branch_len + 1 + 1 + 1
    H = 7
    ct, cost, risk = _empty_grid(H, W)
    features = np.full((H, W, FEATURE_DIM), 0.5, dtype=np.float64)
    ct[:, :] = CellType.WALL
    cost[:, :] = np.inf

    s_col = 1
    fork_col = 2
    merge_col = fork_col + 1 + branch_len
    g_col = merge_col + 1

    # Shared corridor
    for c in [s_col, fork_col, merge_col, g_col]:
        ct[2, c] = CellType.NORMAL
        cost[2, c] = 1.0
        features[2, c] = np.array([0.5, 1.0, 0.0, 0.0])

    for row in [1, 3]:
        ct[row, fork_col] = CellType.NORMAL
        cost[row, fork_col] = 1.0
        ct[row, merge_col] = CellType.NORMAL
        cost[row, merge_col] = 1.0

    # Branches
    branch_a_cells = []
    branch_b_cells = []
    for i in range(branch_len):
        c = fork_col + 1 + i
        ct[1, c] = CellType.NORMAL
        cost[1, c] = 1.0 + rng.uniform(-0.005, 0.005)
        branch_a_cells.append((1, c))
        ct[3, c] = CellType.NORMAL
        cost[3, c] = 1.0 + rng.uniform(-0.005, 0.005)
        branch_b_cells.append((3, c))

    # Oracle assignment
    oracle_safe_branch_id = int(rng.integers(0, 2))
    oracle_risky_branch_id = 1 - oracle_safe_branch_id
    safe_cells = branch_a_cells if oracle_safe_branch_id == 0 else branch_b_cells
    risky_cells = branch_a_cells if oracle_risky_branch_id == 0 else branch_b_cells

    # Staged cues (identical to ELCB-PO)
    base_t1, base_t2 = 0.25, 0.20
    base_risk_level = 0.15
    risky_risk_level = base_risk_level + risk_gap

    for idx, (r, c) in enumerate(safe_cells):
        if idx < reveal_depth:
            t1 = base_t1 + rng.uniform(-weak_contrast, weak_contrast)
            t2 = base_t2 + rng.uniform(-weak_contrast, weak_contrast)
        else:
            t1 = rng.uniform(0.05, 0.12) * cue_strength + (1 - cue_strength) * base_t1
            t2 = rng.uniform(0.04, 0.09) * cue_strength + (1 - cue_strength) * base_t2
        lid = 0.0 if r == 1 else 1.0
        features[r, c] = np.array([lid, 0.0, np.clip(t1, 0.02, 0.95), np.clip(t2, 0.02, 0.90)])
        risk[r, c] = base_risk_level + rng.uniform(-0.02, 0.02) if idx < reveal_depth else base_risk_level + rng.uniform(-0.03, 0.03)

    for idx, (r, c) in enumerate(risky_cells):
        if idx < reveal_depth:
            t1 = base_t1 + rng.uniform(-weak_contrast, weak_contrast)
            t2 = base_t2 + rng.uniform(-weak_contrast, weak_contrast)
        else:
            t1 = rng.uniform(0.45, 0.75) * cue_strength + (1 - cue_strength) * base_t1
            t2 = rng.uniform(0.35, 0.65) * cue_strength + (1 - cue_strength) * base_t2
        lid = 0.0 if r == 1 else 1.0
        features[r, c] = np.array([lid, 0.0, np.clip(t1, 0.02, 0.95), np.clip(t2, 0.02, 0.90)])
        risk[r, c] = (base_risk_level + rng.uniform(-0.02, 0.02) if idx < reveal_depth
                      else risky_risk_level + rng.uniform(-0.03, 0.03))

    # Gate features
    for c in [fork_col, merge_col]:
        features[1, c] = np.array([0.0, 1.0, 0.0, 0.0])
        features[3, c] = np.array([1.0, 1.0, 0.0, 0.0])
        risk[1, c] = 0.01
        risk[3, c] = 0.01

    # Build gridmap
    agent_start = (2, s_col)
    target_pos = (2, g_col)
    gm = _build_gridmap(H, W, ct, cost, risk, agent_start, target_pos, target_pos, [])

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
        gm = _build_gridmap(H, W, ct, cost, risk, agent_start, target_pos, target_pos, [])
    else:
        ww = None

    shortest_any = _bfs_len(gm, agent_start, target_pos, set())
    shortest_safe = _bfs_len(gm, agent_start, target_pos, set(risky_cells))
    path_len = 1 + 1 + branch_len + 1 + 1
    t_max = max(int(time_ratio * path_len), path_len + 5)
    safe_row = 1 if oracle_safe_branch_id == 0 else 3
    risky_row = 3 if oracle_safe_branch_id == 0 else 1

    seg = SegmentMeta(
        index=0, col_start=fork_col, col_end=merge_col,
        risky_row=risky_row, safe_row=safe_row,
        L_risky=branch_len, L_safe=branch_len,
        detour_len=0, risky_cells=risky_cells, safe_cells=safe_cells,
        risky_entry_gate=(risky_row, fork_col),
        safe_entry_gate=(safe_row, fork_col),
        trap_cell=None, weak_cue_cells=[],
    )

    meta = LatticeV2Meta(
        segments=[seg],
        all_gate_cells=[(1, fork_col), (3, fork_col), (1, merge_col), (3, merge_col)],
        all_door_positions=[], shortest_any=shortest_any, shortest_safe=shortest_safe,
        cell_features=features, world_weights=ww, latent_mode=latent_mode,
    )

    cfg = FamilyConfig(
        max_steps=t_max, risk_budget=1.0, prior_risk_mean=0.02, prior_risk_var=0.20,
        search_budget=40, budget_class=10,
    )

    sc = ScenarioConfig(
        family_name="delayed_corridor", difficulty=difficulty,
        primary_intervention="WARN", cue_reliability=cue_strength,
        expected_failure_mode="commitment",
    )
    sc.branch_a_cells = branch_a_cells
    sc.branch_b_cells = branch_b_cells
    sc.oracle_safe_branch_id = oracle_safe_branch_id
    sc.oracle_risky_branch_id = oracle_risky_branch_id
    sc.fork_cell = (2, fork_col)
    sc.merge_cell = (2, merge_col)
    sc.safe_cells = safe_cells
    sc.risky_cells = risky_cells
    sc.safe_row = safe_row
    sc.risky_row = risky_row
    sc.branch_len = branch_len
    sc.risk_gap = risk_gap
    sc.reveal_depth = reveal_depth
    sc.commit_depth = commit_depth
    sc.delta_timing = commit_depth - reveal_depth
    sc.weak_cue_indices = list(range(reveal_depth))
    sc.strong_cue_indices = list(range(reveal_depth, branch_len))

    return gm, cfg, meta, sc


# ══════════════════════════════════════════════════════════════════════
# Distractor Cue Corridor
#
# Branch cells carry TWO types of cues:
#   - Diagnostic cues (dims 2,3 = texture): truly correlate with risk
#   - Distractor cues (dim 1 = gate_flag, repurposed): high salience
#     but zero correlation with branch safety
#
# Tests whether tutor uses real information vs salient noise.
# ══════════════════════════════════════════════════════════════════════

DISTRACTOR_CUE_PARAMS = {
    "easy":   {"branch_len": 10, "risk_gap": 0.30, "cue_strength": 0.9,
               "reveal_depth": 3, "distractor_salience": 0.6, "time_ratio": 2.5},
    "medium": {"branch_len": 10, "risk_gap": 0.25, "cue_strength": 0.8,
               "reveal_depth": 3, "distractor_salience": 0.8, "time_ratio": 2.5},
    "hard":   {"branch_len": 12, "risk_gap": 0.18, "cue_strength": 0.6,
               "reveal_depth": 4, "distractor_salience": 0.95, "time_ratio": 2.5},
}


def generate_distractor_cue(
    seed: int,
    difficulty: DifficultyLevel = "medium",
    latent_mode: bool = True,
    **kwargs,
) -> tuple[GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig]:
    """Distractor Cue Corridor: branches have diagnostic + distractor cues.

    Diagnostic cues (texture dims 2,3): genuinely correlate with risk.
    Distractor cues (dim 1): high salience, random w.r.t. safety label.

    Tests whether tutor (and planner) rely on decision-relevant info
    vs being distracted by salient but non-diagnostic features.
    """
    rng = np.random.default_rng(seed)
    params = DISTRACTOR_CUE_PARAMS[difficulty]
    branch_len = params["branch_len"]
    risk_gap = params["risk_gap"]
    cue_strength = params["cue_strength"]
    reveal_depth = params["reveal_depth"]
    distractor_salience = kwargs.get("distractor_salience", params["distractor_salience"])
    time_ratio = params["time_ratio"]

    W = 1 + 1 + 1 + branch_len + 1 + 1 + 1
    H = 7
    ct, cost, risk = _empty_grid(H, W)
    features = np.full((H, W, FEATURE_DIM), 0.5, dtype=np.float64)
    ct[:, :] = CellType.WALL
    cost[:, :] = np.inf

    s_col = 1
    fork_col = 2
    merge_col = fork_col + 1 + branch_len
    g_col = merge_col + 1

    for c in [s_col, fork_col, merge_col, g_col]:
        ct[2, c] = CellType.NORMAL
        cost[2, c] = 1.0
        features[2, c] = np.array([0.5, 0.0, 0.0, 0.0])

    for row in [1, 3]:
        ct[row, fork_col] = CellType.NORMAL
        cost[row, fork_col] = 1.0
        ct[row, merge_col] = CellType.NORMAL
        cost[row, merge_col] = 1.0

    branch_a_cells = []
    branch_b_cells = []
    for i in range(branch_len):
        c = fork_col + 1 + i
        ct[1, c] = CellType.NORMAL
        cost[1, c] = 1.0 + rng.uniform(-0.005, 0.005)
        branch_a_cells.append((1, c))
        ct[3, c] = CellType.NORMAL
        cost[3, c] = 1.0 + rng.uniform(-0.005, 0.005)
        branch_b_cells.append((3, c))

    oracle_safe_branch_id = int(rng.integers(0, 2))
    oracle_risky_branch_id = 1 - oracle_safe_branch_id
    safe_cells = branch_a_cells if oracle_safe_branch_id == 0 else branch_b_cells
    risky_cells = branch_a_cells if oracle_risky_branch_id == 0 else branch_b_cells

    base_t1, base_t2 = 0.25, 0.20
    weak_contrast = 0.02
    base_risk_level = 0.15
    risky_risk_level = base_risk_level + risk_gap

    # Track which cells have distractors
    distractor_cells_safe = []
    distractor_cells_risky = []

    for idx, (r, c) in enumerate(safe_cells):
        if idx < reveal_depth:
            t1 = base_t1 + rng.uniform(-weak_contrast, weak_contrast)
            t2 = base_t2 + rng.uniform(-weak_contrast, weak_contrast)
        else:
            t1 = rng.uniform(0.05, 0.12) * cue_strength + (1 - cue_strength) * base_t1
            t2 = rng.uniform(0.04, 0.09) * cue_strength + (1 - cue_strength) * base_t2
        lid = 0.0 if r == 1 else 1.0
        # Distractor: dim 1 is RANDOM (high salience, no diagnostic value)
        distractor_val = rng.uniform(0.0, 1.0) * distractor_salience
        features[r, c] = np.array([lid, distractor_val,
                                    np.clip(t1, 0.02, 0.95), np.clip(t2, 0.02, 0.90)])
        risk[r, c] = (base_risk_level + rng.uniform(-0.02, 0.02) if idx < reveal_depth
                      else base_risk_level + rng.uniform(-0.03, 0.03))
        if distractor_val > 0.5:
            distractor_cells_safe.append((r, c))

    for idx, (r, c) in enumerate(risky_cells):
        if idx < reveal_depth:
            t1 = base_t1 + rng.uniform(-weak_contrast, weak_contrast)
            t2 = base_t2 + rng.uniform(-weak_contrast, weak_contrast)
        else:
            t1 = rng.uniform(0.45, 0.75) * cue_strength + (1 - cue_strength) * base_t1
            t2 = rng.uniform(0.35, 0.65) * cue_strength + (1 - cue_strength) * base_t2
        lid = 0.0 if r == 1 else 1.0
        distractor_val = rng.uniform(0.0, 1.0) * distractor_salience
        features[r, c] = np.array([lid, distractor_val,
                                    np.clip(t1, 0.02, 0.95), np.clip(t2, 0.02, 0.90)])
        risk[r, c] = (base_risk_level + rng.uniform(-0.02, 0.02) if idx < reveal_depth
                      else risky_risk_level + rng.uniform(-0.03, 0.03))
        if distractor_val > 0.5:
            distractor_cells_risky.append((r, c))

    # Gate features
    for c in [fork_col, merge_col]:
        features[1, c] = np.array([0.0, 0.0, 0.0, 0.0])
        features[3, c] = np.array([1.0, 0.0, 0.0, 0.0])
        risk[1, c] = 0.01
        risk[3, c] = 0.01

    agent_start = (2, s_col)
    target_pos = (2, g_col)
    gm = _build_gridmap(H, W, ct, cost, risk, agent_start, target_pos, target_pos, [])

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
        gm = _build_gridmap(H, W, ct, cost, risk, agent_start, target_pos, target_pos, [])
    else:
        ww = None

    shortest_any = _bfs_len(gm, agent_start, target_pos, set())
    shortest_safe = _bfs_len(gm, agent_start, target_pos, set(risky_cells))
    path_len = 1 + 1 + branch_len + 1 + 1
    t_max = max(int(time_ratio * path_len), path_len + 5)
    safe_row = 1 if oracle_safe_branch_id == 0 else 3
    risky_row = 3 if oracle_safe_branch_id == 0 else 1

    seg = SegmentMeta(
        index=0, col_start=fork_col, col_end=merge_col,
        risky_row=risky_row, safe_row=safe_row,
        L_risky=branch_len, L_safe=branch_len,
        detour_len=0, risky_cells=risky_cells, safe_cells=safe_cells,
        risky_entry_gate=(risky_row, fork_col),
        safe_entry_gate=(safe_row, fork_col),
        trap_cell=None, weak_cue_cells=[],
    )

    meta = LatticeV2Meta(
        segments=[seg],
        all_gate_cells=[(1, fork_col), (3, fork_col), (1, merge_col), (3, merge_col)],
        all_door_positions=[], shortest_any=shortest_any, shortest_safe=shortest_safe,
        cell_features=features, world_weights=ww, latent_mode=latent_mode,
    )

    cfg = FamilyConfig(
        max_steps=t_max, risk_budget=1.0, prior_risk_mean=0.02, prior_risk_var=0.20,
        search_budget=40, budget_class=10,
    )

    sc = ScenarioConfig(
        family_name="distractor_cue", difficulty=difficulty,
        primary_intervention="WARN", cue_reliability=cue_strength,
        expected_failure_mode="cue_error",
    )
    sc.branch_a_cells = branch_a_cells
    sc.branch_b_cells = branch_b_cells
    sc.oracle_safe_branch_id = oracle_safe_branch_id
    sc.oracle_risky_branch_id = oracle_risky_branch_id
    sc.fork_cell = (2, fork_col)
    sc.merge_cell = (2, merge_col)
    sc.safe_cells = safe_cells
    sc.risky_cells = risky_cells
    sc.safe_row = safe_row
    sc.risky_row = risky_row
    sc.branch_len = branch_len
    sc.risk_gap = risk_gap
    sc.reveal_depth = reveal_depth
    sc.commit_depth = reveal_depth  # same as reveal for distractor family
    sc.delta_timing = 0
    sc.distractor_salience = distractor_salience
    sc.distractor_cells_safe = distractor_cells_safe
    sc.distractor_cells_risky = distractor_cells_risky
    sc.weak_cue_indices = list(range(reveal_depth))
    sc.strong_cue_indices = list(range(reveal_depth, branch_len))

    return gm, cfg, meta, sc


# ══════════════════════════════════════════════════════════════════════
# Temptation Branch Corridor
#
# Two branches: one safe, one risky but with temptation lure.
# Agent has a HIDDEN latent preference type θ ∈ {safe, shiny, neutral}
# that biases its internal utility. Robot doesn't know θ, must infer.
#
# Tests:
#   - Robot's ability to infer agent preference from behavior
#   - WAIT value when observing agent choice reveals preference
#   - WARN urgency when temptation is dangerous
# ══════════════════════════════════════════════════════════════════════

TEMPTATION_PARAMS = {
    "easy":   {"branch_len": 10, "risk_gap": 0.30, "cue_strength": 0.9,
               "reveal_depth": 3, "time_ratio": 2.5},
    "medium": {"branch_len": 10, "risk_gap": 0.25, "cue_strength": 0.8,
               "reveal_depth": 3, "time_ratio": 2.5},
    "hard":   {"branch_len": 12, "risk_gap": 0.18, "cue_strength": 0.6,
               "reveal_depth": 4, "time_ratio": 2.5},
}


def generate_temptation_corridor(
    seed: int,
    difficulty: DifficultyLevel = "medium",
    latent_mode: bool = True,
    temptation_strength: float = 0.7,
    **kwargs,
) -> tuple[GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig]:
    """Temptation Branch Corridor: safe branch vs tempting risky branch.

    The risky branch has high-salience lure cues (dim 1) that attract
    certain preference types. Robot doesn't know agent's θ.

    temptation_strength: how salient the lure cue is [0, 1]
    """
    rng = np.random.default_rng(seed)
    params = TEMPTATION_PARAMS[difficulty]
    branch_len = params["branch_len"]
    risk_gap = params["risk_gap"]
    cue_strength = params["cue_strength"]
    reveal_depth = params["reveal_depth"]
    temptation_strength = kwargs.get("temptation_strength", temptation_strength)
    time_ratio = params["time_ratio"]

    W = 1 + 1 + 1 + branch_len + 1 + 1 + 1
    H = 7
    ct, cost, risk = _empty_grid(H, W)
    features = np.full((H, W, FEATURE_DIM), 0.5, dtype=np.float64)
    ct[:, :] = CellType.WALL
    cost[:, :] = np.inf

    s_col = 1
    fork_col = 2
    merge_col = fork_col + 1 + branch_len
    g_col = merge_col + 1

    for c in [s_col, fork_col, merge_col, g_col]:
        ct[2, c] = CellType.NORMAL
        cost[2, c] = 1.0
        features[2, c] = np.array([0.5, 0.0, 0.0, 0.0])

    for row in [1, 3]:
        ct[row, fork_col] = CellType.NORMAL
        cost[row, fork_col] = 1.0
        ct[row, merge_col] = CellType.NORMAL
        cost[row, merge_col] = 1.0

    branch_a_cells = []
    branch_b_cells = []
    for i in range(branch_len):
        c = fork_col + 1 + i
        ct[1, c] = CellType.NORMAL
        cost[1, c] = 1.0 + rng.uniform(-0.005, 0.005)
        branch_a_cells.append((1, c))
        ct[3, c] = CellType.NORMAL
        cost[3, c] = 1.0 + rng.uniform(-0.005, 0.005)
        branch_b_cells.append((3, c))

    oracle_safe_branch_id = int(rng.integers(0, 2))
    oracle_risky_branch_id = 1 - oracle_safe_branch_id
    safe_cells = branch_a_cells if oracle_safe_branch_id == 0 else branch_b_cells
    risky_cells = branch_a_cells if oracle_risky_branch_id == 0 else branch_b_cells

    # Assign latent preference type for this episode
    pref_types = ["safe", "shiny", "neutral"]
    latent_preference = pref_types[int(rng.integers(0, len(pref_types)))]

    base_t1, base_t2 = 0.25, 0.20
    weak_contrast = 0.02
    base_risk_level = 0.15
    risky_risk_level = base_risk_level + risk_gap

    temptation_cells = []

    # Safe branch: diagnostic cues, LOW temptation (dim 1 near 0)
    for idx, (r, c) in enumerate(safe_cells):
        if idx < reveal_depth:
            t1 = base_t1 + rng.uniform(-weak_contrast, weak_contrast)
            t2 = base_t2 + rng.uniform(-weak_contrast, weak_contrast)
        else:
            t1 = rng.uniform(0.05, 0.12) * cue_strength + (1 - cue_strength) * base_t1
            t2 = rng.uniform(0.04, 0.09) * cue_strength + (1 - cue_strength) * base_t2
        lid = 0.0 if r == 1 else 1.0
        tempt = rng.uniform(0.0, 0.15)  # Low temptation on safe branch
        features[r, c] = np.array([lid, tempt,
                                    np.clip(t1, 0.02, 0.95), np.clip(t2, 0.02, 0.90)])
        risk[r, c] = (base_risk_level + rng.uniform(-0.02, 0.02) if idx < reveal_depth
                      else base_risk_level + rng.uniform(-0.03, 0.03))

    # Risky branch: diagnostic cues + HIGH temptation lure
    for idx, (r, c) in enumerate(risky_cells):
        if idx < reveal_depth:
            t1 = base_t1 + rng.uniform(-weak_contrast, weak_contrast)
            t2 = base_t2 + rng.uniform(-weak_contrast, weak_contrast)
        else:
            t1 = rng.uniform(0.45, 0.75) * cue_strength + (1 - cue_strength) * base_t1
            t2 = rng.uniform(0.35, 0.65) * cue_strength + (1 - cue_strength) * base_t2
        lid = 0.0 if r == 1 else 1.0
        # High temptation on risky branch
        tempt = rng.uniform(0.6, 1.0) * temptation_strength
        features[r, c] = np.array([lid, tempt,
                                    np.clip(t1, 0.02, 0.95), np.clip(t2, 0.02, 0.90)])
        risk[r, c] = (base_risk_level + rng.uniform(-0.02, 0.02) if idx < reveal_depth
                      else risky_risk_level + rng.uniform(-0.03, 0.03))
        if tempt > 0.5:
            temptation_cells.append((r, c))

    for c in [fork_col, merge_col]:
        features[1, c] = np.array([0.0, 0.0, 0.0, 0.0])
        features[3, c] = np.array([1.0, 0.0, 0.0, 0.0])
        risk[1, c] = 0.01
        risk[3, c] = 0.01

    agent_start = (2, s_col)
    target_pos = (2, g_col)
    gm = _build_gridmap(H, W, ct, cost, risk, agent_start, target_pos, target_pos, [])

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
        gm = _build_gridmap(H, W, ct, cost, risk, agent_start, target_pos, target_pos, [])
    else:
        ww = None

    shortest_any = _bfs_len(gm, agent_start, target_pos, set())
    shortest_safe = _bfs_len(gm, agent_start, target_pos, set(risky_cells))
    path_len = 1 + 1 + branch_len + 1 + 1
    t_max = max(int(time_ratio * path_len), path_len + 5)
    safe_row = 1 if oracle_safe_branch_id == 0 else 3
    risky_row = 3 if oracle_safe_branch_id == 0 else 1

    seg = SegmentMeta(
        index=0, col_start=fork_col, col_end=merge_col,
        risky_row=risky_row, safe_row=safe_row,
        L_risky=branch_len, L_safe=branch_len,
        detour_len=0, risky_cells=risky_cells, safe_cells=safe_cells,
        risky_entry_gate=(risky_row, fork_col),
        safe_entry_gate=(safe_row, fork_col),
        trap_cell=None, weak_cue_cells=[],
    )

    meta = LatticeV2Meta(
        segments=[seg],
        all_gate_cells=[(1, fork_col), (3, fork_col), (1, merge_col), (3, merge_col)],
        all_door_positions=[], shortest_any=shortest_any, shortest_safe=shortest_safe,
        cell_features=features, world_weights=ww, latent_mode=latent_mode,
    )

    cfg = FamilyConfig(
        max_steps=t_max, risk_budget=1.0, prior_risk_mean=0.02, prior_risk_var=0.20,
        search_budget=40, budget_class=10,
    )

    sc = ScenarioConfig(
        family_name="temptation_corridor", difficulty=difficulty,
        primary_intervention="WARN", cue_reliability=cue_strength,
        expected_failure_mode="cue_error",
    )
    sc.branch_a_cells = branch_a_cells
    sc.branch_b_cells = branch_b_cells
    sc.oracle_safe_branch_id = oracle_safe_branch_id
    sc.oracle_risky_branch_id = oracle_risky_branch_id
    sc.fork_cell = (2, fork_col)
    sc.merge_cell = (2, merge_col)
    sc.safe_cells = safe_cells
    sc.risky_cells = risky_cells
    sc.safe_row = safe_row
    sc.risky_row = risky_row
    sc.branch_len = branch_len
    sc.risk_gap = risk_gap
    sc.reveal_depth = reveal_depth
    sc.commit_depth = reveal_depth
    sc.delta_timing = 0
    sc.temptation_strength = temptation_strength
    sc.temptation_cells = temptation_cells
    sc.latent_preference = latent_preference
    sc.tempt_score_a = 0.1 if oracle_safe_branch_id == 0 else temptation_strength * 0.8
    sc.tempt_score_b = temptation_strength * 0.8 if oracle_safe_branch_id == 0 else 0.1
    sc.weak_cue_indices = list(range(reveal_depth))
    sc.strong_cue_indices = list(range(reveal_depth, branch_len))

    return gm, cfg, meta, sc


# ══════════════════════════════════════════════════════════════════════
# Joint-Latent Conflict Corridor
#
# Goal cue → branch A, Temptation/preference cue → branch B
# Robot must disentangle: "is agent heading to B because of goal or
# because of temptation?" Staggered reveal depths for asynchrony.
# ══════════════════════════════════════════════════════════════════════

CONFLICT_PARAMS = {
    "easy":   {"branch_len": 10, "risk_gap": 0.25, "cue_strength": 0.85,
               "goal_reveal": 2, "pref_reveal": 4, "time_ratio": 2.5},
    "medium": {"branch_len": 10, "risk_gap": 0.20, "cue_strength": 0.75,
               "goal_reveal": 3, "pref_reveal": 5, "time_ratio": 2.5},
    "hard":   {"branch_len": 12, "risk_gap": 0.15, "cue_strength": 0.60,
               "goal_reveal": 4, "pref_reveal": 6, "time_ratio": 2.5},
}


def generate_joint_conflict_corridor(
    seed: int,
    difficulty: DifficultyLevel = "medium",
    latent_mode: bool = True,
    conflict_strength: float = 0.7,
    **kwargs,
) -> tuple[GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig]:
    """Joint-Latent Conflict Corridor.

    Branch A: goal-aligned (high goal_cue on dim 2)
    Branch B: temptation-aligned (high tempt_cue on dim 1)
    Goal cue reveals earlier than pref/tempt cue (staggered).
    """
    rng = np.random.default_rng(seed)
    params = CONFLICT_PARAMS[difficulty]
    branch_len = params["branch_len"]
    risk_gap = params["risk_gap"]
    cue_strength = params["cue_strength"]
    goal_reveal = kwargs.get("goal_reveal_depth", params["goal_reveal"])
    pref_reveal = kwargs.get("pref_reveal_depth", params["pref_reveal"])
    conflict_strength = kwargs.get("conflict_strength", conflict_strength)
    time_ratio = params["time_ratio"]

    W = 1 + 1 + 1 + branch_len + 1 + 1 + 1
    H = 7
    ct, cost, risk = _empty_grid(H, W)
    features = np.full((H, W, FEATURE_DIM), 0.5, dtype=np.float64)
    ct[:, :] = CellType.WALL
    cost[:, :] = np.inf

    s_col = 1
    fork_col = 2
    merge_col = fork_col + 1 + branch_len
    g_col = merge_col + 1

    for c in [s_col, fork_col, merge_col, g_col]:
        ct[2, c] = CellType.NORMAL
        cost[2, c] = 1.0
        features[2, c] = np.array([0.5, 0.0, 0.0, 0.0])

    for row in [1, 3]:
        ct[row, fork_col] = CellType.NORMAL
        cost[row, fork_col] = 1.0
        ct[row, merge_col] = CellType.NORMAL
        cost[row, merge_col] = 1.0

    branch_a_cells = []
    branch_b_cells = []
    for i in range(branch_len):
        c = fork_col + 1 + i
        ct[1, c] = CellType.NORMAL
        cost[1, c] = 1.0 + rng.uniform(-0.005, 0.005)
        branch_a_cells.append((1, c))
        ct[3, c] = CellType.NORMAL
        cost[3, c] = 1.0 + rng.uniform(-0.005, 0.005)
        branch_b_cells.append((3, c))

    oracle_safe_branch_id = int(rng.integers(0, 2))
    oracle_risky_branch_id = 1 - oracle_safe_branch_id
    safe_cells = branch_a_cells if oracle_safe_branch_id == 0 else branch_b_cells
    risky_cells = branch_a_cells if oracle_risky_branch_id == 0 else branch_b_cells

    # Assign latent preference AND latent goal for this episode
    pref_types = ["safe", "shiny", "neutral"]
    goal_types = ["goal_safe_short", "goal_collect", "goal_explore"]
    latent_preference = pref_types[int(rng.integers(0, len(pref_types)))]
    latent_goal = goal_types[int(rng.integers(0, len(goal_types)))]

    base_t1, base_t2 = 0.25, 0.20
    weak_contrast = 0.02
    base_risk_level = 0.15
    risky_risk_level = base_risk_level + risk_gap

    # CONFLICT DESIGN:
    # Branch A (safe): LOW temptation, HIGH goal_cue (dim 2)
    # Branch B (risky): HIGH temptation (dim 1), LOW goal_cue
    # Goal cue appears at goal_reveal, tempt cue at pref_reveal

    temptation_cells = []

    for idx, (r, c) in enumerate(safe_cells):
        lid = 0.0 if r == 1 else 1.0
        # Goal cue: strong on safe branch after goal_reveal
        if idx < goal_reveal:
            goal_cue = rng.uniform(0.2, 0.3)
        else:
            goal_cue = rng.uniform(0.6, 0.85) * cue_strength * conflict_strength
        # Temptation: low on safe branch
        tempt = rng.uniform(0.0, 0.12)
        # Safety cue
        if idx < max(goal_reveal, pref_reveal):
            t1 = base_t1 + rng.uniform(-weak_contrast, weak_contrast)
        else:
            t1 = rng.uniform(0.05, 0.15) * cue_strength + (1 - cue_strength) * base_t1
        features[r, c] = np.array([lid, tempt,
                                    np.clip(goal_cue, 0.02, 0.95),
                                    np.clip(t1, 0.02, 0.90)])
        risk[r, c] = base_risk_level + rng.uniform(-0.02, 0.02)

    for idx, (r, c) in enumerate(risky_cells):
        lid = 0.0 if r == 1 else 1.0
        # Goal cue: weak on risky branch
        goal_cue = rng.uniform(0.1, 0.25)
        # Temptation: strong after pref_reveal
        if idx < pref_reveal:
            tempt = rng.uniform(0.15, 0.30)
        else:
            tempt = rng.uniform(0.6, 0.95) * cue_strength * conflict_strength
        # Safety cue
        if idx < max(goal_reveal, pref_reveal):
            t1 = base_t1 + rng.uniform(-weak_contrast, weak_contrast)
        else:
            t1 = rng.uniform(0.45, 0.75) * cue_strength + (1 - cue_strength) * base_t1
        features[r, c] = np.array([lid, tempt,
                                    np.clip(goal_cue, 0.02, 0.95),
                                    np.clip(t1, 0.02, 0.90)])
        risk[r, c] = (base_risk_level + rng.uniform(-0.02, 0.02) if idx < pref_reveal
                      else risky_risk_level + rng.uniform(-0.03, 0.03))
        if tempt > 0.5:
            temptation_cells.append((r, c))

    for c in [fork_col, merge_col]:
        features[1, c] = np.array([0.0, 0.0, 0.0, 0.0])
        features[3, c] = np.array([1.0, 0.0, 0.0, 0.0])
        risk[1, c] = 0.01
        risk[3, c] = 0.01

    agent_start = (2, s_col)
    target_pos = (2, g_col)
    gm = _build_gridmap(H, W, ct, cost, risk, agent_start, target_pos, target_pos, [])

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
        gm = _build_gridmap(H, W, ct, cost, risk, agent_start, target_pos, target_pos, [])
    else:
        ww = None

    shortest_any = _bfs_len(gm, agent_start, target_pos, set())
    shortest_safe = _bfs_len(gm, agent_start, target_pos, set(risky_cells))
    path_len = 1 + 1 + branch_len + 1 + 1
    t_max = max(int(time_ratio * path_len), path_len + 5)
    safe_row = 1 if oracle_safe_branch_id == 0 else 3
    risky_row = 3 if oracle_safe_branch_id == 0 else 1

    seg = SegmentMeta(
        index=0, col_start=fork_col, col_end=merge_col,
        risky_row=risky_row, safe_row=safe_row,
        L_risky=branch_len, L_safe=branch_len,
        detour_len=0, risky_cells=risky_cells, safe_cells=safe_cells,
        risky_entry_gate=(risky_row, fork_col),
        safe_entry_gate=(safe_row, fork_col),
        trap_cell=None, weak_cue_cells=[],
    )

    meta = LatticeV2Meta(
        segments=[seg],
        all_gate_cells=[(1, fork_col), (3, fork_col), (1, merge_col), (3, merge_col)],
        all_door_positions=[], shortest_any=shortest_any, shortest_safe=shortest_safe,
        cell_features=features, world_weights=ww, latent_mode=latent_mode,
    )

    cfg = FamilyConfig(
        max_steps=t_max, risk_budget=1.0, prior_risk_mean=0.02, prior_risk_var=0.20,
        search_budget=40, budget_class=10,
    )

    sc = ScenarioConfig(
        family_name="joint_conflict_corridor", difficulty=difficulty,
        primary_intervention="WARN", cue_reliability=cue_strength,
        expected_failure_mode="latent_conflict",
    )
    sc.branch_a_cells = branch_a_cells
    sc.branch_b_cells = branch_b_cells
    sc.oracle_safe_branch_id = oracle_safe_branch_id
    sc.oracle_risky_branch_id = oracle_risky_branch_id
    sc.fork_cell = (2, fork_col)
    sc.merge_cell = (2, merge_col)
    sc.safe_cells = safe_cells
    sc.risky_cells = risky_cells
    sc.safe_row = safe_row
    sc.risky_row = risky_row
    sc.branch_len = branch_len
    sc.risk_gap = risk_gap
    sc.reveal_depth = max(goal_reveal, pref_reveal)
    sc.commit_depth = goal_reveal
    sc.delta_timing = goal_reveal - pref_reveal
    sc.goal_reveal_depth = goal_reveal
    sc.pref_reveal_depth = pref_reveal
    sc.conflict_strength = conflict_strength
    sc.temptation_strength = conflict_strength
    sc.temptation_cells = temptation_cells
    sc.latent_preference = latent_preference
    sc.latent_goal = latent_goal
    sc.tempt_score_a = 0.1
    sc.tempt_score_b = conflict_strength * 0.8
    sc.weak_cue_indices = list(range(min(goal_reveal, pref_reveal)))
    sc.strong_cue_indices = list(range(max(goal_reveal, pref_reveal), branch_len))

    return gm, cfg, meta, sc


# ══════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════

SCENARIO_REGISTRY: dict[str, callable] = {
    "baseline_v2": generate_baseline_v2,
    "fork_trap": generate_fork_trap,
    "hazard_belt": generate_hazard_belt,
    "deadline_gate": generate_deadline_gate,
    "delayed_corridor": generate_delayed_corridor,
    "distractor_cue": generate_distractor_cue,
    "funnel_trap": generate_funnel_trap,
    "elcb": generate_elcb,
    "elcb_po": generate_elcb_po,
    "temptation_corridor": generate_temptation_corridor,
    "joint_conflict_corridor": generate_joint_conflict_corridor,
    "deep_tree_mixed_bottleneck_lattice": generate_dtmb_lattice,
    "goal_preference_temptation_entanglement_lattice": generate_gtet_lattice,
}

SCENARIO_NAMES = list(SCENARIO_REGISTRY.keys())

