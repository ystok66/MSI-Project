"""PP-MRB: Persistent-Profile Mixed-Reveal Branches.

Session-level family generator for persistent tutoring validation.
Produces SessionSpec containing N episodes with parameterized subtypes.

Key design:
  - θ fixed across session; g varies per episode
  - 4 subtypes: wait_clean, wait_lure, boundary_obs, warn_trap
  - Each subtype is a parameterized cluster, not a fixed template
  - Mirror side randomized per episode
  - Compatible with joint posterior / compositional goals via hooks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..envs.scenario_families import (
    generate_scenario, CellType, FEATURE_DIM,
    _empty_grid, _build_gridmap, _bfs_len,
    FamilyConfig, LatticeV2Meta, ScenarioConfig, SegmentMeta,
)


# ══════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════

EPISODE_SUBTYPES = ["wait_clean", "wait_lure", "boundary_obs", "warn_trap"]

# Parameterized subtype clusters (ranges for sampling)
SUBTYPE_PARAMS = {
    "wait_clean": {
        "d_commit": (4, 6), "d_reveal": (1, 2),
        "lure_strength": (0.1, 0.4), "risk_gap": (0.15, 0.25),
    },
    "wait_lure": {
        "d_commit": (4, 6), "d_reveal": (1, 3),
        "lure_strength": (0.6, 1.0), "risk_gap": (0.15, 0.25),
    },
    "boundary_obs": {
        "d_commit": (2, 4), "d_reveal": (2, 4),
        "lure_strength": (0.3, 0.7), "risk_gap": (0.15, 0.25),
    },
    "warn_trap": {
        "d_commit": (1, 3), "d_reveal": (4, 6),
        "lure_strength": (0.8, 1.3), "risk_gap": (0.20, 0.35),
    },
}

GOAL_TYPES_PP = ["goal_safe_long", "goal_collect", "goal_direct"]
PREF_TYPES_PP = ["safe", "shiny", "shortcut", "neutral"]


@dataclass
class EpisodeSpec:
    """Specification for a single episode within a session."""
    episode_idx: int = 0
    goal_type: str = "goal_safe_long"
    episode_subtype: str = "wait_clean"
    mirror_side: int = 0          # 0 = safe on row 1; 1 = safe on row 3
    d_commit: int = 4
    d_reveal: int = 2
    lure_strength: float = 0.3
    risk_gap: float = 0.2
    cue_layout_seed: int = 0
    branch_len: int = 10


@dataclass
class SessionSpec:
    """Specification for a complete tutoring session."""
    session_id: int = 0
    theta_true: str = "safe"
    episodes: list = field(default_factory=list)
    # Future hooks
    latent_goal_vector: Optional[np.ndarray] = None
    latent_preference_vector: Optional[np.ndarray] = None


def generate_session(
    session_id: int,
    n_episodes: int = 10,
    theta_true: str = "safe",
    subtype_mix: Optional[dict] = None,
    branch_len: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> SessionSpec:
    """Generate a parameterized session with mixed episode subtypes.

    Args:
        session_id: Unique session identifier
        n_episodes: Number of episodes in the session
        theta_true: Fixed preference type for this learner
        subtype_mix: Optional dict mapping subtype → probability.
            Default: equal mix of all 4 subtypes.
        branch_len: Length of each branch
        rng: Random generator (seeded from session_id if None)
    """
    if rng is None:
        rng = np.random.default_rng(session_id)

    if subtype_mix is None:
        subtype_mix = {s: 1.0 / len(EPISODE_SUBTYPES) for s in EPISODE_SUBTYPES}

    # Normalize
    total = sum(subtype_mix.values())
    subtypes = list(subtype_mix.keys())
    probs = [subtype_mix[s] / total for s in subtypes]

    episodes = []
    for ep_idx in range(n_episodes):
        # Sample subtype
        subtype = rng.choice(subtypes, p=probs)
        params = SUBTYPE_PARAMS[subtype]

        # Sample parameters from cluster ranges
        d_commit = int(rng.integers(params["d_commit"][0], params["d_commit"][1] + 1))
        d_reveal = int(rng.integers(params["d_reveal"][0], params["d_reveal"][1] + 1))

        # For boundary_obs, enforce |d_commit - d_reveal| <= 1
        if subtype == "boundary_obs":
            d_reveal = d_commit + int(rng.integers(-1, 2))
            d_reveal = max(1, min(d_reveal, branch_len - 1))

        lure_lo, lure_hi = params["lure_strength"]
        lure = float(rng.uniform(lure_lo, lure_hi))

        rg_lo, rg_hi = params["risk_gap"]
        risk_gap = float(rng.uniform(rg_lo, rg_hi))

        goal = rng.choice(GOAL_TYPES_PP)
        mirror = int(rng.integers(0, 2))
        seed = int(rng.integers(0, 100000))

        episodes.append(EpisodeSpec(
            episode_idx=ep_idx,
            goal_type=goal,
            episode_subtype=subtype,
            mirror_side=mirror,
            d_commit=d_commit,
            d_reveal=d_reveal,
            lure_strength=round(lure, 3),
            risk_gap=round(risk_gap, 3),
            cue_layout_seed=seed,
            branch_len=branch_len,
        ))

    return SessionSpec(
        session_id=session_id,
        theta_true=theta_true,
        episodes=episodes,
    )


def generate_episode_scenario(
    ep: EpisodeSpec,
    theta_true: str = "safe",
) -> tuple:
    """Generate a concrete scenario from an EpisodeSpec.

    Returns (gm, cfg, meta, sc) compatible with existing framework.
    """
    rng = np.random.default_rng(ep.cue_layout_seed)
    branch_len = ep.branch_len
    d_commit = ep.d_commit
    d_reveal = ep.d_reveal
    lure = ep.lure_strength
    risk_gap = ep.risk_gap

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

    # Spine
    for c in [s_col, fork_col, merge_col, g_col]:
        ct[2, c] = CellType.NORMAL
        cost[2, c] = 1.0
        features[2, c] = np.array([0.5, 0.0, 0.0, 0.0])

    # Fork/merge gates
    for row in [1, 3]:
        ct[row, fork_col] = CellType.NORMAL
        cost[row, fork_col] = 1.0
        ct[row, merge_col] = CellType.NORMAL
        cost[row, merge_col] = 1.0

    # Branches
    branch_a_cells, branch_b_cells = [], []
    for i in range(branch_len):
        c = fork_col + 1 + i
        ct[1, c] = CellType.NORMAL
        cost[1, c] = 1.0 + rng.uniform(-0.005, 0.005)
        branch_a_cells.append((1, c))
        ct[3, c] = CellType.NORMAL
        cost[3, c] = 1.0 + rng.uniform(-0.005, 0.005)
        branch_b_cells.append((3, c))

    # Mirror: which branch is safe
    if ep.mirror_side == 0:
        oracle_safe_branch_id = 0
    else:
        oracle_safe_branch_id = 1
    oracle_risky_branch_id = 1 - oracle_safe_branch_id
    safe_cells = branch_a_cells if oracle_safe_branch_id == 0 else branch_b_cells
    risky_cells = branch_a_cells if oracle_risky_branch_id == 0 else branch_b_cells

    # Feature layout: 4-segment structure
    # [pre_junction | pre_commit | commit_zone | post_reveal]
    base_risk = 0.12
    weak_contrast = 0.02

    for idx, (r, c) in enumerate(safe_cells):
        lid = 0.5  # neutralized
        tempt = rng.uniform(0.0, 0.1)
        if idx < d_reveal:
            # Weak / ambiguous cue
            safety_cue = 0.45 + rng.uniform(-weak_contrast, weak_contrast)
            diag_cue = 0.45 + rng.uniform(-weak_contrast, weak_contrast)
        else:
            # Strong safe cue
            safety_cue = rng.uniform(0.1, 0.2)
            diag_cue = rng.uniform(0.1, 0.2)
        features[r, c] = np.array([lid, tempt,
                                    np.clip(diag_cue, 0.02, 0.95),
                                    np.clip(safety_cue, 0.02, 0.95)])
        risk[r, c] = base_risk + rng.uniform(-0.02, 0.02)

    tempt_cells = []
    for idx, (r, c) in enumerate(risky_cells):
        lid = 0.5  # neutralized
        if idx < d_reveal:
            tempt = rng.uniform(0.1, 0.25)
            safety_cue = 0.45 + rng.uniform(-weak_contrast, weak_contrast)
            diag_cue = 0.45 + rng.uniform(-weak_contrast, weak_contrast)
        else:
            tempt = rng.uniform(0.5, 0.9) * lure
            safety_cue = rng.uniform(0.55, 0.80)
            diag_cue = rng.uniform(0.55, 0.80)
        features[r, c] = np.array([lid, np.clip(tempt, 0.02, 0.95),
                                    np.clip(diag_cue, 0.02, 0.95),
                                    np.clip(safety_cue, 0.02, 0.95)])
        risk[r, c] = (base_risk + rng.uniform(-0.02, 0.02) if idx < d_reveal
                      else base_risk + risk_gap + rng.uniform(-0.03, 0.03))
        if tempt > 0.4:
            tempt_cells.append((r, c))

    # Gate features (neutral)
    for c in [fork_col, merge_col]:
        features[1, c] = np.array([0.0, 0.0, 0.0, 0.0])
        features[3, c] = np.array([1.0, 0.0, 0.0, 0.0])
        risk[1, c] = 0.01
        risk[3, c] = 0.01

    agent_start = (2, s_col)
    target_pos = (2, g_col)
    gm = _build_gridmap(H, W, ct, cost, risk, agent_start, target_pos, target_pos, [])

    shortest_any = _bfs_len(gm, agent_start, target_pos, set())
    shortest_safe = _bfs_len(gm, agent_start, target_pos, set(risky_cells))
    path_len = 1 + 1 + branch_len + 1 + 1
    t_max = max(int(2.5 * path_len), path_len + 5)
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
        cell_features=features, world_weights=None, latent_mode=False,
    )

    cfg = FamilyConfig(
        max_steps=t_max, risk_budget=1.0, prior_risk_mean=0.02, prior_risk_var=0.20,
        search_budget=40, budget_class=10,
    )

    sc = ScenarioConfig(
        family_name="pp_mrb", difficulty="medium",
        primary_intervention="WARN", cue_reliability=0.75,
        expected_failure_mode="persistent_profile",
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
    sc.reveal_depth = d_reveal
    sc.commit_depth = d_commit
    sc.delta_timing = d_commit - d_reveal
    sc.temptation_strength = lure
    sc.temptation_cells = tempt_cells
    sc.latent_preference = theta_true
    sc.latent_goal = ep.goal_type
    sc.tempt_score_a = 0.1 if oracle_safe_branch_id == 0 else lure * 0.7
    sc.tempt_score_b = lure * 0.7 if oracle_safe_branch_id == 0 else 0.1
    sc.episode_subtype = ep.episode_subtype
    sc.episode_idx = ep.episode_idx
    sc.mirror_side = ep.mirror_side

    return gm, cfg, meta, sc
