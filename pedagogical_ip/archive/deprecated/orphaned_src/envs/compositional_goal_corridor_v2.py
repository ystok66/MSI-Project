"""CGC-v2: Factor-vector-driven Compositional Goal Corridor.

Goals are factor vectors y∈{-1,0,+1}^K sampled from a pool,
not discrete labels. Allows true structural composition generalization.

Train pool: 12 factor vectors (covering diverse combinations)
Held-out pool: 6 novel factor vectors (unseen during training)

Per-episode: sample y and θ, generate branch profiles according to y.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..envs.scenario_families import (
    CellType, FEATURE_DIM,
    _empty_grid, _build_gridmap, _bfs_len,
    FamilyConfig, LatticeV2Meta, ScenarioConfig, SegmentMeta,
)

# Factor semantics: [safety, temptation, novelty, shortcut]
K = 4

# ══════════════════════════════════════════
# Goal vector pools
# ══════════════════════════════════════════

TRAIN_POOL = [
    ( 1, -1,  1,  0),   # safe + explore
    ( 1, -1,  0,  1),   # safe + shortcut
    (-1,  1,  0,  0),   # seek tempt, risk safety
    ( 0,  1,  1,  0),   # tempt + explore
    (-1,  0,  0,  1),   # shortcut, risk
    ( 1,  0,  1, -1),   # safe + explore, avoid shortcut
    ( 1,  0,  0,  0),   # safety only
    ( 0,  0,  1,  0),   # explore only
    ( 0,  1,  0,  1),   # tempt + shortcut
    (-1,  1,  1,  0),   # risk + tempt + explore
    ( 1, -1, -1,  0),   # safe, avoid novelty
    ( 0,  0,  0,  1),   # shortcut only
]

HELDOUT_POOL = [
    ( 1,  1,  0,  0),   # contradictory: safe + tempt
    ( 0,  0,  1,  1),   # explore + shortcut
    (-1,  0,  1, -1),   # risk + explore, avoid shortcut
    ( 1, -1,  1,  1),   # safe + explore + shortcut (3-factor)
    (-1,  1, -1,  0),   # risk + tempt, avoid novelty
    ( 1,  0, -1,  1),   # safe + shortcut, avoid novelty
]

CGC2_SUBTYPES = [
    "aligned", "conflict", "boundary_obs", "decoy",
]

SUBTYPE_PARAMS = {
    "aligned":      {"d_c": (4, 6), "d_r": (1, 2), "lure": (0.1, 0.3), "decoy": 0.0},
    "conflict":     {"d_c": (3, 5), "d_r": (1, 3), "lure": (0.6, 1.0), "decoy": 0.0},
    "boundary_obs": {"d_c": (3, 5), "d_r": (3, 5), "lure": (0.3, 0.6), "decoy": 0.0},
    "decoy":        {"d_c": (3, 5), "d_r": (1, 3), "lure": (0.3, 0.6), "decoy": 0.7},
}


@dataclass
class CGC2EpisodeSpec:
    episode_idx: int = 0
    goal_vec: tuple = (1, -1, 1, 0)
    theta_true: str = "safe"
    subtype: str = "aligned"
    mirror: int = 0
    d_commit: int = 4
    d_reveal: int = 2
    lure_strength: float = 0.3
    decoy_strength: float = 0.0
    branch_len: int = 10
    seed: int = 0


@dataclass
class CGC2SessionSpec:
    session_id: int = 0
    theta_true: str = "safe"
    episodes: list = field(default_factory=list)
    use_heldout: bool = False


def generate_cgc2_session(
    session_id: int,
    n_episodes: int = 12,
    theta_true: str = "safe",
    use_heldout: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> CGC2SessionSpec:
    if rng is None:
        rng = np.random.default_rng(session_id)
    pool = HELDOUT_POOL if use_heldout else TRAIN_POOL
    eps = []
    for i in range(n_episodes):
        st = rng.choice(CGC2_SUBTYPES)
        sp = SUBTYPE_PARAMS[st]
        y = pool[rng.integers(0, len(pool))]
        dc = int(rng.integers(sp["d_c"][0], sp["d_c"][1] + 1))
        dr = int(rng.integers(sp["d_r"][0], sp["d_r"][1] + 1))
        lure = float(rng.uniform(sp["lure"][0], sp["lure"][1]))
        decoy = sp["decoy"] * rng.uniform(0.5, 1.0) if sp["decoy"] > 0 else 0.0
        eps.append(CGC2EpisodeSpec(
            episode_idx=i, goal_vec=y, theta_true=theta_true,
            subtype=st, mirror=int(rng.integers(0, 2)),
            d_commit=dc, d_reveal=dr, lure_strength=round(lure, 3),
            decoy_strength=round(decoy, 3), branch_len=10,
            seed=int(rng.integers(0, 100000)),
        ))
    return CGC2SessionSpec(session_id, theta_true, eps, use_heldout)


def _factor_branch_profile(goal_vec, is_aligned, rng):
    p = np.zeros(4)
    for i in range(4):
        if goal_vec[i] == 0:
            p[i] = rng.uniform(0.3, 0.5)
        elif is_aligned:
            p[i] = rng.uniform(0.6, 0.9) if goal_vec[i] > 0 else rng.uniform(0.1, 0.3)
        else:
            p[i] = rng.uniform(0.1, 0.3) if goal_vec[i] > 0 else rng.uniform(0.6, 0.9)
    return p


def generate_cgc2_scenario(ep: CGC2EpisodeSpec):
    rng = np.random.default_rng(ep.seed)
    blen = ep.branch_len
    W = 1 + 1 + 1 + blen + 1 + 1 + 1
    H = 7
    ct, cost, risk = _empty_grid(H, W)
    features = np.full((H, W, FEATURE_DIM), 0.5, dtype=np.float64)
    ct[:, :] = CellType.WALL
    cost[:, :] = np.inf

    s_col, fork_col = 1, 2
    merge_col = fork_col + 1 + blen
    g_col = merge_col + 1

    for c in [s_col, fork_col, merge_col, g_col]:
        ct[2, c] = CellType.NORMAL; cost[2, c] = 1.0
        features[2, c] = np.array([0.5, 0.0, 0.0, 0.0])
    for row in [1, 3]:
        for c in [fork_col, merge_col]:
            ct[row, c] = CellType.NORMAL; cost[row, c] = 1.0
            features[row, c] = np.zeros(4); risk[row, c] = 0.01

    ba, bb = [], []
    for i in range(blen):
        c = fork_col + 1 + i
        ct[1, c] = CellType.NORMAL; cost[1, c] = 1.0 + rng.uniform(-0.005, 0.005)
        ba.append((1, c))
        ct[3, c] = CellType.NORMAL; cost[3, c] = 1.0 + rng.uniform(-0.005, 0.005)
        bb.append((3, c))

    gid = 0 if ep.mirror == 0 else 1
    goal_cells = ba if gid == 0 else bb
    other_cells = bb if gid == 0 else ba

    prof_goal = _factor_branch_profile(ep.goal_vec, True, rng)
    prof_other = _factor_branch_profile(ep.goal_vec, False, rng)

    for idx, (r, c) in enumerate(goal_cells):
        if idx < ep.d_reveal:
            f = np.array([0.5] + [0.45 + rng.uniform(-0.02, 0.02)] * 3)
        else:
            f = np.array([0.5, prof_goal[1], prof_goal[2], prof_goal[3]])
        features[r, c] = np.clip(f, 0.02, 0.95)
        risk[r, c] = 0.05 + rng.uniform(-0.02, 0.02)

    for idx, (r, c) in enumerate(other_cells):
        if idx < ep.d_reveal:
            f = np.array([0.5] + [0.45 + rng.uniform(-0.02, 0.02)] * 3)
        else:
            tv = prof_other[1] * ep.lure_strength
            f = np.array([0.5, tv, prof_other[2], prof_other[3]])
            if ep.decoy_strength > 0:
                irr = [i for i in range(1, 4) if ep.goal_vec[i] == 0]
                if irr:
                    f[rng.choice(irr)] = min(f[rng.choice(irr)] + ep.decoy_strength * 0.4, 0.95)
        features[r, c] = np.clip(f, 0.02, 0.95)
        risk[r, c] = 0.05 + 0.15 * (1 if idx >= ep.d_reveal else 0) + rng.uniform(-0.02, 0.02)

    agent_start = (2, s_col)
    target = (2, g_col)
    gm = _build_gridmap(H, W, ct, cost, risk, agent_start, target, target, [])
    shortest = _bfs_len(gm, agent_start, target, set())

    safe_row = 1 if gid == 0 else 3
    risky_row = 3 if gid == 0 else 1
    seg = SegmentMeta(0, fork_col, merge_col, risky_row, safe_row,
                      blen, blen, 0, other_cells, goal_cells,
                      (risky_row, fork_col), (safe_row, fork_col), None, [])
    meta = LatticeV2Meta([seg],
        [(1, fork_col), (3, fork_col), (1, merge_col), (3, merge_col)],
        [], shortest, shortest, features, None, False)
    cfg = FamilyConfig(max_steps=max(int(2.5 * (2 + blen + 2)), 2 + blen + 7),
                       risk_budget=1.0, prior_risk_mean=0.02, prior_risk_var=0.20,
                       search_budget=40, budget_class=10)

    gt = float(np.mean([features[r, c][1] for r, c in goal_cells[-3:]]))
    ot = float(np.mean([features[r, c][1] for r, c in other_cells[-3:]]))
    sc = ScenarioConfig(family_name="cgc2", difficulty="medium",
                        primary_intervention="WARN", cue_reliability=0.75,
                        expected_failure_mode="factor_goal")
    sc.branch_a_cells = ba; sc.branch_b_cells = bb
    sc.oracle_safe_branch_id = gid; sc.oracle_risky_branch_id = 1 - gid
    sc.fork_cell = (2, fork_col); sc.merge_cell = (2, merge_col)
    sc.safe_cells = goal_cells; sc.risky_cells = other_cells
    sc.safe_row = safe_row; sc.risky_row = risky_row; sc.branch_len = blen
    sc.reveal_depth = ep.d_reveal; sc.commit_depth = ep.d_commit
    sc.temptation_strength = ep.lure_strength
    sc.tempt_score_a = gt if gid == 0 else ot
    sc.tempt_score_b = ot if gid == 0 else gt
    sc.latent_preference = ep.theta_true
    sc.goal_vector = ep.goal_vec
    sc.episode_subtype = ep.subtype
    sc.episode_idx = ep.episode_idx
    sc.decoy_strength = ep.decoy_strength
    return gm, cfg, meta, sc
