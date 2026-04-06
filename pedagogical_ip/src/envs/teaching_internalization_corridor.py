"""Teaching-Internalization Corridor (TIC).

3-phase session:
  Phase A: tutoring block (8 episodes, tutor active)
  Phase B: no-tutor same-structure (4 episodes)
  Phase C: no-tutor shifted-structure (4 episodes)

Subtypes:
  temptation_repeat — same lure pattern, different surface
  self_discovery_teach — WAIT should yield self-discovery + internal κ update
  warn_rescue — must warn now, short-term rescue
  boundary_obs — ambiguous, near equal branches
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

TIC_SUBTYPES = [
    "temptation_repeat",
    "self_discovery_teach",
    "warn_rescue",
    "boundary_obs",
    # P3-A: Balanced active coverage subtypes
    "soft_gradual",
    "blind_corridor",
]

SUBTYPE_PARAMS = {
    "temptation_repeat":     {"d_c": (3, 5), "d_r": (1, 3), "lure": (0.7, 1.0), "risk": (0.3, 0.5)},
    "self_discovery_teach":  {"d_c": (5, 7), "d_r": (1, 2), "lure": (0.4, 0.7), "risk": (0.2, 0.4)},
    "warn_rescue":           {"d_c": (2, 3), "d_r": (3, 5), "lure": (0.6, 0.9), "risk": (0.4, 0.6)},
    "boundary_obs":          {"d_c": (3, 5), "d_r": (3, 5), "lure": (0.3, 0.5), "risk": (0.15, 0.3)},
    # P3-A: Balanced active coverage
    "soft_gradual":          {"d_c": (3, 4), "d_r": (2, 4), "lure": (0.5, 0.8), "risk": (0.35, 0.5)},
    "blind_corridor":        {"d_c": (2, 3), "d_r": (4, 6), "lure": (0.5, 0.8), "risk": (0.4, 0.55)},
}


@dataclass
class TICEpisodeSpec:
    episode_idx: int = 0
    theta_true: str = "safe"
    subtype: str = "temptation_repeat"
    phase: str = "A"       # A=tutor, B=no-tutor-same, C=no-tutor-shift
    mirror: int = 0
    d_commit: int = 4
    d_reveal: int = 2
    lure_strength: float = 0.7
    risk_level: float = 0.35
    branch_len: int = 10
    cue_layout_seed: int = 0


@dataclass
class TICSessionSpec:
    session_id: int = 0
    theta_true: str = "safe"
    episodes: list = field(default_factory=list)


def generate_tic_session(
    session_id: int,
    theta_true: str = "safe",
    n_tutor: int = 8,
    n_same: int = 4,
    n_shift: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> TICSessionSpec:
    if rng is None:
        rng = np.random.default_rng(session_id)

    eps = []
    idx = 0

    # Phase A: tutor active — mix of subtypes with emphasis on temptation_repeat
    weights_a = [0.35, 0.25, 0.25, 0.15]
    for _ in range(n_tutor):
        st = rng.choice(TIC_SUBTYPES, p=weights_a)
        sp = SUBTYPE_PARAMS[st]
        eps.append(TICEpisodeSpec(
            episode_idx=idx, theta_true=theta_true, subtype=st, phase="A",
            mirror=int(rng.integers(0, 2)),
            d_commit=int(rng.integers(sp["d_c"][0], sp["d_c"][1] + 1)),
            d_reveal=int(rng.integers(sp["d_r"][0], sp["d_r"][1] + 1)),
            lure_strength=round(float(rng.uniform(sp["lure"][0], sp["lure"][1])), 3),
            risk_level=round(float(rng.uniform(sp["risk"][0], sp["risk"][1])), 3),
            cue_layout_seed=int(rng.integers(0, 100000)),
        ))
        idx += 1

    # Phase B: no-tutor same structure (same subtypes)
    for _ in range(n_same):
        st = rng.choice(TIC_SUBTYPES, p=weights_a)
        sp = SUBTYPE_PARAMS[st]
        eps.append(TICEpisodeSpec(
            episode_idx=idx, theta_true=theta_true, subtype=st, phase="B",
            mirror=int(rng.integers(0, 2)),
            d_commit=int(rng.integers(sp["d_c"][0], sp["d_c"][1] + 1)),
            d_reveal=int(rng.integers(sp["d_r"][0], sp["d_r"][1] + 1)),
            lure_strength=round(float(rng.uniform(sp["lure"][0], sp["lure"][1])), 3),
            risk_level=round(float(rng.uniform(sp["risk"][0], sp["risk"][1])), 3),
            cue_layout_seed=int(rng.integers(0, 100000)),
        ))
        idx += 1

    # Phase C: no-tutor shifted (higher lure, different risk pattern)
    for _ in range(n_shift):
        st = rng.choice(["temptation_repeat", "self_discovery_teach"])
        sp = SUBTYPE_PARAMS[st]
        eps.append(TICEpisodeSpec(
            episode_idx=idx, theta_true=theta_true, subtype=st, phase="C",
            mirror=int(rng.integers(0, 2)),
            d_commit=int(rng.integers(sp["d_c"][0], sp["d_c"][1] + 1)),
            d_reveal=int(rng.integers(sp["d_r"][0], sp["d_r"][1] + 1)),
            lure_strength=round(float(rng.uniform(sp["lure"][0] + 0.1, min(sp["lure"][1] + 0.15, 1.0))), 3),
            risk_level=round(float(rng.uniform(sp["risk"][0] + 0.05, sp["risk"][1] + 0.1)), 3),
            cue_layout_seed=int(rng.integers(0, 100000)),
        ))
        idx += 1

    return TICSessionSpec(session_id, theta_true, eps)


def generate_tic_scenario(ep: TICEpisodeSpec):
    """Generate a grid scenario for one TIC episode."""
    rng = np.random.default_rng(ep.cue_layout_seed)
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
        ct[1, c] = CellType.NORMAL; cost[1, c] = 1.0
        ba.append((1, c))
        ct[3, c] = CellType.NORMAL; cost[3, c] = 1.0
        bb.append((3, c))

    safe_id = 0 if ep.mirror == 0 else 1
    safe_cells = ba if safe_id == 0 else bb
    risky_cells = bb if safe_id == 0 else ba

    for idx, (r, c) in enumerate(safe_cells):
        safety_f = max(0.6 + rng.uniform(-0.05, 0.05), 0.55)
        tempt_f = min(0.15 + rng.uniform(-0.05, 0.05), 0.25)
        features[r, c] = np.array([safety_f, tempt_f, rng.uniform(0.3, 0.5), 0.0])
        risk[r, c] = 0.05 + rng.uniform(-0.02, 0.02)

    for idx, (r, c) in enumerate(risky_cells):
        if idx < ep.d_reveal:
            f = np.array([0.5 + rng.uniform(-0.03, 0.03)] * 4)
        else:
            tempt_v = ep.lure_strength * (0.8 + rng.uniform(-0.1, 0.1))
            safety_v = max(0.2 - ep.risk_level * 0.3 + rng.uniform(-0.05, 0.05), 0.05)
            features[r, c] = np.array([safety_v, tempt_v, rng.uniform(0.4, 0.6), rng.uniform(0.0, 0.3)])
        risk[r, c] = ep.risk_level + rng.uniform(-0.05, 0.05)

    agent_start = (2, s_col)
    target = (2, g_col)
    gm = _build_gridmap(H, W, ct, cost, risk, agent_start, target, target, [])
    shortest = _bfs_len(gm, agent_start, target, set())

    safe_row = 1 if safe_id == 0 else 3
    risky_row = 3 if safe_id == 0 else 1
    seg = SegmentMeta(0, fork_col, merge_col, risky_row, safe_row,
                      blen, blen, 0, risky_cells, safe_cells,
                      (risky_row, fork_col), (safe_row, fork_col), None, [])
    meta = LatticeV2Meta([seg],
        [(1, fork_col), (3, fork_col), (1, merge_col), (3, merge_col)],
        [], shortest, shortest, features, None, False)
    cfg = FamilyConfig(max_steps=max(int(2.5 * (2 + blen + 2)), 2 + blen + 7),
                       risk_budget=1.0, prior_risk_mean=0.02, prior_risk_var=0.20,
                       search_budget=40, budget_class=10)

    st = float(np.mean([features[r, c][1] for r, c in safe_cells[-3:]]))
    rt = float(np.mean([features[r, c][1] for r, c in risky_cells[-3:]]))
    sc = ScenarioConfig(family_name="tic", difficulty="medium",
                        primary_intervention="WARN", cue_reliability=0.75,
                        expected_failure_mode="temptation")
    sc.branch_a_cells = ba; sc.branch_b_cells = bb
    sc.oracle_safe_branch_id = safe_id; sc.oracle_risky_branch_id = 1 - safe_id
    sc.fork_cell = (2, fork_col); sc.merge_cell = (2, merge_col)
    sc.safe_cells = safe_cells; sc.risky_cells = risky_cells
    sc.safe_row = safe_row; sc.risky_row = risky_row; sc.branch_len = blen
    sc.reveal_depth = ep.d_reveal; sc.commit_depth = ep.d_commit
    sc.temptation_strength = ep.lure_strength
    sc.tempt_score_a = st if safe_id == 0 else rt
    sc.tempt_score_b = rt if safe_id == 0 else st
    sc.latent_preference = ep.theta_true
    sc.episode_subtype = ep.subtype
    sc.episode_phase = ep.phase
    sc.episode_idx = ep.episode_idx
    sc.risk_level = ep.risk_level
    return gm, cfg, meta, sc
