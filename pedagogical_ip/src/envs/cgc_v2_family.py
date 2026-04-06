"""CGC-v2: Compositional Goal Corridor — Stage 2 family.

Goals are compositional: g = (g_obj, g_constraint).
  Atomic:   collect_red, avoid_blue, use_safe, reach_fast
  Composed: collect_red ∧ avoid_blue, collect_red ∧ use_safe

Episode subtypes:
  goal_aligned:  g and θ naturally agree (e.g., safe θ + safe goal)
  goal_conflict: g and θ disagree (e.g., shiny θ + safe goal)
  goal_boundary: ambiguous — needs observation

Multi-step diagnostic window:
  Step 1: preference-driven (lure-sensitive move at fork)
  Step 2: goal-driven (correction/commitment after partial info)
  Step 3: constraint-sensitive (respect/ignore constraint)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..envs.scenario_families import (
    CellType, FEATURE_DIM, _empty_grid, _build_gridmap, _bfs_len,
    FamilyConfig, LatticeV2Meta, ScenarioConfig, SegmentMeta,
)


# ══════════════════════════════════════════════════════════════
# Goal structure
# ══════════════════════════════════════════════════════════════

ATOMIC_GOALS = ["collect_red", "avoid_blue", "use_safe", "reach_fast"]

COMPOSITE_GOALS = [
    ("collect_red", "avoid_blue"),
    ("collect_red", "use_safe"),
    ("avoid_blue", "use_safe"),
    ("reach_fast", "avoid_blue"),
]

# Goal properties: how each atomic goal attaches to branch attributes
#   [safety_weight, temptation_weight, novelty_weight, speed_weight]
GOAL_WEIGHTS = {
    "collect_red":  np.array([0.0,  2.5, 0.5, 0.0]),
    "avoid_blue":   np.array([2.0, -1.0, 0.0, 0.0]),
    "use_safe":     np.array([3.0, -0.5, 0.0, 0.0]),
    "reach_fast":   np.array([0.0,  0.0, 0.0, 3.0]),
}

CGC_EPISODE_SUBTYPES = ["goal_aligned", "goal_conflict", "goal_boundary"]

CGC_PREF_TYPES = ["safe", "shiny", "shortcut", "neutral"]


# ══════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════

@dataclass
class CGCEpisodeSpec:
    """Specification for one CGC-v2 episode."""
    episode_idx: int = 0
    goal_obj: str = "collect_red"
    goal_constraint: Optional[str] = None  # None = atomic goal
    episode_subtype: str = "goal_aligned"
    mirror_side: int = 0
    d_commit: int = 4
    d_reveal: int = 2
    lure_strength: float = 0.5
    risk_gap: float = 0.2
    cue_layout_seed: int = 0
    branch_len: int = 10
    diagnostic_steps: int = 2  # how many observable steps before commit

    @property
    def goal_label(self) -> str:
        if self.goal_constraint:
            return f"{self.goal_obj}+{self.goal_constraint}"
        return self.goal_obj

    @property
    def is_composite(self) -> bool:
        return self.goal_constraint is not None

    @property
    def delta(self) -> int:
        return self.d_commit - self.d_reveal


@dataclass
class CGCSessionSpec:
    """Session spec for CGC-v2."""
    session_id: int = 0
    theta_true: str = "safe"
    goal_true: str = "collect_red"
    goal_constraint: Optional[str] = None
    episodes: list = field(default_factory=list)


def _is_aligned(goal_obj, goal_constraint, theta):
    """Check if goal and preference naturally agree."""
    safe_goals = {"avoid_blue", "use_safe"}
    risky_goals = {"collect_red", "reach_fast"}
    if theta in ("safe",) and goal_obj in safe_goals:
        return True
    if theta in ("shiny",) and goal_obj in risky_goals:
        return True
    if goal_constraint in safe_goals and theta in ("safe",):
        return True
    return False


def generate_cgc_session(
    session_id: int,
    n_episodes: int = 30,
    theta_true: str = "safe",
    goal_obj: str = "collect_red",
    goal_constraint: Optional[str] = None,
    subtype_mix: Optional[dict] = None,
    branch_len: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> CGCSessionSpec:
    """Generate a CGC-v2 session with parameterized episodes."""
    if rng is None:
        rng = np.random.default_rng(session_id)

    aligned = _is_aligned(goal_obj, goal_constraint, theta_true)

    if subtype_mix is None:
        if aligned:
            subtype_mix = {"goal_aligned": 0.5, "goal_conflict": 0.2, "goal_boundary": 0.3}
        else:
            subtype_mix = {"goal_aligned": 0.2, "goal_conflict": 0.5, "goal_boundary": 0.3}

    total = sum(subtype_mix.values())
    subtypes = list(subtype_mix.keys())
    probs = [subtype_mix[s] / total for s in subtypes]

    episodes = []
    for ei in range(n_episodes):
        subtype = rng.choice(subtypes, p=probs)

        # Δ parameterization by subtype
        if subtype == "goal_aligned":
            dc = int(rng.integers(4, 7))
            dr = int(rng.integers(1, 3))
            lure = float(rng.uniform(0.1, 0.4))
        elif subtype == "goal_conflict":
            dc = int(rng.integers(1, 3))
            dr = int(rng.integers(4, 7))
            lure = float(rng.uniform(0.7, 1.2))
        else:  # boundary
            dc = int(rng.integers(2, 5))
            dr = int(rng.integers(2, 5))
            lure = float(rng.uniform(0.3, 0.7))

        diag_steps = min(dc, int(rng.integers(2, 4)))

        episodes.append(CGCEpisodeSpec(
            episode_idx=ei,
            goal_obj=goal_obj,
            goal_constraint=goal_constraint,
            episode_subtype=subtype,
            mirror_side=int(rng.integers(0, 2)),
            d_commit=dc, d_reveal=dr,
            lure_strength=round(lure, 3),
            risk_gap=round(float(rng.uniform(0.15, 0.30)), 3),
            cue_layout_seed=int(rng.integers(0, 100000)),
            branch_len=branch_len,
            diagnostic_steps=diag_steps,
        ))

    return CGCSessionSpec(
        session_id=session_id, theta_true=theta_true,
        goal_true=goal_obj, goal_constraint=goal_constraint,
        episodes=episodes,
    )


def generate_cgc_episode_scenario(
    ep: CGCEpisodeSpec,
    theta_true: str = "safe",
) -> tuple:
    """Generate concrete scenario from a CGCEpisodeSpec.

    Returns (gm, cfg, meta, sc) compatible with existing framework.
    """
    rng = np.random.default_rng(ep.cue_layout_seed)
    blen = ep.branch_len
    dc = ep.d_commit
    dr = ep.d_reveal
    lure = ep.lure_strength
    rg = ep.risk_gap

    W = 1 + 1 + 1 + blen + 1 + 1 + 1
    H = 7
    ct, cost, risk = _empty_grid(H, W)
    features = np.full((H, W, FEATURE_DIM), 0.5, dtype=np.float64)
    ct[:, :] = CellType.WALL
    cost[:, :] = np.inf

    s_col = 1; fork_col = 2
    merge_col = fork_col + 1 + blen; g_col = merge_col + 1

    for c in [s_col, fork_col, merge_col, g_col]:
        ct[2, c] = CellType.NORMAL; cost[2, c] = 1.0
        features[2, c] = np.array([0.5, 0.0, 0.0, 0.0])
    for row in [1, 3]:
        ct[row, fork_col] = CellType.NORMAL; cost[row, fork_col] = 1.0
        ct[row, merge_col] = CellType.NORMAL; cost[row, merge_col] = 1.0

    ba_cells, bb_cells = [], []
    for i in range(blen):
        c = fork_col + 1 + i
        ct[1, c] = CellType.NORMAL; cost[1, c] = 1.0 + rng.uniform(-0.005, 0.005)
        ba_cells.append((1, c))
        ct[3, c] = CellType.NORMAL; cost[3, c] = 1.0 + rng.uniform(-0.005, 0.005)
        bb_cells.append((3, c))

    if ep.mirror_side == 0:
        osb = 0
    else:
        osb = 1
    orb = 1 - osb
    safe_cells = ba_cells if osb == 0 else bb_cells
    risky_cells = ba_cells if orb == 0 else bb_cells

    # Feature layout with goal-related cues
    base_risk = 0.12
    wc = 0.02
    goal_w = GOAL_WEIGHTS.get(ep.goal_obj, np.array([1.0, 0.0, 0.0, 0.0]))

    for idx, (r, c) in enumerate(safe_cells):
        lid = 0.5
        tempt = rng.uniform(0.0, 0.1)
        if idx < dr:
            safety_cue = 0.45 + rng.uniform(-wc, wc)
            diag_cue = 0.45 + rng.uniform(-wc, wc)
        else:
            safety_cue = rng.uniform(0.1, 0.2)
            diag_cue = rng.uniform(0.1, 0.2)
        features[r, c] = np.array([lid, tempt,
                                    np.clip(diag_cue, 0.02, 0.95),
                                    np.clip(safety_cue, 0.02, 0.95)])
        risk[r, c] = base_risk + rng.uniform(-0.02, 0.02)

    tempt_cells = []
    for idx, (r, c) in enumerate(risky_cells):
        lid = 0.5
        if idx < dr:
            tempt = rng.uniform(0.1, 0.25)
            safety_cue = 0.45 + rng.uniform(-wc, wc)
            diag_cue = 0.45 + rng.uniform(-wc, wc)
        else:
            tempt = rng.uniform(0.5, 0.9) * lure
            safety_cue = rng.uniform(0.55, 0.80)
            diag_cue = rng.uniform(0.55, 0.80)
        features[r, c] = np.array([lid, np.clip(tempt, 0.02, 0.95),
                                    np.clip(diag_cue, 0.02, 0.95),
                                    np.clip(safety_cue, 0.02, 0.95)])
        risk[r, c] = (base_risk + rng.uniform(-0.02, 0.02) if idx < dr
                      else base_risk + rg + rng.uniform(-0.03, 0.03))
        if tempt > 0.4:
            tempt_cells.append((r, c))

    for c in [fork_col, merge_col]:
        features[1, c] = np.array([0.0, 0.0, 0.0, 0.0])
        features[3, c] = np.array([1.0, 0.0, 0.0, 0.0])
        risk[1, c] = 0.01; risk[3, c] = 0.01

    agent_start = (2, s_col); target_pos = (2, g_col)
    gm = _build_gridmap(H, W, ct, cost, risk, agent_start, target_pos, target_pos, [])
    shortest_any = _bfs_len(gm, agent_start, target_pos, set())
    shortest_safe = _bfs_len(gm, agent_start, target_pos, set(risky_cells))
    path_len = 1 + 1 + blen + 1 + 1
    t_max = max(int(2.5 * path_len), path_len + 5)
    safe_row = 1 if osb == 0 else 3
    risky_row = 3 if osb == 0 else 1

    seg = SegmentMeta(
        index=0, col_start=fork_col, col_end=merge_col,
        risky_row=risky_row, safe_row=safe_row,
        L_risky=blen, L_safe=blen, detour_len=0,
        risky_cells=risky_cells, safe_cells=safe_cells,
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
        family_name="cgc_v2", difficulty="medium",
        primary_intervention="WARN", cue_reliability=0.75,
        expected_failure_mode="joint_conflict",
    )
    sc.branch_a_cells = ba_cells; sc.branch_b_cells = bb_cells
    sc.oracle_safe_branch_id = osb; sc.oracle_risky_branch_id = orb
    sc.fork_cell = (2, fork_col); sc.merge_cell = (2, merge_col)
    sc.safe_cells = safe_cells; sc.risky_cells = risky_cells
    sc.safe_row = safe_row; sc.risky_row = risky_row
    sc.branch_len = blen; sc.risk_gap = rg
    sc.reveal_depth = dr; sc.commit_depth = dc
    sc.delta_timing = dc - dr
    sc.temptation_strength = lure; sc.temptation_cells = tempt_cells
    sc.latent_preference = theta_true
    sc.latent_goal = ep.goal_label
    sc.tempt_score_a = 0.1 if osb == 0 else lure * 0.7
    sc.tempt_score_b = lure * 0.7 if osb == 0 else 0.1
    sc.episode_subtype = ep.episode_subtype
    sc.episode_idx = ep.episode_idx
    sc.mirror_side = ep.mirror_side
    sc.diagnostic_steps = ep.diagnostic_steps
    sc.goal_obj = ep.goal_obj
    sc.goal_constraint = ep.goal_constraint

    return gm, cfg, meta, sc
