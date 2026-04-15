"""DTMB-L Specific Tutor Helpers.

Implements family-aware warning dispatch and oracle logic for the 
Deep Tree Mixed-Bottleneck Lattice (DTMB-L) scenario.

Design principles (per user spec):
  - Scenario side: map topology, cues, doors, belts — deterministic given seed+cfg
  - Tutor/dispatch side: warn_variant, oracle_variant, scoring — independent of generation
  - All variant selection is through DTMBDispatchConfig, never through generator user_cfg
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np


# ══════════════════════════════════════════════════════════════════════
# Dispatch Configuration
# ══════════════════════════════════════════════════════════════════════

WarnVariant = Literal["W1", "W2", "W3"]
OracleVariant = Literal["O1", "O2"]


@dataclass
class DTMBDispatchConfig:
    """Tutor-side dispatch configuration for DTMB-L.

    This is NOT a scenario config — it controls how the tutor SCORES
    intervention targets, not how the map is generated.
    """
    warn_variant: WarnVariant = "W1"
    oracle_variant: OracleVariant = "O1"


# ══════════════════════════════════════════════════════════════════════
# Branch discovery (shared by all WARN variants)
# ══════════════════════════════════════════════════════════════════════

def _discover_branches(s) -> tuple[list[int], int, list[bool]]:
    """Discover Stage 1 candidate branch rows and their door status.

    Returns:
        candidate_rows: list of row indices for Stage 1 branches
        fork_col: column of the root fork
        has_door_per_row: parallel list — True if that branch leads to a door
    """
    meta = s.meta

    if not meta.all_gate_cells:
        return [], 0, []

    fork_row, fork_col = meta.all_gate_cells[0]

    # Agent already past commit → too late for Stage 1 warning
    if s.agent_pos[1] > fork_col + 3:
        return [], fork_col, []

    # Passable cells immediately right of fork = branch starts
    candidate_rows = []
    for r in range(s.passable.shape[0]):
        if s.passable[r, fork_col + 1]:
            candidate_rows.append(r)

    if not candidate_rows:
        return [], fork_col, []

    # Map each door to its closest candidate branch
    has_door = [False] * len(candidate_rows)
    for door_pos in meta.all_door_positions:
        if door_pos[1] > fork_col:
            closest_idx = min(
                range(len(candidate_rows)),
                key=lambda i: abs(candidate_rows[i] - door_pos[0]),
            )
            has_door[closest_idx] = True

    return candidate_rows, fork_col, has_door


# ══════════════════════════════════════════════════════════════════════
# WARN target scoring — 3 variants
# ══════════════════════════════════════════════════════════════════════

def _score_w1(r: int, fork_col: int, has_door: bool,
              agent_pos: tuple[int, int], meta) -> float:
    """W1 (current): GT risk + distance + door suppression."""
    score = 0.0
    if not has_door:
        score += 10.0
    dist = abs(agent_pos[0] - r) + abs(agent_pos[1] - (fork_col + 1))
    score -= dist * 1.5
    if has_door:
        score -= 100.0
    return score


def _score_w2(r: int, fork_col: int, has_door: bool,
              agent_pos: tuple[int, int], meta) -> float:
    """W2 (risk-only): Only door presence matters. No distance weighting."""
    if has_door:
        return -100.0
    return 10.0


def _score_w3(r: int, fork_col: int, has_door: bool,
              agent_pos: tuple[int, int], meta) -> float:
    """W3 (risk+commit): W2 + bonus if agent hasn't committed past fork."""
    score = _score_w2(r, fork_col, has_door, agent_pos, meta)
    if not has_door:
        # Extra bonus if agent is still before the commit column
        commit_cols = [cp[1] for cp in meta.commitment_points_by_stage[0]]
        if commit_cols and agent_pos[1] < min(commit_cols):
            score += 5.0
    return score


_SCORE_FNS = {"W1": _score_w1, "W2": _score_w2, "W3": _score_w3}


def compute_dtmb_warn_target(
    s,
    variant: WarnVariant = "W1",
) -> tuple[int, int] | None:
    """Compute the target cell (row, col) to apply a WARNING penalty.

    Scoring is variant-dependent (W1/W2/W3) but branch discovery and
    door-mapping are shared across all variants.

    Returns (row, col) to penalize, or None if no valid target.
    """
    candidate_rows, fork_col, has_door = _discover_branches(s)
    if not candidate_rows:
        return None

    score_fn = _SCORE_FNS[variant]
    best_score = -999.0
    best_target = None

    for i, r in enumerate(candidate_rows):
        sc = score_fn(r, fork_col, has_door[i], s.agent_pos, s.meta)
        if sc > best_score:
            best_score = sc
            best_target = (r, fork_col + 1)

    return best_target


def apply_dtmb_warning(
    s,
    variant: WarnVariant = "W1",
) -> None:
    """Apply belief-side penalty to the warned branch."""
    target = compute_dtmb_warn_target(s, variant=variant)
    if target:
        r, c = target
        if getattr(s, "warned_cell_extra", None) is None:
            s.warned_cell_extra = {}
        s.warned_cell_extra[(r, c)] = s.warned_cell_extra.get((r, c), 0.0) + 50.0
        s.warn_count = getattr(s, "warn_count", 0) + 1


# ══════════════════════════════════════════════════════════════════════
# Oracle — stage-aware GT intervention
# ══════════════════════════════════════════════════════════════════════

def _detect_stage(s) -> int:
    """Detect which DTMB stage the agent is in (1, 2, or 3).

    Uses band boundaries stored in meta rather than hardcoded columns.
    Falls back to column heuristics if band info unavailable.
    """
    meta = s.meta
    c = s.agent_pos[1]

    # Try to use commitment_points_by_stage to infer boundaries
    if hasattr(meta, "commitment_points_by_stage") and meta.commitment_points_by_stage:
        # Stage 1 → 2 boundary: max commit col of Stage 1
        s1_commits = meta.commitment_points_by_stage[0]
        if s1_commits:
            s1_end = max(cp[1] for cp in s1_commits)
            if c <= s1_end + 2:
                return 1

        # Stage 2 → 3 boundary: max commit col of Stage 2
        s2_commits = meta.commitment_points_by_stage[1]
        if s2_commits:
            s2_end = max(cp[1] for cp in s2_commits)
            if c <= s2_end + 2:
                return 2

    # Belt cells mark Stage 3
    if hasattr(meta, "belt_cells_by_stage") and meta.belt_cells_by_stage:
        belt_cells = meta.belt_cells_by_stage[2] if len(meta.belt_cells_by_stage) > 2 else []
        if belt_cells:
            belt_start = min(bc[1] for bc in belt_cells)
            if c >= belt_start - 2:
                return 3

    # Fallback: use door positions as Stage 2 marker
    if meta.all_door_positions:
        door_cols = [dp[1] for dp in meta.all_door_positions]
        min_door, max_door = min(door_cols), max(door_cols)
        if c < min_door - 3:
            return 1
        elif c <= max_door + 3:
            return 2

    return 3


def compute_dtmb_oracle_stage_schedule(s) -> list[tuple[int, str, str]]:
    """Return the GT stage schedule: [(stage_id, dominant_bottleneck, action)].

    Reads from meta.dominant_bottleneck_gt_by_stage if available.
    """
    meta = s.meta
    schedule = []

    if hasattr(meta, "dominant_bottleneck_gt_by_stage") and meta.dominant_bottleneck_gt_by_stage:
        bottlenecks = meta.dominant_bottleneck_gt_by_stage
    else:
        bottlenecks = ["epistemic", "structural", "outcome"]

    action_map = {
        "epistemic": "WARN",
        "structural": "UNLOCK",
        "outcome": "ITEM_DROP",
        "none": "WAIT",
    }

    for i, bn in enumerate(bottlenecks):
        schedule.append((i + 1, bn, action_map.get(bn, "WAIT")))

    return schedule


def _oracle_o1(s, stage: int) -> str:
    """O1 (simple): GT bottleneck → threshold-based action.

    Stage 1: WARN once if agent hasn't been warned yet
    Stage 2: UNLOCK if locked door nearby
    Stage 3: ITEM_DROP if belt ahead and no shield
    """
    meta = s.meta

    if stage == 1:
        # Only warn once per episode
        if getattr(s, '_dtmb_oracle_warned', False):
            return "WAIT"
        # Check if there's a valid warn target and agent is close to commit
        target = compute_dtmb_warn_target(s, variant="W1")
        if target is None:
            return "WAIT"

        # p_fail proxy: distance to commit vs distance to reveal
        s1_commits = (meta.commitment_points_by_stage[0]
                      if hasattr(meta, "commitment_points_by_stage")
                      and meta.commitment_points_by_stage else [])
        s1_reveals = (meta.reveal_events_by_stage[0]
                      if hasattr(meta, "reveal_events_by_stage")
                      and meta.reveal_events_by_stage else [])

        if s1_commits and s1_reveals:
            d_commit = min(abs(s.agent_pos[1] - cp[1]) for cp in s1_commits)
            d_reveal = min(abs(s.agent_pos[1] - re[1]) for re in s1_reveals)
            # p_fail = sigmoid((d_commit - d_reveal) / tau_f)
            tau_f = 2.0
            p_fail = 1.0 / (1.0 + np.exp(-(d_commit - d_reveal) / tau_f))
            if p_fail > 0.4:
                return "WARN"
        else:
            # No commit/reveal info — warn if target exists and agent is nearby
            if target and abs(s.agent_pos[0] - target[0]) <= 2:
                return "WARN"

        return "WAIT"

    elif stage == 2:
        # UNLOCK if locked door is nearby and on the agent's path
        for door_pos in meta.all_door_positions:
            dr, dc = door_pos
            if not s.passable[dr, dc]:
                if abs(s.agent_pos[0] - dr) <= 3 and 0 <= dc - s.agent_pos[1] <= 6:
                    return "UNLOCK"
        return "WAIT"

    elif stage == 3:
        # ITEM_DROP if belt ahead
        already_shielded = False
        if hasattr(s, 'inventory') and s.inventory is not None:
            already_shielded = s.inventory.has_shield()
        if getattr(s, '_dtmb_oracle_item_dropped', False):
            already_shielded = True
        if not already_shielded:
            belt_cells = set()
            if hasattr(meta, 'belt_cells_by_stage') and len(meta.belt_cells_by_stage) > 2:
                belt_cells = set(meta.belt_cells_by_stage[2])
            for look_c in range(s.agent_pos[1], min(s.agent_pos[1] + 6, s.passable.shape[1])):
                if (s.agent_pos[0], look_c) in belt_cells:
                    return "ITEM_DROP"
        return "WAIT"

    return "WAIT"


def _oracle_o2(s, stage: int) -> str:
    """O2 (route-aware): O1 + branch targeting for WARN.

    In Stage 1, specifically targets the branch the agent is heading toward.
    Stages 2-3 identical to O1.
    """
    if stage == 1:
        target = compute_dtmb_warn_target(s, variant="W1")
        if target is None:
            return "WAIT"
        # O2 additionally checks if agent is actually heading toward the bad branch
        agent_r = s.agent_pos[0]
        target_r = target[0]
        if abs(agent_r - target_r) <= 1:
            return "WARN"
        return "WAIT"
    else:
        return _oracle_o1(s, stage)


def compute_dtmb_oracle_action(
    s,
    variant: OracleVariant = "O1",
) -> str:
    """Compute the GT-optimal action for the current stage."""
    stage = _detect_stage(s)
    if variant == "O1":
        return _oracle_o1(s, stage)
    else:
        return _oracle_o2(s, stage)


def apply_dtmb_oracle_action(
    s,
    dispatch_cfg: DTMBDispatchConfig | None = None,
) -> None:
    """Execute the oracle action. Records intervention for metrics."""
    cfg = dispatch_cfg or DTMBDispatchConfig()
    action = compute_dtmb_oracle_action(s, variant=cfg.oracle_variant)

    if action == "WARN":
        apply_dtmb_warning(s, variant=cfg.warn_variant)
        s._dtmb_oracle_warned = True

    elif action == "UNLOCK":
        for door_pos in s.meta.all_door_positions:
            r, c = door_pos
            if not s.passable[r, c]:
                if abs(s.agent_pos[0] - r) <= 3 and 0 <= c - s.agent_pos[1] <= 6:
                    s.passable[r, c] = True
                    s.belief_cost[r, c] = 1.0
                    s.unlock_count = getattr(s, "unlock_count", 0) + 1
                    break

    elif action == "ITEM_DROP":
        if s.inventory is not None:
            s.inventory.add_shield()
        else:
            # SEMANTIC FENCE: ITEM_DROP without inventory is a config error.
            # NEVER modify gridmap.true_risk — that violates the affordance-only
            # intervention semantics required by the proposal.
            import warnings
            warnings.warn(
                "DTMB oracle ITEM_DROP called without inventory system. "
                "Skipping — cannot provide shield without InventoryState.",
                RuntimeWarning, stacklevel=2,
            )
        s._dtmb_oracle_item_dropped = True

    # Record intervention for metrics
    from dataclasses import dataclass as _dc

    @_dc
    class _OracleIntervention:
        action: str
        gain: float = 1.0

    s.last_intervention = _OracleIntervention(action=action, gain=1.0)
