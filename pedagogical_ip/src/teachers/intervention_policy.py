"""
Intervention Policy — Phase 7+8+10.

Scores WAIT / WARN / UNLOCK / ITEM_DROP based on counterfactual surrogate rollouts.
All scores come from the SAME prediction mechanism (AgentPredictor),
not from separate ad-hoc heuristics.

Phase 8: adds ITEM_DROP scoring via same counterfactual framework.
Phase 10: adds bottleneck diagnosis + perceptual access + redundancy penalty.
Phase 1B: adds boredom/frustration penalty to Q_WAIT.

CRITICAL: All diagnostics are prefix-based and read-only.
CRITICAL: Planner-side and execution-time shield semantics must match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .robot_belief import RobotBelief
from .agent_predictor import (
    predict_agent_prefix,
    predict_agent_prefix_after_warn,
    predict_agent_prefix_after_unlock,
    predict_agent_prefix_after_item_drop,
    estimate_learning_gain,
    AgentPrediction,
)
from .interventions import (
    InterventionType, MAIN_INTERVENTION_FAMILY,
    ItemEffect, ItemType, InventoryState,
    SHIELD_DEFAULT_RISK_REDUCTION,
)
from ..agents.belief_planning import BeliefPlan, FailureModeEstimate
from .perceptual_model import PerceptualAccessState, compute_redundancy
from .bottleneck_diagnosis import (
    BottleneckScores, diagnose_bottleneck, match_intervention_to_bottleneck,
)


VALID_ACTIONS = frozenset({"WAIT", "WARN", "UNLOCK", "ITEM_DROP"})


@dataclass
class InterventionDecision:
    """Structured decision from the robot's intervention policy."""
    action: str                         # "WAIT" | "WARN" | "UNLOCK" | "ITEM_DROP"
    scores: dict                        # action → final score (after all bonuses/penalties)
    reason: str                         # dominant reason
    decision_margin: float              # gap between best and 2nd best
    # Predictions backing the decision
    predicted_prefix: list[tuple[int, int]]
    predicted_failure_modes: FailureModeEstimate
    # Counterfactual predictions
    counterfactual_scores: dict         # action → (predicted_risk, predicted_cost)
    # Item effect (Phase 8)
    expected_item_effect: Optional[ItemEffect] = None
    # Phase 10: bottleneck diagnosis
    bottleneck: Optional[BottleneckScores] = None
    # Phase 10+: Score decomposition for diagnostics (Step A logging)
    score_decomposition: Optional[dict] = None


@dataclass
class InterventionConfig:
    """Weights for intervention scoring."""
    catastrophe_weight: float = 5.0
    learning_gain_weight: float = 1.0
    warn_effect_weight: float = 3.0
    unlock_effect_weight: float = 3.0
    item_drop_weight: float = 3.0
    autonomy_penalty: float = 1.0
    deadline_weight: float = 2.0
    item_drop_cost: float = 1.5         # higher than autonomy: items are expensive
    item_drop_enabled: bool = True
    # Phase 10: TPM weights
    bottleneck_match_weight: float = 2.0   # β_b: bottleneck-intervention matching
    redundancy_penalty_weight: float = 1.5  # β_R: redundancy suppression
    # Phase 10: Ablation toggles (all True = full TPM)
    use_bottleneck_matching: bool = True
    use_warn_damping: bool = True
    use_unlock_memory: bool = True
    use_perceptual_access: bool = True
    # Phase 1B: boredom/frustration penalty
    boredom_weight: float = 0.3  # β_bore: canonical (Phase 1B verified)


def score_interventions(
    rb: RobotBelief,
    agent_pos: tuple[int, int],
    goal: tuple[int, int],
    belief_cost: np.ndarray,
    passable: np.ndarray,
    meta,
    warned_cell_extra: Optional[dict] = None,
    warned_segments: Optional[set] = None,
    prefix_horizon: int = 5,
    t: int = 0,
    t_max: int = 100,
    config: Optional[InterventionConfig] = None,
    inventory_state: Optional[InventoryState] = None,
    allowed_actions: Optional[frozenset] = None,
    # Phase 10: Tutor Perceptual Model
    perceptual_access: Optional[PerceptualAccessState] = None,
) -> InterventionDecision:
    """Score WAIT / WARN / UNLOCK / ITEM_DROP using counterfactual surrogate rollouts.

    Phase 10 additions:
    - Bottleneck diagnosis: epistemic / structural / outcome
    - Bottleneck-intervention matching bonus
    - Redundancy penalty for WARN (suppresses wasted warnings)
    - Perceptual access state for decision-relevant unseen estimation
    """
    cfg = config or InterventionConfig()
    warned_segs = warned_segments or set()
    inv = inventory_state or InventoryState()

    # 1. WAIT rollout (baseline)
    pred_wait = predict_agent_prefix(
        rb, agent_pos, goal, belief_cost, passable,
        warned_cell_extra=warned_cell_extra,
        prefix_horizon=prefix_horizon, t=t, t_max=t_max,
        inventory_state=inv,
    )
    wait_risk = pred_wait.predicted_plan.expected_risk
    wait_unc = pred_wait.predicted_plan.uncertainty
    learning_gain = estimate_learning_gain(
        rb, pred_wait.predicted_plan.planned_prefix)
    wait_fm = pred_wait.predicted_failure_modes

    wait_score = (
        cfg.learning_gain_weight * learning_gain
        - cfg.catastrophe_weight * wait_risk
        - cfg.deadline_weight * wait_fm.deadline_miss
    )

    # Phase 1B: boredom penalty
    #   B_wait = avg_prefix_cost / (ε + LG)
    #   Penalizes WAIT when learning gain is near zero but cost accumulates
    prefix_len_wait = len(pred_wait.predicted_plan.planned_prefix)
    wait_expected_cost = pred_wait.predicted_plan.expected_cost
    avg_prefix_cost = wait_expected_cost / max(1, prefix_len_wait)
    boredom_penalty = avg_prefix_cost / (1e-6 + max(0.0, learning_gain))
    wait_score_pre_boredom = wait_score  # snapshot for diagnostics
    wait_score -= cfg.boredom_weight * boredom_penalty

    # 2. WARN rollout (counterfactual)
    warn_extra = _build_warn_extra(meta, warned_segs, agent_pos)
    pred_warn = predict_agent_prefix_after_warn(
        rb, agent_pos, goal, belief_cost, passable,
        warned_cell_extra=warned_cell_extra,
        warn_extra_cost=warn_extra,
        prefix_horizon=prefix_horizon, t=t, t_max=t_max,
    )
    warn_risk = pred_warn.predicted_plan.expected_risk
    catastrophe_reduction = max(0.0, wait_risk - warn_risk)

    warn_score = (
        cfg.warn_effect_weight * catastrophe_reduction
        - cfg.autonomy_penalty
    )

    # 3. UNLOCK rollout (counterfactual)
    unlock_cells = _find_unlockable_cells(meta, passable, agent_pos)
    if unlock_cells:
        pred_unlock = predict_agent_prefix_after_unlock(
            rb, agent_pos, goal, belief_cost, passable,
            unlock_cells=unlock_cells,
            warned_cell_extra=warned_cell_extra,
            prefix_horizon=prefix_horizon, t=t, t_max=t_max,
        )
        unlock_risk = pred_unlock.predicted_plan.expected_risk
        unlock_path_len = len(pred_unlock.predicted_plan.full_path)
        wait_path_len = len(pred_wait.predicted_plan.full_path)
        topology_improvement = max(0, wait_path_len - unlock_path_len)
        unlock_cat_reduction = max(0.0, wait_risk - unlock_risk)

        unlock_score = (
            cfg.unlock_effect_weight * (unlock_cat_reduction + topology_improvement * 0.1)
            - cfg.autonomy_penalty
        )
    else:
        unlock_score = -cfg.autonomy_penalty * 2  # no unlock available
        unlock_risk = wait_risk

    # 4. ITEM_DROP rollout (counterfactual) — Phase 8
    item_effect = None
    if cfg.item_drop_enabled and not inv.has_shield():
        pred_item = predict_agent_prefix_after_item_drop(
            rb, agent_pos, goal, belief_cost, passable,
            inventory_state=inv,
            warned_cell_extra=warned_cell_extra,
            prefix_horizon=prefix_horizon, t=t, t_max=t_max,
        )
        item_risk = pred_item.predicted_plan.expected_risk
        item_cat_reduction = max(0.0, wait_risk - item_risk)

        item_score = (
            cfg.item_drop_weight * item_cat_reduction
            - cfg.item_drop_cost
        )
        item_effect = ItemEffect(
            item_type=ItemType.SHIELD,
            risk_reduction=inv.shield_risk_reduction,
            location="current_cell",
        )
    else:
        item_score = -cfg.item_drop_cost * 2  # not available or already has shield
        item_risk = wait_risk

    # ═══ Phase 10: Bottleneck diagnosis + matching ═══════════════

    # Compute risk uncertainty map for bottleneck diagnosis
    # Uses predictor's external interface — no internal weight access
    risk_unc_map = None
    if rb._predictor_snapshot is not None:
        pred = rb._predictor_snapshot
        H, W = rb.agent_belief_var.shape[:2]
        risk_unc_map = np.zeros((H, W), dtype=np.float64)
        for r in range(H):
            for c in range(W):
                x_var = rb.agent_belief_var[r, c]
                risk_unc_map[r, c] = pred.predict_risk_uncertainty_from_var(x_var)

    # Structural slack estimation
    from ..agents.route_necessity import compute_route_necessity
    slack = max(0, (t_max - t) - _estimate_shortest_path_len(
        agent_pos, goal, passable))

    has_locked = len(unlock_cells) > 0 if unlock_cells else False

    # Estimate unavoidable risk on best feasible path
    prefix_len = len(pred_wait.predicted_plan.planned_prefix)
    min_path_risk = wait_risk  # proxy: risk of best wait path

    bn = diagnose_bottleneck(
        agent_pos=agent_pos, goal=goal, passable=passable,
        t=t, t_max=t_max, pa=perceptual_access,
        q_wait=wait_score, q_warn=warn_score,
        q_unlock=unlock_score, q_item=item_score,
        prefix_cells=pred_wait.predicted_plan.planned_prefix,
        risk_uncertainty_map=risk_unc_map,
        has_locked_doors=has_locked,
        slack_steps=slack,
        min_path_risk=min_path_risk,
    )

    # ── Score decomposition tracking (Step A diagnostics) ──────────
    item_available = cfg.item_drop_enabled and not inv.has_shield()
    raw_q = {"WAIT": wait_score, "WARN": warn_score,
             "UNLOCK": unlock_score, "ITEM_DROP": item_score}
    action_available = {
        "WAIT": True,
        "WARN": True,
        "UNLOCK": bool(unlock_cells),
        "ITEM_DROP": item_available,
    }
    match_bonus = {"WAIT": 0.0, "WARN": 0.0, "UNLOCK": 0.0, "ITEM_DROP": 0.0}
    redundancy_val = 0.0
    warn_damping_val = 0.0
    warn_repeat_penalty = 0.0
    unlock_memory_penalty = 0.0

    # Add bottleneck-intervention matching bonus
    if cfg.use_bottleneck_matching:
        match_bonus["WARN"] = cfg.bottleneck_match_weight * match_intervention_to_bottleneck("WARN", bn)
        if unlock_cells:
            match_bonus["UNLOCK"] = cfg.bottleneck_match_weight * match_intervention_to_bottleneck("UNLOCK", bn)
        if item_available:
            match_bonus["ITEM_DROP"] = cfg.bottleneck_match_weight * match_intervention_to_bottleneck("ITEM_DROP", bn)
        scores = {
            "WAIT": wait_score,
            "WARN": warn_score + match_bonus["WARN"],
            "UNLOCK": unlock_score + match_bonus["UNLOCK"],
            "ITEM_DROP": item_score + match_bonus["ITEM_DROP"],
        }
    else:
        scores = {
            "WAIT": wait_score,
            "WARN": warn_score,
            "UNLOCK": unlock_score,
            "ITEM_DROP": item_score,
        }

    # Redundancy penalty for WARN (Phase 10)
    pa_active = perceptual_access if cfg.use_perceptual_access else None
    if pa_active is not None:
        warn_cells = []
        for seg in meta.segments:
            if seg.index not in warned_segs and seg.col_start > agent_pos[1]:
                warn_cells.extend(seg.risky_cells)
        if warn_cells:
            redundancy_val = compute_redundancy(
                pa_active, warn_cells,
                risk_uncertainty=risk_unc_map)
            scores["WARN"] -= cfg.redundancy_penalty_weight * redundancy_val

    # Outcome-dominant WARN damping (Phase 10)
    if cfg.use_warn_damping and pa_active is not None:
        if item_available and bn.dominant == "outcome" and bn.outcome > bn.epistemic:
            outcome_dominance = bn.outcome / max(bn.epistemic + 0.01, 0.01)
            warn_damping_val = min(2.0, outcome_dominance * 0.5)
            scores["WARN"] -= warn_damping_val

        # Escalate redundancy if WARN already issued multiple times
        n_warns = pa_active.intervention_memory.get("warn_count", 0)
        if n_warns > 0:
            warn_repeat_penalty = n_warns * 0.5
            scores["WARN"] -= warn_repeat_penalty

    # Suppress repeated UNLOCK (check intervention memory)
    if cfg.use_unlock_memory and pa_active is not None:
        n_unlocks = pa_active.intervention_memory.get("unlock_count", 0)
        if n_unlocks > 0 and not has_locked:
            unlock_memory_penalty = cfg.autonomy_penalty * 2
            scores["UNLOCK"] -= unlock_memory_penalty

    # Build decomposition dict
    _decomp = {
        "raw_q": raw_q,
        "match_bonus": match_bonus,
        "redundancy_penalty": redundancy_val,
        "warn_damping": warn_damping_val,
        "warn_repeat_penalty": warn_repeat_penalty,
        "unlock_memory_penalty": unlock_memory_penalty,
        "final_q": dict(scores),
        "action_available": action_available,
        # Phase 1B: boredom diagnostics
        "boredom_penalty": boredom_penalty,
        "boredom_weight": cfg.boredom_weight,
        "avg_prefix_cost": avg_prefix_cost,
        "learning_gain": learning_gain,
        "wait_score_pre_boredom": wait_score_pre_boredom,
    }

    counterfactual = {
        "WAIT": (wait_risk, pred_wait.predicted_plan.expected_cost),
        "WARN": (warn_risk, pred_warn.predicted_plan.expected_cost),
        "UNLOCK": (unlock_risk, pred_wait.predicted_plan.expected_cost),
        "ITEM_DROP": (item_risk, pred_wait.predicted_plan.expected_cost),
    }

    # Filter to allowed actions (WAIT is always available as fallback)
    if allowed_actions is not None:
        allowed = allowed_actions | {"WAIT"}
        scores = {k: v for k, v in scores.items() if k in allowed}

    # Select best
    best_action = max(scores, key=scores.get)
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else 0.0

    reason = _determine_reason(best_action, wait_risk, catastrophe_reduction,
                                learning_gain, wait_fm, item_effect, bn)

    # Track intervention in perceptual memory
    if perceptual_access is not None and best_action != "WAIT":
        key = f"{best_action.lower()}_count"
        perceptual_access.intervention_memory[key] = \
            perceptual_access.intervention_memory.get(key, 0) + 1

    return InterventionDecision(
        action=best_action,
        scores=scores,
        reason=reason,
        decision_margin=margin,
        predicted_prefix=pred_wait.predicted_plan.planned_prefix,
        predicted_failure_modes=pred_wait.predicted_failure_modes,
        counterfactual_scores=counterfactual,
        expected_item_effect=item_effect,
        bottleneck=bn,
        score_decomposition=_decomp,
    )


def _estimate_shortest_path_len(
    start: tuple[int, int], goal: tuple[int, int],
    passable: np.ndarray,
) -> int:
    """Quick BFS shortest path length (for structural slack estimation)."""
    from collections import deque
    H, W = passable.shape
    visited = set()
    queue = deque([(start, 0)])
    visited.add(start)
    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == goal:
            return dist
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and passable[nr, nc] and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return 999  # unreachable


def _build_warn_extra(meta, warned_segs, agent_pos):
    """Build hypothetical warning extra-cost dict for unwarned segments ahead."""
    extra = {}
    for seg in meta.segments:
        if seg.index in warned_segs:
            continue
        # Only warn segments ahead of agent
        if seg.col_start <= agent_pos[1]:
            continue
        for r, c in seg.risky_cells:
            extra[(r, c)] = 5.0  # warning bias
    return extra


def _find_unlockable_cells(meta, passable, agent_pos):
    """Find closed doors ahead of agent that could be unlocked."""
    cells = []
    # Check all_door_positions first (for unlock_shortcut families)
    if hasattr(meta, 'all_door_positions') and meta.all_door_positions:
        for door_pos in meta.all_door_positions:
            r, c = int(door_pos[0]), int(door_pos[1])
            if not passable[r, c]:
                cells.append((r, c))
    if not cells:
        # Fallback: legacy segment gates
        for seg in meta.segments:
            gate = seg.risky_entry_gate
            if not passable[gate[0], gate[1]] and gate[1] > agent_pos[1]:
                cells.append(gate)
    return cells


def _determine_reason(action, wait_risk, cat_reduction, learning_gain, fm,
                       item_effect=None, bottleneck=None):
    """Heuristic reason for the chosen action."""
    # Phase 10: use bottleneck if available
    if bottleneck is not None and action != "WAIT":
        bn_type = bottleneck.dominant
        return f"{bn_type}_bottleneck"

    if action == "WAIT":
        if learning_gain > 0.1:
            return "agent_can_learn"
        return "predicted_path_safe"
    elif action == "WARN":
        if cat_reduction > 0.2:
            return "warning_reduces_catastrophe"
        return "warning_shifts_prefix"
    elif action == "UNLOCK":
        return "topology_improvement"
    elif action == "ITEM_DROP":
        return "shield_reduces_risk"
    return "unknown"

