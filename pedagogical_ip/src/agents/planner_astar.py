"""
Bounded-budget A* planner.

Expands at most `budget` nodes, returns partial plan.
Supports online replanning by calling plan() each step with updated belief.

Architecture:
  _astar_core()        — shared A* engine (cost_fn callable)
  bounded_astar()      — V0 wrapper: scalar belief → path
  plan_next_action()   — V0 wrapper: scalar belief → (action, next_pos)
  cell_cost_v2()       — V2 cost function: feature belief + risk head
  plan_next_action_v2()— V2 wrapper: feature belief → (action, next_pos)
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Cardinal moves: (dr, dc, action_name)
MOVES = [
    (-1, 0, "UP"),
    (1, 0, "DOWN"),
    (0, -1, "LEFT"),
    (0, 1, "RIGHT"),
]

# Discrete budget classes (approx NegBin)
BUDGET_CLASSES = {
    4:  np.array([3, 4, 5]),
    8:  np.array([6, 8, 12]),
    16: np.array([14, 16, 20]),
}
BUDGET_CLASS_PROBS = np.array([0.25, 0.50, 0.25])  # peaked at center


def sample_search_budget(
    budget_class: int = 8,
    rng: np.random.Generator | None = None,
) -> int:
    """
    Sample a search budget from a discrete approximation to Negative-Binomial.

    budget_class ∈ {4, 8, 16} maps to a small set of candidate budgets.
    """
    if rng is None:
        rng = np.random.default_rng()
    candidates = BUDGET_CLASSES.get(budget_class, np.array([budget_class]))
    return int(rng.choice(candidates, p=BUDGET_CLASS_PROBS))


@dataclass(order=True)
class AStarNode:
    """Priority-queue node for A*."""
    f_score: float
    g_score: float = field(compare=False)
    pos: tuple[int, int] = field(compare=False)
    path: list[tuple[int, int]] = field(compare=False, default_factory=list)


def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ── Shared A* core ──────────────────────────────────────────────────

def _astar_core(
    start: tuple[int, int],
    goal: tuple[int, int],
    cost_fn: Callable[[int, int], float],
    H: int,
    W: int,
    budget: int = 30,
    passable_mask: Optional[np.ndarray] = None,
) -> list[tuple[int, int]]:
    """
    Shared bounded A* engine.

    Both V0 (bounded_astar) and V2 (plan_next_action_v2) delegate here.
    The caller defines the cost semantics via cost_fn(row, col) -> float.

    Returns path from start toward goal (or best partial if budget exhausted).
    """
    root = AStarNode(
        f_score=heuristic(start, goal),
        g_score=0.0,
        pos=start,
        path=[start],
    )
    open_set: list[AStarNode] = [root]
    visited: set[tuple[int, int]] = set()
    best_node = root
    expansions = 0

    while open_set and expansions < budget:
        node = heapq.heappop(open_set)
        if node.pos in visited:
            continue
        visited.add(node.pos)
        expansions += 1

        # Track node closest to goal (by heuristic)
        if heuristic(node.pos, goal) < heuristic(best_node.pos, goal):
            best_node = node

        if node.pos == goal:
            return node.path

        for dr, dc, _ in MOVES:
            nr, nc = node.pos[0] + dr, node.pos[1] + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) in visited:
                continue
            if passable_mask is not None and not passable_mask[nr, nc]:
                continue

            step_cost = cost_fn(nr, nc)
            if np.isinf(step_cost):
                continue

            new_g = node.g_score + step_cost
            new_f = new_g + heuristic((nr, nc), goal)
            heapq.heappush(open_set, AStarNode(
                f_score=new_f,
                g_score=new_g,
                pos=(nr, nc),
                path=node.path + [(nr, nc)],
            ))

    # Budget exhausted — return best partial path
    return best_node.path


def _path_to_action(
    agent_pos: tuple[int, int],
    path: list[tuple[int, int]],
) -> tuple[str, tuple[int, int]]:
    """Convert an A* path into a (action_name, next_pos) pair."""
    if len(path) < 2:
        return "STAY", agent_pos

    next_pos = path[1]
    dr = next_pos[0] - agent_pos[0]
    dc = next_pos[1] - agent_pos[1]

    for mdr, mdc, name in MOVES:
        if dr == mdr and dc == mdc:
            return name, next_pos

    return "STAY", agent_pos


# ── V0 Cost Function (scalar belief) ────────────────────────────────

def cell_cost(
    row: int,
    col: int,
    belief_cost_mean: np.ndarray,
    belief_risk_mean: np.ndarray,
    belief_cost_var: np.ndarray,
    lambda_risk: float,
    lambda_uncertainty: float,
) -> float:
    """
    Compute the planning cost for entering a cell, using beliefs.

    score = μ_c + λ_r · φ(μ_ρ) + λ_u · σ²_ρ

    where φ(μ_ρ) = -log(1 - clip(μ_ρ, ε, 1-ε))  [survival form]
    """
    H, W = belief_cost_mean.shape
    if not (0 <= row < H and 0 <= col < W):
        return np.inf

    cost = belief_cost_mean[row, col]
    if cost > 50.0:
        # Treat as impassable (wall / locked door in belief)
        return np.inf

    eps = 1e-4
    mu_rho = np.clip(belief_risk_mean[row, col], eps, 1.0 - eps)
    risk_penalty = lambda_risk * (-np.log(1.0 - mu_rho))
    unc_penalty = lambda_uncertainty * belief_cost_var[row, col]
    return cost + risk_penalty + unc_penalty


def bounded_astar(
    start: tuple[int, int],
    goal: tuple[int, int],
    belief_cost_mean: np.ndarray,
    belief_risk_mean: np.ndarray,
    belief_cost_var: np.ndarray,
    budget: int = 30,
    lambda_risk: float = 3.0,
    lambda_uncertainty: float = 0.5,
    passable_mask: Optional[np.ndarray] = None,
) -> list[tuple[int, int]]:
    """
    Bounded A*: expand at most `budget` nodes.

    Returns a path (list of (row, col) from start to goal or best partial).
    If no path found within budget, returns the partial path toward
    the most promising frontier node.

    Args:
        start: starting position
        goal: goal position
        belief_cost_mean: (H, W) estimated move costs
        belief_risk_mean: (H, W) estimated risk probabilities
        belief_cost_var: (H, W) cost variance (epistemic uncertainty)
        budget: max nodes to expand
        lambda_risk: weight on risk
        lambda_uncertainty: weight on uncertainty
        passable_mask: optional (H, W) bool array; True = passable
    """
    H, W = belief_cost_mean.shape

    def cost_fn(r: int, c: int) -> float:
        return cell_cost(
            r, c,
            belief_cost_mean, belief_risk_mean, belief_cost_var,
            lambda_risk, lambda_uncertainty,
        )

    return _astar_core(start, goal, cost_fn, H, W, budget, passable_mask)


def plan_next_action(
    agent_pos: tuple[int, int],
    goal: tuple[int, int],
    belief_cost_mean: np.ndarray,
    belief_risk_mean: np.ndarray,
    belief_cost_var: np.ndarray,
    budget: int = 30,
    lambda_risk: float = 3.0,
    lambda_uncertainty: float = 0.5,
    passable_mask: Optional[np.ndarray] = None,
) -> tuple[str, tuple[int, int]]:
    # DEPRECATED: V0 legacy — only used by archival BoundedRationalAgent.
    """
    Plan and return the next single action.

    Returns (action_name, next_pos).
    If stuck, returns ("STAY", current_pos).
    """
    path = bounded_astar(
        agent_pos, goal,
        belief_cost_mean, belief_risk_mean, belief_cost_var,
        budget=budget,
        lambda_risk=lambda_risk,
        lambda_uncertainty=lambda_uncertainty,
        passable_mask=passable_mask,
    )
    return _path_to_action(agent_pos, path)


# ── V2 Cost Function (feature-based) ────────────────────────────

def cell_cost_v2(
    row: int,
    col: int,
    belief_cost_mean: np.ndarray,
    feature_belief_mean: np.ndarray,   # (H, W, d) from FeatureBeliefMap
    risk_model,                         # BayesianRiskHead instance
    lambda_risk: float = 5.0,
    lambda_uncertainty: float = 0.1,
) -> float:
    # DEPRECATED: legacy V2 path (no latent predictor). Canonical path uses cell_cost_v2_latent.
    """
    V2 cell cost using feature-based risk prediction (LEGACY path).

    score = μ_c + λ_r · φ(ρ̂(x_belief)) + λ_u · σ²_pred(x_belief)

    KEY: uses feature_belief_mean (agent's noisy belief), NOT true features.
    """
    H, W = belief_cost_mean.shape
    if not (0 <= row < H and 0 <= col < W):
        return np.inf

    cost = belief_cost_mean[row, col]
    if cost > 50.0:
        return np.inf

    x_belief = feature_belief_mean[row, col]
    rho_hat = risk_model.predict_risk(x_belief)
    sigma_hat = risk_model.predict_uncertainty(x_belief)

    eps = 1e-4
    mu_rho = float(np.clip(rho_hat, eps, 1.0 - eps))
    risk_penalty = lambda_risk * (-np.log(1.0 - mu_rho))
    unc_penalty = lambda_uncertainty * sigma_hat

    return cost + risk_penalty + unc_penalty


def cell_cost_v2_latent(
    row: int,
    col: int,
    feature_belief_mean: np.ndarray,   # (H, W, d)
    latent_predictor,                   # LatentCostRiskHead (latent_predictor protocol)
    passable: np.ndarray,               # (H, W) bool
    lambda_c: float = 1.0,
    lambda_risk: float = 5.0,
    lambda_uc: float = 0.1,
    lambda_ur: float = 0.1,
    inventory_state=None,               # InventoryState or None
    feature_belief_var: Optional[np.ndarray] = None,  # (H, W, d)
    route_necessity: float = 0.0,       # scalar n ∈ [0,1]
) -> float:
    """
    V2 cell cost using joint latent predictor (LATENT path).

    Phase 10 formula:
      J = λ_c · ĉ_i
        + λ_r · φ(r̂_i)  [× (1-γ_shield) if shield]
        + λ_uc · (1-n) · u_c_i
        + λ_ur · (1-n) · u_r_i

    Where:
      - ĉ_i, r̂_i: predicted cost and risk from latent predictor
      - u_c_i, u_r_i: directional uncertainty from posterior variance
      - n: route necessity (high → discount uncertainty penalty)
      - φ(r̂): risk penalty transform = -ln(1-r̂)

    Key principle: unknown ≠ dangerous. When route_necessity is high
    (no safe alternative), uncertainty penalty drops to zero.
    """
    H, W = passable.shape
    if not (0 <= row < H and 0 <= col < W):
        return np.inf
    if not passable[row, col]:
        return np.inf

    x_belief = feature_belief_mean[row, col]

    cost_hat = latent_predictor.predict_cost(x_belief)
    risk_hat = latent_predictor.predict_risk(x_belief)

    # Directional uncertainty from posterior variance (Phase 10)
    # or fallback to Hessian-based scalar (legacy)
    if feature_belief_var is not None:
        x_var = feature_belief_var[row, col]
        cost_unc = latent_predictor.predict_cost_uncertainty_from_var(x_var)
        risk_unc = latent_predictor.predict_risk_uncertainty_from_var(x_var)
    else:
        cost_unc = latent_predictor.predict_cost_uncertainty(x_belief)
        risk_unc = latent_predictor.predict_risk_uncertainty(x_belief)

    # NaN safety: online weight updates can produce extreme values
    if np.isnan(cost_hat) or np.isnan(risk_hat):
        return 10.0  # fallback: moderate cost
    cost_unc = max(0.0, float(np.nan_to_num(cost_unc, nan=0.0)))
    risk_unc = max(0.0, float(np.nan_to_num(risk_unc, nan=0.0)))

    eps = 1e-4
    mu_rho = float(np.clip(risk_hat, eps, 1.0 - eps))
    risk_penalty_full = lambda_risk * (-np.log(1.0 - mu_rho))

    # Shield reduces risk penalty (same factor as execution)
    if inventory_state is not None and inventory_state.has_shield():
        risk_penalty_full *= (1.0 - inventory_state.shield_risk_reduction)

    # Necessity-aware risk/uncertainty penalty (Phase 10).
    # "Unknown ≠ dangerous": discount epistemic portions by necessity.
    #
    # The risk head starts with w=0, b=0 → sigmoid(0)=0.5 = maximum entropy.
    # This is NOT a genuine risk prediction — it's epistemic uncertainty
    # masquerading as predicted risk. As the model trains (n_updates grows),
    # the prediction becomes data-driven and should be trusted.
    #
    # learning_factor: 0.0 when untrained (all epistemic), 1.0 when trained
    n_updates = latent_predictor.n_updates if hasattr(latent_predictor, 'n_updates') else 0
    learning_factor = min(1.0, n_updates / 10.0)
    # Blend: trained portion kept, untrained portion discounted by necessity
    necessity_discount = 1.0 - route_necessity
    risk_penalty = risk_penalty_full * (learning_factor + (1.0 - learning_factor) * necessity_discount)

    result = (lambda_c * cost_hat
              + risk_penalty
              + lambda_uc * necessity_discount * cost_unc
              + lambda_ur * necessity_discount * risk_unc)

    # Final NaN safety
    return result if np.isfinite(result) else 10.0


def plan_next_action_v2(
    agent_pos: tuple[int, int],
    goal: tuple[int, int],
    belief_cost_mean: np.ndarray,
    feature_belief_mean: np.ndarray,
    risk_model,
    budget: int = 30,
    lambda_risk: float = 5.0,
    lambda_uncertainty: float = 0.1,
    passable_mask: Optional[np.ndarray] = None,
    warned_cell_extra_cost: Optional[dict] = None,
    latent_predictor=None,
    lambda_c: float = 1.0,
    lambda_uc: float = 0.1,
    lambda_ur: float = 0.1,
    feature_belief_var: Optional[np.ndarray] = None,
    route_necessity: float = 0.0,
) -> tuple[str, tuple[int, int], list[tuple[int, int]]]:
    # DEPRECATED: legacy V2 fallback. Canonical path uses plan_from_belief().
    """V2 planning — LEGACY PATH.

    .. deprecated::
        All canonical experiments use belief_planning_mode=True, which calls
        plan_from_belief() → plan_with_alternatives_v2() instead. This function
        is only reached when belief_planning_mode=False (the default, but unused
        by any experiment script). Does NOT support inventory_state / shield.

    If latent_predictor is provided, uses joint cost+risk prediction from
    the latent model. Otherwise uses legacy risk-only path.

    Returns (action_name, next_pos, full_path).
    """
    H, W = belief_cost_mean.shape

    if latent_predictor is not None:
        passable = passable_mask if passable_mask is not None else np.ones((H, W), dtype=bool)

        def cost_fn(r: int, c: int) -> float:
            base = cell_cost_v2_latent(
                r, c, feature_belief_mean, latent_predictor, passable,
                lambda_c, lambda_risk, lambda_uc, lambda_ur,
                feature_belief_var=feature_belief_var,
                route_necessity=route_necessity,
            )
            if warned_cell_extra_cost and (r, c) in warned_cell_extra_cost:
                base += warned_cell_extra_cost[(r, c)]
            return base
    else:
        def cost_fn(r: int, c: int) -> float:
            base = cell_cost_v2(
                r, c,
                belief_cost_mean, feature_belief_mean, risk_model,
                lambda_risk, lambda_uncertainty,
            )
            if warned_cell_extra_cost and (r, c) in warned_cell_extra_cost:
                base += warned_cell_extra_cost[(r, c)]
            return base

    path = _astar_core(agent_pos, goal, cost_fn, H, W, budget, passable_mask)
    action, next_pos = _path_to_action(agent_pos, path)
    return action, next_pos, path


def plan_with_alternatives_v2(
    agent_pos: tuple[int, int],
    goal: tuple[int, int],
    belief_cost_mean: np.ndarray,
    feature_belief_mean: np.ndarray,
    risk_model,
    budget: int = 30,
    lambda_risk: float = 5.0,
    lambda_uncertainty: float = 0.1,
    passable_mask: Optional[np.ndarray] = None,
    warned_cell_extra_cost: Optional[dict] = None,
    latent_predictor=None,
    lambda_c: float = 1.0,
    lambda_uc: float = 0.1,
    lambda_ur: float = 0.1,
    inventory_state=None,               # InventoryState or None
    feature_belief_var: Optional[np.ndarray] = None,
    route_necessity: float = 0.0,
) -> tuple[str, tuple[int, int], list[tuple[int, int]], dict[str, float]]:
    """V2 planning with path-level candidate scores for all first-step options.

    Returns (action, next_pos, path, candidate_scores).
    candidate_scores maps action_name → path-level total cost for that
    first-step direction, computed via constrained short A*.
    """
    H, W = belief_cost_mean.shape

    if latent_predictor is not None:
        passable = passable_mask if passable_mask is not None else np.ones((H, W), dtype=bool)

        def cost_fn(r: int, c: int) -> float:
            base = cell_cost_v2_latent(
                r, c, feature_belief_mean, latent_predictor, passable,
                lambda_c, lambda_risk, lambda_uc, lambda_ur,
                inventory_state=inventory_state,
                feature_belief_var=feature_belief_var,
                route_necessity=route_necessity,
            )
            if warned_cell_extra_cost and (r, c) in warned_cell_extra_cost:
                base += warned_cell_extra_cost[(r, c)]
            return base
    else:
        def cost_fn(r: int, c: int) -> float:
            base = cell_cost_v2(
                r, c,
                belief_cost_mean, feature_belief_mean, risk_model,
                lambda_risk, lambda_uncertainty,
            )
            if warned_cell_extra_cost and (r, c) in warned_cell_extra_cost:
                base += warned_cell_extra_cost[(r, c)]
            return base

    # Best path via full A*
    path = _astar_core(agent_pos, goal, cost_fn, H, W, budget, passable_mask)
    action, next_pos = _path_to_action(agent_pos, path)

    # Path-level candidate scores: for each valid first-step neighbor,
    # compute cost_fn(neighbor) + A* path cost from neighbor to goal
    candidate_scores: dict[str, float] = {}
    short_budget = max(budget // 2, 8)

    for dr, dc, move_name in MOVES:
        nr, nc = agent_pos[0] + dr, agent_pos[1] + dc
        if not (0 <= nr < H and 0 <= nc < W):
            continue
        if passable_mask is not None and not passable_mask[nr, nc]:
            continue
        step_cost = cost_fn(nr, nc)
        if np.isinf(step_cost):
            continue
        # Short A* from this neighbor toward goal
        sub_path = _astar_core((nr, nc), goal, cost_fn, H, W, short_budget, passable_mask)
        # Total path cost = first step + sum of remaining steps
        total = step_cost
        for i in range(1, len(sub_path)):
            total += cost_fn(sub_path[i][0], sub_path[i][1])
        candidate_scores[move_name] = total

    return action, next_pos, path, candidate_scores
