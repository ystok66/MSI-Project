"""
v1d Cause-Aware Teacher Scoring.

Introduces a latent failure-cause variable z ∈ {explore, belief, plan, hazard}
and a two-stage decision framework:
  Stage 1: Infer q(z | h_t) — why is the learner likely to fail?
  Stage 2: Select intervention conditioned on dominant cause.

All heavy computations are vectorized with numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..agents.belief import BeliefMap, apply_rsa_warning
from ..agents.planner_astar import bounded_astar


# ── Cause types ──────────────────────────────────────────────────────

CAUSES = ["safe_exploration", "belief_error", "planning_bottleneck", "immediate_hazard"]


@dataclass
class CauseScores:
    """Scores for each latent failure cause at one timestep."""
    explore: float = 0.0
    belief: float = 0.0
    plan: float = 0.0
    hazard: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.explore, self.belief, self.plan, self.hazard])

    def posterior(self, kappa: float = 3.0) -> np.ndarray:
        """Softmax posterior q(z | h_t)."""
        s = self.as_array() * kappa
        s -= s.max()  # numerical stability
        e = np.exp(s)
        return e / e.sum()

    def dominant_cause(self, kappa: float = 3.0) -> str:
        idx = int(np.argmax(self.posterior(kappa)))
        return CAUSES[idx]


@dataclass
class CauseDiagnostics:
    """Per-step diagnostics from cause-aware scoring."""
    cause_posterior: np.ndarray    # (4,) softmax over causes
    dominant_cause: str
    p_fatal_wait: float
    p_timeout_wait: float
    warn_precision: float         # 1.0 if warning was on truly risky region
    unlock_useful: float          # 1.0 if unlock changed plan
    wait_safe: float              # 1.0 if WAIT didn't lead to failure within H
    # Per-modality scores
    s_explore: float
    s_belief: float
    s_plan: float
    s_hazard: float


# ── Vectorized helpers ───────────────────────────────────────────────

def compute_survival_prob(
    path: list[tuple[int, int]],
    true_risk: np.ndarray,
    shielded: bool = False,
) -> float:
    """Vectorized survival probability along a path. Π(1 - ρ_effective)."""
    if not path or len(path) < 2:
        return 0.5
    rows = np.array([r for r, c in path[1:]])
    cols = np.array([c for r, c in path[1:]])
    risks = true_risk[rows, cols]
    if shielded:
        risks = risks * 0.067  # ≈ 0.02/0.3
    return float(np.prod(np.maximum(1.0 - risks, 0.01)))


def compute_path_cost(
    path: list[tuple[int, int]],
    true_cost: np.ndarray,
    true_risk: np.ndarray,
    lambda_risk: float = 3.0,
    shielded: bool = False,
) -> float:
    """Risk-aware additive path cost: L = Σc + λ_r·Σ[-log(1-ρ)]."""
    if not path or len(path) < 2:
        return 999.0
    rows = np.array([r for r, c in path[1:]])
    cols = np.array([c for r, c in path[1:]])
    costs = true_cost[rows, cols]
    risks = true_risk[rows, cols]
    if shielded:
        risks = risks * 0.067
    risks_clamped = np.minimum(risks, 0.99)
    return float(np.sum(costs) + lambda_risk * np.sum(-np.log(1.0 - risks_clamped)))


def compute_success_prob(
    path: list[tuple[int, int]],
    goal: tuple[int, int],
    true_risk: np.ndarray,
    time_left: int,
    shielded: bool = False,
) -> float:
    """Success probability: survival × time feasibility."""
    if not path or path[-1] != goal:
        return 0.05
    path_len = len(path) - 1
    if path_len > time_left:
        return 0.1
    survival = compute_survival_prob(path, true_risk, shielded)
    time_margin = (time_left - path_len) / max(time_left, 1)
    return survival * min(1.0, 0.6 + 0.4 * time_margin)


def compute_calibration_gain(
    est_belief: BeliefMap,
    sim_belief: BeliefMap,
    true_risk: np.ndarray,
    critical_cells: list[tuple[int, int]],
) -> float:
    """Calibration gain: Σ (old_error² - new_error²) on critical cells."""
    if not critical_cells:
        return 0.0
    rows = np.array([r for r, c in critical_cells])
    cols = np.array([c for r, c in critical_cells])
    true_crit = true_risk[rows, cols]
    old_err = est_belief.risk_mean[rows, cols] - true_crit
    new_err = sim_belief.risk_mean[rows, cols] - true_crit
    return float(max(0.0, np.sum(old_err ** 2 - new_err ** 2)))


def get_critical_cells(
    path: list[tuple[int, int]],
    agent_pos: tuple[int, int],
    H: int, W: int,
    max_cells: int = 8,
) -> list[tuple[int, int]]:
    """Critical cells = plan prefix + 3×3 around agent."""
    cells = set()
    if path and len(path) > 1:
        for r, c in path[1:min(max_cells + 1, len(path))]:
            cells.add((r, c))
    # Add neighborhood
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            nr, nc = agent_pos[0] + dr, agent_pos[1] + dc
            if 0 <= nr < H and 0 <= nc < W:
                cells.add((nr, nc))
    return list(cells)


# ── Core: compute cause scores ───────────────────────────────────────

def compute_cause_scores(
    est_belief: BeliefMap,
    agent_pos: tuple[int, int],
    goal: tuple[int, int],
    true_risk: np.ndarray,
    true_cost: np.ndarray,
    time_left: int,
    risk_budget_left: float,
    passable_mask: np.ndarray,
    locked_doors: set[tuple[int, int]],
    door_positions: list[tuple[int, int]],
    # Warning simulation
    best_utterance: str,
    sim_belief_warn: BeliefMap,
    warn_region_mask: np.ndarray | None,
    # Params
    lambda_cal: float = 1.0,
    lambda_plan_change: float = 0.5,
    lambda_fp: float = 0.5,
    rho_warn_thresh: float = 0.15,
    lambda_reach: float = 0.5,
    lambda_dist: float = 0.3,
    gamma_risk: float = 0.3,
    gamma_time: float = 0.7,
    lambda_risk_hazard: float = 2.0,
    lambda_time_hazard: float = 1.5,
    search_budget: int = 40,
) -> CauseScores:
    """
    Compute four cause scores in one pass.

    Returns CauseScores with explore, belief, plan, hazard.
    All heavy operations are vectorized.
    """
    H, W = true_risk.shape

    # ── Shared: current plan under WAIT ──
    wait_plan = bounded_astar(
        agent_pos, goal,
        est_belief.cost_mean, est_belief.risk_mean, est_belief.cost_var,
        budget=search_budget, lambda_risk=3.0, passable_mask=passable_mask,
    ) or []

    p_succ_wait = compute_success_prob(wait_plan, goal, true_risk, time_left)
    survival_wait = compute_survival_prob(wait_plan, true_risk)

    # Fatal/timeout probabilities under WAIT
    p_fatal_wait = max(0.0, 1.0 - survival_wait)
    if wait_plan and wait_plan[-1] == goal:
        path_len = len(wait_plan) - 1
        p_timeout_wait = max(0.0, 1.0 - max(0.0, (time_left - path_len) / max(time_left, 1)))
    else:
        p_timeout_wait = 0.8  # no path → high timeout probability

    critical = get_critical_cells(wait_plan, agent_pos, H, W)

    # ═══ S_explore: safe to wait + exploration has value ═══
    wait_safe = (p_fatal_wait < gamma_risk) and (p_timeout_wait < gamma_time)
    # Exploration gain: how much unvisited critical terrain remains
    if critical:
        crit_r = np.array([r for r, c in critical])
        crit_c = np.array([c for r, c in critical])
        uncertainty = float(np.mean(est_belief.risk_var[crit_r, crit_c]))
    else:
        uncertainty = 0.0
    ig_wait = uncertainty  # higher uncertainty → more to learn by exploring
    s_explore = float(wait_safe) * ig_wait

    # ═══ S_belief: WARN would fix a real misconception ═══
    warn_plan = bounded_astar(
        agent_pos, goal,
        sim_belief_warn.cost_mean, sim_belief_warn.risk_mean,
        sim_belief_warn.cost_var,
        budget=search_budget, lambda_risk=3.0, passable_mask=passable_mask,
    ) or []

    delta_p_succ_warn = compute_success_prob(
        warn_plan, goal, true_risk, time_left,
    ) - p_succ_wait

    ig_cal_warn = compute_calibration_gain(
        est_belief, sim_belief_warn, true_risk, critical,
    )

    # Plan-change check (vectorized comparison)
    k = min(4, len(wait_plan), len(warn_plan))
    plan_changed = (k > 0 and wait_plan[:k] != warn_plan[:k])

    # False-warning penalty
    false_penalty = 0.0
    warn_precision_flag = 1.0
    if warn_region_mask is not None and warn_region_mask.any():
        mean_true_risk_region = float(true_risk[warn_region_mask].mean())
        if mean_true_risk_region < rho_warn_thresh:
            false_penalty = lambda_fp
            warn_precision_flag = 0.0

    s_belief = (
        delta_p_succ_warn
        + lambda_cal * ig_cal_warn
        + lambda_plan_change * float(plan_changed)
        - false_penalty
    )

    # ═══ S_plan: UNLOCK enables structurally better path (v1e) ═══
    # Pure structural indicators — NO belief calibration here.
    # S_plan = λ₁·ΔP_succ + λ₂·Δd_bounded + λ₃·ΔReach(safe) + λ₄·ΔPlanPrefix
    s_plan = 0.0
    unlock_useful_flag = 0.0
    if locked_doors and door_positions:
        # Construct unlocked passable mask
        unlock_passable = passable_mask.copy()
        unlock_cost_mean = est_belief.cost_mean.copy()
        unlock_cost_var = est_belief.cost_var.copy()
        for dr, dc in door_positions:
            if (dr, dc) in locked_doors:
                unlock_passable[dr, dc] = True
                unlock_cost_mean[dr, dc] = 1.0
                unlock_cost_var[dr, dc] = 0.01

        # Learner replans with SAME bounded budget on unlocked topology
        unlock_plan = bounded_astar(
            agent_pos, goal,
            unlock_cost_mean, est_belief.risk_mean, unlock_cost_var,
            budget=search_budget, lambda_risk=3.0, passable_mask=unlock_passable,
        ) or []

        # λ₁: Success probability improvement
        delta_p_succ_unlock = compute_success_prob(
            unlock_plan, goal, true_risk, time_left,
        ) - p_succ_wait

        # λ₂: Bounded cost-to-go reduction (risk-aware)
        cost_before = compute_path_cost(wait_plan, true_cost, true_risk)
        cost_after = compute_path_cost(unlock_plan, true_cost, true_risk)
        delta_cost = max(0.0, cost_before - cost_after) / max(cost_before, 1.0)

        # λ₃: Safe route reachability — does unlock make a goal-reaching path exist?
        wait_reaches_goal = bool(wait_plan and wait_plan[-1] == goal)
        unlock_reaches_goal = bool(unlock_plan and unlock_plan[-1] == goal)
        safe_route_gained = float(not wait_reaches_goal and unlock_reaches_goal)

        # λ₄: Plan prefix structural change (first k steps differ)
        ku = min(4, len(wait_plan), len(unlock_plan))
        plan_prefix_changed = float(ku > 0 and wait_plan[:ku] != unlock_plan[:ku])

        unlock_useful_flag = float(
            (delta_p_succ_unlock > 0 and plan_prefix_changed > 0)
            or safe_route_gained > 0
        )

        # Weighted structural score
        lambda_succ = 2.0     # success improvement weight
        lambda_cost = 0.5     # cost reduction weight
        lambda_route = 3.0    # safe-route unlock weight (high!)
        lambda_prefix = 0.3   # plan change weight

        s_plan = (
            lambda_succ * delta_p_succ_unlock
            + lambda_cost * delta_cost
            + lambda_route * safe_route_gained
            + lambda_prefix * plan_prefix_changed
        )

    # ═══ S_hazard: immediate danger ═══
    s_hazard = (
        lambda_risk_hazard * p_fatal_wait
        + lambda_time_hazard * p_timeout_wait
    )

    return CauseScores(
        explore=s_explore,
        belief=s_belief,
        plan=s_plan,
        hazard=s_hazard,
    )


def make_diagnostics(
    scores: CauseScores,
    p_fatal: float,
    p_timeout: float,
    warn_precision: float,
    unlock_useful: float,
    wait_safe: float,
    kappa: float = 3.0,
) -> CauseDiagnostics:
    """Build diagnostics from cause scores."""
    posterior = scores.posterior(kappa)
    return CauseDiagnostics(
        cause_posterior=posterior,
        dominant_cause=scores.dominant_cause(kappa),
        p_fatal_wait=p_fatal,
        p_timeout_wait=p_timeout,
        warn_precision=warn_precision,
        unlock_useful=unlock_useful,
        wait_safe=wait_safe,
        s_explore=scores.explore,
        s_belief=scores.belief,
        s_plan=scores.plan,
        s_hazard=scores.hazard,
    )
