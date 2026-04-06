"""Necessity Gate Variants — Step 5A.1.

Three necessity gate implementations for the risk-sensitive planner:
  N1: Path-level binary gate (hard threshold)
  N2: Soft margin gate (sigmoid of safe-alternative margin)
  N3: Per-cell prefix-aware gate

Plus two monotonicity fixes:
  M1: Normalize uncertainty by feasible range
  M2: Hazard-aware uncertainty discount

Usage:
    from src.agents.necessity_gate_variants import (
        compute_necessity_n1, compute_necessity_n2, compute_necessity_n3,
        apply_monotonicity_m1, apply_monotonicity_m2,
    )
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np


# ═══════════════════════════════════════════════════════════
# N1: Binary gate — uses path-level safe-alternative check
# ═══════════════════════════════════════════════════════════

def compute_necessity_n1(
    candidate_profiles: list,
    current_idx: int,
    hazard_threshold: float = 0.3,
    timeout_threshold: float = 0.5,
) -> float:
    """N1: Binary necessity gate.

    g_N = 1 if safe alternative exists, else 0.

    A safe alternative is any OTHER path with:
      P(hazard) ≤ τ_h AND P(timeout) ≤ τ_t

    Returns:
        g_N ∈ {0, 1}: 1 = full uncertainty surcharge; 0 = no surcharge
    """
    for i, p in enumerate(candidate_profiles):
        if i == current_idx:
            continue
        if (p.hazard_prob <= hazard_threshold and
                p.timeout_prob <= timeout_threshold):
            return 1.0  # safe alternative exists
    return 0.0  # no safe alternative


# ═══════════════════════════════════════════════════════════
# N2: Soft margin gate — sigmoid of safe-alt quality gap
# ═══════════════════════════════════════════════════════════

def compute_safe_alternative_margin(
    candidate_profiles: list,
    current_idx: int,
    lambda_h: float = 3.0,
    lambda_t: float = 2.0,
) -> float:
    """Compute M_safe: how much worse is the best safe alt vs best overall.

    M_safe = score(best_safe_alt) - score(best_overall)

    Small M_safe → safe alt is viable → g_N should be high (full surcharge)
    Large M_safe → safe alt is expensive → g_N should be low (discount)
    """
    def base_score(p):
        return (p.expected_cost + lambda_h * p.hazard_prob +
                lambda_t * p.timeout_prob)

    scores = [base_score(p) for p in candidate_profiles]
    best_overall = min(scores)

    # Find best "safe" alternative (low hazard, low timeout)
    safe_scores = []
    for i, p in enumerate(candidate_profiles):
        if i == current_idx:
            if p.hazard_prob <= 0.3 and p.timeout_prob <= 0.5:
                safe_scores.append(scores[i])
        else:
            if p.hazard_prob <= 0.3 and p.timeout_prob <= 0.5:
                safe_scores.append(scores[i])

    if not safe_scores:
        return 999.0  # no safe alternative at all

    return min(safe_scores) - best_overall


def compute_necessity_n2(
    candidate_profiles: list,
    current_idx: int,
    delta: float = 1.0,
    temperature: float = 2.0,
    lambda_h: float = 3.0,
    lambda_t: float = 2.0,
) -> float:
    """N2: Soft margin necessity gate.

    g_N = σ(-(M_safe - δ) / T_N)

    where M_safe is the cost gap between best safe alt and best overall.

    Large M_safe (no viable safe alt) → g_N → 0 → no surcharge
    Small M_safe (good safe alt exists) → g_N → 1 → full surcharge

    Returns:
        g_N ∈ [0, 1]
    """
    m_safe = compute_safe_alternative_margin(
        candidate_profiles, current_idx, lambda_h, lambda_t)

    # Sigmoid: σ(-(M_safe - δ) / T_N)
    # When M_safe > δ: exponent is negative → σ → 0 (no surcharge)
    # When M_safe < δ: exponent is positive → σ → 1 (full surcharge)
    z = -(m_safe - delta) / max(temperature, 0.01)
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))


# ═══════════════════════════════════════════════════════════
# N3: Per-cell prefix-aware gate
# ═══════════════════════════════════════════════════════════

def compute_necessity_n3(
    path: List[Tuple[int, int]],
    passable: np.ndarray,
    risk_map: np.ndarray,
    t: int,
    t_max: int,
    goal: Tuple[int, int],
    delta: float = 1.0,
    temperature: float = 2.0,
) -> float:
    """N3: Per-cell prefix-aware necessity gate.

    g_N = mean over prefix cells of per-cell necessity.

    For each cell on the path, compute whether a safe detour exists
    from that cell's perspective.

    Returns:
        g_N ∈ [0, 1]: averaged per-cell gate
    """
    from .route_necessity import _bfs_shortest

    cells = path[1:] if len(path) > 1 else path
    if not cells:
        return 0.0

    H, W = passable.shape
    gates = []

    for step_i, (r, c) in enumerate(cells):
        # For this cell: is there a safe detour avoiding it?
        avoid_mask = passable.copy()
        avoid_mask[r, c] = False

        current_t = t + step_i + 1
        remaining = max(t_max - current_t, 1)

        best_len = _bfs_shortest((r, c), goal, passable, H, W)
        avoid_len = _bfs_shortest(
            path[step_i] if step_i < len(path) else (r, c),
            goal, avoid_mask, H, W)

        if best_len >= 999:
            gates.append(0.5)  # can't reach anyway
            continue

        if avoid_len >= 999 or avoid_len > remaining:
            gates.append(0.0)  # no viable detour → no surcharge
        else:
            detour_cost = avoid_len - best_len
            z = -(detour_cost - delta) / max(temperature, 0.01)
            gates.append(float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))))

    return float(np.mean(gates)) if gates else 0.0


# ═══════════════════════════════════════════════════════════
# M1: Normalize uncertainty by feasible range
# ═══════════════════════════════════════════════════════════

def apply_monotonicity_m1(
    epi_unc: float,
    all_uncertainties: List[float],
    eps: float = 0.01,
) -> float:
    """M1: Normalize uncertainty by max across candidates.

    Ũ_epi(π) = U_epi(π) / (ε + max_π' U_epi(π'))

    Prevents absolute uncertainty scale from dominating.
    """
    max_unc = max(all_uncertainties) if all_uncertainties else 0.0
    return epi_unc / (eps + max_unc)


# ═══════════════════════════════════════════════════════════
# M2: Hazard-aware uncertainty discount
# ═══════════════════════════════════════════════════════════

def apply_monotonicity_m2(
    epi_unc: float,
    hazard_prob: float,
    eta_rho: float = 2.0,
) -> float:
    """M2: Discount uncertainty when hazard is already high.

    U_epi_eff(π) = U_epi(π) · exp(-η_ρ · P̄(hazard))

    When hazard is already explicit (high P), uncertainty surcharge
    should not double-penalize.
    """
    return epi_unc * float(np.exp(-eta_rho * hazard_prob))
