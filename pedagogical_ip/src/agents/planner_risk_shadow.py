"""Risk-Sensitive Planner Shadow — Step 5A.

Shadow module for evaluating candidate paths with risk-sensitive objectives.
Sits on top of the existing belief_planning + planner_astar infrastructure.
Does NOT modify any canonical planner module.

Three scoring modes (A1, A2, A3):
  A1. Asymmetric Expected Loss
  A2. + Epistemic Surcharge + Necessity Gate
  A3. CVaR / Tail-Risk Proxy

Formula (A2, recommended):
  J_RS(π) = E[C(π)] + λ_r·P(hazard|π) + λ_u·U_epi(π)·1[safe_alt_exists]
            + λ_t·P(timeout|π)

Key principle: unknown ≠ dangerous. Necessity gate discounts uncertainty
cost when no safe alternative exists.

Usage:
    shadow = PlannerRiskShadow(mode="A2")
    result = shadow.evaluate_path(path, cost_map, risk_map, unc_map, ...)
    comparison = shadow.compare_paths(paths, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np


@dataclass
class RiskShadowConfig:
    """Hyperparameters for risk-sensitive path evaluation.

    Minimal set: 4 weights + 1 necessity temperature.
    """
    lambda_cost: float = 1.0        # expected cost weight
    lambda_hazard: float = 3.0      # hazard probability weight
    lambda_uncertainty: float = 0.5 # epistemic surcharge weight
    lambda_timeout: float = 2.0     # timeout probability weight
    lambda_detour: float = 0.3      # detour cost weight
    necessity_tau: float = 3.0      # necessity gate temperature
    cvar_alpha: float = 0.2         # CVaR confidence level (A3 only)


@dataclass
class PathRiskProfile:
    """Risk profile of a candidate path."""
    path: List[Tuple[int, int]]
    path_length: int

    # Component scores
    expected_cost: float
    hazard_prob: float           # cumulative P(at least one hazard)
    epistemic_uncertainty: float # mean Var[ρ] along path
    timeout_prob: float          # P(path > remaining time)
    detour_cost: float           # excess length vs shortest path

    # Necessity gate
    necessity: float             # ∈ [0,1]: 1 = no safe alternative
    safe_alt_exists: bool
    effective_unc_surcharge: float  # U_epi · (1 - necessity)

    # Composite scores
    score_a1: float   # asymmetric expected loss
    score_a2: float   # + epistemic surcharge + necessity
    score_a3: float   # CVaR proxy

    # Breakdown for diagnostics
    score_breakdown: Dict[str, float] = field(default_factory=dict)


class PlannerRiskShadow:
    """Risk-sensitive planner shadow evaluator.

    Evaluates candidate paths with proper risk-sensitive objectives.
    Does NOT replace the canonical planner — runs in parallel for comparison.

    Usage:
        shadow = PlannerRiskShadow(mode="A2", config=RiskShadowConfig())
        profile = shadow.evaluate_path(path, env_data)
        best, profiles = shadow.rank_paths(candidate_paths, env_data)
    """

    def __init__(self,
                 mode: str = "A2",
                 config: Optional[RiskShadowConfig] = None):
        assert mode in ("A1", "A2", "A3"), f"Unknown mode: {mode}"
        self.mode = mode
        self.cfg = config or RiskShadowConfig()

    def evaluate_path(
        self,
        path: List[Tuple[int, int]],
        cost_map: np.ndarray,
        risk_map: np.ndarray,
        uncertainty_map: np.ndarray,
        passable: np.ndarray,
        t: int,
        t_max: int,
        goal: Tuple[int, int],
        shortest_path_len: Optional[int] = None,
    ) -> PathRiskProfile:
        """Evaluate a single path with risk-sensitive objective.

        Args:
            path: list of (row, col) positions
            cost_map: (H, W) expected traversal cost
            risk_map: (H, W) estimated hazard probability per cell
            uncertainty_map: (H, W) epistemic variance of risk estimate
            passable: (H, W) bool
            t: current timestep
            t_max: episode deadline
            goal: target position
            shortest_path_len: if known, for detour computation
        """
        cfg = self.cfg
        cells = path[1:] if len(path) > 1 else path  # exclude current pos
        n = max(len(cells), 1)

        # ── Expected cost ──
        exp_cost = sum(float(cost_map[r, c]) for r, c in cells)

        # ── Hazard probability (independence approximation) ──
        # P(at least one hazard) = 1 - Π(1 - risk_i)
        survival = 1.0
        for r, c in cells:
            survival *= (1.0 - float(np.clip(risk_map[r, c], 0, 1)))
        hazard_prob = 1.0 - survival

        # ── Epistemic uncertainty (mean variance along path) ──
        if len(cells) > 0:
            epi_unc = float(np.mean([uncertainty_map[r, c] for r, c in cells]))
        else:
            epi_unc = 0.0

        # ── Timeout probability ──
        remaining = max(t_max - t, 1)
        path_len = len(path) - 1 if len(path) > 1 else 0
        timeout_prob = 0.0
        if path_len > remaining:
            timeout_prob = 1.0
        elif path_len > 0.8 * remaining:
            # Soft threshold: increasing probability as path approaches deadline
            timeout_prob = float((path_len - 0.8 * remaining) / (0.2 * remaining))

        # ── Detour cost ──
        if shortest_path_len is not None and shortest_path_len > 0:
            detour = max(0, path_len - shortest_path_len)
        else:
            detour = 0

        # ── Necessity gate ──
        necessity = self._compute_necessity(
            path, passable, t, t_max, goal)
        safe_alt = necessity < 0.8
        eff_unc = epi_unc * (1.0 - necessity) if safe_alt else epi_unc * 0.1

        # ── Composite scores ──
        # A1: Asymmetric Expected Loss
        s_a1 = (cfg.lambda_cost * exp_cost
                + cfg.lambda_hazard * hazard_prob
                + cfg.lambda_timeout * timeout_prob
                + cfg.lambda_detour * detour)

        # A2: + Epistemic Surcharge + Necessity Gate
        s_a2 = s_a1 + cfg.lambda_uncertainty * eff_unc

        # A3: CVaR proxy (pessimistic cost estimate)
        # Approximate: cost + risk/alpha penalty
        alpha = max(cfg.cvar_alpha, 0.01)
        cvar_proxy = exp_cost + hazard_prob / alpha
        s_a3 = (cfg.lambda_cost * cvar_proxy
                + cfg.lambda_timeout * timeout_prob
                + cfg.lambda_detour * detour)

        breakdown = {
            "cost": cfg.lambda_cost * exp_cost,
            "hazard": cfg.lambda_hazard * hazard_prob,
            "uncertainty": cfg.lambda_uncertainty * eff_unc,
            "timeout": cfg.lambda_timeout * timeout_prob,
            "detour": cfg.lambda_detour * detour,
            "necessity": necessity,
        }

        return PathRiskProfile(
            path=path,
            path_length=path_len,
            expected_cost=exp_cost,
            hazard_prob=hazard_prob,
            epistemic_uncertainty=epi_unc,
            timeout_prob=timeout_prob,
            detour_cost=float(detour),
            necessity=necessity,
            safe_alt_exists=safe_alt,
            effective_unc_surcharge=eff_unc,
            score_a1=s_a1,
            score_a2=s_a2,
            score_a3=s_a3,
            score_breakdown=breakdown,
        )

    def score(self, profile: PathRiskProfile) -> float:
        """Get the score for the current mode."""
        if self.mode == "A1":
            return profile.score_a1
        elif self.mode == "A2":
            return profile.score_a2
        else:
            return profile.score_a3

    def rank_paths(
        self,
        candidate_paths: List[List[Tuple[int, int]]],
        cost_map: np.ndarray,
        risk_map: np.ndarray,
        uncertainty_map: np.ndarray,
        passable: np.ndarray,
        t: int,
        t_max: int,
        goal: Tuple[int, int],
        shortest_path_len: Optional[int] = None,
    ) -> Tuple[int, List[PathRiskProfile]]:
        """Rank candidate paths by risk-sensitive objective.

        Returns:
            (best_idx, profiles) — index of best path and all profiles
        """
        profiles = [
            self.evaluate_path(
                p, cost_map, risk_map, uncertainty_map,
                passable, t, t_max, goal, shortest_path_len)
            for p in candidate_paths
        ]

        scores = [self.score(p) for p in profiles]
        best_idx = int(np.argmin(scores))  # lower is better
        return best_idx, profiles

    def compare_to_baseline(
        self,
        baseline_score: float,
        baseline_path_idx: int,
        profiles: List[PathRiskProfile],
    ) -> Dict[str, object]:
        """Compare shadow ranking to baseline planner.

        Returns diagnostic dict with agreement, score deltas, etc.
        """
        shadow_scores = [self.score(p) for p in profiles]
        shadow_best = int(np.argmin(shadow_scores))

        agrees = shadow_best == baseline_path_idx
        if len(profiles) > 1:
            shadow_gap = sorted(shadow_scores)[1] - sorted(shadow_scores)[0]
        else:
            shadow_gap = 0.0

        return {
            "agrees_with_baseline": agrees,
            "shadow_best_idx": shadow_best,
            "baseline_best_idx": baseline_path_idx,
            "shadow_gap": shadow_gap,
            "shadow_best_score": shadow_scores[shadow_best],
            "baseline_shadow_score": shadow_scores[baseline_path_idx] if baseline_path_idx < len(shadow_scores) else None,
        }

    # ── Monotonicity / Sanity Checks ──

    def check_risk_monotonicity(
        self,
        profiles: List[PathRiskProfile],
    ) -> Dict[str, bool]:
        """Verify risk monotonicity: higher hazard → lower ranking.

        Returns per-pair monotonicity checks.
        """
        n = len(profiles)
        violations = 0
        total_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                pi, pj = profiles[i], profiles[j]
                si, sj = self.score(pi), self.score(pj)
                # If i has higher hazard, it should have higher (worse) score
                if pi.hazard_prob > pj.hazard_prob + 0.01:
                    if si < sj:  # violation: riskier path preferred
                        violations += 1
                    total_pairs += 1
                elif pj.hazard_prob > pi.hazard_prob + 0.01:
                    if sj < si:
                        violations += 1
                    total_pairs += 1

        return {
            "total_pairs": total_pairs,
            "violations": violations,
            "monotonic": violations == 0,
        }

    def check_necessity_sanity(
        self,
        profiles: List[PathRiskProfile],
    ) -> Dict[str, object]:
        """Verify necessity gate: when no safe alt, unc surcharge drops.

        Returns diagnostic indicating whether necessity correctly modulates
        uncertainty cost.
        """
        high_nec = [p for p in profiles if p.necessity > 0.8]
        low_nec = [p for p in profiles if p.necessity < 0.3]

        if high_nec and low_nec:
            avg_eff_high = float(np.mean([p.effective_unc_surcharge for p in high_nec]))
            avg_eff_low = float(np.mean([p.effective_unc_surcharge for p in low_nec]))
            return {
                "high_necessity_avg_surcharge": avg_eff_high,
                "low_necessity_avg_surcharge": avg_eff_low,
                "gate_working": avg_eff_high < avg_eff_low,
            }

        return {"gate_working": None, "note": "insufficient necessity range"}

    # ── Internal ──

    def _compute_necessity(
        self,
        path: List[Tuple[int, int]],
        passable: np.ndarray,
        t: int,
        t_max: int,
        goal: Tuple[int, int],
    ) -> float:
        """Compute route necessity using route_necessity module."""
        from .route_necessity import compute_necessity_for_path

        if len(path) < 2:
            return 0.0

        return compute_necessity_for_path(
            path, path[0], goal, passable, t, t_max,
            tau=self.cfg.necessity_tau)
