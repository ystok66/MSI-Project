"""V6.3 — Branch Reranker.

Two-stage planner wrapper:
  Stage 1: rank candidate branches by J_hybrid = J_cell - λ_b · S_branch
  Stage 2: delegate within-branch path planning to existing A*

When λ_b = 0, degenerates to old planner behavior.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .branch_candidates import BranchCandidate, extract_elcb_branches, should_activate_branch_reranker
from .branch_semantic_score import compute_branch_scores, BranchScore
from ..agents.branch_concepts import BranchConceptLibrary
from ..agents.branch_scorer_probe import BranchScorerProbe
from ..agents.planner_astar import cell_cost_v2_latent


def branch_cell_cost(
    cells: list[tuple[int, int]],
    entry_gate: tuple[int, int],
    exit_gate: tuple[int, int],
    belief_mean: np.ndarray,
    cost_risk_head,
    passable: np.ndarray,
    belief_var: np.ndarray | None = None,
    necessity: float = 0.0,
) -> float:
    """Sum cell-level costs along a branch (entry + cells + exit).

    This is J_cell(π) — the baseline planner's aggregated path cost.
    """
    if belief_var is None:
        belief_var = np.full_like(belief_mean, 0.5)

    total = 0.0
    # Entry gate
    r, c = entry_gate
    total += cell_cost_v2_latent(r, c, belief_mean, cost_risk_head, passable,
                                  feature_belief_var=belief_var,
                                  route_necessity=necessity)
    # Branch cells
    for r, c in cells:
        total += cell_cost_v2_latent(r, c, belief_mean, cost_risk_head, passable,
                                      feature_belief_var=belief_var,
                                      route_necessity=necessity)
    # Exit gate
    r, c = exit_gate
    total += cell_cost_v2_latent(r, c, belief_mean, cost_risk_head, passable,
                                  feature_belief_var=belief_var,
                                  route_necessity=necessity)
    return total


def choose_branch(
    candidates: list[BranchCandidate],
    belief_mean: np.ndarray,
    belief_var: np.ndarray | None,
    cost_risk_head,
    passable: np.ndarray,
    concept_lib: BranchConceptLibrary,
    scorer: Optional[BranchScorerProbe] = None,
    *,
    lambda_b: float = 1.0,
    score_mode: str = "hybrid",
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.0,
    tie_rng: np.random.Generator | None = None,
) -> tuple[BranchCandidate, dict]:
    """Choose the best branch using hybrid scoring.

    J_hybrid(π) = J_cell(π) - λ_b · S_branch(π)

    Lower J_hybrid = better (cost minimization convention).

    Parameters
    ----------
    lambda_b : weight on branch semantic score. 0 = old planner.
    score_mode : 'hybrid', 'concept_only', 'scorer_only', 'pointwise_only'

    Returns
    -------
    (best_candidate, details_dict)
    """
    results = []

    for cand in candidates:
        # Cell-level cost
        j_cell = branch_cell_cost(
            cand.cells, cand.entry_gate, cand.exit_gate,
            belief_mean, cost_risk_head, passable, belief_var)

        # Semantic score
        if score_mode == "pointwise_only" or lambda_b == 0:
            s_branch = 0.0
            bs = None
        else:
            bs = compute_branch_scores(
                cand.cells, cand.branch_id,
                belief_mean, belief_var, cost_risk_head,
                concept_lib, scorer,
                alpha=alpha, beta=beta, gamma=gamma)

            if score_mode == "concept_only":
                s_branch = bs.concept_score
            elif score_mode == "scorer_only":
                s_branch = bs.scorer_score
            else:  # hybrid
                s_branch = bs.hybrid_score

        j_hybrid = j_cell - lambda_b * s_branch

        results.append({
            "candidate": cand,
            "j_cell": j_cell,
            "s_branch": s_branch,
            "j_hybrid": j_hybrid,
            "branch_score": bs,
        })

    # Sort by j_hybrid (lower = better)
    results.sort(key=lambda x: x["j_hybrid"])

    # Tie-breaking
    if len(results) >= 2 and abs(results[0]["j_hybrid"] - results[1]["j_hybrid"]) < 1e-4:
        rng = tie_rng or np.random.default_rng()
        if rng.integers(0, 2) == 1:
            results[0], results[1] = results[1], results[0]

    best = results[0]
    return best["candidate"], {
        "j_cell_chosen": best["j_cell"],
        "s_branch_chosen": best["s_branch"],
        "j_hybrid_chosen": best["j_hybrid"],
        "all_results": results,
    }
