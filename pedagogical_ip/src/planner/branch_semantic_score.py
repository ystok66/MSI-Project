"""V6.2 — Branch Semantic Score.

Computes branch-level scores from V5 modules for planner consumption.

Three modes:
  - scorer_only:  Q_branch(π)
  - concept_only: log P(safe|π) - log P(risky|π)
  - hybrid:       α·Q + β·S_concept - γ·C_novel
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..agents.branch_summary import summarize_branch, SUMMARY_DIM
from ..agents.branch_concepts import BranchConceptLibrary, log_inclusion_score
from ..agents.familiarity import familiarity_score
from ..agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input


@dataclass
class BranchScore:
    """All branch-level scores for a single candidate."""
    branch_id: int
    concept_score: float     # log P(safe) - log P(risky)
    scorer_score: float      # Q_branch(π)
    hybrid_score: float      # combined
    familiarity: float       # F(π)
    best_concept: str        # name of best-matching concept
    summary: np.ndarray      # raw 8D summary


def compute_branch_scores(
    branch_cells: list[tuple[int, int]],
    branch_id: int,
    belief_mean: np.ndarray,
    belief_var: np.ndarray | None,
    cost_risk_head,
    concept_lib: BranchConceptLibrary,
    scorer: Optional[BranchScorerProbe] = None,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.0,
    obs_var: float = 0.01,
    tau: float = 1.0,
) -> BranchScore:
    """Compute all semantic scores for a branch.

    Parameters
    ----------
    alpha : weight on scorer score
    beta : weight on concept score
    gamma : weight on novelty cost (0 = disabled)
    """
    # Summarize
    summary = summarize_branch(branch_cells, belief_mean, belief_var,
                                cost_risk_head)

    # Concept scores
    scores = concept_lib.score_all(summary, obs_var=obs_var, tau=tau)
    safe_score = scores.get("safe_branch", -100.0)
    risky_score = scores.get("risky_branch", -100.0)
    concept_score = safe_score - risky_score  # positive = more safe-like

    # Best concept
    best_name = max(scores, key=scores.get)

    # Familiarity
    fam = familiarity_score(summary, concept_lib, obs_var=obs_var, tau=tau)

    # Scorer score
    if scorer is not None:
        inp = build_scorer_input(summary, concept_lib, obs_var=obs_var, tau=tau)
        scorer_val = scorer.score(inp)
    else:
        scorer_val = 0.0

    # Novelty cost
    novelty_cost = max(0.0, 0.5 - fam) if gamma > 0 else 0.0

    # Hybrid
    hybrid = alpha * scorer_val + beta * concept_score - gamma * novelty_cost

    return BranchScore(
        branch_id=branch_id,
        concept_score=concept_score,
        scorer_score=scorer_val,
        hybrid_score=hybrid,
        familiarity=fam,
        best_concept=best_name,
        summary=summary,
    )
