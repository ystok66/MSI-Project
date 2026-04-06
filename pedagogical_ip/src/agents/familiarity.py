"""V5.3 — Familiarity / Novelty Score.

Computes how familiar a branch summary is relative to the concept library.
Two roles:
  - Agent side:  low familiarity → conservative or explore
  - Tutor side:  low familiarity + feasible → teaching WARN
                 high familiarity + danger → rescue intervention

Definitions
-----------
  F(π) = Σ_k 1[score(π,k) > τ_m] · log(1 + κ_k)

  novel(π) = 1[F(π) < τ_novel]
"""

from __future__ import annotations

import numpy as np

from .branch_concepts import BranchConceptLibrary, log_inclusion_score


def familiarity_score(
    x_pi: np.ndarray,
    concepts: BranchConceptLibrary,
    match_threshold: float = -5.0,
    obs_var: float = 0.01,
    tau: float = 1.0,
) -> float:
    """Compute familiarity score for a branch summary.

    F(π) = Σ_k 1[score(π,k) > τ_m] · log(1 + κ_k)

    Higher = more familiar (agent has seen similar branches before).
    Low values indicate genuinely novel patterns.

    Parameters
    ----------
    x_pi : (SUMMARY_DIM,) branch summary vector
    concepts : BranchConceptLibrary
    match_threshold : float
        Minimum log-inclusion score to count as a match.
    obs_var, tau : float
        KL scoring parameters.

    Returns
    -------
    F : float >= 0
    """
    F = 0.0
    for name, concept in concepts.concepts.items():
        score = log_inclusion_score(x_pi, concept, obs_var=obs_var, tau=tau)
        if score > match_threshold:
            F += np.log1p(concept.kappa)
    return float(F)


def is_novel(
    x_pi: np.ndarray,
    concepts: BranchConceptLibrary,
    novelty_threshold: float = 0.5,
    **kwargs,
) -> bool:
    """Check if branch is genuinely novel (no mature concept matches well).

    Parameters
    ----------
    novelty_threshold : float
        Below this F score, the branch is considered novel.
    """
    F = familiarity_score(x_pi, concepts, **kwargs)
    return F < novelty_threshold


def teaching_priority(
    x_pi: np.ndarray,
    concepts: BranchConceptLibrary,
    risk_estimate: float,
    **kwargs,
) -> dict:
    """Compute teaching priority for tutor decision-making.

    Returns a dict with:
      - familiarity: F(π)
      - is_novel: bool
      - risk: float
      - teaching_mode: 'teach' | 'rescue' | 'wait'

    Logic:
      - Low familiarity + feasible risk → TEACH (WARN to build understanding)
      - High familiarity + high risk    → RESCUE (direct intervention)
      - High familiarity + low risk     → WAIT (already knows, no danger)
      - Low familiarity + fatal risk    → RESCUE (can't afford to teach now)
    """
    F = familiarity_score(x_pi, concepts, **kwargs)
    novel = F < 0.5
    high_risk = risk_estimate > 0.4
    fatal_risk = risk_estimate > 0.7

    if fatal_risk:
        mode = "rescue"
    elif novel and not high_risk:
        mode = "teach"
    elif novel and high_risk:
        mode = "rescue"  # too dangerous to teach right now
    elif not novel and high_risk:
        mode = "rescue"  # knows the concept but still dangerous
    else:
        mode = "wait"    # familiar and safe

    return {
        "familiarity": F,
        "is_novel": novel,
        "risk": risk_estimate,
        "teaching_mode": mode,
    }
