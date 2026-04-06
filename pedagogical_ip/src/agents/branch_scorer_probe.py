"""V5.6 — Branch Scorer Feasibility Probe.

Offline ranking: does a branch-level linear scorer outperform
pointwise aggregation for branch choice?

Q_branch(π) = v · [x_π, F(π), concept_scores] + b

NOT connected to planner — pure diagnostic.
"""

from __future__ import annotations

import numpy as np

from .branch_summary import SUMMARY_DIM, summarize_branch
from .branch_concepts import BranchConceptLibrary
from .familiarity import familiarity_score


# Input dim: summary(8) + familiarity(1) + concept_scores(3) = 12
SCORER_INPUT_DIM = SUMMARY_DIM + 1 + 3  # 12


def build_scorer_input(
    x_pi: np.ndarray,
    concepts: BranchConceptLibrary,
    obs_var: float = 0.01,
    tau: float = 1.0,
) -> np.ndarray:
    """Build scorer input vector from branch summary + concept features.

    Returns (SCORER_INPUT_DIM,) = [x_π, F(π), score_safe, score_risky, score_ambig]
    """
    F = familiarity_score(x_pi, concepts)
    scores = concepts.score_all(x_pi, obs_var=obs_var, tau=tau)

    return np.concatenate([
        x_pi,
        [F],
        [scores.get("safe_branch", -10.0)],
        [scores.get("risky_branch", -10.0)],
        [scores.get("ambiguous_branch", -10.0)],
    ])


class BranchScorerProbe:
    """Linear branch scorer for feasibility testing.

    Q(π) = v · input(π) + b

    Trained by SGD to predict oracle safety label (1=safe, 0=risky).
    """

    def __init__(self, lr: float = 0.1, l2: float = 0.01):
        self.v = np.zeros(SCORER_INPUT_DIM, dtype=np.float64)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2
        self.n_updates = 0

    def score(self, inp: np.ndarray) -> float:
        """Raw score: higher = safer."""
        return float(self.v @ inp + self.b)

    def predict_safe_prob(self, inp: np.ndarray) -> float:
        """Sigmoid probability that branch is safe."""
        logit = self.v @ inp + self.b
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10))))

    def rank_branches(self, inputs: list[np.ndarray]) -> list[int]:
        """Rank branches by safety score (highest first).

        Returns list of indices sorted by descending score.
        """
        scores = [self.score(inp) for inp in inputs]
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    def update(self, inp: np.ndarray, label: float, weight: float = 1.0):
        """SGD update: label=1 (safe), label=0 (risky).

        Binary cross-entropy loss + L2 regularization.
        """
        p = self.predict_safe_prob(inp)
        error = label - p

        grad_v = -error * inp * weight + self.l2 * self.v
        grad_b = -error * weight

        gn = float(np.linalg.norm(grad_v))
        if gn > 5.0:
            grad_v *= 5.0 / gn

        self.v -= self.lr * grad_v
        self.b -= self.lr * float(np.clip(grad_b, -5.0, 5.0))
        self.n_updates += 1

    def reset(self):
        self.v[:] = 0.0
        self.b = 0.0
        self.n_updates = 0


def pointwise_branch_score(
    cells: list[tuple[int, int]],
    belief_mean: np.ndarray,
    cost_risk_head,
) -> float:
    """Baseline: simple mean risk across branch cells.

    This is the comparison target for the branch scorer.
    Lower = safer.
    """
    if not cells:
        return 0.5
    risks = [cost_risk_head.predict_risk(belief_mean[r, c]) for r, c in cells]
    return float(np.mean(risks))
