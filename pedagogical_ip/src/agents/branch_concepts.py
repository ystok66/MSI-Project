"""V5.2 — Gaussian Branch Concepts.

Each concept is a diagonal Gaussian over the branch summary space (8D).
Concepts: safe_branch, risky_branch, ambiguous_branch.

Semantic compatibility is measured by KL-based log-inclusion score:
    score(π, k) = -1/τ · KL(A_π || C_k)

Concepts are updated incrementally using the standard Bayesian
sufficient-statistics update (Welford-style):
    κ' = κ + w
    μ' = μ + w/κ' · (x - μ)
    M₂'= M₂ + w · (x - μ)(x - μ')
    σ² = M₂'/κ' + var_floor
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .branch_summary import SUMMARY_DIM


@dataclass
class GaussianConcept:
    """A single Gaussian concept over branch summary space."""
    name: str
    mu: np.ndarray            # (SUMMARY_DIM,) mean
    var: np.ndarray           # (SUMMARY_DIM,) diagonal variance
    kappa: float = 0.0        # maturity (total weight seen)
    _m2: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self._m2 is None:
            self._m2 = self.var.copy() * max(self.kappa, 1.0)


def _kl_diag_gaussian(mu_a: np.ndarray, var_a: np.ndarray,
                       mu_b: np.ndarray, var_b: np.ndarray) -> float:
    """KL(A || B) for diagonal Gaussians.

    KL = 0.5 * Σ_d [var_a/var_b + (μ_b-μ_a)²/var_b - 1 + log(var_b/var_a)]
    """
    d = len(mu_a)
    ratio = var_a / (var_b + 1e-10)
    diff_sq = (mu_b - mu_a) ** 2 / (var_b + 1e-10)
    kl = 0.5 * np.sum(ratio + diff_sq - 1.0 + np.log(1.0 / (ratio + 1e-10)))
    return max(float(kl), 0.0)


def log_inclusion_score(x_pi: np.ndarray, concept: GaussianConcept,
                        obs_var: float = 0.01, tau: float = 1.0) -> float:
    """Log-inclusion score: how well does branch summary x_π fit concept.

    score = -1/τ · KL(A_π || C_k)
    where A_π = N(x_π, obs_var · I)  (observation distribution)
          C_k = N(μ_k, diag(σ²_k))  (concept distribution)

    Higher = better fit.
    """
    obs_variance = np.full(SUMMARY_DIM, obs_var)
    kl = _kl_diag_gaussian(x_pi, obs_variance, concept.mu, concept.var)
    return -kl / tau


def update_concept(concept: GaussianConcept, x_pi: np.ndarray,
                   weight: float = 1.0, var_floor: float = 0.01) -> None:
    """Incremental Bayesian update of concept from branch observation.

    Uses Welford-Knuth online algorithm with weighted updates.
    """
    concept.kappa += weight
    delta = x_pi - concept.mu
    concept.mu = concept.mu + (weight / concept.kappa) * delta
    delta2 = x_pi - concept.mu
    concept._m2 = concept._m2 + weight * delta * delta2
    concept.var = concept._m2 / max(concept.kappa, 1.0) + var_floor


class BranchConceptLibrary:
    """Library of Gaussian branch concepts.

    Default concepts:
    - safe_branch:      low risk, low uncertainty
    - risky_branch:     high risk, any uncertainty
    - ambiguous_branch: medium risk, high uncertainty
    """

    def __init__(self, dim: int = SUMMARY_DIM):
        self.dim = dim
        # Broad initial priors
        high_var = np.full(dim, 1.0)

        self.concepts = {
            "safe_branch": GaussianConcept(
                name="safe_branch",
                mu=np.array([0.1, 0.15, 1.0, 0.1, 0.1, 0.3, 0.02, 0.5]),
                var=high_var.copy(),
                kappa=1.0,
            ),
            "risky_branch": GaussianConcept(
                name="risky_branch",
                mu=np.array([0.5, 0.7, 1.0, 0.1, 0.1, 0.7, 0.05, 0.5]),
                var=high_var.copy(),
                kappa=1.0,
            ),
            "ambiguous_branch": GaussianConcept(
                name="ambiguous_branch",
                mu=np.array([0.3, 0.4, 1.0, 0.5, 0.5, 0.5, 0.10, 0.5]),
                var=high_var.copy(),
                kappa=1.0,
            ),
        }

    def score_all(self, x_pi: np.ndarray, obs_var: float = 0.01,
                  tau: float = 1.0) -> dict[str, float]:
        """Compute log-inclusion scores for all concepts."""
        return {
            name: log_inclusion_score(x_pi, c, obs_var=obs_var, tau=tau)
            for name, c in self.concepts.items()
        }

    def best_concept(self, x_pi: np.ndarray, **kwargs) -> tuple[str, float]:
        """Return (name, score) of the best-matching concept."""
        scores = self.score_all(x_pi, **kwargs)
        best = max(scores, key=scores.get)
        return best, scores[best]

    def update(self, concept_name: str, x_pi: np.ndarray,
               weight: float = 1.0, var_floor: float = 0.01) -> None:
        """Update a specific concept from verified branch outcome."""
        if concept_name in self.concepts:
            update_concept(self.concepts[concept_name], x_pi,
                          weight=weight, var_floor=var_floor)

    def get_maturity(self, concept_name: str) -> float:
        """Return concept maturity (κ)."""
        if concept_name in self.concepts:
            return self.concepts[concept_name].kappa
        return 0.0

    def reset(self):
        """Reset all concepts to broad priors."""
        self.__init__(dim=self.dim)
