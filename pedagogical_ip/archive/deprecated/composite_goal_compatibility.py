"""Composite Goal Compatibility — Structural prior for composite goal separation.

Provides two mechanisms to help the joint posterior discriminate
composite goals from nearby atomic explanations:

1. Subgoal progress tracking: C_t(g) = mean progress across subgoals
2. Complexity penalty: π_0(g) ∝ exp(-λ_comp · |g|)
3. Redundancy penalty: discounts composites whose subgoals predict
   nearly identical action traces

Does NOT modify any existing module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np

from ..agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from .compositional_goal_hypotheses import (
    GoalHypothesisSpace, GoalHypothesis, DEFAULT_GOAL_SPACE,
)


@dataclass
class CompatibilityConfig:
    """Config for composite goal compatibility scoring."""
    beta_compat: float = 0.5       # strength of compatibility term
    lambda_comp: float = 0.3       # complexity penalty per extra subgoal
    lambda_redund: float = 0.2     # redundancy penalty strength
    min_divergence: float = 0.05   # min KL for non-redundant subgoals


class CompositeGoalCompatibility:
    """Computes structural compatibility scores for composite goals.

    Two signals:
    1. Subgoal evidence accumulation: tracks whether observed actions
       are consistent with EACH subgoal independently
    2. Redundancy detection: penalizes composites whose components
       predict nearly identical action distributions

    Usage:
        cgc = CompositeGoalCompatibility(goal_space)
        cgc.observe(branches, observed_action, params)
        scores = cgc.compatibility_scores()  # {goal_label: float}
    """

    def __init__(self,
                 goal_space: Optional[GoalHypothesisSpace] = None,
                 config: Optional[CompatibilityConfig] = None,
                 params: Optional[AgentPolicyParams] = None):
        self._goal_space = goal_space or DEFAULT_GOAL_SPACE
        self._config = config or CompatibilityConfig()
        self._params = params or AgentPolicyParams()

        # Per-atomic-goal evidence accumulator (log-likelihood sum)
        self._atomic_evidence: Dict[str, float] = {}
        for gh in self._goal_space.atomic_goals:
            self._atomic_evidence[gh.label] = 0.0

        self._n_obs = 0

    def observe(self,
                branches: list[BranchAttributes],
                observed_action: int,
                theta: str = "safe"):
        """Accumulate evidence for each atomic subgoal."""
        for gh in self._goal_space.atomic_goals:
            probs = self._goal_space.compute_choice_probs(
                branches, gh, theta, self._params)
            ll = np.log(max(probs[observed_action], 1e-15))
            self._atomic_evidence[gh.label] += ll
        self._n_obs += 1

    def subgoal_progress(self, goal: GoalHypothesis) -> float:
        """Normalized progress score for a goal.

        For atomics: direct evidence
        For composites: mean evidence across components
        """
        if self._n_obs == 0:
            return 0.0
        evidences = []
        for comp in goal.components:
            ev = self._atomic_evidence.get(comp, 0.0)
            # Normalize by observation count (mean log-likelihood)
            evidences.append(ev / max(self._n_obs, 1))
        return float(np.mean(evidences))

    def redundancy_score(self,
                          goal: GoalHypothesis,
                          branches: list[BranchAttributes],
                          theta: str = "safe") -> float:
        """Measure how redundant a composite goal's subgoals are.

        High redundancy = subgoals predict nearly identical actions.
        Returns value in [0, 1], where 1 = perfectly redundant.
        """
        if not goal.is_composite or len(goal.components) < 2:
            return 0.0

        # Compute action distributions for each component
        dists = []
        for comp in goal.components:
            comp_gh = self._goal_space.get(comp)
            probs = self._goal_space.compute_choice_probs(
                branches, comp_gh, theta, self._params)
            dists.append(probs)

        # Pairwise KL divergence
        total_kl = 0.0
        n_pairs = 0
        for i in range(len(dists)):
            for j in range(i + 1, len(dists)):
                p, q = dists[i], dists[j]
                kl = float(np.sum(p * np.log((p + 1e-10) / (q + 1e-10))))
                total_kl += abs(kl)
                n_pairs += 1

        mean_kl = total_kl / max(n_pairs, 1)
        # Low KL = high redundancy
        redundancy = np.exp(-mean_kl / max(self._config.min_divergence, 0.01))
        return float(redundancy)

    def complexity_penalty(self, goal: GoalHypothesis) -> float:
        """Penalty for composite goals: exp(-λ_comp · |g|).

        Atomic goals get penalty 0, composites get > 0.
        """
        n_components = len(goal.components)
        if n_components <= 1:
            return 0.0
        return self._config.lambda_comp * (n_components - 1)

    def compatibility_score(self,
                             goal: GoalHypothesis,
                             branches: Optional[list[BranchAttributes]] = None,
                             theta: str = "safe") -> float:
        """Full compatibility score C(g) for use in posterior.

        C(g) = progress(g) - λ_redund · redundancy(g) - λ_comp · (|g|-1)

        Higher is better for the goal hypothesis.
        """
        progress = self.subgoal_progress(goal)
        redund = 0.0
        if branches is not None and goal.is_composite:
            redund = self.redundancy_score(goal, branches, theta)
        penalty = self.complexity_penalty(goal)

        return (progress
                - self._config.lambda_redund * redund
                - penalty)

    def compatibility_scores(self,
                              branches: Optional[list[BranchAttributes]] = None,
                              theta: str = "safe") -> Dict[str, float]:
        """Compute compatibility scores for all goals."""
        scores = {}
        for gh in self._goal_space.hypotheses:
            scores[gh.label] = self.compatibility_score(gh, branches, theta)
        return scores

    def log_compatibility_bonus(self,
                                 goal: GoalHypothesis,
                                 branches: Optional[list[BranchAttributes]] = None,
                                 theta: str = "safe") -> float:
        """Log-space bonus for posterior: β_C · C(g).

        To be added to log q(g,θ) during update.
        """
        c = self.compatibility_score(goal, branches, theta)
        return self._config.beta_compat * c

    def subgoal_marginals(self,
                           posterior_marginal_goal: Dict[str, float]) -> Dict[str, float]:
        """Compute per-atomic-subgoal marginal from goal posterior.

        q(u) = Σ_{g ∋ u} q(g)

        Useful for reporting even when exact composite label is hard to recover.
        """
        marginals = {}
        for gh in self._goal_space.atomic_goals:
            u = gh.label
            total = 0.0
            for other_gh in self._goal_space.hypotheses:
                if u in other_gh.components:
                    total += posterior_marginal_goal.get(other_gh.label, 0.0)
            marginals[u] = total
        return marginals

    def reset(self):
        """Reset accumulated evidence."""
        for k in self._atomic_evidence:
            self._atomic_evidence[k] = 0.0
        self._n_obs = 0
