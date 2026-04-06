"""Compositional Goal Hypotheses — Goal hypothesis space for CGC-v2.

Defines atomic and composite goal structures from CGC-v2,
with goal-conditioned reward weights for action likelihood computation.

Uses CGC-v2's atomic goals: {collect_red, avoid_blue, use_safe, reach_fast}
and their valid conjunctions.

Does NOT modify any existing module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, FrozenSet
import numpy as np


# ═══ Atomic goals (from cgc_v2_family.py) ═══

ATOMIC_GOALS = ("collect_red", "avoid_blue", "use_safe", "reach_fast")

# Goal reward weights: how each atomic goal values branch attributes
# Columns: [safety_bonus, temptation_bonus, texture_novelty, shortcut_bonus]
ATOMIC_GOAL_WEIGHTS = {
    "collect_red":  np.array([0.0,  2.5, 0.5, 0.0]),
    "avoid_blue":   np.array([2.0, -1.0, 0.0, 0.0]),
    "use_safe":     np.array([3.0, -0.5, 0.0, 0.0]),
    "reach_fast":   np.array([0.0,  0.0, 0.0, 3.0]),
}


# ═══ Composite goals ═══

VALID_COMPOSITES = (
    ("collect_red", "avoid_blue"),
    ("collect_red", "use_safe"),
    ("avoid_blue", "use_safe"),
    ("reach_fast", "avoid_blue"),
)


@dataclass(frozen=True)
class GoalHypothesis:
    """A single goal hypothesis — atomic or composite."""
    components: Tuple[str, ...]   # tuple of atomic goal names
    label: str = ""               # human-readable label

    def __post_init__(self):
        if not self.label:
            object.__setattr__(self, 'label',
                               "+".join(self.components))

    @property
    def is_composite(self) -> bool:
        return len(self.components) > 1

    @property
    def reward_weights(self) -> np.ndarray:
        """Combined reward weight vector for this goal.

        Composite goals: average of component weights (normalized).
        """
        ws = [ATOMIC_GOAL_WEIGHTS[c] for c in self.components]
        combined = np.mean(ws, axis=0)
        return combined


class GoalHypothesisSpace:
    """Manages the space of valid goal hypotheses.

    Provides:
      - list of all hypotheses (atomic + composite)
      - goal-conditioned utility computation
      - reward weight lookup

    Usage:
        ghs = GoalHypothesisSpace()
        for gh in ghs.hypotheses:
            print(gh.label, gh.reward_weights)
    """

    def __init__(self,
                 include_composites: bool = True,
                 custom_composites: Optional[List[Tuple[str, ...]]] = None):
        """
        Args:
            include_composites: whether to include composite goals
            custom_composites: override default valid composites
        """
        self._hypotheses: List[GoalHypothesis] = []

        # Add atomic goals
        for a in ATOMIC_GOALS:
            self._hypotheses.append(GoalHypothesis(components=(a,)))

        # Add composite goals
        if include_composites:
            composites = custom_composites or VALID_COMPOSITES
            for comp in composites:
                self._hypotheses.append(GoalHypothesis(components=comp))

        # Build lookup
        self._label_to_idx = {h.label: i for i, h in enumerate(self._hypotheses)}

    @property
    def hypotheses(self) -> List[GoalHypothesis]:
        return list(self._hypotheses)

    @property
    def labels(self) -> List[str]:
        return [h.label for h in self._hypotheses]

    @property
    def n_goals(self) -> int:
        return len(self._hypotheses)

    @property
    def atomic_goals(self) -> List[GoalHypothesis]:
        return [h for h in self._hypotheses if not h.is_composite]

    @property
    def composite_goals(self) -> List[GoalHypothesis]:
        return [h for h in self._hypotheses if h.is_composite]

    def get(self, label: str) -> GoalHypothesis:
        """Get hypothesis by label."""
        idx = self._label_to_idx.get(label)
        if idx is None:
            raise KeyError(f"Unknown goal: {label}")
        return self._hypotheses[idx]

    def index(self, label: str) -> int:
        """Get index of hypothesis by label."""
        idx = self._label_to_idx.get(label)
        if idx is None:
            raise KeyError(f"Unknown goal: {label}")
        return idx

    def goal_conditioned_utility(self,
                                  branch_attrs,
                                  goal: GoalHypothesis,
                                  theta: str,
                                  params=None) -> float:
        """U(branch | g, θ) = R_goal(branch; g) + λ·R_pref(branch; θ) - J_risk.

        Uses goal's reward weights for R_goal, and existing PREF_REWARD for R_pref.
        """
        from ..agents.stochastic_agent_policy import PREF_REWARD, AgentPolicyParams
        if params is None:
            params = AgentPolicyParams()

        x = branch_attrs.to_array()
        r_goal = float(np.dot(goal.reward_weights, x))
        r_pref = float(np.dot(PREF_REWARD[theta], x))
        j_risk = branch_attrs.risk_penalty

        return r_goal + params.lambda_theta * r_pref - j_risk

    def compute_choice_probs(self,
                              branches: list,
                              goal: GoalHypothesis,
                              theta: str,
                              params=None) -> np.ndarray:
        """P(branch | g, θ) via softmax + lapse."""
        from ..agents.stochastic_agent_policy import AgentPolicyParams
        if params is None:
            params = AgentPolicyParams()

        utilities = np.array([
            self.goal_conditioned_utility(b, goal, theta, params)
            for b in branches
        ])
        scaled = params.beta * utilities
        scaled -= np.max(scaled)
        exp_u = np.exp(scaled)
        softmax_p = exp_u / (exp_u.sum() + 1e-10)
        n = len(branches)
        uniform = np.ones(n) / n
        return (1 - params.epsilon) * softmax_p + params.epsilon * uniform


# ═══ Default instance ═══
DEFAULT_GOAL_SPACE = GoalHypothesisSpace(include_composites=True)
ATOMIC_ONLY_GOAL_SPACE = GoalHypothesisSpace(include_composites=False)
