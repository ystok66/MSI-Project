"""ActionPredictor — Clean predictive action model.

Maps (WorldState, AgentBelief) → P(a | s_world, b_A).
Wraps the current bounded-rational utility machinery.

P(a | s, b) = (1-ε) · softmax(β · U(a; φ(s, b))) + ε · Uniform

This makes inverse planning explicit:
  log P(a_obs | s, b) = standard action-likelihood for belief update.

POMDP-interface shell (Task 3 Phase A).
Does not change any existing behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

from ..agents.stochastic_agent_policy import (
    compute_choice_probs, BranchAttributes, AgentPolicyParams,
)


@dataclass
class ActionDistribution:
    """Predicted action distribution over branches."""
    probs: np.ndarray           # (n_actions,)
    log_probs: np.ndarray       # (n_actions,)
    utilities: np.ndarray       # (n_actions,) — pre-softmax utilities
    branch_labels: tuple = ()   # optional labels for each action

    @property
    def entropy(self) -> float:
        p = self.probs[self.probs > 0]
        return -float(np.sum(p * np.log(p)))

    @property
    def top1_prob(self) -> float:
        return float(np.max(self.probs))

    @property
    def top1_idx(self) -> int:
        return int(np.argmax(self.probs))


class ActionPredictor:
    """Clean predictive action model.

    Wraps the current bounded-rational utility + softmax + lapse machinery.
    Provides:
      - predict(): full action distribution
      - score(): log-likelihood of observed action (for inverse planning)
      - nll(): negative log-likelihood
    """

    def __init__(self, params: Optional[AgentPolicyParams] = None):
        self.params = params or AgentPolicyParams()
        self._call_count = 0
        self._nll_history = []

    def predict(self, world_state, agent_belief,
                branches: list[BranchAttributes],
                context: Optional[Dict] = None) -> ActionDistribution:
        """Predict P(a | s_world, b_A) over branches.

        Args:
            world_state: WorldState (true env, used for feature extraction)
            agent_belief: AgentBelief (agent's uncertain model)
            branches: list of BranchAttributes for each action
            context: optional dict with extra features

        Returns:
            ActionDistribution with probs, log_probs, utilities
        """
        theta = agent_belief.theta if agent_belief is not None else "safe"
        probs = compute_choice_probs(branches, theta, self.params)

        # Compute utilities for diagnostics
        from ..agents.stochastic_agent_policy import compute_agent_utility
        utilities = np.array([
            compute_agent_utility(b, theta, self.params) for b in branches
        ])

        log_probs = np.log(np.clip(probs, 1e-10, 1.0))
        self._call_count += 1

        return ActionDistribution(
            probs=probs,
            log_probs=log_probs,
            utilities=utilities,
        )

    def score(self, world_state, agent_belief,
              branches: list[BranchAttributes],
              observed_action: int) -> float:
        """Log P(a_obs | s_world, b_A) — action likelihood for belief update.

        This is the core inverse-planning signal.
        """
        dist = self.predict(world_state, agent_belief, branches)
        ll = float(dist.log_probs[observed_action])
        self._nll_history.append(-ll)
        return ll

    def nll(self, world_state, agent_belief,
            branches: list[BranchAttributes],
            observed_action: int) -> float:
        """Negative log-likelihood of observed action."""
        return -self.score(world_state, agent_belief, branches,
                          observed_action)

    @property
    def mean_nll(self) -> float:
        if not self._nll_history:
            return 0.0
        return float(np.mean(self._nll_history))

    def reset_stats(self):
        self._call_count = 0
        self._nll_history = []
