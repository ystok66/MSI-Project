"""RobotBeliefOverAgent — Nested ToM belief tracker.

Robot-side belief over the agent's belief state.
This is the core Target B object.

B_t^{R→A}(b^A) ≈ Σ_k w_k · δ(b^A - b_k)

Minimal parametric implementation (v1):
  - mean belief state + diagonal confidence
  - optional particle bank in shadow mode
  - update from observed actions via ActionPredictor likelihood

POMDP-interface shell (Task 3 Phase A).
Default-off shadow mode. Does not change any existing behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np

from ..agents.agent_belief_state import AgentBelief
from ..agents.stochastic_agent_policy import BranchAttributes


@dataclass
class BeliefParticle:
    """Single hypothesis about agent's belief state."""
    agent_belief: AgentBelief
    weight: float = 1.0
    log_weight: float = 0.0


@dataclass
class RobotBeliefState:
    """Robot's summary of its belief over the agent's internal state."""
    # Parametric summary (always available)
    mean_m: Dict[str, float] = field(default_factory=lambda: {
        "tau": 0.3, "nu": 0.1, "gamma_gen": 0.0,
        "gamma_spec": 0.0, "kappa": 0.3,
    })
    confidence: Dict[str, float] = field(default_factory=lambda: {
        "tau": 0.2, "nu": 0.2, "gamma_gen": 0.2,
    })
    theta_posterior: Dict[str, float] = field(default_factory=lambda: {
        "safe": 0.5, "shiny": 0.3, "risky": 0.1, "neutral": 0.1,
    })

    # Entropy of belief (lower = more certain)
    entropy: float = 1.0

    # Diagnostics
    n_updates: int = 0
    cumulative_nll: float = 0.0


class RobotBeliefOverAgent:
    """Nested-ToM belief tracker (shadow mode).

    Maintains robot's belief about what the agent believes,
    updated from observed actions via inverse planning.

    Usage:
        rboa = RobotBeliefOverAgent(predictor)
        rboa.update_from_action(world_state, branches, observed_action)
        state = rboa.get_state()
    """

    def __init__(self, action_predictor=None,
                 n_particles: int = 0,
                 use_particles: bool = False):
        """
        Args:
            action_predictor: ActionPredictor for computing P(a|s,b)
            n_particles: number of particles (0 = parametric only)
            use_particles: whether to maintain particle bank
        """
        self._predictor = action_predictor
        self._state = RobotBeliefState()
        self._particles: List[BeliefParticle] = []
        self._use_particles = use_particles and n_particles > 0
        self._n_particles = n_particles
        self._shadow_log: List[Dict] = []

    def predict_after_observation(self, world_state, observation) -> AgentBelief:
        """Predict agent's updated belief after receiving an observation.

        v1: returns current mean belief (no real belief propagation yet).
        """
        return AgentBelief(
            m_state=dict(self._state.mean_m),
            theta=self._most_likely_theta(),
        )

    def update_from_action(self, world_state, branches: list[BranchAttributes],
                           observed_action: int,
                           agent_belief_hint: Optional[AgentBelief] = None):
        """Update belief from observed agent action via inverse planning.

        w_k ∝ w_{k-1} · P(a_obs | s_world, b_k^A)

        v1 (parametric): updates theta_posterior via action likelihood.
        """
        if self._predictor is None:
            return

        # Compute likelihood for each theta hypothesis
        for theta in self._state.theta_posterior:
            ab = AgentBelief(
                m_state=dict(self._state.mean_m),
                theta=theta,
            )
            if agent_belief_hint is not None:
                ab.belief_mean = agent_belief_hint.belief_mean
                ab.belief_var = agent_belief_hint.belief_var

            ll = self._predictor.score(world_state, ab, branches,
                                       observed_action)
            # Bayesian update: posterior ∝ prior · likelihood
            self._state.theta_posterior[theta] *= np.exp(ll)

        # Normalize
        total = sum(self._state.theta_posterior.values())
        if total > 0:
            for k in self._state.theta_posterior:
                self._state.theta_posterior[k] /= total

        # Update diagnostics
        self._state.n_updates += 1
        self._state.entropy = self._compute_entropy()

        # Shadow log
        self._shadow_log.append({
            "step": self._state.n_updates,
            "theta_post": dict(self._state.theta_posterior),
            "entropy": self._state.entropy,
            "observed_action": observed_action,
        })

    def update_from_observer(self, observer_estimate: Dict[str, float],
                              observer_confidence: Dict[str, float]):
        """Sync parametric belief from observer's current estimate.

        This bridges the existing observer system with the new interface.
        """
        for k, v in observer_estimate.items():
            if k in self._state.mean_m:
                self._state.mean_m[k] = v
        for k, v in observer_confidence.items():
            if k in self._state.confidence:
                self._state.confidence[k] = v

    def get_state(self) -> RobotBeliefState:
        """Return current belief summary."""
        return self._state

    def mean_belief(self) -> Dict[str, float]:
        """Return mean estimated agent state."""
        return dict(self._state.mean_m)

    def confidence(self) -> Dict[str, float]:
        """Return per-dimension confidence."""
        return dict(self._state.confidence)

    def most_likely_theta(self) -> str:
        return self._most_likely_theta()

    def get_shadow_log(self) -> List[Dict]:
        """Return shadow-mode diagnostic log."""
        return list(self._shadow_log)

    def reset(self):
        """Reset to prior."""
        self._state = RobotBeliefState()
        self._particles = []
        self._shadow_log = []

    # --- Internal ---

    def _most_likely_theta(self) -> str:
        return max(self._state.theta_posterior,
                   key=self._state.theta_posterior.get)

    def _compute_entropy(self) -> float:
        probs = np.array(list(self._state.theta_posterior.values()))
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
        return -float(np.sum(probs * np.log(probs)))
