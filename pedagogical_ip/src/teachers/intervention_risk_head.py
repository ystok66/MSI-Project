"""Intervention Risk Head — Timing-aware risk scores from POMDP interfaces.

Consumes WorldState + RobotBeliefOverAgent + ActionPredictor to produce:
  - p_timeout: probability agent will miss deadline
  - p_blind: probability agent commits before disambiguating evidence
  - U_int: combined intervention urgency

Shadow-mode only. Does NOT affect canonical tutor decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

from ..agents.world_state import WorldState
from ..agents.agent_belief_state import AgentBelief
from ..agents.stochastic_agent_policy import BranchAttributes
from .action_predictor import ActionPredictor
from .robot_belief_over_agent import RobotBeliefOverAgent


@dataclass
class InterventionRisk:
    """Intervention timing risk scores."""
    p_timeout: float = 0.0    # P(finish_time > T_remain)
    p_blind: float = 0.0      # P(commit before reveal)
    u_int: float = 0.0        # combined urgency
    lead_time: int = 0        # steps before failure
    flagged: bool = False      # above threshold?


class InterventionRiskHead:
    """Timing-aware intervention risk estimator.

    Uses the POMDP interface to estimate:
    1. Timeout risk: will the agent run out of time?
    2. Blind-commit risk: will the agent commit before seeing evidence?

    These are computed from the action predictor + scenario geometry,
    not from oracle knowledge.
    """

    def __init__(self,
                 lambda_time: float = 1.0,
                 lambda_blind: float = 1.0,
                 tau_time: float = 3.0,     # softness for timeout sigmoid
                 tau_blind: float = 1.5,    # softness for blind-commit sigmoid
                 threshold: float = 0.5):   # urgency threshold for flagging
        self.lambda_time = lambda_time
        self.lambda_blind = lambda_blind
        self.tau_time = tau_time
        self.tau_blind = tau_blind
        self.threshold = threshold
        self._history = []

    def predict(self,
                world_state: Optional[WorldState],
                robot_belief: RobotBeliefOverAgent,
                action_predictor: ActionPredictor,
                branches: list[BranchAttributes],
                agent_belief: Optional[AgentBelief] = None,
                d_commit: int = 3,
                d_reveal: int = 2,
                path_length_estimate: int = 10,
                ) -> InterventionRisk:
        """Compute intervention risk scores.

        Args:
            world_state: true environment state (for t_remain)
            robot_belief: robot's belief over agent
            action_predictor: P(a|s,b)
            branches: current action options
            agent_belief: optional agent belief for prediction
            d_commit: depth to irreversible commitment
            d_reveal: depth to informative evidence reveal
            path_length_estimate: estimated plan length
        """
        # --- Timeout risk ---
        t_remain = world_state.remaining_budget if world_state else 50
        # Soft sigmoid: σ((L_hat - T_remain) / τ_time)
        timeout_gap = path_length_estimate - t_remain
        p_timeout = self._sigmoid(timeout_gap / max(self.tau_time, 0.1))

        # --- Blind-commit risk ---
        # P(agent commits before reveal) using action distribution
        if agent_belief is not None:
            dist = action_predictor.predict(world_state, agent_belief, branches)
        else:
            # Use robot's mean belief
            mb = robot_belief.mean_belief()
            theta_map = robot_belief.most_likely_theta()
            ab = AgentBelief(m_state=mb, theta=theta_map)
            dist = action_predictor.predict(world_state, ab, branches)

        # Per-action blind-commit: commit before reveal
        r_blind = 0.0
        for i, p_a in enumerate(dist.probs):
            # Action i: check if d_commit < d_reveal
            # For now: action 1 (risky) has d_commit < d_reveal more often
            # We use branch risk_penalty as proxy for commit urgency
            risk_p = branches[i].risk_penalty if i < len(branches) else 0.0
            # Higher risk → more likely to be blind commit
            commit_gap = d_reveal - d_commit  # positive = reveal comes later
            blind_score = self._sigmoid(
                (commit_gap + risk_p * 2) / max(self.tau_blind, 0.1))
            r_blind += p_a * blind_score
        p_blind = float(np.clip(r_blind, 0, 1))

        # --- Combined urgency ---
        u_int = self.lambda_time * p_timeout + self.lambda_blind * p_blind
        flagged = u_int > self.threshold

        lead_time = max(t_remain - path_length_estimate, 0) if t_remain > 0 else 0

        result = InterventionRisk(
            p_timeout=round(p_timeout, 4),
            p_blind=round(p_blind, 4),
            u_int=round(u_int, 4),
            lead_time=lead_time,
            flagged=flagged,
        )
        self._history.append(result)
        return result

    def get_history(self):
        return list(self._history)

    def reset(self):
        self._history = []

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))
