"""Continuous Reward Weight Shadow — Step 5B.

Shadow predictor that augments the discrete reward table with small
continuous residuals. Does NOT modify canonical posterior or discrete θ.

Three modes (B1, B2, B3):
  B1: 1D scale residual — w_shadow = (1+α)·w_disc
  B2: 2D interpretable residual — w_shadow = w_disc + α·v_risk + β·v_tempt
  B3: 4D Gaussian residual — w_shadow = w_disc + δ, δ ~ N(0, σ²I)

The shadow predictor fits residuals to observed actions via MAP estimation,
then reports held-out predictive NLL/Brier for promotion assessment.

Usage:
    shadow = ContinuousRewardShadow(mode="B2")
    shadow.observe(branches, action, theta="safe")
    nll = shadow.predictive_nll(branches, action, theta="safe")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np

from .stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, PREF_REWARD, PREFERENCE_TYPES,
)
from ..teachers.compositional_goal_hypotheses import (
    GoalHypothesisSpace, GoalHypothesis, DEFAULT_GOAL_SPACE,
    ATOMIC_GOAL_WEIGHTS,
)


@dataclass
class RewardShadowConfig:
    """Hyperparameters for continuous reward shadow."""
    learning_rate: float = 0.01     # MAP gradient step size
    prior_var: float = 1.0          # prior variance for regularization
    beta_softmax: float = 4.0       # softmax temperature
    epsilon_lapse: float = 0.1      # lapse rate


# ═══════════════════════════════════════════════════════════
# Interpretable basis vectors for B2
# ═══════════════════════════════════════════════════════════

V_RISK = np.array([1.0, -1.0, 0.0, 0.0])    # safety↑, temptation↓
V_TEMPT = np.array([0.0, 1.0, 0.0, 0.0])     # temptation direction


class ContinuousRewardShadow:
    """Shadow predictor with continuous reward residuals.

    Maintains per-(goal, theta) residual parameters that are fit
    online from observed actions. Does NOT write back to canonical posterior.

    Usage:
        shadow = ContinuousRewardShadow(mode="B2")
        for t in range(n_steps):
            shadow.observe(branches, action, "avoid_blue", "safe")
            nll = shadow.predictive_nll(branches, action, "avoid_blue", "safe")
    """

    def __init__(self,
                 mode: str = "B2",
                 config: Optional[RewardShadowConfig] = None,
                 goal_space: Optional[GoalHypothesisSpace] = None):
        assert mode in ("B1", "B2", "B3"), f"Unknown mode: {mode}"
        self.mode = mode
        self.cfg = config or RewardShadowConfig()
        self._goal_space = goal_space or DEFAULT_GOAL_SPACE

        # Per-(goal, theta) residual parameters
        self._residuals: Dict[Tuple[str, str], np.ndarray] = {}
        self._n_obs: Dict[Tuple[str, str], int] = {}
        self._history: List[Dict] = []

        # Initialize residuals to zero
        for gh in self._goal_space.hypotheses:
            for theta in PREFERENCE_TYPES:
                key = (gh.label, theta)
                if mode == "B1":
                    self._residuals[key] = np.array([0.0])  # 1D scale
                elif mode == "B2":
                    self._residuals[key] = np.array([0.0, 0.0])  # 2D
                else:  # B3
                    self._residuals[key] = np.zeros(4)  # 4D
                self._n_obs[key] = 0

    def get_effective_weights(self,
                              goal: GoalHypothesis,
                              theta: str) -> np.ndarray:
        """Get effective reward weights = base + residual."""
        # Base weights from discrete table
        base = goal.reward_weights + self.cfg.beta_softmax * 0.1 * PREF_REWARD.get(theta, np.zeros(4))

        key = (goal.label, theta)
        r = self._residuals.get(key, np.zeros(4))

        if self.mode == "B1":
            # Scale: w_shadow = (1 + α) · w_base
            alpha = r[0]
            return (1.0 + alpha) * base
        elif self.mode == "B2":
            # 2D interpretable: w + α·v_risk + β·v_tempt
            return base + r[0] * V_RISK + r[1] * V_TEMPT
        else:
            # 4D full residual
            return base + r

    def predict_probs(self,
                      branches: List[BranchAttributes],
                      goal_label: str,
                      theta: str) -> np.ndarray:
        """Predict choice probs using shadow weights."""
        gh = self._goal_space.get(goal_label)
        w = self.get_effective_weights(gh, theta)

        utilities = np.array([np.dot(w, b.to_array()) - b.risk_penalty
                              for b in branches])
        scaled = self.cfg.beta_softmax * utilities
        scaled -= np.max(scaled)
        exp_u = np.exp(scaled)
        softmax_p = exp_u / (exp_u.sum() + 1e-10)
        n = len(branches)
        return (1 - self.cfg.epsilon_lapse) * softmax_p + \
               self.cfg.epsilon_lapse * np.ones(n) / n

    def predictive_nll(self,
                       branches: List[BranchAttributes],
                       action: int,
                       goal_label: str,
                       theta: str) -> float:
        """NLL of observed action under shadow predictor."""
        probs = self.predict_probs(branches, goal_label, theta)
        return -np.log(max(float(probs[action]), 1e-15))

    def observe(self,
                branches: List[BranchAttributes],
                action: int,
                goal_label: str,
                theta: str):
        """Update residual via one-step MAP gradient."""
        key = (goal_label, theta)
        r = self._residuals[key]
        cfg = self.cfg

        # Current prediction
        probs = self.predict_probs(branches, goal_label, theta)

        # Gradient of log-likelihood w.r.t. residual
        gh = self._goal_space.get(goal_label)
        base = gh.reward_weights + cfg.beta_softmax * 0.1 * PREF_REWARD.get(theta, np.zeros(4))

        # Feature vectors for each branch
        xs = np.array([b.to_array() for b in branches])

        # d log P(action) / d r
        if self.mode == "B1":
            # d/dα: scale residual
            # U_i = (1+α) · base · x_i
            # dU_i/dα = base · x_i
            du = np.array([float(np.dot(base, x)) for x in xs])
            grad_ll = cfg.beta_softmax * (du[action] - np.sum(probs * du))
            grad = np.array([grad_ll])
        elif self.mode == "B2":
            # d/d[α,β]: 2D residual
            du_risk = np.array([float(np.dot(V_RISK, x)) for x in xs])
            du_tempt = np.array([float(np.dot(V_TEMPT, x)) for x in xs])
            grad_risk = cfg.beta_softmax * (du_risk[action] - np.sum(probs * du_risk))
            grad_tempt = cfg.beta_softmax * (du_tempt[action] - np.sum(probs * du_tempt))
            grad = np.array([grad_risk, grad_tempt])
        else:
            # d/dδ: 4D residual
            du = cfg.beta_softmax * (xs[action] - np.sum(probs[:, None] * xs, axis=0))
            grad = du

        # MAP regularization: -r / σ²
        reg = -r / cfg.prior_var

        # Update
        self._residuals[key] = r + cfg.learning_rate * (grad + reg)
        self._n_obs[key] = self._n_obs.get(key, 0) + 1

    def residual_norm(self, goal_label: str, theta: str) -> float:
        """L2 norm of current residual."""
        key = (goal_label, theta)
        return float(np.linalg.norm(self._residuals.get(key, np.zeros(1))))

    def all_residual_norms(self) -> Dict[str, float]:
        """All residual norms for audit."""
        return {f"{g}|{t}": float(np.linalg.norm(r))
                for (g, t), r in self._residuals.items()
                if self._n_obs.get((g, t), 0) > 0}

    def n_params(self) -> int:
        """Total free parameters."""
        if self.mode == "B1":
            return 1
        elif self.mode == "B2":
            return 2
        else:
            return 4

    def reset(self):
        """Reset all residuals to zero."""
        for key in self._residuals:
            self._residuals[key] = np.zeros_like(self._residuals[key])
            self._n_obs[key] = 0
