"""
Cost-Risk Model — joint Bayesian linear heads for cost and risk prediction.

Phase 4: feature-as-latent semantics.
The cell's true latent state is represented by the existing 4D feature vector,
which is hidden to the agent but available to the simulator.

The agent learns to predict BOTH cost and risk from its noisy belief over the
feature vector, using two independent Bayesian linear heads:

  cost_hat  = w_c · z + b_c             (Gaussian likelihood)
  risk_hat  = sigmoid(w_r · z + b_r)    (Bernoulli likelihood)

The LatentCostRiskHead composes both and exposes a unified predictor interface
(latent_predictor protocol) for the planner.

Supervision modes for risk:
  "oracle_visited"  — use true risk value for visited cells
  "binary_outcome"  — use 0/1 hazard outcome only
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .risk_model import BayesianRiskHead, _sigmoid


class BayesianCostHead:
    """Bayesian linear model for feature → cost prediction.

    Uses Gaussian likelihood with online MAP estimation.
    Prior: w ~ N(0, prior_var * I), b ~ N(1.0, prior_var).
    """

    def __init__(self, d: int = 4, prior_var: float = 1.0,
                 learning_rate: float = 0.1):
        self.d = d
        self.w = np.zeros(d, dtype=np.float64)
        self.b = 1.0  # prior: cost ≈ 1.0 for normal cells
        self.prior_var = prior_var
        self.lr = learning_rate

        # Sufficient statistics for uncertainty estimation
        self.n_updates = 0
        self.xx_sum = np.zeros((d, d), dtype=np.float64)
        self.xy_sum = np.zeros(d, dtype=np.float64)

    def predict_cost(self, x: np.ndarray) -> float:
        """Predict traversal cost from feature vector."""
        return float(max(self.w @ x + self.b, 0.1))

    def predict_cost_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict cost for (H, W, d) or (N, d) feature array."""
        shape = X.shape[:-1]
        X_flat = X.reshape(-1, self.d)
        costs = X_flat @ self.w + self.b
        return np.clip(costs, 0.1, None).reshape(shape)

    def predict_uncertainty(self, x: np.ndarray) -> float:
        """Approximate predictive uncertainty (variance of cost prediction)."""
        if self.n_updates < 2:
            return 1.0  # high uncertainty under prior

        H = self.xx_sum / max(self.n_updates, 1) + np.eye(self.d) / self.prior_var
        try:
            H_inv = np.linalg.inv(H)
            return float(max(x @ H_inv @ x, 0.01))
        except np.linalg.LinAlgError:
            return 1.0

    def update_from_label(self, x: np.ndarray, y: float, weight: float = 1.0):
        """Online MAP update: (feature, cost_label) pair.

        y: realized traversal cost for this cell.
        """
        pred = self.w @ x + self.b
        error = y - pred

        # SGD on negative log-posterior (Gaussian likelihood + Gaussian prior)
        grad_w = -error * x * weight + self.w / self.prior_var
        grad_b = -error * weight

        # Gradient clipping: prevent weight explosions
        grad_norm = float(np.linalg.norm(grad_w))
        max_grad_norm = 5.0
        if not np.isfinite(grad_norm):
            return  # skip update on NaN/Inf gradient
        if grad_norm > max_grad_norm:
            grad_w *= max_grad_norm / grad_norm

        self.w -= self.lr * grad_w
        self.b -= self.lr * float(np.clip(grad_b, -max_grad_norm, max_grad_norm))

        # Weight norm clamping
        max_w_norm = 10.0
        w_norm = float(np.linalg.norm(self.w))
        if w_norm > max_w_norm:
            self.w *= max_w_norm / w_norm

        self.xx_sum += weight * np.outer(x, x)
        self.xy_sum += weight * y * x
        self.n_updates += 1

    def reset(self):
        self.w[:] = 0.0
        self.b = 1.0
        self.xx_sum[:] = 0.0
        self.xy_sum[:] = 0.0
        self.n_updates = 0


@dataclass
class WorldWeights:
    """Fixed world parameters that derive cost and risk from latent z.

    Stored in environment metadata, reproducible from seed.
    """
    w_cost: np.ndarray    # (d,) cost weight vector
    b_cost: float         # cost bias
    w_risk: np.ndarray    # (d,) risk weight vector
    b_risk: float         # risk bias

    def true_cost(self, z: np.ndarray) -> float:
        """Ground truth cost from latent vector."""
        return float(max(self.w_cost @ z + self.b_cost, 0.1))

    def true_risk(self, z: np.ndarray) -> float:
        """Ground truth risk probability from latent vector."""
        logit = float(self.w_risk @ z + self.b_risk)
        return float(_sigmoid(np.array([logit]))[0])


def generate_world_weights(rng: np.random.Generator, d: int = 4) -> WorldWeights:
    """Generate fixed world weights from RNG (reproducible from seed).

    Design: texture dimensions (indices 2,3) drive risk;
    all dimensions contribute to cost variation.
    """
    # Cost weights: mild variation, base cost ≈ 1.0
    w_cost = rng.uniform(-0.3, 0.3, size=d).astype(np.float64)
    b_cost = 1.0

    # Risk weights: texture dims (2,3) are strong drivers, others weak
    w_risk = np.zeros(d, dtype=np.float64)
    w_risk[0] = rng.uniform(-0.5, 0.5)   # lane_id: mild effect
    w_risk[1] = rng.uniform(-0.3, 0.3)   # gate_flag: mild
    w_risk[2] = rng.uniform(2.0, 4.0)    # texture_1: strong positive
    w_risk[3] = rng.uniform(1.5, 3.5)    # texture_2: strong positive
    b_risk = rng.uniform(-3.0, -1.5)     # bias: shift so most cells are low risk

    return WorldWeights(w_cost=w_cost, b_cost=b_cost,
                        w_risk=w_risk, b_risk=b_risk)


class LatentCostRiskHead:
    """Joint Bayesian predictor: feature-as-latent → (cost, risk).

    Composes a BayesianCostHead and a BayesianRiskHead.
    Exposes the latent_predictor protocol for the planner:
      - predict_cost(x)
      - predict_risk(x)
      - predict_cost_uncertainty(x)
      - predict_risk_uncertainty(x)
    """

    def __init__(
        self,
        d: int = 4,
        cost_prior_var: float = 1.0,
        risk_prior_var: float = 1.0,
        cost_lr: float = 0.1,
        risk_lr: float = 0.3,
        risk_supervision: Literal["oracle_visited", "binary_outcome"] = "oracle_visited",
    ):
        self.d = d
        self.cost_head = BayesianCostHead(d=d, prior_var=cost_prior_var,
                                           learning_rate=cost_lr)
        self.risk_head = BayesianRiskHead(d=d, prior_var=risk_prior_var,
                                           learning_rate=risk_lr)
        self.risk_supervision = risk_supervision

    def predict_cost(self, x: np.ndarray) -> float:
        return self.cost_head.predict_cost(x)

    def predict_risk(self, x: np.ndarray) -> float:
        return self.risk_head.predict_risk(x)

    def predict_cost_uncertainty(self, x: np.ndarray) -> float:
        return self.cost_head.predict_uncertainty(x)

    def predict_risk_uncertainty(self, x: np.ndarray) -> float:
        return self.risk_head.predict_uncertainty(x)

    def predict_cost_uncertainty_from_var(self, x_var: np.ndarray) -> float:
        """Directional cost uncertainty: w_c^T · diag(x_var) · w_c.

        Uses posterior variance from FeatureBeliefMap.var[r,c] instead of
        the Hessian-based approximation. More principled when belief variance
        is available.
        """
        return float(self.cost_head.w @ (x_var * self.cost_head.w))

    def predict_risk_uncertainty_from_var(self, x_var: np.ndarray) -> float:
        """Directional risk uncertainty: w_r^T · diag(x_var) · w_r.

        Uses posterior variance from FeatureBeliefMap.var[r,c] instead of
        the Hessian-based approximation.
        """
        return float(self.risk_head.w @ (x_var * self.risk_head.w))

    def update_from_outcome(
        self,
        x: np.ndarray,
        cost_label: float,
        risk_label: float,
        weight: float = 1.0,
    ):
        """Update both heads from observation.

        cost_label: realized traversal cost (strong supervision).
        risk_label: depends on risk_supervision mode:
          - "oracle_visited": true risk value for cell
          - "binary_outcome": 0.0 (safe) or 1.0 (died)
        """
        self.cost_head.update_from_label(x, cost_label, weight=weight)
        self.risk_head.update_from_label(x, risk_label, weight=weight)

    def reset(self):
        self.cost_head.reset()
        self.risk_head.reset()

    @property
    def n_updates(self):
        return self.risk_head.n_updates
