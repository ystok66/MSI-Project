"""
risk_belief.py — Bayesian danger type belief model.

Multi-class safe/danger type classification using Gaussian prototypes
with online moment-matching updates.

Latent type z_i ∈ {0, 1, ..., K}:
  0 = safe
  1..K = danger types

Observation model: x_i ~ N(μ_{z_i}, Σ_{z_i} + Σ_obs)
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

from ..config import LearnerConfig


class DangerTypeBelief:
    """Bayesian danger type classifier.

    Maintains Gaussian prototypes for each type and computes
    posterior P(z_i = k | x_i) for each observed ball.
    """

    def __init__(
        self,
        n_danger_types: int,
        danger_dim: int,
        obs_sigma: float = 0.3,
        prior_safe: float = 0.7,
    ):
        self.n_danger_types = n_danger_types
        self.n_types = 1 + n_danger_types  # 0=safe + K danger
        self.danger_dim = danger_dim
        self.obs_sigma = obs_sigma

        # Type priors: P(z_i = k)
        # Safe gets prior_safe, danger types share the rest equally
        self.type_prior = np.zeros(self.n_types)
        self.type_prior[0] = prior_safe
        if n_danger_types > 0:
            self.type_prior[1:] = (1.0 - prior_safe) / n_danger_types

        # Prototype means: μ_k for each type
        # Initialized to zero; set by set_prototypes() or learn from data
        self.proto_mu = np.zeros((self.n_types, danger_dim))

        # Prototype diagonal variances: σ²_k for each type
        self.proto_var = np.ones((self.n_types, danger_dim))

        # Online moment-matching accumulators
        self._counts = np.zeros(self.n_types)
        self._sum_x = np.zeros((self.n_types, danger_dim))
        self._sum_x2 = np.zeros((self.n_types, danger_dim))

    def set_prototypes(
        self,
        proto_mu: np.ndarray,
        proto_var: Optional[np.ndarray] = None,
    ):
        """Set prototype parameters.

        Args:
            proto_mu: (n_types, danger_dim) prototype means
            proto_var: (n_types, danger_dim) diagonal variances (optional)
        """
        self.proto_mu = proto_mu.copy()
        if proto_var is not None:
            self.proto_var = proto_var.copy()

    def single_ball_posterior(self, x: np.ndarray) -> np.ndarray:
        """Compute P(z_i = k | x_i) for a single observation.

        Uses diagonal Gaussian likelihood + prior:
        P(z=k|x) ∝ P(x|z=k) · P(z=k)
        P(x|z=k) = N(x; μ_k, Σ_k + Σ_obs)

        Args:
            x: (danger_dim,) observed vector

        Returns:
            (n_types,) posterior probabilities, sums to 1.
        """
        log_probs = np.zeros(self.n_types)
        obs_var = self.obs_sigma ** 2

        for k in range(self.n_types):
            total_var = self.proto_var[k] + obs_var  # (d,)
            diff = x - self.proto_mu[k]
            # Diagonal Gaussian log-likelihood
            log_lik = -0.5 * np.sum(
                np.log(2 * np.pi * total_var) + diff**2 / total_var
            )
            log_probs[k] = log_lik + np.log(max(self.type_prior[k], 1e-30))

        # Numerically stable softmax
        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        probs /= probs.sum() + 1e-30
        return probs

    def batch_posterior(self, X: np.ndarray) -> np.ndarray:
        """Compute posteriors for multiple observations (vectorized).

        Args:
            X: (m, danger_dim) observed vectors

        Returns:
            (m, n_types) posterior matrix
        """
        m = X.shape[0]
        obs_var = self.obs_sigma ** 2

        # Compute all log-likelihoods at once: (m, n_types)
        log_probs = np.zeros((m, self.n_types))
        for k in range(self.n_types):
            total_var = self.proto_var[k] + obs_var  # (d,)
            diff = X - self.proto_mu[k]  # (m, d)
            log_lik = -0.5 * np.sum(
                np.log(2 * np.pi * total_var) + diff**2 / total_var,
                axis=1
            )  # (m,)
            log_probs[:, k] = log_lik + np.log(max(self.type_prior[k], 1e-30))

        # Numerically stable softmax per row
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True) + 1e-30
        return probs

    def set_danger_probability(self, X: np.ndarray) -> float:
        """P(∃ danger in set) = 1 - Π P(z_i=0 | x_i).

        Args:
            X: (m, danger_dim) observed vectors for selected set

        Returns:
            Probability that at least one ball is dangerous.
        """
        posteriors = self.batch_posterior(X)  # (m, n_types)
        p_all_safe = np.prod(posteriors[:, 0])
        return 1.0 - p_all_safe

    def update_from_death(self, x: np.ndarray):
        """Update prototypes after death: this ball is definitely danger.

        Hard label: z_i ≠ 0. Distribute update across danger types
        proportional to current posterior.
        """
        post = self.single_ball_posterior(x)
        # Zero out safe, renormalize among danger types
        post[0] = 0.0
        if post.sum() < 1e-30:
            post[1:] = 1.0 / self.n_danger_types
        else:
            post /= post.sum()

        self._accumulate_update(x, post)

    def update_from_safe_observation(self, x: np.ndarray):
        """Update after confirmed safe observation (e.g., placed without death)."""
        post = np.zeros(self.n_types)
        post[0] = 1.0
        self._accumulate_update(x, post)

    def _accumulate_update(self, x: np.ndarray, soft_label: np.ndarray):
        """Online moment-matching update from one observation.

        Updates prototype means and variances using soft assignment.
        """
        for k in range(self.n_types):
            w = soft_label[k]
            if w < 1e-10:
                continue
            self._counts[k] += w
            self._sum_x[k] += w * x
            self._sum_x2[k] += w * x**2

            # Update prototype via moment matching
            n = self._counts[k]
            if n > 0.5:
                new_mu = self._sum_x[k] / n
                new_var = self._sum_x2[k] / n - new_mu**2
                new_var = np.maximum(new_var, 1e-4)  # floor variance

                self.proto_mu[k] = new_mu
                self.proto_var[k] = new_var

    def reset(self):
        """Reset to prior state."""
        self._counts[:] = 0.0
        self._sum_x[:] = 0.0
        self._sum_x2[:] = 0.0
        self.type_prior[0] = 0.7
        if self.n_danger_types > 0:
            self.type_prior[1:] = 0.3 / self.n_danger_types
