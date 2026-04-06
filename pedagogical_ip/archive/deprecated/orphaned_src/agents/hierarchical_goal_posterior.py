"""Hierarchical Goal-Factor Posterior.

Approximation:
  q(y, θ) ≈ q(θ) · ∏_k q(y_k) · exp(ψ(y, θ))

where:
  q(θ)  : [N_PREF] marginal preference posterior
  q(y_k): [3] per-factor marginal for factor k ∈ {-1, 0, +1}
  ψ(y,θ): lightweight coupling (K × N_PREF pairwise terms)

Advantages over exact 405-cell table:
  - Stabilizes faster: each factor marginal updates from every observation
  - Calibrates better: fewer parameters = tighter posteriors
  - Scales: adding factor K+1 adds 3 params, not 3^K cells

Update:
  q_k(v) ∝ q_k(v) · Σ_{θ,y_{-k}} q(θ)·∏_{j≠k}q_j(y_j)·P(a|y,θ)
  Then renormalize per factor and pref
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import itertools

import numpy as np

from .stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, PREFERENCE_TYPES, PREF_REWARD,
)
from .goal_factor_posterior import (
    FACTOR_VALUES, DEFAULT_K, FACTOR_WEIGHTS,
    compute_factor_likelihood, _all_goal_vectors,
)

N_PREF = len(PREFERENCE_TYPES)


@dataclass
class HierarchicalGoalPosterior:
    """Factorized posterior: q(θ)·∏q(y_k)·exp(ψ)."""

    K: int = DEFAULT_K

    # Factor marginals: log q(y_k) for each factor k, each value in {-1,0,+1}
    factor_log_probs: np.ndarray = field(default=None)   # [K, 3]

    # Preference marginal: log q(θ)
    pref_log_probs: np.ndarray = field(default=None)      # [N_PREF]

    # Coupling: ψ_{k,θ} — how factor k and pref θ interact
    psi: np.ndarray = field(default=None)                  # [K, N_PREF]

    observation_count: int = 0
    _all_y: list = field(default=None, repr=False)

    def __post_init__(self):
        if self._all_y is None:
            self._all_y = _all_goal_vectors(self.K)
        if self.factor_log_probs is None:
            self.factor_log_probs = np.zeros((self.K, 3))  # uniform
        if self.pref_log_probs is None:
            self.pref_log_probs = np.zeros(N_PREF)          # uniform
        if self.psi is None:
            # Weak initial coupling from preference-factor affinity
            self.psi = np.zeros((self.K, N_PREF))
            for pi, p in enumerate(PREFERENCE_TYPES):
                self.psi[:, pi] = np.array(PREF_REWARD[p][:self.K]) * 0.05

    @property
    def n_y(self) -> int:
        return len(self._all_y)

    def _factor_probs(self, k: int) -> np.ndarray:
        """Normalized q(y_k) for factor k."""
        lp = self.factor_log_probs[k] - np.max(self.factor_log_probs[k])
        p = np.exp(lp)
        return p / (p.sum() + 1e-10)

    def _pref_probs(self) -> np.ndarray:
        """Normalized q(θ)."""
        lp = self.pref_log_probs - np.max(self.pref_log_probs)
        p = np.exp(lp)
        return p / (p.sum() + 1e-10)

    def _joint_unnorm(self, yi: int, pi: int) -> float:
        """Unnormalized q(y, θ) = q(θ)·∏q(y_k)·exp(ψ)."""
        y = self._all_y[yi]
        q_theta = self._pref_probs()[pi]
        q_factors = 1.0
        psi_sum = 0.0
        for k in range(self.K):
            vidx = FACTOR_VALUES.index(y[k])
            q_factors *= self._factor_probs(k)[vidx]
            psi_sum += self.psi[k, pi] * y[k]
        return float(q_theta * q_factors * np.exp(psi_sum))

    @property
    def table(self) -> np.ndarray:
        """Full joint [n_y, N_PREF] normalized."""
        t = np.zeros((self.n_y, N_PREF))
        for yi in range(self.n_y):
            for pi in range(N_PREF):
                t[yi, pi] = self._joint_unnorm(yi, pi)
        s = t.sum()
        return t / max(s, 1e-10)

    @property
    def entropy(self) -> float:
        t = self.table.ravel()
        return float(-np.sum(t * np.log(t + 1e-10)))

    @property
    def max_entropy(self) -> float:
        return float(np.log(self.n_y * N_PREF))

    @property
    def marginal_pref(self) -> np.ndarray:
        return self._pref_probs()

    def predicted_factor(self, k: int) -> int:
        """MAP for factor k."""
        p = self._factor_probs(k)
        return FACTOR_VALUES[int(np.argmax(p))]

    @property
    def predicted_pref(self) -> str:
        return PREFERENCE_TYPES[int(np.argmax(self._pref_probs()))]

    @property
    def predicted_goal_vec(self) -> tuple[int, ...]:
        return tuple(self.predicted_factor(k) for k in range(self.K))

    @property
    def predicted_joint(self) -> tuple[tuple[int, ...], str]:
        t = self.table
        idx = np.unravel_index(np.argmax(t), t.shape)
        return self._all_y[idx[0]], PREFERENCE_TYPES[idx[1]]

    @property
    def joint_confidence(self) -> float:
        return float(np.max(self.table))

    def factor_accuracy(self, true_goal_vec: tuple[int, ...]) -> float:
        K = min(len(true_goal_vec), self.K)
        hits = sum(1 for k in range(K) if self.predicted_factor(k) == true_goal_vec[k])
        return hits / max(K, 1)

    def factor_confidence(self, k: int) -> float:
        """Top-1 probability for factor k."""
        return float(np.max(self._factor_probs(k)))

    def avg_factor_confidence(self) -> float:
        return float(np.mean([self.factor_confidence(k) for k in range(self.K)]))

    def update_from_choice(
        self,
        chosen_idx: int,
        branches: list[BranchAttributes],
        params: AgentPolicyParams,
        factor_weights: Optional[np.ndarray] = None,
    ):
        """Coordinate ascent update: update each factor marginal and pref marginal."""
        # Pre-compute all likelihoods
        all_liks = {}
        for yi, y in enumerate(self._all_y):
            for pi, p in enumerate(PREFERENCE_TYPES):
                all_liks[(yi, pi)] = compute_factor_likelihood(
                    chosen_idx, branches, y, p, params, factor_weights)

        # Update each factor k's marginal
        for k in range(self.K):
            new_log = np.zeros(3)
            for vidx, v in enumerate(FACTOR_VALUES):
                # Marginalize over all y_{-k} and θ
                acc = 0.0
                for yi, y in enumerate(self._all_y):
                    if y[k] != v:
                        continue
                    for pi in range(N_PREF):
                        w = self._joint_unnorm(yi, pi)
                        acc += w * all_liks[(yi, pi)]
                new_log[vidx] = np.log(max(acc, 1e-15))
            self.factor_log_probs[k] = new_log - np.mean(new_log)

        # Update preference marginal
        new_pref_log = np.zeros(N_PREF)
        for pi in range(N_PREF):
            acc = 0.0
            for yi in range(self.n_y):
                w = self._joint_unnorm(yi, pi)
                acc += w * all_liks[(yi, pi)]
            new_pref_log[pi] = np.log(max(acc, 1e-15))
        self.pref_log_probs = new_pref_log - np.mean(new_pref_log)

        # Update coupling ψ with small gradient step
        eta_psi = 0.02
        t = self.table
        for k in range(self.K):
            for pi in range(N_PREF):
                # Gradient: E_q[y_k · 1_{θ=pi}] - E_data[y_k · 1_{θ=pi}]
                expected = 0.0
                for yi, y in enumerate(self._all_y):
                    expected += t[yi, pi] * y[k]
                # Data term: approximate as weighted likelihood
                data_term = 0.0
                total_w = 0.0
                for yi, y in enumerate(self._all_y):
                    w = t[yi, pi] * all_liks[(yi, pi)]
                    data_term += w * y[k]
                    total_w += w
                if total_w > 1e-10:
                    data_term /= total_w
                self.psi[k, pi] += eta_psi * (data_term - expected)

        self.observation_count += 1

    def posterior_predictive_variance(
        self,
        branches: list[BranchAttributes],
        params: AgentPolicyParams,
        factor_weights: Optional[np.ndarray] = None,
    ) -> float:
        t = self.table.ravel()
        p_cells = np.array([
            compute_factor_likelihood(0, branches, self._all_y[yi],
                                      PREFERENCE_TYPES[pi], params, factor_weights)
            for yi in range(self.n_y) for pi in range(N_PREF)
        ])
        mean_p = float(np.dot(t, p_cells))
        return float(np.dot(t, (p_cells - mean_p) ** 2))
