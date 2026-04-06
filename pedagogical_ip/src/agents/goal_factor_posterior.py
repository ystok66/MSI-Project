"""Goal Factor Posterior: exact joint table q(y, θ).

y ∈ {-1, 0, +1}^K is a factorized goal vector.
θ ∈ PREFERENCE_TYPES is the latent preference.

Joint state space: 3^K × N_PREF cells.
For K=4: 81 × 5 = 405 cells (exact table feasible).

q_t(y,θ) ∝ q_{t-1}(y,θ) · P_A(a_t | s_t, y, θ)

Utility:  U(π | y, θ) = Σ_k w_k · y_k · x_k(π) + λ_θ · R_pref(π;θ) - J_risk(π)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import itertools

import numpy as np

from .stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, PREFERENCE_TYPES, PREF_REWARD,
)

N_PREF = len(PREFERENCE_TYPES)
FACTOR_VALUES = [-1, 0, 1]
DEFAULT_K = 4
FACTOR_WEIGHTS = np.array([1.5, 1.5, 1.0, 1.0])  # per-factor importance


def _all_goal_vectors(K: int) -> list[tuple[int, ...]]:
    """All 3^K goal vectors."""
    return list(itertools.product(FACTOR_VALUES, repeat=K))


def compute_factor_utility(
    branch: BranchAttributes,
    goal_vec: tuple[int, ...],
    theta: str,
    params: AgentPolicyParams,
    factor_weights: Optional[np.ndarray] = None,
) -> float:
    """U(π | y, θ) = Σ_k w_k·y_k·x_k + λ_θ·R_pref(π;θ) - J_risk."""
    x = branch.to_array()  # [safety, tempt, novelty, shortcut]
    K = len(goal_vec)
    w = factor_weights if factor_weights is not None else FACTOR_WEIGHTS[:K]
    r_goal = float(np.sum(w * np.array(goal_vec) * x[:K]))
    r_pref = float(np.dot(PREF_REWARD[theta], x))
    return r_goal + params.lambda_theta * r_pref - branch.risk_penalty


def compute_factor_likelihood(
    chosen_idx: int,
    branches: list[BranchAttributes],
    goal_vec: tuple[int, ...],
    theta: str,
    params: AgentPolicyParams,
    factor_weights: Optional[np.ndarray] = None,
) -> float:
    """P(chose branch_idx | y, θ) via softmax + lapse."""
    utilities = np.array([
        compute_factor_utility(b, goal_vec, theta, params, factor_weights)
        for b in branches
    ])
    scaled = params.beta * utilities
    scaled -= np.max(scaled)
    exp_u = np.exp(scaled)
    sm = exp_u / (exp_u.sum() + 1e-10)
    n = len(branches)
    uniform = np.ones(n) / n
    mixed = (1 - params.epsilon) * sm + params.epsilon * uniform
    return float(mixed[chosen_idx])


# Compatibility prior: some (y, θ) pairs are more natural
def _build_compat_prior(K: int) -> np.ndarray:
    """C(y, θ) = η · Σ_k y_k · b_{k,θ}."""
    all_y = _all_goal_vectors(K)
    n_y = len(all_y)
    eta = 0.1
    # b_{k,θ}: how much pref θ aligns with factor k
    # safety-oriented prefs like goal factor k=0 (safety)
    b = np.zeros((DEFAULT_K, N_PREF))
    for pi, p in enumerate(PREFERENCE_TYPES):
        b[:, pi] = PREF_REWARD[p][:DEFAULT_K] * 0.1

    compat = np.zeros((n_y, N_PREF))
    for yi, y in enumerate(all_y):
        for pi in range(N_PREF):
            compat[yi, pi] = eta * float(np.sum(np.array(y[:DEFAULT_K]) * b[:DEFAULT_K, pi]))
    return compat


@dataclass
class GoalFactorPosterior:
    """Exact joint posterior q(y, θ) over factor goals × preferences."""
    K: int = DEFAULT_K
    log_table: np.ndarray = field(default=None)
    observation_count: int = 0
    forgetting_rate: float = 0.0
    _all_y: list = field(default=None, repr=False)

    def __post_init__(self):
        if self._all_y is None:
            self._all_y = _all_goal_vectors(self.K)
        if self.log_table is None:
            self.log_table = _build_compat_prior(self.K)

    @property
    def n_y(self) -> int:
        return len(self._all_y)

    @property
    def table(self) -> np.ndarray:
        lt = self.log_table - np.max(self.log_table)
        t = np.exp(lt)
        return t / (t.sum() + 1e-10)

    @property
    def entropy(self) -> float:
        t = self.table.ravel()
        return float(-np.sum(t * np.log(t + 1e-10)))

    @property
    def max_entropy(self) -> float:
        return float(np.log(self.n_y * N_PREF))

    @property
    def marginal_goal(self) -> np.ndarray:
        """[n_y] marginal over θ."""
        return self.table.sum(axis=1)

    @property
    def marginal_pref(self) -> np.ndarray:
        """[N_PREF] marginal over y."""
        return self.table.sum(axis=0)

    @property
    def predicted_goal_vec(self) -> tuple[int, ...]:
        mg = self.marginal_goal
        return self._all_y[int(np.argmax(mg))]

    @property
    def predicted_pref(self) -> str:
        mp = self.marginal_pref
        return PREFERENCE_TYPES[int(np.argmax(mp))]

    @property
    def predicted_joint(self) -> tuple[tuple[int, ...], str]:
        t = self.table
        idx = np.unravel_index(np.argmax(t), t.shape)
        return self._all_y[idx[0]], PREFERENCE_TYPES[idx[1]]

    @property
    def joint_confidence(self) -> float:
        return float(np.max(self.table))

    def predicted_factor(self, k: int) -> int:
        """Marginal MAP for factor k."""
        mg = self.marginal_goal  # [n_y]
        factor_mass = {v: 0.0 for v in FACTOR_VALUES}
        for yi, y in enumerate(self._all_y):
            factor_mass[y[k]] += mg[yi]
        return max(factor_mass, key=factor_mass.get)

    def factor_accuracy(self, true_goal_vec: tuple[int, ...]) -> float:
        """Per-factor hit rate."""
        K = min(len(true_goal_vec), self.K)
        hits = sum(1 for k in range(K) if self.predicted_factor(k) == true_goal_vec[k])
        return hits / max(K, 1)

    def update_from_choice(
        self,
        chosen_idx: int,
        branches: list[BranchAttributes],
        params: AgentPolicyParams,
        factor_weights: Optional[np.ndarray] = None,
    ):
        for yi, y in enumerate(self._all_y):
            for pi, p in enumerate(PREFERENCE_TYPES):
                lik = compute_factor_likelihood(
                    chosen_idx, branches, y, p, params, factor_weights)
                self.log_table[yi, pi] += np.log(lik + 1e-10)
        self.log_table -= np.mean(self.log_table)
        if self.forgetting_rate > 0:
            t = self.table
            u = np.ones_like(t) / t.size
            td = (1 - self.forgetting_rate) * t + self.forgetting_rate * u
            self.log_table = np.log(td + 1e-10)
            self.log_table -= np.mean(self.log_table)
        self.observation_count += 1

    def posterior_predictive_variance(
        self, branches: list[BranchAttributes],
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
