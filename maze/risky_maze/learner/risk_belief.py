from __future__ import annotations

import numpy as np


class GaussianRiskBelief:
    def __init__(
        self,
        risk_dim: int,
        n_trap_types: int,
        seed: int = 0,
    ) -> None:
        self.risk_dim = risk_dim
        self.n_classes = n_trap_types + 1
        rng = np.random.default_rng(seed)
        prior_means = rng.normal(0.0, 0.8, size=(self.n_classes, risk_dim))
        prior_means[0] *= 0.25
        prior_vars = np.full((self.n_classes, risk_dim), 2.0, dtype=float)
        self.counts = np.ones(self.n_classes, dtype=float)
        self.sum_x = prior_means.copy()
        self.sum_x2 = prior_vars + prior_means**2
        self.min_var = 0.15

    def copy(self) -> "GaussianRiskBelief":
        other = GaussianRiskBelief(self.risk_dim, self.n_classes - 1)
        other.counts = self.counts.copy()
        other.sum_x = self.sum_x.copy()
        other.sum_x2 = self.sum_x2.copy()
        other.min_var = self.min_var
        return other

    def _means(self) -> np.ndarray:
        return self.sum_x / self.counts[:, None]

    def _vars(self) -> np.ndarray:
        mean = self._means()
        var = self.sum_x2 / self.counts[:, None] - mean**2
        return np.maximum(var, self.min_var)

    def posterior(self, x: np.ndarray) -> np.ndarray:
        means = self._means()
        vars_ = self._vars()
        priors = self.counts / np.sum(self.counts)
        logp = np.log(np.clip(priors, 1e-12, 1.0))
        diff = x[None, :] - means
        log_det = np.sum(np.log(vars_), axis=1)
        quad = np.sum(diff * diff / vars_, axis=1)
        logp = logp - 0.5 * (log_det + quad)
        logp -= np.max(logp)
        p = np.exp(logp)
        return p / np.sum(p)

    def danger_probability(self, x: np.ndarray) -> float:
        post = self.posterior(x)
        return float(1.0 - post[0])

    def update_labeled(self, x: np.ndarray, class_idx: int, weight: float = 1.0) -> None:
        self.counts[class_idx] += weight
        self.sum_x[class_idx] += weight * x
        self.sum_x2[class_idx] += weight * (x * x)

    def update_soft(self, x: np.ndarray, weights: np.ndarray) -> None:
        for idx, weight in enumerate(weights):
            if weight <= 0.0:
                continue
            self.update_labeled(x, idx, float(weight))

    def warning_update(self, features: list[np.ndarray]) -> np.ndarray:
        if not features:
            return np.zeros((0, self.n_classes), dtype=float)
        post = np.stack([self.posterior(x) for x in features], axis=0)
        p_safe = np.clip(post[:, 0], 1e-12, 1.0)
        p_all_safe = float(np.prod(p_safe))
        denom = max(1e-12, 1.0 - p_all_safe)
        updated = np.zeros_like(post)
        for i in range(post.shape[0]):
            p_rest_safe = p_all_safe / p_safe[i]
            updated[i, 0] = post[i, 0] * max(0.0, 1.0 - p_rest_safe) / denom
            updated[i, 1:] = post[i, 1:] / denom
            updated[i] /= np.sum(updated[i])
            self.update_soft(features[i], updated[i])
        return updated
