"""Hybrid Response Model: Hierarchical backbone + contextual residual + pairwise dueling.

Architecture:
  μ_hyb(x, ℓ) = μ_hier(ℓ, b)  +  r_ctx(x, ℓ)
  
  μ_hier: hierarchical EB shrinkage (from v8 LessonResponseModelV3)
  r_ctx:  Bayesian linear residual head over φ(x, ℓ)

Pairwise dueling:
  ΔV(x, ℓ_i, ℓ_j) = w'(φ(x,ℓ_i) - φ(x,ℓ_j))
  
  Trained on observed outcome differences between lessons.
  Forces the model to learn *relative* lesson value, maximizing cross-lesson discriminability.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .lesson_response_model_v3 import LessonResponseModelV3, mastery_bucket, PROBE_NAMES

N_LESSONS = 12


def _build_phi(lesson_name: str, catalog_names: list,
               theta: str, mastery: dict,
               nu: float, tau: float, gg: float, gs: float, kappa: float,
               budget: float, recent: dict,
               severity: float = 0.5, novelty: float = 0.3) -> np.ndarray:
    one_hot = np.zeros(N_LESSONS)
    if lesson_name in catalog_names:
        one_hot[catalog_names.index(lesson_name)] = 1.0
    theta_enc = np.array([1.0, 0.0] if theta == "safe" else [0.0, 1.0])
    u = np.array([mastery.get(p, 0.5) for p in PROBE_NAMES])
    state = np.array([kappa, tau, nu, gs, gg])
    bud = np.array([min(budget / 4.0, 2.0)])
    rec = np.array([recent.get(lesson_name, 0) / 3.0])
    ep = np.array([severity, novelty])
    return np.concatenate([one_hot, theta_enc, u, state, bud, rec, ep])


@dataclass
class BayesianResidualHead:
    """Bayesian linear residual: r = w'φ."""
    prior_precision: float = 2.0
    noise_var: float = 0.2
    _A: np.ndarray = None
    _b: np.ndarray = None
    _n: int = 0

    def init_dim(self, d: int):
        self._A = self.prior_precision * np.eye(d)
        self._b = np.zeros(d)

    def update(self, phi: np.ndarray, y: float):
        if self._A is None: self.init_dim(len(phi))
        self._A += np.outer(phi, phi) / self.noise_var
        self._b += phi * y / self.noise_var
        self._n += 1

    def predict(self, phi: np.ndarray) -> tuple[float, float]:
        if self._A is None: self.init_dim(len(phi))
        try: Sigma = np.linalg.inv(self._A)
        except np.linalg.LinAlgError: Sigma = np.eye(len(phi)) / self.prior_precision
        mu = Sigma @ self._b
        return float(phi @ mu), float(max(phi @ Sigma @ phi + self.noise_var, 1e-8))


@dataclass
class PairwiseDuelingHead:
    """Pairwise dueling: ΔV = w'(φ_i - φ_j). Learns relative lesson value."""
    prior_precision: float = 1.0
    noise_var: float = 0.3
    _A: np.ndarray = None
    _b: np.ndarray = None
    _n: int = 0

    def init_dim(self, d: int):
        self._A = self.prior_precision * np.eye(d)
        self._b = np.zeros(d)

    def update(self, phi_i: np.ndarray, phi_j: np.ndarray, delta_y: float):
        """Update with observed outcome difference: y_i - y_j."""
        dphi = phi_i - phi_j
        if self._A is None: self.init_dim(len(dphi))
        self._A += np.outer(dphi, dphi) / self.noise_var
        self._b += dphi * delta_y / self.noise_var
        self._n += 1

    def predict_delta(self, phi_i: np.ndarray, phi_j: np.ndarray) -> tuple[float, float]:
        dphi = phi_i - phi_j
        if self._A is None: self.init_dim(len(dphi))
        try: Sigma = np.linalg.inv(self._A)
        except np.linalg.LinAlgError: Sigma = np.eye(len(dphi)) / self.prior_precision
        mu = Sigma @ self._b
        return float(dphi @ mu), float(max(dphi @ Sigma @ dphi + self.noise_var, 1e-8))

    def score(self, phi: np.ndarray) -> tuple[float, float]:
        """Score a single lesson (for ranking): w'φ."""
        if self._A is None: self.init_dim(len(phi))
        try: Sigma = np.linalg.inv(self._A)
        except np.linalg.LinAlgError: Sigma = np.eye(len(phi)) / self.prior_precision
        mu = Sigma @ self._b
        return float(phi @ mu), float(max(phi @ Sigma @ phi, 1e-8))


@dataclass
class HybridResponseModel:
    """Hybrid = hierarchical backbone + contextual residual + pairwise dueling."""
    hier: LessonResponseModelV3 = None
    catalog_names: list = field(default_factory=list)
    theta: str = "safe"

    # Contextual residual heads
    res_gain: BayesianResidualHead = field(default_factory=lambda: BayesianResidualHead(prior_precision=2.0, noise_var=0.15))
    res_harm: BayesianResidualHead = field(default_factory=lambda: BayesianResidualHead(prior_precision=2.0, noise_var=0.2))

    # Pairwise dueling heads
    duel_gain: PairwiseDuelingHead = field(default_factory=lambda: PairwiseDuelingHead(prior_precision=1.0, noise_var=0.25))
    duel_harm: PairwiseDuelingHead = field(default_factory=lambda: PairwiseDuelingHead(prior_precision=1.0, noise_var=0.3))

    # Recent observation buffer for pairwise training
    _recent_obs: list = field(default_factory=list)
    _max_recent: int = 30

    def __post_init__(self):
        if self.hier is None:
            self.hier = LessonResponseModelV3()

    def _phi(self, lesson_name: str, mastery: dict,
             nu: float, tau: float, gg: float, gs: float, kappa: float,
             budget: float, recent: dict,
             severity: float = 0.5, novelty: float = 0.3) -> np.ndarray:
        return _build_phi(lesson_name, self.catalog_names, self.theta, mastery,
                          nu, tau, gg, gs, kappa, budget, recent, severity, novelty)

    def predict_gain(self, lesson_name: str, mastery: dict,
                     nu: float, tau: float, gg: float, gs: float, kappa: float,
                     budget: float, recent: dict,
                     severity: float = 0.5, novelty: float = 0.3) -> dict:
        bucket = mastery_bucket(mastery)
        phi = self._phi(lesson_name, mastery, nu, tau, gg, gs, kappa, budget, recent, severity, novelty)

        # Hierarchical backbone
        hier_gain = sum(self.hier.gain_expected(lesson_name, p, bucket) for p in PROBE_NAMES) / len(PROBE_NAMES)
        hier_var = sum(self.hier.gain_variance(lesson_name, p, bucket) for p in PROBE_NAMES) / len(PROBE_NAMES)

        # Contextual residual
        res_mean, res_var = self.res_gain.predict(phi)

        # Dueling score
        duel_mean, duel_var = self.duel_gain.score(phi)

        # Combine: hybrid = hier + residual, with dueling bonus
        mean = hier_gain + 0.5 * res_mean + 0.3 * duel_mean
        var = hier_var + 0.25 * res_var + 0.09 * duel_var

        return {"mean": mean, "var": var, "phi": phi,
                "hier": hier_gain, "res": res_mean, "duel": duel_mean}

    def predict_harm(self, lesson_name: str, mastery: dict,
                     nu: float, tau: float, gg: float, gs: float, kappa: float,
                     budget: float, recent: dict,
                     severity: float = 0.5, novelty: float = 0.3) -> dict:
        bucket = mastery_bucket(mastery)
        phi = self._phi(lesson_name, mastery, nu, tau, gg, gs, kappa, budget, recent, severity, novelty)
        hier_harm_raw = self.hier.total_harm(lesson_name, bucket)
        # Normalize: weights sum to 6.0, so divide to get [0,1] range
        hier_harm = hier_harm_raw / 6.0
        res_mean, res_var = self.res_harm.predict(phi)
        duel_mean, duel_var = self.duel_harm.score(phi)
        mean = hier_harm + 0.3 * res_mean + 0.2 * duel_mean
        var = 0.09 * res_var + 0.04 * duel_var + 0.01
        return {"mean": mean, "var": var, "hier": hier_harm, "res": res_mean, "duel": duel_mean}

    def update(self, lesson_name: str, mastery: dict,
               nu: float, tau: float, gg: float, gs: float, kappa: float,
               budget: float, recent: dict,
               severity: float, novelty: float,
               mastery_before: dict, mastery_after: dict,
               nu_after: float, gg_after: float,
               otr_before: float, otr_after: float):
        bucket = mastery_bucket(mastery_before)
        phi = self._phi(lesson_name, mastery_before, nu, tau, gg, gs, kappa, budget, recent, severity, novelty)

        # Update hierarchical backbone
        self.hier.update_gain(lesson_name, bucket, mastery_before, mastery_after)
        self.hier.update_harm(lesson_name, bucket, nu, nu_after, gg, gg_after, otr_before, otr_after)

        # Compute outcomes for residual
        gain_outcome = sum(mastery_after.get(p, 0.5) - mastery_before.get(p, 0.5) for p in PROBE_NAMES) / len(PROBE_NAMES)
        harm_outcome = (otr_after - otr_before) + 0.5 * (nu_after - nu) + 0.5 * (gg_after - gg)

        # Residual target = outcome - hierarchical prediction
        hier_gain = sum(self.hier.gain_expected(lesson_name, p, bucket) for p in PROBE_NAMES) / len(PROBE_NAMES)
        hier_harm = self.hier.total_harm(lesson_name, bucket)
        self.res_gain.update(phi, gain_outcome - hier_gain)
        self.res_harm.update(phi, harm_outcome - hier_harm)

        # Store for pairwise training
        obs = {"phi": phi, "gain": gain_outcome, "harm": harm_outcome, "lesson": lesson_name}
        self._recent_obs.append(obs)
        if len(self._recent_obs) > self._max_recent:
            self._recent_obs.pop(0)

        # Pairwise updates: compare with recent observations of DIFFERENT lessons
        for prev in self._recent_obs[:-1]:
            if prev["lesson"] != lesson_name:
                self.duel_gain.update(phi, prev["phi"], gain_outcome - prev["gain"])
                self.duel_harm.update(phi, prev["phi"], harm_outcome - prev["harm"])

    def n_updated(self) -> int:
        return self.hier.n_updated() + self.res_gain._n + self.duel_gain._n
