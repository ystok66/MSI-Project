"""Contextual Bayesian Response Model.

Replaces discrete bucket posteriors with Bayesian linear regression heads
that take context φ(x_t, ℓ) as input and predict:
  - gain_C, gain_E: expected Phase C/E improvement
  - harm_otr, harm_nu, harm_gg: risk outcomes

Context vector φ:
  [lesson_id (one-hot), θ_hat, mastery(5), state(5), budget, recent_counts, severity, novelty]

Posterior: w ~ N(μ, Σ), updated via standard Bayesian linear regression.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

PROBE_NAMES = ["RC", "TR", "EP", "VA", "IA"]
N_LESSONS = 12  # max lesson catalog size


def build_context(lesson_name: str, lesson_catalog_names: list,
                  theta: str, mastery: dict,
                  nu: float, tau: float, gamma_gen: float, gamma_spec: float, kappa: float,
                  budget_remaining: float, recent_counts: dict,
                  severity: float = 0.5, novelty: float = 0.3) -> np.ndarray:
    """Build context feature vector φ(x_t, ℓ)."""
    # Lesson one-hot (padded to N_LESSONS)
    one_hot = np.zeros(N_LESSONS)
    if lesson_name in lesson_catalog_names:
        one_hot[lesson_catalog_names.index(lesson_name)] = 1.0

    # θ encoding
    theta_enc = np.array([1.0, 0.0] if theta == "safe" else [0.0, 1.0])

    # Mastery
    u = np.array([mastery.get(p, 0.5) for p in PROBE_NAMES])

    # Internalization state
    state = np.array([kappa, tau, nu, gamma_spec, gamma_gen])

    # Budget & recency
    budget_frac = np.array([min(budget_remaining / 4.0, 2.0)])
    recent = np.array([recent_counts.get(lesson_name, 0) / 3.0])

    # Episode params
    ep_params = np.array([severity, novelty])

    phi = np.concatenate([one_hot, theta_enc, u, state, budget_frac, recent, ep_params])
    return phi


@dataclass
class BayesianLinearHead:
    """Single Bayesian linear regression head: y = w'φ + ε."""
    dim: int = 0
    prior_precision: float = 1.0
    noise_var: float = 0.1

    # Sufficient statistics
    _A: np.ndarray = None   # precision matrix (dim × dim)
    _b: np.ndarray = None   # weighted sum (dim,)
    _n: int = 0

    def __post_init__(self):
        if self._A is None and self.dim > 0:
            self._A = self.prior_precision * np.eye(self.dim)
            self._b = np.zeros(self.dim)

    def init_dim(self, d: int):
        self.dim = d
        self._A = self.prior_precision * np.eye(d)
        self._b = np.zeros(d)

    def update(self, phi: np.ndarray, y: float):
        """Bayesian update: A += φφ'/σ², b += φy/σ²."""
        if self._A is None:
            self.init_dim(len(phi))
        self._A += np.outer(phi, phi) / self.noise_var
        self._b += phi * y / self.noise_var
        self._n += 1

    def predict(self, phi: np.ndarray) -> tuple[float, float]:
        """Return (mean, variance) of prediction at φ."""
        if self._A is None:
            self.init_dim(len(phi))
        try:
            Sigma = np.linalg.inv(self._A)
        except np.linalg.LinAlgError:
            Sigma = np.eye(self.dim) / self.prior_precision
        mu = Sigma @ self._b
        pred_mean = float(phi @ mu)
        pred_var = float(phi @ Sigma @ phi) + self.noise_var
        return pred_mean, max(pred_var, 1e-8)

    @property
    def n_updates(self):
        return self._n


@dataclass
class ContextualResponseModel:
    """Multi-head contextual Bayesian response model."""
    lesson_names: list = field(default_factory=list)
    theta: str = "safe"
    prior_precision: float = 0.5
    noise_var: float = 0.15

    # Heads
    gain_C: BayesianLinearHead = None
    gain_E: BayesianLinearHead = None
    harm_otr: BayesianLinearHead = None
    harm_nu: BayesianLinearHead = None
    harm_gg: BayesianLinearHead = None

    def __post_init__(self):
        if self.gain_C is None:
            self.gain_C = BayesianLinearHead(prior_precision=self.prior_precision, noise_var=self.noise_var)
            self.gain_E = BayesianLinearHead(prior_precision=self.prior_precision, noise_var=self.noise_var)
            self.harm_otr = BayesianLinearHead(prior_precision=self.prior_precision, noise_var=0.2)
            self.harm_nu = BayesianLinearHead(prior_precision=self.prior_precision, noise_var=0.2)
            self.harm_gg = BayesianLinearHead(prior_precision=self.prior_precision, noise_var=0.2)

    def _phi(self, lesson_name: str, mastery: dict,
             nu: float, tau: float, gg: float, gs: float, kappa: float,
             budget: float, recent: dict,
             severity: float = 0.5, novelty: float = 0.3) -> np.ndarray:
        return build_context(lesson_name, self.lesson_names,
                             self.theta, mastery,
                             nu, tau, gg, gs, kappa,
                             budget, recent, severity, novelty)

    def predict_gain(self, lesson_name: str, mastery: dict,
                     nu: float, tau: float, gg: float, gs: float, kappa: float,
                     budget: float, recent: dict,
                     severity: float = 0.5, novelty: float = 0.3):
        phi = self._phi(lesson_name, mastery, nu, tau, gg, gs, kappa, budget, recent, severity, novelty)
        mc, vc = self.gain_C.predict(phi)
        me, ve = self.gain_E.predict(phi)
        return {"C_mean": mc, "C_var": vc, "E_mean": me, "E_var": ve, "phi": phi}

    def predict_harm(self, lesson_name: str, mastery: dict,
                     nu: float, tau: float, gg: float, gs: float, kappa: float,
                     budget: float, recent: dict,
                     severity: float = 0.5, novelty: float = 0.3):
        phi = self._phi(lesson_name, mastery, nu, tau, gg, gs, kappa, budget, recent, severity, novelty)
        mo, vo = self.harm_otr.predict(phi)
        mn, vn = self.harm_nu.predict(phi)
        mg, vg = self.harm_gg.predict(phi)
        return {"otr_mean": mo, "otr_var": vo, "nu_mean": mn, "nu_var": vn, "gg_mean": mg, "gg_var": vg}

    def update(self, lesson_name: str, mastery: dict,
               nu: float, tau: float, gg: float, gs: float, kappa: float,
               budget: float, recent: dict,
               severity: float, novelty: float,
               # Outcome targets (scaled to [0,1] or small range)
               gain_c: float = 0.0, gain_e: float = 0.0,
               delta_otr: float = 0.0, delta_nu: float = 0.0, delta_gg: float = 0.0):
        phi = self._phi(lesson_name, mastery, nu, tau, gg, gs, kappa, budget, recent, severity, novelty)
        self.gain_C.update(phi, gain_c)
        self.gain_E.update(phi, gain_e)
        self.harm_otr.update(phi, delta_otr)
        self.harm_nu.update(phi, delta_nu)
        self.harm_gg.update(phi, delta_gg)

    def n_updated(self) -> int:
        return self.gain_C.n_updates
